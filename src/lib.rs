mod co_tree;
mod graph;

pub use co_tree::*;
pub use graph::{Graph, Representation::*};

use std::collections::BTreeMap;
use std::thread;

#[derive(Debug, Clone, Copy)]
pub struct Options {
    degree_filter: bool,
    no_unnecessary_type_comparisons: bool,
    prefetch_neighbors: bool,
}

impl Options {
    #[must_use]
    pub const fn new(
        degree_filter: bool,
        no_unnecessary_type_comparisons: bool,
        prefetch_neighbors: bool,
    ) -> Self {
        Self {
            degree_filter,
            no_unnecessary_type_comparisons,
            prefetch_neighbors,
        }
    }

    #[must_use]
    pub const fn naive() -> Self {
        Self {
            degree_filter: false,
            no_unnecessary_type_comparisons: false,
            prefetch_neighbors: false,
        }
    }

    #[must_use]
    pub const fn optimized() -> Self {
        Self {
            degree_filter: true,
            no_unnecessary_type_comparisons: true,
            prefetch_neighbors: true,
        }
    }
}

#[must_use]
pub fn calc_nd_classes(graph: &Graph, options: Options) -> Vec<Vec<usize>> {
    let mut type_connectivity_graph =
        Graph::null_graph(graph.vertex_count(), graph.representation());

    // collect degrees for all vertices
    let degrees: Vec<usize> = if options.degree_filter {
        (0..graph.vertex_count())
            .map(|vertex| graph.degree(vertex))
            .collect()
    } else {
        vec![]
    };

    // prefetch neighbors for all vertices
    let neighbors: Vec<Vec<bool>> = if options.prefetch_neighbors {
        (0..graph.vertex_count())
            .map(|vertex| graph.neighbors_as_bool_vector(vertex))
            .collect()
    } else {
        vec![]
    };

    for u in 0..graph.vertex_count() {
        for v in u..graph.vertex_count() {
            // only compare neighborhoods if vertices have same degree
            if options.degree_filter && degrees[u] != degrees[v] {
                continue;
            }

            // only compare neighborhoods if v is not already in an equivalence class
            if options.no_unnecessary_type_comparisons && type_connectivity_graph.degree(v) != 0 {
                continue;
            }

            if options.prefetch_neighbors {
                {
                    let mut u_neighbors = neighbors[u].clone();
                    let mut v_neighbors = neighbors[v].clone();

                    // N(u) \ v
                    u_neighbors[v] = false;
                    // N(v) \ u
                    v_neighbors[u] = false;

                    // equal comparison works because neighbors are bool vector
                    if u_neighbors == v_neighbors {
                        type_connectivity_graph
                            .insert_edge(u, v)
                            .expect("u and v are elements of range 0..vertex_count");
                    }
                }
            } else if same_type(graph, u, v) {
                type_connectivity_graph
                    .insert_edge(u, v)
                    .expect("u and v are elements of range 0..vertex_count");
            }
        }
    }

    type_connectivity_graph.connected_components()
}

#[must_use]
pub fn calc_nd_btree(graph: &Graph) -> Vec<Vec<usize>> {
    let mut neighborhood_partition: Vec<Vec<usize>> = Vec::new();
    let mut cliques: BTreeMap<Vec<bool>, usize> = BTreeMap::new();
    let mut independent_sets: BTreeMap<Vec<bool>, usize> = BTreeMap::new();

    for vertex in 0..graph.vertex_count() {
        let mut clique_type: Vec<bool> = graph.neighbors_as_bool_vector(vertex);
        let independent_set_type = clique_type.clone();
        clique_type[vertex] = true;

        if let Some(&vertex_type) = cliques.get(&clique_type) {
            neighborhood_partition[vertex_type].push(vertex);
        } else if let Some(&vertex_type) = independent_sets.get(&independent_set_type) {
            neighborhood_partition[vertex_type].push(vertex);
        } else {
            let vertex_type = neighborhood_partition.len();
            neighborhood_partition.push(vec![vertex]);
            cliques.insert(clique_type, vertex_type);
            independent_sets.insert(independent_set_type, vertex_type);
        }
    }

    neighborhood_partition
}

#[must_use]
pub fn calc_nd_btree_degree(graph: &Graph) -> Vec<Vec<usize>> {
    let mut neighborhood_partition: Vec<Vec<usize>> = Vec::new();
    let mut cliques: Vec<BTreeMap<Vec<bool>, usize>> = vec![BTreeMap::new(); graph.vertex_count()];
    let mut independent_sets: Vec<BTreeMap<Vec<bool>, usize>> =
        vec![BTreeMap::new(); graph.vertex_count()];
    for vertex in 0..graph.vertex_count() {
        let neighbors = graph.neighbors(vertex);
        let degree = neighbors.len();

        let mut clique_type: Vec<bool> = graph.neighbors_as_bool_vector(vertex);
        let independent_set_type = clique_type.clone();
        clique_type[vertex] = true;

        let cliques = cliques.get_mut(degree).unwrap();
        let independent_sets = independent_sets.get_mut(degree).unwrap();

        if let Some(&vertex_type) = cliques.get(&clique_type) {
            neighborhood_partition[vertex_type].push(vertex);
        } else if let Some(&vertex_type) = independent_sets.get(&independent_set_type) {
            neighborhood_partition[vertex_type].push(vertex);
        } else {
            let vertex_type = neighborhood_partition.len();
            neighborhood_partition.push(vec![vertex]);
            cliques.insert(clique_type, vertex_type);
            independent_sets.insert(independent_set_type, vertex_type);
        }
    }

    neighborhood_partition
}

#[must_use]
pub fn calc_nd_btree_concurrent(graph: &Graph) -> Vec<Vec<usize>> {
    #[derive(Debug, Default, Clone)]
    struct Data {
        neighborhood_partition: Vec<Vec<usize>>,
        cliques: BTreeMap<Vec<bool>, usize>,
        independent_sets: BTreeMap<Vec<bool>, usize>,
    }

    const MAGIC_NUMBER: usize = 100;
    let thread_count = (graph.vertex_count() / MAGIC_NUMBER + 1)
        .min(thread::available_parallelism().map_or(8, std::convert::Into::into));
    let mut thread_data: Vec<Data> = vec![Data::default(); thread_count];

    thread::scope(|scope| {
        for (thread_id, data) in thread_data.iter_mut().enumerate() {
            scope.spawn(move || {
                let start = thread_id * graph.vertex_count() / thread_count;
                let end = (thread_id + 1) * graph.vertex_count() / thread_count;

                for vertex in start..end {
                    let mut clique_type: Vec<bool> = graph.neighbors_as_bool_vector(vertex);
                    let independent_set_type = clique_type.clone();
                    clique_type[vertex] = true;

                    if let Some(&vertex_type) = data.cliques.get(&clique_type.clone()) {
                        data.neighborhood_partition[vertex_type].push(vertex);
                    } else if let Some(&vertex_type) =
                        data.independent_sets.get(&independent_set_type.clone())
                    {
                        data.neighborhood_partition[vertex_type].push(vertex);
                    } else {
                        let vertex_type = data.neighborhood_partition.len();
                        data.neighborhood_partition.push(vec![vertex]);
                        data.cliques.insert(clique_type.clone(), vertex_type);
                        data.independent_sets
                            .insert(independent_set_type.clone(), vertex_type);
                    }
                }
            });
        }
    });

    // collect into last element
    let mut collection = thread_data.pop().unwrap();

    for data in &thread_data {
        let cliques_inverted: BTreeMap<usize, Vec<bool>> =
            data.cliques.iter().map(|(k, v)| (*v, k.clone())).collect();

        let independent_sets_inverted: BTreeMap<usize, Vec<bool>> = data
            .independent_sets
            .iter()
            .map(|(k, v)| (*v, k.clone()))
            .collect();

        for (vertex_type, vertices) in data.neighborhood_partition.iter().enumerate() {
            if let Some(&get) = collection
                .cliques
                .get(cliques_inverted.get(&vertex_type).unwrap())
            {
                collection.neighborhood_partition[get].extend(vertices);
            } else if let Some(&get) = collection
                .independent_sets
                .get(independent_sets_inverted.get(&vertex_type).unwrap())
            {
                collection.neighborhood_partition[get].extend(vertices);
            } else {
                // insert clique type into collection
                collection.cliques.insert(
                    cliques_inverted.get(&vertex_type).unwrap().clone(),
                    collection.neighborhood_partition.len(),
                );
                // insert independent set type into collection
                collection.independent_sets.insert(
                    independent_sets_inverted.get(&vertex_type).unwrap().clone(),
                    collection.neighborhood_partition.len(),
                );
                // add vertices as new type
                collection.neighborhood_partition.push(vertices.clone());
            }
        }
    }

    collection.neighborhood_partition
}

#[must_use]
fn same_type(graph: &Graph, u: usize, v: usize) -> bool {
    let mut u_neighbors = graph.neighbors_as_bool_vector(u);
    let mut v_neighbors = graph.neighbors_as_bool_vector(v);

    // N(u) \ v
    u_neighbors[v] = false;
    // N(v) \ u
    v_neighbors[u] = false;

    // equal comparison works because neighbors are bool vector
    u_neighbors == v_neighbors
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    const VERTEX_COUNT: usize = 1e2 as usize;
    const DENSITY: f32 = 0.5;
    const ND_LIMIT: usize = 20;
    const REPRESENTATIONS: &[graph::Representation] = &[AdjacencyMatrix, AdjacencyList];

    fn test_graphs() -> Vec<Graph> {
        REPRESENTATIONS
            .iter()
            .map(|&representation| {
                Graph::random_graph_nd_limited(VERTEX_COUNT, DENSITY, ND_LIMIT, representation)
            })
            .collect()
    }

    fn baseline(graph: &Graph) -> Vec<Vec<usize>> {
        let mut type_connectivity_graph = Graph::null_graph(graph.vertex_count(), AdjacencyMatrix);

        for u in 0..graph.vertex_count() {
            for v in u..graph.vertex_count() {
                if same_type(graph, u, v) {
                    type_connectivity_graph
                        .insert_edge(u, v)
                        .expect("u and v are elements of range 0..vertex_count");
                }
            }
        }

        type_connectivity_graph.connected_components()
    }

    fn same_type(graph: &Graph, u: usize, v: usize) -> bool {
        let mut u_neighbors = graph.neighbors(u);
        let mut v_neighbors = graph.neighbors(v);

        // N(u) \ v
        if let Some(pos) = u_neighbors.iter().position(|&x| x == v) {
            u_neighbors.remove(pos);
        }
        // N(v) \ u
        if let Some(pos) = v_neighbors.iter().position(|&x| x == u) {
            v_neighbors.remove(pos);
        }

        u_neighbors.sort_unstable();
        v_neighbors.sort_unstable();

        u_neighbors == v_neighbors
    }

    #[test]
    fn baseline_on_example() {
        let path = "examples/nd_01.txt";
        let input = std::fs::read_to_string(path)
            .unwrap_or_else(|error| panic!("error reading '{}': {}", path, error));

        let graph = input
            .parse::<Graph>()
            .unwrap_or_else(|error| panic!("error parsing input: {}", error));

        let neighborhood_diversity = baseline(&graph).len();

        assert_eq!(neighborhood_diversity, 6);
    }

    #[test]
    fn baseline_on_example_shuffled() {
        let path = "examples/nd_01_shuffled.txt";
        let input = std::fs::read_to_string(path)
            .unwrap_or_else(|error| panic!("error reading '{}': {}", path, error));

        let graph = input
            .parse::<Graph>()
            .unwrap_or_else(|error| panic!("error parsing input: {}", error));

        let neighborhood_diversity = baseline(&graph).len();

        assert_eq!(neighborhood_diversity, 6);
    }

    #[test]
    fn naive_on_example_shuffled() {
        let path = "examples/nd_01_shuffled.txt";
        let input = std::fs::read_to_string(path)
            .unwrap_or_else(|error| panic!("error reading '{}': {}", path, error));

        let graph = input
            .parse::<Graph>()
            .unwrap_or_else(|error| panic!("error parsing input: {}", error));

        let neighborhood_diversity = calc_nd_classes(&graph, Options::naive()).len();

        assert_eq!(neighborhood_diversity, 6);
    }

    #[test]
    fn baseline_vs_naive() {
        for graph in test_graphs() {
            assert_eq!(
                calc_nd_classes(&graph, Options::naive()).len(),
                baseline(&graph).len()
            );
        }
    }

    #[test]
    fn baseline_vs_degree_filter() {
        // test algorithm with degree filter against naive implementation
        for graph in test_graphs() {
            assert_eq!(
                calc_nd_classes(&graph, Options::new(true, false, false)).len(),
                baseline(&graph).len()
            );
        }
    }

    #[test]
    fn baseline_vs_no_unnecessary_comparisons() {
        // test algorithm with degree filter against naive implementation
        for graph in test_graphs() {
            assert_eq!(
                calc_nd_classes(&graph, Options::new(false, true, false)).len(),
                baseline(&graph).len()
            );
        }
    }

    #[test]
    fn baseline_vs_prefetch_neighbors() {
        // test algorithm with degree filter against naive implementation
        for graph in test_graphs() {
            assert_eq!(
                calc_nd_classes(&graph, Options::new(false, false, true)).len(),
                baseline(&graph).len()
            );
        }
    }

    #[test]
    fn baseline_vs_optimized() {
        // test algorithm with degree filter against naive implementation
        for graph in test_graphs() {
            assert_eq!(
                calc_nd_classes(&graph, Options::optimized()).len(),
                baseline(&graph).len()
            );
        }
    }

    #[test]
    fn baseline_vs_btree() {
        // test algorithm with degree filter against naive implementation
        for graph in test_graphs() {
            assert_eq!(calc_nd_btree(&graph).len(), baseline(&graph).len());
        }
    }

    #[test]
    fn baseline_vs_btree_degree() {
        // test algorithm with degree filter against naive implementation
        for graph in test_graphs() {
            assert_eq!(calc_nd_btree_degree(&graph).len(), baseline(&graph).len());
        }
    }

    #[test]
    fn baseline_vs_btree_concurrent() {
        // test algorithm with degree filter against naive implementation
        for graph in test_graphs() {
            assert_eq!(
                calc_nd_btree_concurrent(&graph).len(),
                baseline(&graph).len()
            );
        }
    }

    #[test]
    fn empty_graph() {
        for &representation in REPRESENTATIONS {
            let null_graph = Graph::null_graph(0, representation);
            let expected = 0;

            // baseline
            assert_eq!(baseline(&null_graph).len(), expected);

            // naive
            assert_eq!(
                calc_nd_classes(&null_graph, Options::naive()).len(),
                expected
            );

            // degree_filter
            assert_eq!(
                calc_nd_classes(&null_graph, Options::new(true, false, false)).len(),
                expected
            );

            // no unnecessary comparisons
            assert_eq!(
                calc_nd_classes(&null_graph, Options::new(false, true, false)).len(),
                expected
            );

            // optimized
            assert_eq!(
                calc_nd_classes(&null_graph, Options::optimized()).len(),
                expected
            );

            // btree
            assert_eq!(calc_nd_btree(&null_graph).len(), expected);
        }
    }

    #[test]
    fn null_graph() {
        for &representation in REPRESENTATIONS {
            let null_graph = Graph::null_graph(VERTEX_COUNT, representation);
            let expected = 1;

            // baseline
            assert_eq!(baseline(&null_graph).len(), expected);

            // naive
            assert_eq!(
                calc_nd_classes(&null_graph, Options::naive()).len(),
                expected
            );

            // degree_filter
            assert_eq!(
                calc_nd_classes(&null_graph, Options::new(true, false, false)).len(),
                expected
            );

            // no unnecessary comparisons
            assert_eq!(
                calc_nd_classes(&null_graph, Options::new(false, true, false)).len(),
                expected
            );

            // optimized
            assert_eq!(
                calc_nd_classes(&null_graph, Options::optimized()).len(),
                expected
            );

            // btree
            assert_eq!(calc_nd_btree(&null_graph).len(), expected);

            // btree degree
            assert_eq!(calc_nd_btree_degree(&null_graph).len(), expected);

            // btree concurrent
            assert_eq!(calc_nd_btree_concurrent(&null_graph).len(), expected);
        }
    }

    #[test]
    fn complete_graph() {
        for &representation in REPRESENTATIONS {
            let complete_graph = Graph::complete_graph(VERTEX_COUNT, representation);
            let expected = 1;

            // baseline
            assert_eq!(baseline(&complete_graph).len(), expected);

            // naive
            assert_eq!(
                calc_nd_classes(&complete_graph, Options::naive()).len(),
                expected
            );

            // degree_filter
            assert_eq!(
                calc_nd_classes(&complete_graph, Options::new(true, false, false)).len(),
                expected
            );

            // no unnecessary comparisons
            assert_eq!(
                calc_nd_classes(&complete_graph, Options::new(false, true, false)).len(),
                expected
            );

            // optimized
            assert_eq!(
                calc_nd_classes(&complete_graph, Options::optimized()).len(),
                expected
            );

            // btree
            assert_eq!(calc_nd_btree(&complete_graph).len(), expected);

            // btree degree
            assert_eq!(calc_nd_btree_degree(&complete_graph).len(), expected);

            // btree concurrent
            assert_eq!(calc_nd_btree_concurrent(&complete_graph).len(), expected);
        }
    }

    #[test]
    fn shuffled() {
        for graph in &mut test_graphs() {
            let expected = baseline(graph).len();
            graph.shuffle();

            // baseline
            assert_eq!(baseline(graph).len(), expected);

            // naive
            assert_eq!(calc_nd_classes(graph, Options::naive()).len(), expected);

            // degree_filter
            assert_eq!(
                calc_nd_classes(graph, Options::new(true, false, false)).len(),
                expected
            );

            // no unnecessary comparisons
            assert_eq!(
                calc_nd_classes(graph, Options::new(false, true, false)).len(),
                expected
            );

            // optimized
            assert_eq!(calc_nd_classes(graph, Options::optimized()).len(), expected);

            // btree
            assert_eq!(calc_nd_btree(graph).len(), expected);

            // btree degree
            assert_eq!(calc_nd_btree_degree(graph).len(), expected);

            // btree concurrent
            assert_eq!(calc_nd_btree_concurrent(graph).len(), expected);
        }
    }
}
