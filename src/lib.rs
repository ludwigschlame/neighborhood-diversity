mod graph;
mod graph_adj_list;

// pub use graph::Graph;
pub use graph_adj_list::Graph;

use std::collections::BTreeMap;

#[derive(Debug)]
pub struct Options {
    degree_filter: bool,
    no_unnecessary_type_comparisons: bool,
}

impl Options {
    pub fn new(degree_filter: bool, no_unnecessary_neighborhood_comparisons: bool) -> Self {
        Self {
            degree_filter,
            no_unnecessary_type_comparisons: no_unnecessary_neighborhood_comparisons,
        }
    }

    pub fn naive() -> Self {
        Self {
            degree_filter: false,
            no_unnecessary_type_comparisons: false,
        }
    }

    pub fn optimized() -> Self {
        Self {
            degree_filter: true,
            no_unnecessary_type_comparisons: true,
        }
    }
}

pub fn calc_nd_classes(graph: &Graph, options: Options) -> Vec<Vec<usize>> {
    let mut type_connectivity_graph = Graph::null_graph(graph.vertex_count());

    let mut degrees: Vec<usize> = Vec::new();
    if options.degree_filter {
        // collect degrees for all vertices
        degrees = (0..graph.vertex_count())
            .map(|vertex| graph.degree(vertex))
            .collect();
    }

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

            if same_type(graph, u, v) {
                type_connectivity_graph
                    .insert_edge(u, v)
                    .expect("u and v are elements of range 0..vertex_count");
            }
        }
    }

    type_connectivity_graph.connected_components()
}

pub fn calc_nd_btree(graph: &Graph) -> Vec<Vec<usize>> {
    let mut types: Vec<Vec<usize>> = Vec::new();

    let mut cliques: BTreeMap<Vec<_>, usize> = BTreeMap::new();
    let mut independent_sets: BTreeMap<Vec<_>, usize> = BTreeMap::new();

    for vertex in 0..graph.vertex_count() {
        // let mut clique_type: Vec<_> = graph
        //     .neighbors(vertex)
        //     .into_iter()
        //     .chain([vertex])
        //     .collect();
        // clique_type.sort();
        // let mut independent_set_type: Vec<_> = clique_type.clone();
        // if let Ok(pos) = independent_set_type.binary_search(&vertex) {
        //     independent_set_type.remove(pos);
        // }

        let mut clique_type: Vec<bool> = vec![false; graph.vertex_count()];
        for neighbor in graph.neighbors(vertex) {
            clique_type[neighbor] = true;
        }
        let independent_set_type = clique_type.clone();
        clique_type[vertex] = true;

        if let Some(vertex_type) = cliques.get(&clique_type) {
            types[*vertex_type].push(vertex);
        } else if let Some(vertex_type) = independent_sets.get(&independent_set_type) {
            types[*vertex_type].push(vertex);
        } else {
            let vertex_type = types.len();
            types.push(vec![vertex]);
            cliques.insert(clique_type, vertex_type);
            independent_sets.insert(independent_set_type, vertex_type);
        }
    }

    types
}

fn same_type(graph: &Graph, u: usize, v: usize) -> bool {
    let mut u_neighbors = graph.neighbors(u);
    let mut v_neighbors = graph.neighbors(v);

    // N(u) \ v
    if let Some(pos) = u_neighbors.iter().position(|x| *x == v) {
        u_neighbors.remove(pos);
    }
    // N(v) \ u
    if let Some(pos) = v_neighbors.iter().position(|x| *x == u) {
        v_neighbors.remove(pos);
    }

    // equal comparison works because 'graph.neighbors()' always results in same order
    u_neighbors == v_neighbors
}

#[cfg(test)]
mod tests {
    use super::*;

    const VERTEX_COUNT: usize = 1e2 as usize;
    const DENSITY: f64 = 0.5;
    const ND_LIMIT: usize = 20;

    fn test_graph() -> Graph {
        Graph::random_graph_nd_limited(VERTEX_COUNT, DENSITY, ND_LIMIT)
    }

    fn baseline(graph: &Graph) -> Vec<Vec<usize>> {
        let mut type_connectivity_graph = Graph::null_graph(graph.vertex_count());

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
    fn naive_vs_baseline() {
        let random_graph = test_graph();

        assert_eq!(
            calc_nd_classes(&random_graph, Options::naive()).len(),
            baseline(&random_graph).len()
        );
    }

    #[test]
    fn degree_filter_vs_baseline() {
        let random_graph = test_graph();

        // test algorithm with degree filter against naive implementation
        assert_eq!(
            calc_nd_classes(&random_graph, Options::new(true, false)).len(),
            baseline(&random_graph).len()
        );
    }

    #[test]
    fn no_unnecessary_comparisons_vs_baseline() {
        let random_graph = test_graph();

        // test algorithm with degree filter against naive implementation
        assert_eq!(
            calc_nd_classes(&random_graph, Options::new(false, true)).len(),
            baseline(&random_graph).len()
        );
    }

    #[test]
    fn optimized_vs_baseline() {
        let random_graph = test_graph();

        // test algorithm with degree filter against naive implementation
        assert_eq!(
            calc_nd_classes(&random_graph, Options::optimized()).len(),
            baseline(&random_graph).len()
        );
    }

    #[test]
    fn btree_vs_baseline() {
        let random_graph = test_graph();

        // test algorithm with degree filter against naive implementation
        assert_eq!(
            calc_nd_btree(&random_graph).len(),
            baseline(&random_graph).len()
        );
    }

    #[test]
    fn empty_graph() {
        let null_graph = Graph::null_graph(0);
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
            calc_nd_classes(&null_graph, Options::new(true, false)).len(),
            expected
        );

        // no unnecessary comparisons
        assert_eq!(
            calc_nd_classes(&null_graph, Options::new(false, true)).len(),
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

    #[test]
    fn null_graph() {
        let null_graph = Graph::null_graph(VERTEX_COUNT);
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
            calc_nd_classes(&null_graph, Options::new(true, false)).len(),
            expected
        );

        // no unnecessary comparisons
        assert_eq!(
            calc_nd_classes(&null_graph, Options::new(false, true)).len(),
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

    #[test]
    fn complete_graph() {
        let null_graph = Graph::complete_graph(VERTEX_COUNT);
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
            calc_nd_classes(&null_graph, Options::new(true, false)).len(),
            expected
        );

        // no unnecessary comparisons
        assert_eq!(
            calc_nd_classes(&null_graph, Options::new(false, true)).len(),
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
