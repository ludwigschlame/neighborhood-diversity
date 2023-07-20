#![warn(clippy::undocumented_unsafe_blocks)]
#![warn(clippy::use_debug)]
#![warn(clippy::nursery)]
#![warn(clippy::pedantic)]
#![allow(clippy::uninlined_format_args)] // inlined format args don's support batch renaming (yet?)
#![allow(clippy::missing_panics_doc)] // missing docs in general (todo!)
#![allow(clippy::missing_errors_doc)] // missing docs in general (todo!)

pub mod co_tree;
pub mod graph;
pub mod md_tree;
pub mod prelude;

use co_tree::CoTree;
use graph::Graph;
use md_tree::MDTree;

use std::collections::BTreeMap;
use std::num::NonZeroUsize;
use std::thread;

pub type Partition = Vec<Vec<usize>>;

#[derive(Debug, Clone, Copy)]
pub struct Options {
    degree_filter: bool,
    no_unnecessary_type_comparisons: bool,
}

impl Options {
    #[must_use]
    pub const fn new(degree_filter: bool, no_unnecessary_type_comparisons: bool) -> Self {
        Self {
            degree_filter,
            no_unnecessary_type_comparisons,
        }
    }

    #[must_use]
    pub const fn naive() -> Self {
        Self {
            degree_filter: false,
            no_unnecessary_type_comparisons: false,
        }
    }

    #[must_use]
    pub const fn optimized() -> Self {
        Self {
            degree_filter: true,
            no_unnecessary_type_comparisons: true,
        }
    }
}

#[must_use]
fn same_type(graph: &Graph, u: usize, v: usize) -> bool {
    let u_neighbors = graph.neighbors_as_bool_vector(u);
    let v_neighbors = graph.neighbors_as_bool_vector(v);

    let small = u.min(v);
    let large = u.max(v);

    // compare neighborhoods excluding u and v
    // u_neighbors[u] (v_neighbors[v]) always false because no self-loops
    // u_neighbors[v] (v_neighbors[u]) always false because N(u)\{v} (N(v)\{u})
    u_neighbors[..small] == v_neighbors[..small]
        && u_neighbors[small + 1..large] == v_neighbors[small + 1..large]
        && u_neighbors[large + 1..] == v_neighbors[large + 1..]
}

#[must_use]
pub fn calc_nd_classes(graph: &Graph, options: Options) -> Partition {
    let vertex_count = graph.vertex_count();
    let mut neighborhood_partition: Partition = vec![];
    let mut classes = vec![None::<usize>; vertex_count];

    // collect degrees for all vertices
    let degrees: Vec<usize> = if options.degree_filter {
        (0..vertex_count)
            .map(|vertex| graph.degree(vertex))
            .collect()
    } else {
        vec![]
    };

    let mut nd: usize = 0;

    for u in 0..vertex_count {
        // only compare neighborhoods if v is not already in an equivalence class
        if options.no_unnecessary_type_comparisons && classes[u].is_some() {
            continue;
        }

        if classes[u].is_none() {
            classes[u] = Some(nd);
            neighborhood_partition.push(vec![u]);
            nd += 1;
        }

        for v in (u + 1)..vertex_count {
            if options.no_unnecessary_type_comparisons && classes[v].is_some()
                || options.degree_filter && degrees[u] != degrees[v]
            {
                continue;
            }

            if same_type(graph, u, v) {
                if classes[v].is_none() {
                    neighborhood_partition[classes[u].unwrap()].push(v);
                }
                classes[v] = classes[u];
            }
        }
    }

    neighborhood_partition
}

#[must_use]

#[must_use]
pub fn calc_nd_classes_improved(graph: &Graph, options: Options) -> Partition {
    let vertex_count = graph.vertex_count();
    let mut neighborhood_partition: Partition = vec![];
    let mut classified = vec![false; vertex_count];

    // collect degrees for all vertices
    let degrees: Vec<usize> = if options.degree_filter {
        (0..vertex_count)
            .map(|vertex| graph.degree(vertex))
            .collect()
    } else {
        vec![]
    };

    for u in 0..vertex_count {
        if classified[u] {
            continue;
        }

        let mut neighborhood_class = vec![u];
        for v in (u + 1)..vertex_count {
            if options.no_unnecessary_type_comparisons && classified[v]
                || options.degree_filter && degrees[u] != degrees[v]
            {
                continue;
            }

            if same_type(graph, u, v) {
                classified[v] = true;
                neighborhood_class.push(v);
            }
        }
        neighborhood_partition.push(neighborhood_class);
    }

    neighborhood_partition
}

#[must_use]
pub fn calc_nd_btree(graph: &Graph) -> Partition {
    let mut neighborhood_partition: Partition = Vec::new();
    let mut cliques: BTreeMap<Vec<bool>, usize> = BTreeMap::new();
    let mut independent_sets: BTreeMap<Vec<bool>, usize> = BTreeMap::new();

    for vertex in 0..graph.vertex_count() {
        let independent_set_type: &Vec<bool> = &graph.neighbors_as_bool_vector(vertex);
        let clique_type: &mut Vec<bool> = &mut independent_set_type.clone();
        clique_type[vertex] = true;

        if let Some(&vertex_type) = independent_sets.get(independent_set_type) {
            neighborhood_partition[vertex_type].push(vertex);
        } else if let Some(&vertex_type) = cliques.get(clique_type) {
            neighborhood_partition[vertex_type].push(vertex);
        } else {
            let vertex_type = neighborhood_partition.len();
            neighborhood_partition.push(vec![vertex]);
            cliques.insert(clique_type.clone(), vertex_type);
            independent_sets.insert(independent_set_type.clone(), vertex_type);
        }
    }

    neighborhood_partition
}

#[must_use]
pub fn calc_nd_btree_degree(graph: &Graph) -> Partition {
    let mut neighborhood_partition: Partition = Vec::new();
    let mut cliques: Vec<BTreeMap<Vec<bool>, usize>> = vec![BTreeMap::new(); graph.vertex_count()];
    let mut independent_sets: Vec<BTreeMap<Vec<bool>, usize>> =
        vec![BTreeMap::new(); graph.vertex_count()];
    for vertex in 0..graph.vertex_count() {
        let degree = graph.degree(vertex);

        let independent_set_type: &Vec<bool> = &graph.neighbors_as_bool_vector(vertex);
        let clique_type: &mut Vec<bool> = &mut independent_set_type.clone();
        clique_type[vertex] = true;

        let cliques = cliques.get_mut(degree).unwrap();
        let independent_sets = independent_sets.get_mut(degree).unwrap();

        if let Some(&vertex_type) = cliques.get(clique_type) {
            neighborhood_partition[vertex_type].push(vertex);
        } else if let Some(&vertex_type) = independent_sets.get(independent_set_type) {
            neighborhood_partition[vertex_type].push(vertex);
        } else {
            let vertex_type = neighborhood_partition.len();
            neighborhood_partition.push(vec![vertex]);
            cliques.insert(clique_type.clone(), vertex_type);
            independent_sets.insert(independent_set_type.clone(), vertex_type);
        }
    }

    neighborhood_partition
}

#[must_use]
pub fn calc_nd_btree_concurrent(graph: &Graph, thread_count: NonZeroUsize) -> Partition {
    #[derive(Debug, Default, Clone)]
    struct Data {
        neighborhood_partition: Partition,
        cliques: BTreeMap<Vec<bool>, usize>,
        independent_sets: BTreeMap<Vec<bool>, usize>,
    }

    let mut thread_data: Vec<Data> = vec![Data::default(); thread_count.into()];

    thread::scope(|scope| {
        for (thread_id, data) in thread_data.iter_mut().enumerate() {
            scope.spawn(move || {
                let start = thread_id * graph.vertex_count() / thread_count;
                let end = (thread_id + 1) * graph.vertex_count() / thread_count;

                for vertex in start..end {
                    let independent_set_type: &Vec<bool> = &graph.neighbors_as_bool_vector(vertex);
                    let clique_type: &mut Vec<bool> = &mut independent_set_type.clone();
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
    let mut collection = thread_data.pop().expect("len is non-zero");

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;
    use pretty_assertions::assert_eq;
    use rand::{thread_rng, Rng};
    use rayon::prelude::{IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

    const VERTEX_COUNT: usize = 100;
    const DENSITY: f32 = 0.5;
    const ND_LIMIT: usize = 20;
    const REPRESENTATIONS: &[graph::Representation] = &[AdjacencyMatrix, AdjacencyList];
    const THREAD_COUNT: NonZeroUsize = {
        // SAFETY: 3 is non-zero.
        unsafe { NonZeroUsize::new_unchecked(3) }
    };

    fn baseline(graph: &Graph) -> Partition {
        // closure replacing the same_type() function
        let same_type = |u: usize, v: usize| -> bool {
            let mut u_neighbors: Vec<bool> = graph.neighbors_as_bool_vector(u).to_vec();
            let mut v_neighbors: Vec<bool> = graph.neighbors_as_bool_vector(v).to_vec();

            // N(u) \ v
            u_neighbors[v] = false;
            // N(v) \ u
            v_neighbors[u] = false;

            // equal comparison works because neighbors are bool vector
            u_neighbors == v_neighbors
        };

        let vertex_count: usize = graph.vertex_count();
        let mut partition: Partition = Vec::new();
        let mut classes: Vec<Option<usize>> = vec![None; vertex_count];
        let mut nd: usize = 0;

        for u in 0..vertex_count {
            if classes[u].is_none() {
                classes[u] = Some(nd);
                partition.resize((nd + 1).max(partition.len()), vec![]);
                partition[nd].push(u);
                nd += 1;
            }
            for v in (u + 1)..vertex_count {
                if same_type(u, v) {
                    classes[v] = classes[u];
                    if !partition[classes[u].unwrap()].contains(&v) {
                        partition[classes[u].unwrap()].push(v);
                    }
                }
            }
        }

        partition
    }

    fn test_graphs() -> Vec<Graph> {
        const GRAPHS_PER_REPRESENTATION: usize = 10;

        REPRESENTATIONS
            .iter()
            .map(|&representation| {
                (0..GRAPHS_PER_REPRESENTATION)
                    .map(|_| {
                        Graph::random_graph_nd_limited(
                            VERTEX_COUNT,
                            DENSITY,
                            ND_LIMIT,
                            representation,
                        )
                    })
                    .collect::<Vec<Graph>>()
            })
            .collect::<Vec<Vec<Graph>>>()
            .concat()
    }

    fn all_unique(partition: &Partition, vertex_count: usize) {
        let mut counter = std::collections::HashMap::new();

        for class in partition {
            for &vertex in class {
                *counter.entry(vertex).or_insert(0) += 1;
            }
        }

        if vertex_count == 0 {
            assert_eq!(counter.len(), vertex_count);
            assert_eq!(counter.keys().min(), None);
            assert_eq!(counter.keys().max(), None);
        } else {
            assert_eq!(counter.len(), vertex_count);
            assert_eq!(counter.keys().min(), Some(&0));
            assert_eq!(counter.keys().max(), Some(&(vertex_count - 1)));
            assert!(counter.values().all(|&count| count == 1));
        }
    }

    fn compare_all(graph: &Graph, expected: usize) {
        let vertex_count = graph.vertex_count();

        let partitions = &[
            baseline(graph),
            calc_nd_classes(graph, Options::naive()),
            calc_nd_classes(graph, Options::new(true, false)),
            calc_nd_classes(graph, Options::new(false, true)),
            calc_nd_classes(graph, Options::optimized()),
            calc_nd_classes_improved(graph, Options::naive()),
            calc_nd_classes_improved(graph, Options::optimized()),
            calc_nd_btree(graph),
            calc_nd_btree_degree(graph),
            calc_nd_btree_concurrent(graph, THREAD_COUNT),
        ];

        for partition in partitions {
            // check for correct value of neighborhood diversity
            assert_eq!(partition.len(), expected);
            // check for uniqueness of vertices in partition
            all_unique(partition, vertex_count);
        }
    }

    #[test]
    fn all_algorithms_on_example() {
        let path = "examples/nd_01_shuffled.txt";
        let input = std::fs::read_to_string(path)
            .unwrap_or_else(|error| panic!("error reading '{}': {}", path, error));

        let graph = input
            .parse::<Graph>()
            .unwrap_or_else(|error| panic!("error parsing input: {}", error));

        compare_all(&graph, 6);
    }

    #[test]
    fn all_algorithms_on_example_shuffled() {
        let path = "examples/nd_01_shuffled.txt";
        let input = std::fs::read_to_string(path)
            .unwrap_or_else(|error| panic!("error reading '{}': {}", path, error));

        let mut graph = input
            .parse::<Graph>()
            .unwrap_or_else(|error| panic!("error parsing input: {}", error));

        graph.shuffle();

        compare_all(&graph, 6);
    }

    #[test]
    fn all_algorithms_on_test_graphs() {
        for graph in &test_graphs() {
            let expected = baseline(graph).len();

            compare_all(graph, expected);
        }
    }

    #[test]
    fn all_algorithms_on_test_graphs_shuffled() {
        test_graphs().par_iter_mut().for_each(|graph| {
            let expected = baseline(graph).len();
            graph.shuffle();

            compare_all(graph, expected);
        });
    }

    #[test]
    fn fuzzing() {
        REPRESENTATIONS.into_par_iter().for_each(|&representation| {
            (0..100).into_par_iter().for_each(|_| {
                let mut rng = thread_rng();
                let vertex_count = rng.gen_range(0..=100);
                let probability = rng.gen::<f32>();

                let mut fuzzy_graph =
                    Graph::random_graph(vertex_count, probability, representation);

                let expected = baseline(&fuzzy_graph).len();

                fuzzy_graph.shuffle();

                compare_all(&fuzzy_graph, expected);
            });
        });
    }

    #[test]
    fn fuzzing_nd_limited() {
        REPRESENTATIONS.into_par_iter().for_each(|&representation| {
            (0..100).into_par_iter().for_each(|_| {
                let mut rng = thread_rng();
                let vertex_count = rng.gen_range(2..=100);
                let neighborhood_diversity_limit = rng.gen_range(0..=vertex_count);
                let probability = rng.gen::<f32>();

                let mut fuzzy_graph = Graph::random_graph_nd_limited(
                    vertex_count,
                    probability,
                    neighborhood_diversity_limit,
                    representation,
                );

                let expected = baseline(&fuzzy_graph).len();

                fuzzy_graph.shuffle();

                compare_all(&fuzzy_graph, expected);
            });
        });
    }

    #[test]
    fn empty_graph() {
        REPRESENTATIONS.into_par_iter().for_each(|&representation| {
            let null_graph = Graph::null_graph(0, representation);
            let expected = 0;

            compare_all(&null_graph, expected);
        });
    }

    #[test]
    fn null_graph() {
        REPRESENTATIONS.into_par_iter().for_each(|&representation| {
            let null_graph = Graph::null_graph(VERTEX_COUNT, representation);
            let expected = 1;

            compare_all(&null_graph, expected);
        });
    }

    #[test]
    fn complete_graph() {
        REPRESENTATIONS.into_par_iter().for_each(|&representation| {
            let complete_graph = Graph::complete_graph(VERTEX_COUNT, representation);
            let expected = 1;

            compare_all(&complete_graph, expected);
        });
    }

    #[test]
    fn convert_representation() {
        for graph in &mut test_graphs() {
            let expected = baseline(graph).len();
            match graph.representation() {
                AdjacencyMatrix => graph.convert_representation(AdjacencyList),
                AdjacencyList => graph.convert_representation(AdjacencyMatrix),
            }

            assert_eq!(baseline(graph).len(), expected);
        }
    }
}
