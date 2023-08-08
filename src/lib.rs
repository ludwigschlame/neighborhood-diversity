#![warn(clippy::undocumented_unsafe_blocks)]
#![warn(clippy::use_debug)]
#![warn(clippy::nursery)]
#![warn(clippy::pedantic)]
#![allow(clippy::uninlined_format_args)] // inlined format args don's support batch renaming (yet?)
#![allow(clippy::missing_panics_doc)] // missing docs in general (todo!)
#![allow(clippy::missing_errors_doc)] // missing docs in general (todo!)
#![warn( // useful lints that are allowed by default
    missing_debug_implementations,
    missing_copy_implementations,
    trivial_casts,
    trivial_numeric_casts,
    unused_qualifications
)]

pub mod cotree;
pub mod graph;
pub mod md_tree;
pub mod prelude;

use cotree::Cotree;
use graph::Graph;
use md_tree::MDTree;

use std::collections::BTreeMap;
use std::num::NonZeroUsize;
use std::ops::Range;
use std::sync::mpsc;
use std::thread;

pub type Partition = Vec<Vec<usize>>;

#[derive(Debug, Clone, Copy)]
pub struct Optimizations {
    pub degree_filter: bool,
    pub transitivity: bool,
}

impl Optimizations {
    #[must_use]
    pub const fn new(degree_filter: bool, transitivity: bool) -> Self {
        Self {
            degree_filter,
            transitivity,
        }
    }

    #[must_use]
    pub const fn none() -> Self {
        Self {
            degree_filter: false,
            transitivity: false,
        }
    }

    #[must_use]
    pub const fn all() -> Self {
        Self {
            degree_filter: true,
            transitivity: true,
        }
    }

    #[must_use]
    pub const fn degree_filter() -> Self {
        Self {
            degree_filter: true,
            transitivity: false,
        }
    }

    #[must_use]
    pub const fn transitivity() -> Self {
        Self {
            degree_filter: false,
            transitivity: true,
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
    let before_small_eq = u_neighbors[..small] == v_neighbors[..small];
    let in_between_eq = u_neighbors[small + 1..large] == v_neighbors[small + 1..large];
    let after_large_eq = u_neighbors[large + 1..] == v_neighbors[large + 1..];

    before_small_eq && in_between_eq && after_large_eq
}

#[must_use]
pub fn calc_nd_naive(graph: &Graph, optimizations: Optimizations) -> Partition {
    let vertex_count = graph.vertex_count();
    let mut neighborhood_partition: Partition = vec![];
    let mut classes = vec![None::<usize>; vertex_count];

    // collect degrees for all vertices
    let degrees: Vec<usize> = if optimizations.degree_filter {
        (0..vertex_count)
            .map(|vertex| graph.degree(vertex))
            .collect()
    } else {
        vec![]
    };

    let mut nd: usize = 0;

    for u in 0..vertex_count {
        // only compare neighborhoods if v is not already in an equivalence class
        if optimizations.transitivity && classes[u].is_some() {
            continue;
        }

        if classes[u].is_none() {
            classes[u] = Some(nd);
            neighborhood_partition.push(vec![u]);
            nd += 1;
        }

        for v in (u + 1)..vertex_count {
            if optimizations.transitivity && classes[v].is_some()
                || optimizations.degree_filter && degrees[u] != degrees[v]
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
pub fn calc_nd_naive_concurrent(
    graph: &Graph,
    optimizations: Optimizations,
    thread_count: NonZeroUsize,
) -> Partition {
    struct Data {
        partition: Partition,
        range: (usize, usize),
    }

    let vertex_count = graph.vertex_count();
    let mut neighborhood_partition: Partition;
    let mut thread_data: Vec<Data> = vec![];

    for thread_id in 0..thread_count.into() {
        thread_data.push(Data {
            partition: vec![],
            range: (
                thread_id * vertex_count / thread_count,
                (thread_id + 1) * vertex_count / thread_count,
            ),
        });
    }

    thread::scope(|scope| {
        for data in &mut thread_data {
            scope.spawn(move || {
                let mut classified = vec![false; data.range.1 - data.range.0];

                // collect degrees for all vertices
                let degrees: Vec<usize> = if optimizations.degree_filter {
                    (data.range.0..data.range.1)
                        .map(|vertex| graph.degree(vertex))
                        .collect()
                } else {
                    vec![]
                };

                for u in data.range.0..data.range.1 {
                    if classified[u - data.range.0] {
                        continue;
                    }

                    let mut neighborhood_class = vec![u];
                    for v in (u + 1)..data.range.1 {
                        if optimizations.transitivity && classified[v - data.range.0]
                            || optimizations.degree_filter
                                && degrees[u - data.range.0] != degrees[v - data.range.0]
                        {
                            continue;
                        }

                        if same_type(graph, u, v) {
                            classified[v - data.range.0] = true;
                            neighborhood_class.push(v);
                        }
                    }
                    data.partition.push(neighborhood_class);
                }
            });
        }
    });

    neighborhood_partition = thread_data.pop().unwrap().partition;

    for partition in &mut thread_data {
        for class1 in &mut partition.partition {
            let mut found = false;
            for class2 in &mut neighborhood_partition {
                if same_type(graph, class1[0], class2[0]) {
                    class2.append(class1);
                    found = true;
                    break;
                }
            }
            if !found {
                let mut i = vec![];
                i.append(class1);
                neighborhood_partition.push(i);
            }
        }
    }

    neighborhood_partition
}

#[must_use]
pub fn calc_nd_classes_improved(graph: &Graph, optimizations: Optimizations) -> Partition {
    let vertex_count = graph.vertex_count();
    let mut neighborhood_partition: Partition = vec![];
    let mut classified = vec![false; vertex_count];

    // collect degrees for all vertices
    let degrees: Vec<usize> = if optimizations.degree_filter {
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
            if optimizations.transitivity && classified[v]
                || optimizations.degree_filter && degrees[u] != degrees[v]
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
pub fn calc_nd_merge(graph: &Graph, thread_count: NonZeroUsize) -> Partition {
    let vertex_count = graph.vertex_count();

    if vertex_count == 0 {
        return vec![];
    } else if vertex_count == 1 {
        return vec![vec![0]];
    }

    let (tx, rx) = mpsc::channel();
    thread::scope(|scope| {
        (0..thread_count.into()).for_each(|thread_id| {
            let thread_tx = tx.clone();
            let range = thread_id * vertex_count / thread_count
                ..(thread_id + 1) * vertex_count / thread_count;

            scope.spawn(move || thread_tx.send(_calc_nd_merge(graph, range)));
        });
    });
    drop(tx);

    merge_partitions(graph, rx.iter().collect())
}

#[must_use]
fn _calc_nd_merge(graph: &Graph, range: Range<usize>) -> Partition {
    if range.is_empty() {
        return vec![];
    } else if range.len() == 1 {
        return vec![vec![range.start]];
    }

    let middle = range.start + range.len() / 2;
    merge_partitions(
        graph,
        vec![
            _calc_nd_merge(graph, range.start..middle),
            _calc_nd_merge(graph, middle..range.end),
        ],
    )
}

#[must_use]
fn merge_partitions(graph: &Graph, mut partitions: Vec<Partition>) -> Partition {
    let mut destination_partition = vec![];
    while destination_partition.is_empty() {
        destination_partition = partitions.pop().unwrap();
    }

    for source_partition in partitions {
        for mut source_class in source_partition {
            let mut found = false;
            for destination_class in &mut destination_partition {
                if same_type(graph, destination_class[0], source_class[0]) {
                    destination_class.append(&mut source_class);
                    found = true;
                    break;
                }
            }
            if !found {
                destination_partition.push(source_class);
            }
        }
    }

    destination_partition
}

#[must_use]
pub fn calc_nd_btree(graph: &Graph) -> Partition {
    let mut neighborhood_partition: Partition = Vec::new();
    let mut independent_sets: BTreeMap<&Vec<bool>, usize> = BTreeMap::new();
    let mut cliques: BTreeMap<Vec<bool>, usize> = BTreeMap::new();

    for vertex in 0..graph.vertex_count() {
        let independent_set_type = graph.neighbors_as_bool_vector(vertex);
        let mut clique_type;

        if let Some(&vertex_type) = independent_sets.get(independent_set_type) {
            neighborhood_partition[vertex_type].push(vertex);
        } else if let Some(&vertex_type) = cliques.get({
            clique_type = independent_set_type.clone();
            clique_type[vertex] = true;
            &clique_type
        }) {
            neighborhood_partition[vertex_type].push(vertex);
        } else {
            let vertex_type = neighborhood_partition.len();
            neighborhood_partition.push(vec![vertex]);
            independent_sets.insert(independent_set_type, vertex_type);
            cliques.insert(clique_type, vertex_type);
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

        let independent_set_type: &Vec<bool> = graph.neighbors_as_bool_vector(vertex);
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
    type VertexType = Vec<bool>;

    #[derive(Debug, Default, Clone)]
    struct Data<'is> {
        neighborhood_partition: Partition,
        independent_sets: BTreeMap<&'is Vec<bool>, usize>,
        cliques: BTreeMap<Vec<bool>, usize>,
    }

    let mut thread_data: Vec<Data> = vec![Data::default(); thread_count.into()];

    thread::scope(|scope| {
        for (thread_id, data) in thread_data.iter_mut().enumerate() {
            scope.spawn(move || {
                let range = thread_id * graph.vertex_count() / thread_count
                    ..(thread_id + 1) * graph.vertex_count() / thread_count;

                for vertex in range {
                    let independent_set_type: &VertexType = graph.neighbors_as_bool_vector(vertex);
                    let mut clique_type;

                    if let Some(&vertex_type) = data.independent_sets.get(independent_set_type) {
                        data.neighborhood_partition[vertex_type].push(vertex);
                    } else if let Some(&vertex_type) = data.cliques.get({
                        clique_type = independent_set_type.clone();
                        clique_type[vertex] = true;
                        &clique_type
                    }) {
                        data.neighborhood_partition[vertex_type].push(vertex);
                    } else {
                        let vertex_type = data.neighborhood_partition.len();
                        data.neighborhood_partition.push(vec![vertex]);
                        data.independent_sets
                            .insert(independent_set_type, vertex_type);
                        data.cliques.insert(clique_type, vertex_type);
                    }
                }
            });
        }
    });

    // collect into last element
    let mut collection = thread_data.pop().expect("len is non-zero");

    // merge neighborhood partitions
    for data in &mut thread_data {
        let mut not_found: Vec<(Option<&VertexType>, Option<&VertexType>)> =
            vec![(None, None); data.neighborhood_partition.len()];

        for (clique_type, &class) in &data.cliques {
            if let Some(&get) = collection.cliques.get(clique_type) {
                collection.neighborhood_partition[get]
                    .append(&mut data.neighborhood_partition[class]);
            } else {
                not_found[class].0 = Some(clique_type);
            }
        }

        for (is_type, &class) in &data.independent_sets {
            if not_found[class].0.is_none() {
                continue;
            }
            if let Some(&get) = collection.independent_sets.get(is_type) {
                collection.neighborhood_partition[get]
                    .append(&mut data.neighborhood_partition[class]);
            } else {
                not_found[class].1 = Some(is_type);
            }
        }

        // insert remaining classes into collection
        for (class, (clique_type, is_type)) in not_found
            .iter()
            .enumerate()
            .filter(|(_class, found)| found.1.is_some() && found.0.is_some())
        {
            // insert clique type into collection
            collection
                .cliques
                .insert(clique_type.unwrap().clone(), class);
            // insert independent set type into collection
            collection.independent_sets.insert(is_type.unwrap(), class);
            // add vertices as new neighborhood class
            collection
                .neighborhood_partition
                .push(data.neighborhood_partition[class].clone());
        }
    }

    collection.neighborhood_partition
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;
    use pretty_assertions::assert_eq;
    use rand::Rng;
    use rayon::prelude::{IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

    const VERTEX_COUNT: usize = 100;
    const DENSITY: f32 = 0.5;
    const ND_LIMIT: usize = 20;
    const REPRESENTATIONS: &[graph::Representation] = &[AdjacencyMatrix /* , AdjacencyList */];
    const THREAD_COUNT: NonZeroUsize = {
        // SAFETY: 3 is non-zero.
        unsafe { NonZeroUsize::new_unchecked(3) }
    };

    fn baseline(graph: &Graph) -> Partition {
        // closure replacing the same_type() function
        let same_type = |u: usize, v: usize| -> bool {
            let mut u_neighbors: Vec<bool> = graph.neighbors_as_bool_vector(u).clone();
            let mut v_neighbors: Vec<bool> = graph.neighbors_as_bool_vector(v).clone();

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
            .flat_map(|&representation| {
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
            .collect::<Vec<Graph>>()
    }

    fn all_unique(partition: &Partition, vertex_count: usize) {
        let mut counter = std::collections::HashMap::new();

        for class in partition {
            for &vertex in class {
                *counter.entry(vertex).or_insert(0) += 1;
            }
        }

        if vertex_count == 0 {
            assert_eq!(counter.len(), vertex_count, "counter len != vertex count");
            assert_eq!(counter.keys().min(), None, "counter min != None");
            assert_eq!(counter.keys().max(), None, "counter max != None");
        } else {
            assert_eq!(counter.len(), vertex_count, "counter len != vertex count");
            assert_eq!(counter.keys().min(), Some(&0), "counter min != 0");
            assert_eq!(
                counter.keys().max(),
                Some(&(vertex_count - 1)),
                "counter max != vertex count - 1"
            );
            assert!(counter.values().all(|&count| count == 1), "duplicate value");
        }
    }

    fn compare_all(graph: &Graph, expected: usize) {
        let vertex_count = graph.vertex_count();

        let partitions = &[
            baseline(graph),
            calc_nd_naive(graph, Optimizations::none()),
            calc_nd_naive(graph, Optimizations::degree_filter()),
            calc_nd_naive(graph, Optimizations::transitivity()),
            calc_nd_naive(graph, Optimizations::all()),
            calc_nd_classes_improved(graph, Optimizations::none()),
            calc_nd_classes_improved(graph, Optimizations::degree_filter()),
            calc_nd_classes_improved(graph, Optimizations::transitivity()),
            calc_nd_classes_improved(graph, Optimizations::all()),
            calc_nd_naive_concurrent(graph, Optimizations::none(), THREAD_COUNT),
            calc_nd_naive_concurrent(graph, Optimizations::degree_filter(), THREAD_COUNT),
            calc_nd_naive_concurrent(graph, Optimizations::transitivity(), THREAD_COUNT),
            calc_nd_naive_concurrent(graph, Optimizations::all(), THREAD_COUNT),
            calc_nd_merge(graph, NonZeroUsize::new(1).expect("should be non-zero")),
            calc_nd_merge(graph, THREAD_COUNT),
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
        test_graphs().par_iter_mut().for_each(|graph| {
            let expected = baseline(graph).len();

            compare_all(graph, expected);
        });
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
                let mut rng = rand::thread_rng();
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
                let mut rng = rand::thread_rng();
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
    #[ignore = "currently broken"]
    fn convert_representation() {
        for graph in &mut test_graphs() {
            let expected = baseline(graph).len();
            match graph.representation() {
                AdjacencyMatrix => graph.convert_representation(AdjacencyList),
                AdjacencyList => graph.convert_representation(AdjacencyMatrix),
            };

            assert_eq!(baseline(graph).len(), expected);
        }
    }
}
