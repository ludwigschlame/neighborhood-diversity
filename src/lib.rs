// useful lints that are allowed by default
#![warn(
    missing_debug_implementations,
    missing_copy_implementations,
    trivial_casts,
    trivial_numeric_casts,
    unused_qualifications
)]
// enable more aggressive clippy lints
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]
#![warn(clippy::undocumented_unsafe_blocks)]
#![warn(clippy::use_debug)]
// disable lints that are too aggressive
#![allow(clippy::uninlined_format_args)] // inlined format args don't support F2 batch renaming (yet?)
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::missing_errors_doc)]

pub mod graph;
pub mod prelude;

use graph::Graph;

use std::collections::BTreeMap;

// returns true if u and v are of the same type
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
pub fn calc_nd_naive(graph: &Graph) -> Vec<Vec<usize>> {
    let order = graph.order();
    let mut neighborhood_partition: Vec<Vec<usize>> = Vec::new();
    let mut classified = vec![false; order];

    // collect degrees for all vertices
    let degrees: Vec<usize> = (0..order).map(|vertex| graph.degree(vertex)).collect();

    for u in 0..order {
        if classified[u] {
            continue;
        }

        let mut neighborhood_class = vec![u];
        for v in (u + 1)..order {
            if classified[v] || degrees[u] != degrees[v] {
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
pub fn calc_nd_btree(graph: &Graph) -> Vec<Vec<usize>> {
    let mut partition: Vec<Vec<usize>> = Vec::new();
    let mut independent_sets: BTreeMap<&Vec<bool>, usize> = BTreeMap::new();
    let mut cliques: BTreeMap<Vec<bool>, usize> = BTreeMap::new();

    for vertex in 0..graph.order() {
        let independent_set_type = graph.neighbors_as_bool_vector(vertex);
        let mut clique_type; // will only be constructed if first search fails

        if let Some(&vertex_type) = independent_sets.get(independent_set_type) {
            // vertex type found in the 'independent set' BTree
            partition[vertex_type].push(vertex);
        } else if let Some(&vertex_type) = cliques.get({
            clique_type = independent_set_type.clone();
            clique_type[vertex] = true;
            &clique_type
        }) {
            // vertex type found in the 'clique' BTree
            partition[vertex_type].push(vertex);
        } else {
            // vertex type found in neither BTree
            // create new class and insert types into both BTrees
            let vertex_type = partition.len();
            partition.push(vec![vertex]);
            independent_sets.insert(independent_set_type, vertex_type);
            cliques.insert(clique_type, vertex_type);
        }
    }

    partition
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;
    use pretty_assertions::assert_eq;
    use rand::{distributions::Uniform, prelude::Distribution, seq::SliceRandom, Rng};
    // use rayon::prelude::{IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
    use std::collections::{HashMap, HashSet};

    const ORDER: usize = 101;
    const DENSITY: f64 = 0.1;
    const ND_LIMIT: usize = 11;
    const TEST_GRAPH_COUNT: usize = 10;

    // graph with a neighborhood diversity of 6
    const EXAMPLE_GRAPH: &str = "# Number of vertices
12

# Edges
0,1
0,2
1,2
2,9
2,10
2,11
2,3
3,9
3,10
3,11
3,4
3,5
4,6
4,7
4,8
5,6
5,7
5,8
6,7
6,8
7,8
9,10
9,11
10,11
";

    fn baseline(graph: &Graph) -> Vec<Vec<usize>> {
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

        let order: usize = graph.order();
        let mut partition: Vec<Vec<usize>> = Vec::new();
        let mut classes: Vec<Option<usize>> = vec![None; order];
        let mut nd: usize = 0;

        for u in 0..order {
            if classes[u].is_none() {
                classes[u] = Some(nd);
                partition.resize((nd + 1).max(partition.len()), vec![]);
                partition[nd].push(u);
                nd += 1;
            }
            for v in (u + 1)..order {
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

    // shuffles vertex ids while retaining the original graph structure
    pub fn shuffle(graph: &mut Graph) -> &mut Graph {
        let vertex_count = graph.order();
        let mut rng = rand::thread_rng();
        let mut vertex_ids: Vec<usize> = (0..vertex_count).collect();
        vertex_ids.shuffle(&mut rng);

        let mapping: HashMap<usize, usize> = vertex_ids.into_iter().enumerate().collect();

        let mut shuffled_adjacency_matrix = vec![vec![false; vertex_count]; vertex_count];

        shuffled_adjacency_matrix
            .iter_mut()
            .enumerate()
            .for_each(|(u, neighborhood)| {
                neighborhood
                    .iter_mut()
                    .enumerate()
                    .for_each(|(v, is_neighbor)| {
                        *is_neighbor = graph.is_edge(mapping[&u], mapping[&v]);
                    });
            });

        // SAFETY: should be correct if it was correct before.
        *graph = unsafe { Graph::from_adjacency_matrix_unchecked(shuffled_adjacency_matrix) };
        graph
    }

    // constructs a random graph in the spirit of Gilbert's model G(n, p)
    // the additional parameter specifies an upper limit for the neighborhood diversity
    // first, a generator graph is constructed by generating a random graph with
    // #neighborhood_diversity_limit many vertices and the given edge probability
    // afterwards, for every vertex in the generator graph, a clique or an independent set
    // (based on the edge probability) is inserted into the resulting graph
    // finally, the sets of vertices are connected by edges analogous to the generator graph
    #[must_use]
    pub fn random_graph_nd_limited(
        order: usize,
        probability: f64,
        neighborhood_diversity_limit: usize,
    ) -> Graph {
        let mut rng = rand::thread_rng();
        let generator_graph = Graph::random_graph(neighborhood_diversity_limit, probability);
        let mut random_graph = Graph::null_graph(order);

        // randomly divides vertices into #neighborhood_diversity_limit many chunks
        // collects these dividers into sorted array as starting positions for the sets
        let set_start: Vec<usize> = {
            // vertex index 0 is reserved for the initial starting position
            let vertex_range = Uniform::from(1..=order);
            let mut set_dividers: HashSet<usize> =
                HashSet::with_capacity(neighborhood_diversity_limit);

            // avoids excessive iterations by generating at most vertex_count / 2 dividers
            if neighborhood_diversity_limit <= order / 2 {
                // insert into empty HashSet
                set_dividers.insert(0);
                while set_dividers.len() < neighborhood_diversity_limit {
                    set_dividers.insert(vertex_range.sample(&mut rng));
                }
            } else {
                // remove from 'full' HashSet
                set_dividers = (0..order).collect();
                while set_dividers.len() > neighborhood_diversity_limit {
                    set_dividers.remove(&vertex_range.sample(&mut rng));
                }
            }

            let mut set_start = Vec::from_iter(set_dividers);
            set_start.sort_unstable();
            set_start
        };

        for u_gen in 0..generator_graph.order() {
            let set_end_u = if u_gen == generator_graph.order() - 1 {
                order
            } else {
                set_start[u_gen + 1]
            };

            // decides wether the neighborhood is a clique or an independent set
            // if neighborhood is a clique, inserts all edges between distinct vertices
            if rng.gen_bool(probability) {
                for u in set_start[u_gen]..set_end_u {
                    for v in (u + 1)..set_end_u {
                        // SAFETY: each pair of vertices is only visited once.
                        unsafe { random_graph.insert_edge_unchecked(u, v) };
                    }
                }
            }

            // inserts edges between vertex sets based on edges in the generator_graph
            for &v_gen in generator_graph
                .neighbors(u_gen)
                .iter()
                .filter(|&&neighbor| neighbor > u_gen)
            {
                let set_end_v = if v_gen == generator_graph.order() - 1 {
                    order
                } else {
                    set_start[v_gen + 1]
                };
                for u in set_start[u_gen]..set_end_u {
                    for v in set_start[v_gen]..set_end_v {
                        // SAFETY: each pair of vertices is only visited once.
                        unsafe { random_graph.insert_edge_unchecked(u, v) };
                    }
                }
            }
        }

        random_graph
    }

    fn test_graphs() -> Vec<Graph> {
        (0..TEST_GRAPH_COUNT)
            .map(|_| random_graph_nd_limited(ORDER, DENSITY, ND_LIMIT))
            .collect::<Vec<Graph>>()
    }

    fn all_unique(partition: &Vec<Vec<usize>>, order: usize) {
        let mut counter = std::collections::HashMap::new();

        for class in partition {
            for &vertex in class {
                *counter.entry(vertex).or_insert(0) += 1;
            }
        }

        if order == 0 {
            assert_eq!(counter.len(), order, "counter len != order");
            assert_eq!(counter.keys().min(), None, "counter min != None");
            assert_eq!(counter.keys().max(), None, "counter max != None");
        } else {
            assert_eq!(counter.len(), order, "counter len != order");
            assert_eq!(counter.keys().min(), Some(&0), "counter min != 0");
            assert_eq!(
                counter.keys().max(),
                Some(&(order - 1)),
                "counter max != order - 1"
            );
            assert!(counter.values().all(|&count| count == 1), "duplicate value");
        }
    }

    fn compare_all(graph: &Graph, expected: usize) {
        let order = graph.order();

        let partitions = &[baseline(graph), calc_nd_naive(graph), calc_nd_btree(graph)];

        for partition in partitions {
            // check for correct value of neighborhood diversity
            assert_eq!(partition.len(), expected);
            // check for uniqueness of vertices in partition
            all_unique(partition, order);
        }
    }

    #[test]
    fn all_algorithms_on_example() {
        let graph = EXAMPLE_GRAPH
            .parse::<Graph>()
            .unwrap_or_else(|error| panic!("error parsing input: {}", error));

        compare_all(&graph, 6);
    }

    #[test]
    fn all_algorithms_on_example_shuffled() {
        let mut graph = EXAMPLE_GRAPH
            .parse::<Graph>()
            .unwrap_or_else(|error| panic!("error parsing input: {}", error));

        shuffle(&mut graph);

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
        for graph in &mut test_graphs() {
            let expected = baseline(graph).len();

            shuffle(graph);

            compare_all(graph, expected);
        }
    }

    #[test]
    fn fuzzing_gilbert() {
        (0..TEST_GRAPH_COUNT).for_each(|_| {
            let mut rng = rand::thread_rng();
            let order = rng.gen_range(0..=ORDER);
            let probability = rng.gen::<f64>();

            let mut fuzzy_graph = Graph::random_graph(order, probability);

            let expected = baseline(&fuzzy_graph).len();

            shuffle(&mut fuzzy_graph);

            compare_all(&fuzzy_graph, expected);
        });
    }

    #[test]
    fn fuzzing_nd_limit() {
        (0..TEST_GRAPH_COUNT).for_each(|_| {
            let mut rng = rand::thread_rng();
            let order = rng.gen_range(2..=ORDER);
            let neighborhood_diversity_limit = rng.gen_range(0..=order);
            let probability: f64 = rng.gen();

            let mut fuzzy_graph =
                random_graph_nd_limited(order, probability, neighborhood_diversity_limit);

            let expected = baseline(&fuzzy_graph).len();

            shuffle(&mut fuzzy_graph);

            compare_all(&fuzzy_graph, expected);
        });
    }

    #[test]
    fn empty_graph() {
        let null_graph = Graph::null_graph(0);
        let expected = 0;
        compare_all(&null_graph, expected);
    }

    #[test]
    fn null_graph() {
        let null_graph = Graph::null_graph(ORDER);
        let expected = 1;
        compare_all(&null_graph, expected);
    }

    #[test]
    fn complete_graph() {
        let complete_graph = Graph::complete_graph(ORDER);
        let expected = 1;
        compare_all(&complete_graph, expected);
    }
}
