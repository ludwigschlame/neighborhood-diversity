use neighborhood_diversity::prelude::*;
use pretty_assertions::assert_eq;
use rand::{distributions::Uniform, prelude::Distribution, seq::SliceRandom, Rng};
use std::collections::{HashMap, HashSet};

const ORDER_MAX: usize = 101;
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
        // Safety: u is less than graph.order()
        let mut u_neighbors: Vec<bool> = unsafe { graph.get_row_unchecked(u).clone() };
        // Safety: v is less than graph.order()
        let mut v_neighbors: Vec<bool> = unsafe { graph.get_row_unchecked(v).clone() };

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
                    // Safety: u and v are in 0..vertex_count
                    *is_neighbor = unsafe { graph.is_edge_unchecked(mapping[&u], mapping[&v]) };
                });
        });

    // Safety: should be correct if it was correct before.
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
        let mut set_dividers: HashSet<usize> = HashSet::with_capacity(neighborhood_diversity_limit);

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
                    // Safety: each pair of vertices is only visited once.
                    unsafe { random_graph.insert_edge_unchecked(u, v) };
                }
            }
        }

        // inserts edges between vertex sets based on edges in the generator_graph
        // Safety: u_gen is in 0..generator_graph.order()
        for &v_gen in unsafe {
            generator_graph
                .neighbors_unchecked(u_gen)
                .iter()
                .filter(|&&neighbor| neighbor > u_gen)
        } {
            let set_end_v = if v_gen == generator_graph.order() - 1 {
                order
            } else {
                set_start[v_gen + 1]
            };
            for u in set_start[u_gen]..set_end_u {
                for v in set_start[v_gen]..set_end_v {
                    // Safety: each pair of vertices is only visited once.
                    unsafe { random_graph.insert_edge_unchecked(u, v) };
                }
            }
        }
    }

    random_graph
}

fn all_unique(partition: &Vec<Vec<usize>>, order: usize) {
    let mut counter = HashMap::new();

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

    let partitions = &[baseline(graph), calc_neighborhood_partition(graph)];

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
        .unwrap_or_else(|error| panic!("error parsing input: {error}"));

    compare_all(&graph, 6);
}

#[test]
fn all_algorithms_on_example_shuffled() {
    let mut graph = EXAMPLE_GRAPH
        .parse::<Graph>()
        .unwrap_or_else(|error| panic!("error parsing input: {error}"));

    shuffle(&mut graph);

    compare_all(&graph, 6);
}

#[test]
fn empty_graph() {
    let null_graph = Graph::null_graph(0);
    let expected = 0;
    compare_all(&null_graph, expected);
}

#[test]
fn null_graph() {
    let null_graph = Graph::null_graph(ORDER_MAX);
    let expected = 1;
    compare_all(&null_graph, expected);
}

#[test]
fn complete_graph() {
    let complete_graph = Graph::complete_graph(ORDER_MAX);
    let expected = 1;
    compare_all(&complete_graph, expected);
}

#[test]
fn fuzzing_gilbert() {
    (0..TEST_GRAPH_COUNT).for_each(|_| {
        let mut rng = rand::thread_rng();
        let order = rng.gen_range(0..=ORDER_MAX);
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
        let order = rng.gen_range(2..=ORDER_MAX);
        let neighborhood_diversity_limit = rng.gen_range(0..=order);
        let probability: f64 = rng.gen();

        let mut fuzzy_graph =
            random_graph_nd_limited(order, probability, neighborhood_diversity_limit);

        let expected = baseline(&fuzzy_graph).len();

        shuffle(&mut fuzzy_graph);

        compare_all(&fuzzy_graph, expected);
    });
}
