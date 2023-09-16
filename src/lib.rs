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
