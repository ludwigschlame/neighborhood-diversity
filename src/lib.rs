//! Crate for computing the neighborhood diversity of simple, undirected
//! graphs.
//!
//! # Quick Start
//! ```
//! use neighborhood_diversity::prelude::*;
//!
//! let graph = Graph::random_graph(10, 0.1);
//! let neighborhood_partition = calc_neighborhood_partition(&graph);
//! let neighborhood_diversity = neighborhood_partition.len();
//! ```

pub mod graph;
pub mod prelude;

pub use graph::Graph;

use std::collections::BTreeMap;

/// Returns the optimal neighborhood partition of the provided graph.
///
/// Has an asymptotical running time of O(n<sup>2</sup> log n) where n is the
/// order of the graph.
///
/// # Examples
///
/// ```
/// # use neighborhood_diversity::graph::Graph;
/// # use neighborhood_diversity::calc_neighborhood_partition;
/// let graph = Graph::random_graph(10, 0.1);
/// let neighborhood_partition = calc_neighborhood_partition(&graph);
/// ```
#[must_use]
pub fn calc_neighborhood_partition(graph: &Graph) -> Vec<Vec<usize>> {
    let mut neighborhood_partition: Vec<Vec<usize>> = Vec::new();
    let mut independent_sets: BTreeMap<&Vec<bool>, usize> = BTreeMap::new();
    let mut cliques: BTreeMap<Vec<bool>, usize> = BTreeMap::new();

    for vertex in 0..graph.order() {
        // Safety: vertex is less than graph.order()
        let independent_set_type = unsafe { graph.get_row_unchecked(vertex) };
        let mut clique_type; // will only be constructed if first search fails

        if let Some(&vertex_type) = independent_sets.get(independent_set_type) {
            // vertex type found in the 'independent set' BTree
            neighborhood_partition[vertex_type].push(vertex);
        } else if let Some(&vertex_type) = cliques.get({
            clique_type = independent_set_type.clone();
            clique_type[vertex] = true;
            &clique_type
        }) {
            // vertex type found in the 'clique' BTree
            neighborhood_partition[vertex_type].push(vertex);
        } else {
            // vertex type found in neither BTree
            // create new class and insert types into both BTrees
            let vertex_type = neighborhood_partition.len();
            neighborhood_partition.push(vec![vertex]);
            independent_sets.insert(independent_set_type, vertex_type);
            cliques.insert(clique_type, vertex_type);
        }
    }

    neighborhood_partition
}
