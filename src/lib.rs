mod graph;

pub use graph::Graph;

pub enum Algorithm {
    Naive, // O(n^3)
}

pub fn calc_nd(graph: Graph, algorithm: Algorithm) -> usize {
    match algorithm {
        Algorithm::Naive => nd_naive(graph),
    }
}

fn nd_naive(graph: Graph) -> usize {
    let mut type_connectivity_graph = Graph::null_graph(graph.vertex_count);

    for u in 0..graph.vertex_count {
        for v in u..graph.vertex_count {
            let mut u_neighbors = graph.neighbors(u);
            let mut v_neighbors = graph.neighbors(v);
            u_neighbors.remove(&v); // N(u) \ v
            v_neighbors.remove(&u); // N(v) \ u

            if u_neighbors == v_neighbors {
                // u and v have the same type
                type_connectivity_graph
                    .insert_edge(u, v)
                    .expect("u and v are elements of range 0..vertex_count");
            }
        }
    }

    type_connectivity_graph.count_connected_components()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn naive_algorithm() {
        let path = "examples/nd_01.txt";
        let input = std::fs::read_to_string(path)
            .unwrap_or_else(|err| panic!("error reading '{}': {}", path, err));

        let graph = input
            .parse::<Graph>()
            .unwrap_or_else(|err| panic!("error parsing input: {}", err));

        let neighborhood_diversity = calc_nd(graph, Algorithm::Naive);

        assert_eq!(neighborhood_diversity, 6);
    }
}
