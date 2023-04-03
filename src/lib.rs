mod graph;

pub use graph::Graph;

pub enum Algorithm {
    Naive, // O(n^3)
    DegreeFilter,
}

// #[derive(Debug)]
// pub struct AlgorithmParameters {
//     degree_filter: bool,
//     binary_search: bool,
// }

pub fn calc_nd(graph: &Graph, algorithm: Algorithm) -> usize {
    match algorithm {
        Algorithm::Naive => nd_naive(graph),
        Algorithm::DegreeFilter => nd_degree_filter(graph),
    }
}

fn nd_naive(graph: &Graph) -> usize {
    let mut type_connectivity_graph = Graph::null_graph(graph.vertex_count);

    for u in 0..graph.vertex_count {
        for v in u..graph.vertex_count {
            if same_type(graph, u, v) {
                type_connectivity_graph
                    .insert_edge(u, v)
                    .expect("u and v are elements of range 0..vertex_count");
            }
        }
    }

    type_connectivity_graph.count_connected_components()
}

fn nd_degree_filter(graph: &Graph) -> usize {
    // collect degrees for all vertices
    let degrees = (0..graph.vertex_count)
        .map(|vertex| graph.degree(vertex))
        .collect::<Vec<usize>>();

    let mut type_connectivity_graph = Graph::null_graph(graph.vertex_count);

    for u in 0..graph.vertex_count {
        for v in u..graph.vertex_count {
            // only compare neighborhoods if vertices have same degree
            if degrees[u] != degrees[v] {
                continue;
            }

            if same_type(graph, u, v) {
                type_connectivity_graph
                    .insert_edge(u, v)
                    .expect("u and v are elements of range 0..vertex_count");
            }
        }
    }

    type_connectivity_graph.count_connected_components()
}

pub fn same_type(graph: &Graph, u: usize, v: usize) -> bool {
    let mut u_neighbors = graph.neighbors(u);
    let mut v_neighbors = graph.neighbors(v);
    u_neighbors.remove(&v); // N(u) \ v
    v_neighbors.remove(&u); // N(v) \ u

    u_neighbors == v_neighbors
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

        let neighborhood_diversity = calc_nd(&graph, Algorithm::Naive);

        assert_eq!(neighborhood_diversity, 6);
    }

    #[test]
    fn degree_filter_vs_naive() {
        let random_graph = Graph::random_graph(1e2 as usize, 1e-2);

        // test algorithm with degree filter against naive implementation
        assert_eq!(
            calc_nd(&random_graph, Algorithm::DegreeFilter),
            calc_nd(&random_graph, Algorithm::Naive)
        );
    }
}
