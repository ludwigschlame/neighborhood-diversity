mod graph;

pub use graph::Graph;

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
    let mut degrees = Vec::new();
    if options.degree_filter {
        // collect degrees for all vertices
        degrees = (0..graph.vertex_count)
            .map(|vertex| graph.degree(vertex))
            .collect::<Vec<usize>>();
    }

    let mut type_connectivity_graph = Graph::null_graph(graph.vertex_count);

    for u in 0..graph.vertex_count {
        for v in u..graph.vertex_count {
            // only compare neighborhoods if vertices have same degree
            if options.degree_filter && degrees[u] != degrees[v] {
                continue;
            }

            // only compare neighborhoods if v is not already in an equivalence class
            if options.no_unnecessary_type_comparisons
                && !type_connectivity_graph.neighbors(v).is_empty()
            {
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

#[cfg(test)]
// naive algorithm that should be easily verifiable
// used to test correctness of improved algorithms
fn baseline(graph: &Graph) -> Vec<Vec<usize>> {
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

    type_connectivity_graph.connected_components()
}

fn same_type(graph: &Graph, u: usize, v: usize) -> bool {
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
        let random_graph = Graph::random_graph(1e2 as usize, 0.5);

        assert_eq!(
            calc_nd_classes(&random_graph, Options::naive()).len(),
            baseline(&random_graph).len()
        );
    }

    #[test]
    fn degree_filter_vs_baseline() {
        let random_graph = Graph::random_graph(1e2 as usize, 0.5);

        // test algorithm with degree filter against naive implementation
        assert_eq!(
            calc_nd_classes(&random_graph, Options::new(true, false)).len(),
            baseline(&random_graph).len()
        );
    }

    #[test]
    fn optimized_vs_baseline() {
        let random_graph = Graph::random_graph(1e2 as usize, 0.5);

        // test algorithm with degree filter against naive implementation
        assert_eq!(
            calc_nd_classes(&random_graph, Options::optimized()).len(),
            baseline(&random_graph).len()
        );
    }
}
