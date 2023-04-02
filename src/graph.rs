use rand::prelude::*;
use std::collections::HashSet;

#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Graph {
    pub vertex_count: usize,
    pub adjacency_matrix: Vec<Vec<bool>>,
}

impl Graph {
    // constructs a graph with no edges
    pub fn null_graph(vertex_count: usize) -> Self {
        Graph {
            vertex_count,
            adjacency_matrix: vec![vec![false; vertex_count]; vertex_count],
        }
    }

    // constructs a graph where every pair of vertices is connected by an edge
    pub fn complete_graph(vertex_count: usize) -> Self {
        Graph {
            vertex_count,
            adjacency_matrix: vec![vec![true; vertex_count]; vertex_count],
        }
    }

    // constructs a random graph after Gilbert's model G(n, p)
    // every possible edge independently exists with probability p in (0,1)
    pub fn random_graph(
        vertex_count: usize,
        probability: f32,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // check if probability is in open interval (0, 1)
        if !(probability > 0.0 && probability < 1.0) {
            return Err(
                format!("probability: {probability}, expected: 0 < probability < 1").into(),
            );
        }

        let mut rng = rand::thread_rng();
        let mut random_graph = Self::null_graph(vertex_count);

        for u in 0..random_graph.vertex_count {
            for v in u..random_graph.vertex_count {
                if rng.gen::<f32>() <= probability {
                    random_graph.insert_edge(u, v)?;
                }
            }
        }
        Ok(random_graph)
    }

    // inserts edge (u, v) into the adjacency matrix
    // returns wether the edge was newly inserted:
    // graph did not contain edge:   returns true
    // graph already contained edge: returns false
    pub fn insert_edge(&mut self, u: usize, v: usize) -> Result<bool, Box<dyn std::error::Error>> {
        // returns error if index is out of bounds
        if u >= self.vertex_count || v >= self.vertex_count {
            return Err("index out of bounds".into());
        }

        // undirected graph -> symmetrical adjacency matrix
        // thus we only need to check for one direction but change both
        let newly_inserted = !self.adjacency_matrix[u][v];
        self.adjacency_matrix[u][v] = true;
        self.adjacency_matrix[v][u] = true;
        Ok(newly_inserted)
    }

    // removes edge (u, v) from the adjacency matrix
    // returns wether the edge was present:
    // graph did contain the edge: returns true
    // graph did not contain edge: returns false
    pub fn remove_edge(&mut self, u: usize, v: usize) -> Result<bool, Box<dyn std::error::Error>> {
        // returns error if index is out of bounds
        if u >= self.vertex_count || v >= self.vertex_count {
            return Err("index out of bounds".into());
        }

        // undirected graph -> symmetrical adjacency matrix
        // thus we only need to check for one direction but change both
        let was_present = self.adjacency_matrix[u][v];
        self.adjacency_matrix[u][v] = false;
        self.adjacency_matrix[v][u] = false;
        Ok(was_present)
    }

    // returns neighbors as a HashSet to enable neighborhood comparisons
    // a vertex is not it's own neighbor except for self-loops
    pub fn neighbors(&self, vertex: usize) -> HashSet<usize> {
        (0..self.vertex_count)
            .filter(|neighbor| self.adjacency_matrix[vertex][*neighbor])
            .collect()
    }

    // returns degree of given vertex
    // a vertex is not it's own neighbor except for self-loops
    pub fn degree(&self, vertex: usize) -> usize {
        self.adjacency_matrix[vertex]
            .iter()
            .map(|is_neighbor| *is_neighbor as usize)
            .sum::<usize>()
    }

    // keeps track of visited vertices and starts DFS from unvisited ones
    pub fn count_connected_components(&self) -> usize {
        let mut visited = vec![false; self.vertex_count];
        let mut count = 0;

        for vertex in 0..self.vertex_count {
            if !visited[vertex] {
                count += 1;
                self.depth_first_search(vertex, &mut visited);
            }
        }

        count
    }

    // marks start as visited and recursively does DFS from unvisited neighbors
    fn depth_first_search(&self, start: usize, visited: &mut Vec<bool>) {
        visited[start] = true;

        for neighbor in self.neighbors(start) {
            if !visited[neighbor] {
                self.depth_first_search(neighbor, visited);
            }
        }
    }
}

// creates a graph from an input string
// first line contains number of vertices
// following lines list one edge each
// vertex indices are separated by a comma: u,v
// ignores: empty lines; lines starting with '#' or '//'
impl std::str::FromStr for Graph {
    type Err = Box<dyn std::error::Error>;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        // filter out comments and empty lines
        let mut relevant_lines = input
            .lines()
            .map(|line| line.trim())
            .filter(|line| !(line.is_empty() || line.starts_with('#') || line.starts_with("//")));
        // first relevant line should contain the number of vertices
        let vertex_count = relevant_lines
            .next()
            .ok_or("input does not contain graph data")?
            .parse()?;
        let mut graph = Graph::null_graph(vertex_count);

        // for each remaining line, tries to split once at comma
        // then tries to parse both sides as vertex indices
        // finally, tries inserting new edge into the graph
        relevant_lines.try_for_each(|edge| -> Result<(), Self::Err> {
            let parse_error = format!("expected comma-separated vertex ids, received: '{}'", edge);

            let vertices = edge.split_once(',').ok_or(parse_error.clone())?;
            let u = vertices.0.parse().map_err(|_| parse_error.clone())?;
            let v = vertices.1.parse().map_err(|_| parse_error)?;
            graph.insert_edge(u, v)?;
            Ok(())
        })?;

        Ok(graph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_from_str() {
        let input = "3
0,1
0,2
";

        let graph_parsed = input.parse::<Graph>().unwrap();

        let graph_truth = Graph {
            vertex_count: 3,
            adjacency_matrix: vec![
                vec![false, true, true],
                vec![true, false, false],
                vec![true, false, false],
            ],
        };

        assert_eq!(graph_parsed, graph_truth);
    }

    #[test]
    fn graph_from_str_with_comments() {
        let input = "# VERTICES
3
# EDGES
0,1

0,2
// More comments
 //even with space in front
";

        let graph_parsed = input.parse::<Graph>().unwrap();

        let graph_truth = Graph {
            vertex_count: 3,
            adjacency_matrix: vec![
                vec![false, true, true],
                vec![true, false, false],
                vec![true, false, false],
            ],
        };

        assert_eq!(graph_parsed, graph_truth);
    }
}
