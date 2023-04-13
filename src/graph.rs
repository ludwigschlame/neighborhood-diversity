use rand::{distributions::Uniform, prelude::*};
use std::collections::HashSet;

#[derive(Debug)]
#[cfg_attr(test, derive(PartialEq))]
pub struct Graph {
    pub vertex_count: usize,
    pub adjacency_matrix: Vec<Vec<bool>>,
}

impl Graph {
    // constructs a graph with no edges
    pub fn null_graph(vertex_count: usize) -> Self {
        Self {
            vertex_count,
            adjacency_matrix: vec![vec![false; vertex_count]; vertex_count],
        }
    }

    // constructs a graph where every distinct pair of vertices is connected by an edge
    pub fn complete_graph(vertex_count: usize) -> Self {
        let mut adjacency_matrix = vec![vec![true; vertex_count]; vertex_count];
        // remove self-loops
        (0..vertex_count).for_each(|i| {
            adjacency_matrix[i][i] = false;
        });

        Self {
            vertex_count,
            adjacency_matrix,
        }
    }

    // constructs a random graph after Gilbert's model G(n, p)
    // every edge between distinct vertices independently exists with probability p
    pub fn random_graph(vertex_count: usize, probability: f64) -> Self {
        let mut rng = rand::thread_rng();
        let mut random_graph = Self::null_graph(vertex_count);

        for u in 0..random_graph.vertex_count {
            for v in (u + 1)..random_graph.vertex_count {
                if rng.gen_bool(probability.clamp(0.0, 1.0)) {
                    random_graph
                        .insert_edge(u, v)
                        .expect("u and v are in range 0..vertex_count");
                }
            }
        }
        random_graph
    }

    // constructs a random graph in the spirit of Gilbert's model G(n, p)
    // the additional parameter specifies an upper limit for the neighborhood diversity
    // first, a generator graph is constructed by generating a random graph with
    // #neighborhood_diversity_limit many vertices and the given edge probability
    // afterwards, for every vertex in the generator graph, a clique or an independent set
    // (based on the edge probability) is inserted into the resulting graph
    // finally, the sets of vertices are connected by edges analogous to the generator graph
    pub fn random_graph_nd_limited(
        vertex_count: usize,
        probability: f64,
        neighborhood_diversity_limit: usize,
    ) -> Self {
        let mut rng = rand::thread_rng();
        let generator_graph = Self::random_graph(neighborhood_diversity_limit, probability);
        let mut random_graph = Self::null_graph(vertex_count);

        // randomly divides vertices into #neighborhood_diversity_limit many chunks
        // collects these dividers into sorted array as starting positions for the sets
        let set_start: Vec<usize> = {
            // vertex index 0 is reserved for the initial starting position
            let vertex_range = Uniform::from(1..vertex_count);
            let mut set_dividers: HashSet<usize> =
                HashSet::with_capacity(neighborhood_diversity_limit);

            // avoids excessive iterations by generating at most vertex_count / 2 dividers
            if neighborhood_diversity_limit <= vertex_count / 2 {
                // insert into empty HashSet
                set_dividers.insert(0);
                while set_dividers.len() < neighborhood_diversity_limit {
                    set_dividers.insert(vertex_range.sample(&mut rng));
                }
            } else {
                // remove from 'full' HashSet
                set_dividers = (0..vertex_count).collect();
                while set_dividers.len() > neighborhood_diversity_limit {
                    set_dividers.remove(&vertex_range.sample(&mut rng));
                }
            }

            let mut set_start = Vec::from_iter(set_dividers);
            set_start.sort();
            set_start
        };

        for u_gen in 0..generator_graph.vertex_count {
            let set_end_u = match u_gen == generator_graph.vertex_count - 1 {
                true => vertex_count,
                false => set_start[u_gen + 1],
            };

            // decides wether the neighborhood is a clique or an independent set
            // if neighborhood is a clique, inserts all edges between distinct vertices
            if rng.gen_bool(probability) {
                for u in set_start[u_gen]..set_end_u {
                    for v in (u + 1)..set_end_u {
                        random_graph.insert_edge(u, v).unwrap();
                    }
                }
            }

            // inserts edges between vertex sets based on edges in the generator_graph
            for v_gen in generator_graph.neighbors(u_gen) {
                let set_end_v = match v_gen == generator_graph.vertex_count - 1 {
                    true => vertex_count,
                    false => set_start[v_gen + 1],
                };
                for u in set_start[u_gen]..set_end_u {
                    for v in set_start[v_gen]..set_end_v {
                        random_graph.insert_edge(u, v).unwrap();
                    }
                }
            }
        }

        random_graph
    }

    // inserts edge (u, v) into the adjacency matrix
    // returns wether the edge was newly inserted:
    // graph did not contain edge: returns true
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

    // returns actual density of given graph (number of edges / possible edges)
    pub fn density(&self) -> f64 {
        self.adjacency_matrix
            .iter()
            .map(|row| row.iter().filter(|b| **b).count())
            .sum::<usize>() as f64
            / (self.vertex_count * self.vertex_count) as f64
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

    pub fn export(&self) -> String {
        let mut output = format!("# Number of Vertices\n{}\n\n# Edges\n", self.vertex_count);
        for u in 0..self.vertex_count {
            for v in u..self.vertex_count {
                if self.adjacency_matrix[u][v] {
                    output.push_str(&format!("{},{}\n", u, v));
                }
            }
        }

        output
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
        let mut graph = Self::null_graph(vertex_count);

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
