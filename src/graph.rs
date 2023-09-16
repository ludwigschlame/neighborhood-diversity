mod error;

use error::Error;

use rand::Rng;

#[derive(Debug, Clone)]
#[cfg_attr(test, derive(PartialEq, Eq))]
pub struct Graph {
    adjacency_matrix: Vec<Vec<bool>>,
}

impl Graph {
    pub fn from_adjacency_matrix(adjacency_matrix: Vec<Vec<bool>>) -> Result<Self, Error> {
        let order = adjacency_matrix.len();

        // ensure adjacency matrix is square
        if let Some(row) = adjacency_matrix
            .iter()
            .enumerate()
            .find(|(_, row)| row.len() != order)
        {
            return Err(Error::NotSquare(order, row.0, row.1.len()));
        }

        // ensure there are no self-loops by checking diagonal
        if let Some(vertex) = (0..order).find(|&vertex| adjacency_matrix[vertex][vertex]) {
            return Err(Error::SelfLoop(vertex));
        }

        // ensure adjacency matrix is symmetrical
        for u in 0..order {
            for v in (u + 1)..order {
                if adjacency_matrix[u][v] != adjacency_matrix[v][u] {
                    return Err(Error::NotSymmetrical(u, v));
                }
            }
        }

        Ok(Self { adjacency_matrix })
    }

    /// # Safety
    /// Ensure that the adjacency matrix is square, symmetrical and contains no true values on its diagonal
    #[must_use]
    pub unsafe fn from_adjacency_matrix_unchecked(adjacency_matrix: Vec<Vec<bool>>) -> Self {
        Self { adjacency_matrix }
    }

    // constructs a graph with no edges
    #[must_use]
    pub fn null_graph(order: usize) -> Self {
        Self {
            adjacency_matrix: vec![vec![false; order]; order],
        }
    }

    // constructs a graph where every distinct pair of vertices is connected by an edge
    #[must_use]
    pub fn complete_graph(order: usize) -> Self {
        let mut adjacency_matrix = vec![vec![true; order]; order];
        // remove self-loops
        (0..order).for_each(|i| {
            adjacency_matrix[i][i] = false;
        });

        Self { adjacency_matrix }
    }

    // constructs a random graph after the Gilbert Model G(n, p)
    // every edge between distinct vertices independently exists with probability p
    #[must_use]
    pub fn random_graph<F>(order: usize, probability: F) -> Self
    where
        F: Into<f64>,
    {
        let probability: f64 = probability.into();
        let probability = probability.clamp(0.0, 1.0);

        let mut rng = rand::thread_rng();
        let mut random_graph = Self::null_graph(order);

        for u in 0..order {
            for v in (u + 1)..order {
                if rng.gen_bool(probability) {
                    // SAFETY: each pair of vertices is only visited once.
                    unsafe { random_graph.insert_edge_unchecked(u, v) };
                }
            }
        }

        random_graph
    }

    // inserts edge (u, v) into the graph
    // returns wether the edge was newly inserted:
    // graph did not contain edge: returns true
    // graph already contained edge: returns false
    pub fn insert_edge(&mut self, u: usize, v: usize) -> Result<bool, Error> {
        // returns error if index is out of bounds
        let order = self.order();
        for vertex in [u, v] {
            if vertex >= order {
                return Err(Error::OutOfBounds(order, vertex));
            }
        }
        if u == v {
            return Err(Error::SelfLoop(u));
        }

        // undirected graph -> symmetrical adjacency matrix
        // thus we only need to check for one direction but change both
        let newly_inserted = !self.adjacency_matrix[u][v];
        self.adjacency_matrix[u][v] = true;
        self.adjacency_matrix[v][u] = true;
        Ok(newly_inserted)
    }

    /// # Safety
    /// Ensure that the indices are in bounds and that u != v.
    // inserts edge (u, v) into the graph without doing any sanity-checks
    pub unsafe fn insert_edge_unchecked(&mut self, u: usize, v: usize) {
        self.adjacency_matrix[u][v] = true;
        self.adjacency_matrix[v][u] = true;
    }

    // removes edge (u, v) from the graph
    // returns wether the edge was present:
    // graph did contain the edge: returns true
    // graph did not contain edge: returns false
    pub fn remove_edge(&mut self, u: usize, v: usize) -> Result<bool, Error> {
        // returns error if index is out of bounds
        let order = self.order();
        for vertex in [u, v] {
            if vertex >= order {
                return Err(Error::OutOfBounds(order, vertex));
            }
        }

        // undirected graph -> symmetrical adjacency matrix
        // thus we only need to check for one direction but change both
        let was_present = self.adjacency_matrix[u][v];
        self.adjacency_matrix[u][v] = false;
        self.adjacency_matrix[v][u] = false;
        Ok(was_present)
    }

    /// # Safety
    /// Ensure that the indices are in bounds.
    // inserts edge (u, v) into the graph without doing any sanity-checks
    pub unsafe fn remove_edge_unchecked(&mut self, u: usize, v: usize) {
        self.adjacency_matrix[u][v] = false;
        self.adjacency_matrix[v][u] = false;
    }

    // returns number of vertices in the graph
    #[must_use]
    pub fn order(&self) -> usize {
        self.adjacency_matrix.len()
    }

    // returns neighbors of given vertex
    #[must_use]
    pub fn neighbors(&self, vertex: usize) -> Vec<usize> {
        (0..self.order())
            .filter(|&neighbor| self.adjacency_matrix[vertex][neighbor])
            .collect()
    }

    // returns reference to the vertices row in the adjacency matrix
    #[must_use]
    pub fn neighbors_as_bool_vector(&self, vertex: usize) -> &Vec<bool> {
        &self.adjacency_matrix[vertex]
    }

    // returns true if there is an edge between u and v
    #[must_use]
    pub fn is_edge(&self, u: usize, v: usize) -> bool {
        self.adjacency_matrix[u][v]
    }

    // returns degree of given vertex
    // a vertex is not it's own neighbor
    #[must_use]
    pub fn degree(&self, vertex: usize) -> usize {
        self.neighbors_as_bool_vector(vertex)
            .iter()
            .fold(0, |acc, &is_neighbor| acc + usize::from(is_neighbor))
    }

    // returns graph's density (number of edges / possible edges)
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn density(&self) -> f64 {
        let order = self.order();

        let edge_count = self.adjacency_matrix.iter().fold(0, |acc, row| {
            acc + row
                .iter()
                .fold(0, |acc, &is_neighbor| acc + usize::from(is_neighbor))
        });

        // possible edges := order^2 (counted every edge twice) - order (no self-loops)
        let edge_max = (order as f64).mul_add(order as f64, -(order as f64));

        edge_count as f64 / edge_max
    }

    // saves a text representation that can be parsed back into a graph
    // first line contains number of vertices
    // following lines list one edge each
    // vertex indices are separated by a comma: u,v
    // adds comments starting with '#' and whitespace except when 'raw_output' is true
    pub fn export<P>(&self, path: P, raw_output: bool) -> std::io::Result<()>
    where
        P: AsRef<std::path::Path>,
    {
        let order = self.order();

        let mut output = if raw_output {
            format!("{}\n", order)
        } else {
            format!("# Number of Vertices\n{}\n\n# Edges\n", order)
        };

        for u in 0..order {
            for v in u..order {
                if self.adjacency_matrix[u][v] {
                    output.push_str(&format!("{},{}\n", u, v));
                }
            }
        }

        std::fs::write(path, output)
    }
}

// creates a graph from an input string
// first line contains number of vertices
// following lines list one edge each
// vertex indices are separated by a comma: u,v
// ignores: empty lines; lines starting with '#', '//' or '%'
impl std::str::FromStr for Graph {
    type Err = Error;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        // filter out comments and empty lines
        let mut relevant_lines = input.lines().map(str::trim).filter(|line| {
            !(line.is_empty()
                || line.starts_with('#')
                || line.starts_with("//")
                || line.starts_with('%'))
        });

        // first relevant line should contain the number of vertices
        let order = relevant_lines
            .next()
            .ok_or_else(|| Error::InvalidInput("no lines to parse".to_string()))?
            .parse()
            .map_err(|_| Error::InvalidInput("error parsing order".to_string()))?;

        let mut graph = Self::null_graph(order);

        // for each remaining line, tries to split once at comma
        // then tries to parse both sides as vertex indices
        // finally, tries inserting new edge into the graph
        relevant_lines.try_for_each(|edge| -> Result<(), Self::Err> {
            let parse_error = format!("expected comma-separated vertex ids, received: '{}'", edge);

            let vertices = edge
                .split_once(',')
                .ok_or_else(|| Error::InvalidInput(parse_error.clone()))?;
            let u = vertices
                .0
                .parse()
                .map_err(|_| Error::InvalidInput(parse_error.clone()))?;
            let v = vertices
                .1
                .parse()
                .map_err(|_| Error::InvalidInput(parse_error.clone()))?;
            graph.insert_edge(u, v)?;
            Ok(())
        })?;

        Ok(graph)
    }
}
