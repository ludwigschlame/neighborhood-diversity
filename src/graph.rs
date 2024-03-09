//! Undirected graph represented by an adjacency matrix.

mod error;

pub use error::{Error, Result};

use rand::Rng;

/// Undirected graph represented by an adjacency matrix.
#[derive(Debug, Clone, Default)]
#[cfg_attr(test, derive(PartialEq, Eq))]
pub struct Graph {
    adjacency_matrix: Vec<Vec<bool>>,
}

impl Graph {
    /// Creates a new [`Graph`] with an order (number of vertices) and size
    /// (number of edges) of zero.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            adjacency_matrix: Vec::new(),
        }
    }

    /// Constructs a graph from a provided adjacency matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// # use neighborhood_diversity::prelude::Graph;
    /// # use neighborhood_diversity::graph::Error;
    /// # fn main() -> Result<(), Error> {
    /// let adjacency_matrix = vec![
    ///     vec![false, true, true],
    ///     vec![true, false, false],
    ///     vec![true, false, false],
    /// ];
    /// let graph = Graph::from_adjacency_matrix(adjacency_matrix)?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// This function will return an error if the provided adjacency matrix:
    /// - is not square.
    /// - is not symmetrical.
    /// - has any true values on its diagonal.
    pub fn from_adjacency_matrix(adjacency_matrix: Vec<Vec<bool>>) -> Result<Self> {
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

    /// Constructs a graph from a provided adjacency matrix.
    ///
    /// # Safety
    ///
    /// Ensure that the adjacency matrix:
    /// - is square.
    /// - is symmetrical.
    /// - has no true values on its diagonal.
    ///
    /// # Examples
    ///
    /// ```
    /// # use neighborhood_diversity::prelude::Graph;
    /// let adjacency_matrix = vec![
    ///     vec![false, true, true],
    ///     vec![true, false, false],
    ///     vec![true, false, false],
    /// ];
    /// let graph = unsafe { Graph::from_adjacency_matrix_unchecked(adjacency_matrix) };
    /// ```
    #[must_use]
    pub unsafe fn from_adjacency_matrix_unchecked(adjacency_matrix: Vec<Vec<bool>>) -> Self {
        Self { adjacency_matrix }
    }

    /// Constructs a [`Graph`] of the given order (number of vertices) and a
    /// size (number of edges) of zero.
    ///
    /// # Examples
    ///
    /// ```
    /// # use neighborhood_diversity::graph::Graph;
    /// let graph = Graph::null_graph(10);
    /// assert_eq!(graph.order(), 10);
    /// assert_eq!(graph.size(), 0);
    /// ```
    #[must_use]
    pub fn null_graph(order: usize) -> Self {
        Self {
            adjacency_matrix: vec![vec![false; order]; order],
        }
    }

    /// Constructs a graph where every distinct pair of vertices is connected
    /// by an edge.
    ///
    /// The size of such a graph is (order<sup>2</sup> - order) / 2
    ///
    /// # Examples
    ///
    /// ```
    /// # use neighborhood_diversity::graph::Graph;
    /// let graph = Graph::complete_graph(10);
    /// assert_eq!(graph.order(), 10);
    /// assert_eq!(graph.size(), (10 * 10 - 10) / 2);
    /// ```
    #[must_use]
    pub fn complete_graph(order: usize) -> Self {
        let mut adjacency_matrix = vec![vec![true; order]; order];
        // remove self-loops
        (0..order).for_each(|i| {
            adjacency_matrix[i][i] = false;
        });

        Self { adjacency_matrix }
    }

    /// Constructs a random graph after the Gilbert Model `G(n, p)`.
    ///
    /// The resulting graph has an order of `n`.
    ///
    /// Every edge between distinct vertices independently exists with
    /// probability `p`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use neighborhood_diversity::graph::Graph;
    /// let graph = Graph::random_graph(10, 0.5);
    /// assert_eq!(graph.order(), 10);
    /// ```
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
                    // Safety: each pair of vertices is only visited once.
                    unsafe { random_graph.insert_edge_unchecked(u, v) };
                }
            }
        }

        random_graph
    }

    /// Resizes the [`Graph`] in-place so that its `order` is equal to
    /// `new_order`.
    ///
    /// If `new_order` is greater than `order`, the [`Graph`] is extended by
    /// the difference, without introducing any new edges.
    ///
    /// If `new_order` is less than `order`, the [`Graph`] is simply truncated.
    pub fn resize(&mut self, new_order: usize) {
        self.adjacency_matrix
            .iter_mut()
            .for_each(|row| row.resize(new_order, false));
        self.adjacency_matrix
            .resize(new_order, vec![false; new_order]);
    }

    /// Inserts edge `{u, v}` into the graph.
    ///
    /// Returns `true` if the edge was newly inserted; `false` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// # use neighborhood_diversity::prelude::Graph;
    /// # use neighborhood_diversity::graph::Error;
    /// # fn main() -> Result<(), Error> {
    /// let mut graph = Graph::null_graph(10);
    /// let newly_inserted = graph.insert_edge(3,8)?;
    /// assert!(newly_inserted);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - `u` or `v` are out of bounds.
    /// - `u == v` (self-loop).
    pub fn insert_edge(&mut self, u: usize, v: usize) -> Result<bool> {
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

    /// Inserts edge `{u, v}` into the graph.
    ///
    /// # Safety
    ///
    /// Ensure that `u` and `v` are in bounds and that `u != v`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use neighborhood_diversity::prelude::Graph;
    /// let mut graph = Graph::null_graph(10);
    /// unsafe { graph.insert_edge_unchecked(3,8) } ;
    /// ```
    pub unsafe fn insert_edge_unchecked(&mut self, u: usize, v: usize) {
        self.adjacency_matrix[u][v] = true;
        self.adjacency_matrix[v][u] = true;
    }

    /// Removes edge `{u, v}` from the graph.
    ///
    /// Returns `true` if the edge was present; `false` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// # use neighborhood_diversity::prelude::Graph;
    /// # use neighborhood_diversity::graph::Error;
    /// # fn main() -> Result<(), Error> {
    /// let mut graph = Graph::complete_graph(10);
    /// let was_present = graph.remove_edge(3,8)?;
    /// assert!(was_present);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// This function will return an error if `u` or `v` are out of bounds.
    pub fn remove_edge(&mut self, u: usize, v: usize) -> Result<bool> {
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

    /// Removes edge `{u, v}` from the graph.
    ///
    /// # Safety
    ///
    /// Ensure that `u` and `v` are in bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// # use neighborhood_diversity::prelude::Graph;
    /// let mut graph = Graph::complete_graph(10);
    /// unsafe { graph.remove_edge_unchecked(3,8) } ;
    /// ```
    pub unsafe fn remove_edge_unchecked(&mut self, u: usize, v: usize) {
        self.adjacency_matrix[u][v] = false;
        self.adjacency_matrix[v][u] = false;
    }

    /// Returns the number of vertices in the graph.
    ///
    /// # Examples
    ///
    /// ```
    /// # use neighborhood_diversity::graph::Graph;
    /// let graph = Graph::complete_graph(10);
    /// assert_eq!(graph.order(), 10);
    /// ```
    #[must_use]
    pub fn order(&self) -> usize {
        self.adjacency_matrix.len()
    }

    /// Returns a reference to the adjacency matrix of the [`Graph`].
    #[must_use]
    pub const fn adjacency_matrix(&self) -> &Vec<Vec<bool>> {
        &self.adjacency_matrix
    }

    /// Returns the number of edges in the graph.
    ///
    /// # Examples
    ///
    /// ```
    /// # use neighborhood_diversity::graph::Graph;
    /// let graph = Graph::complete_graph(10);
    /// assert_eq!(graph.size(), (10 * 10 - 10) / 2);
    /// ```
    #[must_use]
    pub fn size(&self) -> usize {
        self.adjacency_matrix.iter().fold(0, |acc, row| {
            acc + row
                .iter()
                .fold(0, |acc, &is_neighbor| acc + usize::from(is_neighbor))
        }) / 2
    }

    /// Returns the neighbors of the given vertex as a [`Vec`].
    ///
    /// # Examples
    ///
    /// ```
    /// # use neighborhood_diversity::graph::Graph;
    /// # use neighborhood_diversity::graph::Error;
    /// # fn main() -> Result<(), Error> {
    /// let adjacency_matrix = vec![
    ///     vec![false, true, true],
    ///     vec![true, false, false],
    ///     vec![true, false, false],
    /// ];
    /// let graph = Graph::from_adjacency_matrix(adjacency_matrix)?;
    /// assert_eq!(graph.neighbors(0), vec![1, 2]);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn neighbors(&self, vertex: usize) -> Vec<usize> {
        (0..self.order())
            .filter(|&neighbor| self.adjacency_matrix[vertex][neighbor])
            .collect()
    }

    /// Returns a reference to the corresponding row in the adjacency matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// # use neighborhood_diversity::graph::Graph;
    /// # use neighborhood_diversity::graph::Error;
    /// # fn main() -> Result<(), Error> {
    /// let adjacency_matrix = vec![
    ///     vec![false, true, true],
    ///     vec![true, false, false],
    ///     vec![true, false, false],
    /// ];
    /// let graph = Graph::from_adjacency_matrix(adjacency_matrix)?;
    /// assert_eq!(graph.get_row(0)?, &vec![false, true, true]);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an [`Error`] if `vertex` is greater than or equal to the order of the [`Graph`].
    pub fn get_row(&self, vertex: usize) -> Result<&Vec<bool>> {
        if vertex >= self.order() {
            Err(Error::OutOfBounds(self.order(), vertex))
        } else {
            Ok(&self.adjacency_matrix[vertex])
        }
    }

    /// Returns a reference to the corresponding row in the adjacency matrix.
    ///
    /// # Safety
    ///
    /// Ensure that `vertex` is smaller than the order of the [`Graph`]
    ///
    /// # Examples
    ///
    /// ```
    /// # use neighborhood_diversity::graph::Graph;
    /// # use neighborhood_diversity::graph::Error;
    /// # fn main() -> Result<(), Error> {
    /// let adjacency_matrix = vec![
    ///     vec![false, true, true],
    ///     vec![true, false, false],
    ///     vec![true, false, false],
    /// ];
    /// let graph = Graph::from_adjacency_matrix(adjacency_matrix)?;
    /// assert_eq!(unsafe { graph.get_row_unchecked(0) }, &vec![false, true, true]);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub unsafe fn get_row_unchecked(&self, vertex: usize) -> &Vec<bool> {
        &self.adjacency_matrix[vertex]
    }

    /// Returns `true` if there is an edge between `u` and `v`.
    #[must_use]
    pub fn is_edge(&self, u: usize, v: usize) -> bool {
        self.adjacency_matrix[u][v]
    }

    /// Returns the degree of the given vertex.
    ///
    /// # Errors
    ///
    /// Returns an [`Error`] if `vertex` is greater than or equal to the order of the [`Graph`].
    pub fn degree(&self, vertex: usize) -> Result<usize> {
        Ok(self
            .get_row(vertex)?
            .iter()
            .fold(0, |acc, &is_neighbor| acc + usize::from(is_neighbor)))
    }

    /// Returns the density of the graph (size of graph / max possible size).
    ///
    /// # Examples
    ///
    /// ```
    /// # use neighborhood_diversity::graph::Graph;
    /// let graph = Graph::complete_graph(10);
    /// assert_eq!(graph.density(), 1.0);
    /// ```
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn density(&self) -> f64 {
        let order = self.order();

        let edge_count = self.size();

        // possible edges := (order^2 (counts every edge twice) - order (no self-loops)) / 2.0
        let edge_max = (order as f64).mul_add(order as f64, -(order as f64)) / 2.0;

        edge_count as f64 / edge_max
    }

    /// Saves the graph in a text representation.
    ///
    /// The first line contains the order of the graph; subsequent lines list
    /// one edge each.
    ///
    /// Vertex indices are separated by a comma: `u,v`
    ///
    /// Adds whitespace and comments starting with `#` except when `raw_output`
    /// is true.
    ///
    /// # Errors
    ///
    /// This function will return an error if [`write`] returns an error.
    ///
    /// [`write`]: std::fs::write
    pub fn export<P>(&self, path: P, raw_output: bool) -> std::io::Result<()>
    where
        P: AsRef<std::path::Path>,
    {
        let order = self.order();

        let mut output = if raw_output {
            format!("{order}\n")
        } else {
            format!("# Number of Vertices\n{order}\n\n# Edges\n")
        };

        for u in 0..order {
            for v in u..order {
                if self.adjacency_matrix[u][v] {
                    output.push_str(&format!("{u},{v}\n"));
                }
            }
        }

        std::fs::write(path, output)
    }

    /// Returns the neighborhood partition of the graph.
    ///
    /// Acts as a wrapper around [`calc_neighborhood_partition()`].
    ///
    /// [`calc_neighborhood_partition()`]: crate::calc_neighborhood_partition()
    #[must_use]
    pub fn neighborhood_partition(&self) -> Vec<Vec<usize>> {
        crate::calc_neighborhood_partition(self)
    }
}

/// Creates a graph from an input string.
///
/// The first line contains number of vertices; following lines list one edge
/// each.
///
/// Vertex indices are separated by a comma: `u,v`.
///
/// Ignores: empty lines; lines starting with `#`, `//` or `%`.
impl std::str::FromStr for Graph {
    type Err = Error;

    fn from_str(input: &str) -> Result<Self> {
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
        relevant_lines.try_for_each(|edge| -> Result<()> {
            let parse_error = format!("expected comma-separated vertex ids, received: '{edge}'");

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
