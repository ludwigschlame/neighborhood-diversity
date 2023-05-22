use crate::Options;
use colors_transform::{Color, Hsl};
use network_vis::{network::Network, node_options::NodeOptions};
use rand::{distributions::Uniform, prelude::*};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Representation {
    AdjacencyMatrix,
    AdjacencyList,
}

#[derive(Debug, PartialEq, Eq)]
enum InternalRepresentation {
    // 2d-vector where vec[u][v] == true iff there is an edge between u and v
    AdjacencyMatrix(Vec<Vec<bool>>),
    // vector of vectors where vec[u] contains all neighbors of u
    AdjacencyList(Vec<Vec<usize>>),
}

impl From<&InternalRepresentation> for Representation {
    #[must_use]
    fn from(representation: &InternalRepresentation) -> Self {
        match representation {
            InternalRepresentation::AdjacencyMatrix(_) => Self::AdjacencyMatrix,
            InternalRepresentation::AdjacencyList(_) => Self::AdjacencyList,
        }
    }
}

#[derive(Debug)]
#[cfg_attr(test, derive(PartialEq, Eq))]
pub struct Graph {
    vertex_count: usize,
    representation: InternalRepresentation,
}

impl Graph {
    // constructs a graph with no edges
    #[must_use]
    pub fn null_graph(vertex_count: usize, representation: Representation) -> Self {
        match representation {
            Representation::AdjacencyMatrix => Self {
                vertex_count,
                representation: InternalRepresentation::AdjacencyMatrix(vec![
                    vec![
                        false;
                        vertex_count
                    ];
                    vertex_count
                ]),
            },
            Representation::AdjacencyList => Self {
                vertex_count,
                representation: InternalRepresentation::AdjacencyList(vec![vec![]; vertex_count]),
            },
        }
    }

    // constructs a graph where every distinct pair of vertices is connected by an edge
    #[must_use]
    pub fn complete_graph(vertex_count: usize, representation: Representation) -> Self {
        match representation {
            Representation::AdjacencyMatrix => {
                let mut adjacency_matrix = vec![vec![true; vertex_count]; vertex_count];
                // remove self-loops
                (0..vertex_count).for_each(|i| {
                    adjacency_matrix[i][i] = false;
                });

                Self {
                    vertex_count,
                    representation: InternalRepresentation::AdjacencyMatrix(adjacency_matrix),
                }
            }
            Representation::AdjacencyList => {
                let mut adjacency_list: Vec<Vec<usize>> = Vec::with_capacity(vertex_count);
                for vertex in 0..vertex_count {
                    adjacency_list.push(
                        (0..vertex_count)
                            .filter(|neighbor| *neighbor != vertex)
                            .collect::<Vec<usize>>(),
                    );
                }

                Self {
                    vertex_count,
                    representation: InternalRepresentation::AdjacencyList(adjacency_list),
                }
            }
        }
    }

    // converts graph to the specified representation
    // does nothing if graph is already in the correct representation
    fn convert_representation(graph: Self, representation: Representation) -> Self {
        match (graph.representation(), representation) {
            (r, s) if r == s => graph,
            (Representation::AdjacencyMatrix, Representation::AdjacencyList) => {
                let adjacency_list = (0..graph.vertex_count())
                    .map(|vertex| graph.neighbors(vertex))
                    .collect::<Vec<Vec<usize>>>();

                Self {
                    vertex_count: adjacency_list.len(),
                    representation: InternalRepresentation::AdjacencyList(adjacency_list),
                }
            }
            (Representation::AdjacencyList, Representation::AdjacencyMatrix) => {
                let mut adjacency_matrix =
                    vec![vec![false; graph.vertex_count()]; graph.vertex_count()];

                (0..graph.vertex_count()).for_each(|u| {
                    graph.neighbors(u).iter().for_each(|&v| {
                        adjacency_matrix[u][v] = true;
                        adjacency_matrix[v][u] = true;
                    })
                });

                Self {
                    vertex_count: graph.vertex_count(),
                    representation: InternalRepresentation::AdjacencyMatrix(adjacency_matrix),
                }
            }
            _ => panic!("should be unreachable"),
        }
    }

    // constructs a random graph after Gilbert's model G(n, p)
    // every edge between distinct vertices independently exists with probability p
    #[must_use]
    pub fn random_graph(
        vertex_count: usize,
        probability: f32,
        representation: Representation,
    ) -> Self {
        let mut rng = rand::thread_rng();
        let mut random_graph = Self::null_graph(vertex_count, representation);

        for u in 0..random_graph.vertex_count {
            for v in (u + 1)..random_graph.vertex_count {
                if rng.gen_bool(probability.clamp(0.0, 1.0).into()) {
                    random_graph.insert_edge_unchecked(u, v)
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
    #[must_use]
    pub fn random_graph_nd_limited(
        vertex_count: usize,
        probability: f32,
        neighborhood_diversity_limit: usize,
        representation: Representation,
    ) -> Self {
        let mut rng = rand::thread_rng();
        let generator_graph = Self::random_graph(
            neighborhood_diversity_limit,
            probability,
            Representation::AdjacencyMatrix,
        );
        let mut random_graph = Self::null_graph(vertex_count, Representation::AdjacencyMatrix);

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
            set_start.sort_unstable();
            set_start
        };

        for u_gen in 0..generator_graph.vertex_count {
            let set_end_u = if u_gen == generator_graph.vertex_count - 1 {
                vertex_count
            } else {
                set_start[u_gen + 1]
            };

            // decides wether the neighborhood is a clique or an independent set
            // if neighborhood is a clique, inserts all edges between distinct vertices
            if rng.gen_bool(probability.into()) {
                for u in set_start[u_gen]..set_end_u {
                    for v in (u + 1)..set_end_u {
                        random_graph.insert_edge_unchecked(u, v);
                    }
                }
            }

            // inserts edges between vertex sets based on edges in the generator_graph
            for &v_gen in generator_graph
                .neighbors(u_gen)
                .iter()
                .filter(|&&neighbor| neighbor > u_gen)
            {
                let set_end_v = if v_gen == generator_graph.vertex_count - 1 {
                    vertex_count
                } else {
                    set_start[v_gen + 1]
                };
                for u in set_start[u_gen]..set_end_u {
                    for v in set_start[v_gen]..set_end_v {
                        random_graph.insert_edge_unchecked(u, v);
                    }
                }
            }
        }

        Graph::convert_representation(random_graph, representation)
    }

    // shuffles vertex ids while retaining the original graph structure
    pub fn shuffle(&mut self) {
        let vertex_count = self.vertex_count;
        let mut rng = rand::thread_rng();
        let mut vertex_ids: Vec<usize> = (0..vertex_count).collect();
        vertex_ids.shuffle(&mut rng);

        let mapping: HashMap<usize, usize> = vertex_ids.into_iter().enumerate().collect();

        match &mut self.representation {
            InternalRepresentation::AdjacencyMatrix(adjacency_matrix) => {
                let mut shuffled_adjacency_matrix = vec![vec![false; vertex_count]; vertex_count];

                for u in 0..vertex_count {
                    for v in 0..vertex_count {
                        shuffled_adjacency_matrix[mapping[&u]][mapping[&v]] =
                            adjacency_matrix[u][v];
                    }
                }

                *adjacency_matrix = shuffled_adjacency_matrix;
            }
            InternalRepresentation::AdjacencyList(adjacency_list) => {
                adjacency_list.iter_mut().for_each(|neighborhood| {
                    neighborhood
                        .iter_mut()
                        .for_each(|vertex_id| *vertex_id = mapping[vertex_id])
                });

                let mut shuffled_adjacency_list = vec![vec![]; vertex_count];

                for (idx, idx_mapped) in mapping {
                    shuffled_adjacency_list[idx_mapped] = adjacency_list[idx].clone();
                }

                *adjacency_list = shuffled_adjacency_list;
            }
        }
    }

    // inserts edge (u, v) into the graph
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
        match &mut self.representation {
            InternalRepresentation::AdjacencyMatrix(adjacency_matrix) => {
                let newly_inserted = !adjacency_matrix[u][v];
                adjacency_matrix[u][v] = true;
                adjacency_matrix[v][u] = true;
                Ok(newly_inserted)
            }
            InternalRepresentation::AdjacencyList(adjacency_list) => {
                if adjacency_list[u].contains(&v) {
                    Ok(false)
                } else {
                    adjacency_list[u].push(v);
                    adjacency_list[v].push(u);
                    Ok(true)
                }
            }
        }
    }

    // inserts edge (u, v) into the graph without doing any sanity-checks
    fn insert_edge_unchecked(&mut self, u: usize, v: usize) {
        match &mut self.representation {
            InternalRepresentation::AdjacencyMatrix(adjacency_matrix) => {
                adjacency_matrix[u][v] = true;
                adjacency_matrix[v][u] = true;
            }
            InternalRepresentation::AdjacencyList(adjacency_list) => {
                adjacency_list[u].push(v);
                adjacency_list[v].push(u);
            }
        }
    }

    // removes edge (u, v) from the graph
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
        match &mut self.representation {
            InternalRepresentation::AdjacencyMatrix(adjacency_matrix) => {
                let was_present = adjacency_matrix[u][v];
                adjacency_matrix[u][v] = false;
                adjacency_matrix[v][u] = false;
                Ok(was_present)
            }
            InternalRepresentation::AdjacencyList(adjacency_list) => {
                if adjacency_list[u].contains(&v) {
                    if let Some(pos) = adjacency_list[u].iter().position(|x| *x == v) {
                        adjacency_list[u].remove(pos);
                    }
                    if let Some(pos) = adjacency_list[v].iter().position(|x| *x == u) {
                        adjacency_list[v].remove(pos);
                    }
                    Ok(true)
                } else {
                    Ok(false)
                }
            }
        }
    }

    // returns number of vertices in the graph
    #[must_use]
    pub const fn vertex_count(&self) -> usize {
        self.vertex_count
    }

    #[must_use]
    pub fn representation(&self) -> Representation {
        Representation::from(&self.representation)
    }

    // returns neighbors of given vertex
    // a vertex is not it's own neighbor except for self-loops
    #[must_use]
    pub fn neighbors(&self, vertex: usize) -> Vec<usize> {
        match &self.representation {
            InternalRepresentation::AdjacencyMatrix(adjacency_matrix) => (0..self.vertex_count)
                .filter(|neighbor| adjacency_matrix[vertex][*neighbor])
                .collect(),
            InternalRepresentation::AdjacencyList(adjacency_list) => adjacency_list[vertex].clone(),
        }
    }

    // returns neighbors of given vertex in sorted order
    // takes time O(|V|)
    #[must_use]
    pub fn neighbors_sorted(&self, vertex: usize) -> Vec<usize> {
        let mut neighbors = Vec::new();

        let neighbors = match &self.representation {
            InternalRepresentation::AdjacencyMatrix(adjacency_matrix) => &adjacency_matrix[vertex],
            InternalRepresentation::AdjacencyList(adjacency_list) => {
                neighbors.resize(self.vertex_count, false);
                adjacency_list[vertex]
                    .iter()
                    .for_each(|&neighbor| neighbors[neighbor] = true);
                &neighbors
            }
        };

        (0..self.vertex_count)
            .filter(|&neighbor| neighbors[neighbor])
            .collect()
    }

    // returns neighbors of given vertex as a bool vector
    // takes time O(1) for AdjacencyMatrix and O(|V|) for AdjacencyList
    #[must_use]
    pub fn neighbors_as_bool_vector(&self, vertex: usize) -> Vec<bool> {
        match &self.representation {
            InternalRepresentation::AdjacencyMatrix(adjacency_matrix) => {
                adjacency_matrix[vertex].clone()
            }
            InternalRepresentation::AdjacencyList(adjacency_list) => {
                let mut neighbors = vec![false; self.vertex_count];
                adjacency_list[vertex]
                    .iter()
                    .for_each(|&neighbor| neighbors[neighbor] = true);
                neighbors
            }
        }
    }

    // returns degree of given vertex
    // a vertex is not it's own neighbor except for self-loops
    #[must_use]
    pub fn degree(&self, vertex: usize) -> usize {
        match &self.representation {
            InternalRepresentation::AdjacencyMatrix(adjacency_matrix) => adjacency_matrix[vertex]
                .iter()
                .fold(0, |acc, &is_neighbor| acc + usize::from(is_neighbor)),
            InternalRepresentation::AdjacencyList(adjacency_list) => adjacency_list[vertex].len(),
        }
    }

    // returns actual density of given graph (number of edges / possible edges)
    #[must_use]
    pub fn density(&self) -> f32 {
        match &self.representation {
            InternalRepresentation::AdjacencyMatrix(adjacency_matrix) => {
                adjacency_matrix.iter().fold(0, |acc, row| {
                    acc + row
                    .iter()
                        .fold(0, |acc, &is_neighbor| acc + usize::from(is_neighbor))
                }) as f32
                    / (self.vertex_count * self.vertex_count) as f32
            }
            InternalRepresentation::AdjacencyList(adjacency_list) => {
                // adjacency_list.iter().map(Vec::len).sum::<usize>() as f32
                adjacency_list
                    .iter()
                    .fold(0, |acc, neighborhood| acc + neighborhood.len()) as f32
                    / (self.vertex_count * self.vertex_count) as f32
            }
        }
    }

    // keeps track of visited vertices and starts DFS from unvisited ones
    // for each unvisited vertex a new entry is made in the connected_components vector
    // it will then be filled by the DFS
    #[must_use]
    pub fn connected_components(&self) -> Vec<Vec<usize>> {
        let mut visited = vec![false; self.vertex_count];
        let mut connected_components: Vec<Vec<usize>> = vec![];

        for vertex in 0..self.vertex_count {
            if !visited[vertex] {
                let mut new_component = Vec::new();
                self.depth_first_search(vertex, &mut visited, &mut new_component);
                connected_components.push(new_component);
            }
        }

        connected_components
    }

    // does stack-based DFS in order to avoid recursion
    // 1. moves start on the stack and marks it as visited
    // 2. adds top of stack to connected_component
    // 3. adds unvisited neighbors to top of stack, marks them as visited and goes back to 2.
    fn depth_first_search(
        &self,
        start: usize,
        visited: &mut [bool],
        connected_component: &mut Vec<usize>,
    ) {
        let mut stack = vec![start];
        visited[start] = true;

        while let Some(vertex) = stack.pop() {
            connected_component.push(vertex);

            for neighbor in self.neighbors(vertex) {
                if !visited[neighbor] {
                    // marks visited immediately so vertex isn't pushed onto stack multiple times
                    visited[neighbor] = true;
                    stack.push(neighbor);
                }
            }
        }
    }

    // returns graph in string representation in a format that can be parsed back into a graph
    // first line contains number of vertices
    // following lines list one edge each
    // vertex indices are separated by a comma: u,v
    // adds comments starting with '#'
    #[must_use]
    pub fn export(&self) -> String {
        let mut output = format!("# Number of Vertices\n{}\n\n# Edges\n", self.vertex_count);
        for u in 0..self.vertex_count {
            for v in u..self.vertex_count {
                if match &self.representation {
                    InternalRepresentation::AdjacencyMatrix(adjacency_matrix) => {
                        adjacency_matrix[u][v]
                    }
                    InternalRepresentation::AdjacencyList(adjacency_list) => {
                        adjacency_list[u].contains(&v)
                    }
                } {
                    output.push_str(&format!("{},{}\n", u, v));
                }
            }
        }

        output
    }

    // saves an html document showing the graph in visual form
    // optional: vertices can be colored by group if a coloring vector is provided
    pub fn visualize(&self, path: &str, coloring: Option<&Vec<Vec<usize>>>) {
        const SATURATION: f32 = 80.0;
        const LUMINANCE: f32 = 80.0;
        const DEFAULT_HUE: f32 = 180.0; // Teal
        const VERTEX_SHAPE: &str = "circle";

        let default_color = Hsl::from(DEFAULT_HUE, SATURATION, LUMINANCE)
            .to_rgb()
            .to_css_hex_string();
        let na_color = Hsl::from(0.0, 0.0, 95.0).to_rgb().to_css_hex_string();
        let mut colors: Vec<String> = Vec::new();

        let mut vis_network = Network::new();

        if let Some(coloring) = coloring {
            // selects colors for vertex groups by splitting hue into equal parts
            // avoids borrow checker by generating vector of colors in it's own loop
            let group_count = coloring.len();
            for group_id in 0..group_count {
                let hue = 360.0 / group_count as f32 * group_id as f32;
                colors.push(
                    Hsl::from(hue, SATURATION, LUMINANCE)
                        .to_rgb()
                        .to_css_hex_string(),
                );
            }

            // inserts vertices by group with their corresponding color and group id
            let mut remaining_vertices = (0..self.vertex_count).collect::<HashSet<usize>>();
            for (group_id, color_group) in coloring.iter().enumerate() {
                for vertex_id in color_group {
                    remaining_vertices.remove(vertex_id);
                    vis_network.add_node(
                        *vertex_id as u128,
                        &format!("{}", group_id),
                        Some(vec![
                            NodeOptions::Hex(&colors[group_id]),
                            NodeOptions::Shape(VERTEX_SHAPE),
                        ]),
                    );
                }
            }

            // inserts remaining vertices (if not all vertices are included in the coloring)
            for vertex_id in remaining_vertices {
                vis_network.add_node(
                    vertex_id as u128,
                    "N/A",
                    Some(vec![
                        NodeOptions::Hex(&na_color),
                        NodeOptions::Shape(VERTEX_SHAPE),
                    ]),
                );
            }
        } else {
            // no coloring provided, thus inserts all vertices with default color and no group id
            for vertex_id in 0..self.vertex_count {
                vis_network.add_node(
                    vertex_id as u128,
                    "",
                    Some(vec![
                        NodeOptions::Hex(&default_color),
                        NodeOptions::Shape(VERTEX_SHAPE),
                    ]),
                );
            }
        }

        // inserts edges corresponding to those from the original graph
        for u in 0..self.vertex_count {
            for v in (u + 1)..self.vertex_count {
                if match &self.representation {
                    InternalRepresentation::AdjacencyMatrix(adjacency_matrix) => {
                        adjacency_matrix[u][v]
                    }
                    InternalRepresentation::AdjacencyList(adjacency_list) => {
                        adjacency_list[u].contains(&v)
                    }
                } {
                    vis_network.add_edge(u as u128, v as u128, None, false);
                }
            }
        }

        vis_network.create(path).unwrap();
    }

    // enables the neighborhood density calculation to be called as a method on the graph
    #[must_use]
    pub fn calc_nd_classes(&self, options: Options) -> Vec<Vec<usize>> {
        crate::calc_nd_classes(self, options)
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
            .map(str::trim)
            .filter(|line| !(line.is_empty() || line.starts_with('#') || line.starts_with("//")));
        // first relevant line should contain the number of vertices
        let vertex_count = relevant_lines
            .next()
            .ok_or("input does not contain graph data")?
            .parse()?;
        let mut graph = Self::null_graph(vertex_count, Representation::AdjacencyList);

        // for each remaining line, tries to split once at comma
        // then tries to parse both sides as vertex indices
        // finally, tries inserting new edge into the graph
        relevant_lines.try_for_each(|edge| -> Result<(), Self::Err> {
            let parse_error = format!("expected comma-separated vertex ids, received: '{}'", edge);

            let vertices = edge.split_once(',').ok_or_else(|| parse_error.clone())?;
            let u = vertices.0.parse().map_err(|_| parse_error.clone())?;
            let v = vertices.1.parse().map_err(|_| parse_error.clone())?;
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
            representation: InternalRepresentation::AdjacencyList(vec![
                vec![1, 2],
                vec![0],
                vec![0],
            ]),
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
            representation: InternalRepresentation::AdjacencyList(vec![
                vec![1, 2],
                vec![0],
                vec![0],
            ]),
        };

        assert_eq!(graph_parsed, graph_truth);
    }
}
