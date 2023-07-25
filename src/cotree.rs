use std::collections::HashMap;

use rand::{
    distributions::{Distribution, Uniform},
    seq::SliceRandom,
    thread_rng, Rng,
};

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Operation {
    DisjointUnion,
    DisjointSum,
}

#[derive(Debug, Clone)]
pub enum Cotree {
    // corresponds to an empty graph
    Empty,
    // vertex of co-graph
    // args: (vertex id)
    Leaf(usize),
    // inner node of co-tree
    // args: (vertex count, operation, left child, right child)
    Inner(usize, Operation, Box<Self>, Box<Self>),
}

impl Cotree {
    // calls random tree generator with initial offset 0
    #[must_use]
    pub fn random_tree(vertex_count: usize, density: f32) -> Self {
        Self::_random_tree(vertex_count, density, 0)
    }

    // generates a random co-tree
    // starting from the root, it is randomly decided how many vertices each child tree should have
    // the decision between a disjoint union and disjoint sum is based on the density
    fn _random_tree(vertex_count: usize, density: f32, offset: usize) -> Self {
        if vertex_count == 0 {
            return Self::Empty;
        }
        if vertex_count == 1 {
            return Self::Leaf(offset);
        }

        let mut rng = thread_rng();
        let left_vertex_count = Uniform::from(1..vertex_count).sample(&mut rng);
        let right_vertex_count = vertex_count - left_vertex_count;
        let right_offset = offset + left_vertex_count;
        let left_child = Self::_random_tree(left_vertex_count, density, offset);
        let right_child = Self::_random_tree(right_vertex_count, density, right_offset);

        Self::Inner(
            vertex_count,
            if rng.gen_bool(density.into()) {
                Operation::DisjointSum
            } else {
                Operation::DisjointUnion
            },
            Box::new(left_child),
            Box::new(right_child),
        )
    }

    // returns vertex count
    #[must_use]
    pub const fn vertex_count(&self) -> usize {
        match self {
            Self::Empty => 0,
            Self::Leaf(_) => 1,
            Self::Inner(vertex_count, ..) => *vertex_count,
        }
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        matches!(self, Self::Empty)
    }

    #[must_use]
    pub const fn is_leaf(&self) -> bool {
        matches!(self, Self::Leaf(..))
    }

    #[must_use]
    pub const fn is_inner(&self) -> bool {
        matches!(self, Self::Inner(..))
    }

    pub fn shuffle(&mut self) {
        let mut rng = rand::thread_rng();
        let mut vertex_ids: Vec<usize> = (0..self.vertex_count()).collect();
        vertex_ids.shuffle(&mut rng);
        let mapping: HashMap<usize, usize> = vertex_ids.into_iter().enumerate().collect();

        self._shuffle(&mapping);
    }

    fn _shuffle(&mut self, mapping: &HashMap<usize, usize>) {
        match self {
            Self::Empty => {}
            Self::Leaf(id) => {
                *id = mapping[id];
            }
            Self::Inner(.., left_child, right_child) => {
                left_child._shuffle(mapping);
                right_child._shuffle(mapping);
            }
        }
    }

    #[must_use]
    pub fn leaves(&self) -> Vec<usize> {
        match self {
            Self::Empty => vec![],
            Self::Leaf(id) => vec![*id],
            Self::Inner(.., left_child, right_child) => [left_child, right_child]
                .iter()
                .flat_map(|&child| child.leaves())
                .collect::<Vec<usize>>(),
        }
    }

    // returns neighborhood partition of given co-tree
    #[must_use]
    pub fn neighborhood_partition(&self) -> Vec<Vec<usize>> {
        let mut neighborhood_partition: Vec<Vec<usize>> = vec![];

        let (_, neighborhood_class) = self._neighborhood_partition(&mut neighborhood_partition);

        if !neighborhood_class.is_empty() {
            neighborhood_partition.push(neighborhood_class);
        }

        neighborhood_partition
    }

    // recursively merges equivalence classes from left and right children
    fn _neighborhood_partition(
        &self,
        neighborhood_partition: &mut Vec<Vec<usize>>,
    ) -> (Option<Operation>, Vec<usize>) {
        match self {
            Self::Empty => (None, vec![]),
            Self::Leaf(id) => (None, vec![*id]),
            Self::Inner(_, operation, left_child, right_child) => {
                let mut neighborhood_class = vec![];
                let (left_operation, mut left_class) =
                    left_child._neighborhood_partition(neighborhood_partition);
                let (right_operation, mut right_class) =
                    right_child._neighborhood_partition(neighborhood_partition);

                if left_operation == Some(*operation) || left_child.is_leaf() {
                    neighborhood_class.append(&mut left_class);
                } else if !left_class.is_empty() {
                    neighborhood_partition.push(left_class);
                }
                if right_operation == Some(*operation) || right_child.is_leaf() {
                    neighborhood_class.append(&mut right_class);
                } else if !right_class.is_empty() {
                    neighborhood_partition.push(right_class);
                }

                (Some(*operation), neighborhood_class)
            }
        }
    }
}
