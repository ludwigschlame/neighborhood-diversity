use rand::{distributions::Uniform, prelude::*};

use crate::{calc_nd_classes, graph::Graph, CoTree, Options};

#[derive(Debug, Clone)]
pub enum MDTree {
    // corresponds to an empty graph
    Empty,
    // vertex of graph
    // args: (vertex id)
    Leaf(usize),
    // inner node of md-tree
    // args: (vertex count, node type, children)
    Inner(usize, NodeType, Vec<MDTree>),
}

#[derive(Debug, Clone)]
pub enum NodeType {
    Prime(Box<Graph>),
    Series,
    Parallel,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum ClassType {
    Clique,
    IndependentSet,
}

impl MDTree {
    // // calls random tree generator with initial offset 0
    // #[must_use]
    // pub fn random_tree(vertex_count: usize, density: f32, modular_width: usize) -> Self {
    //     Self::_random_tree(vertex_count, density, modular_width, 0)
    // }

    // // generates a random co-tree
    // // starting from the root, it is randomly decided how many vertices each child tree should have
    // // the decision between a disjoint union and disjoint sum is based on the density
    // fn _random_tree(
    //     vertex_count: usize,
    //     density: f32,
    //     modular_width: usize,
    //     offset: usize,
    // ) -> Self {
    //     if vertex_count == 0 {
    //         return Self::Empty;
    //     }
    //     if vertex_count == 1 {
    //         return Self::Leaf(offset);
    //     }

    //     let mut rng = thread_rng();
    //     let child_count = Uniform::from(1..vertex_count).sample(&mut rng);
    //     let left_vertex_count = Uniform::from(1..vertex_count).sample(&mut rng);
    //     let right_vertex_count = vertex_count - left_vertex_count;
    //     let left_child = Self::_random_tree(left_vertex_count, density, modular_width, offset);
    //     let right_child = Self::_random_tree(
    //         right_vertex_count,
    //         density,
    //         modular_width,
    //         offset + left_vertex_count,
    //     );

    //     Self::Inner(
    //         vertex_count,
    //         if rng.gen_bool(density.into()) {
    //             NodeType::Parallel
    //         } else {
    //             NodeType::Series
    //         },
    //         Box::new(left_child),
    //         Box::new(right_child),
    //     )
    // }

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

    // pub fn shuffle(&mut self) {
    //     let mut rng = rand::thread_rng();
    //     let mut vertex_ids: Vec<usize> = (0..self.vertex_count()).collect();
    //     vertex_ids.shuffle(&mut rng);
    //     let mapping: HashMap<usize, usize> = vertex_ids.into_iter().enumerate().collect();

    //     self._shuffle(&mapping);
    // }

    // fn _shuffle(&mut self, mapping: &HashMap<usize, usize>) {
    //     match self {
    //         Self::Empty => {}
    //         Self::Leaf(id) => {
    //             *id = mapping[id];
    //         }
    //         Self::Inner(.., left_child, right_child) => {
    //             left_child._shuffle(mapping);
    //             right_child._shuffle(mapping);
    //         }
    //     }
    // }

    #[must_use]
    pub fn leaves(&self) -> Vec<usize> {
        match self {
            Self::Empty => vec![],
            Self::Leaf(id) => vec![*id],
            Self::Inner(.., children) => children
                .iter()
                .map(Self::leaves)
                .collect::<Vec<Vec<usize>>>()
                .concat(),
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

    fn _neighborhood_partition(
        &self,
        neighborhood_partition: &mut Vec<Vec<usize>>,
    ) -> (Option<ClassType>, Vec<usize>) {
        match self {
            Self::Empty => (None, vec![]),
            Self::Leaf(id) => (None, vec![*id]),
            Self::Inner(_, node_type, children) => {
                let mut neighborhood_class: Vec<usize> = vec![];

                let mut results = children
                    .iter()
                    .map(|child| child._neighborhood_partition(neighborhood_partition))
                    .collect::<Vec<(Option<ClassType>, Vec<usize>)>>();

                match node_type {
                    NodeType::Prime(graph) => {
                        let classes = calc_nd_classes(graph, Options::optimized());
                        let mut class_type = None;

                        for class in classes {
                            let mut new_class: Vec<usize> = vec![];
                            let is_clique =
                                graph.is_edge(*class.first().unwrap(), *class.last().unwrap());
                            if is_clique {
                                // CL
                                for &idx in &class {
                                    if results[idx].0 == Some(ClassType::Clique)
                                        || results[idx].0.is_none()
                                    {
                                        new_class.append(&mut results[idx].1);
                                    } else if !results[idx].1.is_empty() {
                                        neighborhood_partition.push(results[idx].1.clone());
                                    }
                                }
                            } else {
                                // IS
                                for &idx in &class {
                                    if results[idx].0 == Some(ClassType::IndependentSet)
                                        || results[idx].0.is_none()
                                    {
                                        new_class.append(&mut results[idx].1);
                                    } else if !results[idx].1.is_empty() {
                                        neighborhood_partition.push(results[idx].1.clone());
                                    }
                                }
                            }

                            if new_class.is_empty() {
                                continue;
                            }

                            if graph.degree(class[0]) == 0 {
                                neighborhood_class = new_class;
                                class_type = Some(ClassType::IndependentSet);
                            } else if graph.degree(class[0]) == graph.vertex_count() {
                                neighborhood_class = new_class;
                                class_type = Some(ClassType::Clique);
                            } else {
                                neighborhood_partition.push(new_class);
                            }
                        }

                        (class_type, neighborhood_class)
                    }
                    NodeType::Series => {
                        for result in &mut results {
                            if result.0 == Some(ClassType::Clique) || result.0.is_none() {
                                neighborhood_class.append(&mut result.1);
                            } else if !result.1.is_empty() {
                                neighborhood_partition.push(result.1.clone());
                            }
                        }

                        (Some(ClassType::Clique), neighborhood_class)
                    }
                    NodeType::Parallel => {
                        for result in &mut results {
                            if result.0 == Some(ClassType::IndependentSet) || result.0.is_none() {
                                neighborhood_class.append(&mut result.1);
                            } else if !result.1.is_empty() {
                                neighborhood_partition.push(result.1.clone());
                            }
                        }

                        (Some(ClassType::IndependentSet), neighborhood_class)
                    }
                }
            }
        }
    }
}

impl From<CoTree> for MDTree {
    fn from(co_tree: crate::CoTree) -> Self {
        fn convert(co_tree: CoTree) -> MDTree {
            match co_tree {
                CoTree::Empty => MDTree::Empty,
                CoTree::Leaf(id) => MDTree::Leaf(id),
                CoTree::Inner(vertex_count, operation, left_child, right_child) => {
                    let node_type = match operation {
                        crate::Operation::DisjointUnion => NodeType::Parallel,
                        crate::Operation::DisjointSum => NodeType::Series,
                    };
                    MDTree::Inner(
                        vertex_count,
                        node_type,
                        vec![convert(*left_child), convert(*right_child)],
                    )
                }
            }
        }

        convert(co_tree)
    }
}
