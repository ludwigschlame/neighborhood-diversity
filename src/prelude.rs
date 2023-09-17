//! Convenience re-export of common members.
//!
//! This module simplifies importing of common items.
//!
//! The contents of this module can be imported like this:
//!
//! ```
//! use neighborhood_diversity::prelude::*;
//! # let graph = Graph::random_graph(10, 0.2);
//! ```

pub use crate::calc_neighborhood_partition;
pub use crate::graph::Graph;
