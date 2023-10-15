# Neighborhood Diversity

[<img alt="crates.io" src="https://img.shields.io/crates/v/neighborhood-diversity.svg?style=for-the-badge&color=ffc933&logo=rust" height="20">](https://crates.io/crates/neighborhood-diversity)
[<img alt="build status" src="https://img.shields.io/github/actions/workflow/status/ludwigschlame/neighborhood-diversity/rust.yml?style=for-the-badge" height="20">](https://github.com/ludwigschlame/neighborhood-diversity/actions)
[<img alt="docs.rs" src="https://img.shields.io/docsrs/neighborhood-diversity?style=for-the-badge&color=66c2a5&logo=docs.rs" height="20">](https://docs.rs/neighborhood-diversity)
[<img alt="license" src="https://img.shields.io/crates/l/neighborhood-diversity?style=for-the-badge" height="20">](#license)

A [`Rust`](https://www.rust-lang.org) Library for computing the neighborhood diversity of simple, undirected graphs.

## Usage

```rust
use neighborhood_diversity::prelude::*;
let graph = Graph::random_graph(10, 0.1);
let neighborhood_partition = calc_neighborhood_partition(&graph);
let neighborhood_diversity = neighborhood_partition.len();
```


## Definitions

A graph's neighborhood diversity quantifies the variety of neighborhoods of its vertices.
In loose terms, it says that two vertices have the same type if they have the same neighbors, irrespective of whether they are adjacent or not.
Two vertices having the same type is an equivalence relation which means that reflexivity, symmetry and transitivity apply.
For the order-zero graph $K_0$, the neighborhood diversity is zero.
Graphs $G = (V, E)$ of higher order produce values between one and $|V|$.
One, if the graph's vertices form a singular clique or independent set and $|V|$ if no two vertices have the same type.
The definitions this work is based on closely adhere to the ones presented by [Lampis (2012)](https://doi.org/10.1007/s00453-011-9554-x "Algorithmic Meta-theorems for Restrictions of Treewidth"):

>**Definition 1.1**
Given a graph $G = (V, E)$, two vertices $v, v' \in V$ have the same *type* if $N(v) \setminus \{v'\} = N(v') \setminus \{v\}$.

>**Definition 1.2**
Given a graph $G = (V, E)$, a subset $M \subseteq V$ is called a *neighborhood class* of $G$ if $\forall v, v' \in M: N(v) \setminus \{v'\} = N(v') \setminus \{v\}$.

>**Definition 1.3**
A *neighborhood partition* divides the vertices of a graph into subsets in such a way that each subset forms a neighborhood class.
    Such a partition is *optimal* if it is made up exclusively of maximal neighborhood classes.

>**Definition 1.4**
The *neighborhood diversity* of a graph is defined by the number of parts in the optimal neighborhood partition of its vertices.

## License

Licensed under the Apache License, Version 2.0 <[LICENSE-APACHE.txt](LICENSE-APACHE.txt) or [https://www.apache.org/licenses/LICENSE-2.0](https://www.apache.org/licenses/LICENSE-2.0)> or the MIT license <[LICENSE-MIT.txt](LICENSE-MIT.txt) or [https://opensource.org/licenses/MIT](https://opensource.org/licenses/MIT)>, at your option. 
Files in the project may not be copied, modified, or distributed except according to those terms.
