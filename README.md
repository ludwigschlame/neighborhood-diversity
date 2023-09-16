# Neighborhood Diversity

A `Rust` Library for computing the neighborhood diversity of simple, undirected graphs $G = (V, E)$.
A graph's neighborhood diversity quantifies the variety of neighborhoods of its vertices.
In loose terms, it says that two vertices have the same type if they have the same neighbors, irrespective of whether they are adjacent or not.
Two vertices having the same type is an equivalence relation which means that reflexivity, symmetry and transitivity apply.
For the order-zero graph $K_0$, the neighborhood diversity is zero.
Graphs of higher order produce values between one and $|V|$.
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
