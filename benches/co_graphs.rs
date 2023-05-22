use criterion::{criterion_group, criterion_main, Criterion};
use neighborhood_diversity::*;

fn nd_calc(c: &mut Criterion) {
    const VERTEX_COUNT: usize = 1e5 as usize;
    const DENSITY: f32 = 0.5;

    c.bench_function("Co-Tree", |b| {
        let co_tree = CoTree::random_tree(VERTEX_COUNT, DENSITY);
        b.iter(|| co_tree.neighborhood_partition());
    });

    c.bench_function("Co-Graph BTree", |b| {
        let co_tree = CoTree::random_tree(VERTEX_COUNT, DENSITY);
        let co_graph = Graph::from(co_tree);
        b.iter(|| calc_nd_btree(&co_graph));
    });
}

criterion_group!(benches, nd_calc);
criterion_main!(benches);
