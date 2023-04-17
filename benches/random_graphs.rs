use criterion::{criterion_group, criterion_main, Criterion};
use neighborhood_diversity::*;

fn nd_calc(c: &mut Criterion) {
    c.bench_function("Naive ALgorithm (10^2)", |b| {
        let graph_2 = Graph::random_graph_nd_limited(1e2 as usize, 0.5, 20);
        b.iter(|| calc_nd_classes(&graph_2, Options::naive()))
    });

    c.bench_function("Limited Comparisons (10^2)", |b| {
        let graph_2 = Graph::random_graph_nd_limited(1e2 as usize, 0.5, 20);
        b.iter(|| calc_nd_classes(&graph_2, Options::new(false, true)))
    });

    c.bench_function("Degree Filter (10^2)", |b| {
        let graph_2 = Graph::random_graph_nd_limited(1e2 as usize, 0.5, 20);
        b.iter(|| calc_nd_classes(&graph_2, Options::new(true, false)))
    });

    c.bench_function("All (10^2)", |b| {
        let graph_2 = Graph::random_graph_nd_limited(1e2 as usize, 0.5, 20);
        b.iter(|| calc_nd_classes(&graph_2, Options::optimized()))
    });

    c.bench_function("BTree (10^2)", |b| {
        let graph_2 = Graph::random_graph_nd_limited(1e2 as usize, 0.5, 20);
        b.iter(|| calc_nd_btree(&graph_2))
    });
}

criterion_group!(benches, nd_calc);
criterion_main!(benches);
