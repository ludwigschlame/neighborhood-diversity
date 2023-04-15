use criterion::{criterion_group, criterion_main, Criterion};
use neighborhood_diversity::*;

fn nd_calc(c: &mut Criterion) {
    let graph_2 = Graph::random_graph_nd_limited(1e2 as usize, 0.2, 20);

    c.bench_function("Naive ALgorithm (10^2)", |b| {
        b.iter(|| calc_nd_classes(&graph_2, Options::naive()))
    });

    c.bench_function("Degree Filter (10^2)", |b| {
        b.iter(|| calc_nd_classes(&graph_2, Options::new(true, false)))
    });
    c.bench_function("ALL (10^2)", |b| {
        b.iter(|| calc_nd_classes(&graph_2, Options::optimized()))
    });
}

criterion_group!(benches, nd_calc);
criterion_main!(benches);
