use criterion::{criterion_group, criterion_main, Criterion};
use neighborhood_diversity::*;

fn nd_calc(c: &mut Criterion) {
    let graph_2 = Graph::random_graph(1e2 as usize, 1e-2).unwrap();
    let graph_3 = Graph::random_graph(1e3 as usize, 1e-3).unwrap();

    c.bench_function("Naive Algorithm (10^2)", |b| {
        b.iter(|| {
            calc_nd(&graph_2, Algorithm::Naive);
        })
    });

    c.bench_function("Naive Algorithm + Degree Filter (10^2)", |b| {
        b.iter(|| {
            calc_nd(&graph_2, Algorithm::DegreeFilter);
        })
    });

    c.bench_function("Naive Algorithm (10^3)", |b| {
        b.iter(|| {
            calc_nd(&graph_3, Algorithm::Naive);
        })
    });

    c.bench_function("Naive Algorithm + Degree Filter (10^3)", |b| {
        b.iter(|| {
            calc_nd(&graph_3, Algorithm::DegreeFilter);
        })
    });
}

criterion_group!(benches, nd_calc);
criterion_main!(benches);
