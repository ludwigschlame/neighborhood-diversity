use criterion::{criterion_group, criterion_main, Criterion};
use neighborhood_diversity::*;

fn nd_calc(c: &mut Criterion) {
    const VERTEX_COUNT: usize = 1e2 as usize;
    const ND_LIMIT: usize = 20;
    const DENSITY: f64 = 0.5;

    println!("|-------------------|");
    println!("|   Graph Details   |");
    println!("|-------------------|");
    println!("| Vertices | {: >6} |", format!("{:+e}", VERTEX_COUNT));
    println!("| ND Limit | {: >6} |", format!("{:+e}", ND_LIMIT));
    println!("| Density  | {: >6} |", DENSITY);
    println!("|-------------------|");

    c.bench_function("Naive ALgorithm", |b| {
        let graph_2 = Graph::random_graph_nd_limited(VERTEX_COUNT, DENSITY, ND_LIMIT);
        b.iter(|| calc_nd_classes(&graph_2, Options::naive()))
    });

    c.bench_function("Limited Comparisons", |b| {
        let graph_2 = Graph::random_graph_nd_limited(VERTEX_COUNT, DENSITY, ND_LIMIT);
        b.iter(|| calc_nd_classes(&graph_2, Options::new(false, true)))
    });

    c.bench_function("Degree Filter", |b| {
        let graph_2 = Graph::random_graph_nd_limited(VERTEX_COUNT, DENSITY, ND_LIMIT);
        b.iter(|| calc_nd_classes(&graph_2, Options::new(true, false)))
    });

    c.bench_function("Optimized", |b| {
        let graph_2 = Graph::random_graph_nd_limited(VERTEX_COUNT, DENSITY, ND_LIMIT);
        b.iter(|| calc_nd_classes(&graph_2, Options::optimized()))
    });

    c.bench_function("BTree", |b| {
        let graph_2 = Graph::random_graph_nd_limited(VERTEX_COUNT, DENSITY, ND_LIMIT);
        b.iter(|| calc_nd_btree(&graph_2))
    });
}

criterion_group!(benches, nd_calc);
criterion_main!(benches);
