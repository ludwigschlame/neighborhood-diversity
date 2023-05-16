use criterion::{criterion_group, criterion_main, Criterion};
use neighborhood_diversity::*;

fn nd_calc(c: &mut Criterion) {
    const VERTEX_COUNT: usize = 1e2 as usize;
    const ND_LIMIT: usize = 20;
    const DENSITY: f32 = 0.5;
    let representations = [AdjacencyMatrix, AdjacencyList];

    println!("|-------------------|");
    println!("|   Graph Details   |");
    println!("|-------------------|");
    println!("| Vertices | {: >6} |", format!("{:+e}", VERTEX_COUNT));
    println!("| ND Limit | {: >6} |", format!("{:+e}", ND_LIMIT));
    println!("| Density  | {: >6} |", DENSITY);
    println!("|-------------------|");
    println!();

    for (idx, &representation) in representations.iter().enumerate() {
        println!("|---------------------|");
        println!("| Representation #{:0>2}: |", idx + 1);
        println!("| {: <19} |", format!("{:?}", representation));
        println!("|---------------------|");
        println!();

        c.bench_function("Naive Algorithm", |b| {
            let mut graph =
                Graph::random_graph_nd_limited(VERTEX_COUNT, DENSITY, ND_LIMIT, representation);
            graph.shuffle();
            b.iter(|| calc_nd_classes(&graph, Options::naive()));
        });

        c.bench_function("Limited Comparisons", |b| {
            let mut graph =
                Graph::random_graph_nd_limited(VERTEX_COUNT, DENSITY, ND_LIMIT, representation);
            graph.shuffle();
            b.iter(|| calc_nd_classes(&graph, Options::new(false, true)));
        });

        c.bench_function("Degree Filter", |b| {
            let mut graph =
                Graph::random_graph_nd_limited(VERTEX_COUNT, DENSITY, ND_LIMIT, representation);
            graph.shuffle();
            b.iter(|| calc_nd_classes(&graph, Options::new(true, false)));
        });

        c.bench_function("Optimized", |b| {
            let mut graph =
                Graph::random_graph_nd_limited(VERTEX_COUNT, DENSITY, ND_LIMIT, representation);
            graph.shuffle();
            b.iter(|| calc_nd_classes(&graph, Options::optimized()));
        });

        c.bench_function("BTree", |b| {
            let mut graph =
                Graph::random_graph_nd_limited(VERTEX_COUNT, DENSITY, ND_LIMIT, representation);
            graph.shuffle();
            b.iter(|| calc_nd_btree(&graph));
        });

        c.bench_function("BTree + Degree Filter", |b| {
            let mut graph =
                Graph::random_graph_nd_limited(VERTEX_COUNT, DENSITY, ND_LIMIT, representation);
            graph.shuffle();
            b.iter(|| calc_nd_btree_degree(&graph));
        });

        c.bench_function("BTree + Concurrency", |b| {
            let mut graph =
                Graph::random_graph_nd_limited(VERTEX_COUNT, DENSITY, ND_LIMIT, representation);
            graph.shuffle();
            b.iter(|| calc_nd_btree_concurrent(&graph));
        });
    }
}

criterion_group!(benches, nd_calc);
criterion_main!(benches);
