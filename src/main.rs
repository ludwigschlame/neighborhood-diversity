use neighborhood_diversity::*;

fn main() {
    // example
    let path = "examples/nd_01.txt";
    let input = std::fs::read_to_string(path)
        .unwrap_or_else(|err| panic!("error reading '{}': {}", path, err));

    let example_graph = input
        .parse::<Graph>()
        .unwrap_or_else(|err| panic!("error parsing input: {}", err));

    let neighborhood_diversity = calc_nd(&example_graph, Algorithm::Naive);

    println!("The neighborhood diversity is {}.", neighborhood_diversity);

    // random graph
    for i in 0..11 {
        let edge_probability = 0.1 * i as f64;
        println!(
            "edge probability probability: {:.2}; neighborhood diversity {}",
            edge_probability,
            calc_nd(
                &Graph::random_graph(1e2 as usize, edge_probability),
                Algorithm::Naive
            )
        );
    }
}
