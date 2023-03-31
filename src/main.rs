use neighborhood_diversity::*;

fn main() {
    // example
    let path = "examples/nd_01.txt";
    let input = std::fs::read_to_string(path)
        .unwrap_or_else(|err| panic!("error reading '{}': {}", path, err));

    let example_graph = input
        .parse::<Graph>()
        .unwrap_or_else(|err| panic!("error parsing input: {}", err));

    let neighborhood_diversity = calc_nd(example_graph, Algorithm::Naive);

    println!("The neighborhood diversity is {}.", neighborhood_diversity);

    // random graph
    println!(
        "The neighborhood diversity is {}.",
        calc_nd(
            Graph::random_graph(1e2 as usize, 1e-2).unwrap(),
            Algorithm::Naive
        )
    );
}
