#[derive(Debug, Clone)]
pub enum Error {
    OutOfBounds(/* order: */ usize, /* vertex: */ usize),
    SelfLoop(/* vertex: */ usize),
    NotSquare(
        /* row_count: */ usize,
        /* row_id: */ usize,
        /* row_len: */ usize,
    ),
    NotSymmetrical(/* vertex_1: */ usize, /* vertex_2: */ usize),
    InvalidInput(String),
}

impl std::error::Error for Error {}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::OutOfBounds(order, vertex) => format!(
                "index out of bounds: the order is {} but the index is {}",
                order, vertex
            ),
            Self::SelfLoop(u) => format!(
                "error inserting edge {{{}, {}}}: self-loops are not allowed",
                u, u
            ),
            Self::NotSquare(row_count, row_id, row_len) => format!(
                "adjacency matrix is not square: has {} rows but row #{} has {} elements",
                row_count, row_id, row_len
            ),
            Self::NotSymmetrical(u, v) => format!(
                "adjacency matrix is not symmetrical: matrix[{}][{}] != matrix[{}][{}]",
                u, v, v, u
            ),
            Self::InvalidInput(input) => format!("invalid input: {}", input),
        };

        write!(f, "{}", message)
    }
}
