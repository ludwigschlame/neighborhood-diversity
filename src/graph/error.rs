/// A specialized [`Result`] type for [`Graph`] operations.
///
/// This type is broadly used across [`graph`] for any operation which may
/// produce an error.
///
/// This typedef is generally used to avoid writing out [`graph::Error`] directly and
/// is otherwise a direct mapping to [`Result`].
///
/// [`Result`]: std::result::Result
/// [`graph::Error`]: Error
/// [`Graph`]: crate::graph::Graph
/// [`graph`]: crate::graph
pub type Result<T> = std::result::Result<T, Error>;

/// Error type for operations on the [`Graph`] struct.
///
/// [`Graph`]: crate::graph::Graph
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Provided vertex ID is greater or equal to order.
    OutOfBounds(/* order */ usize, /* vertex */ usize),

    /// True value on the diagonal of the adjacency matrix.
    SelfLoop(/* vertex */ usize),

    /// The length of at least one row of the adjacency matrix is not equal to
    /// the total number of rows.
    NotSquare(
        /* row_count: */ usize,
        /* row_id: */ usize,
        /* row_len: */ usize,
    ),

    /// The value of `adjacency_matrix`\[u\]\[v\] is not equal to `adjacency_matrix`\[v\]\[u\].
    NotSymmetrical(/* vertex_1: */ usize, /* vertex_2: */ usize),

    /// Input could not be parsed into a graph.
    InvalidInput(String),
}

impl std::error::Error for Error {}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::OutOfBounds(order, vertex) => {
                format!("Index out of bounds: the order is {order} but the index is {vertex}")
            }
            Self::SelfLoop(u) => {
                format!("error inserting edge {{{u}, {u}}}: self-loops are not allowed")
            }
            Self::NotSquare(row_count, row_id, row_len) => format!(
                "adjacency matrix is not square: has {row_count} rows but row #{row_id} has {row_len} elements"
            ),
            Self::NotSymmetrical(u, v) => format!(
                "adjacency matrix is not symmetrical: matrix[{u}][{v}] != matrix[{v}][{u}]"
            ),
            Self::InvalidInput(input) => format!("invalid input: {input}"),
        };

        write!(f, "{message}")
    }
}
