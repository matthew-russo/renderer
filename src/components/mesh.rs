use crate::primitives::vertex::Vertex;

#[derive(Clone, Debug)]
pub struct Mesh {
    pub key: String,
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub rendered: bool,
}
