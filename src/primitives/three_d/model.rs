use crate::primitives::vertex::Vertex;
use crate::components::transform::Transform;

#[derive(Clone)]
pub struct Model {
    pub key: String,
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,

    pub transform: Transform
}
