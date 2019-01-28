use crate::primitives::vertex::Vertex;
use specs::Component;
use specs::VecStorage;

pub struct Mesh {
    pub key: String,
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub rendered: bool,
}

impl Component for Mesh {
    type Storage = VecStorage<Self>;
}