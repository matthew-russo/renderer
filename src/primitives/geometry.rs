use vulkano::pipeline::vertex::Vertex;

pub struct Geometry<V> where V: Vertex {
    pub vertices: Vec<V>,
    pub indices: Vec<u32>,
}

impl<V> Geometry<V> where V: Vertex {
    pub fn new(vertices: Vec<V>, indices: Vec<u32>) -> Geometry<V> {
        Geometry {
            vertices,
            indices,
        }
    }
}
