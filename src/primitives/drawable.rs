use crate::components::mesh::Mesh;
use crate::components::transform::Transform;

pub struct Drawable {
    pub mesh: Option<Mesh>,
    pub transform: Option<Transform>
}

impl Drawable {
    pub fn new(m: Mesh, t: Transform) -> Self {
        Self {
            mesh: Some(m),
            transform: Some(t)
        }
    }
}
