use crate::primitives::vertex::Vertex;

#[derive(Clone)]
pub struct Model {
    pub key: String,
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,

    pub transform: Transform
}

#[derive(Clone)]
pub struct Transform {
    pub position: glm::Vec3,
    pub scale: glm::Vec3,
    pub rotation: glm::Vec3,
}

impl Transform {
    pub fn new() -> Transform {
        Transform {
            position: glm::Vec3 {x: 0.0, y: 0.0, z: 0.0},
            scale: glm::Vec3 {x: 1.0, y: 1.0, z: 1.0},
            rotation: glm::Vec3 {x: 0.0, y: 0.0, z: 0.0},
        }
    }

    pub fn translate(&mut self, to_move: glm::Vec3) {
        self.position = self.position +  to_move;
    }
}
