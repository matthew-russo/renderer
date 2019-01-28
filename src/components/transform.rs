use specs::{
    VecStorage,
    Component,
};

#[derive(Clone)]
pub struct Transform {
    pub position: glm::Vec3,
    pub scale: glm::Vec3,
    pub rotation: glm::Vec3,
}

impl Component for Transform {
    type Storage = VecStorage<Self>;
}

impl Transform {
    pub fn new() -> Transform {
        Transform {
            position: glm::Vec3 {x: 0.0, y: 0.0, z: 0.0},
            scale: glm::Vec3 {x: 1.0, y: 1.0, z: 1.0},
            rotation: glm::Vec3 {x: 0.0, y: 0.0, z: 0.0},
        }
    }

    // todo -> this should probably just take 3 params: x, y, and z since thats more intuitive
    pub fn translate(&mut self, to_move: glm::Vec3) {
        self.position = self.position +  to_move;
    }
}