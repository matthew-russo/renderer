use specs::{
    VecStorage,
    Component,
};

use cgmath::{
    Vector3,
    Quaternion,
    Euler,
    Deg,
    Angle
};

#[derive(Clone, Debug)]
pub struct Transform {
    pub position: Vector3<f32>,
    pub scale: Vector3<f32>,
    pub rotation: Quaternion<f32>,
}

impl Component for Transform {
    type Storage = VecStorage<Self>;
}

impl Transform {
    pub fn new() -> Transform {
        Transform {
            position: Vector3::new(0.0, 0.0, 0.0),
            scale: Vector3::new(1.0, 1.0, 1.0),
            rotation: Quaternion::from(Euler { x: Deg(0.0), y: Deg(0.0), z: Deg(0.0) }),
        }
    }

    // todo -> this should probably just take 3 params: x, y, and z since thats more intuitive
    pub fn translate(&mut self, to_move: Vector3<f32>) {
        self.position = self.position +  to_move;
    }

    pub fn rotate(&mut self, x: f32, y: f32, z: f32) {
        let to_rotate_by = Quaternion::from(Euler {
            x: Deg(x),
            y: Deg(y),
            z: Deg(z)
        });

        self.rotation = self.rotation * to_rotate_by;
        // println!("rotation: {:?}", Euler::from(self.rotation));
    }

    pub fn forward(&self) -> Vector3<f32> {
        let rotation = Euler::from(self.rotation);

        Vector3::new(
            rotation.x.sin(),
            rotation.y.tan() * -1.0,
            rotation.x.cos(),
        )
    }
}
