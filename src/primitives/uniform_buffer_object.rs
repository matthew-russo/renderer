use cgmath::{Matrix4, SquareMatrix};

use crate::renderer::presenter::DIMS;

#[derive(Clone, Copy, Debug)]
pub struct CameraUniformBufferObject {
    pub view: Matrix4<f32>,
    pub proj: Matrix4<f32>,
}

impl CameraUniformBufferObject {
    pub fn new(view: Matrix4<f32>, proj: Matrix4<f32>) -> Self {
        Self {
            view,
            proj
        }
    }
}

impl std::default::Default for CameraUniformBufferObject {
    fn default() -> Self {
        let view = Matrix4::look_at(
            cgmath::Point3::new(5.0, 5.0, 5.0),
            cgmath::Point3::new(0.0, 0.0, 0.0),
            cgmath::Vector3::new(0.0, 1.0, 0.0)
        );
        let mut proj = cgmath::perspective(
            cgmath::Deg(45.0),
            DIMS.width as f32 / DIMS.height as f32,
            0.1,
            1000.0
        );
        proj.y.y *= -1.0;

        CameraUniformBufferObject::new(
            view,
            proj
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ObjectUniformBufferObject {
    pub model: Matrix4<f32>,
}

impl ObjectUniformBufferObject {
    pub fn new(model: Matrix4<f32>) -> Self {
        Self {
            model,
        }
    }
}

impl std::default::Default for ObjectUniformBufferObject {
    fn default() -> Self {
        ObjectUniformBufferObject::new(
            Matrix4::identity(),
        )
    }
}
