use cgmath::Matrix4;

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
