use glm::Mat4;

pub struct UniformBufferObject {
    model: Mat4,
    view: Mat4,
    proj: Mat4,
}

impl UniformBufferObject {
    pub fn new(model: Mat4, view: Mat4, proj: Mat4) -> Self {
        Self {
            model,
            view,
            proj
        }
    }
}