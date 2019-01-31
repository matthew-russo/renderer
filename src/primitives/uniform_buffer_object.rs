use cgmath::Matrix4;

#[derive(Clone, Copy)]
pub struct UniformBufferObject {
    pub model: Matrix4<f32>,
    pub view: Matrix4<f32>,
    pub proj: Matrix4<f32>,
}

impl UniformBufferObject {
    pub fn new(model: Matrix4<f32>, view: Matrix4<f32>, proj: Matrix4<f32>) -> Self {
        Self {
            model,
            view,
            proj
        }
    }

    pub fn as_arrays(&self) -> ([[f32; 4]; 4], [[f32; 4]; 4], [[f32; 4]; 4]) {
        let model = [
            [self.model.x.x, self.model.x.y, self.model.x.z, self.model.x.w],
            [self.model.y.x, self.model.y.y, self.model.y.z, self.model.y.w],
            [self.model.z.x, self.model.z.y, self.model.z.z, self.model.z.w],
            [self.model.w.x, self.model.w.y, self.model.w.z, self.model.w.w],
        ];

        let view = [
            [self.view.x.x, self.view.x.y, self.view.x.z, self.view.x.w],
            [self.view.y.x, self.view.y.y, self.view.y.z, self.view.y.w],
            [self.view.z.x, self.view.z.y, self.view.z.z, self.view.z.w],
            [self.view.w.x, self.view.w.y, self.view.w.z, self.view.w.w],
        ];

         let proj = [
            [self.proj.x.x, self.proj.x.y, self.proj.x.z, self.proj.x.w],
            [self.proj.y.x, self.proj.y.y, self.proj.y.z, self.proj.y.w],
            [self.proj.z.x, self.proj.z.y, self.proj.z.z, self.proj.z.w],
            [self.proj.w.x, self.proj.w.y, self.proj.w.z, self.proj.w.w],
        ];

        (model, view, proj)
    }
}