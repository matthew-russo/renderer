use vulkano::impl_vertex;

#[derive(Copy, Clone, Debug)]
pub struct Vertex {
    inPosition: [f32; 3],
    inColor: [f32; 3],
    inTexCoord: [f32; 2],
}

impl Vertex {
    pub fn new(inPosition: [f32; 3], inColor: [f32; 3], inTexCoord: [f32; 2]) -> Self {
        Self {inPosition, inColor, inTexCoord}
    }
}

impl_vertex!(Vertex, inPosition, inColor, inTexCoord);