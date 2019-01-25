use vulkano::impl_vertex;

#[derive(Copy, Clone, Debug)]
pub struct Vertex {
    in_position: [f32; 3],
    in_color: [f32; 3],
    in_tex_coord: [f32; 2],
}

impl Vertex {
    pub fn new(in_position: [f32; 3], in_color: [f32; 3], in_tex_coord: [f32; 2]) -> Self {
        Self {in_position, in_color, in_tex_coord}
    }

    pub fn x(&self) -> f32 {
        self.in_position[0]
    }

    pub fn y(&self) -> f32 {
        self.in_position[1]
    }
}

impl_vertex!(Vertex, in_position, in_color, in_tex_coord);