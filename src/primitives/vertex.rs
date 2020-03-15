#[derive(Copy, Clone, Debug)]
pub struct Vertex {
    pub in_position: [f32; 3],
    pub in_color: [f32; 3],
    pub in_tex_coord: [f32; 2],
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

    pub fn normalize(&mut self, width: u32, height: u32) {
        let x = self.in_position[0];
        let y = self.in_position[1];

        let new_x = (x / width as f32) * 2.0 - 1.0;
        let new_y = (y / height as f32) * 2.0 - 1.0;

        self.in_position[0] = new_x;
        self.in_position[1] = new_y;
    }
}

impl From<&imgui::DrawVert> for Vertex {
    fn from(draw_vert: &imgui::DrawVert) -> Self {
        let in_position = [draw_vert.pos[0], draw_vert.pos[1], 0.0];
        let in_color = [
            draw_vert.col[0] as f32,
            draw_vert.col[1] as f32,
            draw_vert.col[2] as f32
        ];
        let in_tex_coord = [draw_vert.uv[0], draw_vert.uv[1]];

        Self { in_position, in_color, in_tex_coord, }
    }
}
