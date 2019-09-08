use crate::components::mesh::Mesh;
use crate::components::transform::Transform;
use crate::components::texture::Texture;
use crate::components::color::Color;

pub struct Drawable {
    pub mesh: Mesh,
    pub transform: Transform,
    pub color: Option<Color>,
    pub texture: Option<Texture>
}

impl Drawable {
    pub fn new(m: Mesh, t: Transform) -> Self {
        Self {
            mesh: m,
            transform: t,
            color: None,
            texture: None,
        }
    }

    pub fn with_color(&mut self, c: Color) -> &Self {
        self.color = Some(c);
        self
    }

    pub fn with_texture(&mut self, t: Texture) -> &Self {
        self.texture = Some(t);
        self
    }
}
