enum Color {
    Rgb(u8, u8, u8),
    Rgba(u8, u8, u8, f32),
    Hex(String),
}

struct Rect {
    geometry: Geometry,
    color: Color,
}

impl Rect {
    fn new(top_left_corner: f32, width: f32, height: f32, color: Color) -> Rect {
        let geometry = Geometry {

        };
    }
}
