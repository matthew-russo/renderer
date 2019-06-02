use uuid::Uuid;

use crate::primitives::vertex::Vertex;
use crate::components::mesh::Mesh;
use crate::components::transform::Transform;

pub struct Cube {
    key: String,
}

impl Cube {
    pub fn new() -> (Transform, Mesh) {
        let red = [1.0, 0.0, 0.0];
        let green = [0.0, 1.0, 0.0];
        let blue = [0.0, 0.0, 1.0];

        let yellow = [1.0, 1.0, 0.0];
        let purple = [1.0, 0.0, 1.0];
        let blue_green = [0.0, 1.0, 1.0];

        let lower_x = -0.5;
        let lower_y = -0.5;
        let lower_z = -0.5;
        let upper_x = 0.5;
        let upper_y = 0.5;
        let upper_z = 0.5;

        let vertices = vec![
            // back face
            Vertex::new([lower_x, lower_y, lower_z], red, [0.0, 0.0]),
            Vertex::new([lower_x, upper_y, lower_z], red, [1.0, 0.0]),
            Vertex::new([upper_x, upper_y, lower_z], red, [1.0, 1.0]),
            Vertex::new([upper_x, lower_y, lower_z], red, [0.0, 1.0]),

            // front face
            Vertex::new([lower_x, lower_y, upper_z], green, [0.0, 0.0]),
            Vertex::new([upper_x, lower_y, upper_z], green, [1.0, 0.0]),
            Vertex::new([upper_x, upper_y, upper_z], green, [1.0, 1.0]),
            Vertex::new([lower_x, upper_y, upper_z], green, [0.0, 1.0]),

            // left face
            Vertex::new([lower_x, lower_y, upper_z], blue, [0.0, 0.0]),
            Vertex::new([lower_x, upper_y, upper_z], blue, [1.0, 0.0]),
            Vertex::new([lower_x, upper_y, lower_z], blue, [1.0, 1.0]),
            Vertex::new([lower_x, lower_y, lower_z], blue, [0.0, 1.0]),

            // right face
            Vertex::new([upper_x, lower_y, upper_z], yellow, [0.0, 0.0]),
            Vertex::new([upper_x, lower_y, lower_z], yellow, [1.0, 0.0]),
            Vertex::new([upper_x, upper_y, lower_z], yellow, [1.0, 1.0]),
            Vertex::new([upper_x, upper_y, upper_z], yellow, [0.0, 1.0]),

            // top face
            Vertex::new([lower_x, upper_y, upper_z], purple, [0.0, 0.0]),
            Vertex::new([upper_x, upper_y, upper_z], purple, [1.0, 0.0]),
            Vertex::new([upper_x, upper_y, lower_z], purple, [1.0, 1.0]),
            Vertex::new([lower_x, upper_y, lower_z], purple, [0.0, 1.0]),

            // bottom face
            Vertex::new([lower_x, lower_y, upper_z], blue_green, [0.0, 0.0]),
            Vertex::new([lower_x, lower_y, lower_z], blue_green, [1.0, 0.0]),
            Vertex::new([upper_x, lower_y, lower_z], blue_green, [1.0, 1.0]),
            Vertex::new([upper_x, lower_y, upper_z], blue_green, [0.0, 1.0]),
        ];

        let indices = vec![
            0, 1, 2, 2, 3, 0,
            4, 5, 6, 6, 7, 4,
            8, 9, 10, 10, 11, 8,
            12, 13, 14, 14, 15, 12,
            16, 17, 18, 18, 19, 16,
            20, 21, 22, 22, 23, 20,
        ];

        let mesh = Mesh {
            key: Uuid::new_v4().to_string(),
            vertices,
            indices,
            rendered: true
        };

        (Transform::new(), mesh)
    }
}

//impl Model for Cube {
//
//}
