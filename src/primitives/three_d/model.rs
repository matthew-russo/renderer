extern crate tobj;

use std::path::Path;

use crate::primitives::vertex::Vertex;
use crate::components::transform::Transform;

// TODO -> need to reconcile Mesh && Model
#[derive(Clone)]
pub struct Model {
    pub id: String,
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

impl Model {
    pub fn load(id: String, path: &Path) -> Self {
        use tobj::{load_obj};

        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        info!("Starting to load model: {:?}", path);
        let obj = load_obj(path);
        info!("Finishing model load: {:?}", path);

        let (models, materials) = obj.unwrap();

        for (i, m) in models.iter().enumerate() {
            let mesh = &m.mesh;

            for index in &mesh.indices {
                let ind_usize = *index as usize;
                let pos = [
                    mesh.positions[ind_usize * 3 + 0],
                    mesh.positions[ind_usize * 3 + 1],
                    mesh.positions[ind_usize * 3 + 2],
                ];

                let color = [1.0, 1.0, 1.0];

                let tex_coord = [
                    mesh.texcoords[ind_usize * 2],
                    1.0 - mesh.texcoords[ind_usize * 2 + 1],
                ];

                let vertex = Vertex::new(pos, color, tex_coord);
                vertices.push(vertex);
                indices.push(indices.len() as u32);
            }
        }

        Model {
            id,
            vertices,
            indices,
        }
    }
}

