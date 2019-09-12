use std::sync::{
    Arc,
    RwLock
};

use specs::{
    System,
    ReadStorage,
    WriteStorage,
    Join,
};

use rand::Rng;

use crate::components::mesh::Mesh;
use crate::components::transform::Transform;
use crate::timing::Time;

pub struct Rotation {
    pub time: Arc<RwLock<Time>>,
}

impl<'a> System<'a> for Rotation {
    type SystemData = (
        WriteStorage<'a, Transform>,
        ReadStorage<'a, Mesh>
    );

    fn run(&mut self, (mut transform_storage, mesh_storage): Self::SystemData) {
        let mut rng = rand::thread_rng();
        
        let delta_time = self.time.read().unwrap().delta_time as f32 * 0.01;

        for (transform, _mesh) in (&mut transform_storage, &mesh_storage).join() {
            let r = rng.gen::<f32>();
            let unique_delta = r * delta_time * 2.0;
            transform.rotate(2.0 * unique_delta, 10.0 * unique_delta, 20.0 * unique_delta);
        }
    }
}
