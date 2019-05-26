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
        //let total_time = self.time.read().unwrap().total_time() as f32 * 0.0001;
        let delta_time = self.time.read().unwrap().delta_time as f32 * 0.01;

        for (transform, mesh) in (&mut transform_storage, &mesh_storage).join() {
            transform.rotate(2.0 * delta_time, 10.0 * delta_time, 20.0 * delta_time);
        }
    }
}
