use std::sync::{
    Arc,
    RwLock
};

use rand::Rng;

use crate::components::mesh::Mesh;
use crate::components::transform::Transform;
use crate::timing::Time;

use legion::World;
use legion::query::{Read, Write, IntoQuery, Query};

pub struct Rotation {
    pub time: Arc<RwLock<Time>>,
}

impl Rotation {
    pub fn new(time: &Arc<RwLock<Time>>) -> Self {
        Self {
            time: Arc::clone(time),
        }
    }

    pub fn run(&self, world: &World) {
       let mut rng = rand::thread_rng();
       let delta_time = self.time.read().unwrap().delta_time as f32 * 0.01;

       <(Write<Transform>, Read<Mesh>)>::query()
           .iter(&world)
           .for_each(|(transform, _mesh)| {
               let r = rng.gen::<f32>();
               let unique_delta = r * delta_time * 2.0;
               transform.rotate(2.0 * unique_delta, 10.0 * unique_delta, 20.0 * unique_delta);
           });
   }
}
