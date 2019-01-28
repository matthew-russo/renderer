use specs::Component;
use specs::VecStorage;

pub struct Camera {
   pub displaying: bool,
}

impl Component for Camera {
    type Storage = VecStorage<Camera>;
}