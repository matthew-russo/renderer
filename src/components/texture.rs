use specs::Component;
use specs::VecStorage;

#[derive(Eq, PartialEq, Hash, Clone, Debug)]
pub struct Texture {
    pub(crate) path: String,
}

impl Component for Texture {
    type Storage = VecStorage<Self>;
}
