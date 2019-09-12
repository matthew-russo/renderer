use specs::Component;
use specs::VecStorage;

#[derive(Eq, PartialEq, Hash, Clone, Debug)]
pub struct Color {
    r: u8,
    g: u8,
    b: u8
}

impl Component for Color {
    type Storage = VecStorage<Self>;
}
