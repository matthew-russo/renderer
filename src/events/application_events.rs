#[derive(Clone)]
pub enum ApplicationEvent {
    KeyPress(KeyPress)
}

#[derive(Clone)]
pub enum KeyPress {
    EscKey,
}
