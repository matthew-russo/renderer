#[derive(Clone)]
pub enum ApplicationEvent {
    KeyPress(KeyPress)
}

#[derive(Clone)]
pub enum KeyPress {
    EscKey,
    W,
    A,
    S,
    D,
    Space,
    LShift,
}
