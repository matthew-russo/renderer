#[derive(Clone)]
pub enum ApplicationEvent {
    KeyPress(KeyPress),
    MouseMotion { x: f64, y: f64 },
    MouseScroll { delta: f64 },
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
