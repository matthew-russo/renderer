#[derive(Clone, Debug)]
pub struct Config where {
    pub should_record_commands: bool,
}

impl Config {
    pub fn new() -> Self {
        Self {
            should_record_commands: true,
        }
    }
}