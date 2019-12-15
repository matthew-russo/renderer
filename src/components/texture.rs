use std::cmp::Ordering;

#[derive(Eq, PartialEq, Hash, Clone, Debug)]
pub struct Texture {
    pub(crate) path: String,
}

impl Ord for Texture {
    fn cmp(&self, other: &Self) -> Ordering {
        self.path.cmp(&other.path)
    }
}

impl PartialOrd for Texture {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
