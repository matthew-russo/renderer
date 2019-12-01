use crate::primitives::vertex::Vertex;

use legion::EntitySource;
use legion::EntityAllocator;
use legion::storage::Chunk;
use legion::storage::ChunkBuilder;
use legion::storage::Archetype;

use std::collections::HashSet;
use std::hash::BuildHasherDefault;
use std::any::TypeId;

#[derive(Clone, Debug)]
pub struct Mesh {
    pub key: String,
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub rendered: bool,
}
