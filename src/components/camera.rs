
use legion::EntitySource;
use legion::EntityAllocator;
use legion::storage::Chunk;
use legion::storage::ChunkBuilder;
use legion::storage::Archetype;

use std::collections::HashSet;
use std::hash::BuildHasherDefault;
use std::any::TypeId;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
   pub displaying: bool,
}
