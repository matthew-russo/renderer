use std::hash::Hash;
use std::hash::Hasher;

use crate::primitives::vertex::Vertex;

const ABSOLUTE_LOWER_BOUND: f32 = -1.0;
const ABSOLUTE_UPPER_BOUND: f32 = 1.0;
const ABSOLUTE_RANGE: f32 = ABSOLUTE_UPPER_BOUND - ABSOLUTE_LOWER_BOUND;

#[derive(Clone, Debug)]
pub struct Quad where {
    pub key: String,
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
    children: Vec<Quad>,
    pub rendered: bool
}

impl Quad {
    // top and bottom are reversed because y is reversed.
    pub fn new(key: String, top: f32, left: f32, bottom: f32, right: f32, parent: Option<&Quad>) -> Quad {
        let vertices = Self::calculate_absolute_vertices(top, left, bottom, right, parent);

        println!("VERTICES: {:?}\n\n", &vertices);

        let mut indices = vec![
            0, 1, 2, 2, 3, 0
        ];

        Quad {
            key,
            vertices,
            indices,
            children: vec![],
            rendered: true,
        }
    }

    pub fn with_children(mut self, children: Vec<Quad>) -> Self {
        let max_index = self.indices.iter().max().unwrap();

        let new_children = children
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let incremented_index = i as u32 + 1;
                let new_indices = c.indices
                    .iter()
                    .map(|val| (*max_index * incremented_index) + *val + incremented_index)
                    .collect();

                let mut child = c.clone();
                child.indices = new_indices;
                child
            })
            .collect();

        self.children = new_children;

        self
    }

    fn calculate_normalized_distance(range: f32, relative_location: f32) -> f32 {
        let normalized_ratio = relative_location / ABSOLUTE_RANGE;
        (normalized_ratio * range).abs()
    }

    fn calculate_absolute_lower_bound(range: f32, half_point: f32, relative_location: f32) -> f32 {
        half_point - Self::calculate_normalized_distance(range, relative_location)
    }

    fn calculate_absolute_upper_bound(range: f32, half_point: f32, relative_location: f32) -> f32 {
        half_point + Self::calculate_normalized_distance(range, relative_location)
    }

    fn calculate_absolute_vertices(top: f32, left: f32, bottom: f32, right: f32, parent: Option<&Quad>) -> Vec<Vertex> {
        // calc absolute positions of own_vertices;
        // pass those in to children vertices

        // 1. take bounds of parent
        // 2. calculate range of (1)
        // 3. calculate normalized ratio of own coordinates (static)
        // 4. normalizedRatio * range
        // 5. from half point between parent bounds, move w/ quants from 4.

        let (parent_lower_x, parent_upper_x, parent_lower_y, parent_upper_y) = match parent {
            Some(parent_quad) => {
                let lowers = &parent_quad.vertices[0];
                let uppers = &parent_quad.vertices[2];
                (lowers.x(), uppers.x(), lowers.y(), uppers.y())
            },
            None => (-1.0, 1.0, -1.0, 1.0)
        };

        println!("PLXX: {} PUX: {} PLY: {} PUY: {}", parent_lower_x, parent_upper_x, parent_lower_y, parent_upper_y);

        let x_range = parent_upper_x - parent_lower_x;
        let x_half_point = (x_range / 2.0) + parent_lower_x;

        let y_range = parent_upper_y - parent_lower_y;
        let y_half_point = (y_range / 2.0) + parent_lower_y;

        let lower_x = Self::calculate_absolute_lower_bound(x_range, x_half_point, left);
        let upper_x = Self::calculate_absolute_upper_bound(x_range, x_half_point, right);
        let lower_y = Self::calculate_absolute_lower_bound(y_range, y_half_point, bottom);
        let upper_y = Self::calculate_absolute_upper_bound(y_range, y_half_point, top);

        println!("LX: {} UX: {} LY: {} UY: {}", lower_x, upper_x, lower_y, upper_y);

        vec![
            Vertex::new([lower_x, lower_y, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0]),
            Vertex::new([upper_x, lower_y, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0]),
            Vertex::new([upper_x, upper_y, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0]),
            Vertex::new([lower_x, upper_y, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0]),
        ]
    }

    pub fn vertices(&self) -> Vec<Vertex> {
        let mut vertices = self.vertices.clone();
        vertices.append(&mut self.children.iter().flat_map(|quad| quad.vertices()).collect());

        println!("{:?}: {:?}", self.key, &vertices);

        vertices
    }

    pub fn indices(&self) -> Vec<u32> {
        let mut indices = self.indices.clone();
        indices.append(&mut self.children.iter().flat_map(|quad| quad.indices()).collect());

        println!("{:?}: {:?}", self.key, &indices);

        indices
    }
}

impl PartialEq for Quad {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

impl Hash for Quad {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.key.hash(state)
    }
}

impl Eq for Quad {}