use vulkano::command_buffer::pool::standard::StandardCommandPoolBuilder;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use std::collections::HashMap;
use crate::components::transform::Transform;

pub trait RenderLayer where {
     fn draw_indexed(&mut self, builder: AutoCommandBufferBuilder<StandardCommandPoolBuilder>, renderables: &HashMap<String, Transform>) -> AutoCommandBufferBuilder<StandardCommandPoolBuilder>;
     fn recreate_graphics_pipeline(&mut self);
}

