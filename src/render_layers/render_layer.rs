use std::sync::Arc;

use vulkano::pipeline::{GraphicsPipelineAbstract};
use vulkano::buffer::{TypedBufferAccess};
use vulkano::command_buffer::DynamicState;
use vulkano::pipeline::vertex::Vertex;
use vulkano::descriptor::descriptor_set::DescriptorSetsCollection;
use vulkano::buffer::BufferAccess;
use vulkano::command_buffer::pool::standard::StandardCommandPoolBuilder;
use vulkano::command_buffer::AutoCommandBufferBuilder;

pub trait RenderLayer where {
     fn draw_indexed(&mut self, builder: AutoCommandBufferBuilder<StandardCommandPoolBuilder>) -> AutoCommandBufferBuilder<StandardCommandPoolBuilder>;
     fn recreate_graphics_pipeline(&mut self);
}

