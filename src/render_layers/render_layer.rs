use vulkano::command_buffer::pool::standard::StandardCommandPoolBuilder;
use vulkano::command_buffer::AutoCommandBufferBuilder;

pub trait RenderLayer where {
     fn draw_indexed(&mut self, builder: AutoCommandBufferBuilder<StandardCommandPoolBuilder>) -> AutoCommandBufferBuilder<StandardCommandPoolBuilder>;
     fn recreate_graphics_pipeline(&mut self);
}

