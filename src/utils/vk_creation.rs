use std::sync::Arc;

use vulkano::pipeline::vertex::Vertex;
use vulkano::device::{Device, Queue};
use vulkano::buffer::{TypedBufferAccess};
use vulkano::buffer::ImmutableBuffer;
use vulkano::buffer::BufferUsage;
use vulkano::framebuffer::RenderPassAbstract;
use vulkano::pipeline::GraphicsPipelineAbstract;
use vulkano::pipeline::viewport::Viewport;
use vulkano::descriptor::descriptor::ShaderStages;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::descriptor::pipeline_layout::PipelineLayoutDescUnion;
use vulkano::pipeline::depth_stencil::DepthStencil;
use vulkano::framebuffer::Subpass;
use vulkano::sync::GpuFuture;
use vulkano::buffer::BufferAccess;

pub fn create_vertex_buffer<V>(graphics_queue: &Arc<Queue>, vertices: &Vec<V>) -> Arc<BufferAccess + Send + Sync>
        where V: Vertex + Clone {
    let (buffer, future) = ImmutableBuffer::from_iter(
        vertices.iter().cloned(),
        BufferUsage::vertex_buffer(),
        graphics_queue.clone()
    ).unwrap();

    future.flush().unwrap();

    buffer
}

pub fn create_index_buffer(graphics_queue: &Arc<Queue>, indices: &Vec<u32>) -> Arc<TypedBufferAccess<Content=[u32]> + Send + Sync> {
    let (buffer, future) = ImmutableBuffer::from_iter(
        indices.iter().cloned(),
        BufferUsage::index_buffer(),
        graphics_queue.clone()
    ).unwrap();

    future.flush().unwrap();

    buffer
}

// todo ->
// pub fn create_uniform_buffer<U>(graphics_queue: &Arc<Queue>, ubo: U'static) -> Arc<TypedBufferAccess<Content=U> + Send + Sync>
//     where U: Send + Sync{
//     let (buffer, future) = ImmutableBuffer::from_data(
//         ubo,
//         BufferUsage::uniform_buffer(),
//         graphics_queue.clone()
//     ).unwrap();
//
//     future.flush().unwrap();
//
//     buffer
// }

// todo -> issue is because macro requires str lit and i want a var.
// todo -> other options is to blow this out so it doesn't use the macro. more code but can reuse more
// pub fn create_basic_graphics_pipeline(
//     device: &Arc<Device>,
//     render_pass: &Arc<RenderPassAbstract + Send + Send>,
//     dimensions: [f32; 2],
//     shader: &'static str
// ) -> Arc<GraphicsPipelineAbstract + Send + Sync> {
//     mod vs {
//         vulkano_shaders::shader! {
//                ty: "vertex",
//                path: "src\\data\\shaders\\standard.vert"
//             }
//         }
//
//         mod fs {
//             vulkano_shaders::shader! {
//                 ty: "fragment",
//                 path: "src\\data\\shaders\\standard.frag"
//             }
//         }
//
//         let vertex = vs::Shader::load(device.clone()).expect("failed to create vertex shader modules");
//         let frag = fs::Shader::load(device.clone()).expect("failed to create frag shader modules");
//
//         let viewport = Viewport {
//             origin: [0.0, 0.0],
//             dimensions,
//             depth_range: 0.0..1.00,
//         };
//
//         let vs_layout = vs::Layout(ShaderStages { vertex: true, ..ShaderStages::none() }).build(device.clone()).unwrap();
//         let fs_layout = fs::Layout(ShaderStages { fragment: true, ..ShaderStages::none() }).build(device.clone()).unwrap();
//         let layout_union = PipelineLayoutDescUnion::new(vs_layout, fs_layout).build(device.clone()).unwrap();
//
//         Arc::new(GraphicsPipeline::start()
//             .vertex_input_single_buffer::<Vertex>()
//             .vertex_shader(vertex.main_entry_point(), ())
//             .triangle_list()
//             .primitive_restart(false)
//             .viewports(vec![viewport])
//             .fragment_shader(frag.main_entry_point(), ())
//             .depth_clamp(false)
//             .polygon_mode_fill()
//             .line_width(1.00)
//             .cull_mode_back()
//             .front_face_counter_clockwise()
//             .blend_pass_through()
//             .depth_stencil(DepthStencil::simple_depth_test())
//             .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
//             .with_pipeline_layout(device.clone(), layout_union)
//             .unwrap()
//         )
// }
