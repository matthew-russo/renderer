use std::sync::{Arc, Mutex};

use vulkano::device::{Device, Queue};
use vulkano::buffer::{TypedBufferAccess};
use vulkano::pipeline::{GraphicsPipelineAbstract};
use vulkano::framebuffer::RenderPassAbstract;

use super::render_layer::RenderLayer;
use vulkano::descriptor::descriptor_set::FixedSizeDescriptorSetBuilder;
use vulkano::descriptor::descriptor_set::FixedSizeDescriptorSetsPool;
use crate::primitives::vertex::Vertex;
use vulkano::command_buffer::DynamicState;
use vulkano::pipeline::depth_stencil::DepthStencil;
use vulkano::framebuffer::Subpass;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::descriptor::pipeline_layout::PipelineLayoutDescUnion;
use vulkano::descriptor::descriptor::ShaderStages;
use vulkano::pipeline::viewport::Viewport;
use vulkano::descriptor::pipeline_layout::PipelineLayoutDesc;
use vulkano::buffer::BufferAccess;
use crate::utils::vk_creation;
use vulkano::descriptor::descriptor_set::DescriptorSetDesc;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::command_buffer::pool::standard::StandardCommandPoolBuilder;
use vulkano::descriptor::descriptor_set::FixedSizeDescriptorSet;
use crate::primitives;
use vulkano::pipeline::vertex::SingleBufferDefinition;
use vulkano::descriptor::pipeline_layout::PipelineLayout;

mod ui_vertex_shader {
    vulkano_shaders::shader! {
       ty: "vertex",
       path: "src\\data\\shaders\\ui.vert"
    }
}

mod ui_fragment_shader {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src\\data\\shaders\\ui.frag"
    }
}

type UiGraphicsPipeline = Arc<GraphicsPipeline<SingleBufferDefinition<primitives::vertex::Vertex>, PipelineLayout<PipelineLayoutDescUnion<PipelineLayout<ui_vertex_shader::Layout>, PipelineLayout<ui_fragment_shader::Layout>>>, Arc<dyn RenderPassAbstract + Send + Sync>>>;
// todo -> this and scene layer share a ton of logic. probably best to move some of the stuff into a default trait impl
pub struct UiLayer {
    device: Arc<Device>,
    render_pass: Arc<RenderPassAbstract + Send + Sync>,
    dimensions: [f32; 2],

    graphics_queue: Arc<Queue>,
    graphics_pipeline: UiGraphicsPipeline,

    // descriptor_sets_pool: Arc<Mutex<FixedSizeDescriptorSetsPool<UiGraphicsPipeline>>>,
    // descriptor_set: Arc<FixedSizeDescriptorSet<UiGraphicsPipeline, ()>>,

    vertices: Vec<Vertex>,
    vertex_buffer: Arc<BufferAccess + Send + Sync>,

    indices: Vec<u32>,
    index_buffer: Arc<TypedBufferAccess<Content=[u32]> + Send + Sync>,

    need_to_rebuild_buffers: bool,
}

impl UiLayer {
    pub fn new(
        graphics_queue: &Arc<Queue>,
        device: &Arc<Device>,
        render_pass: &Arc<RenderPassAbstract + Send + Sync>,
        dimensions: [f32; 2]
    ) -> UiLayer {
        let local_queue = graphics_queue.clone();

        let graphics_pipeline = Self::create_graphics_pipeline(device, render_pass, dimensions);

        // todo -> build descriptors dynamically?
       // let descriptor_sets_pool = Arc::new(
       //     Mutex::new(
       //         FixedSizeDescriptorSetsPool::new(graphics_pipeline.clone(), 0)
       //     )
       // );

       // let descriptor_set = Self::build_descriptor_set(descriptor_sets_pool.clone());

        UiLayer {
            render_pass: render_pass.clone(),
            device: device.clone(),
            dimensions: dimensions.clone(),

            graphics_queue: graphics_queue.clone(),
            graphics_pipeline,

            // descriptor_sets_pool,
            // descriptor_set,

            vertices: vec![],
            vertex_buffer: vk_creation::create_vertex_buffer::<Vertex>(&graphics_queue, &vec![]),

            indices: vec![],
            index_buffer: vk_creation::create_index_buffer(&graphics_queue, &vec![]),

            need_to_rebuild_buffers: false,
        }
    }

    fn create_graphics_pipeline(
        device: &Arc<Device>,
        render_pass: &Arc<RenderPassAbstract + Send + Sync>,
        dimensions: [f32; 2]
    ) -> UiGraphicsPipeline {
        Self::create_basic_graphics_pipeline(device, render_pass, dimensions, "ui")
    }

    fn recreate_graphics_pipeline(&mut self) {
        self.graphics_pipeline = Self::create_basic_graphics_pipeline(&self.device, &self.render_pass, self.dimensions.clone(), "ui");
    }

    fn build_descriptor_set(
        pool: Arc<Mutex<FixedSizeDescriptorSetsPool<UiGraphicsPipeline>>>
    ) -> Arc<FixedSizeDescriptorSet<UiGraphicsPipeline, ()>> {
        Arc::new(pool.lock().unwrap().next()
            // todo -> add stuff?
            .build()
            .unwrap())
    }

    fn rebuild_buffers_if_necessary(&mut self) {
        if self.need_to_rebuild_buffers {
            println!("rebuilding buffers");
            self.vertex_buffer = vk_creation::create_vertex_buffer(&self.graphics_queue, &self.vertices);
            self.index_buffer = vk_creation::create_index_buffer(&self.graphics_queue, &self.indices);
            self.need_to_rebuild_buffers = false;
        }
    }

    fn get_vertex_buffer(&mut self) -> Arc<BufferAccess + Send + Sync> {
        self.rebuild_buffers_if_necessary();
        self.vertex_buffer.clone()
    }

    fn get_index_buffer(&mut self) -> Arc<TypedBufferAccess<Content=[u32]> + Send + Sync> {
        self.rebuild_buffers_if_necessary();
        self.index_buffer.clone()
    }

    pub fn add_geometry(&mut self, mut vertices: Vec<Vertex>, indices: Vec<u32>) {
        let mut new_indices = indices
            .iter()
            .map(|i| *i + self.indices.len() as u32)
            .collect::<Vec<u32>>();

        self.vertices.append(&mut vertices);
        self.indices.append(&mut new_indices);

        println!("added geometry: {:?}, {:?}", self.vertices, self.indices);

        self.need_to_rebuild_buffers = true;
    }

    fn create_basic_graphics_pipeline(
        device: &Arc<Device>,
        render_pass: &Arc<RenderPassAbstract + Send + Sync>,
        dimensions: [f32; 2],
        shader: &'static str
    ) -> UiGraphicsPipeline {
        let vertex = ui_vertex_shader::Shader::load(device.clone()).expect("failed to create vertex shader modules");
        let frag = ui_fragment_shader::Shader::load(device.clone()).expect("failed to create frag shader modules");

        let viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions,
            depth_range: 0.0..1.00,
        };

        let ui_vertex_shader_layout = ui_vertex_shader::Layout(ShaderStages { vertex: true, ..ShaderStages::none() }).build(device.clone()).unwrap();
        let ui_fragment_shader_layout = ui_fragment_shader::Layout(ShaderStages { fragment: true, ..ShaderStages::none() }).build(device.clone()).unwrap();

        let layout_union = PipelineLayoutDescUnion::new(ui_vertex_shader_layout, ui_fragment_shader_layout).build(device.clone()).unwrap();

        Arc::new(GraphicsPipeline::start()
            .vertex_input_single_buffer::<Vertex>()
            .vertex_shader(vertex.main_entry_point(), ())
            .triangle_list()
            .primitive_restart(false)
            .viewports(vec![viewport])
            .fragment_shader(frag.main_entry_point(), ())
            .depth_clamp(false)
            .polygon_mode_fill()
            .line_width(1.00)
            .cull_mode_disabled()
            .front_face_counter_clockwise()
            .blend_pass_through()
            //.depth_stencil(DepthStencil::simple_depth_test())
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .with_pipeline_layout(device.clone(), layout_union)
            .unwrap()
        )
    }
}

impl RenderLayer for UiLayer {
    fn draw_indexed(&mut self, builder: AutoCommandBufferBuilder<StandardCommandPoolBuilder>) -> AutoCommandBufferBuilder<StandardCommandPoolBuilder> {
        self.rebuild_buffers_if_necessary();

        builder.draw_indexed(
           self.graphics_pipeline.clone(),
            &DynamicState::none(),
            vec![self.vertex_buffer.clone()],
            self.index_buffer.clone(),
           (), //self.descriptor_set.clone(),
           ()
        ).unwrap()
    }

    fn recreate_graphics_pipeline(&mut self) {
        self.recreate_graphics_pipeline();
    }
}