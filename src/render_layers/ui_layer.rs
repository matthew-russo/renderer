use std::sync::{Arc, Mutex};
use std::collections::HashSet;

use vulkano::device::{Device, Queue};
use vulkano::buffer::{TypedBufferAccess, BufferAccess};
use vulkano::framebuffer::{RenderPassAbstract, Subpass};
use vulkano::command_buffer::{
    DynamicState,
    AutoCommandBufferBuilder,
    pool::standard::StandardCommandPoolBuilder,
};
use vulkano::pipeline::{
    GraphicsPipeline,
    viewport::Viewport,
    vertex::SingleBufferDefinition
};
use vulkano::descriptor::{
    pipeline_layout::PipelineLayoutDescUnion,
    pipeline_layout::PipelineLayout,
    pipeline_layout::PipelineLayoutDesc,
    descriptor::ShaderStages,
    descriptor_set::FixedSizeDescriptorSet,
    descriptor_set::FixedSizeDescriptorSetsPool
};

use crate::utils::vk_creation;
use crate::primitives;
use crate::primitives::vertex::Vertex;
use crate::render_layers::render_layer::RenderLayer;
use crate::events::application_events::ApplicationEvent;
use crate::primitives::two_d::{
    widget::EscMenu,
    widget::Widget,
    quad::Quad,
};

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

    // vertices: HashMap<String, Vec<Vertex>>,
    vertex_buffer: Arc<BufferAccess + Send + Sync>,

    // indices: HashMap<String, Vec<u32>>,
    index_buffer: Arc<TypedBufferAccess<Content=[u32]> + Send + Sync>,

    need_to_rebuild_buffers: bool,

    widgets: Vec<Arc<Mutex<Widget>>>,

    rendered_geometry: HashSet<Quad>,
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

            vertex_buffer: vk_creation::create_vertex_buffer::<Vertex>(&graphics_queue, &vec![]),

            index_buffer: vk_creation::create_index_buffer(&graphics_queue, &vec![]),

            need_to_rebuild_buffers: false,

            widgets: vec![Arc::new(Mutex::new(EscMenu::new()))],
            rendered_geometry: HashSet::new(),
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

            let mut vertices: Vec<Vertex> = Vec::new();
            let mut indices: Vec<u32> = Vec::new();

            for geo in self.rendered_geometry.iter() {
                vertices.append(&mut geo.vertices());
                indices.append(&mut geo.indices());
            }

            self.vertex_buffer = vk_creation::create_vertex_buffer(&self.graphics_queue, &vertices);
            self.index_buffer = vk_creation::create_index_buffer(&self.graphics_queue, &indices);
            self.need_to_rebuild_buffers = false;
        }
    }

    pub fn add_quad(&mut self, mut quad: Quad) {
        if !self.rendered_geometry.contains(&quad) {
            self.rendered_geometry.insert(quad);
            self.need_to_rebuild_buffers = true;
        }
    }

    pub fn remove_quad(&mut self, quad: Quad) {
        if self.rendered_geometry.contains(&quad) {
            self.rendered_geometry.remove(&quad);
            self.need_to_rebuild_buffers = true;
        }
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

    pub fn push_events_to_widgets(&mut self, events: Vec<ApplicationEvent>) {
        let mut need_to_rebuild_buffers = self.need_to_rebuild_buffers;

        events.iter().for_each(|e|
            self.widgets.iter().for_each(|w| need_to_rebuild_buffers = w.lock().unwrap().on(e))
        );

        self.need_to_rebuild_buffers = need_to_rebuild_buffers;
    }
}

impl RenderLayer for UiLayer {
    fn draw_indexed(&mut self, builder: AutoCommandBufferBuilder<StandardCommandPoolBuilder>) -> AutoCommandBufferBuilder<StandardCommandPoolBuilder> {
        let widgets = self.widgets.clone();
        for widget in widgets.iter() {

            let quad = widget.lock().unwrap().quad();
            if quad.rendered {
                self.add_quad(quad);
            } else {
                self.remove_quad(quad);
            }
        }

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