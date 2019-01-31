use std::sync::{Arc, Mutex};
use std::time::Instant;

use vulkano::buffer::TypedBufferAccess;
use vulkano::framebuffer::RenderPassAbstract;
use vulkano::device::Device;
use vulkano::device::Queue;
use vulkano::image::ImmutableImage;
use vulkano::sampler::Sampler;
use vulkano::descriptor::descriptor_set::FixedSizeDescriptorSetsPool;
use vulkano::image::Dimensions;
use vulkano::command_buffer::DynamicState;
use vulkano::pipeline::viewport::Viewport;
use vulkano::descriptor::descriptor::ShaderStages;
use vulkano::descriptor::pipeline_layout::PipelineLayoutDescUnion;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::pipeline::depth_stencil::DepthStencil;
use vulkano::framebuffer::Subpass;
use vulkano::format::Format;
use vulkano::descriptor::pipeline_layout::PipelineLayoutDesc;
use vulkano::sync::GpuFuture;
use vulkano::buffer::BufferAccess;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::command_buffer::pool::standard::StandardCommandPoolBuilder;
use vulkano::descriptor::descriptor_set::FixedSizeDescriptorSet;
use vulkano::pipeline::vertex::SingleBufferDefinition;
use vulkano::descriptor::pipeline_layout::PipelineLayout;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSetImg;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSetSampler;

use image::GenericImageView;

use crate::primitives;
use crate::primitives::uniform_buffer_object::UniformBufferObject;
use crate::render_layers::render_layer::RenderLayer;
use crate::utils::vk_creation;
use crate::primitives::vertex::Vertex;
use crate::primitives::three_d::model::Model;
use crate::components::transform::Transform;

use std::collections::HashMap;
use cgmath::{
    Deg,
    Basis3,
    Point3,
    Vector3,
    Matrix4,
    SquareMatrix,
    Transform as cgTransform,
    Quaternion,
    perspective,
};

const MODEL_PATH: &'static str = "C:\\Users\\mcr43\\IdeaProjects\\vulkan_tutorial\\src\\data\\models\\chalet.obj";
// const TEXTURE_PATH: &'static str = "C:\\Users\\mcr43\\IdeaProjects\\vulkan_tutorial\\src\\data\\textures\\chalet.jpg";
const TEXTURE_PATH: &'static str = "C:\\Users\\mcr43\\IdeaProjects\\rendering_engine\\src\\data\\textures\\demo.jpg";

mod vs {
    vulkano_shaders::shader! {
       ty: "vertex",
       path: "src\\data\\shaders\\standard.vert"
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src\\data\\shaders\\standard.frag"
    }
}

type SceneGraphicsPipeline = GraphicsPipeline<SingleBufferDefinition<primitives::vertex::Vertex>, PipelineLayout<PipelineLayoutDescUnion<PipelineLayout<vs::Layout>, PipelineLayout<fs::Layout>>>, Arc<dyn RenderPassAbstract + Send + Sync>>;

pub struct SceneLayer {
    device: Arc<Device>,
    render_pass: Arc<RenderPassAbstract + Send + Sync>,
    dimensions: [f32; 2],

    graphics_queue: Arc<Queue>,
    graphics_pipeline: Arc<SceneGraphicsPipeline>,

    descriptor_sets_pool: Arc<Mutex<FixedSizeDescriptorSetsPool<Arc<SceneGraphicsPipeline>>>>,
    descriptor_set: Arc<FixedSizeDescriptorSet<Arc<SceneGraphicsPipeline>, (((), PersistentDescriptorSetImg<Arc<ImmutableImage<Format>>>), PersistentDescriptorSetSampler)>>,

    // todo ->
    // uniform_buffers: Vec<UniformBufferObject>,
    // images: Vec<Arc<(ImmutableImage<Format>, Sampler)>>,

    image_view: Arc<ImmutableImage<Format>>,
    image_sampler: Arc<Sampler>,

    pub camera_transform: Transform,

    models: Vec<Model>,

    vertices: Vec<Vertex>,
    indices: Vec<u32>,

    vertex_buffer: Arc<BufferAccess + Send + Sync>,
    index_buffer: Arc<TypedBufferAccess<Content=[u32]> + Send + Sync>,

    need_to_rebuild_buffers: bool,
    start_time: Instant,
}

// todo -> this and ui layer share a ton of logic. probably best to move some of the stuff into a default trait impl
impl SceneLayer {
    pub fn new(
        graphics_queue: &Arc<Queue>,
        device: &Arc<Device>,
        render_pass: &Arc<RenderPassAbstract + Send + Sync>,
        dimensions: [f32; 2]
    ) -> SceneLayer {
        let graphics_pipeline = Self::create_graphics_pipeline(device, render_pass, dimensions.clone());

        let descriptor_sets_pool = Arc::new(
            Mutex::new(
                FixedSizeDescriptorSetsPool::new(graphics_pipeline.clone(), 0)
            )
        );

        // todo -> make geometry and images dynamic
        //let model = load_model(Path::new(MODEL_PATH));

        let mut models = Vec::new();


        let image_view = Self::create_image_view(graphics_queue);
        let image_sampler = Self::create_image_sampler(&device);

        let start_time = Instant::now();

        let descriptor_set = Self::build_descriptor_set(
            &graphics_queue,
            descriptor_sets_pool.clone(),
            &image_view,
            &image_sampler,
        );

        let mut scene_layer = SceneLayer {
            render_pass: render_pass.clone(),
            device: device.clone(),
            dimensions: dimensions.clone(),

            graphics_queue: graphics_queue.clone(),
            graphics_pipeline,

            descriptor_sets_pool,
            descriptor_set,

            image_view,
            image_sampler,

            camera_transform: Transform::new(),

            models,

            vertices: vec![],
            indices: vec![],

            vertex_buffer: vk_creation::create_vertex_buffer(graphics_queue, &vec![Vertex::new([1.0,1.0,1.0],[1.0,1.0,1.0],[1.0,1.0])]),
            index_buffer: vk_creation::create_index_buffer(graphics_queue, &vec![0]),

            need_to_rebuild_buffers: false,

            start_time
        };

        scene_layer
    }

    fn create_graphics_pipeline(
        device: &Arc<Device>,
        render_pass: &Arc<RenderPassAbstract + Send + Sync>,
        dimensions: [f32; 2]
    ) -> Arc<SceneGraphicsPipeline> {
        Self::create_basic_graphics_pipeline(device, render_pass, dimensions, "standard")
    }

    fn recreate_graphics_pipeline(&mut self) {
        self.graphics_pipeline = Self::create_basic_graphics_pipeline(&self.device, &self.render_pass, self.dimensions.clone(), "standard");
    }

    fn build_descriptor_set(
        graphics_queue: &Arc<Queue>,
        pool: Arc<Mutex<FixedSizeDescriptorSetsPool<Arc<SceneGraphicsPipeline>>>>,
        image_view: &Arc<ImmutableImage<Format>>,
        image_sampler: &Arc<Sampler>,
    ) -> Arc<FixedSizeDescriptorSet<Arc<SceneGraphicsPipeline>, (((), PersistentDescriptorSetImg<Arc<ImmutableImage<Format>>>), PersistentDescriptorSetSampler)>> {
        Arc::new(pool.lock().unwrap().next()
            .add_sampled_image(image_view.clone(), image_sampler.clone()) .unwrap()
            .build()
            .unwrap())
    }

    fn update_uniform_buffer(start_time: Instant, dimensions: [f32; 2], model_transform: &Transform, camera_transform: &Transform) -> UniformBufferObject {
        let translation = Matrix4::from_translation(model_transform.position);
        // let rotation = Matrix4::from::<Quaternion<f32>>(&model_transform.rotation);

        let scale = Matrix4::from_nonuniform_scale(model_transform.scale.x, model_transform.scale.y, model_transform.scale.z);
        let model = Matrix4::identity()
            .concat(&translation)
            .concat(&model_transform.rotation.into())
            .concat(&scale);


        let position = camera_transform.position;
        let view = Matrix4::look_at(
            Point3::new(position.x, position.y, position.z),
            Point3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0)
        );

        let mut proj = perspective(
            Deg(45.0),
            dimensions[0] as f32 / dimensions[1] as f32,
            0.1,
            1000.0
        );

        proj.y.y *= -1.0;

        UniformBufferObject::new(model, view, proj)
    }

    fn create_image_view(graphics_queue: &Arc<Queue>) -> Arc<ImmutableImage<Format>> {
        //let image = image::open("/Users/matthewrusso/rust/vulkano_tutorial/src/data.textures/demo.jpg").unwrap();
        // "C:\\Users\\mcr43\\IdeaProjects\\vulkan_tutorial\\src\\data.textures\\demo.jpg"
        let image = image::open(TEXTURE_PATH).unwrap();
        let width = image.width();
        let height = image.height();

        let rbgba = image.to_rgba();

        let (image_view, future) = ImmutableImage::from_iter(
            rbgba.into_raw().iter().cloned(),
            Dimensions::Dim2d{width, height},
            Format::R8G8B8A8Unorm,
            graphics_queue.clone()
        ).unwrap();

        future.flush().unwrap();

        return image_view
    }

    fn create_image_sampler(device: &Arc<Device>) -> Arc<Sampler> {
        Sampler::simple_repeat_linear(device.clone())
    }

    fn create_basic_graphics_pipeline(
        device: &Arc<Device>,
        render_pass: &Arc<RenderPassAbstract + Send + Sync>,
        dimensions: [f32; 2],
        shader: &'static str
    ) -> Arc<SceneGraphicsPipeline> {
        let vertex = vs::Shader::load(device.clone()).expect("failed to create vertex shader modules");
        let frag = fs::Shader::load(device.clone()).expect("failed to create frag shader modules");

        let viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions,
            depth_range: 0.0..1.00,
        };

        let vs_layout = vs::Layout(ShaderStages { vertex: true, ..ShaderStages::none() }).build(device.clone()).unwrap();
        let fs_layout = fs::Layout(ShaderStages { fragment: true, ..ShaderStages::none() }).build(device.clone()).unwrap();
        let layout_union = PipelineLayoutDescUnion::new(vs_layout, fs_layout).build(device.clone()).unwrap();

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
            .cull_mode_back()
            .front_face_counter_clockwise()
            .blend_pass_through()
            .depth_stencil(DepthStencil::simple_depth_test())
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .with_pipeline_layout(device.clone(), layout_union)
            .unwrap()
        )
    }


    pub fn add_models(&mut self, mut models: Vec<Model>) {
        let mut prev_max_index = match self.indices.iter().max() {
            Some(max) => *max,
            None => 0,
        };

        for model in models.iter() {
            self.vertices.append(&mut model.vertices.iter().cloned().collect());

            // todo -> normalize model indices to start at 0.
            let mut max_model_index = model.indices.iter().max().unwrap();

            let mut model_indices = model.indices.iter().map(|i| *i + max_model_index + 1).collect();
            self.indices.append(&mut model_indices);

            prev_max_index = prev_max_index + max_model_index;
        }

        self.vertex_buffer = vk_creation::create_vertex_buffer(&self.graphics_queue, &self.vertices);
        self.index_buffer = vk_creation::create_index_buffer(&self.graphics_queue, &self.indices);
        self.models.append(&mut models);
    }
}

impl RenderLayer for SceneLayer {
    fn draw_indexed(&mut self, mut builder: AutoCommandBufferBuilder<StandardCommandPoolBuilder>, renderables: &HashMap<String, Transform>) -> AutoCommandBufferBuilder<StandardCommandPoolBuilder> {
        for model in self.models.iter_mut() {
            match renderables.get(&model.key) {
                Some(transform) => model.transform = transform.clone(),
                None => (),
            }
        }

        self.descriptor_set = Self::build_descriptor_set(
            &self.graphics_queue,
            self.descriptor_sets_pool.clone(),
            &self.image_view,
            &self.image_sampler,
        );

        for (i, model) in self.models.iter().enumerate() {
            let ubo = Self::update_uniform_buffer(self.start_time, self.dimensions, &model.transform, &self.camera_transform);
            let (model, view, proj) = ubo.as_arrays();

            let push_constant = vs::ty::UniformBufferObject {
                model,
                view,
                proj
            };

            builder = builder.draw_indexed(
                self.graphics_pipeline.clone(),
                &DynamicState::none(),
                vec![self.vertex_buffer.clone()],
                self.index_buffer.clone(),
                self.descriptor_set.clone(),
                push_constant
            ).unwrap()
        }

        builder
    }

    fn recreate_graphics_pipeline(&mut self) {
        self.recreate_graphics_pipeline();
    }
}

