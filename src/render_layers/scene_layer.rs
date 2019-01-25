use std::sync::{Arc, Mutex};
use std::path::Path;
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
use vulkano::buffer::ImmutableBuffer;
use vulkano::buffer::BufferUsage;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::command_buffer::pool::standard::StandardCommandPoolBuilder;
use vulkano::descriptor::descriptor_set::FixedSizeDescriptorSet;
use vulkano::pipeline::vertex::SingleBufferDefinition;
use vulkano::descriptor::pipeline_layout::PipelineLayout;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSetBuf;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSetImg;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSetSampler;

use image::GenericImageView;

use crate::primitives;
use crate::primitives::uniform_buffer_object::UniformBufferObject;
use crate::render_layers::render_layer::RenderLayer;
use crate::utils::asset_loading::*;
use crate::utils::vk_creation;
use crate::primitives::vertex::Vertex;
use crate::primitives::three_d::model::Model;
use crate::primitives::three_d::model::Transform;

use rand::Rng;

const MODEL_PATH: &'static str = "C:\\Users\\mcr43\\IdeaProjects\\vulkan_tutorial\\src\\data\\models\\chalet.obj";
const TEXTURE_PATH: &'static str = "C:\\Users\\mcr43\\IdeaProjects\\vulkan_tutorial\\src\\data\\textures\\chalet.jpg";

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
    descriptor_set: Arc<FixedSizeDescriptorSet<Arc<SceneGraphicsPipeline>, ((((), PersistentDescriptorSetBuf<Arc<ImmutableBuffer<[UniformBufferObject; 256]>>>), PersistentDescriptorSetImg<Arc<ImmutableImage<Format>>>), PersistentDescriptorSetSampler)>>,

    // todo ->
    // uniform_buffers: Vec<UniformBufferObject>,
    // images: Vec<Arc<(ImmutableImage<Format>, Sampler)>>,

    image_view: Arc<ImmutableImage<Format>>,
    image_sampler: Arc<Sampler>,

    models: Vec<Model>,

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
        let mut rng = rand::thread_rng();

        for i in 0..255 {
            let mut cube = Self::new_cube();
            let x = rng.gen_range(-5.0, 5.0);
            let y = rng.gen_range(-5.0, 5.0);
            let z = rng.gen_range(-5.0, 5.0);
            cube.transform.translate(glm::vec3(x, y, z));
            models.push(cube);
        }

        let image_view = Self::create_image_view(graphics_queue);
        let image_sampler = Self::create_image_sampler(&device);

        let start_time = Instant::now();

        let descriptor_set = Self::build_descriptor_set(
            &graphics_queue,
            descriptor_sets_pool.clone(),
            &image_view,
            &image_sampler,
            start_time,
            dimensions,
            &models,
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

            models,

            vertex_buffer: vk_creation::create_vertex_buffer(graphics_queue, &vec![Vertex::new([1.0,1.0,1.0],[1.0,1.0,1.0],[1.0,1.0])]),
            index_buffer: vk_creation::create_index_buffer(graphics_queue, &vec![0]),

            need_to_rebuild_buffers: false,

            start_time
        };

        scene_layer.rebuild_buffers();

        scene_layer
    }

    // todo -> move to cube.rs and make Model a trait that all standard 3d geo implements
    fn new_cube() -> Model {
        let red = [1.0, 0.0, 0.0];
        let green = [0.0, 1.0, 0.0];
        let blue = [0.0, 0.0, 1.0];

        let yellow = [1.0, 1.0, 0.0];
        let purple = [1.0, 0.0, 1.0];
        let blue_green = [0.0, 1.0, 1.0];

        let lower_x = -0.5;
        let lower_y = -0.5;
        let lower_z = -0.5;
        let upper_x = 0.5;
        let upper_y = 0.5;
        let upper_z = 0.5;


        let vertices = vec![
            // back face
            Vertex::new([lower_x, lower_y, lower_z], red, [0.0, 0.0]),
            Vertex::new([lower_x, upper_y, lower_z], red, [1.0, 0.0]),
            Vertex::new([upper_x, upper_y, lower_z], red, [1.0, 1.0]),
            Vertex::new([upper_x, lower_y, lower_z], red, [0.0, 1.0]),

            // front face
            Vertex::new([lower_x, lower_y, upper_z], green, [0.0, 0.0]),
            Vertex::new([upper_x, lower_y, upper_z], green, [1.0, 0.0]),
            Vertex::new([upper_x, upper_y, upper_z], green, [1.0, 1.0]),
            Vertex::new([lower_x, upper_y, upper_z], green, [0.0, 1.0]),

            // left face
            Vertex::new([lower_x, lower_y, upper_z], blue, [0.0, 0.0]),
            Vertex::new([lower_x, upper_y, upper_z], blue, [1.0, 0.0]),
            Vertex::new([lower_x, upper_y, lower_z], blue, [1.0, 1.0]),
            Vertex::new([lower_x, lower_y, lower_z], blue, [0.0, 1.0]),

            // right face
            Vertex::new([upper_x, lower_y, upper_z], yellow, [0.0, 0.0]),
            Vertex::new([upper_x, lower_y, lower_z], yellow, [1.0, 0.0]),
            Vertex::new([upper_x, upper_y, lower_z], yellow, [1.0, 1.0]),
            Vertex::new([upper_x, upper_y, upper_z], yellow, [0.0, 1.0]),

            // top face
            Vertex::new([lower_x, upper_y, upper_z], purple, [0.0, 0.0]),
            Vertex::new([upper_x, upper_y, upper_z], purple, [1.0, 0.0]),
            Vertex::new([upper_x, upper_y, lower_z], purple, [1.0, 1.0]),
            Vertex::new([lower_x, upper_y, lower_z], purple, [0.0, 1.0]),

            // bottom face
            Vertex::new([lower_x, lower_y, upper_z], blue_green, [0.0, 0.0]),
            Vertex::new([lower_x, lower_y, lower_z], blue_green, [1.0, 0.0]),
            Vertex::new([upper_x, lower_y, lower_z], blue_green, [1.0, 1.0]),
            Vertex::new([upper_x, lower_y, upper_z], blue_green, [0.0, 1.0]),
        ];

        let indices = vec![
            0, 1, 2, 2, 3, 0,
            4, 5, 6, 6, 7, 4,
            8, 9, 10, 10, 11, 8,
            12, 13, 14, 14, 15, 12,
            16, 17, 18, 18, 19, 16,
            20, 21, 22, 22, 23, 20,
        ];

        Model {
            key: "cube".to_string(),
            vertices,
            indices,

            transform: Transform::new()
        }
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
        start_time: Instant,
        dimensions: [f32; 2],
        models: &Vec<Model>
    ) -> Arc<FixedSizeDescriptorSet<Arc<SceneGraphicsPipeline>, ((((), PersistentDescriptorSetBuf<Arc<ImmutableBuffer<[UniformBufferObject; 256]>>>), PersistentDescriptorSetImg<Arc<ImmutableImage<Format>>>), PersistentDescriptorSetSampler)>> {
        let mut ubos: [UniformBufferObject; 256] = [Self::update_uniform_buffer(start_time, dimensions, &Transform::new()); 256];

        for (i, model) in models.iter().enumerate() {
           ubos[i] = Self::update_uniform_buffer(start_time, dimensions, &model.transform);
        }

        let (buffer, future) = ImmutableBuffer::from_data(
           ubos,
           BufferUsage::uniform_buffer(),
           graphics_queue.clone()
        ).unwrap();

        future.flush().unwrap();

        Arc::new(pool.lock().unwrap().next()
            .add_buffer(buffer).unwrap()
            .add_sampled_image(image_view.clone(), image_sampler.clone()) .unwrap()
            .build()
            .unwrap())
    }

    fn rebuild_buffers(&mut self) {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        for (i, model) in self.models.iter().enumerate() {
            let mut new_vertices = model.vertices.clone();
            vertices.append(&mut new_vertices);

            let mut new_indices = model.indices.iter().map(|index| *index + (4 * i as u32)).collect();
            indices.append(&mut new_indices);
        }

        self.vertex_buffer = vk_creation::create_vertex_buffer(&self.graphics_queue, &vertices);
        self.index_buffer = vk_creation::create_index_buffer(&self.graphics_queue, &indices);
    }

    //fn get_vertex_buffer(&mut self, vertices: &Vec<Vertex>, indices: &Vec<u32>) -> Arc<BufferAccess + Send + Sync> {
    //    self.rebuild_buffers_if_necessary(vertices, indices);
    //    self.vertex_buffer.clone()
    //}

    //fn get_index_buffer(&mut self, vertices: &Vec<Vertex>, indices: &Vec<u32>) -> Arc<TypedBufferAccess<Content=[u32]> + Send + Sync> {
    //    self.rebuild_buffers_if_necessary(vertices, indices);
    //    self.index_buffer.clone()
    //}

    fn update_uniform_buffer(start_time: Instant, dimensions: [f32; 2], transform: &Transform) -> UniformBufferObject {
        let duration = Instant::now().duration_since(start_time);
        let elapsed = duration.as_millis();

        let identity_matrix = glm::mat4(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        );

        let mut model = glm::ext::translate(&identity_matrix, transform.position);
        model = glm::ext::scale(&model, transform.scale);
        model = glm::ext::rotate(&model, (elapsed as f32) * glm::radians(0.180), glm::vec3(0.0, 0.5, 1.0) /*transform.rotation*/);

        let view = glm::ext::look_at(
            glm::vec3(0.0, 2.0, -15.0),
            glm::vec3(0.0, 0.0, 0.0),
            glm::vec3(0.0, 1.0, 0.0)
        );
        let mut proj = glm::ext::perspective(
            glm::radians(45.0,),
            dimensions[0] as f32 / dimensions[1] as f32,
            0.1,
            1000.0
        );

        proj.c1.y *= -1.0;

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
}

#[repr(C)]
struct PushConstant {
    value: u32,
}

impl RenderLayer for SceneLayer {
    fn draw_indexed(&mut self, mut builder: AutoCommandBufferBuilder<StandardCommandPoolBuilder>) -> AutoCommandBufferBuilder<StandardCommandPoolBuilder> {
        self.descriptor_set = Self::build_descriptor_set(
            &self.graphics_queue,
            self.descriptor_sets_pool.clone(),
            &self.image_view,
            &self.image_sampler,
            self.start_time,
            self.dimensions.clone(),
            &self.models
        );

        let models = self.models.clone();

        for (i, model) in models.iter().enumerate() {
            let push_constant = vs::ty::PushConstant {
                value: i as u32,
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

