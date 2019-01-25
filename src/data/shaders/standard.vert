#version 450
#extension GL_ARB_separate_shader_objects : enable

struct UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
};

layout(set = 0, binding = 0) uniform UniformBufferObjects {
    UniformBufferObject ubos[1];
} ubos;

layout(push_constant) uniform PushConstant {
    uint value;
} uboIndex;

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;
layout(location = 2) in vec2 in_tex_coord;

layout(location = 0) out vec3 frag_color;
layout(location = 1) out vec2 frag_tex_coord;

void main() {
    UniformBufferObject ubo = ubos.ubos[uboIndex.value];
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(in_position, 1.0);

    frag_color = in_color;
    frag_tex_coord = in_tex_coord;
}