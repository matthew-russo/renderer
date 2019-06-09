#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 1, binding = 0) uniform StaticUnfiorms {
    mat4 view;
    mat4 proj;
} s_ubo;

layout(set = 2, binding = 0) uniform DynamicUniforms {
    mat4 model;
} d_ubo;

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;
layout(location = 2) in vec2 in_tex_coord;

layout(location = 0) out vec3 frag_color;
layout(location = 1) out vec2 frag_tex_coord;

out gl_PerVertex {
    vec4 gl_Position;
};

void main() {
    frag_color = in_color;
    frag_tex_coord = in_tex_coord;

    gl_Position = s_ubo.proj * s_ubo.view * d_ubo.model * vec4(in_position, 1.0);
}

