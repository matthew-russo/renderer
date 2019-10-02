#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 2, binding = 0) uniform sampler2D tex_sampler;

layout(location = 0) in vec3 frag_color;
layout(location = 1) in vec2 frag_tex_coord;

layout(location = 0) out vec4 outColor;

void main() {
    // outColor = texture(tex_sampler, frag_tex_coord);
    // outColor = vec4(frag_color, 1.0);
    outColor = vec4(0.5, 0.0, 0.0, 1.0);
}

