#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 frag_color;
layout(location = 1) in vec2 frag_tex_coord;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(frag_color, 1.0);
}