#version 410

uniform mat4 vertex_model_to_world;
uniform mat4 vertex_world_to_clip;

layout(location = 0) in vec3 vertex;
layout(location = 1) in vec2 texcoord; // Input texture coordinates

out vec2 fragTexcoord; // Output texture coordinates to the fragment shader

void main()
{
    vec4 world_position = vertex_model_to_world * vec4(vertex, 1.0); // Transform to world space
    gl_Position = vertex_world_to_clip * world_position; // Transform to clip space
    fragTexcoord = texcoord; // Pass texture coordinates
}