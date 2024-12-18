#version 410

// Uniforms
uniform mat4 model;         // Model transformation matrix
uniform mat4 view;          // View transformation matrix
uniform mat4 projection;    // Projection matrix

// Inputs
layout(location = 0) in vec3 vertex; // Vertex position
layout(location = 1) in vec3 normal; // Vertex normal

// Outputs to the fragment shader
out vec3 fragPosition; // Position in world space
out vec3 fragNormal;   // Normal in world space

void main() {
    // Transform vertex position to world space
    vec4 worldPosition = model * vec4(vertex, 1.0);
    fragPosition = worldPosition.xyz;

    // Transform normal to world space
    fragNormal = mat3(transpose(inverse(model))) * normal;

    // Transform position to clip space
    gl_Position = projection * view * worldPosition;
}
