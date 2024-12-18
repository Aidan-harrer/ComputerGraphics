#version 410

// Inputs from vertex shader
in vec3 fragPosition;
in vec3 fragNormal;

// Uniforms
uniform vec3 lightPosition;  // Position of the light in world space
uniform vec3 viewPosition;   // Position of the camera in world space
uniform vec3 lightColor;     // Light color
uniform vec3 baseColor;      // Base color of the diamond
uniform float shininess;     // Shininess factor for specular reflection

// Output color
out vec4 fragColor;

void main() {
    // Normalize the normal vector
    vec3 normal = normalize(fragNormal);

    // Calculate the light direction
    vec3 lightDir = normalize(lightPosition - fragPosition);

    // Calculate diffuse lighting
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    // Calculate the view direction
    vec3 viewDir = normalize(viewPosition - fragPosition);

    // Calculate the reflection vector
    vec3 reflectDir = reflect(-lightDir, normal);

    // Calculate specular lighting
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
    vec3 specular = spec * lightColor;

    // Combine results
    vec3 lighting = diffuse + specular;

    // Apply lighting to the base color
    vec3 finalColor = baseColor * lighting;

    // Output the final color
    fragColor = vec4(finalColor, 1.0);
}
