#version 410

out vec4 fragColor;

uniform vec3 glowColor;  // The color of the glow
uniform float glowIntensity; // Intensity of the glow

void main() {
    fragColor = vec4(glowColor, 1.0f) * glowIntensity;
}