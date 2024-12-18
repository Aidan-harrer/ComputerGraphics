#version 410

out vec4 fragColor;
in vec2 fragTexcoord;
uniform vec3 glowColor;  // The color of the glow
uniform float glowIntensity; // Intensity of the glow

void main() {
	vec2 h = fragTexcoord 	;
    fragColor = vec4(glowColor, 1.0f) * glowIntensity;
}