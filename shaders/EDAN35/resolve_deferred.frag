#version 410

uniform sampler2D diffuse_texture;
uniform sampler2D specular_texture;
uniform sampler2D light_d_texture;
uniform sampler2D light_s_texture;
uniform float ambient;
uniform vec3 fog_color;               // Color of the fog
uniform float fog_bottom;             // Height where fog starts
uniform float fog_top;
uniform float time;
uniform sampler2D position_texture;

layout(pixel_center_integer) in vec4 gl_FragCoord;

out vec4 frag_color;

void main() {
	ivec2 pixel_coord = ivec2(gl_FragCoord.xy);

	vec3 diffuse = texelFetch(diffuse_texture, pixel_coord, 0).rgb;
	vec3 specular = texelFetch(specular_texture, pixel_coord, 0).rgb;

	vec3 light_d = texelFetch(light_d_texture, pixel_coord, 0).rgb;
	vec3 light_s = texelFetch(light_s_texture, pixel_coord, 0).rgb;

	//vec3 world_position = texelFetch(position_texture, pixel_coord, 0).rgb;
	float wave = sin(3 * 2.0 + time * 1.0) * 0.5; // Adjust frequency and amplitude
	float adjusted_fog_bottom = fog_bottom + wave;
	float adjusted_fog_top = fog_top + wave;

	float fragment_height = 2;                   // Fragment's height             
    float fog_factor = clamp((fog_top - fragment_height) / (fog_top - fog_bottom), 0.0, 1.0);

	vec3 scene_color = vec3((ambient + light_d) * diffuse + light_s * specular);

	vec3 final_color = mix(scene_color, fog_color, fog_factor);

	frag_color = vec4(final_color, 1.0);

}