#version 450

uniform bool      u_is_screen;       // true = LED screen (emissive, textured from channel)
uniform bool      u_has_texture;     // true = channel texture bound
uniform sampler2D u_texture;         // channel output texture
uniform vec3      u_base_color;      // diffuse base color for non-screen meshes
uniform vec3      u_light_dir;       // directional light direction (world space, normalised)
uniform float     u_ambient;         // ambient light intensity

in vec3 v_WorldPos;
in vec3 v_Normal;
in vec2 v_TexCoord;

out vec4 FragColor;

void main()
{
    if (u_is_screen) {
        // LED screen: emissive, sample channel texture directly
        if (u_has_texture) {
            FragColor = texture(u_texture, v_TexCoord);
        } else {
            // No texture bound: show dark gray placeholder
            FragColor = vec4(0.15, 0.15, 0.15, 1.0);
        }
    } else {
        // Non-screen geometry: basic diffuse lighting
        vec3 N = normalize(v_Normal);
        float NdotL = max(dot(N, u_light_dir), 0.0);
        vec3 color = u_base_color * (u_ambient + (1.0 - u_ambient) * NdotL);
        FragColor = vec4(color, 1.0);
    }
}
