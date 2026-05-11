#version 450

uniform mat4 u_mvp;
uniform mat4 u_model;

in vec3 a_Position;
in vec3 a_Normal;
in vec2 a_TexCoord;

out vec3 v_WorldPos;
out vec3 v_Normal;
out vec2 v_TexCoord;

void main()
{
    vec4 world = u_model * vec4(a_Position, 1.0);
    v_WorldPos = world.xyz;
    v_Normal   = mat3(u_model) * a_Normal;
    v_TexCoord = a_TexCoord;
    gl_Position = u_mvp * vec4(a_Position, 1.0);
}
