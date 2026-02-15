import pygame
import moderngl
import numpy as np
import sys
import time
from scipy.spatial import KDTree

# --- 1. INITIALIZATION ---
pygame.init()
pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 2)

info = pygame.display.Info()
SW, SH = info.current_w, info.current_h
screen = pygame.display.set_mode((SW, SH), pygame.OPENGL | pygame.DOUBLEBUF)
ctx = moderngl.create_context()

# --- 2. PARAMETERS ---
SUBDIVISIONS = 7
NUM_STARS = 5000
SUN_DIST, SUN_SCALE, SUN_SPEED = 14.0, 0.8, 0.15
MOON_DIST, MOON_SCALE, MOON_ORBIT_SPEED = 3.5, 0.2, -0.3

CHANCE_SPREAD_DIRECT = 1.0
CHANCE_SPREAD_DIAG = 0.005
BASE_IGNITION_CHANCE = 0.02
BASE_REGROW_CHANCE = 0.0003
SIM_SPEED_MODIFIER = 1.0
BURN_DURATION = 60

INTRO_DURATION = 20.0
BASE_ASH_DURATION = 80
FRICTION = 0.96
SELF_ROT_SPEED = 0.002
TPS_TARGET = 60
TICK_TIME = 1.0 / TPS_TARGET
MAX_PHYSICS_STEPS = 4
MIN_ZOOM, MAX_ZOOM = 2.5, 25.0

# MINING
STATION_RADIUS = 0.09
COOLDOWN_TIME = 20.0
MINING_TICK = 0.1

START_TIME = time.time()

# --- GAME STATE ---
station_indices = []
station_center_local = np.array([0, 0, 0], dtype='f4')
station_deployed = False

emeralds = 0.0
highscore = 0.0
next_deploy_time = 0.0
last_mining_tick = 0.0
current_rate = 0

ui_stats = {"terra": 0, "exo": 0, "fire_terra": 0, "fire_exo": 0}
last_stat_update = 0.0

game_msg = "READY TO DEPLOY"
game_msg_col = (150, 200, 200)

# --- 3. COLORS & FONTS ---
C_MATCHA_BG = (15, 25, 20, 230)
C_MATCHA_LINE = (100, 140, 110)
C_MATCHA_TEXT = (210, 230, 215)
C_BAR_GREEN, C_BAR_BLUE = (80, 200, 120), (100, 160, 240)
C_FIRE_ORANGE, C_FIRE_VIOLET = (255, 120, 40), (180, 100, 255)
C_SUN_YELLOW, C_MOON_GREY = (255, 220, 100), (180, 200, 220)
C_EMERALD = (50, 255, 180)
C_ALERT = (255, 50, 50)
C_HOSE = (90, 110, 100)
C_PROBE_METAL = (50, 60, 65)

try:
    UI_FONT_L = pygame.font.SysFont('Consolas', 28, bold=True)
    UI_FONT = pygame.font.SysFont('Consolas', 20, bold=True)
    UI_FONT_S = pygame.font.SysFont('Consolas', 13, bold=True)
    UI_FONT_T = pygame.font.SysFont('Consolas', 11, bold=True)
except:
    UI_FONT_L = UI_FONT = UI_FONT_S = UI_FONT_T = pygame.font.SysFont('Arial', 14, bold=True)


# --- 4. MATH ---
def projection_matrix(fov, aspect, near, far):
    f = 1.0 / np.tan(np.radians(fov) / 2.0)
    return np.array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) / (near - far), -1, 0, 0,
                     (2.0 * far * near) / (near - far), 0], dtype='f4')


def get_model_matrix(translation=(0, 0, 0), scale=1.0, rotation=0.0):
    tx, ty, tz = translation
    cr, sr = np.cos(rotation), np.sin(rotation)
    return np.array([scale * cr, 0, -sr, 0, 0, scale, 0, 0, sr, 0, scale * cr, 0, tx, ty, tz, 1], dtype='f4')


def get_mvp_matrix(proj_flat, view_flat, model_flat):
    return model_flat.reshape(4, 4) @ view_flat.reshape(4, 4) @ proj_flat.reshape(4, 4)


# Robuste View-Matrix
def create_view_matrix(rx, ry, zoom):
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    U = np.array([cy, 0, -sy], dtype='f4')
    V = np.array([sx * sy, cx, sx * cy], dtype='f4')
    N = np.array([cx * sy, -sx, cx * cy], dtype='f4')
    return np.array([U[0], V[0], N[0], 0, U[1], V[1], N[1], 0, U[2], V[2], N[2], 0, 0, 0, -zoom, 1], dtype='f4')


def project_point_robust(point_3d, mvp_matrix):
    pt = np.append(point_3d, 1.0)
    clip = pt @ mvp_matrix
    w = clip[3]
    if w <= 0.1: return None
    ndc = clip[:2] / w
    sx = (ndc[0] + 1) * SW / 2
    sy = (1 - ndc[1]) * SH / 2
    return (sx, sy)


def gen_bolt_long():
    pts = [np.array([0.0, 0.0, 0.0])]
    curr = pts[0]
    for i in range(60):
        f = i * 0.05
        rnd = np.random.uniform(-0.4 * (1 + f), 0.4 * (1 + f), 3)
        rnd[1] = abs(rnd[1]) + 2.5 * (1 + f * 0.4)
        curr = curr + rnd
        pts.append(curr)
    res = []
    for i in range(len(pts) - 1): res.extend([pts[i], pts[i + 1]])
    return np.array(res, dtype='f4')


def create_flat_mesh(subdiv=3):
    t = (1.0 + 5.0 ** 0.5) / 2.0
    v = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t], [t, 0, -1],
         [t, 0, 1], [-t, 0, -1], [-t, 0, 1]]
    f = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6],
         [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7],
         [9, 8, 1]]
    for _ in range(subdiv):
        nf = []
        for tri in f:
            v1, v2, v3 = np.array(v[tri[0]]), np.array(v[tri[1]]), np.array(v[tri[2]])
            m1, m2, m3 = (v1 + v2) / 2, (v2 + v3) / 2, (v3 + v1) / 2
            sid = len(v);
            v.extend([m1.tolist(), m2.tolist(), m3.tolist()])
            nf.extend(
                [[tri[0], sid, sid + 2], [tri[1], sid + 1, sid], [tri[2], sid + 2, sid + 1], [sid, sid + 1, sid + 2]])
        f = nf
    return np.array([np.array(v[i]) / np.linalg.norm(v[i]) for tri in f for i in tri], dtype='f4')


# --- 5. DATA SETUP ---
VERTS = create_flat_mesh(SUBDIVISIONS)
NUM_VERTS = len(VERTS)
NUM_FACES = NUM_VERTS // 3
FACE_CENTERS = np.array([np.mean(VERTS[i:i + 3], axis=0) for i in range(0, NUM_VERTS, 3)])
tree = KDTree(FACE_CENTERS)
_, ALL_NB = tree.query(FACE_CENTERS, k=14)
NB_DIR, NB_DIAG = ALL_NB[:, 1:5], ALL_NB[:, 5:14]
_, NB_BOLT = tree.query(FACE_CENTERS, k=45)

# --- 6. SHADERS ---
v_shader = """#version 330
in vec3 in_pos;
in float in_type;    
in float in_burn;    
in float in_ash;     
in float in_fire_timer;
in float in_state;   

uniform mat4 u_proj, u_view, u_model; 
uniform vec3 u_sun_pos, u_moon_pos; 
uniform float u_time; 
uniform int u_mode; 

flat out vec3 v_color;
flat out float v_state_out;
out vec3 v_pos_local; 

const vec3 C_WATER = vec3(0.01, 0.01, 0.03);
const vec3 C_ASH   = vec3(0.015, 0.015, 0.015);
const vec3 C_GREEN = vec3(0.05, 0.4, 0.12);
const vec3 C_BLUE  = vec3(0.08, 0.15, 0.5);
const vec3 C_FIRE1 = vec3(1.6, 0.5, 0.1);
const vec3 C_FIRE2 = vec3(0.9, 0.1, 1.7);

void main() {
    v_state_out = in_state;
    v_pos_local = in_pos; 

    float f_fac = 0.0;
    if (u_mode == 0) f_fac = clamp(in_fire_timer / 60.0, 0.0, 1.0);

    vec4 w_pos = u_model * vec4(in_pos * (1.0 + f_fac * 0.05 * (1.2 + sin(u_time * 15.0))), 1.0);
    gl_Position = u_proj * u_view * w_pos;

    if (u_mode == 1) {
        v_color = vec3(2.5, 2.0, 1.2); 
    } else {
        vec3 norm = normalize((u_model * vec4(in_pos, 0.0)).xyz);
        float d_sun = max(dot(norm, normalize(u_sun_pos - w_pos.xyz)), 0.0);
        float d_moon = max(dot(norm, normalize(u_moon_pos - w_pos.xyz)), 0.0) * 0.5;
        vec3 light = (d_sun * vec3(1.0, 0.95, 0.85)) + (d_moon * vec3(0.7, 0.8, 1.0));

        if (u_mode == 2) {
            v_color = vec3(0.6, 0.65, 0.75) * (light + 0.1); 
        } else {
            vec3 base_col = C_WATER;
            if (in_ash > 0.0) base_col = C_ASH;
            else {
                int t = int(in_type);
                if (t == 1) base_col = C_GREEN;
                else if (t == 2) base_col = C_BLUE;
            }
            int b = int(in_burn);
            if (b == 1) base_col = C_FIRE1;
            else if (b == 2) base_col = C_FIRE2;
            v_color = base_col * (light + 0.15) + (base_col * f_fac * 4.0);
        }
    }
}"""

f_shader = """#version 330
flat in vec3 v_color;
flat in float v_state_out;
in vec3 v_pos_local;
uniform float u_time;
uniform vec3 u_station_center; 
uniform float u_station_radius; 
uniform int u_station_active; 
uniform int u_mode;

out vec4 f_out;
void main() {
    vec4 final_color = vec4(v_color, 1.0);
    if (u_mode == 0) {
        if (u_station_active == 1) {
            float dist = distance(v_pos_local, u_station_center);
            if (dist < u_station_radius) {
                float pulse = 0.5 + 0.2 * sin(u_time * 8.0);
                float rim = smoothstep(u_station_radius - 0.02, u_station_radius, dist);
                vec3 red_overlay = vec3(1.0, 0.0, 0.0) * (pulse + rim * 2.0);
                final_color = vec4(mix(v_color, red_overlay, 0.6), 1.0);
            }
        }
        if (v_state_out > 1.9) final_color = vec4(1.0, 1.0, 0.5, 1.0);
    }
    f_out = final_color;
}"""
prog = ctx.program(vertex_shader=v_shader, fragment_shader=f_shader)

prog_bolt = ctx.program(
    vertex_shader="#version 330\n in vec3 in_pos; uniform mat4 u_proj, u_view; uniform vec3 u_offset; void main() { vec3 n = normalize(u_offset); vec3 up = abs(n.y) < 0.9 ? vec3(0,1,0) : vec3(1,0,0); vec3 r = normalize(cross(up, n)), b = cross(n, r); gl_Position = u_proj * u_view * vec4((in_pos.x * r + in_pos.z * b + in_pos.y * n) + u_offset, 1.0); }",
    fragment_shader="#version 330\n uniform vec3 u_color; out vec4 f_out; void main() { f_out = vec4(u_color * 12.0, 1.0); }")
prog_stars = ctx.program(
    vertex_shader="#version 330\n in vec3 in_pos; uniform mat4 u_proj, u_view; void main() { gl_Position = (u_proj * u_view * vec4(in_pos * 90.0, 1.0)).xyww; gl_PointSize = 1.5; }",
    fragment_shader="#version 330\n out vec4 f_out; void main() { f_out = vec4(0.8, 0.85, 0.9, 0.6); }")
prog_atmo = ctx.program(
    vertex_shader="#version 330\n in vec3 in_pos; uniform mat4 u_proj, u_view, u_model; out vec3 v_norm, v_pos; void main() { vec4 vp = u_view * u_model * vec4(in_pos * 1.06, 1.0); v_pos = vp.xyz; v_norm = normalize((u_view * u_model * vec4(in_pos, 0.0)).xyz); gl_Position = u_proj * vp; }",
    fragment_shader="#version 330\n in vec3 v_norm, v_pos; out vec4 f_out; void main() { float f = pow(1.0 - max(dot(v_norm, normalize(-v_pos)), 0.0), 3.0); f_out = vec4(0.3, 0.6, 1.0, f * 0.4); }")
prog_ui = ctx.program(
    vertex_shader="#version 330\n in vec2 in_vert, in_uv; out vec2 v_uv; void main() { gl_Position = vec4(in_vert, 0.0, 1.0); v_uv = in_uv; }",
    fragment_shader="#version 330\n uniform sampler2D u_texture; in vec2 v_uv; out vec4 f_out; void main() { f_out = texture(u_texture, v_uv); }")

vbo_pos = ctx.buffer(VERTS.tobytes())
vbo_type = ctx.buffer(reserve=NUM_VERTS * 4)
vbo_burn = ctx.buffer(reserve=NUM_VERTS * 4)
vbo_ash = ctx.buffer(reserve=NUM_VERTS * 4)
vbo_ftimer = ctx.buffer(reserve=NUM_VERTS * 4)
vbo_state = ctx.buffer(reserve=NUM_VERTS * 4)

vao = ctx.vertex_array(prog, [
    (vbo_pos, '3f', 'in_pos'),
    (vbo_type, '1f', 'in_type'),
    (vbo_burn, '1f', 'in_burn'),
    (vbo_ash, '1f', 'in_ash'),
    (vbo_ftimer, '1f', 'in_fire_timer'),
    (vbo_state, '1f', 'in_state')
])
vao_atmo = ctx.vertex_array(prog_atmo, [(vbo_pos, '3f', 'in_pos')])
ui_vbo = ctx.buffer(np.array([-1, 1, 0, 0, -1, -1, 0, 1, 1, 1, 1, 0, 1, -1, 1, 1], dtype='f4'))
ui_vao = ctx.vertex_array(prog_ui, [(ui_vbo, '2f 2f', 'in_vert', 'in_uv')])
ui_surface = pygame.Surface((SW, SH), pygame.SRCALPHA)
ui_tex = ctx.texture((SW, SH), 4)

vbo_bolt = ctx.buffer(reserve=gen_bolt_long().nbytes * 2)
vao_bolt = ctx.vertex_array(prog_bolt, [(vbo_bolt, '3f', 'in_pos')])

# --- 7. WORLD INITIALIZATION ---
f_type = np.random.choice([0, 1, 2], size=NUM_FACES, p=[0.2, 0.4, 0.4]).astype(np.uint8)
for _ in range(12):
    ax = np.random.uniform(-1, 1, 3);
    ax /= np.linalg.norm(ax)
    f_type[np.abs(np.dot(FACE_CENTERS, ax)) < 0.038] = 1 if np.random.rand() > 0.5 else 2

f_burning, f_timer, f_ash = np.zeros(NUM_FACES, dtype=np.uint8), np.zeros(NUM_FACES, dtype=np.float32), np.zeros(
    NUM_FACES, dtype=np.float32)
active_bolts = []


# --- 8. SIMULATION ---
def update_sim():
    global f_type, f_burning, f_timer, f_ash, active_bolts, emeralds, station_indices, game_msg, game_msg_col, last_mining_tick, highscore, current_rate, station_deployed
    global last_stat_update, ui_stats

    now = time.time()

    if now - last_stat_update > 0.2:
        last_stat_update = now
        ui_stats["terra"] = (np.sum(f_type == 1) / NUM_FACES) * 100.0
        ui_stats["exo"] = (np.sum(f_type == 2) / NUM_FACES) * 100.0
        ui_stats["fire_terra"] = np.sum(f_burning == 1)
        ui_stats["fire_exo"] = np.sum(f_burning == 2)

    if station_deployed and len(station_indices) > 0:
        fire_types = f_burning[station_indices]
        if np.any(fire_types == 1):
            emeralds /= 2;
            station_deployed = False;
            station_indices = [];
            game_msg, game_msg_col = "ALERT: STATION LOST.", C_ALERT

        if station_deployed and (now - last_mining_tick >= MINING_TICK):
            last_mining_tick = now
            types = f_type[station_indices];
            tick_gain = np.sum(types == 1) - np.sum(types == 2)
            emeralds += tick_gain;
            current_rate = tick_gain * 10.0;
            highscore = max(highscore, emeralds)

            if now < next_deploy_time:
                game_msg, game_msg_col = f"MINING... (LOCKED: {next_deploy_time - now:.1f}s)", C_HOSE
            else:
                game_msg, game_msg_col = "MINING... (RELOCATION READY)", C_EMERALD

    is_intro = (now - START_TIME) < INTRO_DURATION
    ign_chance = BASE_IGNITION_CHANCE * (3.0 if is_intro else 1.0)
    for _ in range(max(1, int(SIM_SPEED_MODIFIER))):
        for i in [1, 2]:
            bm = (f_burning == i)
            if not np.any(bm): continue
            nd = NB_DIR[bm].flatten();
            vd = nd[(f_type[nd] == i) & (f_burning[nd] == 0)]
            if len(vd) > 0: f_burning[vd[np.random.rand(len(vd)) < CHANCE_SPREAD_DIRECT]] = i
            nx = NB_DIAG[bm].flatten();
            vx = nx[(f_type[nx] == i) & (f_burning[nx] == 0)]
            if len(vx) > 0: f_burning[vx[np.random.rand(len(vx)) < CHANCE_SPREAD_DIAG]] = i
        if np.random.rand() < ign_chance:
            idx = np.random.randint(NUM_FACES);
            bt = f_type[idx] if f_type[idx] > 0 else np.random.randint(1, 3)
            f_burning[NB_BOLT[idx]] = bt
            active_bolts.append({'pos': FACE_CENTERS[idx], 'col': (1.0, 0.5, 0.1) if bt == 1 else (0.7, 0.1, 1.4),
                                 'expiry': now + 0.15})
        bm_any = f_burning > 0;
        f_timer[bm_any] += 1
        ext = (f_timer >= BURN_DURATION) & bm_any;
        f_type[ext], f_burning[ext], f_timer[ext] = 0, 0, 0
        f_ash[ext] = BASE_ASH_DURATION;
        f_ash[f_ash > 0] -= (10.0 if is_intro else 1.0)
        regrow = (f_type == 0) & (f_ash <= 0) & (np.random.rand(NUM_FACES) < BASE_REGROW_CHANCE)
        if np.any(regrow): f_type[regrow] = np.random.randint(1, 3, size=np.sum(regrow))
    active_bolts = [b for b in active_bolts if b['expiry'] > now]


def update_buffers(hover_idx):
    s = np.zeros(NUM_FACES, dtype='f4')
    if hover_idx != -1: s[hover_idx] = 2.0
    vbo_type.write(np.repeat(f_type, 3).astype('f4').tobytes())
    vbo_burn.write(np.repeat(f_burning, 3).astype('f4').tobytes())
    vbo_ash.write(np.repeat(f_ash, 3).astype('f4').tobytes())
    vbo_ftimer.write(np.repeat(f_timer, 3).astype('f4').tobytes())
    vbo_state.write(np.repeat(s, 3).astype('f4').tobytes())


# --- MATRIX-BASED RAYCASTING (100% ACCURATE) ---
def get_face_at_mouse_raycast(mx, my, view_flat, proj_flat, model_flat):
    # 1. Unproject Mouse Coordinates
    x = (2.0 * mx) / SW - 1.0
    y = 1.0 - (2.0 * my) / SH

    # Invert Combined Matrix (View * Proj)
    pm = proj_flat.reshape(4, 4)
    vm = view_flat.reshape(4, 4)
    inv_pv = np.linalg.inv(vm @ pm)

    # Calculate Ray in World Space
    def unproj(vx, vy, vz):
        v = np.array([vx, vy, vz, 1.0], dtype='f4')
        res = v @ inv_pv  # Row-major multiplication
        if res[3] == 0: return np.array([0, 0, 0], dtype='f4')
        return res[:3] / res[3]

    near = unproj(x, y, -1.0)
    far = unproj(x, y, 1.0)

    ray_origin = near
    ray_dir = far - near
    ray_dir /= np.linalg.norm(ray_dir)

    # 2. Transform Ray to Model Space (Inverse Model Matrix)
    inv_model = np.linalg.inv(model_flat.reshape(4, 4))

    # Transform Origin
    ro_4 = np.append(ray_origin, 1.0) @ inv_model
    ro = ro_4[:3] / ro_4[3]

    # Transform Direction (Vector w=0)
    rd_4 = np.append(ray_dir, 0.0) @ inv_model
    rd = rd_4[:3]
    rd /= np.linalg.norm(rd)  # Re-normalize after scale

    # 3. Intersect Unit Sphere
    # |ro + t*rd|^2 = 1
    a = np.dot(rd, rd)
    b = 2.0 * np.dot(ro, rd)
    c = np.dot(ro, ro) - 1.0

    delta = b * b - 4 * a * c
    if delta < 0: return -1

    # Smallest positive t is front face
    t1 = (-b - np.sqrt(delta)) / (2.0 * a)
    t2 = (-b + np.sqrt(delta)) / (2.0 * a)

    t = -1
    if t1 > 0:
        t = t1
    elif t2 > 0:
        t = t2

    if t < 0: return -1

    hit = ro + rd * t
    dist, idx = tree.query(hit)
    if dist > 0.1: return -1  # Safety margin
    return idx


# --- 9. UI ---
def draw_ui(surf, ox, dt, gpu, tps, sun_p, moon_p, zoom, rx, ry, planet_rot):
    surf.fill((0, 0, 0, 0))

    def txt(t, f, c, p):
        surf.blit(f.render(t, True, (0, 0, 0)), (p[0] + 2, p[1] + 2)); surf.blit(f.render(t, True, c), p)

    pr_x, pr_y = SW - 250, SH - 200
    pr_pts = [(pr_x, SH), (SW, SH), (SW, pr_y), (pr_x + 50, pr_y - 30), (pr_x, pr_y)]
    pygame.draw.polygon(surf, C_PROBE_METAL, pr_pts)
    pygame.draw.polygon(surf, (30, 40, 45), pr_pts, 5)
    pygame.draw.circle(surf, (20, 25, 30), (pr_x + 50, pr_y - 10), 15)

    if station_deployed and target_screen_pos:
        s_pos = (pr_x + 50, pr_y - 10);
        e_pos = target_screen_pos
        c_pos = (s_pos[0] - 100, e_pos[1] + 200)
        curve = []
        for t in np.linspace(0, 1, 30): curve.append(
            ((1 - t) ** 2 * s_pos[0] + 2 * (1 - t) * t * c_pos[0] + t ** 2 * e_pos[0],
             (1 - t) ** 2 * s_pos[1] + 2 * (1 - t) * t * c_pos[1] + t ** 2 * e_pos[1]))
        if len(curve) > 1: pygame.draw.lines(surf, C_HOSE, False, curve, 8); pygame.draw.circle(surf, C_HOSE,
                                                                                                (int(e_pos[0]),
                                                                                                 int(e_pos[1])), 8)

    hx, hy = (SW - 500) // 2, 20
    hpts = [(hx, hy), (hx + 500, hy), (hx + 480, hy + 60), (hx + 20, hy + 60)]
    pygame.draw.polygon(surf, C_MATCHA_BG, hpts);
    pygame.draw.polygon(surf, C_MATCHA_LINE, hpts, 3)
    txt("OPERATION: EMERALD HARVEST // SEA-01", UI_FONT_L, C_MATCHA_TEXT, (hx + 45, hy + 12))

    x1, y1 = SW - 360 + ox, 100
    pts1 = [(x1 + 60, y1), (SW, y1), (SW, y1 + 750), (x1 + 30, y1 + 750), (x1, y1 + 700), (x1, y1 + 50)]
    pygame.draw.polygon(surf, C_MATCHA_BG, pts1);
    pygame.draw.polygon(surf, C_MATCHA_LINE, pts1, 3)
    txt("BIOSPHERE ANALYTICS", UI_FONT, C_MATCHA_TEXT, (x1 + 40, y1 + 30))

    for i, (l, p, c, fc, fcol) in enumerate(
            [("TERRA-A", ui_stats["terra"], C_BAR_GREEN, ui_stats["fire_terra"], C_FIRE_ORANGE),
             ("EXO-B", ui_stats["exo"], C_BAR_BLUE, ui_stats["fire_exo"], C_FIRE_VIOLET)]):
        ty = y1 + 80 + i * 100
        txt(f"{l}: {p:.1f}%", UI_FONT_S, C_MATCHA_TEXT, (x1 + 40, ty))
        pygame.draw.rect(surf, (10, 20, 15), (x1 + 40, ty + 25, 260, 14));
        pygame.draw.rect(surf, c, (x1 + 40, ty + 25, int(p * 2.6), 14))
        txt(f"ACTIVE BLAZE: {fc}", UI_FONT_T, fcol, (x1 + 40, ty + 45))

    y_probe = y1 + 320
    pygame.draw.line(surf, C_MATCHA_LINE, (x1 + 40, y_probe), (SW - 40, y_probe), 1)
    txt("PROBE TELEMETRY", UI_FONT_S, C_MATCHA_LINE, (x1 + 40, y_probe + 20))
    txt(f"ALTITUDE:   {max(0.0, zoom - 1.0):.2f}u", UI_FONT_T, C_MATCHA_TEXT, (x1 + 40, y_probe + 45))
    surface_lon = np.degrees(ry - planet_rot) % 360
    surface_lat = max(-90, min(90, (int(np.degrees(rx)) + 90) % 180 - 90))
    txt(f"COORD:      {surface_lat:03d}°N / {int(surface_lon):03d}°E", UI_FONT_T, C_MATCHA_TEXT,
        (x1 + 40, y_probe + 65))

    y_env = y_probe + 100
    pygame.draw.line(surf, C_MATCHA_LINE, (x1 + 40, y_env), (SW - 40, y_env), 1)
    txt("PLANETARY STATUS", UI_FONT_S, C_MATCHA_LINE, (x1 + 40, y_env + 20))
    txt(f"TEMP INDEX: {280 + (ui_stats['fire_terra'] + ui_stats['fire_exo']) * 5:.0f}K", UI_FONT_T, C_MATCHA_TEXT,
        (x1 + 40, y_env + 45))
    txt(f"ROTATION:   {np.degrees(SELF_ROT_SPEED * 60):.2f}°/s", UI_FONT_T, C_MATCHA_TEXT, (x1 + 40, y_env + 65))

    y_sun = y_env + 100
    pygame.draw.line(surf, C_MATCHA_LINE, (x1 + 40, y_sun), (SW - 40, y_sun), 1)
    txt("STAR SYSTEM", UI_FONT_S, C_SUN_YELLOW, (x1 + 40, y_sun + 20))
    txt(f"SUN DIST:   {np.linalg.norm(sun_p):.2f} AU", UI_FONT_T, C_MATCHA_TEXT, (x1 + 40, y_sun + 45))
    txt(f"MOON DIST:  {np.linalg.norm(moon_p):.2f} km", UI_FONT_T, C_MATCHA_TEXT, (x1 + 40, y_sun + 65))
    ph = ((np.arctan2(moon_p[2], moon_p[0]) - np.arctan2(sun_p[2], sun_p[0])) % (2 * np.pi)) / (2 * np.pi)
    p_name = "NEW" if ph < 0.05 or ph > 0.95 else "WAXING" if ph < 0.45 else "FULL" if ph < 0.55 else "WANING"
    txt(f"MOON PHASE: {p_name} ({ph * 100:.0f}%)", UI_FONT_T, C_MATCHA_TEXT, (x1 + 40, y_sun + 85))

    w5, h5 = 320, 130;
    x5, y5 = SW - w5 - 20, SH - h5 - 20
    pts5 = [(x5, y5), (x5 + w5, y5), (x5 + w5, y5 + h5), (x5 + 30, y5 + h5), (x5, y5 + h5 - 30)]
    pygame.draw.polygon(surf, C_MATCHA_BG, pts5);
    pygame.draw.polygon(surf, C_MATCHA_LINE, pts5, 3)
    txt("MINING OPERATIONS", UI_FONT_S, C_EMERALD, (x5 + 20, y5 + 15))
    txt(f"CARGO: {int(emeralds)}", UI_FONT, C_EMERALD, (x5 + 20, y5 + 40))
    txt(f"YIELD: {int(current_rate)}/s", UI_FONT_T, C_MATCHA_TEXT, (x5 + 180, y5 + 40))
    txt(f"BEST:  {int(highscore)}", UI_FONT_T, C_MATCHA_TEXT, (x5 + 20, y5 + 65))
    txt(f"STATUS: {game_msg}", UI_FONT_T, game_msg_col, (x5 + 20, y5 + 85))
    if time.time() < next_deploy_time:
        rem = next_deploy_time - time.time();
        perc = 1.0 - max(0.0, rem / COOLDOWN_TIME)
        pygame.draw.rect(surf, (50, 20, 20), (x5 + 20, y5 + 105, 280, 6));
        pygame.draw.rect(surf, C_HOSE, (x5 + 20, y5 + 105, int(280 * perc), 6))

    w3, y3 = 360, 150
    pts3 = [(0, y3), (w3 - 60, y3), (w3, y3 + 50), (w3, y3 + 420), (w3 - 30, y3 + 470), (0, y3 + 470)]
    pygame.draw.polygon(surf, C_MATCHA_BG, pts3);
    pygame.draw.polygon(surf, C_MATCHA_LINE, pts3, 3)
    txt("SIM OVERRIDE", UI_FONT, C_MATCHA_TEXT, (20, y3 + 30))
    txt(f"[1/2] IGNITION: {BASE_IGNITION_CHANCE * 100:.1f}%", UI_FONT_T, C_MATCHA_TEXT, (20, y3 + 100))
    txt(f"[3/4] REGROW:   {BASE_REGROW_CHANCE * 10000:.1f}", UI_FONT_T, C_MATCHA_TEXT, (20, y3 + 140))
    txt(f"[5/6] SPEED:    {SIM_SPEED_MODIFIER:.1f}x", UI_FONT_T, C_MATCHA_TEXT, (20, y3 + 180))
    txt(f"[7/8] BURN DUR: {BURN_DURATION} ticks", UI_FONT_T, C_FIRE_ORANGE, (20, y3 + 220))
    txt(f"[9/0] CORNER S: {CHANCE_SPREAD_DIAG * 100:.2f}%", UI_FONT_T, C_FIRE_VIOLET, (20, y3 + 260))

    w4, h4 = 340, 150
    x4, y4 = 20, SH - h4 - 20
    pts4 = [(x4, y4), (x4 + w4, y4), (x4 + w4 - 30, y4 + h4), (x4, y4 + h4)]
    pygame.draw.polygon(surf, C_MATCHA_BG, pts4);
    pygame.draw.polygon(surf, C_MATCHA_LINE, pts4, 2)
    txt("SYSTEM DIAGNOSTICS", UI_FONT_S, C_MATCHA_LINE, (x4 + 20, y4 + 15))
    txt(f"AUTHOR: Tuxi | PY: {sys.version.split(' ')[0]}", UI_FONT_T, C_MATCHA_TEXT, (x4 + 20, y4 + 45))
    txt(f"GPU LOAD: {gpu:.2f}ms", UI_FONT_T, (200, 255, 150), (x4 + 20, y4 + 75))
    txt(f"SIM RATE: {tps:.1f} TPS", UI_FONT_T, (255, 200, 100), (x4 + 20, y4 + 105))


# --- 10. LOOP ---
star_data = (np.random.uniform(-1, 1, (NUM_STARS, 3)).astype('f4'))
vbo_stars = ctx.buffer((star_data / np.linalg.norm(star_data, axis=1, keepdims=True)).tobytes())
vao_stars = ctx.vertex_array(prog_stars, [(vbo_stars, '3f', 'in_pos')])
query = ctx.query(time=True)
proj = projection_matrix(45, SW / SH, 0.1, 150.0)
for p in [prog, prog_bolt, prog_stars, prog_atmo]: p['u_proj'].value = tuple(proj)

rot_x, rot_y, zoom, self_rot = 0.0, 0.0, 6.0, 0.0
vel_x, vel_y, dragging, ox = 0.0, 0.0, False, 450.0
clock, acc, prev = pygame.time.Clock(), 0.0, time.time()
tps_count, tps_time, tps_display = 0, time.time(), 60.0
mouse_pos, target_screen_pos = (0, 0), None

while True:
    now = time.time();
    dt = now - prev;
    prev = now;
    acc += dt;
    mouse_click = False
    for e in pygame.event.get():
        if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE): pygame.quit(); sys.exit()
        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_1: BASE_IGNITION_CHANCE = max(0.0, BASE_IGNITION_CHANCE - 0.005)
            if e.key == pygame.K_2: BASE_IGNITION_CHANCE = min(0.2, BASE_IGNITION_CHANCE + 0.005)
            if e.key == pygame.K_3: BASE_REGROW_CHANCE = max(0.0, BASE_REGROW_CHANCE - 0.0001)
            if e.key == pygame.K_4: BASE_REGROW_CHANCE = min(0.01, BASE_REGROW_CHANCE + 0.0001)
            if e.key == pygame.K_5: SIM_SPEED_MODIFIER = max(1.0, SIM_SPEED_MODIFIER - 0.5)
            if e.key == pygame.K_6: SIM_SPEED_MODIFIER = min(15.0, SIM_SPEED_MODIFIER + 0.5)
            if e.key == pygame.K_7: BURN_DURATION = max(10, BURN_DURATION - 5)
            if e.key == pygame.K_8: BURN_DURATION = min(300, BURN_DURATION + 5)
            if e.key == pygame.K_9: CHANCE_SPREAD_DIAG = max(0.0, CHANCE_SPREAD_DIAG - 0.001)
            if e.key == pygame.K_0: CHANCE_SPREAD_DIAG = min(0.1, CHANCE_SPREAD_DIAG + 0.001)
        if e.type == pygame.MOUSEBUTTONDOWN:
            if e.button == 1: dragging = True
            if e.button == 3: mouse_click = True
            if e.button == 4: zoom = max(MIN_ZOOM, zoom - 0.5)
            if e.button == 5: zoom = min(MAX_ZOOM, zoom + 0.5)
        if e.type == pygame.MOUSEBUTTONUP and e.button == 1: dragging = False
        if e.type == pygame.MOUSEMOTION: mouse_pos = e.pos;
        if dragging and e.type == pygame.MOUSEMOTION: vel_x, vel_y = e.rel[0] * 0.005, e.rel[1] * 0.005

    while acc >= TICK_TIME: update_sim(); tps_count += 1; acc -= TICK_TIME
    if now - tps_time >= 1.0: tps_display, tps_count, tps_time = tps_count / (now - tps_time), 0, now

    self_rot += SELF_ROT_SPEED;
    ox *= 0.94
    if station_deployed: rot_y += SELF_ROT_SPEED
    if not dragging: vel_x *= FRICTION; vel_y *= FRICTION
    rot_x += vel_x;
    rot_y += vel_y

    # 1. KONSISTENTE MATRIX BERECHNUNG
    view = create_view_matrix(rot_x, rot_y, zoom)
    mm = get_model_matrix((0, 0, 0), 1.0, self_rot)
    MVP = get_mvp_matrix(proj, view, mm)

    # 2. RAYCASTING (Matrix-Based & Strict Front-Face)
    hover_idx = get_face_at_mouse_raycast(mouse_pos[0], mouse_pos[1], view, proj, mm)

    if mouse_click and hover_idx != -1:
        if f_burning[hover_idx] == 1:
            pass
        elif now < next_deploy_time and station_deployed:
            pass
        else:
            next_deploy_time = now + COOLDOWN_TIME;
            station_deployed = True;
            station_center_local = FACE_CENTERS[hover_idx]
            station_indices = np.where(np.linalg.norm(FACE_CENTERS - station_center_local, axis=1) < STATION_RADIUS)[0]
            game_msg, game_msg_col = "DEPLOYED.", C_EMERALD

    target_screen_pos = project_point_robust(station_center_local, MVP) if station_deployed else None

    with query:
        ctx.clear(0.008, 0.015, 0.018)
        sun_p = np.array([np.sin(now * SUN_SPEED) * SUN_DIST, 4.0, np.cos(now * SUN_SPEED) * SUN_DIST], dtype='f4')
        moon_p = np.array([np.sin(now * MOON_ORBIT_SPEED) * MOON_DIST, 0.5, np.cos(now * MOON_ORBIT_SPEED) * MOON_DIST],
                          dtype='f4')

        ctx.disable(moderngl.DEPTH_TEST);
        prog_stars['u_view'].value = tuple(view.flatten());
        vao_stars.render(moderngl.POINTS)
        ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE);
        ctx.disable(moderngl.BLEND)

        update_buffers(hover_idx)
        prog['u_view'].value, prog['u_sun_pos'].value, prog['u_moon_pos'].value, prog['u_time'].value = tuple(
            view.flatten()), tuple(sun_p), tuple(moon_p), now

        # PLANET (Mode 0)
        prog['u_mode'].value = 0
        prog['u_station_active'].value = 1 if station_deployed else 0
        prog['u_station_center'].value, prog['u_station_radius'].value = tuple(station_center_local), STATION_RADIUS
        prog['u_model'].value = tuple(mm.flatten());
        vao.render(moderngl.TRIANGLES)

        # CELESTIALS (Mode 1 & 2)
        prog['u_station_active'].value = 0
        prog['u_mode'].value = 2;
        prog['u_model'].value = tuple(get_model_matrix(tuple(moon_p), MOON_SCALE));
        vao.render(moderngl.TRIANGLES)
        prog['u_mode'].value = 1;
        prog['u_model'].value = tuple(get_model_matrix(tuple(sun_p), SUN_SCALE));
        vao.render(moderngl.TRIANGLES)

        ctx.enable(moderngl.BLEND);
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        prog_atmo['u_view'].value, prog_atmo['u_model'].value = tuple(view.flatten()), tuple(mm.flatten());
        vao_atmo.render(moderngl.TRIANGLES)

        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
        if active_bolts:
            prog_bolt['u_view'].value = tuple(view.flatten());
            mr = mm.reshape(4, 4).T
            if np.random.rand() > 0.6: vbo_bolt.write(gen_bolt_long().tobytes())
            for b in active_bolts: prog_bolt['u_offset'].value, prog_bolt['u_color'].value = tuple(
                mr[:3, :3] @ b['pos']), b['col']; vao_bolt.render(moderngl.LINES)

    ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
    draw_ui(ui_surface, ox, dt, query.elapsed * 1e-6, tps_display, sun_p, moon_p, zoom, rot_x, rot_y, self_rot)
    ui_tex.write(pygame.image.tostring(ui_surface, 'RGBA', False));
    ui_tex.use(0);
    ui_vao.render(moderngl.TRIANGLE_STRIP)
    pygame.display.flip();
    clock.tick(60)