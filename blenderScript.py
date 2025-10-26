# Blender Python: Nerfstudio dataset exporter (spherical, Z-up, elevation×azimuth)
# + Sun Light direction matches camera forward
# ------------------------------------------------------------

import bpy, json, math, os, sys, traceback
from math import radians
from mathutils import Matrix, Vector

# ==== 사용자 설정 ====
OUT_DIR = "/Users/jujeong-yeol/Documents/blenderOutput/b1"
IMG_DIR = os.path.join(OUT_DIR, "images")
MASK_DIR = os.path.join(OUT_DIR, "masks")
NORMAL_DIR = os.path.join(OUT_DIR, "normals")
DEPTH_DIR = os.path.join(OUT_DIR, "depth")

W, H = 1024, 1024
FOCAL_MM = 35.0
SENSOR_MM = 36.0

RADIUS = 2.2
CAM_Z_OFFSET = 1.0
TARGET_Z = 1.0

ELEV_STEPS = 20
AZIM_STEPS = 20
MAX_ELEV_DEG = 89.0

BG_TRANSPARENT = False
COLOR_MANAGEMENT_VIEW = "Filmic"
GAMMA = 1.0
CYCLES_SAMPLES = 64

# ==== 준비 ====
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)
os.makedirs(NORMAL_DIR, exist_ok=True)
os.makedirs(DEPTH_DIR, exist_ok=True)
scene = bpy.context.scene

# --- 엔진 자동 선택(3.x/4.x 호환) ---
engine_keys = set(bpy.types.RenderSettings.bl_rna.properties['engine'].enum_items.keys())
preferred_engines = ['CYCLES', 'BLENDER_EEVEE', 'BLENDER_EEVEE_NEXT']
for engine in preferred_engines:
    if engine in engine_keys:
        scene.render.engine = engine
        break
else:
    raise RuntimeError(f"지원 렌더 엔진을 찾을 수 없습니다: {engine_keys}")

if scene.render.engine != 'CYCLES':
    print("⚠️ Cycles 엔진을 찾지 못해", scene.render.engine, "로 렌더링합니다. Normal/Depth 패스가 비활성일 수 있습니다.")

# --- 해상도/색관리/포맷 ---
scene.render.resolution_x = W
scene.render.resolution_y = H
scene.render.resolution_percentage = 100
scene.view_settings.view_transform = COLOR_MANAGEMENT_VIEW
scene.view_settings.gamma = GAMMA
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_depth = '8'
scene.render.film_transparent = BG_TRANSPARENT
scene.render.image_settings.color_mode = 'RGBA' if BG_TRANSPARENT else 'RGB'
if scene.render.engine == 'CYCLES':
    scene.cycles.samples = CYCLES_SAMPLES
scene.render.use_save_buffers = True

# --- View Layer Pass 설정 ---
active_view_layer = bpy.context.view_layer
active_view_layer.use_pass_normal = True
active_view_layer.use_pass_z = True
ACTIVE_VIEW_LAYER_NAME = active_view_layer.name

# --- 카메라 준비 ---
cam_obj = next((o for o in bpy.data.objects if o.type == 'CAMERA'), None)
if cam_obj is None:
    cam_data = bpy.data.cameras.new("NerfCam")
    cam_obj = bpy.data.objects.new("NerfCam", cam_data)
    bpy.context.collection.objects.link(cam_obj)
scene.camera = cam_obj
cam = cam_obj.data
cam.lens = FOCAL_MM
cam.sensor_width = SENSOR_MM

# --- 타깃 ---
target = Vector((0.0, 0.0, TARGET_Z))

# --- 라이트 준비 ---
def ensure_sun_light():
    sun = next((o for o in bpy.data.objects if o.type == 'LIGHT' and o.data.type == 'SUN'), None)
    if sun is None:
        light_data = bpy.data.lights.new(name="SunLight", type='SUN')
        light_data.energy = 5.0
        light_data.angle = radians(5)
        sun = bpy.data.objects.new(name="SunLight", object_data=light_data)
        bpy.context.collection.objects.link(sun)
        print("☀️ Sun Light created.")
    return sun

sun = ensure_sun_light()

# --- 카메라 방향 기반 Sun 회전 ---
def align_sun_to_camera(cam_obj, sun_obj):
    """
    카메라의 forward(-Z) 방향을 Sun의 조명 방향과 일치시킨다.
    즉, 카메라가 바라보는 방향으로 태양빛이 들어오게 함.
    """
    # 카메라의 forward(-Z) 방향
    cam_forward = -cam_obj.matrix_world.to_3x3().col[2].normalized()
    up = Vector((0, 0, 1))
    right = cam_forward.cross(up).normalized()
    up_corrected = right.cross(cam_forward).normalized()

    rot = Matrix((
        (right.x, up_corrected.x, -cam_forward.x),
        (right.y, up_corrected.y, -cam_forward.y),
        (right.z, up_corrected.z, -cam_forward.z)
    ))
    sun_obj.matrix_world = rot.to_4x4()
    # sun_obj.location = cam_obj.location  # (선택) 카메라 위치로 이동시키고 싶다면 활성화

# --- look_at 함수 ---
def look_at(cam_obj, target):
    direction = (target - cam_obj.location).normalized()
    z = -direction
    up = Vector((0.0, 0.0, 1.0))
    x = up.cross(z).normalized()
    y = z.cross(x).normalized()

    rot = Matrix((
        (x.x, y.x, z.x, 0.0),
        (x.y, y.y, z.y, 0.0),
        (x.z, y.z, z.z, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    ))
    cam_obj.matrix_world = Matrix.Translation(cam_obj.location) @ rot

# --- 렌더 패스 추출 + 저장 유틸리티 ---
def _ensure_render_layer(render_image, view_layer_name):
    for attr in ("view_layers", "render_layers", "layers"):
        container = getattr(render_image, attr, None)
        if container is None:
            continue
        layer = None
        if hasattr(container, "get"):
            layer = container.get(view_layer_name)
        else:
            for candidate in container:
                if getattr(candidate, "name", None) == view_layer_name:
                    layer = candidate
                    break
        if layer is not None:
            return layer
    return None

def _extract_render_pass(render_layer, *pass_names):
    passes = getattr(render_layer, "passes", [])
    lookup = {p.name: p for p in passes}
    for name in pass_names:
        if name in lookup:
            return lookup[name]
    return None

def _save_image(filepath, pixels, width, height, *, float_buffer=False, file_format='PNG', colorspace='Linear'):
    image_name = os.path.basename(filepath)
    image = bpy.data.images.new(image_name, width=width, height=height, alpha=True, float_buffer=float_buffer)
    image.pixels = pixels
    image.filepath_raw = filepath
    image.file_format = file_format
    try:
        image.colorspace_settings.name = colorspace
    except Exception:
        pass
    image.save()
    bpy.data.images.remove(image)

def _convert_normal_pixels(pass_rect):
    pixels = []
    for i in range(0, len(pass_rect), 4):
        nx, ny, nz = pass_rect[i], pass_rect[i + 1], pass_rect[i + 2]
        # Blender normal은 [-1, 1] -> [0, 1]로 매핑
        pixels.extend([
            0.5 * nx + 0.5,
            0.5 * ny + 0.5,
            0.5 * nz + 0.5,
            1.0,
        ])
    return pixels

def _convert_depth_pixels(pass_rect):
    raw_pixels = []
    mask_pixels = []
    valid_depths = [d for d in pass_rect[::4] if math.isfinite(d) and d > 0.0]
    min_depth = min(valid_depths) if valid_depths else 0.0
    max_depth = max(valid_depths) if valid_depths else 0.0

    for i in range(0, len(pass_rect), 4):
        depth_value = pass_rect[i]
        if math.isfinite(depth_value) and depth_value > 0.0:
            raw_val = depth_value
            mask_val = 1.0
        else:
            raw_val = 0.0
            mask_val = 0.0

        raw_pixels.extend([raw_val, raw_val, raw_val, 1.0])
        mask_pixels.extend([mask_val, mask_val, mask_val, 1.0])

    return raw_pixels, mask_pixels, min_depth, max_depth

def save_auxiliary_passes(frame_idx, view_layer_name):
    render_result = bpy.data.images.get("Render Result")
    if render_result is None or not render_result.has_data:
        print("  ⚠️ Render result unavailable for auxiliary passes.")
        return

    render_layer = _ensure_render_layer(render_result, view_layer_name)
    if render_layer is None:
        print(f"  ⚠️ View layer '{view_layer_name}' not found in render result.")
        return

    width, height = render_result.size
    normal_pass = _extract_render_pass(render_layer, "Normal", "NORMAL")
    depth_pass = _extract_render_pass(render_layer, "Depth", "Z")

    if normal_pass is not None:
        normal_pixels = _convert_normal_pixels(list(normal_pass.rect))
        normal_path = os.path.join(NORMAL_DIR, f"normal_{frame_idx:04d}.png")
        _save_image(normal_path, normal_pixels, width, height, float_buffer=False, file_format='PNG', colorspace='Non-Color')
        print(f"  └─ Normal saved: {os.path.basename(normal_path)}")
    else:
        print("  ⚠️ Normal pass not available; skipping normal output.")

    if depth_pass is not None:
        raw_pixels, mask_pixels, min_d, max_d = _convert_depth_pixels(list(depth_pass.rect))
        depth_exr_path = os.path.join(DEPTH_DIR, f"depth_{frame_idx:04d}.exr")
        _save_image(depth_exr_path, raw_pixels, width, height, float_buffer=True, file_format='OPEN_EXR', colorspace='Linear')
        print(f"  └─ Depth saved: {os.path.basename(depth_exr_path)} [range {min_d:.4f} ~ {max_d:.4f}]")

        mask_path = os.path.join(MASK_DIR, f"mask_{frame_idx:04d}.png")
        _save_image(mask_path, mask_pixels, width, height, float_buffer=False, file_format='PNG', colorspace='Non-Color')
        print(f"  └─ Depth mask saved: {os.path.basename(mask_path)}")
    else:
        print("  ⚠️ Depth/Z pass not available; skipping depth & mask outputs.")

# --- 카메라 내적 파라미터 ---
fl_x = (cam.lens / cam.sensor_width) * W
fl_y = fl_x
cx = W / 2.0
cy = H / 2.0

data = {"fl_x": fl_x, "fl_y": fl_y, "cx": cx, "cy": cy, "w": W, "h": H,
        "camera_model": "OPENCV", "frames": []}

# --- 이중 반복 (Elev × Azim) ---
ELEV_STEPS_SAFE = max(2, int(ELEV_STEPS))
AZIM_STEPS_SAFE = max(1, int(AZIM_STEPS))
frame_idx = 0
total_frames = ELEV_STEPS_SAFE * AZIM_STEPS_SAFE
print(f"[Info] Total frames = {total_frames}")

try:
    for elev_i in range(ELEV_STEPS_SAFE):
        elev_deg = (MAX_ELEV_DEG * elev_i) / (ELEV_STEPS_SAFE - 1)
        elev = radians(elev_deg)

        for azim_i in range(AZIM_STEPS_SAFE):
            azim = 2.0 * math.pi * (azim_i / AZIM_STEPS_SAFE)

            # 구면좌표계
            x = RADIUS * math.cos(elev) * math.cos(azim)
            y = RADIUS * math.cos(elev) * math.sin(azim)
            z = RADIUS * math.sin(elev)

            # 카메라 위치
            cam_obj.location = Vector((x, y, z + CAM_Z_OFFSET))
            look_at(cam_obj, target)

            # 🔹 카메라 방향에 맞게 라이트 방향 정렬
            align_sun_to_camera(cam_obj, sun)

            # 렌더 (컬러 이미지)
            img_name = f"{frame_idx:04d}.png"
            img_path = os.path.join(IMG_DIR, img_name)
            scene.render.filepath = img_path
            print(f"[Render] {frame_idx+1}/{total_frames}  elev={elev_deg:5.1f}°, azim={math.degrees(azim):6.1f}° → {img_name}")
            bpy.ops.render.render(write_still=True)

            # 노멀/깊이/마스크 패스 저장
            save_auxiliary_passes(frame_idx, ACTIVE_VIEW_LAYER_NAME)

            # c2w 저장
            c2w = cam_obj.matrix_world.copy()
            mat = [[c2w[r][c] for c in range(4)] for r in range(4)]
            data["frames"].append({"file_path": f"images/{img_name}", "transform_matrix": mat})
            frame_idx += 1

    with open(os.path.join(OUT_DIR, "transforms.json"), "w") as f:
        json.dump(data, f, indent=2)
    print(f"[OK] Export done: {OUT_DIR} (frames={len(data['frames'])})")

except Exception as e:
    print("[ERROR] 예외 발생:")
    traceback.print_exc(file=sys.stdout)
    raise
