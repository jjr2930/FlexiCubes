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
scene = bpy.context.scene

# --- 엔진 자동 선택(3.x/4.x 호환) ---
engine_keys = set(bpy.types.RenderSettings.bl_rna.properties['engine'].enum_items.keys())
if 'BLENDER_EEVEE' in engine_keys:
    scene.render.engine = 'BLENDER_EEVEE'
elif 'BLENDER_EEVEE_NEXT' in engine_keys:
    scene.render.engine = 'BLENDER_EEVEE_NEXT'
elif 'CYCLES' in engine_keys:
    scene.render.engine = 'CYCLES'
else:
    raise RuntimeError(f"지원 렌더 엔진을 찾을 수 없습니다: {engine_keys}")

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
    import mathutils
    # 카메라의 forward(-Z) 방향
    cam_forward = -cam_obj.matrix_world.to_3x3().col[2].normalized()
    up = Vector((0, 0, 1))
    right = cam_forward.cross(up).normalized()
    up_corrected = right.cross(cam_forward).normalized()

    rot = mathutils.Matrix((
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

# --- 마스크 렌더링 함수들 (render.py의 안티앨리어싱 마스크 개념 구현) ---
def setup_mask_materials():
    """모든 오브젝트에 마스크용 흰색 재질 적용"""
    # 마스크용 재질 생성 또는 가져오기
    mask_mat_name = "MaskMaterial"
    mask_mat = bpy.data.materials.get(mask_mat_name)
    
    if mask_mat is None:
        mask_mat = bpy.data.materials.new(name=mask_mat_name)
        mask_mat.use_nodes = True
        nodes = mask_mat.node_tree.nodes
        nodes.clear()
        
        # 출력 노드 추가
        output_node = nodes.new(type='ShaderNodeOutputMaterial')
        
        # Emission 셰이더 추가 (흰색 마스크용)
        emission_node = nodes.new(type='ShaderNodeEmission')
        emission_node.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)  # 흰색
        emission_node.inputs['Strength'].default_value = 1.0
        
        # 노드 연결
        mask_mat.node_tree.links.new(emission_node.outputs['Emission'], output_node.inputs['Surface'])
        
    return mask_mat

def apply_mask_materials(mask_material):
    """모든 메시 오브젝트에 마스크 재질 적용"""
    original_materials = {}
    
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and obj.visible_get():
            original_materials[obj.name] = []
            
            # 기존 재질 백업
            for slot in obj.material_slots:
                original_materials[obj.name].append(slot.material)
            
            # 모든 재질 슬롯을 마스크 재질로 교체
            for i, slot in enumerate(obj.material_slots):
                slot.material = mask_material
                
            # 재질 슬롯이 없는 경우 추가
            if len(obj.material_slots) == 0:
                obj.data.materials.append(mask_material)
                
    return original_materials

def restore_original_materials(original_materials):
    """원래 재질로 복원"""
    for obj_name, materials in original_materials.items():
        obj = bpy.data.objects.get(obj_name)
        if obj and obj.type == 'MESH':
            for i, material in enumerate(materials):
                if i < len(obj.material_slots):
                    obj.material_slots[i].material = material

def render_mask_image(frame_idx):
    """마스크 이미지 렌더링 - render.py의 (rast[..., -1:] > 0).float() 개념을 Blender로 구현"""
    # 마스크용 재질 설정
    mask_material = setup_mask_materials()
    
    # 기존 재질 백업 및 마스크 재질 적용
    original_materials = apply_mask_materials(mask_material)
    
    # 마스크 렌더링 설정
    original_bg = scene.render.film_transparent
    original_engine = scene.render.engine
    original_color_mode = scene.render.image_settings.color_mode
    
    scene.render.film_transparent = False  # 마스크는 불투명 배경
    scene.render.engine = 'BLENDER_WORKBENCH'  # 빠른 렌더링을 위해
    scene.render.image_settings.color_mode = 'RGB'  # RGB로 설정
    
    # 월드 배경을 검은색으로 설정
    original_bg_color = None
    if bpy.context.scene.world:
        world = bpy.context.scene.world
        if world.use_nodes:
            bg_node = world.node_tree.nodes.get('Background')
            if bg_node:
                original_bg_color = bg_node.inputs['Color'].default_value[:]
                bg_node.inputs['Color'].default_value = (0.0, 0.0, 0.0, 1.0)  # 검은색
    
    try:
        # 마스크 이미지 렌더링
        mask_name = f"mask_{frame_idx:03d}.png"
        mask_path = os.path.join(MASK_DIR, mask_name)
        scene.render.filepath = mask_path
        bpy.ops.render.render(write_still=True)
        print(f"  └─ Mask saved: {mask_name}")
        
    finally:
        # 설정 복원
        scene.render.film_transparent = original_bg
        scene.render.engine = original_engine
        scene.render.image_settings.color_mode = original_color_mode
        
        # 배경색 복원
        if original_bg_color is not None and bpy.context.scene.world and bpy.context.scene.world.use_nodes:
            bg_node = bpy.context.scene.world.node_tree.nodes.get('Background')
            if bg_node:
                bg_node.inputs['Color'].default_value = original_bg_color
        
        # 재질 복원
        restore_original_materials(original_materials)

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
            
            # 마스크 이미지 렌더링 (render.py의 안티앨리어싱 마스크와 유사한 기능)
            render_mask_image(frame_idx)

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
