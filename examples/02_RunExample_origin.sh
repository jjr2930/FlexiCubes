#!/bin/sh

python optimize.py --ref_mesh "/workspace/FlexiCubes/examples/data/inputmodels/africa_man.obj" \
    --out_dir out/africa_man_origin \
    --voxel_grid_res 128 128 128 \
    --iter=100 \
    -lr=0.01 \
    -r 2048 2048 \
    -fv 3246 9915 \
    -mc 0 \
    --sdf_loss False \ 
    --focus_start_iteration 50 \
    --focus_capture_data "/workspace/FlexiCubes/examples/data/inputmodels/africa_man/positionItem.json" 


# python optimize.py --ref_mesh "/workspace/FlexiCubes/examples/data/inputmodels/africa_man.obj" \
#     --out_dir out/africa_man_origin_fc \
#     --voxel_grid_res 200 200 200 \
#     --iter=400 \
#     -lr=0.01 \
#     -r 2048 2048 \
#     -fc 250 \
#     -fv 3246 9915 \
#     -mc 2 \
#     --sdf_loss False


# python optimize.py --ref_mesh "/workspace/FlexiCubes/examples/data/inputmodels/africa_man.obj" \
#     --out_dir out/africa_man_origin_200 \
#     --voxel_grid_res 200 200 200 \
#     --iter=300 \
#     -lr=0.01 \
#     -r 2048 2048 \
#     -fc 500 \
#     -fv 3246 9915 \
#     -mc 0 \
#     --sdf_loss False \
#     -b 16

# python optimize.py --ref_mesh "/workspace/FlexiCubes/examples/data/inputmodels/africa_man.obj" \
#     --out_dir out/africa_man_origin_fc_200 \
#     --voxel_grid_res 200 200 200 \
#     --iter=300 \
#     -lr=0.01 \
#     -r 2048 2048 \
#     -fc 250 \
#     -fv 3246 9915 \
#     -mc 4 \
#     --sdf_loss False \
#     -b 16
    

# #peasant
# python optimize.py --ref_mesh "/workspace/FlexiCubes/examples/data/inputmodels/Peasant Girl.obj" \
#     --out_dir out/Peasant_origin_200 \
#     --voxel_grid_res 200 200 200 \
#     --iter=400 \
#     -lr=0.005 \
#     -r 8192 8192 \
#     -fc 500 \
#     -fv 2114 1812 \
#     --sdf_loss False \
#     -mc 0 \
#     -b 2

# python optimize.py --ref_mesh "/workspace/FlexiCubes/examples/data/inputmodels/Peasant Girl.obj" \
#     --out_dir out/Peasant_origin_fc_200 \
#     --voxel_grid_res 200 200 200 \
#     --iter=400 \
#     -lr=0.005 \
#     -r 8192 8192 \
#     -fc 300 \
#     -fv 2114 1812 \
#     --sdf_loss False \
#     -mc 2 \
#     -b 4


# #warrok
# python optimize.py --ref_mesh "/workspace/FlexiCubes/examples/data/inputmodels/Warrok W Kurniawan.obj" \
#     --out_dir out/Warrok_origin \
#     --voxel_grid_res 200 200 200 \
#     --iter=400 \
#     -lr=0.005 \
#     -r 2048 2048 \
#     -fc 500 \
#     -fv 5511 3838 \
#     --sdf_loss False

# python optimize.py --ref_mesh "/workspace/FlexiCubes/examples/data/inputmodels/Warrok W Kurniawan.obj" \
#     --out_dir out/Warrok_origin_fc \
#     --voxel_grid_res 200 200 200 \
#     --iter=400 \
#     -lr=0.005 \
#     -r 2048 2048 \
#     -fc 300 \
#     -fv 5511 3838 \
#     --sdf_loss False
