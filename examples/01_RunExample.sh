#!/bin/sh

# #y only
# python optimizeWithImage.py \
#     --out_dir out/africa_man_y_only \
#     --voxel_grid_res 200 200 200 \
#     --iter=400 \
#     --sdf_regularizer 0.02 \
#     --ref_mesh "/workspace/FlexiCubes/examples/data/inputmodels/africa_man.obj" \
#     -lr=0.01 \
#     -r 2048 2048 \
#     -fc 0 \
#     -df "/workspace/FlexiCubes/examples/dataset/africa_man/dataset.json" \
#     -wd "/workspace/FlexiCubes/examples/dataset/africa_man" \
#     -op "africa_man" \
#     -b 12 


##post
# python optimizeWithImage.py \
#     --out_dir out/africa_man_y_with_focus_post \
#     --voxel_grid_res 200 200 200 \
#     --iter=100 \
#     --sdf_regularizer 0.02 \
#     --ref_mesh "/workspace/FlexiCubes/examples/data/inputmodels/africa_man.obj" \
#     -lr=0.01 \
#     -r 2048 2048 \
#     -fuf True True \
#     -fc 6 \
#     -df "/workspace/FlexiCubes/examples/dataset/africa_man/dataset.json" \
#     -wd "/workspace/FlexiCubes/examples/dataset/africa_man" \
#     -op "africa_man" \
#     -fm "post" \
#     -fpi 50 \
#     -b 12 


## middle
python optimizeWithImage.py \
    --out_dir out/africa_man_y_with_focus_middle \
    --voxel_grid_res 200 200 200 \
    --iter=100 \
    --sdf_regularizer 0.02 \
    --ref_mesh "/workspace/FlexiCubes/examples/data/inputmodels/africa_man.obj" \
    -lr=0.01 \
    -r 2048 2048 \
    -fuf True True \
    -fc 2 \
    -df "/workspace/FlexiCubes/examples/dataset/africa_man/dataset.json" \
    -wd "/workspace/FlexiCubes/examples/dataset/africa_man" \
    -op "africa_man" \
    -fm "middle" \
    -b 12 
