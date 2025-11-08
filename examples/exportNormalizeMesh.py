
import argparse
import os
import util
import trimesh

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='flexicubes optimization')
    parser.add_argument('-o', '--out_dir', type=str, default=None)
    parser.add_argument('-fn', '--file_name', type=str)    
    parser.add_argument('-rm', '--ref_mesh', type=str)    

    FLAGS = parser.parse_args()

    device = 'cuda'

    os.makedirs(FLAGS.out_dir, exist_ok=True)

    gt_mesh = util.load_mesh_jy(FLAGS.ref_mesh, device)
    gt_mesh.auto_normals() # compute face normals for visualization

    trimesh.Trimesh(vertices=gt_mesh.vertices.cpu().numpy(), faces=gt_mesh.faces.cpu().numpy()).export(FLAGS.out_dir + f"/{FLAGS.file_name}.obj")
    