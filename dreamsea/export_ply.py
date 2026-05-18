import torch
import numpy as np
import argparse
import os
from plyfile import PlyData, PlyElement

def export_to_ply(checkpoint_path, output_path):
    """
    Converts a 3DGS .pt checkpoint into a standard .ply file for visualization.
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load state dict
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    
    # Extract positions
    xyz = state_dict.get('positions')
    if xyz is None:
        print("Error: Positions not found in the checkpoint.")
        print("Make sure you are using a checkpoint generated with the latest version of generate_3dgs.py.")
        return

    xyz = xyz.numpy()
    normals = np.zeros_like(xyz)
    
    # Appearance params
    f_dc = state_dict['features_dc'].detach().cpu().numpy()
    opacity = state_dict['opacity'].detach().cpu().numpy()
    scale = state_dict['scaling'].detach().cpu().numpy()
    rotation = state_dict['rotation'].detach().cpu().numpy()

    # Construct the attribute list for 3DGS .ply
    # x, y, z, nx, ny, nz, f_dc_0, f_dc_1, f_dc_2, opacity, scale_0, scale_1, scale_2, rot_0, rot_1, rot_2, rot_3
    
    dtype_full = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
                  ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
                  ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
                  ('opacity', 'f4'),
                  ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
                  ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')]
    
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate([xyz, normals, f_dc, opacity, scale, rotation], axis=1)
    elements[:] = list(map(tuple, attributes))
    
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(output_path)
    
    print(f"\nSuccess! Standard 3DGS exported to: {output_path}")
    print("You can now view this file in any 3DGS viewer (e.g., Polycam, SIBERIA, or web-based viewers).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export DreamSea .pt 3DGS to .ply")
    parser.add_argument("--input", type=str, required=True, help="Path to .pt model file.")
    parser.add_argument("--output", type=str, default="model.ply", help="Path to save .ply file.")
    
    args = parser.parse_args()
    export_to_ply(args.input, args.output)
