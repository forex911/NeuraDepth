import numpy as np
import cv2

class ExportService:
    @staticmethod
    def generate_ply(depth_map: np.ndarray, color_image: np.ndarray) -> bytes:
        """Generates an ASCII PLY point cloud from a depth map."""
        h, w = depth_map.shape
        
        # Subsample to avoid 500MB point clouds (cap at around 512x512 equivalent density)
        scale = min(1.0, 512.0 / max(w, h))
        if scale < 1.0:
            depth_map = cv2.resize(depth_map, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            color_image = cv2.resize(color_image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            
        h, w = depth_map.shape
        yy, xx = np.mgrid[0:h, 0:w]
        
        z = depth_map.flatten()
        x = xx.flatten()
        y = yy.flatten()
        
        colors = color_image.reshape(-1, 3)
        
        # Filter out extreme noise
        mask = z > 0.01
        
        x = x[mask]
        y = y[mask]
        # Invert Y for standard 3D coordinate systems, scale Z to be proportional
        z = z[mask] * (max(w, h) * 0.3) 
        colors = colors[mask]
        
        num_points = len(x)
        header = f"""ply
format ascii 1.0
element vertex {num_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
        lines = [header]
        for i in range(num_points):
            b, g, r = colors[i]
            lines.append(f"{x[i]} {-y[i]} {z[i]:.3f} {r} {g} {b}\n")
            
        return "".join(lines).encode('utf-8')

    @staticmethod
    def generate_obj(depth_map: np.ndarray) -> bytes:
        """Generates a 3D mesh OBJ file from a depth map."""
        # Cap resolution to prevent multi-GB OBJ files
        scale = min(1.0, 256.0 / max(depth_map.shape))
        if scale < 1.0:
            small_depth = cv2.resize(depth_map, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            small_depth = depth_map
            
        h, w = small_depth.shape
        lines = ["o DepthForgeMesh\n"]
        
        z_scale = max(w, h) * 0.3
        for y in range(h):
            for x in range(w):
                z = small_depth[y, x] * z_scale
                lines.append(f"v {x} {-y} {z:.3f}\n")
                
        for y in range(h - 1):
            for x in range(w - 1):
                idx1 = y * w + x + 1
                idx2 = idx1 + 1
                idx3 = (y + 1) * w + x + 1
                idx4 = idx3 + 1
                
                lines.append(f"f {idx1} {idx2} {idx3}\n")
                lines.append(f"f {idx2} {idx4} {idx3}\n")
                
        return "".join(lines).encode('utf-8')
