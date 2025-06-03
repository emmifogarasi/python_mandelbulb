import numpy as np
from numba import jit
from tqdm import tqdm

X_BOUNDS_DEFAULT = (-1.5, 1.5)
Y_BOUNDS_DEFAULT = (-1.5, 1.5)
Z_BOUNDS_DEFAULT = (-1.5, 1.5)

@jit(nopython=True)     # decorator tells Numba to compile this code into fast machine mode. Ensures it runs entirely without Python interpreter overhead
def mandelbulb_iterate_point(cx, cy, cz, power, max_iterations, bailout_radius_sq):
    zx, zy, zz = cx, cy, cz     #Start iteration at the point itself

    for i in range(max_iterations):
        # Check for escape
        r_sq = zx*zx + zy*zy + zz*zz
        if r_sq > bailout_radius_sq:
            return i    # Escaped, return iteration count
        
        r = np.sqrt(r_sq)   # square root of r_sq

        # Convert to spherical coordinates
        # theta is polar angle, phi is azimuthal angle
        # guard against zx=0 and sqrt_z_zy_sq=0 for atan2 robustness if robustness needed
        sqrt_zx_zy_sq = np.sqrt(zx*zx + zy* zy)
        theta = np.arctan2(sqrt_zx_zy_sq, zz)
        phi = np.arctan2(zy, zx)

        # Apply power transformation in spherical coords
        r_pow_n = r**power
        new_theta = theta * power
        new_phi = phi * power 

        # Convert back to cartesian coords
        zx_temp = r_pow_n * np.sin(new_theta) * np.cos(new_phi)
        zy_temp = r_pow_n * np.sin(new_theta) * np.sin(new_phi)
        zz_temp =r_pow_n * np.cos(new_theta)

        # Add original point coordinate (like c in mandelbrot)
        zx = zx_temp + cx
        zy = zy_temp + cy
        zz = zz_temp + cz 

    return max_iterations   # Point is considered in the set

# Definining a 3D grid of coordinates, looping through each voxel and call mandelbulb_iterate_point for each voxel
# Store result in an array

def compute_scalar_field(grid_res, power, max_iterations, bailout_radius,
                         x_bounds=X_BOUNDS_DEFAULT,
                         y_bounds=Y_BOUNDS_DEFAULT,
                         z_bounds=Z_BOUNDS_DEFAULT):
    
    print(f"Initialising {grid_res}x{grid_res}x{grid_res} on scalar field...")
    scalar_field = np.zeros((grid_res, grid_res, grid_res), dtype=np.int32)
    bailout_radius_sq = bailout_radius**2

    # Create coord vectors for each point in grid
    # Voxels
    x_coords = np.linspace(x_bounds[0], x_bounds[1], grid_res)
    y_coords = np.linspace(y_bounds[0], y_bounds[1], grid_res)
    z_coords = np.linspace(z_bounds[0], z_bounds[1], grid_res)

    print("Computing scalar field (this may take a while)...")
    # Loop through each voxel by its index (i, j, k)
    # The outer loop uses tqdm for a progress bar
    for i in tqdm(range(grid_res), desc="X slices"):
        for j in range(grid_res):
            for k in range(grid_res):
                # Convert voxel index (i, j, k) to actual 3D coords (cx, cy, cz)
                cx = x_coords[i]
                cy = y_coords[j]
                cz = z_coords[k]

                # Calling iteration function
                scalar_field[i, j, k] = mandelbulb_iterate_point(
                    cx, cy, cz, power, max_iterations, bailout_radius_sq
                )

    print("Scalar field computation complete.")
    return scalar_field
