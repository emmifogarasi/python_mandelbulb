import time
import numpy as np
import argparse
from src.mandelbulb_generator import compute_scalar_field

def run_mandelbulb_computation():
    GRID_RESOLUTION = 16
    POWER = 8
    MAX_ITERATIONS = 15
    BAILOUT_RADIUS = 2.0

    X_BOUNDS = (-1.5, 1.5)
    Y_BOUNDS = (-1.5, 1.5)
    Z_BOUNDS = (-1.5, 1.5)

    print(f"Starting Mandelbulb computation with parameters:")
    print(f"    Grid Resolution: {GRID_RESOLUTION}^3")
    print(f"    Power: {POWER}")
    print(f"    Max Iterations: {MAX_ITERATIONS}")
    print(f"    Bailout Radius: {BAILOUT_RADIUS}")
    
    start_time = time.time()

    scalar_data = compute_scalar_field(
        grid_res=GRID_RESOLUTION,
        power=POWER,
        max_iterations=MAX_ITERATIONS,
        bailout_radius=BAILOUT_RADIUS,
        x_bounds=X_BOUNDS,
        y_bounds=Y_BOUNDS,
        z_bounds=Z_BOUNDS
    )

    end_time = time.time()
    computation_time = end_time - start_time
    print(f"Computation finished in {computation_time:.2f} seconds.")

    if scalar_data is not None:
        print(f"Scalar data generated with shape: {scalar_data.shape}")
        print(f"Min escape value in data: {scalar_data.min()}")
        print(f"Max escape value in data: {scalar_data.max()}")
    else:
        print("Error: Scalar data computation failed.")

if __name__ == "__main__":
    run_mandelbulb_computation()