def visualise_mandelbulb(
    scalar_field,
    bounds,
    isosurface_threshold,
    show_axes=True,
    show_grid=True
):
    import numpy as np
    import pyvista as pv

    print(f"Preparing isosurface for data of shape {scalar_field.shape}...")
    print(f"Using isosurface threshold: {isosurface_threshold}")

    # Build grid
    grid = pv.ImageData()
    grid.dimensions = np.array(scalar_field.shape)
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    grid.origin = (xmin, ymin, zmin)
    sx = (xmax - xmin) / (scalar_field.shape[0] - 1 if scalar_field.shape[0] > 1 else 1)
    sy = (ymax - ymin) / (scalar_field.shape[1] - 1 if scalar_field.shape[1] > 1 else 1)
    sz = (zmax - zmin) / (scalar_field.shape[2] - 1 if scalar_field.shape[2] > 1 else 1)
    grid.spacing = (sx, sy, sz)
    grid.point_data["escape_time"] = scalar_field.flatten(order="F")


    print("Extracting isosurface...")
    isosurface = grid.contour([isosurface_threshold], scalars="escape_time")
    if isosurface.n_points == 0:
        print(f"Warning: No surface generated for threshold {isosurface_threshold}")
        print(f"Data min: {scalar_field.min()}, Data max: {scalar_field.max()}")
        return
    print(f"Isosurface generated with {isosurface.n_points} points.")

    # Smooth + decimate for clean mesh 
    print("Smoothing and decimating mesh...")
    smoothed = isosurface.smooth(n_iter=15, relaxation_factor=0.01)
    smoothed = smoothed.decimate(0.5)
    print(f"Final mesh has {smoothed.n_points} points and {smoothed.n_cells} cells.")

    z_vals = smoothed.points[:, 2].copy()
    smoothed.point_data["Z"] = z_vals

    # Plotter
    plotter = pv.Plotter(window_size=[900, 900])
    plotter.set_background("black")

    # Lights
    light1 = pv.Light(position=(5, 5, 5), focal_point=(0, 0, 0),
                      color="white", intensity=0.8)
    plotter.add_light(light1)
    light2 = pv.Light(position=(-5, -5, 5), focal_point=(0, 0, 0),
                      color="lightblue", intensity=0.4)
    plotter.add_light(light2)

    # Mesh and Shading
    plotter.add_mesh(
        smoothed,
        scalars="Z",                  # use the Z‐values to drive color
        cmap="plasma",               
        smooth_shading=True,
        scalar_bar_args={
            "title": "Z",
            "vertical": True,
            "title_font_size": 12,
            "label_font_size": 10,
        },
        specular=0.2,                 
        specular_power=10,            
        ambient=0.6,                  
        diffuse=0.5,                  
    )

    # 6) Optional axes/grid
    if show_axes:
        plotter.add_axes(line_width=2, labels_off=False)
    if show_grid:
        plotter.show_grid(color="gray")

    plotter.camera_position = "iso"
    plotter.enable_anti_aliasing("fxaa")

    print("Displaying PyVista isosurface. Close window to finish.")
    plotter.show()
    print("Plot window closed.")
