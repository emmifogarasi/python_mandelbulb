import numpy as np
import pyvista as pv

def visualise_mandelbulb(scalar_field,
                                 bounds,
                                 isosurface_threshold=None,
                                 volume_cmap="coolwarm",
                                 isosurface_color="lightblue",
                                 show_axes=True,
                                 show_grid=True):
    """
    Visualises the 3D scalar field as an isosurface using PyVista.
    """

    print(f"Preparing visualisation for data of shape {scalar_field.shape}...")
    print(f"Using isosurface threshold: {isosurface_threshold}")

    grid = pv.ImageData()

    # Set dimensions of grid (number of points along each axis)
    grid.dimensions = np.array(scalar_field.shape)

    # Set the origin (nim corner) and spacing of the grid based on bounds
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    grid.origin = (xmin, ymin, zmin)

    # Spacing between points along each axis
    # Ensure dimensions are not 1 to avoid division by zero
    sx = (xmax - xmin) / (scalar_field.shape[0] - 1 if scalar_field.shape[0] > 1 else 1)
    sy = (ymax - ymin) / (scalar_field.shape[1] - 1 if scalar_field.shape[0] > 1 else 1)
    sz = (zmax - zmin) / (scalar_field.shape[2] - 1 if scalar_field.shape[0] > 1 else 1)
    grid.spacing = (sx, sy, sz)

    # Add scalar data to grid's point data
    # The 'F' order (Fortran order) is important for VTK compatibility
    # It ensures the 1D flattened array matches how VTK expects 3D data
    grid.point_data["escape_time"] = scalar_field.flatten(order="F")
    
    #print("Extracting isosurface...")
    #isosurface = grid.contour([threshold_value], scalars = "escape_time")

    #if isosurface.n_points == 0:
    #    print(f"Warning: No surface generated for threshold {threshold_value}")
    #    print("This might mean your threshold is outside the range of your data,")
    #    print("or the data doesn't form a clear surface at this level.")
    #    print(f"Data min: {scalar_field.min()}, Data max: {scalar_field.max()}")
    #    return
    
    #print(f"Isosurface generated with {isosurface.n_points}.points.")



    # Plot the isosurface
    plotter = pv.Plotter(window_size=[800, 800])

    print(f"Adding volume rendering...")
    opacity_tf = np.linspace(0, 0.4, scalar_field.max() + 1)
    opacity_tf[:5] = 0

    plotter.add_volume(
        grid, 
        scalars="escape_time",
        cmap=volume_cmap,
        opacity=opacity_tf,
        mapper="gpu",
        blending="composite",
        shade=True
        #ambient=0.3, diffuse=0.7, specular=0.3
    )

    #plotter.add_mesh(
    #    isosurface,
    #    cmap=cmap,
    #    smooth_shading=smooth_shading,
    #    specular=0.7,   # Shininess
    #    specular_power=15   #Intensity of shininess
    #)

    #light = pv.Light(position=(5, 5, 5), focal_point=(0, 0, 0), color="white")
    #plotter.add_light(light)

    if show_axes:
        plotter.add_axes()
    if show_grid:
        plotter.show_grid()

    plotter.camera_position = "iso"
    plotter.enable_anti_aliasing("fxaa")

    print("Displaying PyVista plot window. Close the window to exit script.")
    plotter.show()
    print("Plot window closed.")