import Mantis

using Test

# CartesianGeometry (1, 1) homogeneous grid -----------------------------------
dim = 1
nx = 5
breakpoints_cart_1_1 = (collect(LinRange(0.0, 1.0, nx+1)),)
cartesian_geometry_cart_1_1 = Mantis.Geometry.CartesianGeometry(breakpoints_cart_1_1)

# Expected Jacobian per element (the same for all elements)
dx_cart_1_1 = [1.0/nx]
g_ref_cart_1_1 = [dx_cart_1_1[1]^2]
inv_g_ref_cart_1_1 = [dx_cart_1_1[1]^(-2)]
det_g_ref_cart_1_1 = sqrt(prod(dx_cart_1_1.^2))

# Points where to evaluate the metric
nx_evaluate = 3
xi_1_cart_1_1 = collect(LinRange(0.0, 1.0, nx_evaluate + 1))
n_evaluation_points = nx_evaluate

# Evaluate the metric
for element_idx in 1:Mantis.Geometry.get_num_elements(cartesian_geometry_cart_1_1) 
    g, sqrt_g = Mantis.Geometry.metric(cartesian_geometry_cart_1_1, element_idx, (xi_1_cart_1_1,))
    
    # Test g
    for dim_1_idx in 1:dim
        for dim_2_idx in 1:dim
            @test sum(abs.(g[:, dim_1_idx, dim_2_idx] .- g_ref_cart_1_1[dim_1_idx, dim_2_idx])) ≈ 0 atol = 1e-14  
        end
    end

    # Test det_g
    @test sum(abs.(sqrt_g[:] .- det_g_ref_cart_1_1)) ≈ 0 atol = 1e-14
end

# Evaluate the inverse of the metric
for element_idx in 1:Mantis.Geometry.get_num_elements(cartesian_geometry_cart_1_1) 
    inv_g, g, sqrt_g = Mantis.Geometry.inv_metric(cartesian_geometry_cart_1_1, element_idx, (xi_1_cart_1_1,))
    for dim_1_idx in 1:dim
        for dim_2_idx in 1:dim
            @test sum(abs.(g[:, dim_1_idx, dim_2_idx] .- g_ref_cart_1_1[dim_1_idx, dim_2_idx]))  ≈ 0 atol = 1e-14 
            @test sum(abs.(inv_g[:, dim_1_idx, dim_2_idx] .- inv_g_ref_cart_1_1[dim_1_idx, dim_2_idx]))  ≈ 0 atol = 1e-13 
            @test sum(abs.(sqrt_g[:] .- det_g_ref_cart_1_1))  ≈ 0 atol = 1e-14 
        end
    end
end
# -----------------------------------------------------------------------------

# CartesianGeometry (2, 2) homogeneous grid -----------------------------------
dim = 2
nx = 4
ny = 5
breakpoints_cart_2_2 = (collect(LinRange(0.0, 1.0, nx+1)), collect(LinRange(0.0,2.0,ny+1)))
cartesian_geometry_cart_2_2 = Mantis.Geometry.CartesianGeometry(breakpoints_cart_2_2)

# Expected Jacobian per element (the same for all elements)
dx_cart_2_2 = [1.0/nx, 2.0/ny]
g_ref_cart_2_2 = [dx_cart_2_2[1]^2 0.0; 0.0 dx_cart_2_2[2]^2]
inv_g_ref_cart_2_2 = [dx_cart_2_2[1]^(-2) 0.0; 0.0 dx_cart_2_2[2]^(-2)]
det_g_ref_cart_2_2 = prod(dx_cart_2_2)

# Points where to evaluate the metric
nx_evaluate = 3
ny_evaluate = 7
xi_1_cart_2_2 = collect(LinRange(0.0, 1.0, nx_evaluate + 1))
xi_2_cart_2_2 = collect(LinRange(0.0, 1.0, ny_evaluate + 1))
n_evaluation_points = nx_evaluate * ny_evaluate

# Evaluate the metric
for element_idx in 1:Mantis.Geometry.get_num_elements(cartesian_geometry_cart_2_2) 
    g, sqrt_g = Mantis.Geometry.metric(cartesian_geometry_cart_2_2, element_idx, (xi_1_cart_2_2, xi_2_cart_2_2))
    
    # Test g
    for dim_1_idx in 1:dim
        for dim_2_idx in 1:dim
            @test sum(abs.(g[:, dim_1_idx, dim_2_idx] .- g_ref_cart_2_2[dim_1_idx, dim_2_idx])) ≈ 0 atol = 1e-14  
        end
    end

    # Test det_g
    @test sum(abs.(sqrt_g[:] .- det_g_ref_cart_2_2)) ≈ 0 atol = 1e-14
end

# Evaluate the inverse of the metric
for element_idx in 1:Mantis.Geometry.get_num_elements(cartesian_geometry_cart_2_2) 
    inv_g, g, sqrt_g = Mantis.Geometry.inv_metric(cartesian_geometry_cart_2_2, element_idx, (xi_1_cart_2_2, xi_2_cart_2_2))
    
    # Test g and inv_g
    for dim_1_idx in 1:dim
        for dim_2_idx in 1:dim
            @test sum(abs.(g[:, dim_1_idx, dim_2_idx] .- g_ref_cart_2_2[dim_1_idx, dim_2_idx]))  ≈ 0 atol = 1e-14 
            @test sum(abs.(inv_g[:, dim_1_idx, dim_2_idx] .- inv_g_ref_cart_2_2[dim_1_idx, dim_2_idx]))  ≈ 0 atol = 2e-13 
        end
    end

    # Test det_g
    @test sum(abs.(sqrt_g[:] .- det_g_ref_cart_2_2))  ≈ 0 atol = 1e-14
end 
# -----------------------------------------------------------------------------

# CartesianGeometry (2, 2) inhomogeneous grid ---------------------------------
dim = 2
breakpoints_cart_2_2_inh = ([0.0, 0.25, 1.0], [0.0, 0.5, 0.9, 1.0])
cartesian_geometry_cart_2_2_inh = Mantis.Geometry.CartesianGeometry(breakpoints_cart_2_2_inh)

# Points where to evaluate the metric
nx_evaluate = 3
ny_evaluate = 7
xi_1_cart_2_2_inh = collect(LinRange(0.0, 1.0, nx_evaluate + 1))
xi_2_cart_2_2_inh = collect(LinRange(0.0, 1.0, ny_evaluate + 1))
n_evaluation_points = nx_evaluate * ny_evaluate

# Expected metric terms per element (allocation)
dx_cart_2_2_inh = [0.25 0.75 0.25 0.75 0.25 0.75;
      0.5  0.5  0.4  0.4  0.1  0.1]  # the dxs for each element are over the columns

# Evaluate the metric
for element_idx in 1:Mantis.Geometry.get_num_elements(cartesian_geometry_cart_2_2_inh) 
    g, sqrt_g = Mantis.Geometry.metric(cartesian_geometry_cart_2_2_inh, element_idx, (xi_1_cart_2_2_inh, xi_2_cart_2_2_inh))

    # Test g
    for dim_1_idx in 1:dim
        for dim_2_idx in 1:dim
            if dim_1_idx != dim_2_idx
                g_ref_cart_2_2_inh = 0.0
            else
                g_ref_cart_2_2_inh = dx_cart_2_2_inh[dim_1_idx, element_idx]^2 
            end
            @test sum(abs.(g[:, dim_1_idx, dim_2_idx] .- g_ref_cart_2_2_inh)) ≈ 0 atol = 1e-14  
        end
    end

    # Test det_g
    det_g_ref_cart_2_2_inh = prod(dx_cart_2_2_inh[:, element_idx])
    @test sum(abs.(sqrt_g[:] .- det_g_ref_cart_2_2_inh)) ≈ 0 atol = 1e-14
end

# Evaluate the inverse of the metric
for element_idx in 1:Mantis.Geometry.get_num_elements(cartesian_geometry_cart_2_2_inh) 
    inv_g, g, sqrt_g = Mantis.Geometry.inv_metric(cartesian_geometry_cart_2_2_inh, element_idx, (xi_1_cart_2_2_inh, xi_2_cart_2_2_inh))
    
    # Test g and inv_g
    for dim_1_idx in 1:dim
        for dim_2_idx in 1:dim
            if dim_1_idx != dim_2_idx
                g_ref_cart_2_2_inh = 0.0
                inv_g_ref_cart_2_2_inh = 0.0
            else
                g_ref_cart_2_2_inh = dx_cart_2_2_inh[dim_1_idx, element_idx]^2 
                inv_g_ref_cart_2_2_inh = 1.0/g_ref_cart_2_2_inh
            end
            @test sum(abs.(g[:, dim_1_idx, dim_2_idx] .- g_ref_cart_2_2_inh))  ≈ 0 atol = 1e-14 
            @test sum(abs.(inv_g[:, dim_1_idx, dim_2_idx] .- inv_g_ref_cart_2_2_inh))  ≈ 0 atol = 3e-12 
        end
    end

    # Test det_g
    det_g_ref_cart_2_2_inh = prod(dx_cart_2_2_inh[:, element_idx])
    @test sum(abs.(sqrt_g[:] .- det_g_ref_cart_2_2_inh)) ≈ 0 atol = 1e-14
end
# -----------------------------------------------------------------------------
#=
# FEMGeometry (2, 3) ----------------------------------------------------------
deg = 2
Wt = pi/2
b = Mantis.FunctionSpaces.CanonicalFiniteElementSpace(Mantis.FunctionSpaces.GeneralizedTrigonometric(deg, Wt))
B = ntuple( i -> b, 4)
GB = Mantis.FunctionSpaces.GTBSplineSpace(B, [1, 1, 1, 1])
b1 = Mantis.FunctionSpaces.CanonicalFiniteElementSpace(Mantis.FunctionSpaces.Bernstein(1))
TP = Mantis.FunctionSpaces.TensorProductSpace(GB, b1, Dict())
# control points for geometry
geom_coeffs_0 =   [1.0  -1.0
    1.0   1.0
    -1.0   1.0
    -1.0  -1.0]
r0 = 1
r1 = 2
geom_coeffs = [geom_coeffs_0.*r0 -[+1.0, -1.0, +1.0, -1.0]
               geom_coeffs_0.*r1 [+1.0, -1.0, +1.0, -1.0]]
wavy_surface_geom = Mantis.Geometry.FEMGeometry(TP, geom_coeffs)

# Points where to evaluate the metric
nx_evaluate = 20
ny_evaluate = 20
xi_1_fem_2_2 = collect(LinRange(0.0, 1.0, nx_evaluate + 1))
xi_2_fem_2_2 = collect(LinRange(0.0, 1.0, ny_evaluate + 1))
n_evaluation_points = nx_evaluate * ny_evaluate

g_fem, sqrt_g_fem = Mantis.Geometry.metric(wavy_surface_geom, 1, (xi_1_fem_2_2, xi_2_fem_2_2))
inv_g_fem, g_fem_2, sqrt_g_fem = Mantis.Geometry.inv_metric(wavy_surface_geom, 1, (xi_1_fem_2_2, xi_2_fem_2_2))

=#
