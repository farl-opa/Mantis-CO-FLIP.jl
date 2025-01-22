module MetricTests

import Mantis

include("GeometryTestsHelpers.jl")

using Test

# CartesianGeometry (1, 1) homogeneous grid ------------------------------------------------
dim = 1
nx = 5
breakpoints_cart_1_1 = (collect(LinRange(0.0, 1.0, nx + 1)),)
cartesian_geometry_cart_1_1 = Mantis.Geometry.CartesianGeometry(breakpoints_cart_1_1)

# Expected Jacobian per element (the same for all elements)
dx_cart_1_1 = [1.0 / nx]
g_ref_cart_1_1 = [dx_cart_1_1[1]^2]
inv_g_ref_cart_1_1 = [dx_cart_1_1[1]^(-2)]
det_g_ref_cart_1_1 = sqrt(prod(dx_cart_1_1.^2))

# Points where to evaluate the metric
nx_evaluate = 3
xi_1_cart_1_1 = collect(LinRange(0.0, 1.0, nx_evaluate + 1))
n_evaluation_points = nx_evaluate

# Evaluate the metric, its inverse and its determinant 
for element_idx in 1:Mantis.Geometry.get_num_elements(cartesian_geometry_cart_1_1)
    inv_g, g, sqrt_g = Mantis.Geometry.inv_metric(
        cartesian_geometry_cart_1_1, element_idx, (xi_1_cart_1_1,)
    )
    for dim_1_idx in 1:dim
        for dim_2_idx in 1:dim
            @test all(isapprox.(
                g[:, dim_1_idx, dim_2_idx], g_ref_cart_1_1[dim_1_idx, dim_2_idx];
                rtol=rtol
            ))
            @test all(isapprox.(
                inv_g[:, dim_1_idx, dim_2_idx], inv_g_ref_cart_1_1[dim_1_idx, dim_2_idx];
                rtol=rtol
            ))
        end
    end
    
    @test all(isapprox.(sqrt_g[:], det_g_ref_cart_1_1; rtol=rtol))
end
# ------------------------------------------------------------------------------------------

# CartesianGeometry (2, 2) homogeneous grid ------------------------------------------------
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

# Evaluate the metric, its inverse and its determinant 
for element_idx in 1:Mantis.Geometry.get_num_elements(cartesian_geometry_cart_2_2)
    inv_g, g, sqrt_g = Mantis.Geometry.inv_metric(
        cartesian_geometry_cart_2_2, element_idx, (xi_1_cart_2_2, xi_2_cart_2_2)
    )
    for dim_1_idx in 1:dim
        for dim_2_idx in 1:dim
            @test all(isapprox.(
                g[:, dim_1_idx, dim_2_idx], g_ref_cart_2_2[dim_1_idx, dim_2_idx];
                rtol=rtol
            ))
            @test all(isapprox.(
                inv_g[:, dim_1_idx, dim_2_idx], inv_g_ref_cart_2_2[dim_1_idx, dim_2_idx];
                rtol=rtol
            ))
        end
    end
    
    @test all(isapprox.(sqrt_g[:], det_g_ref_cart_2_2; rtol=rtol))
end
# ------------------------------------------------------------------------------------------

# CartesianGeometry (2, 2) inhomogeneous grid ----------------------------------------------
dim = 2
breakpoints_cart_2_2_inh = ([0.0, 0.25, 1.0], [0.0, 0.5, 0.9, 1.0])
cartesian_geometry_cart_2_2_inh = Mantis.Geometry.CartesianGeometry(
    breakpoints_cart_2_2_inh
)

# Points where to evaluate the metric
nx_evaluate = 3
ny_evaluate = 7
xi_1_cart_2_2_inh = collect(LinRange(0.0, 1.0, nx_evaluate + 1))
xi_2_cart_2_2_inh = collect(LinRange(0.0, 1.0, ny_evaluate + 1))
n_evaluation_points = nx_evaluate * ny_evaluate

# Expected metric terms per element (allocation)
dx_cart_2_2_inh = [
    0.25 0.75 0.25 0.75 0.25 0.75;
    0.5  0.5  0.4  0.4  0.1  0.1
]  # the dxs for each element are over the columns

# Evaluate the metric, its inverse and its determinant 
for element_idx in 1:Mantis.Geometry.get_num_elements(cartesian_geometry_cart_2_2_inh)
    inv_g, g, sqrt_g = Mantis.Geometry.inv_metric(
        cartesian_geometry_cart_2_2_inh, element_idx, (xi_1_cart_2_2_inh, xi_2_cart_2_2_inh)
    )
    for dim_1_idx in 1:dim
        for dim_2_idx in 1:dim
            if dim_1_idx != dim_2_idx
                @test all(isapprox.(g[:, dim_1_idx, dim_2_idx], 0.0; atol=atol))
                @test all(isapprox.(inv_g[:, dim_1_idx, dim_2_idx], 0.0; atol=atol))
            else
                g_ref_cart_2_2_inh = dx_cart_2_2_inh[dim_1_idx, element_idx]^2
                inv_g_ref_cart_2_2_inh = 1.0 / g_ref_cart_2_2_inh
                @test all(isapprox.(
                    g[:, dim_1_idx, dim_2_idx], g_ref_cart_2_2_inh; rtol=rtol)
                )
                @test all(isapprox.(
                    inv_g[:, dim_1_idx, dim_2_idx], inv_g_ref_cart_2_2_inh; rtol=rtol)
                )
            end
        end
    end
     
    # Test det_g
    det_g_ref_cart_2_2_inh = prod(dx_cart_2_2_inh[:, element_idx])
    @test all(isapprox.(sqrt_g[:], det_g_ref_cart_2_2_inh; rtol=rtol))
end
# ------------------------------------------------------------------------------------------
end
