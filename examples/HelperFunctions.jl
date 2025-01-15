function unit_square_biquadratic(num_ref::Int = 0)
    deg = (2,2)
    TP = tensor_product_bsplines(deg, (1,1), (-1,-1), (1.0,1.0))
    geom_coeffs =  [0.0 0.0
                    0.5 0.0
                    1.0 0.0
                    0.0 0.5
                    0.95 0.95
                    1.0 0.5
                    0.0 1.0
                    0.5 1.0
                    1.0 1.0]
    if num_ref == 0
        return Mantis.Geometry.FEMGeometry(TP, geom_coeffs)
    else
        ts_TP, TP_refined = Mantis.FunctionSpaces.build_two_scale_operator(TP, (2^num_ref,2^num_ref))
        geom_coeffs_refined = ts_TP.global_subdiv_matrix * geom_coeffs
        return Mantis.Geometry.FEMGeometry(TP_refined, geom_coeffs_refined)
    end
end

function bilinear_wedge(num_ref::Int = 0)
    deg = (1,1)
    TP = tensor_product_bsplines(deg, (1,1), (-1,-1), (1.0,1.0))
    geom_coeffs =  [0.0 0.0
                    1.0 0.0
                    0.0 1.0
                    1.5 1.75]
    if num_ref == 0
        return Mantis.Geometry.FEMGeometry(TP, geom_coeffs)
    else
        ts_TP, TP_refined = Mantis.FunctionSpaces.build_two_scale_operator(TP, (2^num_ref,2^num_ref))
        geom_coeffs_refined = ts_TP.global_subdiv_matrix * geom_coeffs
        return Mantis.Geometry.FEMGeometry(TP_refined, geom_coeffs_refined)
    end
end

function quarter_annulus_nurbs(radius_inner::Float64, radius_outer::Float64)
    deg = 2
    num_el = 1
    # B-spline space - angular direction
    Bθ = univariate_bsplines(deg, num_el, -1)
    # NURBS space - angular direction
    Rθ = Mantis.FunctionSpaces.RationalFiniteElementSpace(Bθ, [1, 1/sqrt(2), 1])
    # B-spline space - radial direction
    Br = univariate_bsplines(deg, num_el, -1)
    # tensor-product space
    TP = Mantis.FunctionSpaces.TensorProductSpace(Rθ, Br)
    # control points for quarter-circle
    geom_coeffs_sector =   [0.0 1.0
                            1.0   1.0
                            1.0   0.0]
    # control points for quarter annulus
    radii = LinRange(radius_inner, radius_outer, deg+1)
    geom_coeffs = vcat([geom_coeffs_sector.*radii[i] for i ∈ eachindex(radii)]...)

    return Mantis.Geometry.FEMGeometry(TP, geom_coeffs)
end

function quarter_annulus_trigonometric_bspline(radius_inner::Float64, radius_outer::Float64)
    deg = 2
    Wt = pi/2
    num_el = 1
    # TB-spline space - angular direction
    Bθ = univariate_bsplines(Mantis.FunctionSpaces.GeneralizedTrigonometric(deg,Wt), num_el, -1)
    # B-spline space - radial direction
    Br = univariate_bsplines(deg, 1, -1)
    # tensor-product space
    TP = Mantis.FunctionSpaces.TensorProductSpace(Bθ, Br)
    # control points for quarter-circle
    geom_coeffs_sector =   [0.0 1.0
                            1.0   1.0
                            1.0   0.0]
    # control points for quarter annulus
    radii = LinRange(radius_inner, radius_outer, deg+1)
    geom_coeffs = vcat([geom_coeffs_sector.*radii[i] for i ∈ eachindex(radii)]...)

    return Mantis.Geometry.FEMGeometry(TP, geom_coeffs)
end

function quarter_bulging_annulus_trigonometric_bspline(radius_inner::Float64, radius_outer::Float64)
    deg = 2
    Wt = pi/2
    num_el = 1
    # TB-spline space - angular direction
    Bθ = univariate_bsplines(Mantis.FunctionSpaces.GeneralizedTrigonometric(deg,Wt), num_el, -1)
    # B-spline space - radial direction
    Br = univariate_bsplines(deg, 1, -1)
    # tensor-product space
    TP = Mantis.FunctionSpaces.TensorProductSpace(Bθ, Br)
    # control points for quarter-circle
    geom_coeffs_sector =   [0.0 1.0
                            1.0   1.0
                            1.0   0.0]
    # control points for quarter annulus
    radii = LinRange(radius_inner, radius_outer, deg+1)
    geom_coeffs = vcat([geom_coeffs_sector.*radii[i] for i ∈ eachindex(radii)]...)

    return Mantis.Geometry.FEMGeometry(TP, geom_coeffs)
end
