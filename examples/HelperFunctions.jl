import Mantis

# FUNCTION SPACES -------------------------------------------------------------------

function univariate_breakpoints(num_el::Int, L::Float64 = 1.0)
    return collect(LinRange(0.0, L, num_el+1))
end

function univariate_bsplines(deg::Int, num_el::Int, reg::Int, L::Float64 = 1.0)
    # generate a uniform mesh
    breakpoints = univariate_breakpoints(num_el, L)
    patch = Mantis.Mesh.Patch1D(breakpoints)
    # return B-spline space
    return Mantis.FunctionSpaces.BSplineSpace(patch, deg, reg)
end

function univariate_bsplines(polynomials::F, num_el::Int, reg::Int, L::Float64 = 1.0) where {F <: Mantis.FunctionSpaces.AbstractCanonicalSpace}
    # generate a uniform mesh
    breakpoints = univariate_breakpoints(num_el, L)
    patch = Mantis.Mesh.Patch1D(breakpoints)
    # return B-spline space
    return Mantis.FunctionSpaces.BSplineSpace(patch, polynomials, reg)
end

function tensor_product_bsplines(deg::NTuple{n,Int}, num_el::NTuple{n,Int}, reg::NTuple{n,Int}, L::NTuple{n,Float64}) where {n}
    # generate dimension-wise spaces
    univariate_spaces = univariate_bsplines.(deg, num_el, reg, L)
    # build tensor-product space
    if n == 1
        return univariate_spaces[1]
    else
        TP = Mantis.FunctionSpaces.TensorProductSpace(univariate_spaces[1],univariate_spaces[2])
        for i = 3:n
            TP = Mantis.FunctionSpaces.TensorProductSpace(TP, univariate_spaces[i])
        end
        return TP
    end
end

function tensor_product_bsplines(polynomials::NTuple{n}, num_el::NTuple{n,Int}, reg::NTuple{n,Int}, L::NTuple{n,Float64}) where {n}
    # generate dimension-wise spaces
    univariate_spaces = univariate_bsplines.(polynomials, num_el, reg, L)
    # build tensor-product space
    if n == 1
        return univariate_spaces[1]
    else
        TP = Mantis.FunctionSpaces.TensorProductSpace(univariate_spaces[1],univariate_spaces[2])
        for i = 3:n
            TP = Mantis.FunctionSpaces.TensorProductSpace(TP, univariate_spaces[i])
        end
        return TP
    end
end

function tensor_product_de_rham_complex(□, p⁰, num_el, L, section_space_type, θ = 0)

    if section_space_type == "bernstein"
        # zero-form section spaces
        Wₖ = Mantis.FunctionSpaces.Bernstein.(p⁰)
        s⁰ = p⁰ .- (1,1)
        # one-form section spaces
        Xₖ₁ = Mantis.FunctionSpaces.Bernstein.(p⁰ .- (1,0))
        Xₖ₂ = Mantis.FunctionSpaces.Bernstein.(p⁰ .- (0,1))
        s¹₁ = s⁰ .- (1,0)
        s¹₂ = s⁰ .- (0,1)
        # two-form section spaces
        Yₖ = Mantis.FunctionSpaces.Bernstein.(p⁰ .- (1,1))
        s² = s⁰ .- (1,1)

        # quadrature rule degree for assembly
        q_assembly = p⁰ .+ (1,1)
        # quadrature rule degree for error computation
        q_error = q_assembly .* 2

    elseif section_space_type == "trigonometric"
        # zero-form section spaces
        Wₖ = Mantis.FunctionSpaces.GeneralizedTrigonometric.(p⁰, θ)
        s⁰ = p⁰ .- (1,1)
        # one-form section spaces
        Xₖ₁ = Mantis.FunctionSpaces.GeneralizedTrigonometric.(p⁰ .- (1,0), θ)
        Xₖ₂ = Mantis.FunctionSpaces.GeneralizedTrigonometric.(p⁰ .- (0,1), θ)
        s¹₁ = s⁰ .- (1,0)
        s¹₂ = s⁰ .- (0,1)
        # two-form section spaces
        Yₖ = Mantis.FunctionSpaces.GeneralizedTrigonometric.(p⁰ .- (1,1), θ)
        s² = s⁰ .- (1,1)

        # quadrature rule degree for assembly
        q_assembly = p⁰ .+ (1,1)
        # quadrature rule degree for error computation
        q_error = q_assembly .* 2

    elseif section_space_type == "legendre"
        # zero-form section spaces
        Wₖ = Mantis.FunctionSpaces.LobattoLegendre.(p⁰)
        s⁰ = (0,0)
        # one-form section spaces
        Xₖ₁ = Mantis.FunctionSpaces.LobattoLegendre.(p⁰ .- (1,0))
        Xₖ₂ = Mantis.FunctionSpaces.LobattoLegendre.(p⁰ .- (0,1))
        s¹₁ = s⁰ .- (1,0)
        s¹₂ = s⁰ .- (0,1)
        # two-form section spaces
        Yₖ = Mantis.FunctionSpaces.LobattoLegendre.(p⁰ .- (1,1))
        s² = s⁰ .- (1,1)

        # quadrature rule degree for assembly
        q_assembly = p⁰ .+ (1,1)
        # quadrature rule degree for error computation
        q_error = q_assembly .* 2

    end

    # zero-form space
    W = Mantis.Forms.FormSpace(0, □, (tensor_product_bsplines(Wₖ, num_el, s⁰, L),), "σ")
    # one-form space
    X₁ = tensor_product_bsplines(Xₖ₁, num_el, s¹₁, L)
    X₂ = tensor_product_bsplines(Xₖ₂, num_el, s¹₂, L)
    X = Mantis.Forms.FormSpace(1, □, (X₁,X₂), "σ")
    # two-form space
    Y = tensor_product_bsplines(Yₖ, num_el, s², L)
    Y = Mantis.Forms.FormSpace(2, □, (Y,), "u")

    # quadrature rules
    ∫ₐ = Mantis.Quadrature.tensor_product_rule(q_assembly, Mantis.Quadrature.gauss_legendre)
    ∫ₑ = Mantis.Quadrature.tensor_product_rule(q_error, Mantis.Quadrature.gauss_legendre)

    return W, X, Y, ∫ₐ, ∫ₑ
end

# GEOMETRIES -------------------------------------------------------------------

function unit_cube_cartesian(num_el::NTuple{n,Int}) where {n}
    breakpoints = univariate_breakpoints.(num_el)

    return Mantis.Geometry.CartesianGeometry(breakpoints)
end

function unit_square_curvilinear(num_el::NTuple{2,Int}, crazy_c::Float64 = 0.2)
    # build underlying Cartesian geometry
    unit_cube = unit_cube_cartesian(num_el)

    # build curved mapping
    Lleft = 0.0
    Lright = 1.0
    Lbottom = 0.0
    Ltop = 1.0
    function mapping(x::Vector{Float64})
        x1_new = (2.0/(Lright-Lleft))*x[1] - 2.0*Lleft/(Lright-Lleft) - 1.0
        x2_new = (2.0/(Ltop-Lbottom))*x[2] - 2.0*Lbottom/(Ltop-Lbottom) - 1.0
        return [x[1] + ((Lright-Lleft)/2.0)*crazy_c*sinpi(x1_new)*sinpi(x2_new), x[2] + ((Ltop-Lbottom)/2.0)*crazy_c*sinpi(x1_new)*sinpi(x2_new)]
    end
    function dmapping(x::Vector{Float64})
        x1_new = (2.0/(Lright-Lleft))*x[1] - 2.0*Lleft/(Lright-Lleft) - 1.0
        x2_new = (2.0/(Ltop-Lbottom))*x[2] - 2.0*Lbottom/(Ltop-Lbottom) - 1.0
        return [1.0 + pi*crazy_c*cospi(x1_new)*sinpi(x2_new) ((Lright-Lleft)/(Ltop-Lbottom))*pi*crazy_c*sinpi(x1_new)*cospi(x2_new); ((Ltop-Lbottom)/(Lright-Lleft))*pi*crazy_c*cospi(x1_new)*sinpi(x2_new) 1.0 + pi*crazy_c*sinpi(x1_new)*cospi(x2_new)]
    end
    dimension = (2, 2)
    curved_mapping = Mantis.Geometry.Mapping(dimension, mapping, dmapping)

    return Mantis.Geometry.MappedGeometry(unit_cube, curved_mapping)
end

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

# SOLUTION & GEOMETRY OUTPUT -------------------------------------------------------------------
function visualize_geometry(geo::Mantis.Geometry.AbstractGeometry, filename::String, n_subcells::Int = 1, degree::Int = 4)
    Mantis_folder =  dirname(dirname(pathof(Mantis)))
    data_folder = joinpath(Mantis_folder, "examples", "data")
    output_data_folder = joinpath(data_folder, "output")
    output_file = joinpath(output_data_folder, filename)
    Mantis.Plot.plot(geo; vtk_filename = output_file, n_subcells = n_subcells, degree = degree, ascii = false, compress = false)
    
    return nothing
end

function visualize_solution(form_sols, var_names, filename, geom, n_subcells::Int = 1, degree::Int = 4)

    Mantis_folder =  dirname(dirname(pathof(Mantis)))
    data_folder = joinpath(Mantis_folder, "examples", "data")
    output_data_folder = joinpath(data_folder, "output")
    
    # This is for the plotting.
    for (form_sol, var_name) in zip(form_sols, var_names)
        println("Writing form '$var_name' to file ...")
        output_file = joinpath(output_data_folder, "$filename-$var_name")
        Mantis.Plot.plot(form_sol; vtk_filename = output_file, n_subcells = n_subcells, degree = degree, ascii = false, compress = false)
    end

    return nothing
end

# PROBLEM SETUP -------------------------------------------------------------------
function sinusoidal_solution(form_rank::Int, geo::Mantis.Geometry.AbstractGeometry)
    ω = 2.0 * pi
    function my_sol(x::Matrix{Float64})
        # [u] = [sin(ωx¹)sin(ωx²)...sin(ωxⁿ)]
        y = @. sin(ω * x)
        return [vec(prod(y, dims=2))]
    end

    function grad_my_sol(x::Matrix{Float64})
        # [u₁, u₂, ...] = [ω*cos(ωx¹)sin(ωx²)...sin(ωxⁿ), ω*sin(ωx¹)cos(ωx²)...sin(ωxⁿ), ...]
        y = sin.(ω .* x)
        z = ω .* cos.(ω .* x)
        w = Vector{Vector{Float64}}(undef, size(x, 2))
        for i ∈ 1:size(x,2)
            w[i] = z[:,i] .* prod(y[:,setdiff(1:size(x,2), i)], dims=2)[:,1]
        end
        return w
    end

    function flux_my_sol(x::Matrix{Float64})
        # [u₁, u₂, ...] = ⋆[ω*cos(ωx¹)sin(ωx²)...sin(ωxⁿ), ω*sin(ωx¹)cos(ωx²)...sin(ωxⁿ), ...]
        w = grad_my_sol(x)
        if size(x,2) == 1
            # (a) -> (a)
            return [w[1]]
            
        elseif size(x,2) == 2
            # (a, b) -> (-b, a)
            w = [-w[2], w[1]]
            return w

        elseif size(x,2) == 3
            # (a, b, c) -> (a, b, c)
            return [w[1], w[2], w[3]]

        else
            throw(ArgumentError("Dimension not supported."))
        end
    end

    function laplace_my_sol(x::Matrix{Float64})
        # [-(u₁₁+u₂₂+...uₙₙ)] = [2ω²*sin(ωx¹)sin(ωx²)...sin(ωxⁿ)]
        y = prod(sin.(ω * x), dims=2)
        y = @. 2 * ω * ω * y
        return [vec(y)]
    end

    if form_rank == 0
        u⁰ = Mantis.Forms.AnalyticalFormField(0, my_sol, geo, "u")
        du⁰ = Mantis.Forms.AnalyticalFormField(1, grad_my_sol, geo, "du")
        f⁰ = Mantis.Forms.AnalyticalFormField(0, laplace_my_sol, geo, "f")
        return u⁰, du⁰, f⁰

    elseif form_rank == 2
        u² = Mantis.Forms.AnalyticalFormField(2, my_sol, geo, "u")
        δu² = Mantis.Forms.AnalyticalFormField(1, flux_my_sol, geo, "δu")
        f² = Mantis.Forms.AnalyticalFormField(2, laplace_my_sol, geo, "f")
        return u², δu², f²

    else
        throw(ArgumentError("Form rank not supported."))
    end
    
end

# NORM COMPUTATION -------------------------------------------------------------------

function L2_norm(u, ∫)
    norm = 0.0
    for el_id ∈ 1:Mantis.Geometry.get_num_elements(u.geometry)
        inner_prod = SparseArrays.sparse(Mantis.Forms.evaluate_inner_product(u, u, el_id, ∫)...)
        norm += inner_prod[1,1]
    end
    return sqrt(norm)
end