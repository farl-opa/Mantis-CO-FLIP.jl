import Mantis
import SparseArrays
using Test

function my_sol(x::Matrix{Float64})
    ω = 0.1
    y = prod(sin.(ω * x), dims=2)
    return [vec(y)]
end

function L2_norm(u, ∫)
    norm = 0.0
    for el_id ∈ 1:Mantis.Geometry.get_num_elements(u.geometry)
        inner_prod = SparseArrays.sparse(Mantis.Forms.evaluate_inner_product(u, u, el_id, ∫)...)
        norm += inner_prod[1,1]
    end
    return sqrt(norm)
end

function visualize_geometry(geo::Mantis.Geometry.AbstractGeometry, filename::String, n_subcells::Int = 1, degree::Int = 4)
    Mantis_folder =  dirname(dirname(pathof(Mantis)))
    data_folder = joinpath(Mantis_folder, "examples", "data")
    output_data_folder = joinpath(data_folder, "output")
    output_file = joinpath(output_data_folder, filename)
    Mantis.Plot.plot(geo; vtk_filename = output_file, n_subcells = n_subcells, degree = degree, ascii = false, compress = false)
    
    return nothing
end

function visualize_solution(form_sols, var_names, filename, n_subcells::Int = 1, degree::Int = 4)

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

function visualize_tensor_product_controlnet(control_points::Array{Float64}, manifold_dim::Int, range_dim::Int, periodic::Vector{Bool}, filename::String)
    # create bilinear geometry
    B = [Mantis.FunctionSpaces.BSplineSpace(Mantis.Mesh.Patch1D(collect(LinRange(0.0, 1.0, size(control_points,i)+periodic[i]))), 1, 0) for i in 1:manifold_dim]
    if periodic[1]
        TP = Mantis.FunctionSpaces.GTBSplineSpace((B[1],), [0])
    else
        TP = B[1]
    end
    for i in 2:manifold_dim
        if periodic[i]
            TP = Mantis.FunctionSpaces.TensorProductSpace((TP, Mantis.FunctionSpaces.GTBSplineSpace((B[i],), [0])))
        else
            TP = Mantis.FunctionSpaces.TensorProductSpace((TP, B[i]))
        end
    end

    # create geometry
    geo = Mantis.Geometry.FEMGeometry(TP, reshape(control_points, :, range_dim))
    
    # export to vtk
    visualize_geometry(geo, filename, 1, 1)
    
    return nothing
end

function L2_projection(X, ∫, fₑ)
    # inputs for the mixed weak form
    weak_form_inputs = Mantis.Assemblers.WeakFormInputs(fₑ, X, X, ∫)
    
    # assemble all matrices
    weak_form = Mantis.Assemblers.l2_weak_form
    A, b = Mantis.Assemblers.assemble(weak_form, weak_form_inputs, Dict{Int, Float64}())
    
    # solve for coefficients of solution
    sol = A \ b

    # create solution as forms and return
    uₕ = Mantis.Forms.FormField(X, "u")
    uₕ.coefficients .= sol
    
    return uₕ
end

# Test for L2 projection on polar splines in 2D -------------------------------------

function build_polar_spline_space_and_geometry(deg, nel_r, nel_θ, R; refine::Bool = false, geom_coeffs_tp::Union{Nothing,Array{Float64,3}}=nothing, form_rank::Int = 0)
    # patches in r and θ
    patch_r = Mantis.Mesh.Patch1D(collect(LinRange(0.0, 1.0, nel_r+1)))
    patch_θ = Mantis.Mesh.Patch1D(collect(LinRange(0.0, 1.0, nel_θ+1)))

    # spline space in r
    Br = Mantis.FunctionSpaces.BSplineSpace(patch_r, deg, deg-1)
    # spline space in θ
    Bθ = Mantis.FunctionSpaces.BSplineSpace(patch_θ, deg, deg-1)
    GBθ = Mantis.FunctionSpaces.GTBSplineSpace((Bθ,), [deg-1])
    
    # control points for a degenerate tensor-product mapping
    n_θ = Mantis.FunctionSpaces.get_num_basis(GBθ)
    n_r = Mantis.FunctionSpaces.get_num_basis(Br)

    ts_θ = nothing
    ts_r = nothing
    if isnothing(geom_coeffs_tp)
        geom_coeffs_tp, _, _ = Mantis.FunctionSpaces.build_standard_degenerate_control_points(n_θ, n_r, R)
    else
        # geometry has already been provided, just check if we need to refine it
        if refine
            # refine the univariate spaces
            ts_r, Br_ref = Mantis.FunctionSpaces.build_two_scale_operator(Br, 2)
            _, Bθ_ref = Mantis.FunctionSpaces.build_two_scale_operator(Bθ, 2)
            GBθ_ref = Mantis.FunctionSpaces.GTBSplineSpace((Bθ_ref,), [deg-1])
            ts_θ, _ = Mantis.FunctionSpaces.build_two_scale_operator(GBθ,GBθ_ref,((2,),))
            
            # refine the tensor-product control points
            geom_coeffs_tp = cat(ts_θ.global_subdiv_matrix * geom_coeffs_tp[:,:,1] * ts_r.global_subdiv_matrix', ts_θ.global_subdiv_matrix * geom_coeffs_tp[:,:,2] * ts_r.global_subdiv_matrix'; dims=3)
            
            # update old spaces
            Br = Br_ref
            Bθ = Bθ_ref
            GBθ = GBθ_ref
        end
    end

    # Polar spline space and global extraction matrix for the geometry
    P_geom, E_geom = Mantis.FunctionSpaces.PolarSplineSpace(GBθ, Br, (geom_coeffs_tp[:,1,:],geom_coeffs_tp[:,2,:]); form_rank = 0)
    # control points for the polar spline space
    geom_coeffs_polar = (E_geom[1] * E_geom[1]') \ (E_geom[1] * reshape(geom_coeffs_tp,:, 2))
    # polar spline geometry
    ○ = Mantis.Geometry.FEMGeometry(P_geom[1], geom_coeffs_polar)

    # Polar spline space and global extraction matrix for the solution
    dBr = Mantis.FunctionSpaces.get_derivative_space(Br)
    dBθ = Mantis.FunctionSpaces.get_derivative_space(Bθ)
    dGBθ = Mantis.FunctionSpaces.GTBSplineSpace((dBθ,), [deg-2])
    P_sol, _ = Mantis.FunctionSpaces.PolarSplineSpace(GBθ, Br, (geom_coeffs_tp[:,1,:],geom_coeffs_tp[:,2,:]); form_rank = form_rank, dspace_p = dGBθ, dspace_r = dBr)
    # form space
    X = Mantis.Forms.FormSpace(form_rank, ○, P_sol, "σ")

    return X, ○, E_geom, geom_coeffs_tp, (ts_θ, ts_r)
end

for deg in 2:3
    for form_rank in [0, 2]
        nel_r = 5
        nel_θ = 15
        R = 1.0
        n_ref = 4
        verbose=false
        
        # form rank
        if form_rank == 1
            throw(ArgumentError("Form rank 1 is not supported for this test."))
        end

        # quadrature rule degree for assembly
        q_assembly = (deg, deg) .+ (1,1)
        # quadrature rule degree for error computation
        q_error = q_assembly .* 2
        # quadrature rules
        ∫ₐ = Mantis.Quadrature.tensor_product_rule(q_assembly, Mantis.Quadrature.gauss_legendre)
        ∫ₑ = Mantis.Quadrature.tensor_product_rule(q_error, Mantis.Quadrature.gauss_legendre)

        errors = zeros(n_ref+1)
        geom_coeffs_tp = nothing
        E_○ = nothing
        for r = 0:n_ref
            if verbose
                println("Building polar splines for refinement level $r...")
            end
            if r == 0
                X, ○, E_○, geom_coeffs_tp = build_polar_spline_space_and_geometry(deg, nel_r, nel_θ, R; form_rank = form_rank)
            else
                X, ○, E_○, geom_coeffs_tp = build_polar_spline_space_and_geometry(deg, nel_r, nel_θ, R; refine = true, geom_coeffs_tp = geom_coeffs_tp, form_rank = form_rank)
                nel_r *= 2
                nel_θ *= 2
            end
            # exact solution
            uₑ = Mantis.Forms.AnalyticalFormField(form_rank, my_sol, ○, "u")
            if verbose
                println("Solving the problem...")
            end
            uₕ = L2_projection(X, ∫ₐ, uₑ)
            if verbose
                println("Computing error...")
            end
            global errors[r+1] = L2_norm(uₕ - uₑ, ∫ₑ)
            if verbose
                println("       Error in u: ", errors[r+1])
                println("...done!")
                # println("Visualizing the solution...")
                # visualize_solution((uₕ, uₑ-uₕ), ("uh", "error"), "k_form_L2_projection_polar_$r", 1, 4)
                println("--------------------------------------------")
            end
        end

        error_rates = log.(Ref(2), errors[1:end-1]./errors[2:end])
        if form_rank == 0
            @test isapprox(error_rates[end], deg+1, atol=1e-1)
        elseif form_rank == 2
            @test isapprox(error_rates[end], deg, atol=1e-1)
        end
    end
end

# Test for L2 projection on polar splines in 3D -------------------------------------
function build_toroidal_spline_space_and_geometry(deg, nel_r, nel_θ, nel_ϕ, R_θr, R_ϕ; form_rank::Int = 0, refine::Bool = false, geom_coeffs_tp::Union{Nothing,Array{Float64,4}}=nothing)

    # first, extract the geometry coefficients for the 2D polar splines
    if isnothing(geom_coeffs_tp)
        geom_coeffs_tp_cs = nothing
    else
        geom_coeffs_tp_cs = geom_coeffs_tp[:,:,1,1:2]
    end

    # then, build (and refine if needed) 2D polar spline space and geometry
    if form_rank == 0
        X, ○, E_○_θr, geom_coeffs_tp_cs, (ts_θ, ts_r) = build_polar_spline_space_and_geometry(deg, nel_r, nel_θ, R_θr; refine = refine, geom_coeffs_tp=geom_coeffs_tp_cs, form_rank = 0)
    elseif form_rank == 3
        X, ○, E_○_θr, geom_coeffs_tp_cs, (ts_θ, ts_r) = build_polar_spline_space_and_geometry(deg, nel_r, nel_θ, R_θr; refine = refine, geom_coeffs_tp=geom_coeffs_tp_cs, form_rank = 2)
    end

    # build spline space in ϕ
    patch_ϕ = Mantis.Mesh.Patch1D(collect(LinRange(0.0, 1.0, nel_ϕ+1)))
    Bϕ = Mantis.FunctionSpaces.BSplineSpace(patch_ϕ, deg, deg-1)
    GBϕ = Mantis.FunctionSpaces.GTBSplineSpace((Bϕ,), [deg-1])
    n_ϕ = Mantis.FunctionSpaces.get_num_basis(GBϕ)
    E_ϕ = Mantis.FunctionSpaces.assemble_global_extraction_matrix(GBϕ)

    # build the full, non-ϕ-periodic tensor-product geometry coefficients for toroidal geometry
    if isnothing(geom_coeffs_tp)
        # build the control points from scratch
        n_θ = size(geom_coeffs_tp_cs, 1)
        n_r = size(geom_coeffs_tp_cs, 2)

        # the full, non-ϕ-periodic tensor-product control points
        geom_coeffs_tp_cs = reshape(geom_coeffs_tp_cs, :, 2)
        geom_coeffs_tp_cs = [geom_coeffs_tp_cs.+[R_ϕ 0] zeros(size(geom_coeffs_tp_cs,1))]
        greville_ϕ = Mantis.FunctionSpaces.get_greville_points(Bϕ)[1]
        ϕ = greville_ϕ .* 2π
        geom_coeffs_tp = Vector{Matrix{Float64}}(undef,length(ϕ))
        for i ∈ eachindex(ϕ)
            R = [cos(ϕ[i]) 0 sin(ϕ[i]); 0 1 0; -sin(ϕ[i]) 0 cos(ϕ[i])]
            geom_coeffs_tp[i] = geom_coeffs_tp_cs * R'
        end
        geom_coeffs_tp = reshape(vcat(geom_coeffs_tp...), n_θ * n_r, :, 3)
        # project into the toroidal spline space and back
        geom_coeffs_tp = cat([((geom_coeffs_tp[:,:,i]*E_ϕ) / (E_ϕ' * E_ϕ)) * E_ϕ' for i = 1:3]..., dims=3) # ensure that points are in the image of the extraction operator
        geom_coeffs_tp = reshape(geom_coeffs_tp, n_θ, n_r, :, 3)

    else
        # geometry coefficients already defined, just check if they need to be refined
        # old sizes
        n_θ = size(geom_coeffs_tp, 1)
        n_r = size(geom_coeffs_tp, 2)
        if refine
            # two-scale relations
            ts_ϕ, Bϕ_ref = Mantis.FunctionSpaces.build_two_scale_operator(Bϕ, 2)
            
            # first, refine in ϕ direction
            geom_coeffs_tp = reshape(geom_coeffs_tp, n_θ*n_r, :, 3)
            geom_coeffs_tp = [geom_coeffs_tp[:,:,i] * ts_ϕ.global_subdiv_matrix' for i = 1:3]
            geom_coeffs_tp = reshape(cat(geom_coeffs_tp..., dims=3), n_θ, n_r, :, 3)
            
            # next, refine in θ and r directions
            geom_coeffs_tp = [[ts_θ.global_subdiv_matrix * geom_coeffs_tp[:,:,i,j] * ts_r.global_subdiv_matrix' for i in axes(geom_coeffs_tp,3)] for j = 1:3]
            geom_coeffs_tp = [cat(geom_coeffs_tp[j]..., dims=3) for j = 1:3]
            geom_coeffs_tp = cat(geom_coeffs_tp..., dims=4)

            # update quantities
            Bϕ = Bϕ_ref
            GBϕ = Mantis.FunctionSpaces.GTBSplineSpace((Bϕ_ref,), [deg-1])
            n_θ = size(ts_θ.global_subdiv_matrix, 1)
            n_r = size(ts_r.global_subdiv_matrix, 1)
            n_ϕ = Mantis.FunctionSpaces.get_num_basis(GBϕ)
            E_ϕ = Mantis.FunctionSpaces.assemble_global_extraction_matrix(GBϕ)
        end
    end

    # Toroidal spline space and global extraction matrix for the geometry
    T_geom = Mantis.FunctionSpaces.TensorProductSpace((○.fem_space, GBϕ))
    # control points for the toroidal spline space
    geom_coeffs_toroidal = reshape(geom_coeffs_tp, n_θ*n_r, :, 3)
    geom_coeffs_toroidal = cat([((geom_coeffs_toroidal[:,:,i]*E_ϕ) / (E_ϕ' * E_ϕ)) for i = 1:3]..., dims=3) # impose periodicity in ϕ
    geom_coeffs_toroidal = vcat([(E_○_θr[1] * E_○_θr[1]') \ (E_○_θr[1] * geom_coeffs_toroidal[:,i,:]) for i in axes(geom_coeffs_toroidal,2)]...) # impose polar extraction in (θ,r)
    # toroidal spline geometry
    T = Mantis.Geometry.FEMGeometry(T_geom, geom_coeffs_toroidal)

    # Toroidal form space
    if form_rank == 0
        T_sol = Mantis.FunctionSpaces.TensorProductSpace((X.fem_space[1], GBϕ))
    elseif form_rank == 3
        dBϕ = Mantis.FunctionSpaces.get_derivative_space(Bϕ)
        dGBϕ = Mantis.FunctionSpaces.GTBSplineSpace((dBϕ,), [deg-2])
        T_sol = Mantis.FunctionSpaces.TensorProductSpace((X.fem_space[1], dGBϕ))
    end
    # form space
    X = Mantis.Forms.FormSpace(form_rank, T, (T_sol,), "σ")

    return X, T, geom_coeffs_tp
end

for deg in 2:2
    for form_rank in [0, 3]
        nel_r = 2
        nel_θ = 5
        nel_ϕ = 5
        R_θr = 1.0
        R_ϕ = 2.0
        n_ref = 3
        verbose=false
        
        # form rank
        if form_rank == 1 || form_rank == 2
            throw(ArgumentError("Form ranks 1 and 2 are not supported for this test."))
        end

        # quadrature rule degree for assembly
        q_assembly = (deg, deg, deg) .+ (1,1,1)
        # quadrature rule degree for error computation
        q_error = q_assembly .* 2
        # quadrature rules
        ∫ₐ = Mantis.Quadrature.tensor_product_rule(q_assembly, Mantis.Quadrature.gauss_legendre)
        ∫ₑ = Mantis.Quadrature.tensor_product_rule(q_error, Mantis.Quadrature.gauss_legendre)

        errors = zeros(n_ref+1)
        geom_coeffs_toroidal = nothing
        for r = 0:n_ref
            if verbose
                println("Building toroidal splines for refinement level $r...")
            end
            if r == 0
                X, T, geom_coeffs_toroidal = build_toroidal_spline_space_and_geometry(deg, nel_r, nel_θ, nel_ϕ, R_θr, R_ϕ; form_rank = form_rank)
            else
                X, T, geom_coeffs_toroidal = build_toroidal_spline_space_and_geometry(deg, nel_r, nel_θ, nel_ϕ, R_θr, R_ϕ; form_rank = form_rank, refine = true, geom_coeffs_tp = geom_coeffs_toroidal)
                nel_r *= 2
                nel_θ *= 2
                nel_ϕ *= 2
            end
            # exact solution
            uₑ = Mantis.Forms.AnalyticalFormField(form_rank, my_sol, T, "u")
            if verbose
                n_dofs = Mantis.FunctionSpaces.get_num_basis(X.fem_space[1])
                println("Solving the problem with $n_dofs dofs...")
            end
            uₕ = L2_projection(X, ∫ₐ, uₑ)
            if verbose
                println("Computing error...")
            end
            global errors[r+1] = L2_norm(uₕ - uₑ, ∫ₑ)
            if verbose
                println("       Error in u: ", errors[r+1])
                println("...done!")
            end
            # if n_dofs < 2e5
            #     println("Visualizing the solution...")
            #     n_subcells = 20
            #     visualize_solution((uₕ,), ("u_h",), "$form_rank _form_L2_projection_toroidal_$r _$n_subcells _$deg", n_subcells, deg)
            #     visualize_tensor_product_controlnet(geom_coeffs_toroidal, 3, 3, [true, false, false], "L2_projection_toroidal_$r _controlnet")
            #     println("--------------------------------------------")
            # end
        end

        error_rates = log.(Ref(2), errors[1:end-1]./errors[2:end])
        if form_rank == 0
            @test isapprox(error_rates[end], deg+1, atol=2e-1)
        elseif form_rank == 2
            @test isapprox(error_rates[end], deg, atol=2e-1)
        end
    end
end