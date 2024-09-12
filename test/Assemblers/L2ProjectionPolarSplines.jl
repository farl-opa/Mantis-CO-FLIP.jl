import Mantis
import SparseArrays
using Test

# Problem parameters
function run_polar_spline_L2_projection_test(deg, form_rank)
    nel_r = 5
    nel_θ = 15
    R = 1.0
    n_ref = 4
    verbose=false
    # function to be approximated
    function my_sol(x::Matrix{Float64})
        ω = 1.0
        y = prod(sin.(ω * x), dims=2)
        return [vec(y)]
    end
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

    function build_polar_spline_space_and_geometry(deg, nel_r, nel_θ, R; n_ref = 0, form_rank = 0)
        # patches in r and θ
        patch_r = Mantis.Mesh.Patch1D(collect(LinRange(0.0, 1.0, nel_r+1)))
        patch_θ = Mantis.Mesh.Patch1D(collect(LinRange(0.0, 1.0, nel_θ+1)))

        # spline space in r
        Br = Mantis.FunctionSpaces.BSplineSpace(patch_r, deg, deg-1)
        # spline space in θ
        Bθ = Mantis.FunctionSpaces.BSplineSpace(patch_θ, deg, deg-1)
        GBθ = Mantis.FunctionSpaces.GTBSplineSpace((Bθ,), [deg-1])

        # number of basis functions
        n_r = Mantis.FunctionSpaces.get_num_basis(Br)
        n_θ = Mantis.FunctionSpaces.get_num_basis(GBθ)

        # control points for a degenerate tensor-product mapping
        geom_coeffs_tp, _, _ = Mantis.FunctionSpaces.build_standard_degenerate_control_points(n_θ, n_r, R)
        
        # refine space if needed
        for _ = 1:n_ref
            # refine the univariate spaces
            ts_r, Br_ref = Mantis.FunctionSpaces.build_two_scale_operator(Br, 2)
            _, Bθ_ref = Mantis.FunctionSpaces.build_two_scale_operator(Bθ, 2)
            GBθ_ref = Mantis.FunctionSpaces.GTBSplineSpace((Bθ_ref,), [deg-1])
            ts_θ, _ = Mantis.FunctionSpaces.build_two_scale_operator(GBθ,GBθ_ref,((2,),))
            # refine the tensor-product control points
            geom_coeffs_tp = cat(ts_θ.global_subdiv_matrix * geom_coeffs_tp[:,:,1] * ts_r.global_subdiv_matrix', ts_θ.global_subdiv_matrix * geom_coeffs_tp[:,:,2] * ts_r.global_subdiv_matrix'; dims=3)
            
            # update old spaces for the next iteration
            Br = Br_ref
            Bθ = Bθ_ref
            GBθ = GBθ_ref
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

        return X, ○
    end

    function L2_projection(X, ∫, fₑ)
        # inputs for the mixed weak form
        weak_form_inputs = Mantis.Assemblers.WeakFormInputs(fₑ, X, X, ∫)
        
        # Dirichlet boundary conditions
        dirichlet_bcs = Dict{Int, Float64}()

        # assemble all matrices
        weak_form = Mantis.Assemblers.l2_weak_form
        A, b = Mantis.Assemblers.assemble(weak_form, weak_form_inputs, dirichlet_bcs)
        
        # global_assembler = Mantis.Assemblers.Assembler(Dict{Int, Float64}())
        # A, b = global_assembler(weak_form, weak_form_inputs)

        # solve for coefficients of solution
        sol = A \ b

        # create solution as forms and return
        uₕ = Mantis.Forms.FormField(X, "u")
        uₕ.coefficients .= sol
        
        return uₕ
    end

    errors = zeros(n_ref+1)
    for r = 0:n_ref
        if verbose
            println("Building polar splines for refinement level $r...")
        end
        X, ○ = build_polar_spline_space_and_geometry(deg, nel_r, nel_θ, R; n_ref = r, form_rank = form_rank)
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
        # visualize_solution((uₕ, uₑ-uₕ), ("uh", "error"), "k_form_L2_projection_polar_$r", ○, 1, 4)
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

for deg in 2:3
    for form_rank in [0, 2]
        run_polar_spline_L2_projection_test(deg, form_rank)
    end
end