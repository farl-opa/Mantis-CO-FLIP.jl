import Mantis

using Test
using LinearAlgebra, Plots, SparseArrays


function fe_run(source_function, trial_space, test_space, geom, q_nodes, 
    q_weights, exact_sol, p, k, source, case, bc_dirichlet = Dict{Int, Float64}(), output_to_file=false, test=true, 
    verbose=false)
    if verbose
        println("Starting setup of problem and assembler for case "*case*" ...")
    end
    # Setup the element assembler.
    weak_form_inputs = Mantis.Assemblers.WeakFormInputs(source_function,
                                                trial_space,
                                                test_space,
                                                geom,
                                                q_nodes,
                                                q_weights)

    # Setup the global assembler.
    global_assembler = Mantis.Assemblers.Assembler(bc_dirichlet)

    if verbose
        println("Assembling ...")
    end
    A, b = global_assembler(Mantis.Assemblers.l2_weak_form, weak_form_inputs)

    if test
        if verbose
            println("Running tests ...")
        end
        @test isapprox(A, A', rtol=1e-12)  # Full system matrices need not be symmetric due to the boundary conditions.
        @test isempty(nullspace(Matrix(A)))  # Only works on dense matrices!
        @test LinearAlgebra.cond(Matrix(A)) < 1e10
    end

    # Solve & add bcs.
    if verbose
        println("Solving ",size(A,1)," x ",size(A,2)," sized system ...")
    end
    sol = A \ b
    
    sol_rsh = reshape(sol, :, 1)


    # This is for the plotting. You can visualise the solution in 
    # Paraview, using the 'Plot over line'-filter.
    if output_to_file
        if verbose
        println("Writing to file ...")
        end

        out_deg = maximum([1, maximum(p)])

        nels = Mantis.Geometry.get_num_elements(geom)
        output_filename = "L2-Projection-p$p-k$k-nels$nels-source-"*source*"-case-"*case*".vtu"
        output_filename_error = "L2-Projection-Error-p$p-k$k-nels$nels-case-"*case*".vtu"
        field = Mantis.Fields.FEMField(trial_space, sol_rsh)

        output_file = joinpath(output_data_folder, output_filename)
        output_file_error = joinpath(output_data_folder, output_filename_error)
        Mantis.Plot.plot(geom, field; vtk_filename = output_file, n_subcells = 1, degree = out_deg, ascii = false, compress = false)
        #Mantis.Plot.plot(geom, field, exact_sol; vtk_filename = output_file_error, n_subcells = 1, degree = out_deg, ascii = false, compress = false)
    end

    # Compute error
    if verbose
        println("Computing L^2 error w.r.t. exact solution ...")
    end
    err_assembler = Mantis.Assemblers.AssemblerErrorPerElement(q_nodes, q_weights)
    err_per_element, max_error = err_assembler(trial_space, sol_rsh, geom, exact_sol,"L2", true)
    if verbose
        println("The L^2 error is: ", sqrt(sum(err_per_element)))
    end

    return err_per_element, max_error, A
end

source = "gianelli2012-approximation-test"
source_function(x, y) = 2/(3*exp(sqrt((10*x-3)^2 + (10*y-3)^2))) + 2/(3*exp(sqrt((10*x)^2 + (10*y)^2))) + 2/(3*exp(sqrt((10*x+3)^2 + (10*y+3)^2)))

#=
source ="como2024-example"
source_function(x, y) = 1-tanh((sqrt((2*x-1)^2 + (2*y-1)^2) - 0.3)/(0.05*sqrt(2)))
=#

Mantis_folder = dirname(dirname(pathof(Mantis)))
data_folder = joinpath(Mantis_folder, "test", "data")
output_data_folder = joinpath(data_folder, "output", "L2Projection") # Create this folder first if you haven't done so yet.
output_to_file = false
test = false
verbose = false
verbose_step = false
verbose_convergence = false

all_cases = ["h", "HB", "THB", "THB-L-chain"]
cases = ["h", "HB", "THB", "THB-L-chain"] # Use this to choose the cases being tested.

# Parameters
nsteps = 3
hsteps = 2
dorfler_parameter = 0.3

nels = (10, 10)
p = (2, 2)
k = (1, 1)
nsubdiv = (2, 2)
hsubdiv = 2

plt = plot(xlabel="DoF", ylabel="Error", yscale=:log10, xscale=:log10)
for case ∈ cases
    if case == "h"
        h_errors = Vector{Float64}(undef, hsteps+1)
        h_dofs = Vector{Int}(undef, hsteps+1)
        q_nodes, q_weights = Mantis.Quadrature.tensor_product_rule(p .+ 2, Mantis.Quadrature.gauss_legendre)
        for subdiv_factor ∈ 0:hsteps
            curr_nels = nels .* 2^subdiv_factor
            
            patches = map(n -> Mantis.Mesh.Patch1D(collect(range(-1, 1, n+1))), curr_nels)
            bsplines = [Mantis.FunctionSpaces.BSplineSpace(patches[i], p[i], [-1; fill(k[i], curr_nels[i]-1); -1]) for i ∈ 1:2]
            
            trial_space = Mantis.FunctionSpaces.TensorProductSpace(bsplines...)
            test_space = Mantis.FunctionSpaces.TensorProductSpace(bsplines...)
            bc = Dict{Int, Float64}()
            
            geom_cartesian = Mantis.Geometry.CartesianGeometry(Tuple(patches[i].breakpoints for i ∈ 1:2))
            
            # Setup the quadrature rule.
            if subdiv_factor < hsteps
                h_errors[subdiv_factor+1] = fe_run(source_function, trial_space, test_space, geom_cartesian, q_nodes, q_weights, source_function, p, k, source, case, bc, false, test, verbose)[2]
            else
                h_errors[subdiv_factor+1] = fe_run(source_function, trial_space, test_space, geom_cartesian, q_nodes, q_weights, source_function, p, k, source, case, bc, output_to_file, test, verbose)[2]
            end
            
            h_dofs[subdiv_factor+1] = Mantis.FunctionSpaces.get_dim(trial_space)
            if verbose_step
                println("Step $subdiv_factor:")
                println("DoF: $(h_dofs[subdiv_factor+1]).")
                println("Maximum error: $(h_errors[subdiv_factor+1]). \n")
            end
        end
        plt = plot!(h_dofs, h_errors, label="h")
    end
    if case == "HB"
        patches = map(n -> Mantis.Mesh.Patch1D(collect(range(-1, 1, n+1))), nels)
        bsplines = [Mantis.FunctionSpaces.BSplineSpace(patches[i], p[i], [-1; fill(k[i], nels[i]-1); -1]) for i ∈ 1:2]
        tensor_bspline = Mantis.FunctionSpaces.TensorProductSpace(bsplines...)

        spaces = [tensor_bspline]
        operators = Mantis.FunctionSpaces.AbstractTwoScaleOperator[]
        hspace = Mantis.FunctionSpaces.HierarchicalFiniteElementSpace(spaces, operators, [Int[]])
        hspace_geo = Mantis.Geometry.compute_geometry(hspace)
        bc = Dict{Int, Float64}()
        q_nodes, q_weights = Mantis.Quadrature.tensor_product_rule(p .+ 2, Mantis.Quadrature.gauss_legendre)

        err_per_element, max_error, A = fe_run(source_function, hspace, hspace, hspace_geo, q_nodes, q_weights, source_function, p, k, source, case, bc, false, test, verbose)

        if verbose_step
            println("Step 0:")
            println("Polynomial degrees: $p with regularities: $k.")
            println("Number of elements: $(Mantis.FunctionSpaces.get_num_elements(hspace)).")
            println("DoF: $(Mantis.FunctionSpaces.get_dim(hspace)).")
            println("Maximum error: $max_error. \n")
        end

        global hb_errors = Vector{Float64}(undef, nsteps+1)
        global hb_dofs = Vector{Int}(undef, nsteps+1)
        #hb_nnz = Vector{Int}(undef, nsteps+1)

        hb_dofs[1] = Mantis.FunctionSpaces.get_dim(hspace)
        hb_errors[1] = max_error
        #SparseArrays.dropzeros!(A)
        #hb_nnz[1] = SparseArrays.nnz(A)

        for step ∈ 1:nsteps
            # Solve current hierarchical space solution

            L = Mantis.FunctionSpaces.get_num_levels(hspace)
            new_operator, new_space = Mantis.FunctionSpaces.subdivide_bspline(hspace.spaces[L], nsubdiv)

            dorfler_marking = Mantis.FunctionSpaces.get_dorfler_marking(err_per_element, dorfler_parameter)
            marked_domains = Mantis.FunctionSpaces.get_marked_domains(hspace, dorfler_marking, new_operator, false)

            if length(marked_domains) > L
                push!(spaces, new_space)
                push!(operators, new_operator)
            end
            
            hspace = Mantis.FunctionSpaces.HierarchicalFiniteElementSpace(spaces, operators, marked_domains, false)
            hspace_geo = Mantis.Geometry.compute_geometry(hspace)

            if step<nsteps 
                err_per_element, max_error, A = fe_run(source_function, hspace, hspace, hspace_geo, q_nodes, q_weights, source_function, p, k, source, case, bc, false, test, verbose)
            else
                err_per_element, max_error, A = fe_run(source_function, hspace, hspace, hspace_geo, q_nodes, q_weights, source_function, p, k, source, case, bc, output_to_file, test, verbose)
            end

            if verbose_step
                println("Step $step") 
                println("Maximum error: $(max_error).") 
                println("Number of marked_elements: $(length(dorfler_marking)).")

                println("Number of elements: $(Mantis.FunctionSpaces.get_num_elements(hspace)).")
                println("DoF: $(Mantis.FunctionSpaces.get_dim(hspace)). \n")
            end
            hb_errors[step+1] = max_error
            hb_dofs[step+1] = Mantis.FunctionSpaces.get_dim(hspace)
            #SparseArrays.dropzeros!(A)
            #hb_nnz[step+1] = SparseArrays.nnz(A)
        end
        plt = plot!(hb_dofs, hb_errors, label="HB")
    end
    if case == "THB"
        patches = map(n -> Mantis.Mesh.Patch1D(collect(range(-1, 1, n+1))), nels)
        bsplines = [Mantis.FunctionSpaces.BSplineSpace(patches[i], p[i], [-1; fill(k[i], nels[i]-1); -1]) for i ∈ 1:2]
        tensor_bspline = Mantis.FunctionSpaces.TensorProductSpace(bsplines...)

        spaces = [tensor_bspline]
        operators = Mantis.FunctionSpaces.AbstractTwoScaleOperator[]
        hspace = Mantis.FunctionSpaces.HierarchicalFiniteElementSpace(spaces, operators, [Int[]], true)
        hspace_geo = Mantis.Geometry.compute_geometry(hspace)
        bc = Dict{Int, Float64}()
        q_nodes, q_weights = Mantis.Quadrature.tensor_product_rule(p .+ 2, Mantis.Quadrature.gauss_legendre)

        err_per_element, max_error, A = fe_run(source_function, hspace, hspace, hspace_geo, q_nodes, q_weights, source_function, p, k, source, case, bc, false, test, verbose)

        if verbose_step
            println("Step 0:")
            println("Polynomial degrees: $p with regularities: $k.")
            println("Number of elements: $(Mantis.FunctionSpaces.get_num_elements(hspace)).")
            println("DoF: $(Mantis.FunctionSpaces.get_dim(hspace)).")
            println("Maximum error: $max_error. \n")
        end

        global thb_errors = Vector{Float64}(undef, nsteps+1)
        global thb_dofs = Vector{Int}(undef, nsteps+1)
        #thb_nnz = Vector{Int}(undef, nsteps+1)

        thb_dofs[1] = Mantis.FunctionSpaces.get_dim(hspace)
        thb_errors[1] = max_error
        #SparseArrays.dropzeros!(A)
        #thb_nnz[1] = SparseArrays.nnz(A)

        for step ∈ 1:nsteps
            # Solve current hierarchical space solution

            L = Mantis.FunctionSpaces.get_num_levels(hspace)
            new_operator, new_space = Mantis.FunctionSpaces.subdivide_bspline(hspace.spaces[L], nsubdiv)

            dorfler_marking = Mantis.FunctionSpaces.get_dorfler_marking(err_per_element, dorfler_parameter)
            marked_domains = Mantis.FunctionSpaces.get_marked_domains(hspace, dorfler_marking, new_operator, false)

            if length(marked_domains) > L
                push!(spaces, new_space)
                push!(operators, new_operator)
            end
            
            hspace = Mantis.FunctionSpaces.HierarchicalFiniteElementSpace(spaces, operators, marked_domains, true)
            hspace_geo = Mantis.Geometry.compute_geometry(hspace)

            if step<nsteps 
                err_per_element, max_error, A = fe_run(source_function, hspace, hspace, hspace_geo, q_nodes, q_weights, source_function, p, k, source, case, bc, false, test, verbose)
            else
                err_per_element, max_error, A = fe_run(source_function, hspace, hspace, hspace_geo, q_nodes, q_weights, source_function, p, k, source, case, bc, output_to_file, test, verbose)
            end

            if verbose_step
                println("Step $step") 
                println("Maximum error: $(max_error).") 
                println("Number of marked_elements: $(length(dorfler_marking)).")

                println("Number of elements: $(Mantis.FunctionSpaces.get_num_elements(hspace)).")
                println("DoF: $(Mantis.FunctionSpaces.get_dim(hspace)). \n")
            end
            thb_errors[step+1] = max_error
            thb_dofs[step+1] = Mantis.FunctionSpaces.get_dim(hspace)
            #SparseArrays.dropzeros!(A)
            #thb_nnz[step+1] = SparseArrays.nnz(A)
        end
        plt = plot!(thb_dofs, thb_errors, label="THB")
    end
    if case == "THB-L-chain"
        patches = map(n -> Mantis.Mesh.Patch1D(collect(range(-1, 1, n+1))), nels)
        bsplines = [Mantis.FunctionSpaces.BSplineSpace(patches[i], p[i], [-1; fill(k[i], nels[i]-1); -1]) for i ∈ 1:2]
        tensor_bspline = Mantis.FunctionSpaces.TensorProductSpace(bsplines...)
    
        spaces = [tensor_bspline]
        operators = Mantis.FunctionSpaces.AbstractTwoScaleOperator[]
        hspace = Mantis.FunctionSpaces.HierarchicalFiniteElementSpace(spaces, operators, [Int[]], true)
        hspace_geo = Mantis.Geometry.compute_geometry(hspace)
        bc = Dict{Int, Float64}()
        q_nodes, q_weights = Mantis.Quadrature.tensor_product_rule(p .+ 2, Mantis.Quadrature.gauss_legendre)

        err_per_element, max_error, A = fe_run(source_function, hspace, hspace, hspace_geo, q_nodes, q_weights, source_function, p, k, source, case, bc, false, test, verbose)

        global lchain_errors = Vector{Float64}(undef, nsteps+1)
        global lchain_dofs = Vector{Int}(undef, nsteps+1)
        #lchain_nnz = Vector{Int}(undef, nsteps+1)

        lchain_dofs[1] = Mantis.FunctionSpaces.get_dim(hspace)
        lchain_errors[1] = max_error
        #SparseArrays.dropzeros!(A)
        #lchain_nnz[1] = SparseArrays.nnz(A)
        for step ∈ 1:nsteps
            # Solve current hierarchical space solution

            L = Mantis.FunctionSpaces.get_num_levels(hspace)
            new_operator, new_space = Mantis.FunctionSpaces.subdivide_bspline(hspace.spaces[L], nsubdiv)

            dorfler_marking = Mantis.FunctionSpaces.get_dorfler_marking(err_per_element, dorfler_parameter)
            marked_domains = Mantis.FunctionSpaces.get_marked_domains(hspace, dorfler_marking, new_operator, true)

            if length(marked_domains) > L
                push!(spaces, new_space)
                push!(operators, new_operator)
            end
            
            hspace = Mantis.FunctionSpaces.HierarchicalFiniteElementSpace(spaces, operators, marked_domains, true)
            hspace_geo = Mantis.Geometry.compute_geometry(hspace)

            if step<nsteps 
                err_per_element, max_error, A = fe_run(source_function, hspace, hspace, hspace_geo, q_nodes, q_weights, source_function, p, k, source, case, bc, false, test, verbose)
            else
                err_per_element, max_error, A = fe_run(source_function, hspace, hspace, hspace_geo, q_nodes, q_weights, source_function, p, k, source, case, bc, output_to_file, test, verbose)
            end

            if verbose_step
                println("Step $step") 
                println("Maximum error: $(max_error).") 
                println("Number of marked_elements: $(length(dorfler_marking)).")
                println("Number of elements: $(Mantis.FunctionSpaces.get_num_elements(hspace)).")
                println("DoF: $(Mantis.FunctionSpaces.get_dim(hspace)). \n")
            end
            lchain_errors[step+1] = max_error
            lchain_dofs[step+1] = Mantis.FunctionSpaces.get_dim(hspace)
            #SparseArrays.dropzeros!(A)
            #lchain_nnz[step+1] = SparseArrays.nnz(A)
        end
        plt = plot!(lchain_dofs, lchain_errors, label="THB L-chain")
    end
    display(plt)
end