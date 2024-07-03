import Mantis

using LinearAlgebra
using Test

function fe_run(source_function, trial_space, test_space, geom, q_nodes, 
    q_weights, exact_sol, p, k, case, n, bc_dirichlet = Dict{Int, Float64}(), output_to_file=false, test=true, 
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
        output_filename = "L2-Projection-p$p-k$k-nels$nels-case-"*case*".vtu"
        output_filename_error = "L2-Projection-Error-p$p-k$k-nels$nels-case-"*case*".vtu"
        field = Mantis.Fields.FEMField(trial_space, sol_rsh)

        output_file = joinpath(output_data_folder, output_filename)
        output_file_error = joinpath(output_data_folder, output_filename_error)
        Mantis.Plot.plot(geom, field; vtk_filename = output_file, n_subcells = 1, degree = out_deg, ascii = false, compress = false)
        Mantis.Plot.plot(geom, field, exact_sol; vtk_filename = output_file_error, n_subcells = 1, degree = out_deg, ascii = false, compress = false)
    end

    # Compute error
    if verbose
        println("Computing L^2 error w.r.t. exact solution ...")
    end
    err_assembler = Mantis.Assemblers.AssemblerError(q_nodes, q_weights)
    err = err_assembler(trial_space, sol_rsh, geom, exact_sol)
    if verbose
        println("The L^2 error is: ",err)
    end

    # Extra check to test if the metric computation was correct.
    # err2 = err_assembler(trial_space, reshape(ones(length(sol)-1), :, 1), geom, (x,y) -> 0.0)
    # println(err2)

    return err
end

source_function(x, y) = exp(-y^3)*sin(x^2)
case = "exp_poly_2d"

Mantis_folder = dirname(dirname(pathof(Mantis)))
data_folder = joinpath(Mantis_folder, "test", "data")
output_data_folder = joinpath(data_folder, "output", "L2Projecton") # Create this folder first if you haven't done so yet.
output_to_file = false
test = false
verbose = false
verbose_convergence = true

nsubdivs = 4
min_p = 0
max_p = 4
base_nels = (10, 10)

for p ∈ min_p:max_p
    errors = Vector{Float64}(undef, nsubdivs+1)
    dofs = Vector{Float64}(undef, nsubdivs+1)
    for subdiv_factor ∈ 0:nsubdivs
        nels = base_nels .* 2^subdiv_factor

        patches = map(n -> Mantis.Mesh.Patch1D(collect(range(0, 1, n+1))), nels)
        bsplines = [Mantis.FunctionSpaces.BSplineSpace(patches[i], p, [-1; fill(p-1, nels[i]-1); -1]) for i ∈ 1:2]

        trial_space = Mantis.FunctionSpaces.TensorProductSpace(bsplines...)
        test_space = Mantis.FunctionSpaces.TensorProductSpace(bsplines...)
        bc = Dict{Int, Float64}()

        geom_cartesian = Mantis.Geometry.CartesianGeometry(Tuple(patches[i].breakpoints for i ∈ 1:2))

        # Setup the quadrature rule.
        q_nodes, q_weights = Mantis.Quadrature.tensor_product_rule((p,p) .+ 2, Mantis.Quadrature.gauss_legendre)
        errors[subdiv_factor+1] = fe_run(source_function, trial_space, test_space, geom_cartesian, q_nodes, q_weights, source_function, p, p-1, case, 2, bc, output_to_file, test, verbose)
        dofs[subdiv_factor+1] = Mantis.FunctionSpaces.get_dim(trial_space)
    end
    error_rates = log.(Ref(2), errors[1:end-1]./errors[2:end])
    if verbose_convergence
        println("Degree $p:")
        println("Error convergence rates:", error_rates, "\n")
    end
    if test
        @test isapprox(error_rates[end], p+1, atol=5e-2)
    end

end

