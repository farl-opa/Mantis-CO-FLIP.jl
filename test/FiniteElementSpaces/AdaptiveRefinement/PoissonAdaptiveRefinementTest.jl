import Mantis

using Test
using Random
using LinearAlgebra

function get_thb_geometry(hspace::Mantis.FunctionSpaces.HierarchicalFiniteElementSpace{n, S, T}) where {n, S<:Mantis.FunctionSpaces.AbstractFiniteElementSpace{n}, T<:Mantis.FunctionSpaces.AbstractTwoScaleOperator}
    L = Mantis.FunctionSpaces.get_num_levels(hspace)
    
    coefficients = Matrix{Float64}(undef, (Mantis.FunctionSpaces.get_dim(hspace), 2))

    id_sum = 1
    for level ∈ 1:1:L
        max_ind_basis = Mantis.FunctionSpaces._get_dim_per_space(hspace.spaces[level])
        x_greville_points = Mantis.FunctionSpaces.get_greville_points(hspace.spaces[level].function_space_1.knot_vector)
        y_greville_points = Mantis.FunctionSpaces.get_greville_points(hspace.spaces[level].function_space_2.knot_vector)
        grevile_mesh(x_id,y_id) = x_greville_points[x_id]*y_greville_points[y_id]
        
        _, level_active_basis = Mantis.FunctionSpaces.get_level_active(hspace.active_basis, level)

        for (y_count, y_id) ∈ enumerate(y_greville_points)
            for (x_count, x_id) ∈ enumerate(x_greville_points)
                if Mantis.FunctionSpaces.ordered_to_linear_index((x_count, y_count), max_ind_basis) ∈ level_active_basis
                    coefficients[id_sum, :] .= [x_id, y_id]
                    id_sum += 1
                end
            end
        end
    end

    return Mantis.Geometry.FEMGeometry(hspace, coefficients)
end

function fe_run(forcing_function, trial_space, test_space, geom, q_nodes, q_weights, exact_sol, p, k, step, case, n, output_to_file, bc = (false, 0.0, 0.0))
    element_assembler = Mantis.Assemblers.PoissonBilinearForm(
        forcing_function,
        trial_space,
        test_space,
        geom,
        q_nodes,
        q_weights
    )

    # Setup the global assembler.
    if bc[1] == false
        global_assembler = Mantis.Assemblers.Assembler()
    else
        global_assembler = Mantis.Assemblers.Assembler(bc[2], bc[3])
    end
    A, b = global_assembler(element_assembler)

    if n != 1
        # Add the average = 0 condition for Neumann b.c. (derivatives are 
        # assumed to be zero!)
        A = vcat(A, ones((1,size(A)[2])))
        A = hcat(A, vcat(ones(size(A)[1]-1), 0.0))
        b = vcat(b, 0.0)
    end

    if n == 1
        sol = [bc[2], (A \ Vector(b))..., bc[3]]
    else
        sol = A \ Vector(b)
    end
    if n == 1
        sol_rsh = reshape(sol, :, 1)
    else
        sol_rsh = reshape(sol[1:end-1], :, 1)
    end

    # This is for the plotting. You can visualise the solution in 
    # Paraview, using the 'Plot over line'-filter.
    if output_to_file
        msave = Mantis.Geometry.get_num_elements(geom)
        Mantis_folder = dirname(dirname(pathof(Mantis)))
        data_folder = joinpath(Mantis_folder, "test", "data")
        output_data_folder = joinpath(data_folder, "output", "Field")
        output_filename = "Poisson-step$step-$n-D-p$p-k$k-case-"*case*".vtu"
        output_file = joinpath(output_data_folder, output_filename)
        field = Mantis.Fields.FEMField(trial_space, sol_rsh)
        if n == 1
            out_deg = maximum([1, p])
        else
            out_deg = maximum([1, maximum(p)])
        end
        Mantis.Plot.plot(geom, field; vtk_filename = output_file[1:end-4], n_subcells = 1, degree = out_deg, ascii = false, compress = false)
    end
    err_assembler = Mantis.Assemblers.AssemblerErrorPerElement(q_nodes, q_weights)
    err_per_element = err_assembler(trial_space, sol_rsh, geom, exact_sol)

    return err_per_element
end

const Lleft = 0.25
const Lright = 0.75
const Lbottom = 0.25
const Ltop = 0.75

function test()
    # Dimension
    n_2d = 2
    # Test parameters
    ne1 = 10
    ne2 = 10
    breakpoints1 = collect(range(0, 1, ne1+1))
    patch1 = Mantis.Mesh.Patch1D(breakpoints1)
    breakpoints2 = collect(range(0, 1, ne2+1))
    patch2 = Mantis.Mesh.Patch1D(breakpoints2)

    deg1 = 2
    deg2 = 2
    reg1 = 1
    reg2 = 1

    B1 = Mantis.FunctionSpaces.BSplineSpace(patch1, deg1, [-1; fill(reg1, ne1-1); -1])
    B2 = Mantis.FunctionSpaces.BSplineSpace(patch2, deg2, [-1; fill(reg2, ne2-1); -1])
    TB = Mantis.FunctionSpaces.TensorProductSpace(B1, B2)

    q_nodes1, q_weights1 = Mantis.Quadrature.gauss_legendre(deg1+1)
    q_nodes2, q_weights2 = Mantis.Quadrature.gauss_legendre(deg2+1)
    q_nodes = (q_nodes1, q_nodes2)
    q_weights = Mantis.Quadrature.tensor_product_weights((q_weights1, q_weights2))

    # Domain. The length of the domain is chosen so that the normal 
    # derivatives of the exact solution are zero at the boundary. This is 
    # the only Neumann b.c. that we can specify at the moment.

    function forcing_sine(x::Float64, y::Float64)
        return 8.0 * pi^2 * sinpi(2.0 * x) * sinpi(2.0 * y)
    end

    function exact_sol_sine(x::Float64, y::Float64)
        return sinpi(2.0 * x) * sinpi(2.0 * y)
    end

    nsub1 = 2
    nsub2 = 2

    case = "sine2d-THB-adaptive-refinement"
    verbose = true
    output_to_file = false
    n_steps = 1
    dorfler_parameter = 0.3
    spaces = [TB]
    operators = Mantis.FunctionSpaces.AbstractTwoScaleOperator[]
    hspace = Mantis.FunctionSpaces.HierarchicalFiniteElementSpace(spaces, operators, [Int[]])
    hspace_geo = get_thb_geometry(hspace)
    err_per_element = fe_run(forcing_sine, hspace, hspace, hspace_geo, 
    q_nodes, q_weights, exact_sol_sine, (deg1, deg2), 
    (reg1, reg2), 0, case, n_2d, output_to_file)
    if verbose
        println("Initial data:")
        println("Polynomial degrees: $((deg1, deg2)) with regularities: $((reg1, reg2)).")
        println("Number of elements: $(Mantis.FunctionSpaces.get_num_elements(hspace)).")
        println("DoF: $(Mantis.FunctionSpaces.get_dim(hspace)). \n")
    end

    for step ∈ 1:n_steps
        # Solve current hierarchical space solution

        L = Mantis.FunctionSpaces.get_num_levels(hspace)
        new_operator, new_space = Mantis.FunctionSpaces.subdivide_bspline(hspace.spaces[L], (nsub1, nsub2))
        dorfler_marking = Mantis.FunctionSpaces.get_dorfler_marking(err_per_element, dorfler_parameter)
        marked_domains = Mantis.FunctionSpaces.get_marked_domains(hspace, dorfler_marking, new_operator, false)

        if length(marked_domains) > L
            push!(spaces, new_space)
            push!(operators, new_operator)
        end
        
        hspace = Mantis.FunctionSpaces.HierarchicalFiniteElementSpace(spaces, operators, marked_domains, true)
        hspace_geo = get_thb_geometry(hspace)
        err_per_element = fe_run(forcing_sine, hspace, hspace, hspace_geo, 
        q_nodes, q_weights, exact_sol_sine, (deg1, deg2), 
        (reg1, reg2), step, case, n_2d, output_to_file)

        if verbose
            println("Step $step") 
            println("Maximum error: $(maximum(err_per_element)).") 
            println("Number of marked_elements: $(length(dorfler_marking)).")

            println("Number of elements: $(Mantis.FunctionSpaces.get_num_elements(hspace)).")
            println("DoF: $(Mantis.FunctionSpaces.get_dim(hspace)). \n")
        end
    end
end

test()