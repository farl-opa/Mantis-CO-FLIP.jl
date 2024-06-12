import Mantis

using Test
using Plots
using ColorSchemes
using Random
using LinearAlgebra

function plot_grid(ne1, ne2, marked_elements, marked_bsplines, refinement_domain, hspace)
    # Create a grid of zeros (white background)
    grid = fill(0, ne1, ne2)
    pal = Plots.palette(:tab10)

    # Mark specific elements
    els_per_dim = Mantis.FunctionSpaces._get_num_elements_per_space(hspace.spaces[2])
    for idx in refinement_domain
        row, col = Mantis.FunctionSpaces.linear_to_ordered_index(idx, els_per_dim)
        grid[row, col] = 1
    end
    for idx in marked_bsplines
        row, col = Mantis.FunctionSpaces.linear_to_ordered_index(idx, els_per_dim)
        grid[row, col] = 2
    end
    for idx in marked_elements
        row, col = Mantis.FunctionSpaces.linear_to_ordered_index(idx, els_per_dim)
        grid[row, col] = -1  
    end

    # Plot the grid with background
    custom_gradient = cgrad([:orange, :white, :red, pal[1]])
    plt = heatmap(grid', clim=(-1,2), st=:heatmap, color=custom_gradient, legend=false, aspect_ratio=:equal, framestyle=:box, fmt=:pdf)

    # Add grid lines for better visualization
    for i in 1:ne1+1
        vline!([i-0.5], color=:black, linewidth=0.5)
    end
    for i in 1:ne2+1
        hline!([i-0.5], color=:black, linewidth=0.5)
    end
    xlims!(0.5,ne1+0.5)
    ylims!(0.5,ne2+0.5)

    # Show the plot
    display(plt)
    #savefig("many-marked.pdf")
end

function get_geometry(hspace::Mantis.FunctionSpaces.HierarchicalFiniteElementSpace{n}) where {n}
    degrees = Mantis.FunctionSpaces.get_polynomial_degree_per_dim(hspace.spaces[1])
    nxi_per_dim = maximum(degrees) + 1
    nxi = nxi_per_dim^2
    xi_per_dim = collect(range(0,1, nxi_per_dim))
    xi = Matrix{Float64}(undef, nxi,2)

    xi_eval = (xi_per_dim, xi_per_dim)

    for (idx,x) ∈ enumerate(Iterators.product(xi_per_dim, xi_per_dim))
        xi[idx,:] = [x[1] x[2]]
    end

    xs = Matrix{Float64}(undef, Mantis.FunctionSpaces.get_num_elements(hspace)*nxi,2)
    nx = size(xs)[1]

    A = zeros(nx, Mantis.FunctionSpaces.get_dim(hspace))

    for el ∈ 1:1:Mantis.FunctionSpaces.get_num_elements(hspace)
        level = Mantis.FunctionSpaces.get_active_level(hspace.active_elements, el)
        element_id = Mantis.FunctionSpaces.get_active_id(hspace.active_elements, el)

        max_ind_els = Mantis.FunctionSpaces._get_num_elements_per_space(hspace.spaces[level])
        ordered_index = Mantis.FunctionSpaces.linear_to_ordered_index(element_id, max_ind_els)

        borders_x = Mantis.Mesh.get_element(hspace.spaces[level].function_space_1.knot_vector.patch_1d, ordered_index[1])
        borders_y = Mantis.Mesh.get_element(hspace.spaces[level].function_space_2.knot_vector.patch_1d, ordered_index[2])

        x = [(borders_x[1] .+ xi[:,1] .* (borders_x[2] - borders_x[1])) (borders_y[1] .+ xi[:,2] .* (borders_y[2] - borders_y[1]))]

        idx = (el-1)*nxi+1:el*nxi
        xs[idx,:] = x
        local eval = Mantis.FunctionSpaces.evaluate(hspace, el, xi_eval, 0)
        A[idx, eval[2]] = eval[1][0,0]
    end

    coeffs = A \ xs

    hierarchical_geo = Mantis.Geometry.FEMGeometry(hspace, coeffs)

    return hierarchical_geo
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
    # Number of elements.
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

    case = "sine2d-HB-adaptive-refinement"
    output_to_file = true
    n_steps = 2
    dorfler_parameter = 0.1
    spaces = [TB]
    operators = Mantis.FunctionSpaces.AbstractTwoScaleOperator[]
    hspace = Mantis.FunctionSpaces.HierarchicalFiniteElementSpace(spaces, operators, [Int[]])
    hspace_geo = get_geometry(hspace)
    err_per_element = fe_run(forcing_sine, hspace, hspace, hspace_geo, 
    q_nodes, q_weights, exact_sol_sine, (deg1, deg2), 
    (reg1, reg2), 0, case, n_2d, output_to_file)

    test_domain = [1]
    for step ∈ 1:n_steps
        println("Step $step")
        # Solve current hierarchical space solution

        L = Mantis.FunctionSpaces.get_num_levels(hspace)
        new_operator, new_space = Mantis.FunctionSpaces.subdivide_bspline(hspace.spaces[L], (nsub1, nsub2))

        println("Maximum error is $(maximum(err_per_element))")
        dorfler_marking = Mantis.FunctionSpaces.get_dorfler_marking(err_per_element, dorfler_parameter)
        refinement_domains = Mantis.FunctionSpaces.get_refinement_domain(hspace, dorfler_marking, new_operator)
        if step == 3
            plot_grid(20,20,[117, 118, 119, 137, 138, 139, 157, 158, 159], Int[], Int[], hspace)
        end

        if length(refinement_domains) > L
            push!(spaces, new_space)
            push!(operators, new_operator)
        end
        
        hspace = Mantis.FunctionSpaces.HierarchicalFiniteElementSpace(spaces, operators, refinement_domains, false)
        hspace_geo = get_geometry(hspace)
        err_per_element = fe_run(forcing_sine, hspace, hspace, hspace_geo, 
        q_nodes, q_weights, exact_sol_sine, (deg1, deg2), 
        (reg1, reg2), step, case, n_2d, output_to_file)
    end
end

test()