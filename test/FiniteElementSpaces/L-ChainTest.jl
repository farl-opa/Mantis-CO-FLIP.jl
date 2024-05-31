import Mantis

#using Test
using Plots
using ColorSchemes
using Random

function plot_grid(ne1, ne2, marked_elements, marked_bsplines, refinement_domain)
    # Create a grid of zeros (white background)
    grid = fill(0, ne1, ne2)
    pal = Plots.palette(:tab10)

    # Mark specific elements
    els_per_dim = Mantis.FunctionSpaces._get_num_elements_per_space(TB)
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
    custom_gradient = cgrad([pal[7], :white, pal[2], pal[1]])
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
    #savefig("test.pdf")
end

# Test parameters
ne1 = 40
ne2 = 15
breakpoints1 = collect(range(0, 1, ne1+1))
patch1 = Mantis.Mesh.Patch1D(breakpoints1)
breakpoints2 = collect(range(0, 1, ne2+1))
patch2 = Mantis.Mesh.Patch1D(breakpoints2)

Plots.palette(:tab10)

deg1 = 2
deg2 = 2

B1 = Mantis.FunctionSpaces.BSplineSpace(patch1, deg1, [-1; fill(deg1-1, ne1-1); -1])
B2 = Mantis.FunctionSpaces.BSplineSpace(patch2, deg2, [-1; fill(deg2-1, ne2-1); -1])

nsub1 = 2
nsub2 = 2

ts1, BF1 = Mantis.FunctionSpaces.subdivide_bspline(B1, nsub1)
ts2, BF2 = Mantis.FunctionSpaces.subdivide_bspline(B2, nsub2)

TB = Mantis.FunctionSpaces.TensorProductSpace(B1, B2)
TTS = Mantis.FunctionSpaces.TensorProductTwoScaleOperator(ts1, ts2)

# refinement_domain = Int[]
# for case ∈ 1:8
#     dim = Mantis.FunctionSpaces._get_dim_per_space(TB)

#     basis1 = (8,8) #.+ basis1_offset[case]
#     basis2 = dim .- (6 + case)

#     basis1 = Mantis.FunctionSpaces.ordered_to_linear_index(basis1, dim)
#     basis2 = Mantis.FunctionSpaces.ordered_to_linear_index(basis2, dim)

#     if Mantis.FunctionSpaces.check_problematic_intersection(TTS, basis1, basis2)
#         bsplines = Mantis.FunctionSpaces.build_L_chain(TB, basis1, basis2)
#         for bspline ∈ bsplines
#             append!(refinement_domain, Mantis.FunctionSpaces.get_support(TB, bspline))
#         end
#     else
#         append!(refinement_domain, Mantis.FunctionSpaces.get_support(TB, basis1))
#         append!(refinement_domain, Mantis.FunctionSpaces.get_support(TB, basis2))
#     end
#     plot_grid(ne1, ne2, refinement_domain)
# end

#=
dim = Mantis.FunctionSpaces._get_dim_per_space(TB)

basis1 = Mantis.FunctionSpaces.ordered_to_linear_index((4,4), dim)
basis2 = Mantis.FunctionSpaces.ordered_to_linear_index((6,7), dim)

Mantis.FunctionSpaces.check_shortest_chain(TB, basis1, basis2, [34,35,45,44,54,64,65,66])
=#

#n_marked = 15
#Random.seed!(14)
n_marked = 6
marked_elements = rand(1:Mantis.FunctionSpaces.get_num_elements(TB), n_marked)
marked_elements = [170, 331]
marked_bsplines = []
marked_basis_functions = Mantis.FunctionSpaces.basis_functions_in_marked_elements(marked_elements, TB)
for basis ∈ marked_basis_functions
    append!(marked_bsplines, Mantis.FunctionSpaces.get_support(TB, basis))
end

refinement_domain = Mantis.FunctionSpaces.get_refinement_domain(TB, marked_elements, TTS)

plot_grid(ne1, ne2, marked_elements, marked_bsplines,refinement_domain)

#basis1 = Mantis.FunctionSpaces.get_support(TB, 312)
#basis2 = Mantis.FunctionSpaces.get_support(TB, 354)
#Mantis.FunctionSpaces.check_problematic_intersection(TTS, 88, 45)
#Mantis.FunctionSpaces.check_shortest_chain(TB, 91, 106, [91,92,106])

#plot_grid(ne1, ne2, [1], append!(basis1, basis2), [1])
