module MCMPTests

# This file contains some simple tests for a multi-component, multi-patch space. This simple
# construction does not make use of an existing multi-component or multi-patch space in
# Mantis.

import Mantis
import LinearAlgebra
import SparseArrays

using Test

struct MultiPatchC0Space{num_patches, T} <: Mantis.FunctionSpaces.AbstractFESpace{2, 1, num_patches}
    function_spaces::T
    extraction_op::Mantis.FunctionSpaces.ExtractionOperator
    dof_partition::Vector{Vector{Vector{Int}}}
end

function _edge_number_to_dofpart_division(edge_number::Int)
    if edge_number == 1
        # Bottom edge, so vertices BL and BR, and edge B.
        return 1, 2, 3
    elseif edge_number == 2
        # Right edge, so vertices BR and TR, and edge R.
        return 3, 6, 9
    elseif edge_number == 3
        # Top edge, so vertices TL and TR, and edge T.
        return 7, 8, 9
    elseif edge_number == 4
        # Left edge, so vertices BL and TL, and edge L
        return 1, 4, 7
    end
end

function create_multi_patch_c0_space(
    function_spaces::T, patch_connectivity::NTuple{num_patches, NTuple{4, NTuple{2, Int}}}
) where {num_patches, T <: NTuple{num_patches, Mantis.FunctionSpaces.AbstractFESpace{2, 1, 1}}}
    # Compute the total number of elements and the offsets for each patch.
    num_elements = 0
    elems_per_patch = zeros(Int, num_patches)
    num_basis_per_patch = zeros(Int, num_patches)
    for patch_idx in 1:1:num_patches
        elems_on_patch = Mantis.FunctionSpaces.get_num_elements(function_spaces[patch_idx])
        elems_per_patch[patch_idx] = elems_on_patch
        num_elements += elems_on_patch

        num_basis_per_patch[patch_idx] = Mantis.FunctionSpaces.get_num_basis(function_spaces[patch_idx])
    end
    elems_per_patch_offset = vcat(0, cumsum(elems_per_patch[1:end-1]))

    # Create the dof partition, accounting for shared dofs.
    global_dof = 1
    dof_partition = Vector{Vector{Vector{Int}}}(undef, num_patches)
    global_to_local_dof_dict = Dict{Int, Dict{Int, Int}}()
    local_to_global_dof_dict = Dict{Tuple{Int, Int}, Int}()
    local_dof_partition = [Mantis.FunctionSpaces.get_dof_partition(function_spaces[patch_i])[1] for patch_i in 1:1:num_patches]
    for patch_i in 1:1:num_patches
        # Allocate the dof partition for this patch.
        dof_partition[patch_i] = Vector{Vector{Int}}(undef, 9)

        # Keep track of which dof groups on a patch have already been
        # assigned. Relevant for patch_i > 1.
        dof_groups_assigned = Vector{Int}(undef, 0)

        if patch_i == 1
            # First patch.
            for dof_group_i in eachindex(local_dof_partition[patch_i])
                dof_partition[patch_i][dof_group_i] = Vector{Int}(undef, size(local_dof_partition[patch_i][dof_group_i]))

                for (idx, local_dof_i) in pairs(local_dof_partition[patch_i][dof_group_i])
                    # Directly assign the new global dof numbers to all
                    # dofs, as nothing is shared yet.
                    dof_partition[patch_i][dof_group_i][idx] = global_dof

                    global_to_local_dof_dict[global_dof] = Dict{Int, Int}(patch_i => local_dof_i)
                    local_to_global_dof_dict[(patch_i, local_dof_partition[patch_i][dof_group_i][idx])] = global_dof
                    global_dof += 1
                end
            end

        else
            # Next patch. All numbers that are not shared with the
            # previously processed patches are inherited. The others are
            # given the already assigned number.
            for edge_i in eachindex(patch_connectivity[patch_i])

                # Get the neighbouring patch and the edge number.
                (neighbour_patch, neighbour_edge) = patch_connectivity[patch_i][edge_i]
                # Get the dof groups on the current edge.
                vertex_i1, edge_nr1, vertex_i2 = _edge_number_to_dofpart_division(edge_i)

                # Only look at actual neighbours (skipping 0) and the
                # patches that have already been processed.
                if neighbour_patch != 0 && neighbour_patch < patch_i
                    # Get the dofs on the edge that are shared.

                    # Get the dof groups on the neighbouring edge.
                    vertex_n1, edge_nr2, vertex_n2 = _edge_number_to_dofpart_division(neighbour_edge)

                    # The global partition for the current patch needs
                    # to be updated with the already assigned numbers.
                    # The order of the newly added dofs matters, but in
                    # the C0 case the ordering is the same, so no
                    # shuffle is needed.
                    dof_partition[patch_i][vertex_i1] = dof_partition[neighbour_patch][vertex_n1]

                    dof_partition[patch_i][edge_nr1] = dof_partition[neighbour_patch][edge_nr2]

                    dof_partition[patch_i][vertex_i2] = dof_partition[neighbour_patch][vertex_n2]

                    # Update the global to local dof dict. The global
                    # dofs are now already defined.
                    for dof_group_i in [vertex_i1, edge_nr1, vertex_i2]
                        for (global_dof_i, local_dof_i) in zip(dof_partition[patch_i][dof_group_i], local_dof_partition[patch_i][dof_group_i])
                            global_to_local_dof_dict[global_dof_i][patch_i] = local_dof_i
                            local_to_global_dof_dict[(patch_i, local_dof_i)] = global_dof_i
                        end
                    end

                    push!(dof_groups_assigned, vertex_i1, edge_nr1, vertex_i2)
                end
            end

            # Add the non-shared dofs.
            for dof_group_i in eachindex(local_dof_partition[patch_i])
                # Only add the dofs that have not been assigned yet.
                if !(dof_group_i in dof_groups_assigned)
                    dof_partition[patch_i][dof_group_i] = Vector{Int}(undef, size(local_dof_partition[patch_i][dof_group_i]))

                    for (idx, local_dof_i) in enumerate(local_dof_partition[patch_i][dof_group_i])
                        dof_partition[patch_i][dof_group_i][idx] = global_dof

                        # All the global dofs are unique again
                        global_to_local_dof_dict[global_dof] = Dict{Int, Int}(patch_i => local_dof_i)
                        local_to_global_dof_dict[(patch_i, local_dof_partition[patch_i][dof_group_i][idx])] = global_dof
                        global_dof += 1
                    end
                end
            end
        end
    end
    global_dof -= 1  # Last global dof that was processed.

    # Create the global extraction operator.
    extraction_coefficients = Vector{Matrix{Float64}}(undef, num_elements)
    basis_indices = Vector{Vector{Int}}(undef, num_elements)
    for patch_idx in 1:1:num_patches

        for elem_idx in 1:1:Mantis.FunctionSpaces.get_num_elements(function_spaces[patch_idx])
            global_elem_id = elems_per_patch_offset[patch_idx]+elem_idx

            # Get the local extraction coefficients and basis indices.
            extr_coeffs, indices = Mantis.FunctionSpaces.get_extraction(function_spaces[patch_idx], elem_idx)

            extraction_coefficients[global_elem_id] = Matrix(LinearAlgebra.I, size(extr_coeffs))
            basis_indices[global_elem_id] = [local_to_global_dof_dict[(patch_idx, local_dof)] for local_dof in indices]
        end
    end

    return MultiPatchC0Space{num_patches, T}(
        function_spaces,
        Mantis.FunctionSpaces.ExtractionOperator(extraction_coefficients, basis_indices, num_elements, global_dof),
        dof_partition
    )
end

function Mantis.FunctionSpaces.get_local_basis(
    space::MultiPatchC0Space,
    element_id::Int,
    xi::NTuple{2, Vector{Float64}},
    nderivatives::Int,
    component_id::Int=1,
)
    patch_id, local_element_id = Mantis.FunctionSpaces.get_patch_and_local_element_id(space, element_id)

    # Only keep the evaluations, not the indices.
    return Mantis.FunctionSpaces.evaluate(space.function_spaces[patch_id], local_element_id, xi, nderivatives)[1]
end

function Mantis.FunctionSpaces.get_num_elements_per_patch(space::MultiPatchC0Space)
    return Mantis.FunctionSpaces.get_num_elements.(space.function_spaces)
end

function extract_mcmp_to_tp(
    component_spaces::NTuple{num_components, Mantis.FunctionSpaces.AbstractFESpace},
    global_extr_ops::NTuple{num_components, SparseArrays.SparseMatrixCSC{Float64, Int}},
) where {num_components}
    # All components have the same number of elements, so we can just take the first one.
    num_elements = Mantis.FunctionSpaces.get_num_elements(component_spaces[1])

    # In this example, we set the global number of basis functions to be the same as the sum
    # of the number of basis functions in each component space.
    num_basis_per_component = [Mantis.FunctionSpaces.get_num_basis(spaces) for spaces in component_spaces]
    max_global_dof = sum(num_basis_per_component)
    basis_offset = vcat(0, cumsum(num_basis_per_component[1:(end - 1)]))

    # Convert the global extraction matrix to the local (per element) ones that the
    # ExtractionOperator expects.
    extr_ops = ntuple(num_components) do component_idx

        extraction_coefficients = Vector{Matrix{Float64}}(undef, num_elements)
        basis_indices = Vector{Vector{Int}}(undef, num_elements)
        basis_indices_all = Vector{Vector{Int}}(undef, num_elements)
        for elem_id in 1:1:Mantis.FunctionSpaces.get_num_elements(component_spaces[component_idx])
            support_per_space = [
                Mantis.FunctionSpaces.get_basis_indices(component_spaces[i], elem_id) for i in 1:num_components
            ]

            basis_indices_all[elem_id] = reduce(
                vcat, [support_per_space[i] .+ basis_offset[i] for i in 1:num_components]
            )

            # The local indices are now column indices in the global extraction coefficient
            # matrix. The non-zero row indices are the global indices supported on this
            # element. We need to find the non-zero row indices for each component space to
            # account for the most general case where the global extraction operator is not
            # a block-diagonal matrix.
            nz_row_idxs_per_component = ntuple(num_components) do i
                (nz_row_idxs_i, _, _) = SparseArrays.findnz(
                    global_extr_ops[i][:, support_per_space[i]]
                )
                return nz_row_idxs_i
            end

            # Only the unique indices are needed. unique!(sort!()) is supposed to be more
            # efficient than unique alone.
            # nz_row_idxs = unique!(sort!(reduce(vcat, nz_row_idxs_per_component)))
            nz_row_idxs = unique!(reduce(vcat, nz_row_idxs_per_component))
            basis_indices[elem_id] = nz_row_idxs

            # Add zeros for the basis functions that are supported on this element but have
            # a zero value for the current component. These cannot be found through the
            # search above, so we need to add them manually. Note that the convention is
            # that [constituent_spaces] * [extraction] = [MCMP].
            coeffs = zeros(length(support_per_space[component_idx]), length(nz_row_idxs))
            col_idxs = zeros(Int, length(nz_row_idxs_per_component[component_idx]))
            for j in eachindex(nz_row_idxs_per_component[component_idx])
                for i in eachindex(nz_row_idxs)
                    if nz_row_idxs[i] == nz_row_idxs_per_component[component_idx][j]
                        col_idxs[j] = i
                    end
                end
            end
            coeffs[:, col_idxs] = Matrix(transpose(global_extr_ops[component_idx][nz_row_idxs_per_component[component_idx], support_per_space[component_idx]]))
            extraction_coefficients[elem_id] = coeffs
        end

        return Mantis.FunctionSpaces.ExtractionOperator(
            extraction_coefficients, basis_indices, num_elements, max_global_dof
        )
    end

    return extr_ops
end

struct MCMP{T} <: Mantis.FunctionSpaces.AbstractFESpace{2, 2, 2}
    component_spaces::T
    extraction_ops::NTuple{2, Mantis.FunctionSpaces.ExtractionOperator}

    function MCMP(
        component_spaces::T,
        global_extraction_operators::NTuple{num_components, SparseArrays.SparseMatrixCSC{Float64, Int}},
    ) where {
        num_components, T <: NTuple{num_components, Mantis.FunctionSpaces.AbstractFESpace}
    }
        extraction_operators = extract_mcmp_to_tp(component_spaces, global_extraction_operators)
        new{T}(component_spaces, extraction_operators)
    end
end


breakpoints = collect(LinRange(0.0, 1.0, 5))
patch = Mantis.Mesh.Patch1D(breakpoints);
p = (3, 2)
B1 = Mantis.FunctionSpaces.BSplineSpace(patch, p[1], [-1, 2, 2, 2, -1])
B2 = Mantis.FunctionSpaces.BSplineSpace(patch, p[2], [-1, 1, 1, 1, -1])

patch_1_con = ((0,0), (2,4), (0,0), (0,0))
patch_2_con = ((0,0), (0,0), (0,0), (1,2))
mesh_con_2patch = (patch_1_con, patch_2_con)

TP1 = Mantis.FunctionSpaces.TensorProductSpace((B1, B1))
C0TP1 = create_multi_patch_c0_space((TP1, TP1), mesh_con_2patch)
TP2 = Mantis.FunctionSpaces.TensorProductSpace((B2, B2))
C0TP2 = create_multi_patch_c0_space((TP2, TP2), mesh_con_2patch)

function basic_tests(space, answers)
    # Type-based getters
    @test Mantis.FunctionSpaces.get_manifold_dim(space) == answers[1]
    @test Mantis.FunctionSpaces.get_num_components(space) == answers[2]
    @test Mantis.FunctionSpaces.get_num_patches(space) == answers[3]

    # Full-space properties
    @test Mantis.FunctionSpaces.get_component_spaces(space) == answers[4]
    @test Mantis.FunctionSpaces.get_num_elements_per_patch(space) == answers[5]
    @test Mantis.FunctionSpaces.get_num_basis(space) == answers[6]
    @test Mantis.FunctionSpaces.get_num_elements(space) == answers[7]
end

basic_tests(TP1, (2, 1, 1, (TP1,), (16,), Mantis.FunctionSpaces.get_num_basis(B1)^2, 16))
basic_tests(TP2, (2, 1, 1, (TP2,), (16,), Mantis.FunctionSpaces.get_num_basis(B2)^2, 16))
num_basis_C0TP1 = Mantis.FunctionSpaces.get_num_basis(TP1)*2 - Mantis.FunctionSpaces.get_num_basis(B1)
basic_tests(C0TP1, (2, 1, 2, (C0TP1,), (16, 16), num_basis_C0TP1, 32))
num_basis_C0TP2 = Mantis.FunctionSpaces.get_num_basis(TP2)*2 - Mantis.FunctionSpaces.get_num_basis(B2)
basic_tests(C0TP2, (2, 1, 2, (C0TP2,), (16, 16), num_basis_C0TP2, 32))



component_spaces = (C0TP1, C0TP2)
global_extr_ops = ntuple(2) do component_idx
    if component_idx == 1
        return SparseArrays.sparse(vcat(
            Matrix{Float64}(2.0*LinearAlgebra.I,
                (Mantis.FunctionSpaces.get_num_basis(component_spaces[1]),
                Mantis.FunctionSpaces.get_num_basis(component_spaces[1]))),
            Matrix{Float64}(LinearAlgebra.I,
                (Mantis.FunctionSpaces.get_num_basis(component_spaces[2]),
                Mantis.FunctionSpaces.get_num_basis(component_spaces[1])))
        ))
    else
        return SparseArrays.sparse(vcat(
            Matrix{Float64}(LinearAlgebra.I,
                (Mantis.FunctionSpaces.get_num_basis(component_spaces[1]),
                Mantis.FunctionSpaces.get_num_basis(component_spaces[2]))),
            zeros(Mantis.FunctionSpaces.get_num_basis(component_spaces[2]),
                Mantis.FunctionSpaces.get_num_basis(component_spaces[2]))
        ))
    end
end
mcmpC0 = MCMP(component_spaces, global_extr_ops)
basic_tests(mcmpC0, (2, 2, 2, (C0TP1, C0TP2), (16, 16), num_basis_C0TP1+num_basis_C0TP2, 32))

test_elem_id = 1
mcmpC0_eval, mcmpC0_ind = Mantis.FunctionSpaces.evaluate(
    mcmpC0, test_elem_id, ([0.0, 0.5, 1.0], [0.1, 0.7]), 1
)

C0TP1_eval, C0TP1_ind = Mantis.FunctionSpaces.evaluate(
    C0TP1, test_elem_id, ([0.0, 0.5, 1.0], [0.1, 0.7]), 1
)
C0TP2_eval, C0TP2_ind = Mantis.FunctionSpaces.evaluate(
    C0TP2, test_elem_id, ([0.0, 0.5, 1.0], [0.1, 0.7]), 1
)

# Verify that the evaluation of the sum space is the combination of the evaluations of the
# component spaces per component and zero elsewhere.
n_nonzero_basis_mcmpC0 = Mantis.FunctionSpaces.get_num_basis(mcmpC0, test_elem_id)
n_nonzero_basis_C0TP1 = Mantis.FunctionSpaces.get_num_basis(C0TP1, test_elem_id)
n_nonzero_basis_C0TP2 = Mantis.FunctionSpaces.get_num_basis(C0TP2, test_elem_id)

# This way of computing the answer is probably only correct on the first element.
answer1 = 2.0 .* C0TP1_eval[1][1][1]
ans1_inds = C0TP1_ind
answer2 = C0TP1_eval[1][1][1]
ans2_inds = C0TP1_ind .+ Mantis.FunctionSpaces.get_num_basis(C0TP1)
answer3 = C0TP2_eval[1][1][1]
ans3_inds = C0TP2_ind .+ Mantis.FunctionSpaces.get_num_basis(C0TP1)
answer_c1 = zeros(size(mcmpC0_eval[1][1][1]))
for i in 1:1:n_nonzero_basis_mcmpC0-4
    if mod(i, 2) == 1
        answer_c1[:, i] = answer1[:, div(i, 2) + 1]
    else
        answer_c1[:, i] = answer2[:, div(i, 2)]
    end
end
answer_c2 = zeros(size(mcmpC0_eval[1][1][2]))
for i in eachindex(mcmpC0_ind)
    if mcmpC0_ind[i] in C0TP2_ind
        answer_c2[:, i] = answer3[:, indexin([mcmpC0_ind[i]], C0TP2_ind)]
    end
end

@test all(isapprox(mcmpC0_eval[1][1][1], answer_c1, rtol=1e-14))
@test all(isapprox(mcmpC0_eval[1][1][2], answer_c2, rtol=1e-14))







# This case of MCMP should be equivalent to a DirectSumSpace.
component_spaces = (C0TP1, C0TP2)
global_extr_ops = ntuple(2) do component_idx
    if component_idx == 1
        return SparseArrays.sparse(vcat(
            Matrix{Float64}(LinearAlgebra.I,
                Mantis.FunctionSpaces.get_num_basis(component_spaces[1]),
                Mantis.FunctionSpaces.get_num_basis(component_spaces[1])),
            zeros(
                Mantis.FunctionSpaces.get_num_basis(component_spaces[2]),
                Mantis.FunctionSpaces.get_num_basis(component_spaces[1]))
        ))
    else
        return SparseArrays.sparse(vcat(
            zeros(
                Mantis.FunctionSpaces.get_num_basis(component_spaces[1]),
                Mantis.FunctionSpaces.get_num_basis(component_spaces[2])
            ),
            Matrix{Float64}(LinearAlgebra.I,
                Mantis.FunctionSpaces.get_num_basis(component_spaces[2]),
                Mantis.FunctionSpaces.get_num_basis(component_spaces[2])
            )
        ))
    end
end
mcmpC0_like_DS = MCMP(component_spaces, global_extr_ops)
basic_tests(mcmpC0_like_DS, (2, 2, 2, (C0TP1, C0TP2), (16, 16), num_basis_C0TP1+num_basis_C0TP2, 32))

DS = Mantis.FunctionSpaces.DirectSumSpace(component_spaces)

test_elem_id = 20
mcmpC0_like_DS_eval, mcmpC0_like_DS_ind = Mantis.FunctionSpaces.evaluate(
    mcmpC0_like_DS, test_elem_id, ([0.0, 0.5, 1.0], [0.1, 0.7]), 1
)
DS_eval, DS_ind = Mantis.FunctionSpaces.evaluate(
    DS, test_elem_id, ([0.0, 0.5, 1.0], [0.1, 0.7]), 1
)

@test all(isapprox(mcmpC0_like_DS_eval[1][1][1], DS_eval[1][1][1], rtol=1e-14))
@test all(isapprox(mcmpC0_like_DS_eval[1][1][2], DS_eval[1][1][2], rtol=1e-14))

end
