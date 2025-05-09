"""
    DirectSumSpace{manifold_dim, num_components, num_patches, F} <:
       AbstractFESpace{manifold_dim, num_components, num_patches}

A multi-component space that is the direct sum of `num_components` scalar function spaces.
Consequently, their basis functions are evaluated independently and arranged in a
block-diagonal matrix. Each scalar function space contributes to a separate component of
the multi-component space.

# Fields
- `component_spaces::F`: Tuple of `num_components` scalar function spaces
"""
struct DirectSumSpace{manifold_dim, num_components, num_patches, F} <:
       AbstractFESpace{manifold_dim, num_components, num_patches}
    component_spaces::F
    extraction_ops::NTuple{num_components, ExtractionOperator}

    function DirectSumSpace(
        component_spaces::F
    ) where {
        manifold_dim,
        num_components,
        num_patches,
        F <: NTuple{num_components, AbstractFESpace{manifold_dim, 1, num_patches}},
    }
        num_elements_per_component_space = get_num_elements.(component_spaces)
        if any(num_elements_per_component_space .!= num_elements_per_component_space[1])
            throw(ArgumentError(
                "All component spaces must have the same number of elements."
            ))
        end

        extraction_ops = extract_directsum_to_constituent(component_spaces)

        return new{manifold_dim, num_components, num_patches, F}(
            component_spaces, extraction_ops
        )
    end
end

function extract_directsum_to_constituent(
    component_spaces::NTuple{num_components, AbstractFESpace{manifold_dim, 1, num_patches}}
) where {manifold_dim, num_components, num_patches}
    # All components have the same number of elements, so we can just take the first one.
    num_elements = get_num_elements(component_spaces[1])

    # Since each component in a direct sum space only contributes to its own component, the
    # number of global d.o.f.s is the sum of the number of d.o.f.s in each component space.
    num_basis_per_component = [get_num_basis(spaces) for spaces in component_spaces]
    max_global_dof = sum(num_basis_per_component)
    basis_offset = vcat(0, cumsum(num_basis_per_component[1:(end - 1)]))

    extraction_operators = ntuple(num_components) do component_idx

        extraction_coefficients = Vector{Matrix{Float64}}(undef, num_elements)
        basis_indices = Vector{Vector{Int}}(undef, num_elements)
        for elem_id in 1:1:get_num_elements(component_spaces[component_idx])
            support_per_space = [
                get_basis_indices(component_spaces[i], elem_id) for i in 1:num_components
            ]

            num_support_per_space_offset = vcat(
                0, cumsum([length(support_per_space[i]) for i in 1:num_components-1])
            )

            basis_indices[elem_id] = reduce(
                vcat, [support_per_space[i] .+ basis_offset[i] for i in 1:num_components]
            )

            # The convention is that [constituent_spaces] * [extraction] = [MCMP].
            coeffs = zeros(length(support_per_space[component_idx]), length(basis_indices[elem_id]))
            for i in 1:length(support_per_space[component_idx])
                coeffs[i, i+num_support_per_space_offset[component_idx]] = 1.0
            end
            extraction_coefficients[elem_id] = coeffs
        end

        return ExtractionOperator(
            extraction_coefficients, basis_indices, num_elements, max_global_dof
        )
    end

    return extraction_operators
end

function get_component_spaces(space::DirectSumSpace)
    return space.component_spaces
end

function get_extraction_operators(space::DirectSumSpace)
    return space.extraction_ops
end

function _get_dof_offsets(
    space::DirectSumSpace{manifold_dim, num_components, F}
) where {manifold_dim, num_components, F}
    num_dofs_component = FunctionSpaces.get_num_basis.(get_component_spaces(space))
    dof_offset_component = zeros(Int, num_components)
    dof_offset_component[2:end] .= cumsum(num_dofs_component[1:(num_components - 1)])
    return dof_offset_component
end

function get_max_local_dim(space::DirectSumSpace)
    max_local_dim = 0
    for space in get_component_spaces(space)
        max_local_dim += get_max_local_dim(space)
    end
    return max_local_dim
end

"""
    get_component_dof_partition(space::DirectSumSpace, component_idx::Int)

Get the d.o.f. partition the component of `space` with `component_idx`. The d.o.f.s are
offsetted by the (cumulative) dimension(s) of preceding section spaces.

# Arguments
- `space::DirectSumSpace`: Direct sum space.
- `component_idx::Int`: Index of the component space.

# Returns
- `component_dof_partition::Vector{Vector{Vector{Int}}}`: D.o.f. Partition the component
    space.
"""
function get_component_dof_partition(space::DirectSumSpace, component_idx::Int)
    component_dof_partition = deepcopy(
        get_dof_partition(get_component_spaces(space)[component_idx])
    )
    dof_offset_component = _get_dof_offsets(space)[component_idx]
    for i in eachindex(component_dof_partition)
        for j in eachindex(component_dof_partition[i])
            component_dof_partition[i][j] .+= dof_offset_component
        end
    end

    return component_dof_partition
end

function get_dof_partition(
    space::DirectSumSpace{manifold_dim, num_components, F}
) where {manifold_dim, num_components, F}
    component_spaces = get_component_spaces(space)
    dof_offsets = _get_dof_offsets(space)

    dof_partition_per_component = FunctionSpaces.get_dof_partition.(component_spaces)
    dof_partition = deepcopy(dof_partition_per_component)

    for component in 1:num_components
        for patch in 1:length(dof_partition_per_component[component][1])
            dof_partition[component][1][patch] =
                dof_partition_per_component[component][1][patch] .+ dof_offsets[component]
        end
    end

    return dof_partition
end
