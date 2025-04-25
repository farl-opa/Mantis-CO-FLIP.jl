"""
    DirectSumSpace{manifold_dim, num_components, num_patches, F} <:
       AbstractFESpace{manifold_dim, num_components, num_patches}

A multi-valued space that is the direct sum of `num_components` scalar function spaces.
Consequently, their basis functions are evaluated independently and arranged in a
block-diagonal matrix. Each scalar function space contributes to a separate component of
the multi-valued space.

# Fields
- `component_spaces::F`: Tuple of `num_components` scalar function spaces
"""
struct DirectSumSpace{manifold_dim, num_components, num_patches, F} <:
       AbstractFESpace{manifold_dim, num_components, num_patches}
    component_spaces::F

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

        return new{manifold_dim, num_components, num_patches, F}(component_spaces)
    end
end

get_num_basis(space::DirectSumSpace) = sum(get_num_basis.(get_component_spaces(space)))
function get_num_basis(space::DirectSumSpace, element_id::Int)
    return sum(get_num_basis.(get_component_spaces(space), element_id))
end

function get_component_spaces(
    space::DirectSumSpace{manifold_dim, 1, F}
) where {manifold_dim, F}
    return space.component_spaces
end

function get_basis_indices_w_components(
    space::DirectSumSpace{manifold_dim, num_components, F}, element_id::Int
) where {manifold_dim, num_components, F}
    component_basis_indices =
        FunctionSpaces.get_basis_indices.(get_component_spaces(space), element_id)
    dof_offset_component = _get_dof_offsets(space)

    multivalued_basis_indices = Vector{Vector{Int}}(undef, num_components)
    for component_idx in 1:num_components
        multivalued_basis_indices[component_idx] =
            component_basis_indices[component_idx] .+ dof_offset_component[component_idx]
    end

    return reduce(vcat, multivalued_basis_indices), multivalued_basis_indices
end

function get_basis_indices(space::DirectSumSpace, element_id::Int)
    return get_basis_indices_w_components(space, element_id)[1]
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
