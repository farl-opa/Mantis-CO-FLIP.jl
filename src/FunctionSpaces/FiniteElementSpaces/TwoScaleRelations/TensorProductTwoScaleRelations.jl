"""
    struct TensorProductTwoScaleOperator{manifold_dim, TP, TS} <: AbstractTwoScaleOperator{manifold_dim}

A structure representing a two-scale operator for tensor product spaces, defining the relationships between coarse and fine tensor product spaces.

# Fields
- `coarse_space::TP`: The coarse tensor product space.
- `fine_space::TP`: The fine tensor product space.
- `global_subdiv_matrix::SparseArrays.SparseMatrixCSC{Float64, Int}`: The global subdivision matrix for the tensor product space.
- `twoscale_operators::TS`: A tuple of two-scale operators for each constituent space.
"""
struct TensorProductTwoScaleOperator{manifold_dim, TP, TS} <: AbstractTwoScaleOperator{manifold_dim}
    coarse_space::TP
    fine_space::TP
    global_subdiv_matrix::SparseArrays.SparseMatrixCSC{Float64, Int}
    twoscale_operators::TS

    function TensorProductTwoScaleOperator(coarse_space::TensorProductSpace{manifold_dim, T}, fine_space::TensorProductSpace{manifold_dim, T}, twoscale_operators::TS) where {manifold_dim, num_spaces, T <: NTuple{num_spaces, AbstractFiniteElementSpace}, TS <: NTuple{num_spaces, AbstractTwoScaleOperator}}
        #=
        # Does not work because equality of structures is not straightforward
        
        for space_id ∈ 1:num_spaces
            if get_space(coarse_space, space_id) != twoscale_operators[space_id].coarse_space
                throw(ArgumentError("The coarse space in the two-scale operator does not match the space in the coarse tensor product space at space index $space_id."))
            elseif get_space(fine_space, space_id) != twoscale_operators[space_id].fine_space
                throw(ArgumentError("The fine space in the two-scale operator does not match the space in the fine tensor product space at space index $space_id."))
            end
        end
        =#
        gm = kron([twoscale_operators[i].global_subdiv_matrix for i in 1:num_spaces]...)

        new{manifold_dim, typeof(coarse_space), TS}(coarse_space, fine_space, gm, twoscale_operators)
    end
end

# Basic getters per two scale operator

function _get_element_children_per_space(twoscale_operator::TensorProductTwoScaleOperator, element_id::Int)
    max_ind_c = _get_num_elements_per_space(get_coarse_space(twoscale_operator))
    ordered_index = linear_to_ordered_index(element_id, max_ind_c)

    return map(get_element_children, twoscale_operator.twoscale_operators, ordered_index)
end

function _get_element_parent_per_space(twoscale_operator::TensorProductTwoScaleOperator, element_id::Int)
    max_ind_f = _get_num_elements_per_space(get_fine_space(twoscale_operator))
    ordered_index = linear_to_ordered_index(element_id, max_ind_f)
    return map(get_element_parent, twoscale_operator.twoscale_operators, ordered_index)
end

function _get_basis_children_per_space(twoscale_operator::TensorProductTwoScaleOperator, basis_id::Int)
    max_ind_c = _get_num_basis_per_space(get_coarse_space(twoscale_operator))
    ordered_index = linear_to_ordered_index(basis_id, max_ind_c)

    return map(get_basis_children, twoscale_operator.twoscale_operators, ordered_index)
end

function _get_basis_parents_per_space(twoscale_operator::TensorProductTwoScaleOperator, basis_id::Int)
    max_ind_f = _get_num_basis_per_space(get_fine_space(twoscale_operator))
    ordered_index = linear_to_ordered_index(basis_id, max_ind_f)

    return map(get_basis_parents, twoscale_operator.twoscale_operators, ordered_index)
end

# Basis getters for TensorProductTwoScaleOperator

"""
    get_element_children(twoscale_operator::TensorProductTwoScaleOperator, element_id::Int)

Retrieve and return the child element IDs for a given element ID within a tensor product two-scale operator.

# Arguments
- `twoscale_operator::TensorProductTwoScaleOperator`: The tensor product two-scale operator that defines the parent-child relationships between elements.
- `element_id::Int`: The identifier of the element whose children are to be retrieved.

# Returns
- `::Vector{Int}`: A vector containing the identifiers of the child elements.
"""
function get_element_children(twoscale_operator::TensorProductTwoScaleOperator, element_id::Int)
    element_children_per_space = _get_element_children_per_space(twoscale_operator, element_id)

    max_ind_f = _get_num_elements_per_space(get_fine_space(twoscale_operator))

    element_children = Vector{Int}(undef, prod(length.(element_children_per_space)))
    linear_index = 1
    for child_per_space ∈ Iterators.product(element_children_per_space...)
        element_children[linear_index] = ordered_to_linear_index(child_per_space, max_ind_f)
        linear_index += 1
    end

    return element_children

end

"""
    get_element_parent(twoscale_operator::TensorProductTwoScaleOperator, element_id::Int)

Retrieve and return the parent element ID for a given element ID within a tensor product two-scale operator.

# Arguments
- `twoscale_operator::TensorProductTwoScaleOperator`: The tensor product two-scale operator that defines the parent-child relationships between elements.
- `element_id::Int`: The identifier of the element whose parent is to be retrieved.

# Returns
- `::Int`: The identifier of the parent element.
"""
function get_element_parent(twoscale_operator::TensorProductTwoScaleOperator, element_id::Int)
    element_parent_per_space = _get_element_parent_per_space(twoscale_operator, element_id)

    max_ind_c = _get_num_elements_per_space(get_coarse_space(twoscale_operator))

    return ordered_to_linear_index(element_parent_per_space, max_ind_c)
end

"""
    get_basis_children(twoscale_operator::TensorProductTwoScaleOperator, basis_id::Int)

Retrieve and return the child basis function IDs for a given basis function ID within a tensor product two-scale operator.

# Arguments
- `twoscale_operator::TensorProductTwoScaleOperator`: The tensor product two-scale operator that defines the parent-child relationships between basis functions.
- `basis_id::Int`: The identifier of the basis function whose children are to be retrieved.

# Returns
- `::Vector{Int}`: A vector containing the identifiers of the child basis functions.
"""
function get_basis_children(twoscale_operator::TensorProductTwoScaleOperator, basis_id::Int)
    basis_children_per_space = _get_basis_children_per_space(twoscale_operator, basis_id)

    max_ind_f = _get_num_basis_per_space(get_fine_space(twoscale_operator))

    basis_children = Vector{Int}(undef, prod(length.(basis_children_per_space)))
    linear_index = 1
    for child_per_space ∈ Iterators.product(basis_children_per_space...)
        basis_children[linear_index] = ordered_to_linear_index(child_per_space, max_ind_f)
        linear_index += 1
    end

    return basis_children
end

"""
    get_basis_parents(twoscale_operator::TensorProductTwoScaleOperator, basis_id::Int) -> Int

Retrieve and return the parent basis functions for a given basis function ID within a tensor product two-scale operator.

# Arguments
- `twoscale_operator::TensorProductTwoScaleOperator`: The tensor product two-scale operator that defines the parent-child relationships between basis functions.
- `basis_id::Int`: The identifier of the basis function whose parent is to be retrieved.

# Returns
- `::Vector{Int}`: The identifier of the parent basis functions.
"""
function get_basis_parents(twoscale_operator::TensorProductTwoScaleOperator, basis_id::Int)
    basis_parents_per_space = _get_basis_parents_per_space(twoscale_operator, basis_id)

    max_ind_c = _get_num_basis_per_space(get_coarse_space(twoscale_operator))

    basis_parents = Vector{Int}(undef, prod(length.(basis_parents_per_space)))
    linear_index = 1
    for parents_per_space ∈ Iterators.product(basis_parents_per_space...)
        basis_parents[linear_index] = ordered_to_linear_index(parents_per_space, max_ind_c)
        linear_index += 1
    end

    return basis_parents
end

"""
    subdivide_space(space::TensorProductSpace{manifold_dim, T}, nsubdivisions::NTuple{num_spaces, Int})

Subdivide the finite element spaces within a tensor product space and return the resulting finer tensor product space.

# Arguments
- `space::TensorProductSpace{manifold_dim, T}`: The tensor product space containing finite element spaces to be subdivided.
- `nsubdivisions::NTuple{num_spaces, Int}`: A tuple specifying the number of subdivisions for each finite element space.

# Returns
- `::TensorProductSpace`: The resulting finer tensor product space after subdivision.
"""
function subdivide_space(space::TensorProductSpace{manifold_dim, T}, nsubdivisions::NTuple{num_spaces, Int}) where {manifold_dim, num_spaces, T <: NTuple{num_spaces, AbstractFiniteElementSpace}}
    new_spaces = map(subdivide_space, space.fem_spaces, nsubdivisions)

    return TensorProductSpace(new_spaces)
end

"""
    build_two_scale_operator(space::TensorProductSpace{manifold_dim, T}, nsubdivisions::NTuple{num_spaces, Int})

Build a two-scale operator for a tensor product space by subdividing each constituent finite element space.

# Arguments
- `space::TensorProductSpace{manifold_dim, T}`: The tensor product space to be subdivided.
- `nsubdivisions::NTuple{num_spaces, Int}`: A tuple specifying the number of subdivisions for each constituent finite element space.

# Returns
- `::TensorProductTwoScaleOperator`: The two-scale operator.
- `::TensorProductSpace`: The resulting finer tensor product space after subdivision.
"""
function build_two_scale_operator(space::TensorProductSpace{manifold_dim, T}, nsubdivisions::NTuple{num_spaces, Int}) where {manifold_dim, num_spaces, T <: NTuple{num_spaces, AbstractFiniteElementSpace}}
    fine_spaces = Vector{AbstractFiniteElementSpace}(undef, num_spaces)
    twoscale_operators = Vector{AbstractTwoScaleOperator}(undef, num_spaces)
    for space_id ∈ 1:num_spaces
        twoscale_operators[space_id], fine_spaces[space_id] = build_two_scale_operator(get_space(space, space_id), nsubdivisions[space_id])
    end

    fine_tp_space = TensorProductSpace(Tuple(fine_spaces))

    return TensorProductTwoScaleOperator(space, fine_tp_space, Tuple(twoscale_operators)), fine_tp_space
end