"""
Functions and algorithms used for two scale relations.
"""

"""
    AbstractTwoScaleOperator

Supertype for all two scale relations.
"""
abstract type AbstractTwoScaleOperator end

# TwoScaleOperator structure

"""
    struct TwoScaleOperator

Two-scale operator for a change of basis between two finite element spaces.

# Fields
- `coarse_space::AbstractFiniteElementSpace`: Coarse finite element space.
- `fine_space::AbstractFiniteElementSpace`: Fine finite element space.
- `global_subdiv_matrix::SparseArrays.SparseMatrixCSC{Float64, Int64}`: Change of basis matrix.
- `local_subdiv_matrices::Vector{Matrix{Float64}}`: Local subdivision matrices.
- `coarse_to_fine_elements::Vector{Vector{Int}}`: Relation between coarser and finer elements.
- `fine_to_coarse_elements::Vector{Int}`: Relation from finer to coarser elements.
- `coarse_to_fine_functions::Vector{Vector{Int}}`: Relation between coarser and finer basis functions.
- `fine_to_coarse_functions::Vector{Vector{Int}}`: Relation from finer to coarser basis functions.
"""
struct TwoScaleOperator <: AbstractTwoScaleOperator
    coarse_space::AbstractFiniteElementSpace
    fine_space::AbstractFiniteElementSpace
    global_subdiv_matrix::SparseArrays.SparseMatrixCSC{Tv, Ti} where {Tv, Ti}
    local_subdiv_matrices::Vector{Matrix{Float64}}
    coarse_to_fine_elements::Vector{Vector{Int}}
    fine_to_coarse_elements::Vector{Int}
    coarse_to_fine_functions::Vector{Vector{Int}}
    fine_to_coarse_functions::Vector{Vector{Int}}

    function TwoScaleOperator(coarse_space::S, fine_space::T, global_subdiv_matrix::SparseArrays.SparseMatrixCSC{Tv, Ti}, local_subdiv_matrices::Vector{Matrix{Float64}}, coarse_to_fine_elements::Vector{Vector{Int}}, fine_to_coarse_elements::Vector{Int}) where {S<:AbstractFiniteElementSpace, T<:AbstractFiniteElementSpace, Tv, Ti}
        dims = size(global_subdiv_matrix)
        coarse_to_fine_functions = Vector{Vector{Int}}(undef, dims[2])
        fine_to_coarse_functions = Vector{Vector{Int}}(undef, dims[1])

        gm_data = SparseArrays.findnz(global_subdiv_matrix)
        transpose_matrix = SparseArrays.sparse(gm_data[2], gm_data[1], gm_data[3])

        for i in 1:dims[2]
            coarse_to_fine_functions[i] = global_subdiv_matrix.rowval[SparseArrays.nzrange(global_subdiv_matrix, i)]
        end
        for i in 1:dims[1]
            fine_to_coarse_functions[i] = global_subdiv_matrix.rowval[SparseArrays.nzrange(transpose_matrix, i)]
        end

        new(coarse_space, fine_space, global_subdiv_matrix, local_subdiv_matrices, coarse_to_fine_elements, fine_to_coarse_elements, coarse_to_fine_functions, fine_to_coarse_functions)
    end
end

# Getters for elements

"""
    element_ranges_to_tuple_list(finer_ranges::Vector{Vector{Int}}, nsubdivisions::NTuple{n, Int}) where {n}

Converts `finer_ranges::Vector{Vector{Int}}` to a `Vector{NTuple{n, Int}}` containing all 
permutations of numbers in `finer_ranges` as `NTuple{n, Int}`.

# Examples
julia> element_ranges_to_tuple_list([[1, 2], [3, 4]], (2,2))
[(1, 3), (2, 3), (1, 4), (2, 4)]

# Arguments
- `finer_ranges::Vector{Vector{Int}}`: The numbers to be used in the permutations.
- `nsubdivisions::NTuple{n, Int}`: The number of subdivisions in each dimension.

# Returns
- `finer_elements::Vector{NTuple{n, Int}}`: Vector containing all permutations as `NTuple{n, Int}`.
"""
function element_ranges_to_tuple_list(finer_ranges::Vector{Vector{Int}}, nsubdivisions::NTuple{n, Int}) where {n}
    finer_elements = Vector{NTuple{n, Int}}(undef, prod(nsubdivisions))

    el_count = 1
    @inbounds for el_id in Iterators.product(finer_ranges...)
        finer_elements[el_count] = el_id
        el_count += 1
    end
    
    return finer_elements
end

"""
    get_finer_elements(coarse_element_id::Int, nsubdivisions::Int)

Returns the child elements contained inside `coarse_element_id`, according to the number of
`nsubdivisions`.

# Arguments
- `coarse_element_id::Int`: The parent element where the child elements are contained.
- `nsubdivisions::Int`: The number of subdivisions.

# Returns
- `::Vector{Int}`: The element ids of the child elements.
"""
function get_finer_elements(coarse_element_id::Int, nsubdivisions::Int)
    return [get_finer_elements((coarse_element_id,), (nsubdivisions,))[i][1] for i in 1:nsubdivisions]
end

"""
    get_finer_elements(coarse_element_id::NTuple{n, Int}, nsubdivisions::NTuple{n, Int}) where {n}

Returns the child elements contained inside `coarse_element_id`, according to the number of
`nsubdivisions`.

# Arguments
- `coarse_element_id::NTuple{n, Int}`: The parent element where the child elements are contained.
- `nsubdivisions::NTuple{n, Int}`: The number of subdivisions in each dimension.

# Returns
- `::Vector{NTuple{n, Int}}`: The element ids of the child elements.
"""
function get_finer_elements(coarse_element_id::NTuple{n, Int}, nsubdivisions::NTuple{n, Int}) where {n}
    finer_ranges = Vector{Vector{Int}}(undef, n)

    for d in 1:n
        finer_ranges[d] = collect((1:nsubdivisions[d]) .+ (coarse_element_id[d]-1)*nsubdivisions[d]) 
    end

    return element_ranges_to_tuple_list(finer_ranges, nsubdivisions)
end

"""
    get_coarser_element(fine_element_id::Int, nsubdivisions::Int)

Returns the parent element where `fine_element_id` is contained.

# Arguments
- `fine_element_id::Int`: The child element where parent element containment is checked. 
- `nsubdivisions::Int`: The number of subdivisions in each dimension.

# Returns
- `::Int`: The parent element where `fine_element_id` is contained.
"""
function get_coarser_element(fine_element_id::Int, nsubdivisions::Int)
    return floor(Int, (fine_element_id-1)/nsubdivisions + 1)
end

"""
    get_coarser_element(fine_element_id::NTuple{n, Int}, nsubdivisions::NTuple{n, Int}) where {n}

Returns the parent element where `fine_element_id` is contained.

# Arguments
- `fine_element_id::NTuple{n, Int}`: The child element where parent element containment is checked. 
- `nsubdivisions::NTuple{n, Int}`: The number of subdivisions in each dimension.

# Returns
- `::NTuple{n, Int}`: The parent element where `fine_element_id` is contained.
"""
function get_coarser_element(fine_element_id::NTuple{n, Int}, nsubdivisions::NTuple{n, Int}) where {n}
    return ntuple(d -> get_coarser_element(fine_element_id[d], nsubdivisions[d]), n)
end

# Getters for TwoScaleOperator

"""
    get_coarse_space(twoscale_operator::AbstractTwoScaleOperator)

Returns the coarse finite element space of the two-scale operator.

# Arguments
- `twoscale_operator::AbstractTwoScaleOperator`: The two-scale operator.

# Returns
- `::AbstractFiniteElementSpace`: The coarse finite element space.
"""
function get_coarse_space(twoscale_operator::AbstractTwoScaleOperator) 
    return twoscale_operator.coarse_space
end

"""
    get_fine_space(twoscale_operator::AbstractTwoScaleOperator)

Returns the fine finite element space of the two-scale operator.

# Arguments
- `twoscale_operator::AbstractTwoScaleOperator`: The two-scale operator.

# Returns
- `::AbstractFiniteElementSpace`: The fine finite element space.
"""
function get_fine_space(twoscale_operator::AbstractTwoScaleOperator)
    return twoscale_operator.fine_space
end

"""
    get_coarse_spaces(twoscale_operators::NTuple{m, AbstractTwoScaleOperator}) where {m}

Returns the coarse finite element spaces of multiple two-scale operators.

# Arguments
- `twoscale_operators::NTuple{m, AbstractTwoScaleOperator}`: Tuple of two-scale operators.

# Returns
- `::NTuple{m, AbstractFiniteElementSpace}`: Tuple of coarse finite element spaces.
"""
function get_coarse_spaces(twoscale_operators::NTuple{m, AbstractTwoScaleOperator}) where {m}
    return get_coarse_space.(twoscale_operators)
end

"""
    get_fine_spaces(twoscale_operators::NTuple{m, AbstractTwoScaleOperator}) where {m}

Returns the fine finite element spaces of multiple two-scale operators.

# Arguments
- `twoscale_operators::NTuple{m, AbstractTwoScaleOperator}`: Tuple of two-scale operators.

# Returns
- `::NTuple{m, AbstractFiniteElementSpace}`: Tuple of fine finite element spaces.
"""
function get_fine_spaces(twoscale_operators::NTuple{m, AbstractTwoScaleOperator}) where {m}
    return get_fine_space.(twoscale_operators)
end

# Getters for change of basis

"""
    subdivide_coeffs(coarse_basis_coeffs::Vector{Float64}, twoscale_operator::TwoScaleOperator)

Returns the spline coefficients in a refined basis from coefficients in a coarser B-spline basis.

# Arguments
- `coarse_basis_coeffs::Vector{Float64}`: Coefficients in the coarser basis.
- `twoscale_operator::TwoScaleOperator`: Two-scale operator with the change of basis matrix.

# Returns
- `::Vector{Float64}`: Coefficients in the finer basis.
"""
function subdivide_coeffs(coarse_basis_coeffs::Vector{Float64}, twoscale_operator::TwoScaleOperator)
    return twoscale_operator.global_subdiv_matrix * coarse_basis_coeffs
end

"""
    get_local_subdiv_matrix(twoscale_operator::TwoScaleOperator, fine_el_id::Int)

Returns the local subdivision matrix necessary to represent functions from a coarser space 
in terms of finer functions on element `fine_el_id`.

# Arguments 
- `twoscale_operator::TwoScaleOperator`: Two-scale operator relating two B-spline spaces.
- `fine_el_id::Int`: Id of the fine element.

# Returns
- `::Array{Float64, 2}`: Local refinement matrix.
"""
function get_local_subdiv_matrix(twoscale_operator::TwoScaleOperator, fine_el_id::Int)
    return twoscale_operator.local_subdiv_matrices[fine_el_id]
end


"""
    get_element_basis_subdiv_matrices(gm::SparseArrays.SparseMatrixCSC{Float64, Int}, coarse_space::C, fine_space::F, coarse_el_id::Int, fine_el_id::Int) where {C<:AbstractFiniteElementSpace{1}, F<:AbstractFiniteElementSpace{1}}

Extract a submatrix from the global subdivision matrix for specific coarse and fine elements.

# Arguments
- `gm::SparseArrays.SparseMatrixCSC{Float64, Int}`: Global subdivision matrix.
- `coarse_space::C`: Coarse finite element space.
- `fine_space::F`: Fine finite element space.
- `coarse_el_id::Int`: ID of the coarse element.
- `fine_el_id::Int`: ID of the fine element.

# Returns
- `::Matrix{Float64}`: Submatrix of the global subdivision matrix for the specified elements.
"""
function get_element_basis_subdiv_matrices(gm::SparseArrays.SparseMatrixCSC{Float64, Int}, coarse_space::C, fine_space::F, coarse_el_id::Int, fine_el_id::Int) where {C<:AbstractFiniteElementSpace{1}, F<:AbstractFiniteElementSpace{1}}
    # Get row indices for the fine element
    _, rows = get_extraction(fine_space, fine_el_id)
    # Get column indices for the coarse element
    _, columns = get_extraction(coarse_space, coarse_el_id)

    # Return the submatrix view
    return @views gm[rows, columns]
end

"""
    get_finer_basis_id(twoscale_operator::TwoScaleOperator, basis_id::Int)

Returns the IDs of the child B-splines of `basis_id`, in terms of the change of basis
provided by `twoscale_operator`.

# Arguments
- `twoscale_operator::TwoScaleOperator`: The two-scale operator for the change of basis.
- `basis_id::Int`: ID of the parent B-spline.

# Returns
- `::Vector{Int}`: IDs of the child B-splines.
"""
function get_finer_basis_id(twoscale_operator::TwoScaleOperator, basis_id::Int)
    return twoscale_operator.coarse_to_fine_functions[basis_id]
end

"""
    get_finer_elements(twoscale_operator::TwoScaleOperator, el_id::Int)

Returns the finer elements corresponding to a given coarse element.

# Arguments
- `twoscale_operator::TwoScaleOperator`: The two-scale operator.
- `el_id::Int`: ID of the coarse element.

# Returns
- `::Vector{Int}`: IDs of the finer elements.
"""
function get_finer_elements(twoscale_operator::TwoScaleOperator, el_id::Int)
    return twoscale_operator.coarse_to_fine_elements[el_id]    
end

"""
    get_finer_elements(twoscale_operator::TwoScaleOperator, el_ids::Union{Vector{Int},UnitRange{Int}})

Returns the finer elements corresponding to multiple coarse elements.

# Arguments
- `twoscale_operator::TwoScaleOperator`: The two-scale operator.
- `el_ids::Union{Vector{Int},UnitRange{Int}}`: IDs of the coarse elements.

# Returns
- `::Vector{Int}`: Concatenated vector of IDs of the finer elements.
"""
function get_finer_elements(twoscale_operator::TwoScaleOperator, el_ids::Union{Vector{Int},UnitRange{Int}})
    return vcat(get_finer_elements.((twoscale_operator,), el_ids)...)  
end

"""
    get_coarser_element(twoscale_operator::TwoScaleOperator, el_id::Int)

Returns the coarser element corresponding to a given fine element.

# Arguments
- `twoscale_operator::TwoScaleOperator`: The two-scale operator.
- `el_id::Int`: ID of the fine element.

# Returns
- `::Int`: ID of the coarser element.
"""
function get_coarser_element(twoscale_operator::TwoScaleOperator, el_id::Int)
    return twoscale_operator.fine_to_coarse_elements[el_id]    
end

"""
    get_coarser_basis_id(twoscale_operator::TwoScaleOperator, basis_id::Int)

Returns the ID of the parent B-spline for a given fine basis function.

# Arguments
- `twoscale_operator::TwoScaleOperator`: The two-scale operator.
- `basis_id::Int`: ID of the fine basis function.

# Returns
- `::Int`: ID of the coarser basis function.
"""
function get_coarser_basis_id(twoscale_operator::TwoScaleOperator, basis_id::Int)
    return twoscale_operator.fine_to_coarse_functions[basis_id]
end

"""
    check_support(fe_space::F, basis_id::Int, next_level_domain::SubArray{Int64, 1, Vector{Int64}, Tuple{UnitRange{Int64}}, true}, twoscale_operator::O) where {O<:AbstractTwoScaleOperator, F<:FunctionSpaces.AbstractFiniteElementSpace{n} where {n}}

Checks if the support of a basis function is contained within a given domain at the next finer level.

# Arguments
- `fe_space::F`: The finite element space.
- `basis_id::Int`: ID of the basis function.
- `next_level_domain::SubArray{Int64, 1, Vector{Int64}, Tuple{UnitRange{Int64}}, true}`: Domain at the next finer level.
- `twoscale_operator::O`: The two-scale operator.

# Returns
- `::Tuple{Bool, UnitRange{Int}, Vector{Int}}`: A tuple containing:
  - Boolean indicating if the support is contained.
  - The support of the basis function.
  - The finer support of the basis function.
"""
function check_support(fe_space::F, basis_id::Int, next_level_domain::SubArray{Int64, 1, Vector{Int64}, Tuple{UnitRange{Int64}}, true}, twoscale_operator::O) where {O<:AbstractTwoScaleOperator, F<:FunctionSpaces.AbstractFiniteElementSpace{n} where {n}}
    basis_support = FunctionSpaces.get_support(fe_space, basis_id)
    finer_support = get_finer_support(basis_support, twoscale_operator)

    contained, _ = Mesh.check_contained(finer_support, next_level_domain)

    if contained
        return true, basis_support, finer_support
    end

    return false, basis_support, finer_support
end

"""
    get_finer_support(support::Union{Vector{Int}, UnitRange{Int}}, twoscale_operator::O) where {O<:AbstractTwoScaleOperator}

Returns the finer elements corresponding to the support of a basis function.

# Arguments
- `support::Union{Vector{Int}, UnitRange{Int}}`: The support of a basis function.
- `twoscale_operator::O`: The two-scale operator.

# Returns
- `::Vector{Int}`: The finer elements of the support.
"""
function get_finer_support(support::Union{Vector{Int}, UnitRange{Int}}, twoscale_operator::O) where {O<:AbstractTwoScaleOperator}
    return reduce(vcat, get_finer_elements(twoscale_operator, support))
end

"""
    get_finer_extraction_coeffs(space::S, two_scale_operators::Vector{O}, coarse_element::Int, coarse_level::Int, finer_element::Int, finer_level::Int) where {S<: AbstractFiniteElementSpace, O<:AbstractTwoScaleOperator}

Computes the extraction coefficients for a finer element given a coarse element and multiple levels of refinement.

# Arguments
- `space::S`: The finite element space.
- `two_scale_operators::Vector{O}`: Vector of two-scale operators for each refinement level.
- `coarse_element::Int`: ID of the coarse element.
- `coarse_level::Int`: Level of the coarse element.
- `finer_element::Int`: ID of the finer element.
- `finer_level::Int`: Level of the finer element.

# Returns
- `::Tuple{Matrix{Float64}, Vector{Int}}`: A tuple containing:
  - The refined extraction coefficients.
  - The basis function indices.
"""
function get_finer_extraction_coeffs(space::S, two_scale_operators::Vector{O}, coarse_element::Int, coarse_level::Int, finer_element::Int, finer_level::Int) where {S<: AbstractFiniteElementSpace, O<:AbstractTwoScaleOperator}
    # Initialize the extraction coeffs
    coarse_coeffs, basis_indices = get_extraction(space, coarse_element)

    refinement_matrix = LinearAlgebra.I

    current_fine = finer_element
    
    for level ∈ finer_level:-1:coarse_level+1
        next_fine = get_coarser_element(two_scale_operators[level-1], current_fine) 
        local_subdiv_matrix = get_local_subdiv_matrix(two_scale_operators[level-1], current_fine)

        refinement_matrix = refinement_matrix * local_subdiv_matrix
        
        current_fine = next_fine
    end

    return refinement_matrix * coarse_coeffs, basis_indices
end

"""
    get_finer_extraction_coeffs(space::S, two_scale_operators::Vector{O}, coarse_element::Int, coarse_level::Int, finer_element::Int, finer_level::Int, basis_id::Int) where {S<: AbstractFiniteElementSpace, O<:AbstractTwoScaleOperator}

Computes the extraction coefficients for a specific basis function in a finer element given a coarse element and multiple levels of refinement.

# Arguments
- `space::S`: The finite element space.
- `two_scale_operators::Vector{O}`: Vector of two-scale operators for each refinement level.
- `coarse_element::Int`: ID of the coarse element.
- `coarse_level::Int`: Level of the coarse element.
- `finer_element::Int`: ID of the finer element.
- `finer_level::Int`: Level of the finer element.
- `basis_id::Int`: ID of the specific basis function.

# Returns
- `::Vector{Float64}`: The refined extraction coefficients for the specified basis function.
"""
function get_finer_extraction_coeffs(space::S, two_scale_operators::Vector{O}, coarse_element::Int, coarse_level::Int, finer_element::Int, finer_level::Int, basis_id::Int) where {S<: AbstractFiniteElementSpace, O<:AbstractTwoScaleOperator}
    coarse_coeffs, basis_indices = get_finer_extraction_coeffs(space, two_scale_operators, coarse_element, coarse_level, finer_element, finer_level)
    
    basis_index = findfirst(x -> x == basis_id, basis_indices)

    return @view coarse_coeffs[:, basis_index]
end

"""
    get_coarser_element(two_scale_operators::Vector{O}, coarse_level::Int, finer_element::Int, finer_level::Int) where {O<:AbstractTwoScaleOperator}

Finds the coarser element corresponding to a given finer element across multiple levels of refinement.

# Arguments
- `two_scale_operators::Vector{O}`: Vector of two-scale operators for each refinement level.
- `coarse_level::Int`: Target coarse level.
- `finer_element::Int`: ID of the finer element.
- `finer_level::Int`: Level of the finer element.

# Returns
- `::Int`: ID of the corresponding coarser element.
"""
function get_coarser_element(two_scale_operators::Vector{O}, coarse_level::Int, finer_element::Int, finer_level::Int) where {O<:AbstractTwoScaleOperator}
    current_coarse = finer_element
    current_fine = finer_element
    for level ∈ finer_level:-1:coarse_level+1
        current_coarse = get_coarser_element(two_scale_operators[level-1], current_fine)
        current_fine = current_coarse
    end

    return current_coarse
end

# Includes for concrete two scale relations

include("UnivariateBSplineTwoScaleRelations.jl")
include("TensorProductTwoScaleRelations.jl")
