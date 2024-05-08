struct FEMField{n,m} <: AbstractField{n, m}
    field_coeffs::Array{Float64,2}
    fem_space::FunctionSpaces.AbstractFiniteElementSpace{n}
    n_elements::Int

    function FEMField(fem_space::FunctionSpaces.AbstractFiniteElementSpace{n}, field_coeffs::Array{Float64,2}) where {n}
        m = size(field_coeffs,2)
        n_elements = FunctionSpaces.get_num_elements(fem_space)
        return new{n,m}(field_coeffs, fem_space, n_elements)
    end
end

function get_num_elements(field::FEMField{n,m}) where {n,m}
    return field.n_elements
end

function get_domain_dim(::FEMField{n,m}) where {n, m}
    return n
end

function get_image_dim(::FEMField{n,m}) where {n, m}
    return m
end

# evaluate at matrix containing n-dimensional parametric points
function evaluate(field::FEMField{n,m}, element_idx::Int, ξ::Matrix{Float64}) where {n,m}
    return evaluate.(field, element_idx, ntuple(i -> ntuple(j -> [ξ[i,j]], n), size(ξ,1)))
end

# evaluate at a single n-dimensional parametric point
function evaluate(field::FEMField{n,m}, element_idx::Int, ξ::Vector{Float64}) where {n,m}
    @assert length(ξ) == n "Dimension mismatch"
    return evaluate(field, element_idx, ntuple(i -> [ξ[i]], n))
end

# evaluate at a single n-dimensional parametric point
function evaluate(field::FEMField{n,1}, element_idx::Int, ξ::Vector{Float64}) where {n}
    @assert length(ξ) == n "Dimension mismatch"
    return evaluate(field, element_idx, ntuple(i -> [ξ[i]], n))
end

function evaluate(field::FEMField{1,m}, element_idx::Int, ξ::Float64) where {m}
    return evaluate(field, element_idx, [ξ])
end

# evaluate in each direction at the specific points in the ntuple
function evaluate(field::FEMField{n,m}, element_id::Int, xi::NTuple{n,Vector{Float64}}) where {n,m}
    # evaluate fem space
    fem_basis, fem_basis_indices = FunctionSpaces.evaluate(field.fem_space, element_id, xi, 0)
    # combine with coefficients and return
    key = Tuple(zeros(Float64,n))
    return fem_basis[key...] * field.field_coeffs[fem_basis_indices,:]
end
