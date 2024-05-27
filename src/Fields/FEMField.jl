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

# evaluate in each direction at the specific points in the ntuple
function evaluate(field::FEMField{n,m}, element_id::Int, xi::NTuple{n,Vector{Float64}}) where {n,m}
    # evaluate fem space
    fem_basis, fem_basis_indices = FunctionSpaces.evaluate(field.fem_space, element_id, xi, 0)
    # combine with coefficients and return
    key = Tuple(zeros(Float64,n))
    return fem_basis[key...] * field.field_coeffs[fem_basis_indices,:]
end
