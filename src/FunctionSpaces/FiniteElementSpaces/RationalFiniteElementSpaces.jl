struct RationalFiniteElementSpace{n,F} <: AbstractFiniteElementSpace{n}
    function_space::F
    weights::Vector{Float64}

    function RationalFiniteElementSpace(function_space::F, weights::Vector{Float64}) where {F <: AbstractFiniteElementSpace{n} where {n}}
        @assert get_dim(function_space)==length(weights) "Dimension mismatch"
        n = get_n(function_space)
        new{n,F}(function_space, weights)
    end
end

function get_dim(rat_space::RationalFiniteElementSpace{n,F}) where {n, F <: AbstractFiniteElementSpace{n}}
    return get_dim(rat_space.function_space)
end

function get_num_elements(rat_space::RationalFiniteElementSpace{n,F}) where {n, F <: AbstractFiniteElementSpace{n}}
    return get_num_elements(rat_space.function_space)
end

function evaluate(rat_space::RationalFiniteElementSpace{n,F}, element_id::Int, xi::NTuple{n,Vector{Float64}}, nderivatives::Int) where {n, F <: AbstractFiniteElementSpace{n}}
    homog_basis, basis_indices = evaluate(rat_space.function_space, element_id, xi, nderivatives)
    n_eval = prod(length.(xi))
    for key in keys(homog_basis)
        if sum(key) == 0
            homog_basis[key] .= homog_basis[key] * LinearAlgebra.Diagonal(rat_space.weights[basis_indices])
            weight = reshape(sum(homog_basis[key], dims=2), n_eval)
            homog_basis[key] .= LinearAlgebra.Diagonal(weight) \ homog_basis[key]
        else
            error("High order derivatives of rational spaces not implemented")
        end
    end
    
    return homog_basis, basis_indices
end

# dummy getters
import LinearAlgebra

function get_local_basis(rat_space::RationalFiniteElementSpace{n,F}, element_id::Int, xi::NTuple{n,Vector{Float64}}, nderivatives::Int) where {n, F <: AbstractFiniteElementSpace{n}}

    return evaluate(rat_space, element_id, xi, nderivatives)[1]
end

function get_extraction(rat_space::RationalFiniteElementSpace{n,F}, element_id::Int) where {n, F <: AbstractFiniteElementSpace{n}}

    _, basis_indices = get_extraction(rat_space.function_space, element_id)
    n_supp = length(basis_indices)
        
    return Matrix{Float64}(LinearAlgebra.I, n_supp, n_supp), basis_indices

end