struct RationalSpace{n,F} <: AbstractFiniteElementSpace{n}
    function_space::F
    weights::Vector{Float64}

    function RationalSpace(function_space::F, weights::Vector{Float64}) where {F <: AbstractFiniteElementSpace{n} where {n}}
        @assert get_dim(function_space)==length(weights) "Dimension mismatch"
        n = get_n(function_space)
        new{n,F}(function_space, weights)
    end
end

function get_dim(rat_space::RationalSpace{n,F}) where {n, F <: AbstractFiniteElementSpace{n}}
    return get_dim(rat_space.function_space)
end

function get_num_elements(rat_space::RationalSpace{n,F}) where {n, F <: AbstractFiniteElementSpace{n}}
    return get_num_elements(rat_space.function_space)
end

function evaluate(rat_space::RationalSpace{n,F}, element_id::Int, xi::NTuple{n,Vector{Float64}}, nderivatives::Int) where {n, F <: AbstractFiniteElementSpace{n}}
    homog_basis, basis_indices = evaluate(rat_space.function_space, element_id, xi, nderivatives)
    for key in keys(homog_basis)
        if sum(key) == 0
            homog_basis[key] .= homog_basis[key] * LinearAlgebra.Diagonal(rat_space.weights[basis_indices])
            weight = sum(homog_basis[key],2)
            homog_basis[key] .= homog_basis[key] / LinearAlgebra.Diagonal(weight)
        else
            error("High order derivatives of rational spaces not implemented")
        end
    end
    
    return homog_basis, basis_indices
end

# dummy getters
import LinearAlgebra

function get_local_basis(rat_space::RationalSpace{n,F}, element_id::Int, xi::NTuple{n,Vector{Float64}}, nderivatives::Int) where {n, F <: AbstractFiniteElementSpace{n}}

    return evaluate(rat_space, element_id, xi, nderivatives)[1]
end

function get_extraction(rat_space::RationalSpace{n,F}, element_id::Int) where {n, F <: AbstractFiniteElementSpace{n}}

    _, basis_inds = get_extraction(rat_space.function_space, element_id)
        
    return Matrix{Float64}(LinearAlgebra.I, length(basis_inds), length(basis_inds)), basis_inds

end