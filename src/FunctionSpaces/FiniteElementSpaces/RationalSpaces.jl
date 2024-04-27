struct RationalSpace{n} <: AbstractFiniteElementSpace{n}
    function_space::AbstractFiniteElementSpace{n}
    weights::Vector{Float64}

    function RationalSpace(function_space::AbstractFiniteElementSpace{n}, weights::Vector{Float64}) where {n}
        @assert get_dim(function_space)==length(weights) "Dimension mismatch"
        new{n}(function_spaces, weights)
    end
end

function get_dim(rat_space::RationalSpace)
    return get_dim(rat_space.function_space)
end

function get_num_elements(rat_space::RationalSpace)
    return get_num_elements(rat_space.function_space)
end

function get_local_basis(rat_space::RationalSpace{n}, element_id::Int, xi::Vector{Float64}, nderivatives::Int) where {n}
    return evaluate(rat_space.function_space, element_id, xi, nderivatives)
end

function evaluate(rat_space::RationalSpace{n}, element_id::Int, xi::Vector{Float64}, nderivatives::Int) where {n}
    local_basis, basis_indices = get_local_basis(rat_space, element_id, xi, nderivatives)
    for r = 0:nderivatives
        local_basis[:,:,r+1] .= @views local_basis[:,:,r+1] * weights
    end

    return local_basis, basis_indices
end

function evaluate(rat_space::RationalSpace{1,m}, element_id::Int, xi::Float64, nderivatives::Int) where {m}
    return evaluate(rat_space, element_id, [xi], nderivatives)
end