
struct CompositeGeometry{n, G1, G2} <: AbstractGeometry{n}
    Φ_1::G1
    Φ_2::G2    
    n_elements::Int

    function CompositeGeometry(Φ_1::G1, Φ_2::G2) where {G1<:AbstractGeometry{n} where {n}, G2<:AbstractGeometry{k} where {k}}
        # Φ = Φ_2  ̊ Φ_1
        @assert get_image_dim(Φ_1) == k "Dimension mismatch."
        n_elements = Φ_1.n_elements  # the number of elements need to be the same
        return new{n}(Φ_1, Φ_2, n_elements)
    end
end

function get_domain_dim(_::CompositeGeometry{n, G1, G2}) where {G1<:AbstractGeometry{n} where {n}, G2<:AbstractGeometry{k} where {k}}
    return n
end

function get_image_dim(geometry::CompositeGeometry{n, G1, G2}) where {G1<:AbstractGeometry{n} where {n}, G2<:AbstractGeometry{k} where {k}}
    return get_image_dim(geometry.Φ_2)
end


function evaluate(geometry::CompositeGeometry{n, G1, G2}, element_idx::Int, ξ::Vector{Float64}) where {G1<:AbstractGeometry{n} where {n}, G2<:AbstractGeometry{k} where {k}}
    # We only need to evaluate the last geometry because it is required that Φ₂ is defined as
    # Φ₂ = Φ̃₂ ̊Φ₁⁻¹
    # with
    # Φ₂: Ω₁ ↦ Ω₂
    # Φ₁: Ω₀ ↦ Ω₁
    # Φ̃₂: Ω₀ ↦ Ω₂  
    x = evaluate(geometry.Φ_2, element_idx, ξ)
    return x
end

# I would rename this to: ∂Φ_∂ξ
function jacobian_in_canonical_coordinates(geometry::CompositeGeometry{n, G1, G2}, element_idx::Int, ξ::Vector{Float64}) where {G1<:AbstractGeometry{n} where {n}, G2<:AbstractGeometry{k} where {k}}
    return jacobian_in_canonical_coordinates(geometry.Φ_2, element_idx, ξ)  # We only need the final geometry because it already accounts 
                                                                            # for the whole transformation because we require that geometries
                                                                            # share the same base geometry to be composable
end
