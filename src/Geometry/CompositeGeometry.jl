"""
    CompositeGeometry{n, m} <: AbstractGeometry{n, m}

A struct representing a composite geometry formed by the composition of two geometries.

# Fields
- `dimension::NTuple{2, Int}`: The dimensions of the domain and codomain.
- `Φ_1::AbstractGeometry`: The first geometry in the composition.
- `Φ_2::AbstractGeometry`: The second geometry in the composition.
- `n_elements::Int`: Number of elements in the geometry.

# Type parameters
- `n`: Dimension of the domain of Φ_1.
- `m`: Dimension of the codomain of Φ_2.

Note: The composition is defined as Φ = Φ_2 ∘ Φ_1, where Φ_1: Ω₀ → Ω₁ and Φ_2: Ω₁ → Ω₂.
"""
struct CompositeGeometry{n, m} <: AbstractGeometry{n, m}
    dimension::NTuple{2, Int}
    Φ_1::AbstractGeometry
    Φ_2::AbstractGeometry
    n_elements::Int

    """
        CompositeGeometry{n, m}(Φ_1::AbstractGeometry{n, k}, Φ_2::AbstractGeometry{k, m}) where {n, k, m}

    Construct a new CompositeGeometry instance with explicit type parameters.

    # Arguments
    - `Φ_1`: The first geometry in the composition.
    - `Φ_2`: The second geometry in the composition.

    # Returns
    A new CompositeGeometry instance.
    """
    function CompositeGeometry{n, m}(Φ_1::AbstractGeometry{n, k}, Φ_2::AbstractGeometry{k, m}) where {n, k, m}
        # Φ = Φ_2 ∘ Φ_1
        dimension = (Φ_1.dimension[1], Φ_2.dimension[2])
        n_elements = Φ_1.n_elements  # The number of elements must be the same for both geometries
        return new{n, m}(dimension, Φ_1, Φ_2, n_elements)
    end

    """
        CompositeGeometry(Φ_1::AbstractGeometry{n, k}, Φ_2::AbstractGeometry{k, m}) where {n, k, m}

    Construct a new CompositeGeometry instance with inferred type parameters.

    # Arguments
    - `Φ_1`: The first geometry in the composition.
    - `Φ_2`: The second geometry in the composition.

    # Returns
    A new CompositeGeometry instance.
    """
    function CompositeGeometry(Φ_1::AbstractGeometry{n, k}, Φ_2::AbstractGeometry{k, m}) where {n, k, m}
        # Φ = Φ_2 ∘ Φ_1
        dimension = (Φ_1.dimension[1], Φ_2.dimension[2])
        n_elements = Φ_1.n_elements  # The number of elements must be the same for both geometries
        return new{n, m}(dimension, Φ_1, Φ_2, n_elements)
    end
end

"""
    evaluate(geometry::CompositeGeometry{n, m}, element_idx::Int, ξ::Vector{Float64}) where {n, m}

Evaluate the CompositeGeometry at a specific point in the reference element.

# Arguments
- `geometry`: The CompositeGeometry instance.
- `element_idx`: The index of the element to evaluate.
- `ξ`: A vector representing the evaluation point in the reference element.

# Returns
The evaluated point after applying the composite geometry transformation.

Note: We only need to evaluate the last geometry (Φ_2) because it is required that
Φ₂ = Φ̃₂ ∘ Φ₁⁻¹, where:
- Φ₂: Ω₁ → Ω₂
- Φ₁: Ω₀ → Ω₁
- Φ̃₂: Ω₀ → Ω₂
"""
function evaluate(geometry::CompositeGeometry{n, m}, element_idx::Int, ξ::Vector{Float64}) where {n, m}
    # We only need to evaluate the last geometry (Φ_2)
    x = evaluate(geometry.Φ_2, element_idx, ξ)
    return x
end

"""
    jacobian_in_canonical_coordinates(geometry::CompositeGeometry{n, m}, element_idx::Int, ξ::Vector{Float64}) where {n, m}

Compute the Jacobian of the CompositeGeometry in canonical coordinates.

# Arguments
- `geometry`: The CompositeGeometry instance.
- `element_idx`: The index of the element to evaluate.
- `ξ`: A vector representing the evaluation point in the reference element.

# Returns
The Jacobian matrix at the specified point.

Note: We only need the Jacobian of the final geometry (Φ_2) because it already accounts 
for the whole transformation, as we require that geometries share the same base geometry 
to be composable.
"""
function jacobian_in_canonical_coordinates(geometry::CompositeGeometry{n, m}, element_idx::Int, ξ::Vector{Float64}) where {n, m}
    # We only need the Jacobian of the final geometry (Φ_2)
    return jacobian_in_canonical_coordinates(geometry.Φ_2, element_idx, ξ)
end
