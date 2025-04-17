"""
    StandardQuadrature{manifold_dim, Q} <: AbstractGlobalQuadratureRule{manifold_dim}

Represents a standard quadrature rule for a given manifold dimension, where each element has
the same canonical rule `Q`.

# Fields
- `canonical_qrule::Q`: The canonical quadrature rule used for the elements.
- `num_elements::Int`: The number of elements in the quadrature rule.

# Type parameters
- `manifold_dim`: Dimension of the domain
- `Q`: Type of the canonical quadrature rule.

# Inner Constructors
- `StandardQuadrature(canonical_qrule::Q, num_elements::Int)`: Creates a new
    `StandardQuadrature` instance with the specified canonical quadrature rule and number of
    elements.
"""
struct StandardQuadrature{manifold_dim, Q} <: AbstractGlobalQuadratureRule{manifold_dim}
    canonical_qrule::Q
    num_elements::Int

    function StandardQuadrature(
        canonical_qrule::Q, num_elements::Int
    ) where {manifold_dim, Q <: CanonicalQuadratureRule{manifold_dim}}
        return new{manifold_dim, Q}(canonical_qrule, num_elements)
    end
end

function get_canonical_rule(standard_quadrature::StandardQuadrature)
    return standard_quadrature.canonical_qrule
end

function get_parametric_nodes(standard_quadrature::StandardQuadrature)
    return get_nodes(get_canonical_rule(standard_quadrature))
end

function get_parametric_weights(standard_quadrature::StandardQuadrature)
    return get_weights(get_canonical_rule(standard_quadrature))
end

function get_label(standard_quadrature::StandardQuadrature)
    return get_label(get_canonical_rule(standard_quadrature))
end

function get_num_elements_mask(standard_quadrature::StandardQuadrature)
    return standard_quadrature.num_elements
end

function get_num_elements(standard_quadrature::StandardQuadrature)
    return standard_quadrature.num_elements
end

function get_element_idxs(::StandardQuadrature, element_idx::Int)
    return [element_idx]
end

function get_element_quadrature_rule(standard_quadrature::StandardQuadrature, ::Int)
    return get_canonical_rule(standard_quadrature)
end
