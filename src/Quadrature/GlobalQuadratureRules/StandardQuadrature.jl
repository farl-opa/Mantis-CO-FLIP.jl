struct StandardQuadrature{manifold_dim, Q} <: AbstractGlobalQuadratureRule{manifold_dim}
    canonical_qrule::Q
    num_elements::Int

    function StandardQuadrature(
        canonical_qrule::Q,
        num_elements::Int
    ) where {manifold_dim, Q <: CanonicalQuadratureRule{manifold_dim}}
        return new{manifold_dim, Q}(canonical_qrule, num_elements)
    end
end

function get_parametric_nodes(standard_quadrature::StandardQuadrature)
    return get_nodes(standard_quadrature.canonical_qrule)
end

function get_parametric_weights(standard_quadrature::StandardQuadrature)
    return get_weights(standard_quadrature.canonical_qrule)
end

function get_label(standard_quadrature::StandardQuadrature)
    return get_label(standard_quadrature.canonical_qrule)
end

function get_num_elements_mask(standard_quadrature::StandardQuadrature)
    return standard_quadrature.num_elements
end

function get_num_elements(standard_quadrature::StandardQuadrature)
    return standard_quadrature.num_elements
end

function get_element_idxs(
    _::StandardQuadrature,
    element_idx::Int
)
    return [element_idx]
end

function get_element_quadrature_rule(
    standard_quadrature::StandardQuadrature,
    _::Int
)
    return standard_quadrature.canonical_qrule
end
