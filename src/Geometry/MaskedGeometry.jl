"""
    MaskedGeometry{manifold_dim, G, M}

A masked geometry obtained by a pseudo-refinement of another geometry. The
geometry `base_geometry` is the original geometry and it is composed with the evaluation
mask `eval_mask`.

# Fields
- `base_geometry::G`: The base geometry.
- `eval_mask::EvaluationMask.AbstractEvaluationMask`: The evaluation mask.
"""
struct MaskedGeometry{manifold_dim, G, M} <: AbstractGeometry{manifold_dim}
    base_geometry::G
    eval_mask::M

    function MaskedGeometry(
        base_geometry::G, eval_mask::M
    ) where {
        manifold_dim,
        num_elements,
        num_elements_base,
        G <: AbstractGeometry{manifold_dim},
        M <: EvaluationMask.AbstractEvaluationMask{
            manifold_dim, num_elements, num_elements_base
            }
        }

        if get_num_elements(base_geometry) != EvaluationMask.get_num_elements_base(eval_mask)
            throw(ArgumentError("Number of elements in geometry and evaluation mask do not match."))
        end

        return new{manifold_dim, G, M}(base_geometry, eval_mask)
    end
end

"""
    get_num_elements(geometry::MaskedGeometry)

Get the number of elements in the geometry.

# Arguments
- `geometry::MaskedGeometry`: The masked geometry.

# Returns
- `num_elements::Int`: The number of elements in the geometry.
"""
function get_num_elements(geometry::MaskedGeometry)
    return EvaluationMask.get_num_elements(geometry.eval_mask)
end

"""
    get_image_dim(geometry::MaskedGeometry)

Get the dimension of the image of the geometry.

# Arguments
- `geometry::MaskedGeometry`: The masked geometry.

# Returns
- `image_dim::Int`: The dimension of the image.
"""
function get_image_dim(geometry::MaskedGeometry)
    return get_image_dim(geometry.base_geometry)
end

"""
    get_element_lengths(geometry::MaskedGeometry, element_idx::Int)

Get the lengths of the elements in the geometry.

# Arguments
- `geometry::MaskedGeometry`: The masked geometry.
- `element_idx::Int`: The element index.

# Returns
- `lengths::Vector{Float64}`: The lengths of the element.
"""
function get_element_lengths(geometry::MaskedGeometry, element_idx::Int)
    element_idx_base = EvaluationMask.get_base_element(geometry.eval_mask, element_idx)
    scales = EvaluationMask.get_element_length_scales(geometry.eval_mask, element_idx)
    lengths_base = get_element_lengths(geometry.base_geometry, element_idx_base)

    return lengths_base .* scales
end

"""
    get_element_measure(geometry::MaskedGeometry, element_idx::Int)

Get the measure of the element in the geometry.

# Arguments
- `geometry::MaskedGeometry`: The masked geometry.
- `element_idx::Int`: The element index.

# Returns
- `volume::Float64`: The volume of the element.
"""
function get_element_measure(geometry::MaskedGeometry, element_idx::Int)
    element_idx_base = EvaluationMask.get_base_element(geometry.eval_mask, element_idx)
    scale = EvaluationMask.get_element_measure_scale(geometry.eval_mask, element_idx)
    volume_base = get_element_measure(geometry.base_geometry, element_idx_base)
    return volume_base * scale
end

"""
    evaluate(
        geometry::MaskedGeometry,
        element_idx::Int,
        xi::NTuple{manifold_dim, Vector{Float64}}
    )

Evaluate the geometry.

# Arguments
- `geometry::MaskedGeometry`: The masked geometry.
- `element_idx::Int`: The element index in the masked geometry.
- `xi::NTuple{manifold_dim, Vector{Float64}}`: The canonical points in the masked geometry.

# Returns
- `x::Vector{Float64}`: The points in the image of the geometry.
"""
function evaluate(
    geometry::MaskedGeometry,
    element_idx::Int,
    xi::NTuple{manifold_dim, Vector{Float64}}
) where {manifold_dim}
    # map the canonical points to the base geometry
    element_idx_base, xi_base, _ = EvaluationMask.transform_evaluation_points(
        geometry.eval_mask, element_idx, xi
    )
    # evaluate the base geometry
    x = evaluate(geometry.base_geometry, element_idx_base, xi_base)

    return x
end

"""
    jacobian(
        geometry::MaskedGeometry,
        element_idx::Int,
        xi::NTuple{manifold_dim, Vector{Float64}}
    )

Evaluate the jacobian of the geometry.

# Arguments
- `geometry::MaskedGeometry`: The masked geometry.
- `element_idx::Int`: The element index in the masked geometry.
- `xi::NTuple{manifold_dim, Vector{Float64}}`: The canonical points in the masked geometry.

# Returns
- `J::Array{Float64, 3}`: The jacobian of the geometry.
"""
function jacobian(
    geometry::MaskedGeometry,
    element_idx::Int,
    xi::NTuple{manifold_dim, Vector{Float64}}
) where {manifold_dim}
    # map the canonical points to the base geometry
    element_idx_base, xi_base, scales = EvaluationMask.transform_evaluation_points(
        geometry.eval_mask, element_idx, xi
    )

    # build the evaluation mapping jacobian
    J_map = zeros(Float64, manifold_dim, manifold_dim)
    for i in 1:manifold_dim
        J_map[i, i] = scales[i]
    end

    # evaluate the base geometry
    J_base = jacobian(geometry.base_geometry, element_idx_base, xi_base)
    # scale the base geometry jacobian
    for i in axes(J_base, 1)
        J_base[i, :, :] .= J_base[i, :, :] * J_map
    end

    return J_base
end
