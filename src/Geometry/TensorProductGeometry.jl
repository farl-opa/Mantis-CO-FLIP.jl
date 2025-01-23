
@doc raw"""
Tensor product geometry of two geometries.

An ``(n, m)`` geometry $\Phi$ is a collection of ``L`` mappings ``\left\{\Phi_{i}\right\}_{i=1}^{L}``
that map the canonical ``n``-dimensional domain, ``\Omega^{0} := [0, 1]^{n}`` into ``L``
``n``-dimensional simply connected subdomains, ``\Omega^{1}_{i}`` with ``i = 1, \dots, L``, of ``\mathbb{R}^{m}``.
Moreover, ``\bigcap_{i=1}^{L}\Omega^{1}_{i} = \emptyset`` and
using Core: Argument
``\overline{\Omega}^{1}_{i} \cap \overline{\Omega}^{1}_{j} \subset \partial\Omega^{1}_{i} \cup \partial\Omega^{1}_{j}``
with ``i,j = 1, \dots, L``.

Note that

```math
\Phi_{i}(\xi_{1}, \dots, \xi_{n}) = (x_{1}, \dots, x_{m}),
```

and we use ``\Phi_{i, j} = x_{j}`` to mean the ``j``-th component of the mapping ``\Phi`` of element ``i``.

Given an ``(n_{1}, m_{1})`` geometry ``\Phi^{1}`` of ``L_{1}`` mappings and an ``(n_{2}, m_{2})`` geometry ``\Phi^{2}``
of ``L_{2}`` mappings, i.e.,

```math
\Phi^{1}_{i}: [0, 1]^{n_{1}} \mapsto \Omega^{1}_{i} \subset \mathbb{R}^{m_{1}}, \quad i = 1, \dots, L_{1}
```

and

```math
\Phi^{2}_{i}: [0, 1]^{n_{2}} \mapsto \Omega^{2}_{i} \subset \mathbb{R}^{m_{2}}, \quad i = 1, \dots, L_{2}
```

the tensor product geometry ``\Phi := \Phi^{1}\otimes\Phi^{2}`` is an ``(n_{1} + n_{2}, m_{1} + m_{2})`` geometry
made up of a collection of ``L_{1}L_{2}`` mappings ``\Phi_{k}``

```math
\Phi_{k = L_1(j - 1) + I}: [0, 1]^{n_{1}} \times [0, 1]^{n_{2}} \mapsto \Omega_{k} = \Omega^{1}_{i}\times\Omega^{2}_{j} \subset \mathbb{R}^{m_{1} + m_{2}}, \quad i = 1, \dots, L_{1}, \text{ and } j = 1, \dots, L{2}.
```

Specifically, we have

```math
\Phi_{L_1(j - 1) + i, l}(\xi_{1}, \dots, \xi_{n}) := \left\{
\begin{array}{ll}
\Phi^{1}_{i, l}(\xi_{1}, \dots, \xi_{n_{1}}), & \quad \text{if } l \leq n_{1}\\
\Phi^{2}_{j, l - n_{1}}(\xi_{n_{1} + 1}, \dots, \xi_{n_{1} + n_{2}}), & \quad \text{if } n_{1} < l \leq n_{1} + n_{2}
\end{array}
\right.\,,
\quad i = 1, \dots, L_{1},\quad j = 1, \dots, L_{2}, \text{ and } l = 1, \dots, m_{1} + m_{2}\,.

```

The Jacobian of this geometry

```math
J^{k}_{l,v} := \frac{\partial \Phi_{k, l}}{\partial \xi_{v}}
```

is given by

```math
\frac{\partial\Phi_{L_1(j - 1) + i, l}}{\partial\xi_{v}}(\xi_{1}, \dots, \xi_{n}) := \left\{
\begin{array}{ll}
\frac{\partial\Phi^{1}_{i, l}}{\partial \xi_{v}}(\xi_{1}, \dots, \xi_{n_{1}}), & \quad \text{if } l \leq n_{1}, \text{  and  } v \leq m_{1}\\
\frac{\partial\Phi^{2}_{j, l - n_{1}}}{\partial \xi_{u - m_{1}}}(\xi_{n_{1} + 1}, \dots, \xi_{n_{1} + n_{2}}), & \quad \text{if } n_{1} < l \leq n_{1} + n_{2}, \text{  and  } m_{1} < v \leq m_{1} + m_{2} \\
0 & \quad\text{otherwise}
\end{array}
\right.\,,
\quad i = 1, \dots, L_{1}, \quad j = 1, \dots, L_{2}, \text{ and } l = 1, \dots, m_{1} + m_{2}\,.

```

"""
struct TensorProductGeometry{manifold_dim, T} <: AbstractGeometry{manifold_dim}
    geometries::T
    num_elements::Int
    ordered_indices::CartesianIndices

    function TensorProductGeometry(geometries::T) where {
        num_geometries, T<:NTuple{num_geometries, AbstractGeometry}
    }
        num_elements_per_geometry = ntuple(num_geometries) do k
            return get_num_elements(geometries[k])
        end

        num_elements = prod(num_elements_per_geometry)
        ordered_indices = CartesianIndices(num_elements_per_geometry)
        manifold_dim_per_geometry = ntuple(num_geometries) do k
            return get_manifold_dim(geometries[k])
        end

        manifold_dim = sum(manifold_dim_per_geometry)

        return new{manifold_dim, T}(
            geometries,
            num_elements,
            ordered_indices
        )
    end
end

#Basic getters

function get_num_elements_per_geometry(
    geometry::TensorProductGeometry{manifold_dim, T}
) where {
    manifold_dim, num_geometries, T<:NTuple{num_geometries, AbstractGeometry}
}
    num_elements_per_geometry = ntuple(num_geometries) do k
        return get_num_elements(geometry.geometries[k])
    end

    return num_elements_per_geometry
end

function get_geometry(geometry::TensorProductGeometry, geometry_id::Int)
    return geometry.geometries[geometry_id]
end

get_num_elements(geometry::TensorProductGeometry) = geometry.num_elements

function get_domain_dim(::TensorProductGeometry{manifold_dim, T}) where {manifold_dim, T}
    return manifold_dim
end

function get_ordered_indices(geometry::TensorProductGeometry)
    return geometry.ordered_indices
end

function get_ordered_indices(geometry::TensorProductGeometry, element_id::Int)
    return get_ordered_indices(geometry)[element_id]
end

function _get_element_dimensions(
    geometry::TensorProductGeometry{manifold_dim, T}, element_id::Int
) where {
    manifold_dim, num_geometries, T<:NTuple{num_geometries, AbstractGeometry}
}
    ordered_index = get_ordered_indices(geometry, element_id)
    element_dimensions_per_geometry = [get_element_dimensions(
        geometry.geometries[k], ordered_index[k]
    ) for k ∈ 1:num_geometries]

    element_dimensions = Tuple(reduce(vcat, element_dimensions_per_geometry))

    return element_dimensions
end

function _get_manifold_dim_per_geometry(geometry::TensorProductGeometry{manifold_dim, T}
) where {
    manifold_dim, num_geometries, T<:NTuple{num_geometries, AbstractGeometry}
}
    manifold_dim_per_geometry = ntuple(num_geometries) do k
        return get_manifold_dim(geometry.geometries[k])
    end

    return manifold_dim_per_geometry
end

function _get_cum_manifold_dim_per_geometry(geometry::TensorProductGeometry)
    cum_manifold_dim_per_geometry = cumsum(
        (0, _get_manifold_dim_per_geometry(geometry)...)
    )

    return cum_manifold_dim_per_geometry
end

function _get_manifold_ranges(geometry::TensorProductGeometry{manifold_dim, T}
) where {
    manifold_dim, num_geometries, T<:NTuple{num_geometries, AbstractGeometry}
}
    manifold_dim_per_geometry = _get_manifold_dim_per_geometry(geometry)
    cum_manifold_dim_per_geometry = _get_cum_manifold_dim_per_geometry(geometry)

    manifold_ranges = ntuple(num_geometries) do k
        beg_idx = cum_manifold_dim_per_geometry[k]+1
        end_idx = cum_manifold_dim_per_geometry[k]+manifold_dim_per_geometry[k]

        return beg_idx:end_idx
    end

    return manifold_ranges
end

function _get_image_dim_per_geometry(geometry::TensorProductGeometry{manifold_dim, T}
) where {
    manifold_dim, num_geometries, T<:NTuple{num_geometries, AbstractGeometry}
}
    image_dim_per_geometry = ntuple(num_geometries) do k
        return get_image_dim(geometry.geometries[k])
    end

    return image_dim_per_geometry
end

function _get_cum_image_dim_per_geometry(geometry::TensorProductGeometry)
    cum_image_dim_per_geometry = cumsum(
        (0, _get_image_dim_per_geometry(geometry)...)
    )

    return cum_image_dim_per_geometry
end

function _get_image_ranges(geometry::TensorProductGeometry{manifold_dim, T}
) where {
    manifold_dim, num_geometries, T<:NTuple{num_geometries, AbstractGeometry}
}
    image_dim_per_geometry = _get_image_dim_per_geometry(geometry)
    cum_image_dim_per_geometry = _get_cum_image_dim_per_geometry(geometry)

    image_ranges = ntuple(num_geometries) do k
        beg_idx = cum_image_dim_per_geometry[k]+1
        end_idx = cum_image_dim_per_geometry[k]+image_dim_per_geometry[k]

        return beg_idx:end_idx
    end

    return image_ranges
end

function get_image_dim(geometry::TensorProductGeometry{manifold_dim, T}) where {
    manifold_dim, T
}
    return sum(_get_image_dim_per_geometry(geometry))
end

# Other methods

@doc raw"""
    evaluate(geometry::TensorProductGeometry{manifold_dim, T}, element_idx::Int, ξ::NTuple{n, Vector{Float64}})

Evaluates the tensor product geometry `geometry` at element `element_idx` and at the tensor product of the
`NTuple` coordinates `ξ`.

In other words, given the `NTuple` `ξ` of ``n`` `Vectors`, ``\boldsymbol{\xi}^{i}``, ``i=1, \dots, n``,
each containing ``m_{i}`` unidimensional coordinates ``\xi^{i}_{j}``, ``i = 1, \dots, n`` and `` j = 1, \dots m_{i}``,
evaluates the tensor product geometry at the element `element_idx` and at the ``\prod_{i=1}^{n}m_{i}`` tensor product
points ``V_{k = j_{1} + \sum_{i=2}^{n} (j_{i} - 1)\prod_{l=1}^{i-1}m_{l}} = (\xi^{1}_{j_{1}}, \dots, \xi^{n}_{j_{n}})``,
with ``j_{i} = 1, \dots, m_{i}``.

The output is a matrix, ``\boldsymbol{\mathsf{X}}`` of dimensions ``\left(\prod_{i=1}^{n}m_{i}\right) \times m``
(the number of tensor product points where the geometry is evaluated in element `element_idx`, and the dimension of the
embedding space to where the canonical element is mapped into. Specifically:

```math
\boldsymbol{\mathsf{X}}_{k, l} = \Phi_{r, l}(\xi^{1}_{j_{1}}, \dots, \xi^{n}_{j_{n}}),
```

where ``r = \mathtt{element\_idx}``, and ``k = j_{1} + \sum_{i=2}^{n} (j_{i} - 1)\prod_{l=1}^{i-1}m_{l}``, as before.

# Arguments
- `geometry::TensorProductGeometry{manifold_dim, T}`: tensor product geometry of elements of dimension `n` embedded in a domain of
   dimension `m`.
- `element_idx::Int`: the index of the element of the tensor product geometry where to evaluate the geometry.
- `ξ::NTuple{n,Vector{Float64}}`: the ``n`` unidimensional vectors with ``m_{i}``, ``i = 1, \dots, n``,
   coordinates where to evaluate the geometry in the canonical domain `[0, 1]^{n}`.
   The geometry will be evaluated at the `M` points, ``\mathtt{M} = \prod_{i=1}^{n}m_{i}``, generated by
   the tensor product of the unidimensional coordinates, as explained above.

# Returns
`x::Array{Float64, 2}(M, m)`: the evaluation of the geometry at the element `element_idx`, and the tensor product points
 of the unidimensional coordinates `\xi`. See above for more details.

"""
function evaluate(
    geometry::TensorProductGeometry{manifold_dim, T},
    element_idx::Int,
    ξ::NTuple{manifold_dim, Vector{Float64}}
) where {
    manifold_dim, num_geometries, T<:NTuple{num_geometries, AbstractGeometry}
}
    ordered_indices = get_ordered_indices(geometry, element_idx)
    manifold_ranges = _get_manifold_ranges(geometry)
    # Evaluate the subgeometries at their associated evaluation points
    eval_per_geometry = ntuple(num_geometries) do k
        return evaluate(
            geometry.geometries[k], ordered_indices[k], ξ[manifold_ranges[k]]
        )
    end

    image_ranges = _get_image_ranges(geometry)
    n_points_per_geometry = ntuple(num_geometries) do k
        return prod(size.(ξ[manifold_ranges[k]], 1))
    end
    n_points = prod(n_points_per_geometry) # total number of evaluation points
    image_dim = get_image_dim(geometry)
    eval = zeros(Float64, n_points, image_dim) # evaluation storage
    
    ordered_points = CartesianIndices(n_points_per_geometry)
    linear_points = LinearIndices(ordered_points)
    # loop over all points
    for (lin_point, ord_point) in zip(linear_points, ordered_points)
        for k in 1:num_geometries # loop over geometries
            eval[lin_point, image_ranges[k]] .= @view eval_per_geometry[k][ord_point[k], :]
        end
    end

    return eval
end

@doc raw"""
    jacobian(geometry::TensorProductGeometry{manifold_dim, T}, element_idx::Int, ξ::NTuple{n,Vector{Float64}})

Evaluates the Jacobian of the tensor product geometry `geometry` at element `element_idx` and at the tensor product of the
`NTuple` coordinates `ξ`.

In other words, given the `NTuple` `ξ` of ``n`` `Vectors`, ``\boldsymbol{\xi}^{i}``, ``i=1, \dots, n``,
each containing ``m_{i}`` unidimensional coordinates ``\xi^{i}_{j}``, ``i = 1, \dots, n`` and `` j = 1, \dots m_{i}``,
evaluates the Jacobian of the tensor product geometry at the element `element_idx` and at the ``\prod_{i=1}^{n}m_{i}`` tensor product
points ``V_{k = j_{1} + \sum_{i=2}^{n} (j_{i} - 1)\prod_{l=1}^{i-1}m_{l}} = (\xi^{1}_{j_{1}}, \dots, \xi^{n}_{j_{n}})``,
with ``j_{i} = 1, \dots, m_{i}``.

The output is a matrix, ``\boldsymbol{\mathsf{J}}`` of dimensions ``\left(\prod_{i=1}^{n}m_{i}\right) \times m \times n``
(the number of tensor product points where the geometry is evaluated in element `element_idx`, the dimension of the
embedding space to where the canonical element is mapped into, and the dimension of the canonical element, which is the
same as the dimension of the element's manifold). Specifically:

```math
\boldsymbol{\mathsf{J}}_{k, l, s} = \frac{\partial\Phi_{r, l}}{\partial\xi_{s}}(\xi^{1}_{j_{1}}, \dots, \xi^{n}_{j_{n}}),
```

where ``r = \mathtt{element\_idx}``, and ``k = j_{1} + \sum_{i=2}^{n} (j_{i} - 1)\prod_{l=1}^{i-1}m_{l}``, as before.

# Arguments
- `geometry::TensorProductGeometry{manifold_dim, T}`: tensor product geometry of elements of dimension `n` embedded in a domain of
   dimension `m`.
- `element_idx::Int`: the index of the element of the tensor product geometry where to evaluate the geometry.
- `ξ::NTuple{n,Vector{Float64}}`: the ``n`` unidimensional vectors with ``m_{i}``, ``i = 1, \dots, n``,
   coordinates where to evaluate the geometry in the canonical domain `[0, 1]^{n}`.
   The geometry will be evaluated at the `M` points, ``\mathtt{M} = \prod_{i=1}^{n}m_{i}``, generated by
   the tensor product of the unidimensional coordinates, as explained above.

# Returns
`J::Array{Float64, 2}(M, m)`: the evaluation of the Jacobian of the geometry at the element `element_idx`,
 and the tensor product points of the unidimensional coordinates `\xi`. See above for more details.

"""
function jacobian(
    geometry::TensorProductGeometry{manifold_dim, T},
    element_idx::Int,
    ξ::NTuple{manifold_dim, Vector{Float64}}
) where {
    manifold_dim, num_geometries, T<:NTuple{num_geometries, AbstractGeometry}
}
    ordered_indices = get_ordered_indices(geometry, element_idx)
    manifold_ranges = _get_manifold_ranges(geometry)
    # Evaluate the jacobian for each subgeometry at their associated evaluation points
    jacobian_per_geometry = ntuple(num_geometries) do k
        return jacobian(
            geometry.geometries[k], ordered_indices[k], ξ[manifold_ranges[k]]
        )
    end

    image_ranges = _get_image_ranges(geometry)
    n_points_per_geometry = ntuple(num_geometries) do k
        return prod(size.(ξ[manifold_ranges[k]], 1))
    end
    n_points = prod(n_points_per_geometry) # total number of evaluation points
    image_dim = get_image_dim(geometry)
    jacobian_eval = zeros(Float64, n_points, image_dim, manifold_dim) # evaluation storage

    ordered_points = CartesianIndices(n_points_per_geometry)
    linear_points = LinearIndices(ordered_points)
    # loop over all points
    for (lin_point, ord_point) in zip(linear_points, ordered_points)
        for k in 1:num_geometries # loop over geometries
            jacobian_eval[lin_point, image_ranges[k], manifold_ranges[k]] .= view(
                jacobian_per_geometry[k], ord_point[k], :, :
            )
        end
    end

    return jacobian_eval
end
