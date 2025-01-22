import LinearAlgebra

function metric(
    geometry::AbstractGeometry{manifold_dim},
    element_id::Int,
    xi::NTuple{manifold_dim,Vector{Float64}},
) where {manifold_dim}
    # The metric tensor gᵢⱼ is
    #   gᵢⱼ := ∑ₖᵐ ∂Φᵏ\∂ξᵢ ⋅ ∂Φᵏ\∂ξⱼ
    # Since the Jacobian matrix J is given by
    #   Jᵢⱼ = ∂Φⁱ\∂ξⱼ
    # The metric tensor can be computed as 
    #   g = Jᵗ ⋅ J 
    
    # Compute the jacobian
    J = jacobian(geometry, element_id, xi)

    n_evaluation_points = prod(size.(xi, 1))  # compute the number of nD points evaluated
    g = zeros(Float64, n_evaluation_points, manifold_dim, manifold_dim) 

    ## Compute the metric 
    #for column_idx in 1:manifold_dim 
    #    for row_idx in 1:manifold_dim
    #        for point in 1:n_evaluation_points
    #            for dim in 1:manifold_dim
    #                g[point, row_idx, column_idx] += 
    #                    J[point, dim, row_idx] * J[point, dim, column_idx]
    #            end
    #        end
    #    end
    #end

    # Here we compute: gᵢⱼ := ∑ₖᵐ ∂Φᵏ\∂ξᵢ ⋅ ∂Φᵏ\∂ξⱼ
    # with:
    #   - i := row 
    #   - j := column
    # The summation over k corresponds to the summation over the dimension 2.
    for index in CartesianIndices(g)
        (point, row, column) = Tuple(index)
        for k in 1:manifold_dim
            g[point, row, column] += J[point, k, row] * J[point, k, column]
        end
    end
    
    sqrt_g = sqrt.(abs.(LinearAlgebra.det.(eachslice(g; dims=1))))

    return g, sqrt_g
end

function inv_metric(
    geometry::AbstractGeometry{manifold_dim},
    element_id::Int,
    xi::NTuple{manifold_dim,Vector{Float64}}
) where {manifold_dim}
    # The inverse of the metric tensor gᵢⱼ is
    #   [gⁱʲ] := [gᵢⱼ]⁻¹
    # The metric tensor is computed for all evaluation points. To avoid having to compute
    # the inverse of the metric tensor for each evaluation point, we explicitly encode
    # the inverse in order to be able to do it in one go for all evaluation points.
    # We can do this because we have a limited number of possible inverses we need to
    # compute, namely:
    #   manifold_dim x manifold_dim
    # with manifold_dim = 1, 2, 3
    # in the future if we wish to add space-time approaches we can add the manifold_dim = 4
    # case.

    # First compute the metric 
    g, sqrt_g = metric(geometry, element_id, xi)

    # Then compute the inverse of the metric
    # inv_g = mapslices(LinearAlgebra.inv, g, dims=(2, 3))  # operate LinearAlgebra.inv over
    # the dimension 2 and 3 of g

    # this is needed because inv can't handle non-contiguous columns. This is what happens
    # when using views in the original g storage order
    g_permuted = permutedims(g, (2,3,1))
    inv_g = LinearAlgebra.inv.(eachslice(g_permuted; dims=3))
    

    return inv_g, g, sqrt_g
end
