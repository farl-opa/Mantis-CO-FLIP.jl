import LinearAlgebra

function metric(geometry::AbstractGeometry{n}, element_id::Int, xi::NTuple{n,Vector{Float64}}) where {n}
    # The metric tensor gᵢⱼ is
    #   gᵢⱼ := ∑ₖᵐ ∂Φᵏ\∂ξᵢ ⋅ ∂Φᵏ\∂ξⱼ
    # Since the Jacobian matrix J is given by
    #   Jᵢⱼ = ∂Φⁱ\∂ξⱼ
    # The metric tensor can be computed as 
    #   g = Jᵗ ⋅ J 
    
    # Compute the jacobian
    J = jacobian(geometry, element_id, xi)

    # Compute the metric 
    n_evaluation_points = prod(size.(xi, 1))  # compute the number of nD points evaluated (the tensor product of the n sets of 1D input coordinates)
    g = zeros(Float64, n_evaluation_points, n, n)  # allocate memory space for the metric tensor 

    for column_idx in 1:n 
        for row_idx in 1:n
            # This line computes: gᵢⱼ := ∑ₖᵐ ∂Φᵏ\∂ξᵢ ⋅ ∂Φᵏ\∂ξⱼ
            # with:
            #   - i := row_idx 
            #   - j := column_idx
            # The summation over k corresponds to the summation over the dimension 2.
            # The first dimension are the evaluation points, therefore we do it for all
            # evaluation points in one go.
            g[:, row_idx, column_idx] .= sum(view(J, :, :, row_idx) .* view(J, :, :, column_idx), dims = 2)
        end
    end

    # Compute the squareroot of the determinant of the metric
    sqrt_g = sqrt.(abs.(LinearAlgebra.det.(eachslice(g; dims=1))))

    return g, sqrt_g
end

function inv_metric(geometry::AbstractGeometry{n}, element_id::Int, xi::NTuple{n,Vector{Float64}}) where {n}
    # The inverse of the metric tensor gᵢⱼ is
    #   [gⁱʲ] := [gᵢⱼ]⁻¹
    # The metric tensor is computed for all evaluation points. To avoid having to compute the 
    # inverse of the metric tensor for each evaluation point, we explicitly encode the inverse 
    # in order to be able to do it in one go for all evaluation points.
    # We can do this because we have a limited number of possible inverses we need to compute, namely:
    #   n x n
    # with n = 1, 2, 3
    # in the future if we wish to add space-time approaches we can add the n = 4 case.

    # First compute the metric 
    g, sqrt_g = metric(geometry, element_id, xi)

    # Then compute the inverse of the metric
    # inv_g = mapslices(LinearAlgebra.inv, g, dims=(2, 3))  # operate LinearAlgebra.inv over the dimension 2 and 3 of g

    # Test to see if the inv_g computation can be type stable:
    inv_g = zeros(size(g))
    for point in axes(g, 1)
        inv_g[point,:,:] .= LinearAlgebra.inv(g[point,:,:])
    end

    return inv_g, g, sqrt_g
end
