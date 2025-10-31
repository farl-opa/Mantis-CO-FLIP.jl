module CartesianPointsTests

using Mantis
using Test

manifold_dims = [1, 2, 3]
num_const_points = [(2,), (4, 3), (2, 1, 5)]
num_points = [prod(num_const_points[dim]) for dim in 1:3]
for i in 1:3
    manifold_dim = manifold_dims[i]
    points = ntuple(dim -> rand(num_const_points[i][dim]), manifold_dim)
    range = 1:num_points[i]
    xi = Points.CartesianPoints(points)

    @test Points.get_manifold_dim(xi) == manifold_dim
    @test firstindex(xi) == 1
    @test lastindex(xi) == num_points[i]
    @test keys(xi) == 1:num_points[i]
    @test Points.get_constituent_points(xi) == points
    @test Points.get_constituent_num_points(xi) == num_const_points[i]
    @test Points.get_num_points(xi) == num_points[i]
    for (j, original_point) in zip(eachindex(xi), Iterators.product(points...))
        @test j == range[j]
        @test xi[j] == original_point
    end
end

end
