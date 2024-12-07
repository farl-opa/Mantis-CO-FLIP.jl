################################################################################
# Some standard geometries
################################################################################

"""
    create_cartesian_box(starting_points::NTuple{manifold_dim, Float64}, box_sizes::NTuple{manifold_dim, Float64}, num_elements::NTuple{manifold_dim, Int}) where {manifold_dim}

Create a Cartesian box geometry with `manifold_dim` dimensions, starting at `starting_points` and with `box_sizes` and `num_elements` defining the size of the box.

# Arguments
- `starting_points::NTuple{manifold_dim, Float64}`: The starting points of the box.
- `box_sizes::NTuple{manifold_dim, Float64}`: The size of the box.
- `num_elements::NTuple{manifold_dim, Int}`: The number of elements in each dimension.

# Output
- `geometry::CartesianGeometry{manifold_dim}`: The Cartesian box geometry.
"""
function create_cartesian_box(starting_points::NTuple{manifold_dim, Float64}, box_sizes::NTuple{manifold_dim, Float64}, num_elements::NTuple{manifold_dim, Int}) where {manifold_dim}
    
    breakpoints = map(LinRange, starting_points, starting_points .+ box_sizes, num_elements .+ 1)
    return CartesianGeometry(map(collect,breakpoints))
end