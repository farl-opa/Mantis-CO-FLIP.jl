################################################################################
# Univariate and tensor-product B-spline space creation functions
################################################################################

"""
    create_bspline_space(
        starting_point::Float64,
        box_size::Float64,
        num_elements::Int,
        section_space::F,
        regularity::Int;
        n_dofs_left::Int=1,
        n_dofs_right::Int=1,
    ) where {F <: AbstractCanonicalSpace}

Create a 1D B-spline space based on the provided starting point, box size, number of
elements, section space, and regularity. It constructs a uniform distribution of
breakpoints and uses them to define a `Mesh.Patch1D` object. The function then creates a
regularity vector, with the first and last entries set to -1 to ensure an open knot vector,
and returns the corresponding `BSplineSpace`. Optional arguments include the number of
degrees of freedom on the left and right boundaries.

# Arguments
- `starting_point::Float64`: The coordinate of the starting point of the B-spline space.
- `box_size::Float64`: The size of the space along the single dimension.
- `num_elements::Int`: The number of elements along the dimension.
- `section_space::F`: The section space to use for the B-spline basis functions.
- `regularity::Int`: The regularity (smoothness) between consecutive B-spline basis
    functions, typically between 0 and `degree - 1`.
- `n_dofs_left::Int`: The number of degrees of freedom on the left boundary.
- `n_dofs_right::Int`: The number of degrees of freedom on the right boundary.

# Returns
- `::BSplineSpace`: The generated B-spline space.

# Examples
```julia
# Create a 1D B-spline space
start_1d = 0.0
size_1d = 10.0
elements_1d = 100
degree = 3
regularity = 2
section_space = Bernstein(degree)
bspline_space_1d = create_bspline_space(start_1d, size_1d, elements_1d, section_space, regularity)
```
"""
function create_bspline_space(
    starting_point::Float64,
    box_size::Float64,
    num_elements::Int,
    section_space::F,
    regularity::Int;
    n_dofs_left::Int=1,
    n_dofs_right::Int=1,
) where {F <: AbstractCanonicalSpace}
    if box_size > 0.0
        nothing
    else
        throw(ArgumentError("The argumente 'box_size' must be greater than 0."))
    end

    breakpoints = collect(
        LinRange(starting_point, starting_point + box_size, num_elements + 1)
    )
    patch = Mesh.Patch1D(breakpoints)

    regularity_vector = fill(regularity, (num_elements + 1,))
    regularity_vector[1] = regularity_vector[end] = -1 # Open knot vector

    return BSplineSpace(patch, section_space, regularity_vector, n_dofs_left, n_dofs_right)
end

"""
    create_bspline_space(
        starting_point::Float64,
        box_size::Float64,
        num_elements::Int,
        degree::Int,
        regularity::Int;
        n_dofs_left::Int=1,
        n_dofs_right::Int=1,
    )

Create a piecewise-polynomial 1D B-spline space based on the provided starting point, box
size, number of elements, polynomial degree, and regularity. It constructs a uniform
distribution of breakpoints and uses them to define a `Mesh.Patch1D` object. The function
then creates a regularity vector, with the first and last entries set to -1 to ensure an
open knot vector, and returns the corresponding `BSplineSpace`. Optional arguments include
the number of degrees of freedom on the left and right boundaries.

# Arguments
- `starting_point::Float64`: The coordinate of the starting point of the B-spline space.
- `box_size::Float64`: The size of the space along the single dimension.
- `num_elements::Int`: The number of elements along the dimension.
- `degree::Int`: The polynomial degree on each element.
- `regularity::Int`: The regularity (smoothness) between consecutive B-spline basis
    functions, typically between 0 and `degree - 1`.
- `n_dofs_left::Int`: The number of degrees of freedom on the left boundary.
- `n_dofs_right::Int`: The number of degrees of freedom on the right boundary.

# Returns
- `::BSplineSpace`: The generated B-spline space.
"""
function create_bspline_space(
    starting_point::Float64,
    box_size::Float64,
    num_elements::Int,
    degree::Int,
    regularity::Int;
    n_dofs_left::Int=1,
    n_dofs_right::Int=1,
)
    return create_bspline_space(
        starting_point,
        box_size,
        num_elements,
        Bernstein(degree),
        regularity;
        n_dofs_left=n_dofs_left,
        n_dofs_right=n_dofs_right,
    )
end

"""
    create_bspline_space(
        starting_point::NTuple{1, Float64},
        box_size::NTuple{1, Float64},
        num_elements::NTuple{1, Int},
        section_space::NTuple{1, F},
        regularity::NTuple{1, Int};
        n_dofs_left::NTuple{1, Int}=(1,),
        n_dofs_right::NTuple{1, Int}=(1,),
    ) where {F <: AbstractCanonicalSpace}

Create a 1D B-spline space based on the provided starting point, box size, number of
elements, section space, and regularity. It constructs a uniform distribution of
breakpoints and uses them to define a `Mesh.Patch1D` object. The function then creates a
regularity vector, with the first and last entries set to -1 to ensure an open knot vector,
and returns the corresponding `BSplineSpace`. Optional arguments include the number of
degrees of freedom on the left and right boundaries.

# Arguments
- `starting_point::NTuple{1, Float64}`: The coordinate of the starting point of the
    B-spline space.
- `box_size::NTuple{1, Float64}`: The size of the space along the single dimension.
- `num_elements::NTuple{1, Int}`: The number of elements along the dimension.
- `section_space::NTuple{1,F}`: The section space to use for the B-spline basis functions.
- `regularity::NTuple{1, Int}`: The regularity (smoothness) between consecutive B-spline
    basis functions, typically between 0 and `degree - 1`.
- `n_dofs_left::NTuple{1,Int}`: The number of degrees of freedom on the left boundary.
- `n_dofs_right::NTuple{1,Int}`: The number of degrees of freedom on the right boundary.

# Returns
- `::BSplineSpace`: The generated B-spline space.

# Examples
```julia
# Create a 1D B-spline space
start_1d = 0.0
size_1d = 10.0
elements_1d = 100
degree = 3
regularity = 2
section_space = Bernstein(degree)
bspline_space_1d = create_bspline_space((start_1d,), (size_1d,), (elements_1d,), (section_space,), (regularity,))
```
"""
function create_bspline_space(
    starting_point::NTuple{1, Float64},
    box_size::NTuple{1, Float64},
    num_elements::NTuple{1, Int},
    section_space::NTuple{1, F},
    regularity::NTuple{1, Int};
    n_dofs_left::NTuple{1, Int}=(1,),
    n_dofs_right::NTuple{1, Int}=(1,),
) where {F <: AbstractCanonicalSpace}
    return create_bspline_space(
        starting_point[1],
        box_size[1],
        num_elements[1],
        section_space[1],
        regularity[1];
        n_dofs_left=n_dofs_left[1],
        n_dofs_right=n_dofs_right[1],
    )
end

"""
    create_bspline_space(
        starting_point::NTuple{1, Float64},
        box_size::NTuple{1, Float64},
        num_elements::NTuple{1, Int},
        degree::NTuple{1, Int},
        regularity::NTuple{1, Int};
        n_dofs_left::NTuple{1, Int}=(1,),
        n_dofs_right::NTuple{1, Int}=(1,),
    )

Create a piecewise-polynomial 1D B-spline space based on the provided starting point,
box size, number of elements, polynomial degree, and regularity. It constructs a uniform
distribution of breakpoints and uses them to define a `Mesh.Patch1D` object. The function
then creates a regularity vector, with the first and last entries set to -1 to ensure an
open knot vector, and returns the corresponding `BSplineSpace`. Optional arguments include
the number of degrees of freedom on the left and right boundaries.

# Arguments
- `starting_point::Float64`: The coordinate of the starting point of the B-spline space.
- `box_size::Float64`: The size of the space along the single dimension.
- `num_elements::Int`: The number of elements along the dimension.
- `degree::Int`: The polynomial degree on each element.
- `regularity::Int`: The regularity (smoothness) between consecutive B-spline basis
    functions, typically between 0 and `degree - 1`.
- `n_dofs_left::Int`: The number of degrees of freedom on the left boundary.
- `n_dofs_right::Int`: The number of degrees of freedom on the right boundary.

# Returns
- `::BSplineSpace`: The generated B-spline space.
"""
function create_bspline_space(
    starting_point::NTuple{1, Float64},
    box_size::NTuple{1, Float64},
    num_elements::NTuple{1, Int},
    degree::NTuple{1, Int},
    regularity::NTuple{1, Int};
    n_dofs_left::NTuple{1, Int}=(1,),
    n_dofs_right::NTuple{1, Int}=(1,),
)
    return create_bspline_space(
        starting_point[1],
        box_size[1],
        num_elements[1],
        Bernstein(degree[1]),
        regularity[1];
        n_dofs_left=n_dofs_left[1],
        n_dofs_right=n_dofs_right[1],
    )
end

"""
    create_bspline_space(
        starting_points::NTuple{manifold_dim, Float64},
        box_sizes::NTuple{manifold_dim, Float64},
        num_elements::NTuple{manifold_dim, Int},
        section_spaces::F,
        regularities::NTuple{manifold_dim, Int};
        n_dofs_left::NTuple{manifold_dim, Int}=Tuple(ones(Int, manifold_dim)),
        n_dofs_right::NTuple{manifold_dim, Int}=Tuple(ones(Int, manifold_dim)),
    ) where {manifold_dim, F <: NTuple{manifold_dim, AbstractCanonicalSpace}}

Create a tensor product B-spline space based on the specified parameters for each dimension.

# Arguments
- `starting_points::NTuple{manifold_dim, Float64}`: The starting point of the B-spline
    space in each dimension.
- `box_sizes::NTuple{manifold_dim, Float64}`: The size of the box in each dimension.
- `num_elements::NTuple{manifold_dim, Int}`: The number of elements in each dimension.
- `section_spaces::NTuple{manifold_dim, AbstractCanonicalSpace}`: The section spaces for
    each dimension.
- `regularities::NTuple{manifold_dim, Int}`: The regularities of the B-spline in each
    dimension.

# Returns
- `::TensorProductSpace`: The resulting tensor product B-spline space.
"""
function create_bspline_space(
    starting_points::NTuple{manifold_dim, Float64},
    box_sizes::NTuple{manifold_dim, Float64},
    num_elements::NTuple{manifold_dim, Int},
    section_spaces::F,
    regularities::NTuple{manifold_dim, Int};
    n_dofs_left::NTuple{manifold_dim, Int}=Tuple(ones(Int, manifold_dim)),
    n_dofs_right::NTuple{manifold_dim, Int}=Tuple(ones(Int, manifold_dim)),
) where {manifold_dim, F <: NTuple{manifold_dim, AbstractCanonicalSpace}}
    return TensorProductSpace(
        create_dim_wise_bspline_spaces(
            starting_points,
            box_sizes,
            num_elements,
            section_spaces,
            regularities,
            n_dofs_left,
            n_dofs_right,
        ),
    )
end

"""
    create_dim_wise_bspline_spaces(
        starting_points::NTuple{manifold_dim, Float64},
        box_sizes::NTuple{manifold_dim, Float64},
        num_elements::NTuple{manifold_dim, Int},
        section_spaces::F,
        regularities::NTuple{manifold_dim, Int},
        n_dofs_left::NTuple{manifold_dim, Int},
        n_dofs_right::NTuple{manifold_dim, Int},
    ) where {manifold_dim, F <: NTuple{manifold_dim, AbstractCanonicalSpace}}

Create `manifold_dim` univariate B-spline spaces based on the specified parameters for each
dimension.

# Arguments
- `starting_points::NTuple{manifold_dim, Float64}`: The starting point of the B-spline
    space in each dimension.
- `box_sizes::NTuple{manifold_dim, Float64}`: The size of the box in each dimension.
- `num_elements::NTuple{manifold_dim, Int}`: The number of elements in each dimension.
- `section_spaces::NTuple{manifold_dim, F}`: The section spaces for each dimension.
- `regularities::NTuple{manifold_dim, Int}`: The regularities of the B-spline in each
    dimension.

# Returns
- `::Tuple{BSplineSpace}`: The resulting univariate B-spline spaces.
"""
function create_dim_wise_bspline_spaces(
    starting_points::NTuple{manifold_dim, Float64},
    box_sizes::NTuple{manifold_dim, Float64},
    num_elements::NTuple{manifold_dim, Int},
    section_spaces::F,
    regularities::NTuple{manifold_dim, Int},
    n_dofs_left::NTuple{manifold_dim, Int},
    n_dofs_right::NTuple{manifold_dim, Int},
) where {manifold_dim, F <: NTuple{manifold_dim, AbstractCanonicalSpace}}
    return map(
        (xᵢ, Lᵢ, mᵢ, Fᵢ, kᵢ, lᵢ, rᵢ) ->
            create_bspline_space(xᵢ, Lᵢ, mᵢ, Fᵢ, kᵢ, ; n_dofs_left=lᵢ, n_dofs_right=rᵢ),
        starting_points,
        box_sizes,
        num_elements,
        section_spaces,
        regularities,
        n_dofs_left,
        n_dofs_right,
    )::NTuple{manifold_dim, BSplineSpace}
end

"""
    create_bspline_space(
        starting_points::NTuple{manifold_dim, Float64},
        box_sizes::NTuple{manifold_dim, Float64},
        num_elements::NTuple{manifold_dim, Int},
        degrees::NTuple{manifold_dim, Int},
        regularities::NTuple{manifold_dim, Int};
        n_dofs_left::NTuple{manifold_dim, Int}=Tuple(ones(Int, manifold_dim)),
        n_dofs_right::NTuple{manifold_dim, Int}=Tuple(ones(Int, manifold_dim)),
    ) where {manifold_dim}

Create a tensor product B-spline space based on the specified parameters for each dimension.

# Arguments
- `starting_points::NTuple{manifold_dim, Float64}`: The starting point of the B-spline
    space in each dimension.
- `box_sizes::NTuple{manifold_dim, Float64}`: The size of the box in each dimension.
- `num_elements::NTuple{manifold_dim, Int}`: The number of elements in each dimension.
- `degrees::NTuple{manifold_dim, Int}`: The polynomial degree in each dimension.
- `regularities::NTuple{manifold_dim, Int}`: The regularities of the B-spline in each
    dimension.

# Returns
- `::TensorProductSpace`: The resulting tensor product B-spline space.
"""
function create_bspline_space(
    starting_points::NTuple{manifold_dim, Float64},
    box_sizes::NTuple{manifold_dim, Float64},
    num_elements::NTuple{manifold_dim, Int},
    degrees::NTuple{manifold_dim, Int},
    regularities::NTuple{manifold_dim, Int};
    n_dofs_left::NTuple{manifold_dim, Int}=Tuple(ones(Int, manifold_dim)),
    n_dofs_right::NTuple{manifold_dim, Int}=Tuple(ones(Int, manifold_dim)),
) where {manifold_dim}
    return TensorProductSpace(
        create_dim_wise_bspline_spaces(
            starting_points,
            box_sizes,
            num_elements,
            degrees,
            regularities,
            n_dofs_left,
            n_dofs_right,
        ),
    )
end

"""
    create_dim_wise_bspline_spaces(
        starting_points::NTuple{manifold_dim, Float64},
        box_sizes::NTuple{manifold_dim, Float64},
        num_elements::NTuple{manifold_dim, Int},
        degrees::NTuple{manifold_dim, Int},
        regularities::NTuple{manifold_dim, Int},
        n_dofs_left::NTuple{manifold_dim, Int},
        n_dofs_right::NTuple{manifold_dim, Int},
    ) where {manifold_dim}

Create `manifold_dim` univariate B-spline spaces based on the specified parameters for
each dimension.

# Arguments
- `starting_points::NTuple{manifold_dim, Float64}`: The starting point of the B-spline
    space in each dimension.
- `box_sizes::NTuple{manifold_dim, Float64}`: The size of the box in each dimension.
- `num_elements::NTuple{manifold_dim, Int}`: The number of elements in each dimension.
- `degrees::NTuple{manifold_dim, Int}`: The polynomial degree in each dimension.
- `regularities::NTuple{manifold_dim, Int}`: The regularities of the B-spline in each
    dimension.

# Returns
- `::Tuple{BSplineSpace}`: The resulting univariate B-spline spaces.
"""
function create_dim_wise_bspline_spaces(
    starting_points::NTuple{manifold_dim, Float64},
    box_sizes::NTuple{manifold_dim, Float64},
    num_elements::NTuple{manifold_dim, Int},
    degrees::NTuple{manifold_dim, Int},
    regularities::NTuple{manifold_dim, Int},
    n_dofs_left::NTuple{manifold_dim, Int},
    n_dofs_right::NTuple{manifold_dim, Int},
) where {manifold_dim}
    return map(
        (xᵢ, Lᵢ, mᵢ, pᵢ, kᵢ, lᵢ, rᵢ) ->
            create_bspline_space(xᵢ, Lᵢ, mᵢ, pᵢ, kᵢ, ; n_dofs_left=lᵢ, n_dofs_right=rᵢ),
        starting_points,
        box_sizes,
        num_elements,
        degrees,
        regularities,
        n_dofs_left,
        n_dofs_right,
    )::NTuple{manifold_dim, BSplineSpace{Bernstein}}
end

################################################################################
# Polar spline helpers
################################################################################

"""
    create_polar_spline_space_and_geometry(
        num_elements::NTuple{2, Int},
        degrees::NTuple{2, Int},
        regularities::NTuple{2, Int},
        R::Float64;
        refine::Bool=false,
        geom_coeffs_tp::Union{Nothing, Array{Float64, 3}}=nothing,
        form_rank::Int=0,
    )

Create a polar spline space and geometry based on the provided parameters. The function
constructs a tensor-product B-spline space in the angular and radial directions, and then
creates a polar spline space and geometry based on the tensor-product space. The geometry
is defined by the radius `R` and the number of elements in the angular and radial
directions. Optional arguments include a flag to refine the geometry, the geometry
coefficients, and the form rank.

# Arguments
- `num_elements::NTuple{2, Int}`: The number of elements in the angular and radial
    directions.
- `degrees::NTuple{2, Int}`: The polynomial degrees in the angular and radial directions.
- `regularities::NTuple{2, Int}`: The regularities in the angular and radial directions.
- `R::Float64`: The radius of the domain.
- `refine::Bool`: A flag to refine the geometry.
- `geom_coeffs_tp::Union{Nothing,Array{Float64,3}}`: The geometry coefficients.
- `form_rank::Int`: The form rank.

# Returns
- `(P_sol, E_sol)`: The polar spline solution space and global extraction matrix for the
    solution.
- `(P_geom, E_geom, geom_coeffs_polar)`: The polar spline geometry space, global extraction
    matrix, and geometry coefficients.
- `geom_coeffs_tp`: The tensor-product geometry coefficients.
- `(ts_θ, ts_r)`: The univariate refinement operators for the angular and radial directions.
"""
function create_polar_spline_space_and_geometry(
    num_elements::NTuple{2, Int},
    degrees::NTuple{2, Int},
    regularities::NTuple{2, Int},
    R::Float64;
    refine::Bool=false,
    geom_coeffs_tp::Union{Nothing, Array{Float64, 3}}=nothing,
    form_rank::Int=0,
)
    Bθ, Br = create_dim_wise_bspline_spaces(
        (0.0, 0.0), (1.0, 1.0), num_elements, degrees, regularities, (1, 1), (1, 1)
    )

    return _create_polar_spline_space_and_geometry(
        Bθ, Br, regularities[1], R, refine, geom_coeffs_tp, form_rank
    )
end

"""
    create_polar_spline_space_and_geometry(
        num_elements::NTuple{2, Int},
        section_spaces::F,
        regularities::NTuple{2, Int},
        R::Float64;
        refine::Bool=false,
        geom_coeffs_tp::Union{Nothing, Array{Float64, 3}}=nothing,
        form_rank::Int=0,
    ) where {F <: NTuple{2, AbstractCanonicalSpace}}

Create a polar spline space and geometry based on the provided parameters. The function
constructs a tensor-product B-spline space in the angular and radial directions, and then
creates a polar spline space and geometry based on the tensor-product space. The geometry
is defined by the radius `R` and the number of elements in the angular and radial
directions. Optional arguments include a flag to refine the geometry, the geometry
coefficients, and the form rank.

# Arguments
- `num_elements::NTuple{2, Int}`: The number of elements in the angular and radial
    directions.
- `section_spaces::NTuple{2, AbstractCanonicalSpace}`: The section spaces in the angular
    and radial directions.
- `regularities::NTuple{2, Int}`: The regularities in the angular and radial directions.
- `R::Float64`: The radius of the domain.
- `refine::Bool`: A flag to refine the geometry.
- `geom_coeffs_tp::Union{Nothing,Array{Float64,3}}`: The geometry coefficients.
- `form_rank::Int`: The form rank.

# Returns
- `(P_sol, E_sol)`: The polar spline solution space and global extraction matrix for
    the solution
- `(P_geom, E_geom, geom_coeffs_polar)`: The polar spline geometry space, global extraction
    matrix, and geometry coefficients
- `geom_coeffs_tp`: The tensor-product geometry coefficients
- `(ts_θ, ts_r)`: The univariate refinement operators for the angular and radial directions
"""
function create_polar_spline_space_and_geometry(
    num_elements::NTuple{2, Int},
    section_spaces::F,
    regularities::NTuple{2, Int},
    R::Float64;
    refine::Bool=false,
    geom_coeffs_tp::Union{Nothing, Array{Float64, 3}}=nothing,
    form_rank::Int=0,
) where {F <: NTuple{2, AbstractCanonicalSpace}}
    Bθ, Br = create_dim_wise_bspline_spaces(
        (0.0, 0.0), (1.0, 1.0), num_elements, section_spaces, regularities, (1, 1), (1, 1)
    )

    return _create_polar_spline_space_and_geometry(
        Bθ, Br, regularities[1], R, refine, geom_coeffs_tp, form_rank
    )
end

"""
    _create_polar_spline_space_and_geometry(
        Bθ, Br, endpt_regularity_θ, R, refine, geom_coeffs_tp, form_rank
    )

Create a polar spline space and geometry based on the provided parameters. The function
constructs a polar spline space and geometry based on the tensor-product space constructed
from the input univariate spaces.

# Arguments
- `Bθ::BSplineSpace`: The univariate B-spline space in the angular direction.
- `Br::BSplineSpace`: The univariate B-spline space in the radial direction.
- `endpt_regularity_θ::Int`: The regularity at the endpoints in the angular direction.
- `R::Float64`: The radius of the domain.
- `refine::Bool`: A flag to refine the geometry.
- `geom_coeffs_tp::Union{Nothing,Array{Float64,3}}`: The geometry coefficients.
- `form_rank::Int`: The form rank.

# Returns
- `(P_sol, E_sol)`: The polar spline solution space and global extraction matrix for the
    solution
- `(P_geom, E_geom, geom_coeffs_polar)`: The polar spline geometry space, global extraction
    matrix, and geometry coefficients
- `geom_coeffs_tp`: The tensor-product geometry coefficients
- `(ts_θ, ts_r)`: The univariate refinement operators for the angular and radial directions
"""
function _create_polar_spline_space_and_geometry(
    Bθ, Br, endpt_regularity_θ, R, refine, geom_coeffs_tp, form_rank
)
    GBθ = GTBSplineSpace((Bθ,), [endpt_regularity_θ]) # impose periodicity

    # number of control points for a degenerate tensor-product mapping
    n_θ = get_num_basis(GBθ)
    n_r = get_num_basis(Br)

    # two-scale operators per dimension
    ts_θ = nothing
    ts_r = nothing
    if isnothing(geom_coeffs_tp)
        geom_coeffs_tp, _, _ = build_standard_degenerate_control_points(n_θ, n_r, R)
    else
        # geometry has already been provided, just check if we need to refine it
        if refine
            # refine the univariate spaces
            ts_r, Br_ref = build_two_scale_operator(Br, 2)
            _, Bθ_ref = build_two_scale_operator(Bθ, 2)
            GBθ_ref = GTBSplineSpace((Bθ_ref,), [endpt_regularity_θ])
            ts_θ, _ = build_two_scale_operator(GBθ, GBθ_ref, ((2,),))

            # refine the tensor-product control points
            geom_coeffs_tp = cat(
                ts_θ.global_subdiv_matrix *
                geom_coeffs_tp[:, :, 1] *
                ts_r.global_subdiv_matrix',
                ts_θ.global_subdiv_matrix *
                geom_coeffs_tp[:, :, 2] *
                ts_r.global_subdiv_matrix';
                dims=3,
            )

            # update old spaces
            Br = Br_ref
            Bθ = Bθ_ref
            GBθ = GBθ_ref
        end
    end

    # Polar spline space and global extraction matrix for the geometry
    P_geom, E_geom = PolarSplineSpace(
        GBθ, Br, (geom_coeffs_tp[:, 1, :], geom_coeffs_tp[:, 2, :]); form_rank=0
    )
    # control points for the polar spline space
    geom_coeffs_polar =
        (E_geom[1] * E_geom[1]') \ (E_geom[1] * reshape(geom_coeffs_tp, :, 2))

    # Polar spline space and global extraction matrix for the solution
    dBr = get_derivative_space(Br)
    dBθ = get_derivative_space(Bθ)
    dGBθ = GTBSplineSpace((dBθ,), [endpt_regularity_θ - 1])
    P_sol, E_sol = PolarSplineSpace(
        GBθ,
        Br,
        (geom_coeffs_tp[:, 1, :], geom_coeffs_tp[:, 2, :]);
        form_rank=form_rank,
        dspace_p=dGBθ,
        dspace_r=dBr,
    )

    return (P_sol, E_sol), (P_geom, E_geom, geom_coeffs_polar), geom_coeffs_tp, (ts_θ, ts_r)
end

################################################################################
# Masking for hierarchical spaces
################################################################################

function create_evaluation_mask_from_hierarchical_mesh(
    hierarchical_space::FunctionSpaces.HierarchicalFiniteElementSpace{manifold_dim, S, T};
    exclude_elements::Vector{Int}=Int[]
) where {
    manifold_dim,
    S<:FunctionSpaces.AbstractFESpace{manifold_dim, 1},
    T<:FunctionSpaces.AbstractTwoScaleOperator
}
    # initialize trivial evaluation mask for first level space
    base_space = FunctionSpaces.get_space(hierarchical_space, 1)
    # number of elements in base mesh
    num_elements_base = FunctionSpaces.get_num_elements(base_space)
    # get element vertices for the base space
    base_element_vertices = [
        FunctionSpaces.get_element_vertices(base_space, i) for i in 1:num_elements_base
    ]

    # number of elements in evaluation mask (i.e., the active hierarchical mesh)
    num_elements = FunctionSpaces.get_num_elements(hierarchical_space)
    # get eval to base element index map
    element_idx_map = zeros(Int64, num_elements)
    # get element vertices for the eval mask
    eval_element_vertices = Vector{NTuple{manifold_dim, Vector{Float64}}}(
        undef, num_elements
    )
    for i in 1:num_elements
        element_level, element_level_id =
        FunctionSpaces.convert_to_element_level_and_level_id(
            hierarchical_space, i
        )
        eval_element_vertices[i] = FunctionSpaces.get_element_vertices(
                FunctionSpaces.get_space(hierarchical_space, element_level),
                element_level_id
        )
        element_idx_map[i] = FunctionSpaces.get_element_ancestor(
            hierarchical_space.two_scale_operators,
            element_level_id,
            element_level,
            element_level-1
        )
    end

    # allocate memory for translations and scalings
    translations = Vector{NTuple{manifold_dim, Float64}}(undef, num_elements)
    scalings = Vector{NTuple{manifold_dim, Float64}}(undef, num_elements)

    # create length scales (i.e., ratios of child and ancestor)
    for element_idx in 1:num_elements
        element_idx_base = element_idx_map[element_idx]
        base_el_verts = base_element_vertices[element_idx_base]
        eval_el_verts = eval_element_vertices[element_idx]

        length_scales = [eval_el_verts[k][2] - eval_el_verts[k][1] for k in 1:manifold_dim] ./
        [base_el_verts[k][2] - base_el_verts[k][1] for k in 1:manifold_dim]
        scalings[element_idx] = tuple(length_scales...)

        translations[element_idx] = tuple(
            [(eval_el_verts[k][1] - base_el_verts[k][1]) /
            (base_el_verts[k][2] - base_el_verts[k][1]) for k in 1:manifold_dim]...
        )
    end

    # create corresponding evaluation mask
    E = EvaluationMask.AffineEvaluationMask(
        num_elements,
        num_elements_base,
        element_idx_map,
        translations,
        scalings
    )
    # return trimmed mask
    return EvaluationMask.trim_evaluation_mask(E, exclude_elements)
end
