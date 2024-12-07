################################################################################
# Tensor-product B-spline de Rham complex
################################################################################

import Combinatorics

"""
    get_basis_index_combinations(manifold_dim::Int, form_rank::Int)

Generate all possible k-form basis index combinations.

# Arguments
- `manifold_dim::Int`: the dimension of the manifold.
- `form_rank::Int`: the rank of the form.

# Returns
- `NTuple{binomial(manifold_dim, form_rank), Vector{Int}}`: the basis index combinations.
"""
function get_basis_index_combinations(manifold_dim::Int, form_rank::Int)

    # generate all possible k-form basis index combinations
    if form_rank == 2 && manifold_dim == 3
        # canonical ordering of the degree-deficit combinations
        return ([2, 3], [1, 3], [1, 2])::NTuple{3, Vector{Int}}
    else
        return tuple(Combinatorics.combinations(1:manifold_dim, form_rank)...)::NTuple{binomial(manifold_dim, form_rank), Vector{Int}}
    end
end

"""
    create_tensor_product_bspline_de_rham_complex(starting_points::NTuple{manifold_dim, Float64}, box_sizes::NTuple{manifold_dim, Float64}, num_elements::NTuple{manifold_dim, Int}, section_spaces::NTuple{manifold_dim, F}, regularities::NTuple{manifold_dim, Int}) where {manifold_dim, F <: FunctionSpaces.AbstractCanonicalSpace}

Create a tensor-product B-spline de Rham complex.

# Arguments
- `starting_points::NTuple{manifold_dim, Float64}`: the starting points of the domain.
- `box_sizes::NTuple{manifold_dim, Float64}`: the sizes of the domain.
- `num_elements::NTuple{manifold_dim, Int}`: the number of elements in each direction.
- `section_spaces::NTuple{manifold_dim, F}`: the section spaces.
- `regularities::NTuple{manifold_dim, Int}`: the regularities of the B-spline spaces.
- `geometry::G`: the geometry of the domain.

# Returns
- `Vector{AbstractFormSpace}`: the `manifold_dim+1` form spaces of the complex.
"""
function create_tensor_product_bspline_de_rham_complex(starting_points::NTuple{manifold_dim, Float64}, box_sizes::NTuple{manifold_dim, Float64}, num_elements::NTuple{manifold_dim, Int}, section_spaces::NTuple{manifold_dim, F}, regularities::NTuple{manifold_dim, Int}, geometry::G) where {manifold_dim, F <: FunctionSpaces.AbstractCanonicalSpace, G <: Geometry.AbstractGeometry{manifold_dim}}

    # number of dofs on the left and right boundary of the domain
    n_dofs_left = tuple((1 for _ in 1:manifold_dim)...)
    n_dofs_right = tuple((1 for _ in 1:manifold_dim)...)

    # store all univariate FEM spaces helper
    fem_spaces = Vector{NTuple{manifold_dim,FunctionSpaces.AbstractFiniteElementSpace{1}}}(undef, 2)
    # first, create all univariate FEM spaces corresponding to directional-zero forms
    fem_spaces[1] = FunctionSpaces.create_dim_wise_bspline_spaces(starting_points, box_sizes, num_elements, section_spaces, regularities, n_dofs_left, n_dofs_right)
    # next, create all univariate FEM spaces corresponding to directional-one forms
    fem_spaces[2] = map(FunctionSpaces.get_derivative_space, fem_spaces[1])

    # allocate space for all form spaces
    form_spaces = Vector{AbstractFormSpace}(undef, manifold_dim+1)
    # loop over all form ranks
    for k = 0:manifold_dim
        # get k-form basis indices, these also inform the directional degree-deficits
        k_form_basis_idxs = get_basis_index_combinations(manifold_dim, k)
        n_form_components = length(k_form_basis_idxs)
        # allocate space for storing all k-form finite element spaces
        k_form_fem_spaces = Vector{FunctionSpaces.AbstractFiniteElementSpace{manifold_dim}}(undef, n_form_components)
        # loop over all k-form bases and construct the corresponding tensor-product B-spline space
        for component_idx in 1:n_form_components
            # by default, use direction-zero forms...
            fem_space_idxs = ones(Int, manifold_dim)
            # ...unless the basis index is present in the k-form basis indices
            fem_space_idxs[k_form_basis_idxs[component_idx]] .= 2
            # build and store the corresponding tensor-product FEM space
            k_form_fem_spaces[component_idx] = FunctionSpaces.TensorProductSpace(tuple([fem_spaces[fem_space_idxs[i]][i] for i in 1:manifold_dim]...))
        end
        # create the form space
        form_spaces[k+1] = FormSpace(k, geometry, FunctionSpaces.DirectSumSpace(tuple(k_form_fem_spaces...)), "ω_$k")
    end

    return form_spaces
end

"""
    create_tensor_product_bspline_de_rham_complex(starting_points::NTuple{manifold_dim, Float64}, box_sizes::NTuple{manifold_dim, Float64}, num_elements::NTuple{manifold_dim, Int}, section_spaces::NTuple{manifold_dim, F}, regularities::NTuple{manifold_dim, Int}) where {manifold_dim, F <: FunctionSpaces.AbstractCanonicalSpace}

Create a tensor-product B-spline de Rham complex.

# Arguments
- `starting_points::NTuple{manifold_dim, Float64}`: the starting points of the domain.
- `box_sizes::NTuple{manifold_dim, Float64}`: the sizes of the domain.
- `num_elements::NTuple{manifold_dim, Int}`: the number of elements in each direction.
- `section_spaces::NTuple{manifold_dim, F}`: the section spaces.
- `regularities::NTuple{manifold_dim, Int}`: the regularities of the B-spline spaces.

# Returns
- `Vector{AbstractFormSpace}`: the `manifold_dim+1` form spaces of the complex.
"""
function create_tensor_product_bspline_de_rham_complex(starting_points::NTuple{manifold_dim, Float64}, box_sizes::NTuple{manifold_dim, Float64}, num_elements::NTuple{manifold_dim, Int}, section_spaces::NTuple{manifold_dim, F}, regularities::NTuple{manifold_dim, Int}) where {manifold_dim, F <: FunctionSpaces.AbstractCanonicalSpace}

    # create underlying box geometry
    geometry = Geometry.create_cartesian_box(starting_points, box_sizes, num_elements)

    return create_tensor_product_bspline_de_rham_complex(starting_points, box_sizes, num_elements, section_spaces, regularities, geometry)
end

"""
    create_tensor_product_bspline_de_rham_complex(starting_points::NTuple{manifold_dim, Float64}, box_sizes::NTuple{manifold_dim, Float64}, num_elements::NTuple{manifold_dim, Int}, degrees::NTuple{manifold_dim, Int}, regularities::NTuple{manifold_dim, Int}) where {manifold_dim}

Create a tensor-product B-spline de Rham complex.

# Arguments
- `starting_points::NTuple{manifold_dim, Float64}`: the starting points of the domain.
- `box_sizes::NTuple{manifold_dim, Float64}`: the sizes of the domain.
- `num_elements::NTuple{manifold_dim, Int}`: the number of elements in each direction.
- `degrees::NTuple{manifold_dim, Int}`: the degrees of the B-spline spaces.
- `regularities::NTuple{manifold_dim, Int}`: the regularities of the B-spline spaces.

# Returns
- `Vector{AbstractFormSpace}`: the `manifold_dim+1` form spaces of the complex.
"""
function create_tensor_product_bspline_de_rham_complex(starting_points::NTuple{manifold_dim, Float64}, box_sizes::NTuple{manifold_dim, Float64}, num_elements::NTuple{manifold_dim, Int}, degrees::NTuple{manifold_dim, Int}, regularities::NTuple{manifold_dim, Int}) where {manifold_dim}

    return create_tensor_product_bspline_de_rham_complex(starting_points, box_sizes, num_elements, map(FunctionSpaces.Bernstein, degrees), regularities)
end