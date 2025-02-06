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
- `::Tuple{<:AbstractFormSpace{manifold_dim, form_rank, geometry}}`: Tuple with the form
    spaces of the complex, for each `form_rank` from `0` to `manifold_dim+1`.
"""
function create_tensor_product_bspline_de_rham_complex(starting_points::NTuple{manifold_dim, Float64}, box_sizes::NTuple{manifold_dim, Float64}, num_elements::NTuple{manifold_dim, Int}, section_spaces::NTuple{manifold_dim, F}, regularities::NTuple{manifold_dim, Int}, geometry::G) where {manifold_dim, F <: FunctionSpaces.AbstractCanonicalSpace, G <: Geometry.AbstractGeometry{manifold_dim}}

    # number of dofs on the left and right boundary of the domain
    n_dofs_left = tuple((1 for _ in 1:manifold_dim)...)
    n_dofs_right = tuple((1 for _ in 1:manifold_dim)...)

    # store all univariate FEM spaces helper
    fem_spaces = Vector{NTuple{manifold_dim,FunctionSpaces.AbstractFiniteElementSpace{1}}}(undef, 2)
    # first, create all univariate FEM spaces corresponding to directional-zero forms
    fem_spaces[1] = FunctionSpaces.create_dim_wise_bspline_spaces(
        starting_points,
        box_sizes,
        num_elements,
        section_spaces,
        regularities,
        n_dofs_left,
        n_dofs_right,
    )
    # next, create all univariate FEM spaces corresponding to directional-one forms
    fem_spaces[2] = map(FunctionSpaces.get_derivative_space, fem_spaces[1])

    # Build all the form spaces of the complex.
    form_spaces = ntuple(manifold_dim + 1) do k
        k = k - 1 # Because form ranks range from 0 to manifold_dim.
        # Get k-form basis indices, these also inform the directional degree-deficits.
        k_form_basis_idxs = get_basis_index_combinations(manifold_dim, k)
        num_form_components = length(k_form_basis_idxs)
        # Generate tuple with all the k-form finite element spaces.
        k_form_fem_spaces = ntuple(num_form_components) do component
            # By default, use direction-zero forms...
            fem_space_idxs = ones(Int, manifold_dim)
            # ...unless the basis index is present in the k-form basis indices.
            fem_space_idxs[k_form_basis_idxs[component]] .= 2
            # Build and store constituent spaces of the tensor-product FEM space.
            tp_consituent_spaces = ntuple(manifold_dim) do dim
                return fem_spaces[fem_space_idxs[dim]][dim]
            end
            # Build and return the corresponding tensor-product FEM space.
            return FunctionSpaces.TensorProductSpace(tp_consituent_spaces)
        end

        direct_sum_space = FunctionSpaces.DirectSumSpace(k_form_fem_spaces) 
         
        return FormSpace(
            k, geometry, direct_sum_space, "ω_$k"
        )
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

################################################################################
# Polar B-spline de Rham complex
################################################################################

import SparseArrays

"""
    create_polar_spline_de_rham_complex(num_elements::NTuple{2, Int}, section_spaces::NTuple{2, F}, regularities::NTuple{2, Int}, R::Float64; refine::Bool=false, geom_coeffs_tp::NTuple{2, Vector{Float64}}=undef, form_rank::Int=0) where {F <: FunctionSpaces.AbstractCanonicalSpace}

Create a polar B-spline de Rham complex.

# Arguments
- `num_elements::NTuple{2, Int}`: the number of elements in each direction.
- `section_spaces::NTuple{2, F}`: the section spaces.
- `regularities::NTuple{2, Int}`: the regularities of the B-spline spaces.
- `R::Float64`: the radius of the domain.
- `refine::Bool=false`: whether to refine the domain.
- `geom_coeffs_tp::NTuple{2, Vector{Float64}}=undef`: the geometry coefficients.
- `form_rank::Int=0`: the rank of the form.

# Returns
- `::Vector{AbstractFormSpace}`: the 3 form spaces of the complex.
- `::Vector{NTuple{N,SparseMatrixCSC{Float64,Int}} where {N}}`: the global extraction operators.
- `::NTuple{2, Array{Float64,3}}`: the geometry coefficients for the underlying tensor-product B-spline spaces.
"""
function create_polar_spline_de_rham_complex(num_elements::NTuple{2, Int}, section_spaces::F, regularities::NTuple{2, Int}, R::Float64; refine::Bool=false, geom_coeffs_tp::Union{Nothing, Array{Float64,3}}=nothing) where {F <: NTuple{2, FunctionSpaces.AbstractCanonicalSpace}}

    form_spaces = Vector{AbstractFormSpace}(undef, 3)
    global_extraction_operators = Vector{NTuple{N,SparseArrays.SparseMatrixCSC{Float64,Int}} where {N}}(undef,3)

    ##############################
    # 0-Forms
    ##############################

    # 0-form FEM space (and possibly refined control points)
    (P_sol_0, E_sol_0), (P_geom, E_geom, geom_coeffs_polar), geom_coeffs_tp, (ts_θ_0, ts_r_0) = FunctionSpaces.create_polar_spline_space_and_geometry(num_elements, section_spaces, regularities, R; refine=refine, geom_coeffs_tp=geom_coeffs_tp, form_rank=0)
    global_extraction_operators[1] = E_sol_0

    if !isnothing(ts_r_0) && !isnothing(ts_θ_0)
        # update number of elements
        num_elements = (FunctionSpaces.get_num_elements(ts_θ_0.fine_space), FunctionSpaces.get_num_elements(ts_r_0.fine_space))
    end

    # polar spline geometry
    geometry = Geometry.FEMGeometry(P_geom.component_spaces[1], geom_coeffs_polar)

    # 0-form space
    form_spaces[1] = FormSpace(0, geometry, P_sol_0, "ω_0")


    ##############################
    # 1-Forms
    ##############################

    # 1-form FEM space
    (P_sol_1, E_sol_1), _ = FunctionSpaces.create_polar_spline_space_and_geometry(num_elements, section_spaces, regularities, R; geom_coeffs_tp=geom_coeffs_tp, form_rank=1)
    global_extraction_operators[2] = E_sol_1

    # 1-form space
    form_spaces[2] = FormSpace(1, geometry, P_sol_1, "ω_1")


    ##############################
    # 2-Forms
    ##############################

    # 2-form FEM space
    (P_sol_2, E_sol_2), _ = FunctionSpaces.create_polar_spline_space_and_geometry(num_elements, section_spaces, regularities, R; geom_coeffs_tp=geom_coeffs_tp, form_rank=2)
    global_extraction_operators[3] = E_sol_2

    # 2-form space
    form_spaces[3] = FormSpace(2, geometry, P_sol_2, "ω_2")

    return form_spaces, global_extraction_operators, geom_coeffs_tp
end

"""
    create_polar_spline_de_rham_complex(num_elements::NTuple{2, Int}, degrees::NTuple{2, Int}, regularities::NTuple{2, Int}, R::Float64; refine::Bool=false, geom_coeffs_tp::NTuple{2, Vector{Float64}}=undef, form_rank::Int=0)

Create a polar B-spline de Rham complex.

# Arguments
- `num_elements::NTuple{2, Int}`: the number of elements in each direction.
- `degrees::NTuple{2, Int}`: the polynomial degrees in angular and radial directions.
- `regularities::NTuple{2, Int}`: the regularities of the B-spline spaces.
- `R::Float64`: the radius of the domain.
- `refine::Bool=false`: whether to refine the domain.
- `geom_coeffs_tp::NTuple{2, Vector{Float64}}=undef`: the geometry coefficients.
- `form_rank::Int=0`: the rank of the form.

# Returns
- `::Vector{AbstractFormSpace}`: the 3 form spaces of the complex.
- `::Vector{NTuple{N,SparseMatrixCSC{Float64,Int}} where {N}}`: the global extraction operators.
- `::NTuple{2, Array{Float64,3}}`: the geometry coefficients for the underlying tensor-product B-spline spaces.
"""
function create_polar_spline_de_rham_complex(num_elements::NTuple{2, Int}, degrees::NTuple{2, Int}, regularities::NTuple{2, Int}, R::Float64; refine::Bool=false, geom_coeffs_tp::Union{Nothing, Array{Float64,3}}=nothing)

    form_spaces = Vector{AbstractFormSpace}(undef, 3)
    global_extraction_operators = Vector{NTuple{N,SparseMatrixCSC{Float64,Int}} where {N}}(undef,3)

    ##############################
    # 0-Forms
    ##############################

    # 0-form FEM space (and possibly refined control points)
    (P_sol_0, E_sol_0), (P_geom, E_geom, geom_coeffs_polar), geom_coeffs_tp, (ts_θ_0, ts_r_0) = FunctionSpaces.create_polar_spline_space_and_geometry(num_elements, degrees, regularities, R; refine=refine, geom_coeffs_tp=geom_coeffs_tp, form_rank=0)
    global_extraction_operators[1] = E_sol_0

    if !isnothing(ts_r_0) && !isnothing(ts_θ_0)
        # update number of elements
        num_elements = (FunctionSpaces.get_num_elements(ts_θ_0.fine_space), FunctionSpaces.get_num_elements(ts_r_0.fine_space))
    end

    # polar spline geometry
    geometry = Geometry.FEMGeometry(P_geom.component_spaces[1], geom_coeffs_polar)

    # 0-form space
    form_spaces[1] = FormSpace(0, geometry, P_sol_0, "ω_0")


    ##############################
    # 1-Forms
    ##############################

    # 1-form FEM space
    (P_sol_1, E_sol_1), _ = FunctionSpaces.create_polar_spline_space_and_geometry(num_elements, degrees, regularities, R; geom_coeffs_tp=geom_coeffs_tp, form_rank=1)
    global_extraction_operators[2] = E_sol_1

    # 1-form space
    form_spaces[2] = FormSpace(1, geometry, P_sol_1, "ω_1")


    ##############################
    # 2-Forms
    ##############################

    # 2-form FEM space
    (P_sol_2, E_sol_2), _ = FunctionSpaces.create_polar_spline_space_and_geometry(num_elements, degrees, regularities, R; geom_coeffs_tp=geom_coeffs_tp, form_rank=2)
    global_extraction_operators[3] = E_sol_2

    # 2-form space
    form_spaces[3] = FormSpace(2, geometry, P_sol_2, "ω_2")

    return form_spaces, global_extraction_operators, geom_coeffs_tp
end
