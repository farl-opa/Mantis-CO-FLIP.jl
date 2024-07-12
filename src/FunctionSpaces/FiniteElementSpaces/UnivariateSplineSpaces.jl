"""
    KnotVector

1-dimensional knot vector. 'patch_1d' determines the location of the knots and
'multiplicity' the number of times each of them is repeated.

# Fields
- `patch_1d::Mesh.Patch1D`: Location of each knot.
- `polynomial_degree::Int`: Polynomial degree.
- `multiplicity::Vector{Int}`: Repetition of each knot.
"""
struct KnotVector
    patch_1d::Mesh.Patch1D
    polynomial_degree::Int
    multiplicity::Vector{Int}

    function KnotVector(patch_1d::Mesh.Patch1D, polynomial_degree::Int, multiplicity::Vector{Int})
        # Ensure the number of knots and multiplicity are consistent
        if (size(patch_1d)+1) != length(multiplicity)
            msg1 = "Number of knots and multiplicity must be the same."
            msg2 = " Knots has length $(size(patch_1d)+1) and multiplicity has length $(length(multiplicity))."
            throw(ArgumentError(msg1*msg2))
        end

        # Ensure all multiplicities are greater than 0
        if any(multiplicity .<= 0)
            error_index = findfirst(multiplicity .<= 0)
            msg1 = "Multiplicity must be greater or equal to 1."
            msg2 = " At index $error_index, the multiplicity is $(multiplicity[error_index])."
            throw(ArgumentError(msg1*msg2))
        end

        new(patch_1d, polynomial_degree, multiplicity)
    end
end

"""
    create_knot_vector(patch_1d::Mesh.Patch1D, p::Int, breakpoint_condition::Vector{Int}, condition_type::String)

Creates a uniform knot vector corresponding to B-splines basis functions of polynomial degree `p` and continuity `regularity[i]` on `patch_1d.breakpoint[i]`.

# Arguments
- `patch_1d::Mesh.Patch1D`: Location of each breakpoint.
- `p::Int`: Degree of the polynomial (``p ≥ 0``).
- `breakpoint_condition::Vector{Int}`: Either the regularity or multiplicity of each breakpoint.
- `condition_type::String`: Determines whether `breakpoint_condition` determines the regularity or multiplicity.

# Returns 
- `::KnotVector`: Knot vector.
"""
function create_knot_vector(patch_1d::Mesh.Patch1D, p::Int, breakpoint_condition::Vector{Int}, condition_type::String)
    if condition_type == "regularity"
        multiplicity = ones(Int, size(patch_1d)+1)

        # Calculate multiplicity based on regularity
        for i in eachindex(breakpoint_condition)
            multiplicity[i] = p - breakpoint_condition[i]
        end

        return KnotVector(patch_1d, p, multiplicity)
    elseif condition_type == "multiplicity"
        return KnotVector(patch_1d, p, breakpoint_condition)
    else 
        msg1 = "Allowed breakpoint conditions are `regularity` or `multiplicity`."
        msg2 = " The given condition is $condition_type."
        throw(ArgumentError(msg1*msg2))
    end
end

"""
    get_element_size(knot_vector::KnotVector, element_id::Int)

Returns the size of the element specified by `element_id`.

# Arguments
- `knot_vector::KnotVector`: The B-Spline function space.
- `element_id::Int`: The id of the element.

# Returns
- `::Float64`: The size of the element.
"""
function get_element_size(knot_vector::KnotVector, element_id::Int)
    return Mesh.get_element_size(knot_vector.patch_1d, element_id)
end

"""
    get_knot_vector_length(knot_vector::KnotVector)

Determines the length of `knot_vector` by summing the multiplicities of each knot vector.

# Arguments
- `knot_vector::KnotVector`: Knot vector for length calculation.

# Returns
- `::Int`: Length of the knot vector.
"""
function get_knot_vector_length(knot_vector::KnotVector)
    return sum(knot_vector.multiplicity)
end

"""
    convert_knot_to_breakpoint_idx(knot_vector::KnotVector, knot_index::Int)

Retrieves the breakpoint index corresponding to `knot_vector` at `knot_index`, i.e., the index of the vector where every `breakpoint[i]` appears `knot_vector.multiplicity[i]`-times.

# Arguments
- `knot_vector::KnotVector`: Knot vector for length calculation.
- `knot_index::Int`: Index in the knot vector.

# Returns
- `::Int`: Index of breakpoint corresponding to `knot_index`.
"""
function convert_knot_to_breakpoint_idx(knot_vector::KnotVector, knot_index::Int)
    return findfirst(idx -> idx >= knot_index, cumsum(knot_vector.multiplicity))
end

"""
    get_first_knot_index(knot_vector::KnotVector, breakpoint_index::Int)

Get the index of the first knot corresponding to a given breakpoint.

# Arguments
- `knot_vector::KnotVector`: The knot vector.
- `breakpoint_index::Int`: Index of the breakpoint.

# Returns
- `::Int`: Index of the first knot for the given breakpoint.
"""
function get_first_knot_index(knot_vector::KnotVector, breakpoint_index::Int)
    return cumsum(knot_vector.multiplicity)[breakpoint_index] - knot_vector.multiplicity[breakpoint_index] + 1
end

"""
    get_last_knot_index(knot_vector::KnotVector, breakpoint_index::Int)

Get the index of the last knot corresponding to a given breakpoint.

# Arguments
- `knot_vector::KnotVector`: The knot vector.
- `breakpoint_index::Int`: Index of the breakpoint.

# Returns
- `::Int`: Index of the last knot for the given breakpoint.
"""
function get_last_knot_index(knot_vector::KnotVector, breakpoint_index::Int)
    return cumsum(knot_vector.multiplicity)[breakpoint_index]
end

"""
    get_knot_value(knot_vector::KnotVector, knot_index::Int)

Retrieves the breakpoint corresponding to `knot_vector` at `knot_index`, i.e., the index 
of the vector where every `breakpoint[i]` appears `knot_vector.multiplicity[i]`-times.

# Arguments
- `knot_vector::KnotVector`: Knot vector for length calculation.
- `knot_index::Int`: Index in the knot vector.

# Returns
- `::Float64`: Breakpoint corresponding to `knot_index`.
"""
function get_knot_value(knot_vector::KnotVector, knot_index::Int)
    idx = convert_knot_to_breakpoint_idx(knot_vector, knot_index)
    return Mesh.get_breakpoints(knot_vector.patch_1d)[idx]
end

"""
    get_knot_multiplicity(knot_vector::KnotVector, knot_index::Int)

Retrieves the multiplicity of the breakpoint corresponding to `knot_vector` at `knot_index`, i.e., the index 
of the vector where every `breakpoint[i]` appears `knot_vector.multiplicity[i]`-times.

# Arguments
- `knot_vector::KnotVector`: Knot vector for length calculation.
- `knot_index::Int`: Index in the knot vector.

# Returns
- `::Int`: Multiplicity of the breakpoint corresponding to `knot_index`.
"""
function get_knot_multiplicity(knot_vector::KnotVector, knot_index::Int)
    index = convert_knot_to_breakpoint_idx(knot_vector, knot_index)
    return knot_vector.multiplicity[index]
end

"""
    get_local_knot_vector(knot_vector::KnotVector, basis_id::Int)

Returns the local knot vector necessary to characterize the B-spline identified by `basis_id`.

# Arguments
- `knot_vector::KnotVector`: The knot vector of the full B-spline basis.
- `basis_id::Int`: The id of the B-spline.

# Returns
- `::KnotVector`: The knot vector of the B-spline identified by `basis_id`.
"""
function get_local_knot_vector(knot_vector::KnotVector, basis_id::Int)
    local_idx = get_breakpoint_index(knot_vector, basis_id):get_breakpoint_index(knot_vector, basis_id+knot_vector.polynomial_degree+1)

    local_patch = Mesh.Patch1D(Mesh.get_breakpoints(knot_vector.patch_1d)[local_idx])
    local_multiplicity = knot_vector.multiplicity[local_idx]

    return KnotVector(local_patch, knot_vector.polynomial_degree, local_multiplicity)
end

"""
    get_greville_points(knot_vector::KnotVector)

Compute the Greville points for the given knot vector.

# Arguments
- `knot_vector::KnotVector`: The knot vector.

# Returns
- `::Vector{Float64}`: Vector of Greville points.
"""
function get_greville_points(knot_vector::KnotVector)
    p = knot_vector.polynomial_degree
    n = sum(knot_vector.multiplicity) - p - 1
    cm = cumsum(knot_vector.multiplicity)
    greville_points = zeros(n)
    for i in 1:n
        # Starting breakpoint index
        id0 = findfirst(cm .>= i+1)
        id1 = findfirst(cm .>= i+p)
        if id1 > id0
            pt = knot_vector.patch_1d.breakpoints[id0] * (cm[id0] - i)
            for j = id0+1:id1-1
                pt += knot_vector.patch_1d.breakpoints[j] * knot_vector.multiplicity[j]
            end
            pt += knot_vector.patch_1d.breakpoints[id1] * (i + p - cm[id1-1])
        else
            pt = knot_vector.patch_1d.breakpoints[id0] * p
        end
        greville_points[i] = pt / p
    end
    return greville_points
end


"""
    BSplineSpace

Structure containing information about a univariate B-Spline function space defined on `patch_1d::Mesh.Patch1D`,
with given `polynomial_degree` and `regularity` per breakpoint.

# Fields
- `knot_vector::KnotVector`: 1-dimensional knot vector.
- `extraction_op::ExtractionOperator`: Stores extraction coefficients and basis indices.
- `polynomials::Bernstein`: Reference Bernstein polynomials.
- `boundary_dof_indices::Vector{Int}`: Indices of boundary degrees of freedom.
"""
struct BSplineSpace <: AbstractFiniteElementSpace{1}
    knot_vector::KnotVector
    extraction_op::ExtractionOperator
    polynomials::Bernstein
    boundary_dof_indices::Vector{Int}

    function BSplineSpace(patch_1d::Mesh.Patch1D, polynomial_degree::Int, regularity::Vector{Int})
        # Check for errors in the inputs
        if polynomial_degree < 0
            msg1 = "Polynomial degree must be greater or equal to 0."
            msg2 = " The degree is $polynomial_degree."
            throw(ArgumentError(msg1 * msg2))
        end

        if (size(patch_1d) + 1) != length(regularity)
            msg1 = "Number of regularity conditions should be equal to the number of breakpoints."
            msg2 = " You have $(size(patch_1d) + 1) breakpoints and $(length(regularity)) regularity conditions."
            throw(ArgumentError(msg1 * msg2))
        end

        for i in eachindex(regularity)
            if polynomial_degree <= regularity[i]
                msg1 = "Polynomial degree must be greater than regularity."
                msg2 = " The degree is $polynomial_degree and there is regularity $regularity[i] in index $i."
                throw(ArgumentError(msg1 * msg2))
            end
        end

        # Create the knot vector
        knot_vector = create_knot_vector(patch_1d, polynomial_degree, regularity, "regularity")

        # Extract B-spline to Bernstein coefficients
        extraction_op = extract_bspline_to_bernstein(knot_vector)

        # Get the dimension of the B-spline space
        bspline_dim = get_dim(extraction_op)

        # Initialize the BSplineSpace struct
        new(knot_vector, extraction_op, Bernstein(polynomial_degree), [1, bspline_dim])
    end
end

"""
    get_extraction(bspline::BSplineSpace, element_id::Int)

Returns the supported basis functions of `bspline` on the element specified by `element_id` and their extraction coefficients.

# Arguments
- `bspline::BSplineSpace`: A univariate B-Spline function space.
- `element_id::Int`: The id of the element.

# Returns
- `::Tuple{Matrix{Float64}, Vector{Int}}`: The extraction coefficients and supported bases.
"""
function get_extraction(bspline::BSplineSpace, element_id::Int)
    return get_extraction(bspline.extraction_op, element_id)
end

"""
    get_polynomials(bspline::BSplineSpace)

Returns the reference Bernstein polynomials of `bspline`.

# Arguments
- `bspline::BSplineSpace`: A univariate B-Spline function space.

# Returns
- `::Bernstein`: Bernstein polynomials.
"""
function get_polynomials(bspline::BSplineSpace)
    return bspline.polynomials
end

"""
    get_local_basis(bspline::BSplineSpace, element_id::Int, xi::Vector{Float64}, nderivatives::Int)

Compute the local basis functions and their derivatives for a B-spline element.

# Arguments
- `bspline::BSplineSpace`: A univariate B-Spline function space.
- `element_id::Int`: The id of the element.
- `xi::Vector{Float64}`: Evaluation points.
- `nderivatives::Int`: Number of derivatives to compute.

# Returns
- `::Matrix{Float64}`: Local basis functions and their derivatives.
"""
function get_local_basis(bspline::BSplineSpace, element_id::Int, xi::Vector{Float64}, nderivatives::Int)
    local_basis = bspline.polynomials(xi, nderivatives)
    return local_basis
end

"""
    get_local_basis(bspline::BSplineSpace, element_id::Int, xi::NTuple{1, Vector{Float64}}, nderivatives::Int)

Compute the local basis functions and their derivatives for a B-spline element.

# Arguments
- `bspline::BSplineSpace`: A univariate B-Spline function space.
- `element_id::Int`: The id of the element.
- `xi::NTuple{1, Vector{Float64}}`: Evaluation points.
- `nderivatives::Int`: Number of derivatives to compute.

# Returns
- `::Matrix{Float64}`: Local basis functions and their derivatives.
"""
function get_local_basis(bspline::BSplineSpace, element_id::Int, xi::NTuple{1, Vector{Float64}}, nderivatives::Int)
    return get_local_basis(bspline, element_id, xi[1], nderivatives)
end

"""
    get_polynomial_degree(bspline::BSplineSpace)

Returns the degree of B-splines.

# Arguments
- `bspline::BSplineSpace`: A univariate B-Spline function space.

# Returns
- `polynomial_degree`: Polynomial degree.
"""
function get_polynomial_degree(bspline::BSplineSpace)
    return bspline.knot_vector.polynomial_degree
end

function get_polynomial_degree(bspline::BSplineSpace, ::Int)
    return bspline.knot_vector.polynomial_degree
end

"""
    get_dim(bspline::BSplineSpace)

Returns the dimension of the univariate function space `bspline`.

# Arguments
- `bspline::BSplineSpace`: The B-Spline function space.

# Returns
- `::Int`: The dimension of the B-Spline space.
"""
function get_dim(bspline::BSplineSpace)
    @assert get_dim(bspline.extraction_op) == size(bspline.knot_vector.patch_1d) * (bspline.knot_vector.polynomial_degree + 1) + sum(bspline.knot_vector.multiplicity .- (bspline.knot_vector.polynomial_degree + 1)) "B-spline dimension incorrect."
    return get_dim(bspline.extraction_op)
end

"""
    get_patch(bspline::BSplineSpace)

Returns the patch of the univariate function space `bspline`.

# Arguments
- `bspline::BSplineSpace`: The B-Spline function space.

# Returns
- `::Mesh.Patch1D`: The patch of the B-Spline space.
"""
function get_patch(bspline::BSplineSpace)
    return bspline.knot_vector.patch_1d
end

"""
    get_multiplicity_vector(bspline::BSplineSpace)

Returns the multiplicities of the knot vector associated with the univariate function space `bspline`.

# Arguments
- `bspline::BSplineSpace`: The B-Spline function space.

# Returns
- `::Vector{Int}`: The multiplicity of the knot vector associated with the B-Spline space.
"""
function get_multiplicity_vector(bspline::BSplineSpace)
    return bspline.knot_vector.multiplicity
end

"""
    get_num_elements(bspline::BSplineSpace)

Returns the number of elements in the underlying partition.

# Arguments
- `bspline::BSplineSpace`: The B-Spline function space.

# Returns
- `::Int`: The number of elements.
"""
function get_num_elements(bspline::BSplineSpace)
    return size(bspline.knot_vector.patch_1d)
end

"""
    get_element_size(bspline::BSplineSpace, element_id::Int)

Returns the size of the element specified by `element_id`.

# Arguments
- `bspline::BSplineSpace`: The B-Spline function space.
- `element_id::Int`: The id of the element.

# Returns
- `::Float64`: The size of the element.
"""
function get_element_size(bspline::BSplineSpace, element_id::Int)
    return get_element_size(bspline.knot_vector, element_id)
end

"""
    get_support(bspline::BSplineSpace, basis_id::Int)

Returns the elements where the B-spline given by `basis_id` is supported.

# Arguments
- `bspline::BSplineSpace`: The B-Spline function space.
- `basis_id::Int`: The id of the basis function.

# Returns
- `::UnitRange{Int}`: The support of the basis function.
"""
function get_support(bspline::BSplineSpace, basis_id::Int)
    first_element = convert_knot_to_breakpoint_idx(bspline.knot_vector, basis_id)
    last_element = convert_knot_to_breakpoint_idx(bspline.knot_vector, basis_id + bspline.knot_vector.polynomial_degree + 1) - 1
    return first_element:last_element
end

"""
    get_max_local_dim(bspline::BSplineSpace)

Returns the maximum local dimension of the B-Spline space.

# Arguments
- `bspline::BSplineSpace`: The B-Spline function space.

# Returns
- `::Int`: The maximum local dimension.
"""
function get_max_local_dim(bspline::BSplineSpace)
    return bspline.knot_vector.polynomial_degree + 1
end

"""
    set_boundary_dof_indices(bspline::BSplineSpace, indices::Vector{Int})

Set the boundary degrees of freedom indices for the B-Spline space.

# Arguments
- `bspline::BSplineSpace`: The B-Spline function space.
- `indices::Vector{Int}`: Indices of boundary degrees of freedom.

# Returns
- `nothing`
"""
function set_boundary_dof_indices(bspline::BSplineSpace, indices::Vector{Int})
    bspline.boundary_dof_indices = indices
    return nothing
end

"""
    get_boundary_dof_indices(bspline::BSplineSpace)

Get the boundary degrees of freedom indices for the B-Spline space.

# Arguments
- `bspline::BSplineSpace`: The B-Spline function space.

# Returns
- `::Vector{Int}`: Indices of boundary degrees of freedom.
"""
function get_boundary_dof_indices(bspline::BSplineSpace)
    return bspline.boundary_dof_indices
end


@doc raw"""
    GTBSplineSpace(canonical_spaces::NTuple{m,CanonicalFiniteElementSpace}, regularity::Vector{Int}) where {m}

Constructs a GTBSplineSpace from a tuple of canonical finite element spaces and a vector of regularity conditions.

# Arguments
- `canonical_spaces::NTuple{m,CanonicalFiniteElementSpace}`: A tuple of `m` canonical finite element spaces.
- `regularity::Vector{Int}`: A vector of regularity conditions at the interfaces between the spaces.

# Returns
- `UnstructuredSpace`: The constructed GTBSplineSpace.

# Throws
- `ArgumentError`: If the number of regularity conditions does not match the number of interfaces.
- `ArgumentError`: If the minimal polynomial degree of any pair of adjacent spaces is less than the corresponding regularity condition.
"""
function GTBSplineSpace(canonical_spaces::NTuple{m,CanonicalFiniteElementSpace}, regularity::Vector{Int}) where {m}
    # Check if the number of regularity conditions matches the number of interfaces
    if length(regularity) != m
        msg1 = "Number of regularity conditions should be equal to the number of bspline interfaces."
        msg2 = " You have $(m) interfaces and $(length(regularity)) regularity conditions."
        throw(ArgumentError(msg1 * msg2))
    end

    # Check if the polynomial degree of each pair of adjacent spaces is sufficient for the regularity condition
    for i in 1:m
        j = i
        k = i + 1
        if i == m
            k = 1
        end
        polynomial_degree = min(get_polynomial_degree(canonical_spaces[j]), get_polynomial_degree(canonical_spaces[k]))
        if polynomial_degree < regularity[i]
            msg1 = "Minimal polynomial degrees must be greater than or equal to the regularity."
            msg2 = " The minimal degree is $polynomial_degree and there is regularity $regularity[i] in index $i."
            throw(ArgumentError(msg1 * msg2))
        end
    end

    # Construct and return the UnstructuredSpace
    return UnstructuredSpace(canonical_spaces, extract_gtbspline_to_canonical(canonical_spaces, regularity), Dict("regularity" => regularity))
end

@doc raw"""
    GTBSplineSpace(nurbs::NTuple{m,T}, regularity::Vector{Int}) where {m, T <: Union{BSplineSpace, RationalFiniteElementSpace{1,F} where {F <: Union{BSplineSpace, CanonicalFiniteElementSpace}}}}

Constructs a GTBSplineSpace from a tuple of NURBS or B-spline spaces and a vector of regularity conditions.

# Arguments
- `nurbs::NTuple{m,T}`: A tuple of `m` NURBS or B-spline spaces.
- `regularity::Vector{Int}`: A vector of regularity conditions at the interfaces between the spaces.

# Returns
- `UnstructuredSpace`: The constructed GTBSplineSpace.

# Throws
- `ArgumentError`: If the number of regularity conditions does not match the number of interfaces.
- `ArgumentError`: If the minimal polynomial degree of any pair of adjacent spaces is less than the corresponding regularity condition.
"""
function GTBSplineSpace(nurbs::NTuple{m,T}, regularity::Vector{Int}) where {m, T <: Union{BSplineSpace, RationalFiniteElementSpace{1,F} where {F <: Union{BSplineSpace, CanonicalFiniteElementSpace}}}}
    # Check if the number of regularity conditions matches the number of interfaces
    if length(regularity) != m
        msg1 = "Number of regularity conditions should be equal to the number of bspline interfaces."
        msg2 = " You have $(m) interfaces and $(length(regularity)) regularity conditions."
        throw(ArgumentError(msg1 * msg2))
    end

    # Check if the polynomial degree of each pair of adjacent spaces is sufficient for the regularity condition
    for i in 1:m
        j = i
        k = i + 1
        if i == m
            k = 1
        end
        polynomial_degree = min(get_polynomial_degree(nurbs[j], 1), get_polynomial_degree(nurbs[k], 1))
        if polynomial_degree < regularity[i]
            msg1 = "Minimal polynomial degrees must be greater than or equal to the regularity."
            msg2 = " The minimal degree is $polynomial_degree and there is regularity $regularity[i] in index $i."
            throw(ArgumentError(msg1 * msg2))
        end
    end

    # Construct and return the UnstructuredSpace
    return UnstructuredSpace(nurbs, extract_gtbspline_to_nurbs(nurbs, regularity), Dict("regularity" => regularity))
end
