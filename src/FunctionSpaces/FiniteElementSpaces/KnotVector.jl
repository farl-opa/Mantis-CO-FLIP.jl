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
    function KnotVector(patch_1d::Mesh.Patch1D, polynomial_degree::Int, multiplicity::Int)
       return KnotVector(patch_1d, polynomial_degree, [multiplicity])
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
    return (greville_points,)
end