# @doc raw"""
#     GTBSplineSpace(canonical_spaces::NTuple{m,CanonicalFiniteElementSpace}, regularity::Vector{Int}) where {m}

# Constructs a GTBSplineSpace from a tuple of canonical finite element spaces and a vector of regularity conditions.

# # Arguments
# - `canonical_spaces::NTuple{m,CanonicalFiniteElementSpace}`: A tuple of `m` canonical finite element spaces.
# - `regularity::Vector{Int}`: A vector of regularity conditions at the interfaces between the spaces.

# # Returns
# - `UnstructuredSpace`: The constructed GTBSplineSpace.

# # Throws
# - `ArgumentError`: If the number of regularity conditions does not match the number of interfaces.
# - `ArgumentError`: If the minimal polynomial degree of any pair of adjacent spaces is less than the corresponding regularity condition.
# """
# function GTBSplineSpace(canonical_spaces::NTuple{m,CanonicalFiniteElementSpace}, regularity::Vector{Int}) where {m}
#     # Check if the number of regularity conditions matches the number of interfaces
#     if length(regularity) != m
#         msg1 = "Number of regularity conditions should be equal to the number of bspline interfaces."
#         msg2 = " You have $(m) interfaces and $(length(regularity)) regularity conditions."
#         throw(ArgumentError(msg1 * msg2))
#     end

#     # Check if the polynomial degree of each pair of adjacent spaces is sufficient for the regularity condition
#     for i in 1:m
#         j = i
#         k = i + 1
#         if i == m
#             k = 1
#         end
#         polynomial_degree = min(get_polynomial_degree(canonical_spaces[j]), get_polynomial_degree(canonical_spaces[k]))
#         if polynomial_degree < regularity[i]
#             msg1 = "Minimal polynomial degrees must be greater than or equal to the regularity."
#             msg2 = " The minimal degree is $polynomial_degree and there is regularity $regularity[i] in index $i."
#             throw(ArgumentError(msg1 * msg2))
#         end
#     end

#     # Construct and return the UnstructuredSpace
#     if regularity[m] > -1 # periodic space: non-default dof partitioning
#         return UnstructuredSpace(canonical_spaces, extract_gtbspline_to_canonical(canonical_spaces, regularity), 0, 0, Dict("regularity" => regularity))
#     else
#         return UnstructuredSpace(canonical_spaces, extract_gtbspline_to_canonical(canonical_spaces, regularity), Dict("regularity" => regularity))
#     end
# end

@doc raw"""
    GTBSplineSpace(spline_spaces::NTuple{m,T}, regularity::Vector{Int}) where {m, T <: Union{BSplineSpace, RationalFiniteElementSpace}}

Constructs a GTBSplineSpace from a tuple of NURBS or B-spline spaces and a vector of regularity conditions.

# Arguments
- `spline_spaces::NTuple{m,T}`: A tuple of `m` NURBS or B-spline spaces.
- `regularity::Vector{Int}`: A vector of regularity conditions at the interfaces between the spaces.

# Returns
- `UnstructuredSpace`: The constructed GTBSplineSpace.

# Throws
- `ArgumentError`: If the number of regularity conditions does not match the number of interfaces.
- `ArgumentError`: If the minimal polynomial degree of any pair of adjacent spaces is less than the corresponding regularity condition.
"""
function GTBSplineSpace(spline_spaces::NTuple{m,T}, regularity::Vector{Int}) where {m, T <: Union{BSplineSpace, RationalFiniteElementSpace}}
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
        polynomial_degree = min(get_polynomial_degree(spline_spaces[j], 1), get_polynomial_degree(spline_spaces[k], 1))
        if polynomial_degree < regularity[i]
            msg1 = "Minimal polynomial degrees must be greater than or equal to the regularity."
            msg2 = " The minimal degree is $polynomial_degree and there is regularity $regularity[i] in index $i."
            throw(ArgumentError(msg1 * msg2))
        end
    end

    # Construct and return the UnstructuredSpace
    if regularity[m] > -1 # periodic space: non-default dof partitioning
        return UnstructuredSpace(spline_spaces, extract_gtbspline_to_bspline(spline_spaces, regularity), 0, 0, Dict("regularity" => regularity))
    else
        return UnstructuredSpace(spline_spaces, extract_gtbspline_to_bspline(spline_spaces, regularity), Dict("regularity" => regularity))
    end
end 
