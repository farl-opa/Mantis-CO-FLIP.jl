"""
    struct KnotVector

1-dimensional knot vector. 'breakpoints' determines the location of the knots and
'multiplicity' the number of times each of them is repeated.

# Fields
- `breakpoints::Vector{Float64}`: Location of each knot.
- `multiplicity::Vector{Int}`: Repetition of each knot.
"""
struct KnotVector
    breakpoints::Vector{Float64}
    polynomial_degree::Int
    multiplicity::Vector{Int}

    function KnotVector(breakpoints::Vector{Float64}, polynomial_degree::Int, multiplicity::Vector{Int})
        if length(breakpoints) != length(multiplicity)
            msg1 = "Number of breakpoints and multiplicity must be the same."
            msg2 = " breakpoints as length $(length(breakpoints)) and multiplicity has length $(length(multiplicity))."
            throw(ArgumentError(msg1*msg2))
        end

        if any(multiplicity .<= 0)
            error_index = findfirst(multiplicity .<= 0)
            msg1 = "multiplicity must be greater or equal to 1."
            msg2 = " In index $error_index, the multiplicity is $(multiplicity[error_index])."
            throw(ArgumentError(msg1*msg2))
        end

        new(breakpoints, polynomial_degree, multiplicity)
    end
end

"""
    create_knot_vector(breakpoints::Vector{Float64}, p::Int, regularity::Vector{Int})

Creates a uniform knot vector corresponding to `nel*p - sum(k) + 1` 
B-splines basis functions of polynomial degree `p` and continuity `k[i]` over `nel` 
equally spaced elements in the interval between`left_endpoint` and `right_endpoint`. 
See https://en.wikipedia.org/wiki/B-spline#Definition.

# Arguments
- `left_endpoint::Float64`: first value of the knot vector.
- `right_endpoint::Float64`: last value of the knot vector.
- `nel::Int`: number of elements.
- `p::Int`: degree of the polynomial (``p \\geq 0``).
- `k::Vector{Int}`: continuity at the interfaces between elements. (`` -1 \\leq k[i] \\leq p``).

# Returns 
- `::KnotVector`: uniform knot vector.
"""
function create_knot_vector(breakpoints::Vector{Float64}, p::Int, regularity::Vector{Int})
    multiplicity = ones(Int, length(breakpoints))

    for i in eachindex(regularity)
        multiplicity[i+1] = p - regularity[i]
    end

    return KnotVector(breakpoints, p, multiplicity)
end

function create_knot_vector(breakpoints::Vector{Float64}, p::Int, multiplicity::Vector{Int})
    return KnotVector(breakpoints, p, multiplicity)
end

"""
    struct BSplineSpace

Structure containing information about scalar `n`-dimensional B-Spline basis functions defined on `patch`,
with given `polynomial_degree` and `regularity` conditions in each dimension. Note that the
`polynomial_degree` is the same for every element in each dimension, but the `regularity` is
specific to each element interface. Also, an open knot vector is assumed for this structure, so,
`length(regularity)` should be equal to `size(patch)[d] - 1`.

`extract_bezier_representation()` can be called on this structure to retrieve the corresponding
extraction coefficients.

# Fields
- `patch::Mesh.Patch{n}`: Patch where the B-Spline basis functions are defined.
- `polynomial_degree::NTuple{n, Int}`: Polynomial degree in each dimension.
- `regularity::NTuple{n, Vector{Int}}`: Regularity for each element interface and dimension.
"""
struct BSplineSpace<:AbstractFunctionSpace{1}
    knot_vector::KnotVector

    function BSplineSpace(breakpoints::Vector{Float64}, polynomial_degree::Int, regularity::Vector{Int})
        # Check for errors in the construction 
        if polynomial_degree <0
            msg1 = "Polynomial degree must be greater or equal than 0."
            msg2 = " The degree is $polynomial_degree."
            throw(ArgumentError(msg1*msg2))
        end
        if length(breakpoints) != length(regularity)
            msg1 = "Number of regularity conditions should be equal to the number of breakpoints."
            msg2 = " You have $(length(breakpoints)) breakpoints and $(length(regularity)) regularity conditions."
            throw(ArgumentError(msg1*msg2))
        end
        for i in eachindex(regularity)
            if polynomial_degree <= regularity[i]
                msg1 = "Polynomial degree must be greater than regularity."
                msg2 = " The degree is $polynomial_degree and there is regularity $regularity[i] in index $i."
                throw(ArgumentError(msg1*msg2))
            end
        end
        
        new(create_knot_vector(breakpoints, p, regularity))        
    end
end