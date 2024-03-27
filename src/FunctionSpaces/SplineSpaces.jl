"""
    struct KnotVector

1-dimensional knot vector. 'breakpoints' determines the location of the knots and
'multiplicity' the number of times each of them is repeated.

# Fields
- `breakpoints::Vector{Float64}`: Location of each knot.
- `polynomial_degree::Int`: Polynomial degree.
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

Creates a uniform knot vector corresponding to B-splines basis functions of polynomial degree `p` 
and continuity `regularity[i]` on `breakpoint[i]`. 
See https://en.wikipedia.org/wiki/B-spline#Definition.

# Arguments
- `breakpoints::Vector{Float64}`: Location of each breakpoint.
- `p::Int`: degree of the polynomial (``p \\geq 0``).
- `breakpoint_condition::Vector{Int}`: Either the regularity or multiplicity of each breakpoint.
- `condition_type::String`: Determines whether `breakpoint_condition` determines the regularity or multiplicity.

# Returns 
- `::KnotVector`: Uniform knot vector.
"""
function create_knot_vector(breakpoints::Vector{Float64}, p::Int, breakpoint_condition::Vector{Int}, condition_type::String)
    if condition_type == "regularity"
        multiplicity = ones(Int, length(breakpoints))

        for i in eachindex(breakpoint_condition)
            multiplicity[i] = p - breakpoint_condition[i]
        end

        return KnotVector(breakpoints, p, multiplicity)
    elseif condition_type == "multiplicity"
        return KnotVector(breakpoints, p, vec)
    else 
        msg1 = "Allowed breakpoint conditions are `regularity` or `multiplicity`."
        msg2 = " The given condition is $condition_type."
        throw(ArgumentError(msg1*msg2))
    end
end

"""
    struct BSplineSpace

Structure containing information about a univariate B-Spline function space defined on `breakpoints`,
with given `polynomial_degree` and `regularity` per breakpoint.

# Fields
- `breakpoints::Vector{Float64}`: Location of each breakpoint.
- `polynomial_degree::Int`: Polynomial degree.
- `regularity::Vector{Int}`: Regularity of each breakpoint.
"""
struct BSplineSpace<:AbstractFunctionSpace{1}
    knot_vector::KnotVector
    extraction_coefficients::Array{Float64, 3}
    polynomials::Polynomials.Bernstein
    supported_basis::Matrix{Int}

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
        
        knot_vector = create_knot_vector(breakpoints, polynomial_degree, regularity, "regularity")

        knot_vector = create_knot_vector(breakpoints, polynomial_degree, regularity, "regularity")

        supported_basis = zeros(Int, (length(breakpoints)-1, polynomial_degree+1))
        supported_basis[1,:] = 1:polynomial_degree+1

        for el = 2:length(breakpoints)-1
            supported_basis[el,:] .= (@view supported_basis[el-1,:]) .+ knot_vector.multiplicity[el]
        end
        
        new(knot_vector, extract_bezier_representation(knot_vector), Polynomials.Bernstein(polynomial_degree), supported_basis)        
    end
end

""" 
    get_extraction_coefficients_and_indices(bspline::BSplineSpace, element_id::Int)

Returns the supported bases functions of `bspline` on the element specified by `element_id` and their extraction coefficients.

# Arguments
- `bspline::BSplineSpace`: A univariate B-Spline function space.
- `element_id::Int`: The id of the element.
# Returns
- `::@views (Matrix{Float64}, Vector{Int})`: The extraction coefficients and supported bases.
"""
function get_extraction_coefficients_and_indices(bspline::BSplineSpace, element_id::Int)
    return @views bspline.extraction_coefficients[:,:,element_id], bspline.supported_basis[el,:]
end

""" 
    get_polynomials(bspline::BSplineSpace)

Returns the reference Bernstein polynomials of `bspline`.

# Arguments
- `bspline::BSplineSpace`: A univariate B-Spline function space.
# Returns
- `::Polynomials.Bernstein`: Bernstein polynomials.
"""
function get_polynomials(bspline::BSplineSpace)
    return bspline.polynomials
end

"""
    evaluate(bspline::BSplineSpace, element_id::Int, xi::Vector{Float64})

Evaluates the global basis functions of `bspline` on the element specified by `element_id` and points `xi`.

# Arguments
- `bspline::BSplineSpace`: A univariate B-Spline function space.
- `element_id::Int`: The id of the element.
- `xi::Vector{Float64}`: The points where the global basis is evaluated.
# Returns
- `::Matrix{Float64}`: Gloabal basis functions.
"""
function evaluate(bspline::BSplineSpace, element_id::Int, xi::Vector{Float64})
    extraction_coefficients, supported_bases = get_extraction_coefficients_and_indices(bspline, element_id)

    return extraction_coefficients  * get_polynomials(bspline)(xi), supported_bases
end
