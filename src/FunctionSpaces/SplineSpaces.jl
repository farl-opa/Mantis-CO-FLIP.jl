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
    multiplicity::Vector{Int}

    function KnotVector(breakpoints::Vector{Float64}, multiplicity::Vector{Int})
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

        new(breakpoints, multiplicity)
    end
end

"""
    struct BSplineSpace{n,k}

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
struct BSplineSpace{n, k}<:AbstractFunctionSpace{n, k}
    patch::Mesh.Patch{n}
    polynomial_degree::NTuple{n, Int}
    regularity::NTuple{n, Vector{Int}}
    function BSplineSpace(patch::Mesh.Patch{n}, polynomial_degree::NTuple{n, Int}, regularity::NTuple{n, Vector{Int}}, k::Int) where {n}
        for d in 1:1:n
            if polynomial_degree[d] <0
                msg1 = "Polynomial degree must be greater or equal than 0."
                msg2 = " In dimension $d, the degree is $(polynomial_degree[d])."
                throw(ArgumentError(msg1*msg2))
            end
            if size(patch)[d] - 1 != length(regularity[d])
                msg1 = "Number of regularity conditions should be one less than the number of elements."
                msg2 = " You have $(size(patch)[d]) elements and $(length(regularity[d])) regularity conditions in dimension $d"
                throw(ArgumentError(msg1*msg2))
            end
            for i in 1:1:length(regularity[d])
                if polynomial_degree[d] <= regularity[d][i]
                    msg1 = "Polynomial degree must be greater than regularity."
                    msg2 = " In dimension $d,  the degree is $polynomial_degree[d] and there is regularity $regularity[d][i]"
                    throw(ArgumentError(msg1*msg2))
                end
            end
        end

        new{n, k}(patch, polynomial_degree, regularity)        
    end
end