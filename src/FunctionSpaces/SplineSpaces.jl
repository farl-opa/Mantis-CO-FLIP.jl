struct BSplineSpace{n, k}<:FunctionSpace{n, k} 
    patch::Mesh.Patch{n}
    polynomial_degree::NTuple{n, Vector{Int}}
    regularity::NTuple{n, Vector{Int}}
    function BSplineSpace(patch::Mesh.Patch{n}, polynomial_degree::NTuple{n, Vector{Int}}, regularity::NTuple{n, Vector{Int}}, k::Int) where {n}
        for d in 1:1:n, i in eachindex(polynomial_degree[d])
            if size(patch)[d] != length(polynomial_degree[d])
                msg1 = "Number of elements and given degrees are not the same."
                msg2 = " You have $(size(patch)[d]) elements and $(length(polynomial_degree[d])) degrees."
                throw(ArgumentError(msg1*msg2))
            end
            if polynomial_degree[d][i] < 0 
                msg1 = "Polynomial degree must be greater or equal than 0."
                msg2 = " In dimension $d, there is degree $polynomial_degree[d][i]."
                throw(ArgumentError(msg1*msg2))
            end
        end

        new{n, k}(patch, polynomial_degree, regularity)        
    end
end