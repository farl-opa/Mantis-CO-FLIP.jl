struct BSplineSpace{n, k}<:FunctionSpace{n, k}
    patch::Mesh.Patch{n}
    polynomial_degree::NTuple{n, Vector{Int}}
    regularity::NTuple{n, Vector{Int}}
    function BSplineSpace(patch::Mesh.Patch{n}, polynomial_degree::NTuple{n, Vector{Int}}, regularity::NTuple{n, Vector{Int}}, k::Int) where {n}
        for d in 1:1:n
            if size(patch)[d] != length(polynomial_degree[d])
                msg1 = "Number of elements and number of given degrees are not the same."
                msg2 = " You have $(size(patch)[d]) elements and $(length(polynomial_degree[d])) degrees."
                throw(ArgumentError(msg1*msg2))
            end
            for i in 1:1:length(polynomial_degree[d])
                if polynomial_degree[d][i] < 0 
                    msg1 = "Polynomial degree must be greater or equal than 0."
                    msg2 = " In dimension $d, there is degree $polynomial_degree[d][i]."
                    throw(ArgumentError(msg1*msg2))
                end
                if i==1 && polynomial_degree[d][i] <= regularity[d][i]
                    msg1 = "Regularity has to be strictly less than the polynomial degree."
                    msg2 = " In dimension $d, there is degree $polynomial_degree[d][i] and regularity $regularity[d][i]"
                    throw(ArgumentError(msg1*msg2))
                elseif polynomial_degree[d][i] <= regularity[d][i] 
                    msg1 = "Regularity has to be strictly less than the polynomial degree."
                    msg2 = " In dimension $d, there is degree $polynomial_degree[d][i] and regularity $regularity[d][i]"
                    throw(ArgumentError(msg1*msg2))
                elseif polynomial_degree[d][i] <= regularity[d][i-1]
                    msg1 = "Regularity has to be strictly less than the polynomial degree."
                    msg2 = " In dimension $d, there is degree $polynomial_degree[d][i] and regularity $regularity[d][i-1]"
                    throw(ArgumentError(msg1*msg2))
                elseif i==length(polynomial_degree[d]) && polynomial_degree[d][i] <= regularity[d][i-1]
                    msg1 = "Regularity has to be strictly less than the polynomial degree."
                    msg2 = " In dimension $d, there is degree $polynomial_degree[d][i] and regularity $regularity[d][i-1]"
                    throw(ArgumentError(msg1*msg2))
                end
            end
        end

        new{n, k}(patch, polynomial_degree, regularity)        
    end
end