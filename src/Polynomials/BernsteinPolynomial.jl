struct Bernstein <: AbstractPolynomials
    p::Int  # Polynomial degree
end

function evaluate(polynomial::Bernstein, xi::Vector{Float64})::Array{Float64}
    return evaluate(polynomial, xi, 0)
end

"""
    evaluate(polynomial::Bernstein, xi::Vector{Float64}, nderivatives::Int64)::Array{Float64}

Compute derivatives upto order `nderivatives`` for all Bernstein polynomial of degree `p` at `\\xi` for ``\\xi \\in [0.0, 1.0]``. 

# Arguments
- `polynomial::Bernstein`: Bernstein polynomial
- `xi::Vector{Float64}`: vector of evaluation points \\in [0.0, 1.0].
- `nderivatives::Int64`: maximum order of derivatives to be computed (nderivatives \\leq p).
"""
function evaluate(polynomial::Bernstein, xi::Vector{Float64}, nderivatives::Int64)::Array{Float64}
    # store the values and derivatives here
    neval = length(xi)
    ders = zeros(Float64, neval, polynomial.p + 1, nderivatives + 1)
    for i = 1:neval
        ders[i,:,:] = evaluate(polynomial, xi[i], nderivatives)
    end
    return ders
end

function evaluate(polynomial::Bernstein, xi::Float64)::Array{Float64}
    return evaluate(polynomial, xi, 0)
end

function evaluate(polynomial::Bernstein, xi::Float64, nderivatives::Int64)::Array{Float64}
    # degree
    p = polynomial.p

    if p < 0
        msg = "The Bernstein polynomial must be of degree at least 0."
        throw(ArgumentError(msg))
    elseif xi < 0.0 || xi > 1.0
        msg = "x = $x is outside the interval [0.0, 1.0]."
        throw(ArgumentError(msg))
    end
    
    ndu = zeros(Float64, p+1, p+1)
    left = zeros(Float64, p+1)
    right = zeros(Float64, p+1)
    ndu[1,1] = 1
    for j=1:p
        left[j+1] = xi
        right[j+1] = 1.0 - xi
        saved = 0.0
        for r = 0:j-1
            #lower trinagle
            ndu[j+1,r+1] = right[r+2]+ left[j-r+1]
            temp = ndu[r+1,j] / ndu[j+1,r+1]
            
            # upper triangle
            ndu[r+1,j+1] = saved + right[r+2] * temp
            saved = left[j-r+1]*temp
        end
        ndu[j+1,j+1] = saved
    end
    
    # store the values and derivatives here
    ders = zeros(Float64, 1, p+1, nderivatives+1)
    # values are contained in the last column of ndu
    ders[1,:,1] = ndu[:,p+1]
    
    # if nderivatves>0, the next section computes the derivatives
    if nderivatives>0
        a = zeros(Int64, 2, nderivatives+1)
        for r=0:p
            s1=0; s2=1
            a[1,1] = 1
            # loop to compute kth derivative
            for k=1:nderivatives
                d=0
                rk=r-k; pk = p-k
                if (r >= k)
                    a[s2+1,1] = a[s1+1,1] / ndu[pk+2,rk+1]
                    d = a[s2+1,1] * ndu[rk+1,pk+1]
                end
                if (rk >= -1) 
                    j1 = 1
                else
                    j1 = -rk
                end
                if (r-1 <= pk)
                    j2 = k-1
                else
                    j2 = p-r
                end
                
                for j=j1:j2
                    a[s2+1,j+1] = (a[s1+1,j+1] - a[s1+1,j]) / ndu[pk+2,rk+j+1]
                    d = d + a[s2+1,j+1] * ndu[rk+j+1,pk+1]
                end
                
                if (r <= pk)
                    a[s2+1,k+1] = - a[s1+1,k] / ndu[pk+2,r+1]
                    d = d + a[s2+1,k+1] * ndu[r+1,pk+1]
                end
                ders[1, r+1, k+1] = d
                j = s1; s1 = s2; s2 = j
            end
        end

        # multiply by the correct factors
        r = p
        for k=1:nderivatives
            ders[1,:,k+1] *= r
            r = r*(p-k)
        end
    end
    
    return ders
end

