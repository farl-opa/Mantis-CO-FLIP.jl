@doc raw"""
    get_dorfler_marking(element_errors::Vector{Float64}, dorfler_parameter::Float64) 

Computes the indices of elements with at least 'dorfler_parameter*100'% of the highest error in 'element_errors'.

# Arguments
- `element_errors::Vector{Float64}`: element-wise errors.
- `dorfler_parameter::Float64`: dorfler parameter determing how many elements are selected.
"""
function get_dorfler_marking(element_errors::Vector{Float64}, dorfler_parameter::Float64) 
    0.0 <= dorfler_parameter < 1.0 || throw(ArgumentError("Dorfler parameter should be between 0 and 1. The given value was $dorfler_parameter.")) 
    max_error = maximum(element_errors)
    
    return findall(el -> el > (1.0-dorfler_parameter)*max_error, element_errors)
end