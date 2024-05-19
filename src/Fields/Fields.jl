"""
This (sub-)module provides a collection of fields.
The exported names are:
"""
module Fields

import .. FunctionSpaces

"""
    AbstractField

Supertype for all fields.
"""
abstract type AbstractField{n, m} end
abstract type AbstractAnalyticalField{n, m} <: AbstractField{n, m} end
abstract type AbstractFEMField{n, m} <: AbstractField{n, m} end

include("./FEMField.jl")

end