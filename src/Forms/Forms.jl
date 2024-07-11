"""
    module Forms

Contains all definitions of forms, including form fields, form spaces, and form expressions.
"""
module Forms

import .. FunctionSpaces
import .. Geometry

abstract type AbstractFormExpression{manifold_dim, form_rank} end
abstract type AbstractFormField{manifold_dim, form_rank} <: AbstractFormExpression{manifold_dim, form_rank} end

include("./FormSpaces.jl")
include("./FormExpressions.jl")

end