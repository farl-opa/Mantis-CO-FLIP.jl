"""
    module Forms

Contains all definitions of forms, including form fields, form spaces, and form expressions.
"""
module Forms

import .. FunctionSpaces
import .. Geometry

include("./FormSpaces.jl")
include("./FormExpressions.jl")

# 0-form: no basis
# 1-forms: (dx1, dx2, dx3)
# 2-forms: (dx2dx3, dx3dx1, dx1dx2)
# 3-forms: dx1dx2d3

end