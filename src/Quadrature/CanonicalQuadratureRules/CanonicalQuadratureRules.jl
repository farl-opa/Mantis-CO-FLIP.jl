# One-dimensional quadrature rules.
# Quadrature rules on non-equally spaced nodes.
include("./Gauss.jl")
include("./ClenshawCurtis.jl")
# Quadrature rules on equally spaced nodes.
include("./NewtonCotes.jl")

# Multi-dimensional quadrature rules. The tensor product rules are made by combining one-
# dimensional rules, so this file must be included after the one-dimensional rules.
include("./TensorProduct.jl")
