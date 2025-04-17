# Quadrature
```@meta
CurrentModule = Mantis.Quadrature
```
Mantis' `Quadrature` module contains all functionality related to quadrature rules.

!!! warning "The standard domain for the quadrature rules in Mantis is [0, 1]."
    This is different from the common choice of [-1, 1]. However, the [0, 1] interval is
    more in line with the definition of the function spaces. See [FunctionSpaces](@ref).


## Background on Numerical Quadrature.
Numerical quadrature is the process of approximating the integral of a function by
evaluating the function at specific points and combining the evaluations with weights. The
points are called quadrature nodes and the weights are the corresponding weights for each
node. How the nodes and weights are chosen determines the accuracy of the quadrature rule
and varies per rule.

In general, a quadrature rule can be written as:
```math
\int_{0}^{1} f(x) dx \approx \sum_{i=1}^{N} w_i f(x_i)
```
where ``N`` is the number of quadrature nodes, ``w_i`` are the weights, and ``x_i`` are the
quadrature nodes. Note that the integral is computed on the interval [0, 1].

## Overview of all types and functions from Mantis.Quadrature
The Quadrature modules defines three abstract types:
```@docs
AbstractQuadratureRule
AbstractElementQuadratureRule
AbstractGlobalQuadratureRule
```

Currently, there are two concrete subtypes of `AbstractElementQuadratureRule`:
```@docs
ElementQuadratureRule
CanonicalQuadratureRule
```
and one subtype of `AbstractGlobalQuadratureRule`
```@docs
StandardQuadrature
```
For every element quadrature rule, the following getters are defined:
```@docs
get_label
get_nodes
get_weights
```

### Available Quadrature Rules
Gauss-type rules:
```@docs
gauss_legendre
gauss_lobatto
```

Clenshaw-Curtis rules:
```@docs
clenshaw_curtis
```

Rules on equally spaced nodes:
```@docs
newton_cotes
```

The above rules are all 1D rules. Using a tensor product rule allows thse rules to be used
in a multi-dimensional setting. For this, the following methods can be used:
```@docs
tensor_product_rule
```
These methods use the following internal function to compute the tensor product weights:
```@docs
_compute_tensor_product
```


### Helper Functions
Create several tensor-product rules of the same type:
```@docs
get_canonical_quadrature_rules
```
