# Priority 1: Inner products for abstract form fields

# 0-forms
function inner_product(f1::FormSpace{n,0,G,Tuple{F1}}, f2::FormSpace{n,0,G,Tuple{F2}}, element_id::Int, quad_rule::QuadratureRule{n}) where {n, G<:Geometry.AbstractGeometry{n,m} where {m}, F1 <: FunctionSpaces.AbstractFunctionSpace{n}, F2 <: FunctionSpaces.AbstractFunctionSpace{n}}

    # 1. evaluate geometry and metric on element
    # 2. fill in dense matrix of basis function integrals against each other

end

# n-forms
function inner_product(f1::FormSpace{n,n,G,Tuple{F1}}, f2::FormSpace{n,n,G,Tuple{F2}}, element_id::Int, quad_rule::QuadratureRule{n}) where {n, G<:Geometry.AbstractGeometry{n,m} where {m}, F1 <: FunctionSpaces.AbstractFunctionSpace{n}, F2 <: FunctionSpaces.AbstractFunctionSpace{n}}

    # Computed bases and their derivatives.
    basis_eval_1, basis_inds_1 = FunctionSpaces.evaluate(f1, element_id, quad_rule.points, 0)
    basis_eval_2, basis_inds_2 = FunctionSpaces.evaluate(f2, element_id, quad_rule.points, 0)
    
    

end

# 1-forms in 2D
function inner_product(f1::FormSpace{2,1,G,Tuple{F1x,F1y}}, f2::FormSpace{2,1,G,Tuple{F2x,F2y}}, element_id::Int, quad_rule::QuadratureRule{2}) where {G<:Geometry.AbstractGeometry{2,m} where {m}, F1x <: FunctionSpaces.AbstractFunctionSpace{2}, F1y <: FunctionSpaces.AbstractFunctionSpace{2}, F2x <: FunctionSpaces.AbstractFunctionSpace{2}, F2y <: FunctionSpaces.AbstractFunctionSpace{2}}



end

# 1-forms in 3D
function inner_product(f1::FormSpace{3,1,G,Tuple{F1x,F1y,F1z}}, f2::FormSpace{3,1,G,Tuple{F2x,F2y,F2z}}, element_id::Int, quad_rule::QuadratureRule{3}) where {G<:Geometry.AbstractGeometry{3,m} where {m}, F1x <: FunctionSpaces.AbstractFunctionSpace{3}, F1y <: FunctionSpaces.AbstractFunctionSpace{3}, F1z <: FunctionSpaces.AbstractFunctionSpace{3}, F2x <: FunctionSpaces.AbstractFunctionSpace{3}, F2y <: FunctionSpaces.AbstractFunctionSpace{3}, F2z <: FunctionSpaces.AbstractFunctionSpace{3}}



end

# 2-forms in 3D
function inner_product(f1::FormSpace{3,2,G,Tuple{F1x,F1y,F1z}}, f2::FormSpace{3,2,G,Tuple{F2x,F2y,F2z}}, element_id::Int, quad_rule::QuadratureRule{3}) where {G<:Geometry.AbstractGeometry{3,m} where {m}, F1x <: FunctionSpaces.AbstractFunctionSpace{3}, F1y <: FunctionSpaces.AbstractFunctionSpace{3}, F1z <: FunctionSpaces.AbstractFunctionSpace{3}, F2x <: FunctionSpaces.AbstractFunctionSpace{3}, F2y <: FunctionSpaces.AbstractFunctionSpace{3}, F2z <: FunctionSpaces.AbstractFunctionSpace{3}}



end

# Priority 2: Wedge products

# Priority 3: Triple products