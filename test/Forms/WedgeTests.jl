import Mantis

using Test

using LinearAlgebra
using SparseArrays

# Domain
Lleft = 0.0
Lright = 1.0
Lbottom = 0.0
Ltop = 1.0

# Setup the form spaces -------------------------------------------------------
# First the patches
breakpoints1 = [Lleft, 0.5, Lright]
patch1 = Mantis.Mesh.Patch1D(breakpoints1)
breakpoints2 = [Lbottom, 0.5, 0.6, Ltop]
patch2 = Mantis.Mesh.Patch1D(breakpoints2)

# then the univariate FEM spaces
deg1 = 2
deg2 = 2
B1 = Mantis.FunctionSpaces.BSplineSpace(patch1, deg1, [-1, deg1-1, -1])
B2 = Mantis.FunctionSpaces.BSplineSpace(patch2, deg2, [-1, min(deg2-1,1),  deg2-1, -1])
dB1 = Mantis.FunctionSpaces.get_derivative_space(B1)
dB2 = Mantis.FunctionSpaces.get_derivative_space(B2)

# next, tensor-product B-spline spaces
TP_Space_0 = Mantis.FunctionSpaces.DirectSumSpace((Mantis.FunctionSpaces.TensorProductSpace(B1, B2),))
dTP_Space_dx = Mantis.FunctionSpaces.TensorProductSpace(dB1, B2)
dTP_Space_dy = Mantis.FunctionSpaces.TensorProductSpace(B1, dB2)
TP_Space_1 = Mantis.FunctionSpaces.DirectSumSpace((dTP_Space_dx, dTP_Space_dy))
TP_Space_2 = Mantis.FunctionSpaces.DirectSumSpace((Mantis.FunctionSpaces.TensorProductSpace(dB1, dB2),))

# then, the geometries
# Line 1
line_1_geo = Mantis.Geometry.CartesianGeometry((breakpoints1,))
# Line 2
line_2_geo = Mantis.Geometry.CartesianGeometry((breakpoints2,))
# Tensor product geometry 
geom_cart = Mantis.Geometry.TensorProductGeometry(line_1_geo, line_2_geo)

# curvilinear mesh
crazy_c = 0.2
function mapping(x::Vector{Float64})
    x1_new = (2.0/(Lright-Lleft))*x[1] - 2.0*Lleft/(Lright-Lleft) - 1.0
    x2_new = (2.0/(Ltop-Lbottom))*x[2] - 2.0*Lbottom/(Ltop-Lbottom) - 1.0
    return [x[1] + ((Lright-Lleft)/2.0)*crazy_c*sinpi(x1_new)*sinpi(x2_new), x[2] + ((Ltop-Lbottom)/2.0)*crazy_c*sinpi(x1_new)*sinpi(x2_new)]
end
function dmapping(x::Vector{Float64})
    x1_new = (2.0/(Lright-Lleft))*x[1] - 2.0*Lleft/(Lright-Lleft) - 1.0
    x2_new = (2.0/(Ltop-Lbottom))*x[2] - 2.0*Lbottom/(Ltop-Lbottom) - 1.0
    return [1.0 + pi*crazy_c*cospi(x1_new)*sinpi(x2_new) ((Lright-Lleft)/(Ltop-Lbottom))*pi*crazy_c*sinpi(x1_new)*cospi(x2_new); ((Ltop-Lbottom)/(Lright-Lleft))*pi*crazy_c*cospi(x1_new)*sinpi(x2_new) 1.0 + pi*crazy_c*sinpi(x1_new)*cospi(x2_new)]
end
dimension = (2, 2)
crazy_mapping = Mantis.Geometry.Mapping(dimension, mapping, dmapping)
geom_crazy = Mantis.Geometry.MappedGeometry(geom_cart, crazy_mapping)
# -----------------------------------------------------------------------------

# Test wedge product ----------------------------------------------------------

# Test on multiple geometries
q_rule = Mantis.Quadrature.tensor_product_rule((deg1+25, deg2+25), Mantis.Quadrature.gauss_legendre)
for geom in [geom_cart, geom_crazy]#[geom_cart, geom_crazy]
    println("Geom type: ",typeof(geom))
    
    # Create the form spaces
    ϵ⁰ = Mantis.Forms.FormSpace(0, geom, TP_Space_0, "ν")
    ϵ¹ = Mantis.Forms.FormSpace(1, geom, TP_Space_1, "η")
    ϵⁿ = Mantis.Forms.FormSpace(2, geom, TP_Space_2, "σ")

    # Create the Hodge-star of the form spaces
    ★ϵ⁰ = Mantis.Forms.hodge(ϵ⁰)
    ★ϵ¹ = Mantis.Forms.hodge(ϵ¹)
    ★ϵⁿ = Mantis.Forms.hodge(ϵⁿ)

    
    for elem_id in 1:1:Mantis.Geometry.get_num_elements(geom)
        # Compare the inner product
        #   <ϵ⁰, ϵ⁰>
        # with 
        #   ϵ⁰ ∧ ⋆ϵ⁰
        
        # Compute the inner product: <ϵ⁰, ϵ⁰>
        zero_form_rows_idx, zero_form_columns_idx, zero_form_inner_product_evaluate = Mantis.Forms.evaluate_inner_product(ϵ⁰, ϵ⁰, elem_id, q_rule)
        
        # Compute the wedge product: ϵ⁰ ∧ ⋆ϵ⁰
        zero_form_wedge = Mantis.Forms.wedge(ϵ⁰, ★ϵ⁰)
        
        # Evaluate it at the quadrature nodes
        ξ_quadrature_nodes = Mantis.Quadrature.get_quadrature_nodes(q_rule)
        zero_form_wedge_evaluate, zero_form_wedge_indices = Mantis.Forms.evaluate(zero_form_wedge, elem_id, ξ_quadrature_nodes)
        
        # Integrate
        quadrature_weights = Mantis.Quadrature.get_quadrature_weights(q_rule)
        zero_form_inner_product_from_wedge = sum(quadrature_weights .* zero_form_wedge_evaluate[1], dims=1)[:]
        a = 2

        # Test if it is the same as the inner product
        @test isapprox(sum(abs.(zero_form_inner_product_from_wedge - zero_form_inner_product_evaluate)), 0.0, atol=1e-12)

        # Test if the wedge product expression has the correct expression rank 
        @test Mantis.Forms.get_expression_rank(zero_form_wedge) == 2

        # Compare the inner product
        #   <ϵⁿ, ϵⁿ>
        # with 
        #   ϵⁿ ∧ ⋆ϵⁿ
        
        # Compute the inner product: <ϵⁿ, ϵⁿ>
        top_form_rows_idx, top_form_columns_idx, top_form_inner_product_evaluate = Mantis.Forms.evaluate_inner_product(ϵⁿ, ϵⁿ, elem_id, q_rule)
        
        # Compute the wedge product: ϵⁿ ∧ ⋆ϵⁿ
        top_form_wedge = Mantis.Forms.wedge(ϵⁿ, ★ϵⁿ)

        # Evaluate it at the quadrature nodes
        ξ_quadrature_nodes = Mantis.Quadrature.get_quadrature_nodes(q_rule)
        top_form_wedge_evaluate, top_form_wedge_indices = Mantis.Forms.evaluate(top_form_wedge, elem_id, ξ_quadrature_nodes)
        
        # Integrate
        quadrature_weights = Mantis.Quadrature.get_quadrature_weights(q_rule)
        top_form_inner_product_from_wedge = sum(quadrature_weights .* top_form_wedge_evaluate[1], dims=1)[:]

        # Test if it is the same as the inner product
        @test isapprox(sum(abs.(top_form_inner_product_from_wedge - top_form_inner_product_evaluate)), 0.0, atol=1e-12)

        # Test if the wedge product expression has the correct expression rank 
        @test Mantis.Forms.get_expression_rank(top_form_wedge) == 2

        # Compare the inner product
        #   <ϵ¹, ϵ¹>
        # with 
        #   ϵ¹ ∧ ⋆ϵ¹
        
        # Compute the inner product: <ϵ¹, ϵ¹>
        one_form_rows_idx, one_form_columns_idx, one_form_inner_product_evaluate = Mantis.Forms.evaluate_inner_product(ϵ¹, ϵ¹, elem_id, q_rule)
        
        # Compute the wedge product: ϵ¹ ∧ ⋆ϵ¹
        one_form_wedge = Mantis.Forms.wedge(ϵ¹, ★ϵ¹)
        
        # Evaluate it at the quadrature nodes
        ξ_quadrature_nodes = Mantis.Quadrature.get_quadrature_nodes(q_rule)
        one_form_wedge_evaluate, one_form_wedge_indices = Mantis.Forms.evaluate(one_form_wedge, elem_id, ξ_quadrature_nodes)
        
        # Integrate
        quadrature_weights = Mantis.Quadrature.get_quadrature_weights(q_rule)
        one_form_inner_product_from_wedge = sum(quadrature_weights .* one_form_wedge_evaluate[1], dims=1)[:]

        # Test if it is the same as the inner product
        @test isapprox(sum(abs.(one_form_inner_product_from_wedge - one_form_inner_product_evaluate)), 0.0, atol=1e-12)

        # Test if the wedge product expression has the correct expression rank 
        @test Mantis.Forms.get_expression_rank(one_form_wedge) == 2

        # Test αⁿ ∧ βⁿ (top forms) throws error
        @test_throws ArgumentError Mantis.Forms.wedge(ϵⁿ, ϵⁿ)

        # Test ϵ¹ ∧ ⋆ϵ¹ ∧ ϵ¹ to check if rank 3 expressions are not allowed
        @test_throws ArgumentError Mantis.Forms.wedge(one_form_wedge, ϵ⁰)
    end
end

# -----------------------------------------------------------------------------