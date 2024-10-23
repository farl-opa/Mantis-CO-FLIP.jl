import Mantis

using Test

using LinearAlgebra
using SparseArrays

# 2D Tests --------------------------------------------------------------------

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

        # Test ϵ¹ ∧ ⋆ϵ¹ ∧ ϵ¹ to check if form rank 3 expressions are not allowed
        @test_throws ArgumentError Mantis.Forms.wedge(one_form_wedge, ϵ¹)


    end
end

# -----------------------------------------------------------------------------


# 3D tests --------------------------------------------------------------------

# Setup the geometry 

# Domain
Lleft = 0.0
Lright = 1.0
Lbottom = 0.0
Ltop = 1.0

# Setup the form spaces

# First the FEM spaces
breakpoints = [Lleft, 0.5, Lright]
patch = Mantis.Mesh.Patch1D(breakpoints)

# first B-spline patch
deg = 2
B = Mantis.FunctionSpaces.BSplineSpace(patch, deg, [-1, deg-1, -1])

# tensor-product B-spline patch
TP_Space_2d = Mantis.FunctionSpaces.TensorProductSpace(B, B)
TP_Space_3d = Mantis.FunctionSpaces.TensorProductSpace(TP_Space_2d, B)

# Then the geometry 
# Line
line_geo = Mantis.Geometry.CartesianGeometry((breakpoints,))
geo_3d_cart = Mantis.Geometry.CartesianGeometry((breakpoints, breakpoints, breakpoints))

# Crazy Tensor product geometry in 2D (auxiliary) 
tensor_prod_geo_2d = Mantis.Geometry.TensorProductGeometry(line_geo, line_geo)
geo_2d_cart_aux = Mantis.Geometry.CartesianGeometry((breakpoints, breakpoints))
crazy_geo_2d_cart = Mantis.Geometry.MappedGeometry(geo_2d_cart_aux, crazy_mapping)

# Crazy mesh 3D (in x and y only, z is straight) (this is also a tensor product geometry)
crazy_geo_3d_cart = Mantis.Geometry.TensorProductGeometry(crazy_geo_2d_cart, line_geo)



# Setup the form spaces

# Generate the multivalued FEMSpaces (DirectSumSpace)
dsTP_0_form_3d = Mantis.FunctionSpaces.DirectSumSpace((TP_Space_3d,))  # direct sum space
dsTP_1_form_3d = Mantis.FunctionSpaces.DirectSumSpace((TP_Space_3d, TP_Space_3d, TP_Space_3d))  # direct sum space
dsTP_2_form_3d = dsTP_1_form_3d
dsTP_top_form_3d = dsTP_0_form_3d

# Expression for analytical form fields 
function zero_form_expression(x::Matrix{Float64})
    return [ones(Float64, size(x))]
end

# function one_form_expression(x::Matrix{Float64})
#     return [ones(Float64, size(x)) for _ in 1:3]
# end

# function two_form_expression(x::Matrix{Float64})
#     return [ones(Float64, size(x)) for _ in 1:3]
# end

# function top_form_expression(x::Matrix{Float64})
#     return [ones(Float64, size(x))]
# end

# Quadrature rule
q_rule = Mantis.Quadrature.tensor_product_rule((deg, deg, deg) .+ 1, Mantis.Quadrature.gauss_legendre)
ξ_quadrature_nodes = Mantis.Quadrature.get_quadrature_nodes(q_rule)
quadrature_weights = Mantis.Quadrature.get_quadrature_weights(q_rule)

# Test on multiple geometries. Type-wise and content/metric wise.
for geom in [geo_3d_cart, crazy_geo_3d_cart]
    println("Geom type: ",typeof(geom))

    # Create form spaces
    ϵ⁰ = Mantis.Forms.FormSpace(0, geom, dsTP_0_form_3d, "ν")
    ϵ¹ = Mantis.Forms.FormSpace(1, geom, dsTP_1_form_3d, "η")
    ϵ² = Mantis.Forms.FormSpace(2, geom, dsTP_2_form_3d, "μ")
    ϵ³ = Mantis.Forms.FormSpace(3, geom, dsTP_top_form_3d, "σ")

    # Analytical form fields
    α⁰ = Mantis.Forms.AnalyticalFormField(0, zero_form_expression, geom, "α⁰")
    # α¹ = Mantis.Forms.AnalyticalFormField(1, one_form_expression, geom, "α¹")
    # α² = Mantis.Forms.AnalyticalFormField(2, two_form_expression, geom, "α²")
    # α³ = Mantis.Forms.AnalyticalFormField(3, top_form_expression, geom, "α³")

    # Hodge-⋆ of all form spaces
    ★ϵ⁰ = Mantis.Forms.hodge(ϵ⁰)
    ★ϵ¹ = Mantis.Forms.hodge(ϵ¹)
    ★ϵ² = Mantis.Forms.hodge(ϵ²)
    ★ϵ³ = Mantis.Forms.hodge(ϵ³)
    
    for elem_id in 1:Mantis.Geometry.get_num_elements(geom)
        # These tests the wedge product by checking if 
        #   Since ∫ α⁰ ∧ ⋆βᵏ = (αᵏ, βᵏ)
        # Therefore it computes the left hand side with the wedge and the right hand side 
        # with the inner product. 

        # ϵ⁰ ∧ ⋆ϵ⁰ (0-form with top-form, both expression rank 1) -------------

        # Compute the wedge product
        zero_form_wedge = Mantis.Forms.wedge(ϵ⁰, ★ϵ⁰)
        zero_form_wedge_triple = Mantis.Forms.wedge(zero_form_wedge, α⁰)  # ϵ⁰ ∧ ⋆ϵ⁰ ∧ α⁰ (with α⁰ = 1, so the two expressions should be equal)

        # Evaluate wedge at the quadrature nodes
        zero_form_wedge_evaluate, zero_form_wedge_indices = Mantis.Forms.evaluate(zero_form_wedge, elem_id, ξ_quadrature_nodes)
        zero_form_wedge_triple_evaluate, zero_form_wedge_triple_indices = Mantis.Forms.evaluate(zero_form_wedge_triple, elem_id, ξ_quadrature_nodes)

        # Test if ϵ⁰ ∧ ⋆ϵ⁰ == ϵ⁰ ∧ ⋆ϵ⁰ ∧ α⁰ (since α⁰ = 1) 
        @test isapprox(sum(abs.(zero_form_wedge_evaluate[1] - zero_form_wedge_triple_evaluate[1])), 0.0, atol=1e-12)
        
        # Integrate
        zero_form_inner_product_from_wedge = sum(quadrature_weights .* zero_form_wedge_evaluate[1], dims=1)[:]

        # Compute inner product (reference value)
        zero_form_inner_rows_idx, zero_form_inner_columns_idx, zero_form_inner_evaluate = Mantis.Forms.evaluate_inner_product(ϵ⁰, ϵ⁰, elem_id, q_rule)

        # Test if it is the same as the inner product
        @test isapprox(sum(abs.(zero_form_inner_product_from_wedge - zero_form_inner_evaluate)), 0.0, atol=1e-12)

        # Test if the wedge product expression has the correct expression rank 
        @test Mantis.Forms.get_expression_rank(zero_form_wedge) == 2
        # ---------------------------------------------------------------------
        
        # ϵ¹ ∧ ⋆ϵ¹ (1-form with 2-form, both expression rank 1) ---------------

        # Compute the wedge product
        one_form_wedge = Mantis.Forms.wedge(ϵ¹, ★ϵ¹)
        one_form_wedge_triple = Mantis.Forms.wedge(one_form_wedge, α⁰)  # ϵ¹ ∧ ⋆ϵ¹ ∧ α⁰ (with α⁰ = 1, so the two expressions should be equal)

        # Evaluate wedge at the quadrature nodes
        one_form_wedge_evaluate, one_form_wedge_indices = Mantis.Forms.evaluate(one_form_wedge, elem_id, ξ_quadrature_nodes)
        one_form_wedge_triple_evaluate, one_form_wedge_triple_indices = Mantis.Forms.evaluate(one_form_wedge_triple, elem_id, ξ_quadrature_nodes)

        # Test if ϵ¹ ∧ ⋆ϵ¹ == ϵ¹ ∧ ⋆ϵ¹ ∧ α⁰ (since α⁰ = 1) 
        @test isapprox(sum(abs.(one_form_wedge_evaluate[1] - one_form_wedge_triple_evaluate[1])), 0.0, atol=1e-12)

        # Integrate
        one_form_inner_product_from_wedge = sum(quadrature_weights .* one_form_wedge_evaluate[1], dims=1)[:]

        # Compute inner product (reference value)
        one_form_inner_rows_idx, one_form_inner_columns_idx, one_form_inner_evaluate = Mantis.Forms.evaluate_inner_product(ϵ¹, ϵ¹, elem_id, q_rule)

        # Test if it is the same as the inner product
        @test isapprox(sum(abs.(one_form_inner_product_from_wedge - one_form_inner_evaluate)), 0.0, atol=1e-12)

        # Test if the wedge product expression has the correct expression rank 
        @test Mantis.Forms.get_expression_rank(one_form_wedge) == 2
        # ---------------------------------------------------------------------

        # ϵ² ∧ ⋆ϵ² (2-form with 1-form, both expression rank 1) ---------------

        # Compute the wedge product
        two_form_wedge = Mantis.Forms.wedge(ϵ², ★ϵ²)
        two_form_wedge_triple = Mantis.Forms.wedge(two_form_wedge, α⁰)  # ϵ² ∧ ⋆ϵ² ∧ α⁰ (with α⁰ = 1, so the two expressions should be equal)
        two_form_wedge_triple_alt = Mantis.Forms.wedge(Mantis.Forms.wedge(ϵ², α⁰), ★ϵ²)  # ϵ² ∧ α⁰ ∧ ⋆ϵ²  (with α⁰ = 1, so the three expressions should be equal)

        # Evaluate wedge at the quadrature nodes
        two_form_wedge_evaluate, one_form_wedge_indices = Mantis.Forms.evaluate(two_form_wedge, elem_id, ξ_quadrature_nodes)
        two_form_wedge_triple_evaluate, two_form_wedge_triple_indices = Mantis.Forms.evaluate(two_form_wedge_triple, elem_id, ξ_quadrature_nodes)
        two_form_wedge_triple_alt_evaluate, two_form_wedge_triple_alt_indices = Mantis.Forms.evaluate(two_form_wedge_triple_alt, elem_id, ξ_quadrature_nodes)

        # Test if ϵ² ∧ ⋆ϵ² == ϵ² ∧ ⋆ϵ² ∧ α⁰ == ϵ² ∧ α⁰ ∧ ⋆ϵ² (since α⁰ = 1) 
        @test isapprox(sum(abs.(two_form_wedge_evaluate[1] - two_form_wedge_triple_evaluate[1])), 0.0, atol=1e-12)
        @test isapprox(sum(abs.(two_form_wedge_triple_alt_evaluate[1] - two_form_wedge_triple_evaluate[1])), 0.0, atol=1e-12)

        # Integrate
        two_form_inner_product_from_wedge = sum(quadrature_weights .* two_form_wedge_evaluate[1], dims=1)[:]

        # Compute inner product (reference value)
        two_form_inner_rows_idx, two_form_inner_columns_idx, two_form_inner_evaluate = Mantis.Forms.evaluate_inner_product(ϵ², ϵ², elem_id, q_rule)

        # Test if it is the same as the inner product
        @test isapprox(sum(abs.(two_form_inner_product_from_wedge - two_form_inner_evaluate)), 0.0, atol=1e-12)

        # Test if the wedge product expression has the correct expression rank 
        @test Mantis.Forms.get_expression_rank(two_form_wedge) == 2
        # ---------------------------------------------------------------------

        # ϵ³ ∧ ⋆ϵ³ (2-form with 1-form, both expression rank 1) ---------------

        # Compute the wedge product
        top_form_wedge = Mantis.Forms.wedge(ϵ³, ★ϵ³)
        top_form_wedge_triple = Mantis.Forms.wedge(top_form_wedge, α⁰)  # ϵ³ ∧ ⋆ϵ³ ∧ α⁰ (with α⁰ = 1, so the two expressions should be equal)

        # Evaluate wedge at the quadrature nodes
        top_form_wedge_evaluate, top_form_wedge_indices = Mantis.Forms.evaluate(top_form_wedge, elem_id, ξ_quadrature_nodes)
        top_form_wedge_triple_evaluate, top_form_wedge_triple_indices = Mantis.Forms.evaluate(top_form_wedge_triple, elem_id, ξ_quadrature_nodes)

        # Test if ϵ³ ∧ ⋆ϵ³ == ϵ³ ∧ ⋆ϵ³ ∧ α⁰ (since α⁰ = 1) 
        @test isapprox(sum(abs.(top_form_wedge_evaluate[1] - top_form_wedge_triple_evaluate[1])), 0.0, atol=1e-12)

        # Integrate
        top_form_inner_product_from_wedge = sum(quadrature_weights .* top_form_wedge_evaluate[1], dims=1)[:]

        # Compute inner product (reference value)
        top_form_inner_rows_idx, top_form_inner_columns_idx, top_form_inner_evaluate = Mantis.Forms.evaluate_inner_product(ϵ³, ϵ³, elem_id, q_rule)

        # Test if it is the same as the inner product
        @test isapprox(sum(abs.(top_form_inner_product_from_wedge - top_form_inner_evaluate)), 0.0, atol=1e-12)

        # Test if the wedge product expression has the correct expression rank 
        @test Mantis.Forms.get_expression_rank(top_form_wedge) == 2

        # Test ϵ¹ ∧ ⋆ϵ¹ ∧ ϵ¹ to check if rank 3 expressions are not allowed
        @test_throws ArgumentError Mantis.Forms.wedge(one_form_wedge, ϵ⁰)
        # ---------------------------------------------------------------------
    end
end

# -----------------------------------------------------------------------------
