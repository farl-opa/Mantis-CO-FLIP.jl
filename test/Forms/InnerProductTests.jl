import Mantis

using Test

using LinearAlgebra
using SparseArrays

# Domain
Lleft = 0.0
Lright = 1.0
Lbottom = 0.0
Ltop = 1.0

# Setup the form spaces
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

# then, the geometry 
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

# Test the inner product of the 0-form spaces -----------------------------
q_rule = Mantis.Quadrature.tensor_product_rule((deg1+3, deg2+3), Mantis.Quadrature.gauss_legendre)
for geom in [geom_cart, geom_crazy]
    # build a 0-form space
    zero_form_space = Mantis.Forms.FormSpace(0, geom, TP_Space_0, "ξ")
    # compute inner product assuming unit coefficients for both spaces
    inner_prod_0 = 0.0
    for element_id in 1:1:Mantis.Geometry.get_num_elements(geom)
        prod_form_rows, prod_form_cols, prod_form_eval = Mantis.Forms.evaluate_inner_product(zero_form_space, zero_form_space, element_id, q_rule)
        inner_prod_0 += sum(prod_form_eval)
    end
    @test isapprox(inner_prod_0, 1.0, atol=1e-12)
end
# -------------------------------------------------------------------------

# Test the inner product of analytical form fields -----------------------------
function analytical_1valued_form_func(x::Matrix{Float64})
    return [@. 8.0 * pi^2 * sinpi(2.0 * x[:,1]) * sinpi(2.0 * x[:,2])]
end
function analytical_2valued_form_func(x::Matrix{Float64})
    return @. [8.0 * pi^2 * sinpi(2.0 * x[:,1]) * sinpi(2.0 * x[:,2]), 8.0 * pi^2 * sinpi(2.0 * x[:,1]) * sinpi(2.0 * x[:,2])]
end

# Test on multiple geometries
q_rule = Mantis.Quadrature.tensor_product_rule((deg1+25, deg2+25), Mantis.Quadrature.gauss_legendre)
for geom in [geom_cart, geom_crazy]#[geom_cart, geom_crazy]
    println("Geom type: ",typeof(geom))
    
    # Create form spaces
    zero_form_space = Mantis.Forms.FormSpace(0, geom, TP_Space_0, "ν")
    one_form_space = Mantis.Forms.FormSpace(1, geom, TP_Space_1, "η")
    top_form_space = Mantis.Forms.FormSpace(2, geom, TP_Space_2, "σ")

    # Generate the form expressions
    α⁰ = Mantis.Forms.FormField(zero_form_space, "α")
    α⁰.coefficients .= 1.0
    ζ¹ = Mantis.Forms.FormField(one_form_space, "ζ")
    ζ¹.coefficients .= 1.0
    constdx = Mantis.Forms.FormField(one_form_space, "ζ")
    constdx.coefficients[1:Mantis.FunctionSpaces.get_num_basis(dTP_Space_dx)] .= 1.0
    constdy = Mantis.Forms.FormField(one_form_space, "ζ")
    constdy.coefficients[Mantis.FunctionSpaces.get_num_basis(dTP_Space_dx)+1:end] .= 1.0
    dα⁰ = Mantis.Forms.exterior_derivative(α⁰)
    γ² = Mantis.Forms.FormField(top_form_space, "γ")
    γ².coefficients .= 1.0
    dζ¹ = Mantis.Forms.exterior_derivative(ζ¹)

    ★α⁰ = Mantis.Forms.hodge(α⁰)
    ★ζ¹ = Mantis.Forms.hodge(ζ¹)
    ★γ² = Mantis.Forms.hodge(γ²)

    # AnalyticalFormFields
    f⁰_analytic = Mantis.Forms.AnalyticalFormField(0, analytical_1valued_form_func, geom, "f")
    f¹_analytic = Mantis.Forms.AnalyticalFormField(1, analytical_2valued_form_func, geom, "f")
    f²_analytic = Mantis.Forms.AnalyticalFormField(2, analytical_1valued_form_func, geom, "f")
    
    total_integrated_analytical_field_0 = 0.0
    total_integrated_analytical_field_1 = 0.0
    total_integrated_analytical_field_n = 0.0
    for elem_id in 1:1:Mantis.Geometry.get_num_elements(geom)
        # Tests to see if the integrated metric terms are correctly recovered.
        inv_g, g, det_g = Mantis.Geometry.inv_metric(geom, elem_id, Mantis.Quadrature.get_quadrature_nodes(q_rule))
        # 0-forms
        integrated_metric_0 = sum(Mantis.Quadrature.get_quadrature_weights(q_rule) .* det_g)
        # @test all(isapprox.(Mantis.Forms.evaluate_inner_product(α⁰, α⁰, elem_id, q_rule)[3], integrated_metric_0, atol=1e-12))
        @test isapprox(sum(Mantis.Forms.evaluate_inner_product(zero_form_space, zero_form_space, elem_id, q_rule)[3]), integrated_metric_0, atol=1e-12) # same as the previous, but different wat of doing it (field vs space)
        
        # Because forms are defined on the parametric domain, the constant coefficents equal to 1 will not result 
        # in form equal to 1 evaluated on each element, therefore we need to account for it
        element_dimensions = Mantis.Geometry.get_element_dimensions(geom, elem_id)
        
        # 1-forms
        integrated_metric_1 = zeros((2,2))

        for node_idx in eachindex(Mantis.Quadrature.get_quadrature_weights(q_rule))
            integrated_metric_1 .+= Mantis.Quadrature.get_quadrature_weights(q_rule)[node_idx] .* ((inv_g[node_idx,:,:]).*det_g[node_idx])
        end
        
        # < ζ¹, ζ¹ > with ζ¹ a form with constant coefficients all equal to α then the inner product,
        # due to the partition of unity property over each element will be 
        #   < ζ¹, ζ¹ > = < α dξ¹ + α dξ²,  α dξ¹ + α dξ² > = [α α] gⁱʲ [α α]ᵗ √det(g)
        reference_result = dot(element_dimensions, integrated_metric_1*element_dimensions)
        @test isapprox(Mantis.Forms.evaluate_inner_product(ζ¹, ζ¹, elem_id, q_rule)[3][1], reference_result, atol=1e-12)
        
        # <dα⁰, dα⁰> for α⁰ a form with constant coefficients, is
        #   <dα⁰, dα⁰> = 0 
        # because dα⁰ = 0
        reference_result = 0.0
        @test isapprox(Mantis.Forms.evaluate_inner_product(dα⁰, dα⁰, elem_id, q_rule)[3][1], reference_result, atol=1e-12)
        
        # < α₁ dξ¹, α₂ dξ² > due to the partition of unity property over each element will be 
        #   < α₁ dξ¹, α₂ dξ² > = [α₁ 0] gⁱʲ [0 α₂]ᵗ √det(g) = α₁ g¹² α₂
        reference_result = element_dimensions[1] * integrated_metric_1[1,2] * element_dimensions[2]
        @test isapprox(Mantis.Forms.evaluate_inner_product(constdx, constdy, elem_id, q_rule)[3][1], reference_result, atol=1e-12)
        reference_result = element_dimensions[2] * integrated_metric_1[2,1] * element_dimensions[1]
        @test isapprox(Mantis.Forms.evaluate_inner_product(constdy, constdx, elem_id, q_rule)[3][1], element_dimensions[2] * integrated_metric_1[2,1] * element_dimensions[1], atol=1e-12)
        
        # This test follows the same logive as the < ζ¹, ζ¹ > test above
        reference_result = dot(element_dimensions, integrated_metric_1*element_dimensions)
        @test isapprox(sum(Mantis.Forms.evaluate_inner_product(one_form_space, one_form_space, elem_id, q_rule)[3]), reference_result, atol=1e-12)
        
        # 2-forms
        integrated_metric_2 = sum(Mantis.Quadrature.get_quadrature_weights(q_rule) .* (1.0./det_g))
    
        reference_result = integrated_metric_2 * prod(element_dimensions.^2)
        @test isapprox(Mantis.Forms.evaluate_inner_product(γ², γ², elem_id, q_rule)[3][1], reference_result, atol=1e-12)
        @test isapprox(Mantis.Forms.evaluate_inner_product(dζ¹, dζ¹, elem_id, q_rule)[3][1], 0.0, atol=1e-12)
        @test isapprox(sum(Mantis.Forms.evaluate_inner_product(top_form_space, top_form_space, elem_id, q_rule)[3]), reference_result, atol=1e-12)

        # Test if the inner product of the hodges of the forms equals that of the forms
        @test isapprox(Mantis.Forms.evaluate_inner_product(★α⁰, ★α⁰, elem_id, q_rule)[3], Mantis.Forms.evaluate_inner_product(α⁰, α⁰, elem_id, q_rule)[3], atol=1e-12)
        @test isapprox(Mantis.Forms.evaluate_inner_product(★ζ¹, ★ζ¹, elem_id, q_rule)[3], Mantis.Forms.evaluate_inner_product(ζ¹, ζ¹, elem_id, q_rule)[3], atol=1e-12)
        @test isapprox(Mantis.Forms.evaluate_inner_product(★γ², ★γ², elem_id, q_rule)[3], Mantis.Forms.evaluate_inner_product(γ², γ², elem_id, q_rule)[3], atol=1e-12)

        # AnalyticalFormField tests
        # 0-form
        total_integrated_analytical_field_0 += Mantis.Forms.evaluate_inner_product(f⁰_analytic, f⁰_analytic, elem_id, q_rule)[3][1]
        total_integrated_analytical_field_1 += Mantis.Forms.evaluate_inner_product(f¹_analytic, f¹_analytic, elem_id, q_rule)[3][1]
        total_integrated_analytical_field_n += Mantis.Forms.evaluate_inner_product(f²_analytic, f²_analytic, elem_id, q_rule)[3][1]
    end
    @test isapprox(total_integrated_analytical_field_0, 1558.5454565440389, atol=1e-12)
    @test isapprox(total_integrated_analytical_field_1/2, 1558.5454565440389, atol=1e-12)
    @test isapprox(total_integrated_analytical_field_n, 1558.5454565440389, atol=1e-12)
end


# # Then the geometry 
# # Line 1
# line_1_geo = Mantis.Geometry.CartesianGeometry((breakpoints1,))
# # Line 2
# line_2_geo = Mantis.Geometry.CartesianGeometry((breakpoints2,))
# # Tensor product geometry 
# tensor_prod_geo = Mantis.Geometry.TensorProductGeometry(line_1_geo, line_2_geo)
# geo_2d_cart = Mantis.Geometry.CartesianGeometry((breakpoints1, breakpoints2)) # equivalent to the above tensor-product geometry

# # Crazy mesh
# crazy_c = 0.2
# function mapping(x::Vector{Float64})
#     x1_new = (2.0/(Lright-Lleft))*x[1] - 2.0*Lleft/(Lright-Lleft) - 1.0
#     x2_new = (2.0/(Ltop-Lbottom))*x[2] - 2.0*Lbottom/(Ltop-Lbottom) - 1.0
#     return [x[1] + ((Lright-Lleft)/2.0)*crazy_c*sinpi(x1_new)*sinpi(x2_new), x[2] + ((Ltop-Lbottom)/2.0)*crazy_c*sinpi(x1_new)*sinpi(x2_new)]
# end
# function dmapping(x::Vector{Float64})
#     x1_new = (2.0/(Lright-Lleft))*x[1] - 2.0*Lleft/(Lright-Lleft) - 1.0
#     x2_new = (2.0/(Ltop-Lbottom))*x[2] - 2.0*Lbottom/(Ltop-Lbottom) - 1.0
#     return [1.0 + pi*crazy_c*cospi(x1_new)*sinpi(x2_new) ((Lright-Lleft)/(Ltop-Lbottom))*pi*crazy_c*sinpi(x1_new)*cospi(x2_new); ((Ltop-Lbottom)/(Lright-Lleft))*pi*crazy_c*cospi(x1_new)*sinpi(x2_new) 1.0 + pi*crazy_c*sinpi(x1_new)*cospi(x2_new)]
# end

# dimension = (2, 2)
# crazy_mapping = Mantis.Geometry.Mapping(dimension, mapping, dmapping)
# geom_crazy = Mantis.Geometry.MappedGeometry(geo_2d_cart, crazy_mapping)

# q_rule = Mantis.Quadrature.tensor_product_rule((deg1+25, deg2+25), Mantis.Quadrature.gauss_legendre)

# # Create some non-constant analytical form fields to test the inner product.
# function analytical_form_func(x::Matrix{Float64})
#     return [@. 8.0 * pi^2 * sinpi(2.0 * x[:,1]) * sinpi(2.0 * x[:,2])]
# end
# function analytical_1_form_func(x::Matrix{Float64})
#     return [@. sinpi(2.0 * x[:,1]) * sinpi(2.0 * x[:,2]), @. cospi(2.0 * x[:,1]) * cospi(2.0 * x[:,2])]
# end

# # Test on multiple geometries. Type-wise and content/metric wise.
# for geom in [geo_2d_cart, tensor_prod_geo, geom_crazy]
#     println("Geom type: ",typeof(geom))
#     # Create form spaces
#     zero_form_space = Mantis.Forms.FormSpace(0, geom, (TP_Space,), "ν")
#     one_form_space = Mantis.Forms.FormSpace(1, geom, (TP_Space, TP_Space), "η")
#     top_form_space = Mantis.Forms.FormSpace(2, geom, (TP_Space,), "σ")

#     # Generate the form expressions
#     α⁰ = Mantis.Forms.FormField(zero_form_space, "α")
#     α⁰.coefficients .= 1.0
#     ζ¹ = Mantis.Forms.FormField(one_form_space, "ζ")
#     ζ¹.coefficients .= 1.0
#     constdx = Mantis.Forms.FormField(one_form_space, "ζ")
#     constdx.coefficients[begin:20] .= 1.0
#     constdy = Mantis.Forms.FormField(one_form_space, "ζ")
#     constdy.coefficients[21:end] .= 1.0
#     dα⁰ = Mantis.Forms.exterior_derivative(α⁰)
#     γ² = Mantis.Forms.FormField(top_form_space, "γ")
#     γ².coefficients .= 1.0
#     dζ¹ = Mantis.Forms.exterior_derivative(ζ¹)

#     ★α⁰ = Mantis.Forms.hodge(α⁰)
#     ★ζ¹ = Mantis.Forms.hodge(ζ¹)
#     ★γ² = Mantis.Forms.hodge(γ²)

#     # AnalyticalFormFields
#     f⁰_analytic = Mantis.Forms.AnalyticalFormField(0, analytical_form_func, geom, "f")
#     f¹_analytic = Mantis.Forms.AnalyticalFormField(1, analytical_1_form_func, geom, "f")
#     f²_analytic = Mantis.Forms.AnalyticalFormField(2, analytical_form_func, geom, "f")
    
#     total_integrated_analytical_field_0 = 0.0
#     total_integrated_analytical_field_n = 0.0
#     for elem_id in 1:1:Mantis.Geometry.get_num_elements(geom)
#         # Note that we cannot do mixed inner products

#         # Tests to see if the integrated metric terms are correctly recovered.
#         inv_g, g, det_g = Mantis.Geometry.inv_metric(geom, elem_id, Mantis.Quadrature.get_quadrature_nodes(q_rule))
#         # 0-forms
#         integrated_metric_0 = sum(Mantis.Quadrature.get_quadrature_weights(q_rule) .* det_g)
#         @test all(isapprox.(Mantis.Forms.evaluate_inner_product(α⁰, α⁰, elem_id, q_rule)[3][1], integrated_metric_0, atol=1e-12))
#         @test isapprox(sum(Matrix(SparseArrays.sparse(Mantis.Forms.evaluate_inner_product(zero_form_space, zero_form_space, elem_id, q_rule)...))), integrated_metric_0, atol=1e-12) # same as the previous, but different wat of doing it (field vs space)
        
#         # 1-forms
#         integrated_metric_1 = zeros((2,2))
#         for node_idx in eachindex(Mantis.Quadrature.get_quadrature_weights(q_rule))
#             integrated_metric_1 .+= Mantis.Quadrature.get_quadrature_weights(q_rule)[node_idx] .* (inv_g.*det_g)[node_idx,:,:]
#         end
#         @test isapprox(Matrix(SparseArrays.sparse(Mantis.Forms.evaluate_inner_product(ζ¹, ζ¹, elem_id, q_rule)...))[1], sum(integrated_metric_1), atol=1e-12)
#         @test isapprox(Mantis.Forms.evaluate_inner_product(dα⁰, dα⁰, elem_id, q_rule)[3][1], 0.0, atol=1e-12)
#         @test isapprox(Matrix(SparseArrays.sparse(Mantis.Forms.evaluate_inner_product(constdx, constdy, elem_id, q_rule)...))[1], integrated_metric_1[1,2], atol=1e-12)
#         @test isapprox(Matrix(SparseArrays.sparse(Mantis.Forms.evaluate_inner_product(constdy, constdx, elem_id, q_rule)...))[1], integrated_metric_1[2,1], atol=1e-12)
        
#         @test isapprox(sum(Matrix(SparseArrays.sparse(Mantis.Forms.evaluate_inner_product(one_form_space, one_form_space, elem_id, q_rule)...))), sum(integrated_metric_1), atol=1e-12)
        
#         # n-forms
#         integrated_metric_2 = sum(Mantis.Quadrature.get_quadrature_weights(q_rule) .* (1.0./det_g))
#         @test all(isapprox.(Mantis.Forms.evaluate_inner_product(γ², γ², elem_id, q_rule)[3][1], integrated_metric_2, atol=1e-12))
#         @test isapprox(Mantis.Forms.evaluate_inner_product(dζ¹, dζ¹, elem_id, q_rule)[3][1], 0.0, atol=1e-12)
#         @test isapprox(sum(Matrix(SparseArrays.sparse(Mantis.Forms.evaluate_inner_product(top_form_space, top_form_space, elem_id, q_rule)...))), integrated_metric_2, atol=1e-12)


#         # NOTE: We cannot do inner product of mixed forms yet, so this doesn't work yet!
#         # Test to check if the inner product between a form and its 
#         # Hodge dual is done correctly. This is a duality pairing and 
#         # can be directly computed using the coefficients of the forms.
#         #display(Matrix(SparseArrays.sparse(Mantis.Forms.evaluate_inner_product(★α⁰, α⁰, elem_id, q_rule)...)))

#         # Test if the inner product of the hodges of the forms equals that of the forms
#         @test isapprox(Matrix(SparseArrays.sparse(Mantis.Forms.evaluate_inner_product(★α⁰, ★α⁰, elem_id, q_rule)...)), Matrix(SparseArrays.sparse(Mantis.Forms.evaluate_inner_product(α⁰, α⁰, elem_id, q_rule)...)), atol=1e-12)
#         @test isapprox(Matrix(SparseArrays.sparse(Mantis.Forms.evaluate_inner_product(★ζ¹, ★ζ¹, elem_id, q_rule)...)), Matrix(SparseArrays.sparse(Mantis.Forms.evaluate_inner_product(ζ¹, ζ¹, elem_id, q_rule)...)), atol=1e-12)
#         @test isapprox(Matrix(SparseArrays.sparse(Mantis.Forms.evaluate_inner_product(★γ², ★γ², elem_id, q_rule)...)), Matrix(SparseArrays.sparse(Mantis.Forms.evaluate_inner_product(γ², γ², elem_id, q_rule)...)), atol=1e-12)


#         # AnalyticalFormField tests
#         # 0-form
#         total_integrated_analytical_field_0 += Mantis.Forms.evaluate_inner_product(f⁰_analytic, f⁰_analytic, elem_id, q_rule)[3][1]
#         total_integrated_analytical_field_n += Mantis.Forms.evaluate_inner_product(f²_analytic, f²_analytic, elem_id, q_rule)[3][1]
#     end
#     @test isapprox(total_integrated_analytical_field_0, 1558.5454565440389, atol=1e-12)
#     @test isapprox(total_integrated_analytical_field_n, 1558.5454565440389, atol=1e-12)
# end

# # # 3d
# # geo_3d_cart = Mantis.Geometry.CartesianGeometry((breakpoints1, breakpoints2, breakpoints1))
# # zero_form_space_cart = Mantis.Forms.FormSpace(0, geo_3d_cart, (TP_Space3d,), "ν")
# # one_form_space_cart = Mantis.Forms.FormSpace(1, geo_3d_cart, (TP_Space3d, TP_Space3d, TP_Space3d), "θ")
# # two_form_space_cart = Mantis.Forms.FormSpace(2, geo_3d_cart, (TP_Space3d, TP_Space3d, TP_Space3d), "η")
# # top_form_space_cart = Mantis.Forms.FormSpace(3, geo_3d_cart, (TP_Space3d,), "σ")

# # # Generate the form expressions
# # α⁰ = Mantis.Forms.FormField(zero_form_space_cart, "α")
# # α⁰.coefficients .= 1.0
# # θ¹ = Mantis.Forms.FormField(one_form_space_cart, "θ")
# # θ¹.coefficients .= 1.0
# # ζ² = Mantis.Forms.FormField(two_form_space_cart, "ζ")
# # ζ².coefficients .= 1.0
# # dα⁰ = Mantis.Forms.exterior_derivative(α⁰)
# # γ³ = Mantis.Forms.FormField(top_form_space_cart, "γ")
# # γ³.coefficients .= 1.0

# # q_rule = Mantis.Quadrature.tensor_product_rule((deg1+1, deg2+1, deg1+1), Mantis.Quadrature.gauss_legendre)
# # # Note that we cannot do mixed inner products

# # for elem_id in 1:1:Mantis.Geometry.get_num_elements(geo_3d_cart)
# #     ordered_idx = Tuple(geo_3d_cart.cartesian_idxs[elem_id])
# #     elem_vol = 1.0
# #     for dim_idx in range(1, 3)
# #         start_breakpoint_idx = ordered_idx[dim_idx]
# #         end_breakpoint_idx = ordered_idx[dim_idx] + 1
# #         elem_vol *= geo_3d_cart.breakpoints[dim_idx][end_breakpoint_idx] - geo_3d_cart.breakpoints[dim_idx][start_breakpoint_idx]
# #     end

# #     g_inv, g, det_g = Mantis.Geometry.inv_metric(geo_3d_cart, elem_id, Mantis.Quadrature.get_quadrature_nodes(q_rule))

# #     # 0-forms
# #     @test isapprox(Mantis.Forms.evaluate_inner_product(α⁰, α⁰, elem_id, q_rule)[3][1], elem_vol, atol=1e-12)

# #     # 1-forms
# #     @test isapprox(sum(Mantis.Forms.evaluate_inner_product(θ¹, θ¹, elem_id, q_rule)[3]), sum((g_inv.*det_g)[1,:,:]), atol=1e-12)

# #     # 2-forms
# #     @test isapprox(sum(Mantis.Forms.evaluate_inner_product(ζ², ζ², elem_id, q_rule)[3]), sum((g./det_g)[1,:,:]), atol=1e-12)

# #     # n-forms
# #     @test isapprox(Mantis.Forms.evaluate_inner_product(γ³, γ³, elem_id, q_rule)[3][1], 1/elem_vol, atol=1e-12)
# # end