module ExteriorDerivativeTests

using Mantis

using Test

using LinearAlgebra, SparseArrays

# Domain
Lleft = 0.0
Lright = 1.0
Lbottom = 0.0
Ltop = 1.0

# Setup the form spaces
# First the FEM spaces
breakpoints1 = [Lleft, 0.5, Lright]
patch1 = Mesh.Patch1D(breakpoints1)
breakpoints2 = [Lbottom, 0.5, 0.6, Ltop]
patch2 = Mesh.Patch1D(breakpoints2)

# first B-spline patch
deg1 = 2
deg2 = 2
B1 = FunctionSpaces.BSplineSpace(patch1, deg1, [-1, deg1 - 1, -1])
# second B-spline patch
B2 = FunctionSpaces.BSplineSpace(patch2, deg2, [-1, min(deg2 - 1, 1), deg2 - 1, -1])
# tensor-product B-spline patch
TP_Space_2d = FunctionSpaces.TensorProductSpace((B1, B2))
TP_Space_3d = FunctionSpaces.TensorProductSpace((TP_Space_2d, B1))

# Direct sum spaces for the vector-valued forms.
dsTP_1_form_2d = FunctionSpaces.DirectSumSpace((TP_Space_2d, TP_Space_2d))

# Then the geometry
# Line 1
line_1_geo = Geometry.CartesianGeometry((breakpoints1,))

# Line 2
line_2_geo = Geometry.CartesianGeometry((breakpoints2,))

# Tensor product geometry
tensor_prod_geo = Geometry.TensorProductGeometry((line_1_geo, line_2_geo))
geo_2d_cart = Geometry.CartesianGeometry((breakpoints1, breakpoints2))

# Crazy mesh
crazy_c = 0.2
function mapping_ed_test(x::Vector{Float64})
    x1_new = (2.0 / (Lright - Lleft)) * x[1] - 2.0 * Lleft / (Lright - Lleft) - 1.0
    x2_new = (2.0 / (Ltop - Lbottom)) * x[2] - 2.0 * Lbottom / (Ltop - Lbottom) - 1.0

    return [
        x[1] + ((Lright - Lleft) / 2.0) * crazy_c * sinpi(x1_new) * sinpi(x2_new),
        x[2] + ((Ltop - Lbottom) / 2.0) * crazy_c * sinpi(x1_new) * sinpi(x2_new),
    ]
end

function dmapping_ed_test(x::Vector{Float64})
    x1_new = (2.0 / (Lright - Lleft)) * x[1] - 2.0 * Lleft / (Lright - Lleft) - 1.0
    x2_new = (2.0 / (Ltop - Lbottom)) * x[2] - 2.0 * Lbottom / (Ltop - Lbottom) - 1.0

    return [
        1.0+pi * crazy_c * cospi(x1_new) * sinpi(x2_new) ((Lright - Lleft)/(Ltop - Lbottom))*pi*crazy_c*sinpi(x1_new)*cospi(x2_new)
        ((Ltop - Lbottom)/(Lright - Lleft))*pi*crazy_c*cospi(x1_new)*sinpi(x2_new) 1.0+pi * crazy_c * sinpi(x1_new) * cospi(x2_new)
    ]
end

dimension = (2, 2)
crazy_mapping = Geometry.Mapping(dimension, mapping_ed_test, dmapping_ed_test)
geom_crazy = Geometry.MappedGeometry(geo_2d_cart, crazy_mapping)

q_rule = Quadrature.tensor_product_rule((deg1 + 1, deg2 + 1), Quadrature.gauss_legendre)

# Test on multiple geometries. Type-wise and content/metric wise.
for geom in [geo_2d_cart, tensor_prod_geo, geom_crazy]
    # Create form spaces
    zero_form_space = Forms.FormSpace(0, geom, TP_Space_2d, "ν")
    one_form_space = Forms.FormSpace(1, geom, dsTP_1_form_2d, "η")
    top_form_space = Forms.FormSpace(2, geom, TP_Space_2d, "σ")

    # Generate the form expressions
    # 0-form: constant
    α⁰ = Forms.FormField(zero_form_space, "α")
    α⁰.coefficients .= 1.0

    # 1-form: constant
    ζ¹ = Forms.FormField(one_form_space, "ζ")
    ζ¹.coefficients .= 1.0

    # Compute exterior derivatives
    dα⁰ = Forms.ExteriorDerivative(α⁰)
    dζ¹ = Forms.ExteriorDerivative(ζ¹)

    top_field = Forms.FormField(top_form_space, "top")
    @test_throws ArgumentError Forms.ExteriorDerivative(top_form_space)
    @test_throws ArgumentError Forms.ExteriorDerivative(top_field)

    # Test exterior derivatives
    for elem_id in 1:1:Geometry.get_num_elements(geom)
        # 0-form
        # Exterior derivative of a unity 0-form is a zero 1-form
        @test all(
            isapprox(
                sum(abs.(Forms.evaluate(dα⁰, elem_id, Quadrature.get_nodes(q_rule))[1][1])),
                0.0;
                atol=1e-12,
            ),
        )
        @test all(
            isapprox(
                sum(abs.(Forms.evaluate(dα⁰, elem_id, Quadrature.get_nodes(q_rule))[1][1])),
                0.0;
                atol=1e-12,
            ),
        )
        @test all(
            isapprox(
                sum(abs.(Forms.evaluate(dα⁰, elem_id, Quadrature.get_nodes(q_rule))[1][2])),
                0.0;
                atol=1e-12,
            ),
        )

        # 1-form
        # Exterior derivative of a unity 1-form is a zero 2-form
        @test all(
            isapprox(
                sum(abs.(Forms.evaluate(dζ¹, elem_id, Quadrature.get_nodes(q_rule))[1][1])),
                0.0;
                atol=1e-12,
            ),
        )
    end
end

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
patch = Mesh.Patch1D(breakpoints)

# first B-spline patch
deg = 2
B = FunctionSpaces.BSplineSpace(patch, deg, [-1, deg - 1, -1])

# tensor-product B-spline patch
TP_Space_2d = FunctionSpaces.TensorProductSpace((B, B))
TP_Space_3d = FunctionSpaces.TensorProductSpace((TP_Space_2d, B))

# Then the geometry
line_geo = Geometry.CartesianGeometry((breakpoints,))
geo_3d_cart = Geometry.CartesianGeometry((breakpoints, breakpoints, breakpoints))

# Crazy Tensor product geometry in 2D (auxiliary)
tensor_prod_geo_2d = Geometry.TensorProductGeometry((line_geo, line_geo))
geo_2d_cart_aux = Geometry.CartesianGeometry((breakpoints, breakpoints))
crazy_geo_2d_cart = Geometry.MappedGeometry(geo_2d_cart_aux, crazy_mapping)

# Crazy mesh 3D (in x and y only, z is straight)
crazy_geo_3d_cart = Geometry.TensorProductGeometry((crazy_geo_2d_cart, line_geo))

# Setup the form spaces

# Generate the multicomponent FEMSpaces for the vector-valued forms.
dsTP_1_form_3d = FunctionSpaces.DirectSumSpace((TP_Space_3d, TP_Space_3d, TP_Space_3d))
dsTP_2_form_3d = FunctionSpaces.DirectSumSpace((TP_Space_3d, TP_Space_3d, TP_Space_3d))

# Quadrature rule
q_rule = Quadrature.tensor_product_rule((deg, deg, deg) .+ 1, Quadrature.gauss_legendre)

# Test on multiple geometries. Type-wise and content/metric wise.
for geom in [geo_3d_cart, crazy_geo_3d_cart]
    # Create form spaces
    zero_form_space_3d = Forms.FormSpace(0, geom, TP_Space_3d, "ν")
    one_form_space_3d = Forms.FormSpace(1, geom, dsTP_1_form_3d, "η")
    two_form_space_3d = Forms.FormSpace(2, geom, dsTP_2_form_3d, "μ")
    top_form_space_3d = Forms.FormSpace(3, geom, TP_Space_3d, "u")

    # Generate the form expressions
    # 0-form: constant
    γ⁰ = Forms.FormField(zero_form_space_3d, "γ")
    γ⁰.coefficients .= 1.0

    # 1-form: constant
    ι¹ = Forms.FormField(one_form_space_3d, "ι")
    ι¹.coefficients .= 1.0

    # 2-form: constant
    β² = Forms.FormField(two_form_space_3d, "β")
    β².coefficients .= 1.0

    # Exterior derivative of all forms
    dγ⁰ = Forms.ExteriorDerivative(γ⁰)
    dι¹ = Forms.ExteriorDerivative(ι¹)
    dβ² = Forms.ExteriorDerivative(β²)

    top_field_3d = Forms.FormField(top_form_space_3d, "top")
    @test_throws ArgumentError Forms.ExteriorDerivative(top_form_space_3d)
    @test_throws ArgumentError Forms.ExteriorDerivative(top_field_3d)

    for elem_id in 1:Geometry.get_num_elements(geom)
        # 0-form
        # Exterior derivative of a unity 0-form is a zero 1-form
        @test all(
            isapprox(
                sum(abs.(Forms.evaluate(dγ⁰, elem_id, Quadrature.get_nodes(q_rule))[1][1])),
                0.0;
                atol=1e-12,
            ),
        )
        @test all(
            isapprox(
                sum(abs.(Forms.evaluate(dγ⁰, elem_id, Quadrature.get_nodes(q_rule))[1][2])),
                0.0;
                atol=1e-12,
            ),
        )
        @test all(
            isapprox(
                sum(abs.(Forms.evaluate(dγ⁰, elem_id, Quadrature.get_nodes(q_rule))[1][3])),
                0.0;
                atol=1e-12,
            ),
        )

        # 1-form
        # Exterior derivative of a unity 1-form is a zero 2-form
        @test all(
            isapprox(
                sum(abs.(Forms.evaluate(dι¹, elem_id, Quadrature.get_nodes(q_rule))[1][1])),
                0.0;
                atol=1e-12,
            ),
        )
        @test all(
            isapprox(
                sum(abs.(Forms.evaluate(dι¹, elem_id, Quadrature.get_nodes(q_rule))[1][2])),
                0.0;
                atol=1e-12,
            ),
        )
        @test all(
            isapprox(
                sum(abs.(Forms.evaluate(dι¹, elem_id, Quadrature.get_nodes(q_rule))[1][3])),
                0.0;
                atol=1e-12,
            ),
        )

        # 2-form
        # Exterior derivative of a unity 1-form is a zero 3-form
        @test all(
            isapprox(
                sum(abs.(Forms.evaluate(dβ², elem_id, Quadrature.get_nodes(q_rule))[1][1])),
                0.0;
                atol=1e-12,
            ),
        )
    end
end

# -----------------------------------------------------------------------------
end
