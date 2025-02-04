module WedgeTests

import Mantis

using Test

using LinearAlgebra
using SparseArrays

############################################################################################
##                                       Testing methods                                  ##
############################################################################################
function test_inner_prod_equality(geom::G, q_rule, complex) where {
    manifold_dim, G <: Mantis.Geometry.AbstractGeometry{manifold_dim}
}
    quad_nodes = Mantis.Quadrature.get_nodes(q_rule)
    quad_weights = Mantis.Quadrature.get_weights(q_rule)

    for rank in 0:manifold_dim
        # Generate the form space ϵᵏ, or form rank k, and matching hodge and form field.
        ϵ = complex[rank + 1]
        ★ϵ = Mantis.Forms.hodge(ϵ)
        α⁰ = Mantis.Forms.AnalyticalFormField(
            0, zero_form_expression, Mantis.Forms.get_geometry(ϵ), "α⁰"
        )

        for elem_id in 1:Mantis.Geometry.get_num_elements(geom)
            # Compare the inner product
            #   <ϵ, ϵ>
            # with
            #   ϵ ∧ ★ϵ
            # Compute the inner product: <ϵ, ϵ>
            _, _, inner_product_eval = 
                Mantis.Forms.evaluate_inner_product(ϵ, ϵ, elem_id, q_rule)

            # Compute the wedge product: ϵ ∧ ⋆ϵ
            wedge = Mantis.Forms.wedge(ϵ, ★ϵ)

            # Evaluate it at the quadrature nodes
            wedge_eval, _ = Mantis.Forms.evaluate(
                wedge, elem_id, quad_nodes
            )

            # Integrate the wedge product
            wedge_integral_eval = vec(sum(quad_weights .* wedge_eval[1]; dims=1))

            # Test if wedge is the same as the inner product
            @test all(isapprox.(wedge_integral_eval, inner_product_eval; atol=1e-12))

            # Test if ϵ ∧ ★ϵ == ϵ ∧ ★ϵ ∧ α⁰ (since α⁰ = 1)
            wedge_triple = Mantis.Forms.wedge(wedge, α⁰)
            wedge_triple_eval, _ = Mantis.Forms.evaluate(
                wedge_triple, elem_id, quad_nodes
            )
            @test all(isapprox.(wedge_eval, wedge_triple_eval; atol=1e-12))
                
            # Test if the wedge product expression has the correct expression rank
            @test Mantis.Forms.get_expression_rank(wedge) == 2
        end
        
        # Test ϵ¹ ∧ ⋆ϵ¹ ∧ ϵ¹ to check if form rank 3 expressions are not allowed
        if rank == 1
            wedge = Mantis.Forms.wedge(ϵ, ★ϵ)
            @test_throws ArgumentError Mantis.Forms.wedge(wedge, ϵ)
        end

        # Test ϵⁿ ∧ ϵⁿ (volume forms) throws error
        if rank == manifold_dim
            @test_throws ArgumentError Mantis.Forms.wedge(ϵ, ϵ)
        end
    end
end

function test_combinations_2d(complex, q_rule)

    # Create the form spaces.
    ϵ⁰ = complex[1]
    ϵ¹ = complex[2]
    ϵ² = complex[3]

    # Create the form fields.
    ε⁰ = Mantis.Forms.FormField(ϵ⁰, "ε⁰")
    ε¹ = Mantis.Forms.FormField(ϵ¹, "ε¹")
    ε² = Mantis.Forms.FormField(ϵ², "ε²")
    
    # Test if different form-/expression-rank combinations don't throw errors
    zero_form_space_wedge = Mantis.Forms.wedge(ϵ⁰, ϵ⁰) # 0-form space with 0-form space
    zero_form_field_wedge = Mantis.Forms.wedge(ε⁰, ε⁰) # 0-form field with 0-form field
    zero_form_mixed_1_wedge = Mantis.Forms.wedge(ϵ⁰, ε⁰) # 0-form space with 0-form field
    zero_form_mixed_2_wedge = Mantis.Forms.wedge(ε⁰, ϵ⁰) # 0-form field with 0-form space
    zero_top_form_space_wedge = Mantis.Forms.wedge(ϵ⁰, ϵ²) # 0-form space with top-form space
    top_zero_form_space_wedge = Mantis.Forms.wedge(ϵ², ϵ⁰) # top-form space with 0-form space
    zero_top_form_mixed_1_wedge = Mantis.Forms.wedge(ϵ⁰, ε²) # 0-form space with top-form field
    zero_top_form_mixed_2_wedge = Mantis.Forms.wedge(ε⁰, ϵ²) # 0-form field with top-form space
    top_zero_form_mixed_1_wedge = Mantis.Forms.wedge(ϵ², ε⁰) # top-form space with 0-form field
    top_zero_form_mixed_2_wedge = Mantis.Forms.wedge(ε², ϵ⁰) # top-form field with 0-form space
    top_zero_form_field_wedge = Mantis.Forms.wedge(ε², ε⁰) # top-form field with 0-form field
    zero_top_form_field_wedge = Mantis.Forms.wedge(ε⁰, ε²) # 0-form field with top-form field
    one_form_space_wedge = Mantis.Forms.wedge(ϵ¹, ϵ¹) # 1-form space with 1-form space
    one_form_field_wedge = Mantis.Forms.wedge(ε¹, ε¹) # 1-form field with 1-form field
    one_form_mixed_1_wedge = Mantis.Forms.wedge(ϵ¹, ε¹) # 1-form space with 1-form field
    one_form_mixed_2_wedge = Mantis.Forms.wedge(ε¹, ϵ¹) # 1-form field with 1-form space
    one_zero_form_space_wedge = Mantis.Forms.wedge(ϵ¹, ϵ⁰) # 1-form space with 0-form space
    one_zero_form_field_wedge = Mantis.Forms.wedge(ε¹, ε⁰) # 1-form field with 0-form field
    one_zero_form_mixed_1_wedge = Mantis.Forms.wedge(ϵ¹, ε⁰) # 1-form space with 0-form field
    one_zero_form_mixed_2_wedge = Mantis.Forms.wedge(ε¹, ϵ⁰) # 1-form field with 0-form space
    zero_one_form_space_wedge = Mantis.Forms.wedge(ϵ⁰, ϵ¹) # 0-form space with 1-form space
    zero_one_form_field_wedge = Mantis.Forms.wedge(ε⁰, ε¹) # 0-form field with 1-form field
    zero_one_form_mixed_1_wedge = Mantis.Forms.wedge(ϵ⁰, ε¹) # 0-form space with 1-form field
    zero_one_form_mixed_2_wedge = Mantis.Forms.wedge(ε⁰, ϵ¹) # 0-form field with 1-form space

    quad_nodes = Mantis.Quadrature.get_nodes(q_rule)

    @test_nowarn Mantis.Forms.evaluate(zero_form_space_wedge, 1, quad_nodes)
    @test_nowarn Mantis.Forms.evaluate(zero_form_field_wedge, 1, quad_nodes)
    @test_nowarn Mantis.Forms.evaluate(zero_form_mixed_1_wedge, 1, quad_nodes)
    @test_nowarn Mantis.Forms.evaluate(zero_form_mixed_2_wedge, 1, quad_nodes)
    @test_nowarn Mantis.Forms.evaluate(zero_top_form_space_wedge, 1, quad_nodes)
    @test_nowarn Mantis.Forms.evaluate(top_zero_form_space_wedge, 1, quad_nodes)
    @test_nowarn Mantis.Forms.evaluate(zero_top_form_mixed_1_wedge, 1, quad_nodes)
    @test_nowarn Mantis.Forms.evaluate(zero_top_form_mixed_2_wedge, 1, quad_nodes)
    @test_nowarn Mantis.Forms.evaluate(top_zero_form_mixed_1_wedge, 1, quad_nodes)
    @test_nowarn Mantis.Forms.evaluate(top_zero_form_mixed_2_wedge, 1, quad_nodes)
    @test_nowarn Mantis.Forms.evaluate(top_zero_form_field_wedge, 1, quad_nodes)
    @test_nowarn Mantis.Forms.evaluate(zero_top_form_field_wedge, 1, quad_nodes)
    @test_nowarn Mantis.Forms.evaluate(one_form_space_wedge, 1, quad_nodes)
    @test_nowarn Mantis.Forms.evaluate(one_form_field_wedge, 1, quad_nodes)
    @test_nowarn Mantis.Forms.evaluate(one_form_mixed_1_wedge, 1, quad_nodes)
    @test_nowarn Mantis.Forms.evaluate(one_form_mixed_2_wedge, 1, quad_nodes)
    @test_nowarn Mantis.Forms.evaluate(one_zero_form_space_wedge, 1, quad_nodes)
    @test_nowarn Mantis.Forms.evaluate(one_zero_form_field_wedge, 1, quad_nodes)
    @test_nowarn Mantis.Forms.evaluate(one_zero_form_mixed_1_wedge, 1, quad_nodes)
    @test_nowarn Mantis.Forms.evaluate(one_zero_form_mixed_2_wedge, 1, quad_nodes)
    @test_nowarn Mantis.Forms.evaluate(zero_one_form_space_wedge, 1, quad_nodes)
    @test_nowarn Mantis.Forms.evaluate(zero_one_form_field_wedge, 1, quad_nodes)
    @test_nowarn Mantis.Forms.evaluate(zero_one_form_mixed_1_wedge, 1, quad_nodes)
    @test_nowarn Mantis.Forms.evaluate(zero_one_form_mixed_2_wedge, 1, quad_nodes)
end

function test_combinations_3d(complex, q_rule)
    
    # Create form spaces
    ϵ⁰ = complex[1]
    ϵ¹ = complex[2] 
    ϵ² = complex[3] 
    ϵ³ = complex[4]

    # Create the form fields
    ε⁰ = Mantis.Forms.FormField(ϵ⁰, "ε⁰")
    ε¹ = Mantis.Forms.FormField(ϵ¹, "ε¹")
    ε² = Mantis.Forms.FormField(ϵ², "ε²")
    ε³ = Mantis.Forms.FormField(ϵ³, "ε³")

    # Test if different form-/expression-rank combinations don't throw errors
    zero_form_space_wedge = Mantis.Forms.wedge(ϵ⁰, ϵ⁰) # 0-form space with 0-form space
    zero_form_field_wedge = Mantis.Forms.wedge(ε⁰, ε⁰) # 0-form field with 0-form field
    zero_form_mixed_1_wedge = Mantis.Forms.wedge(ϵ⁰, ε⁰) # 0-form space with 0-form field
    zero_form_mixed_2_wedge = Mantis.Forms.wedge(ε⁰, ϵ⁰) # 0-form field with 0-form space
    zero_one_form_space_wedge = Mantis.Forms.wedge(ϵ⁰, ϵ¹) # 0-form space with 1-form space
    zero_one_form_field_wedge = Mantis.Forms.wedge(ε⁰, ε¹) # 0-form field with 1-form field
    zero_one_form_mixed_1_wedge = Mantis.Forms.wedge(ϵ⁰, ε¹) # 0-form space with 1-form field
    zero_one_form_mixed_2_wedge = Mantis.Forms.wedge(ε⁰, ϵ¹) # 0-form field with 1-form space
    one_form_space_wedge = Mantis.Forms.wedge(ϵ¹, ϵ¹) # 1-form space with 1-form space
    one_form_field_wedge = Mantis.Forms.wedge(ε¹, ε¹) # 1-form field with 1-form field
    one_form_mixed_1_wedge = Mantis.Forms.wedge(ϵ¹, ε¹) # 1-form space with 1-form field
    one_form_mixed_2_wedge = Mantis.Forms.wedge(ε¹, ϵ¹) # 1-form field with 1-form space
    one_zero_form_space_wedge = Mantis.Forms.wedge(ϵ¹, ϵ⁰) # 1-form space with 0-form space
    one_zero_form_field_wedge = Mantis.Forms.wedge(ε¹, ε⁰) # 1-form field with 0-form field
    one_zero_form_mixed_1_wedge = Mantis.Forms.wedge(ϵ¹, ε⁰) # 1-form space with 0-form field
    one_zero_form_mixed_2_wedge = Mantis.Forms.wedge(ε¹, ϵ⁰) # 1-form field with 0-form space
    zero_two_form_space_wedge = Mantis.Forms.wedge(ϵ⁰, ϵ²) # 0-form space with 2-form space
    zero_two_form_field_wedge = Mantis.Forms.wedge(ε⁰, ε²) # 0-form field with 2-form field
    zero_two_form_mixed_1_wedge = Mantis.Forms.wedge(ϵ⁰, ε²) # 0-form space with 2-form field
    zero_two_form_mixed_2_wedge = Mantis.Forms.wedge(ε⁰, ϵ²) # 0-form field with 2-form space
    two_zero_form_space_wedge = Mantis.Forms.wedge(ϵ², ϵ⁰) # 2-form space with 0-form space
    two_zero_form_field_wedge = Mantis.Forms.wedge(ε², ε⁰) # 2-form field with 0-form field
    two_zero_form_mixed_1_wedge = Mantis.Forms.wedge(ϵ², ε⁰) # 2-form space with 0-form field
    two_zero_form_mixed_2_wedge = Mantis.Forms.wedge(ε², ϵ⁰) # 2-form field with 0-form space
    one_two_form_space_wedge = Mantis.Forms.wedge(ϵ¹, ϵ²) # 1-form space with 2-form space
    one_two_form_field_wedge = Mantis.Forms.wedge(ε¹, ε²) # 1-form field with 2-form field
    one_two_form_mixed_1_wedge = Mantis.Forms.wedge(ϵ¹, ε²) # 1-form space with 2-form field
    one_two_form_mixed_2_wedge = Mantis.Forms.wedge(ε¹, ϵ²) # 1-form field with 2-form space
    two_one_form_space_wedge = Mantis.Forms.wedge(ϵ², ϵ¹) # 2-form space with 1-form space
    two_one_form_field_wedge = Mantis.Forms.wedge(ε², ε¹) # 2-form field with 1-form field
    two_one_form_mixed_1_wedge = Mantis.Forms.wedge(ϵ², ε¹) # 2-form space with 1-form field
    two_one_form_mixed_2_wedge = Mantis.Forms.wedge(ε², ϵ¹) # 2-form field with 1-form space
    zero_top_form_space_wedge = Mantis.Forms.wedge(ϵ⁰, ϵ³) # 0-form space with top-form space
    zero_top_form_field_wedge = Mantis.Forms.wedge(ε⁰, ε³) # 0-form field with top-form field
    zero_top_form_mixed_1_wedge = Mantis.Forms.wedge(ϵ⁰, ε³) # 0-form space with top-form field
    zero_top_form_mixed_2_wedge = Mantis.Forms.wedge(ε⁰, ϵ³) # 0-form field with top-form space
    top_zero_form_space_wedge = Mantis.Forms.wedge(ϵ³, ϵ⁰) # top-form space with 0-form space
    top_zero_form_field_wedge = Mantis.Forms.wedge(ε³, ε⁰) # top-form field with 0-form field
    top_zero_form_mixed_1_wedge = Mantis.Forms.wedge(ϵ³, ε⁰) # top-form space with 0-form field
    top_zero_form_mixed_2_wedge = Mantis.Forms.wedge(ε³, ϵ⁰) # top-form field with 0-form space

    quad_nodes = Mantis.Quadrature.get_nodes(q_rule)

    @test_nowarn Mantis.Forms.evaluate(zero_form_space_wedge, 1, quad_nodes)
    @test_nowarn Mantis.Forms.evaluate(zero_form_field_wedge, 1, quad_nodes)
    @test_nowarn Mantis.Forms.evaluate(zero_form_mixed_1_wedge, 1, quad_nodes)
    @test_nowarn Mantis.Forms.evaluate(zero_form_mixed_2_wedge, 1, quad_nodes)
    @test_nowarn Mantis.Forms.evaluate(zero_one_form_space_wedge, 1, quad_nodes)
    @test_nowarn Mantis.Forms.evaluate(zero_one_form_field_wedge, 1, quad_nodes)
    @test_nowarn Mantis.Forms.evaluate(zero_one_form_mixed_1_wedge, 1, quad_nodes)
    @test_nowarn Mantis.Forms.evaluate(zero_one_form_mixed_2_wedge, 1, quad_nodes)
    @test_nowarn Mantis.Forms.evaluate(one_form_space_wedge, 1, quad_nodes)
    @test_nowarn Mantis.Forms.evaluate(one_form_field_wedge, 1, quad_nodes)
    @test_nowarn Mantis.Forms.evaluate(one_form_mixed_1_wedge, 1, quad_nodes)
    @test_nowarn Mantis.Forms.evaluate(one_form_mixed_2_wedge, 1, quad_nodes)
    @test_nowarn Mantis.Forms.evaluate(one_zero_form_space_wedge, 1, quad_nodes)
    @test_nowarn Mantis.Forms.evaluate(one_zero_form_field_wedge, 1, quad_nodes)
    @test_nowarn Mantis.Forms.evaluate(one_zero_form_mixed_1_wedge, 1, quad_nodes)
    @test_nowarn Mantis.Forms.evaluate(one_zero_form_mixed_2_wedge, 1, quad_nodes)
    @test_nowarn Mantis.Forms.evaluate(zero_two_form_space_wedge, 1, quad_nodes)
    @test_nowarn Mantis.Forms.evaluate(zero_two_form_field_wedge, 1, quad_nodes)
    @test_nowarn Mantis.Forms.evaluate(zero_two_form_mixed_1_wedge, 1, quad_nodes)
    @test_nowarn Mantis.Forms.evaluate(zero_two_form_mixed_2_wedge, 1, quad_nodes)
    @test_nowarn Mantis.Forms.evaluate(two_zero_form_space_wedge, 1, quad_nodes)
    @test_nowarn Mantis.Forms.evaluate(two_zero_form_field_wedge, 1, quad_nodes)
    @test_nowarn Mantis.Forms.evaluate(two_zero_form_mixed_1_wedge, 1, quad_nodes)
    @test_nowarn Mantis.Forms.evaluate(two_zero_form_mixed_2_wedge, 1, quad_nodes)
    @test_nowarn Mantis.Forms.evaluate(one_two_form_space_wedge, 1, quad_nodes)
    @test_nowarn Mantis.Forms.evaluate(one_two_form_field_wedge, 1, quad_nodes)
    @test_nowarn Mantis.Forms.evaluate(one_two_form_mixed_1_wedge, 1, quad_nodes)
    @test_nowarn Mantis.Forms.evaluate(one_two_form_mixed_2_wedge, 1, quad_nodes)
    @test_nowarn Mantis.Forms.evaluate(two_one_form_space_wedge, 1, quad_nodes)
    @test_nowarn Mantis.Forms.evaluate(two_one_form_field_wedge, 1, quad_nodes)
    @test_nowarn Mantis.Forms.evaluate(two_one_form_mixed_1_wedge, 1, quad_nodes)
    @test_nowarn Mantis.Forms.evaluate(two_one_form_mixed_2_wedge, 1, quad_nodes)
    @test_nowarn Mantis.Forms.evaluate(zero_top_form_space_wedge, 1, quad_nodes)
    @test_nowarn Mantis.Forms.evaluate(zero_top_form_field_wedge, 1, quad_nodes)
    @test_nowarn Mantis.Forms.evaluate(zero_top_form_mixed_1_wedge, 1, quad_nodes)
    @test_nowarn Mantis.Forms.evaluate(zero_top_form_mixed_2_wedge, 1, quad_nodes)
    @test_nowarn Mantis.Forms.evaluate(top_zero_form_space_wedge, 1, quad_nodes)
    @test_nowarn Mantis.Forms.evaluate(top_zero_form_field_wedge, 1, quad_nodes)
    @test_nowarn Mantis.Forms.evaluate(top_zero_form_mixed_1_wedge, 1, quad_nodes)
    @test_nowarn Mantis.Forms.evaluate(top_zero_form_mixed_2_wedge, 1, quad_nodes)
end

############################################################################################
##                                       Variables setup                                  ##
############################################################################################
# Domains.
const starting_point_2d = (0.0, 0.0)
const box_size_2d = (1.0, 1.0)
const num_elements_2d = (3, 4)
const starting_point_3d = (0.0, 0.0, 0.0)
const box_size_3d = (1.0, 1.0, 1.0)
const num_elements_3d = (3, 4, 5)

# Polynomial degrees.
const degrees_2d = (2, 2)
const regularities_2d = (degrees_2d[1]-1, degrees_2d[2]-1)
const degrees_3d = (2, 2, 2)
const regularities_3d = (degrees_3d[1]-1, degrees_3d[2]-1, degrees_3d[3]-1)

# Crazy mesh.
const crazy_c = 0.2
function mapping(x::AbstractVector)
    x1_new = (2.0 / (Lright - Lleft)) * x[1] - 2.0 * Lleft / (Lright - Lleft) - 1.0
    x2_new = (2.0 / (Ltop - Lbottom)) * x[2] - 2.0 * Lbottom / (Ltop - Lbottom) - 1.0
    return [
        x[1] + ((Lright - Lleft) / 2.0) * crazy_c * sinpi(x1_new) * sinpi(x2_new),
        x[2] + ((Ltop - Lbottom) / 2.0) * crazy_c * sinpi(x1_new) * sinpi(x2_new),
    ]
end
function dmapping(x::AbstractVector)
    x1_new = (2.0 / (Lright - Lleft)) * x[1] - 2.0 * Lleft / (Lright - Lleft) - 1.0
    x2_new = (2.0 / (Ltop - Lbottom)) * x[2] - 2.0 * Lbottom / (Ltop - Lbottom) - 1.0
    return [
        1.0+pi * crazy_c * cospi(x1_new) * sinpi(x2_new) ((Lright - Lleft)/(Ltop - Lbottom))*pi*crazy_c*sinpi(x1_new)*cospi(x2_new)
        ((Ltop - Lbottom)/(Lright - Lleft))*pi*crazy_c*cospi(x1_new)*sinpi(x2_new) 1.0+pi * crazy_c * sinpi(x1_new) * cospi(x2_new)
    ]
end

# Expression for analytical form fields
function zero_form_expression(x::Matrix{Float64})
    return [ones(Float64, size(x))]
end

############################################################################################
##                                       2D TESTS                                         ##
############################################################################################
# Setup the complex 
complex_2d = Mantis.Forms.create_tensor_product_bspline_de_rham_complex(
    starting_point_2d, box_size_2d, num_elements_2d, degrees_2d, regularities_2d
)

# Then, the geometries.
# Cartesian geometry.
geom_cart_2d = Mantis.Geometry.create_cartesian_box(
    starting_point_2d, box_size_2d, num_elements_2d
)
# Crazy mesh geometry.
dimension = (2, 2)
crazy_mapping = Mantis.Geometry.Mapping(dimension, mapping, dmapping)
geom_crazy_2d = Mantis.Geometry.MappedGeometry(geom_cart_2d, crazy_mapping)

# The quadrature information.
q_rule_2d = Mantis.Quadrature.tensor_product_rule(
    degrees_2d .+ 1, Mantis.Quadrature.gauss_legendre
)

# Test the different geometries.
for geom in (geom_cart_2d, geom_crazy_2d)
    test_inner_prod_equality(geom, q_rule_2d, complex_2d)
    test_combinations_2d(complex_2d, q_rule_2d)
end

############################################################################################
##                                       3D TESTS                                         ##
############################################################################################
# Setup the complex 
complex_3d = Mantis.Forms.create_tensor_product_bspline_de_rham_complex(
    starting_point_3d, box_size_3d, num_elements_3d, degrees_3d, regularities_3d
)

# Then, the geometries.
# Cartesian geometry.
geom_cart_3d = Mantis.Geometry.create_cartesian_box(
    starting_point_3d, box_size_3d, num_elements_3d
)
# Crazy mesh geometry.
breakpoints3 = collect(LinRange(
    starting_point_3d[3], starting_point_3d[3]+box_size_3d[3], num_elements_3d[3]
))
line_geo_3 = Mantis.Geometry.CartesianGeometry((breakpoints3,))
geom_crazy_3d = Mantis.Geometry.TensorProductGeometry((geom_crazy_2d, line_geo_3))

# The quadrature information.
q_rule_3d = Mantis.Quadrature.tensor_product_rule(
    degrees_3d .+ 1, Mantis.Quadrature.gauss_legendre
)

# Test the different geometries.
for geom in (geom_cart_3d, geom_crazy_3d)
    test_inner_prod_equality(geom, q_rule_3d, complex_3d)
    test_combinations_3d(complex_3d, q_rule_3d)
end

end
