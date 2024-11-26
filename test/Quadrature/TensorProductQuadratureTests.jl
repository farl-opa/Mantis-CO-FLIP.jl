module TensorProductQuadratureTests

import Mantis

import LinearAlgebra

using Test


const Quad = Mantis.Quadrature


# Defines test functions and tolerances for quadrature rules.
include("HelperQuadratureTests.jl")


# The tensor product quadrature is a product of two or more 1D 
# quadrature rules. The 1D rules have their own tests, so only tests 
# for the tensor product quadrature are needed here.


const one_dim_quad_rules = [Quad.clenshaw_curtis, Quad.gauss_legendre,
                            Quad.gauss_lobatto, (Quad.newton_cotes, "closed"),
                            (Quad.newton_cotes, "open")]

function non_existing_quad_rule()
    return nothing
end

# Different degree but same rule ---------------------------------------

# Check that the tensor product rule reduces to the 1D rule in 1D. As 
# long as we don't define a specialised equality function for quadrature
# rules, we can only check that the rules are the same by manually 
# checking the nodes, weights and rule type.
for one_d_rule in one_dim_quad_rules
    for p in [2, 5, 8]

        deg = (p,)

        if typeof(one_d_rule) <: Tuple
            quad_rule_1d = one_d_rule[1](p, one_d_rule[2])
            quad_rule = Quad.tensor_product_rule(deg, one_d_rule...)
        else
            quad_rule_1d = one_d_rule(p)
            quad_rule = Quad.tensor_product_rule(deg, one_d_rule)
        end

        node_eq = Quad.get_quadrature_nodes(quad_rule) == Quad.get_quadrature_nodes(quad_rule_1d)
        weight_eq = Quad.get_quadrature_weights(quad_rule) == Quad.get_quadrature_weights(quad_rule_1d)
        type_eq = Quad.get_quadrature_rule_type(quad_rule) == Quad.get_quadrature_rule_type(quad_rule_1d)
        @test all([node_eq, weight_eq, type_eq])
    end
end



for one_d_rule in one_dim_quad_rules
    for dimension in [2, 3, 4]
        deg = (3, 5, 7, 8)[1:dimension]
        if typeof(one_d_rule) <: Tuple
            quad_rule = Quad.tensor_product_rule(deg, one_d_rule...)
            label = "Tensor-product of $dimension $(Quad.get_quadrature_rule_type(one_d_rule[1](deg[1], one_d_rule[2]))) rules"
        else
            quad_rule = Quad.tensor_product_rule(deg, one_d_rule)
            label = "Tensor-product of $dimension $(Quad.get_quadrature_rule_type(one_d_rule(deg[1]))) rules"
        end
        
        ξ = Quad.get_quadrature_nodes(quad_rule)
        w = Quad.get_quadrature_weights(quad_rule)


        # Check that the rule type is correct.
        @test Quad.get_quadrature_rule_type(quad_rule) == label


        # Constructor tests.
        @test typeof(quad_rule) == Quad.QuadratureRule{dimension}


        # No invalid degree tests here, as theses are tested in the
        # 1D quadrature tests. We can check what happens if a non-
        # existing rule is used.
        @test_throws MethodError Quad.tensor_product_rule(deg, non_existing_quad_rule)
        

        # Property tests.
        # Test that sum of weights is one.
        @test isapprox(sum(w), 1.0, atol=atol)
        # Test that all weights are positive if they should be. This is 
        # the case for all rules but Newton-Cotes.
        if typeof(one_d_rule) <: Quad.QuadratureRule
            @test all(w .> 0.0)
        end
        # Test that the weights are symmetric.
        @test isapprox(w, reverse(w), atol=atol)


        # Value tests on [0,1]^d.
        f = chebyshev_nd(deg.-1, ξ)
        I_num = LinearAlgebra.dot(w, f)
        I = integrated_chebyshev_nd(deg.-1)
        @test isapprox(I_num, I, atol=atol)
    end
end



# Different rules ------------------------------------------------------

# Only simple tests are performed here. This function relies on the 1D
# rules, and the computation of the tensor product of the weights is the
# same as the above.

# Check that the tensor product rule reduces to the 1D rule in 1D. As 
# long as we don't define a specialised equality function for quadrature
# rules, we can only check that the rules are the same by manually 
# checking the nodes, weights and rule type.
one_dim_rules = (Quad.clenshaw_curtis(3), Quad.gauss_legendre(5),
                 Quad.gauss_lobatto(4), Quad.newton_cotes(2, "closed"),
                 Quad.newton_cotes(3, "open"))

for one_d_rule in one_dim_rules
    quad_rule = Quad.tensor_product_rule((one_d_rule,))

    node_eq = Quad.get_quadrature_nodes(quad_rule) == Quad.get_quadrature_nodes(one_d_rule)
    weight_eq = Quad.get_quadrature_weights(quad_rule) == Quad.get_quadrature_weights(one_d_rule)
    type_eq = Quad.get_quadrature_rule_type(quad_rule) == Quad.get_quadrature_rule_type(one_d_rule)
    @test all([node_eq, weight_eq, type_eq])
end

# Test the tensor product of different rules.
quad_rule = Quad.tensor_product_rule(one_dim_rules)
label = "Tensor-product of (Clenshaw-Curtis, Gauss-Legendre, Gauss-Lobatto, Newton-Cotes (closed), Newton-Cotes (open)) rules"

w = Quad.get_quadrature_weights(quad_rule)

# Check that the rule type is correct.
@test Quad.get_quadrature_rule_type(quad_rule) == label

# Constructor tests.
@test typeof(quad_rule) == Quad.QuadratureRule{5}

# Property tests.
# Test that sum of weights is one.
@test isapprox(sum(w), 1.0, atol=atol)
# Test that the weights are symmetric.
@test isapprox(w, reverse(w), atol=atol)



end