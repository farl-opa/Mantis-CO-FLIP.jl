module TensorProductQuadratureTests

import Mantis

import LinearAlgebra

using Test


const Quad = Mantis.Quadrature


# Defines test functions and tolerances for quadrature rules.
include("QuadratureTestsSetup.jl")


# The tensor product quadrature is a product of two or more 1D quadrature rules. The 1D
# rules have their own tests, so only tests for the tensor product quadrature are needed.


const one_dim_quad_rules = [
    Quad.clenshaw_curtis,
    Quad.gauss_legendre,
    Quad.gauss_lobatto,
    (Quad.newton_cotes, "closed"),
    (Quad.newton_cotes, "open"),
]

non_existing_quad_rule() = nothing


# Different degree but same rule -----------------------------------------------------------

# Check that the tensor product rule reduces to the 1D rule in 1D. As long as we don't
# define a specialised equality function for quadrature rules, we can only check that the
# rules are the same by manually checking the nodes, weights and rule type.
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

        quad_rule_info = (
            Quad.get_quadrature_nodes(quad_rule),
            Quad.get_quadrature_weights(quad_rule),
            Quad.get_quadrature_rule_label(quad_rule),
        )

        one_d_rule_info = (
            Quad.get_quadrature_nodes(quad_rule_1d),
            Quad.get_quadrature_weights(quad_rule_1d),
            Quad.get_quadrature_rule_label(quad_rule_1d),
        )

        @test all(quad_rule_info .== one_d_rule_info)
    end
end



for one_d_rule in one_dim_quad_rules
    for dimension in [2, 3, 4]
        deg = (3, 5, 7, 8)[1:dimension]
        if typeof(one_d_rule) <: Tuple
            quad_rule = Quad.tensor_product_rule(deg, one_d_rule...)
            label_1d = Quad.get_quadrature_rule_label(one_d_rule[1](deg[1], one_d_rule[2]))
            label = """Tensor-product of $dimension $(label_1d) rules"""
        else
            quad_rule = Quad.tensor_product_rule(deg, one_d_rule)
            label_1d = Quad.get_quadrature_rule_label(one_d_rule(deg[1]))
            label = "Tensor-product of $dimension $(label_1d) rules"
        end

        ξ = Quad.get_quadrature_nodes(quad_rule)
        w = Quad.get_quadrature_weights(quad_rule)


        # Check that the rule type is correct.
        @test Quad.get_quadrature_rule_label(quad_rule) == label


        # Constructor tests.
        @test typeof(quad_rule) == Quad.QuadratureRule{dimension}


        # No invalid degree tests here, as theses are tested in the 1D quadrature tests. We
        # can check what happens if a non-existing rule is used.
        @test_throws MethodError Quad.tensor_product_rule(deg, non_existing_quad_rule)


        # Property tests.
        @test isapprox(sum(w), 1.0, atol=atol)
        # Test that all weights are positive if they should be. This is the case for all
        # rules but Newton-Cotes. Note that the Newton-Cotes rules in the list are a tuple,
        # while the others are functions.
        if typeof(one_d_rule) <: Function
            @test all(w .> 0.0)
        end
        @test isapprox(w, reverse(w), atol=atol)


        # Value tests on [0,1]^d.
        f = chebyshev_nd(deg.-1, ξ)
        I_num = LinearAlgebra.dot(w, f)
        I = integrated_chebyshev_nd(deg.-1)
        @test isapprox(I_num, I, atol=atol)
    end
end



# Different rules --------------------------------------------------------------------------

# Only simple tests are performed here. This function relies on the 1D rules, and the
# computation of the tensor product of the weights is the same as the above.

# Check that the tensor product rule reduces to the 1D rule in 1D. As long as we don't
# define a specialised equality function for quadrature rules, we can only check that the
# rules are the same by manually checking the nodes, weights and rule type.
one_dim_rules = (
    Quad.clenshaw_curtis(3),
    Quad.gauss_legendre(5),
    Quad.gauss_lobatto(4),
    Quad.newton_cotes(2, "closed"),
    Quad.newton_cotes(3, "open")
)

for one_d_rule in one_dim_rules
    quad_rule = Quad.tensor_product_rule((one_d_rule,))

    quad_rule_info = (
        Quad.get_quadrature_nodes(quad_rule),
        Quad.get_quadrature_weights(quad_rule),
        Quad.get_quadrature_rule_label(quad_rule),
    )

    one_d_rule_info = (
        Quad.get_quadrature_nodes(one_d_rule),
        Quad.get_quadrature_weights(one_d_rule),
        Quad.get_quadrature_rule_label(one_d_rule),
    )

    @test all(quad_rule_info .== one_d_rule_info)
end

# Test the tensor product of different rules.
quad_rule = Quad.tensor_product_rule(one_dim_rules)
label = """\
    Tensor-product of (Clenshaw-Curtis, Gauss-Legendre, Gauss-Lobatto, Newton-Cotes \
    (closed), Newton-Cotes (open)) rules\
    """

w = Quad.get_quadrature_weights(quad_rule)

# Check that the rule label is correct.
@test Quad.get_quadrature_rule_label(quad_rule) == label

# Constructor tests.
@test typeof(quad_rule) == Quad.QuadratureRule{5}

# Property tests.
@test isapprox(sum(w), 1.0, atol=atol)
@test isapprox(w, reverse(w), atol=atol)

end
