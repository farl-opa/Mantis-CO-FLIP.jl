module TensorProductTests

"""
Tests for tensor-product spline spaces.
"""

import Mantis

using Test
using InteractiveUtils

###
### Basic tensor-product tests
###

# patch breakpoints in x and y
breakpoints1 = [0.0, 0.5, 0.8, 0.9, 1.0]
patch1 = Mantis.Mesh.Patch1D(breakpoints1)
breakpoints2 = [0.0, 0.5, 0.6, 1.0]
patch2 = Mantis.Mesh.Patch1D(breakpoints2)
num_derivatives = 3
for deg1 in 0:5
    for deg2 in 0:5
        # first B-spline patch
        local B1 = Mantis.FunctionSpaces.BSplineSpace(patch1, deg1, [-1, deg1-1, min(deg1-1, 0), deg1-1, -1])
        # second B-spline patch
        local B2 = Mantis.FunctionSpaces.BSplineSpace(patch2, deg2, [-1, min(deg2-1,1),  deg2-1, -1])
        # tensor-product B-spline patch
        TP = Mantis.FunctionSpaces.TensorProductSpace((B1,B2))
        TP1 = Mantis.FunctionSpaces.TensorProductSpace((Mantis.FunctionSpaces.TensorProductSpace((B1, B2)), B1))
        TP2 = Mantis.FunctionSpaces.TensorProductSpace((B1, Mantis.FunctionSpaces.TensorProductSpace((B2, B1))))
        TP3 = Mantis.FunctionSpaces.TensorProductSpace((B1,B2,B1))
        # evaluation points
        qrule = Mantis.Quadrature.tensor_product_rule((deg1+1, deg2+1), Mantis.Quadrature.gauss_legendre)
        qrule3 = Mantis.Quadrature.tensor_product_rule((deg1+1, deg2+1, deg1+1), Mantis.Quadrature.gauss_legendre)
        x_all = Mantis.Quadrature.get_nodes(qrule)
        x_all3 = Mantis.Quadrature.get_nodes(qrule3)
        for el in 1:1:Mantis.FunctionSpaces.get_num_elements(TP)
            # check B-spline evaluation
            TP_eval, _ = Mantis.FunctionSpaces.evaluate(TP, el, x_all, 0)
            TP1_eval, _ = Mantis.FunctionSpaces.evaluate(TP1, el, x_all3, num_derivatives)
            TP2_eval, _ = Mantis.FunctionSpaces.evaluate(TP1, el, x_all3, num_derivatives)
            TP3_eval, _ = Mantis.FunctionSpaces.evaluate(TP3, el, x_all3, num_derivatives)
            # Positivity of the polynomials
            @test minimum(TP_eval[1][1][1]) >= 0.0
            @test minimum(TP1_eval[1][1][1]) >= 0.0
            @test minimum(TP2_eval[1][1][1]) >= 0.0
            @test minimum(TP3_eval[1][1][1]) >= 0.0

            # Partition of unity
            @test all(isapprox.(sum(TP_eval[1][1][1], dims=2), 1.0))
            @test all(isapprox.(sum(TP1_eval[1][1][1], dims=2), 1.0))
            @test all(isapprox.(sum(TP2_eval[1][1][1], dims=2), 1.0))
            @test all(isapprox.(sum(TP3_eval[1][1][1], dims=2), 1.0))

            # Consistency of the evaluation
            for der ∈ 1:num_derivatives
                for der_idx ∈ eachindex(TP1_eval[der])
                    @test isapprox(TP1_eval[der][der_idx], TP2_eval[der][der_idx])
                    @test isapprox(TP2_eval[der][der_idx], TP3_eval[der][der_idx])
                end
            end
        end

        # tests for number of basis functions
        @test Mantis.FunctionSpaces.get_num_basis(TP1) == Mantis.FunctionSpaces.get_num_basis(TP2)
        @test Mantis.FunctionSpaces.get_num_basis(TP2) == Mantis.FunctionSpaces.get_num_basis(TP3)

        # tests for dof partitioning
        @test Mantis.FunctionSpaces.get_dof_partition(TP1) == Mantis.FunctionSpaces.get_dof_partition(TP2)
        @test Mantis.FunctionSpaces.get_dof_partition(TP2) == Mantis.FunctionSpaces.get_dof_partition(TP3)
        @test sum(map(length, Mantis.FunctionSpaces.get_dof_partition(TP3)[1])) == prod(map(Mantis.FunctionSpaces.get_num_basis, TP3.fem_spaces))
    end
end

###
### Combination of tensor-product and multi-patch tests
###

# first B-spline patch
breakpoints1 = [0.0, 0.5, 1.0]
patch1 = Mantis.Mesh.Patch1D(breakpoints1)
breakpoints2 = [0.0, 0.5, 0.6, 1.0]
patch2 = Mantis.Mesh.Patch1D(breakpoints2)
deg1 = 3;
B1_tensor_prod = Mantis.FunctionSpaces.BSplineSpace(patch1, deg1, [-1, deg1-1, -1])
# second B-spline patch
deg2 = 4;
B2_tensor_prod = Mantis.FunctionSpaces.BSplineSpace(patch2, deg2, [-1, min(deg2-1,1),  deg2-1, -1])
# first multi-patch object
MP1 = Mantis.FunctionSpaces.GTBSplineSpace((B1_tensor_prod, B2_tensor_prod), [1, -1])

# third B-spline patch
deg1 = 4;
B3_tensor_prod = Mantis.FunctionSpaces.BSplineSpace(patch1, deg1, [-1, deg1-1, -1])
# second B-spline patch
deg2 = 3;
B4_tensor_prod = Mantis.FunctionSpaces.BSplineSpace(patch2, deg2, [-1, min(deg2-1,1),  deg2-1, -1])
# First multi-patch object
MP2 = Mantis.FunctionSpaces.GTBSplineSpace((B3_tensor_prod, B4_tensor_prod), [1, -1])

# tensor-product B-spline patch
TP = Mantis.FunctionSpaces.TensorProductSpace((MP1, MP2))
# evaluation points
x1 = collect(LinRange(0.0,1.0,11))
x2 = collect(LinRange(0.0,1.0,11))
for el in 1:1:Mantis.FunctionSpaces.get_num_elements(TP)
    # check B-spline evaluation
    TP_eval, _ = Mantis.FunctionSpaces.evaluate(TP, el, (x1,x2))
    # Positivity of the polynomials
    @test minimum(TP_eval[1][1][1]) >= 0.0

    # Partition of unity
    @test all(isapprox.(sum(TP_eval[1][1][1], dims=2), 1.0))
end

end
