import Mantis

using Test

# Tests for a tensor product HierarchicalSplineSpace

ne1 = 5
ne2 = 5
breakpoints1 = collect(range(0,1,ne1+1))
patch1 = Mantis.Mesh.Patch1D(breakpoints1)
breakpoints2 = collect(range(0,1,ne2+1))
patch2 = Mantis.Mesh.Patch1D(breakpoints2)

deg1 = 2
deg2 = 2

CB1 = Mantis.FunctionSpaces.BSplineSpace(patch1, deg1, [-1; fill(deg1-1, ne1-1); -1])
CB2 = Mantis.FunctionSpaces.BSplineSpace(patch2, deg2, [-1; fill(deg2-1, ne2-1); -1])

nsub1 = 2
nsub2 = 2

TS1,FB1 = Mantis.FunctionSpaces.build_two_scale_operator(CB1, nsub1)
TS2, FB2 = Mantis.FunctionSpaces.build_two_scale_operator(CB2, nsub2)

CTP = Mantis.FunctionSpaces.TensorProductSpace((CB1, CB2))
FTP = Mantis.FunctionSpaces.TensorProductSpace((FB1, FB2))
spaces = [CTP, FTP]

CTP_num_els = Mantis.FunctionSpaces.get_num_elements(CTP)

CTS = Mantis.FunctionSpaces.TensorProductTwoScaleOperator(CTP, FTP, (TS1,TS2))

coarse_elements_to_refine = [3,4,5,8,9,10]
refined_elements = vcat(Mantis.FunctionSpaces.get_element_children.(Ref(CTS), coarse_elements_to_refine)...)

refined_domains = Mantis.FunctionSpaces.HierarchicalActiveInfo([collect(1:CTP_num_els),refined_elements])

###
hier_space = Mantis.FunctionSpaces.HierarchicalFiniteElementSpace(spaces, [CTS], refined_domains)

qrule = Mantis.Quadrature.tensor_product_rule((deg1+1, deg2+1), Mantis.Quadrature.gauss_legendre)
xi = Mantis.Quadrature.get_quadrature_nodes(qrule)

# Tests for coefficients and evaluation
for element_id in 1:1:Mantis.FunctionSpaces.get_num_elements(hier_space)

    # check extraction coefficients
    ex_coeffs, _ = Mantis.FunctionSpaces.get_extraction(hier_space, element_id)
    @test all(ex_coeffs .>= 0.0) # Test for non-negativity

    # check Hierarchical B-spline evaluation
    h_eval, _ = Mantis.FunctionSpaces.evaluate(hier_space, element_id, xi, 0)
    # Positivity of the basis
    @test minimum(h_eval[1][1][1]) >= 0.0
end