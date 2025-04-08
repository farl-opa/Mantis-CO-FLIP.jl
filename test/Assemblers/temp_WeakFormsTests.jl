module WeakFormsTests

using Mantis

function n_form_mixed(inputs::Assemblers.WeakFormInputs)
    εⁿ⁻¹, φⁿ = Assemblers.get_trial_forms(inputs)
    uⁿ⁻¹, vⁿ = Assemblers.get_test_forms(inputs)
    f = Assemblers.get_forcing(inputs)[1]

    A11 = uⁿ⁻¹ ∧ ★(εⁿ⁻¹)
    A12 = d(uⁿ⁻¹) ∧ ★(φⁿ)
    A21 = vⁿ ∧ ★(d(εⁿ⁻¹))
    b21 = vⁿ ∧ ★(f)

    lhs_expressions = (A11, A12, A21)
    rhs_expressions = (b21,)

    lhs_row_offsets = (0, 0, Forms.get_num_basis(uⁿ⁻¹))
    lhs_col_offsets = (0, Forms.get_num_basis(εⁿ⁻¹), 0)
    lhs_offsets = (lhs_row_offsets, lhs_col_offsets)

    rhs_row_offsets = (Forms.get_num_basis(uⁿ⁻¹),)
    rhs_col_offsets = (0,)
    rhs_offsets = (rhs_row_offsets, rhs_col_offsets)

    return lhs_expressions, rhs_expressions, lhs_offsets, rhs_offsets
end

starting_point = (0.0, 0.0)
box_size = (1.0, 1.0)
num_elements = (5, 5)
degrees = (2, 2)
regularities = (1, 1)
q_rule = Quadrature.tensor_product_rule(degrees .+ 1, Quadrature.gauss_legendre)

σ⁰, ε¹, φ² = Forms.create_tensor_product_bspline_de_rham_complex(
    starting_point, box_size, num_elements, degrees, regularities
)
f_expr(x) = @. [x[:, 1] * (1 - x[:, 1])]
f² = Forms.AnalyticalFormField(2, f_expr, Forms.get_geometry(φ²), "f²")

wfi = Assemblers.WeakFormInputs((ε¹, φ²), (f²,))
wf = Assemblers.WeakForm(wfi, n_form_mixed)
println(Assemblers.get_problem_size(wf))

end
