module WeakFormsTests

using Mantis

function sinusoidal_solution(geo::Mantis.Geometry.AbstractGeometry)
    ω = 2.0 * pi
    function my_sol(x::Matrix{Float64})
        # [u] = [sin(ωx¹)sin(ωx²)...sin(ωxⁿ)]
        y = @. sin(ω * x)
        return [vec(prod(y, dims=2))]
    end

    function grad_my_sol(x::Matrix{Float64})
        # [u₁, u₂, ...] = [ω*cos(ωx¹)sin(ωx²)...sin(ωxⁿ), ω*sin(ωx¹)cos(ωx²)...sin(ωxⁿ), ...]
        y = sin.(ω .* x)
        z = ω .* cos.(ω .* x)
        w = Vector{Vector{Float64}}(undef, size(x, 2))
        for i ∈ 1:size(x,2)
            w[i] = z[:,i] .* prod(y[:,setdiff(1:size(x,2), i)], dims=2)[:,1]
        end
        return w
    end

    function flux_my_sol(x::Matrix{Float64})
        # [u₁, u₂, ...] = ⋆[ω*cos(ωx¹)sin(ωx²)...sin(ωxⁿ), ω*sin(ωx¹)cos(ωx²)...sin(ωxⁿ), ...]
        w = grad_my_sol(x)
        if size(x,2) == 1
            # (a) -> (a)
            return [w[1]]

        elseif size(x,2) == 2
            # (a, b) -> (b, -a)
            w = [w[2], -w[1]]
            return w

        elseif size(x,2) == 3
            # (a, b, c) -> (a, b, c)
            return [w[1], w[2], w[3]]

        else
            throw(ArgumentError("Dimension not supported."))
        end
    end

    function laplace_my_sol(x::Matrix{Float64})
        # [-(u₁₁+u₂₂+...uₙₙ)] = [2ω²*sin(ωx¹)sin(ωx²)...sin(ωxⁿ)]
        y = prod(sin.(ω * x), dims=2)
        y = @. 2 * ω * ω * y
        return [vec(y)]
    end

    ϕ² = Mantis.Forms.AnalyticalFormField(2, my_sol, geo, "ϕ")
    δϕ² = Mantis.Forms.AnalyticalFormField(1, flux_my_sol, geo, "δϕ")
    f² = Mantis.Forms.AnalyticalFormField(2, laplace_my_sol, geo, "f")

    return ϕ², δϕ², f²
end

function n_form_mixed(inputs::Assemblers.WeakFormInputs)
    uⁿ⁻¹, vⁿ = Assemblers.get_trial_forms(inputs)
    εⁿ⁻¹, φⁿ = Assemblers.get_test_forms(inputs)
    fⁿ = Assemblers.get_forcing(inputs)
    A11 = ∫(uⁿ⁻¹ ∧ ★(εⁿ⁻¹))
    A12 = -∫((d(uⁿ⁻¹) ∧ ★(φⁿ)))
    A21 = ∫(vⁿ ∧ ★(d(εⁿ⁻¹)))
    b21 = ∫(vⁿ ∧ ★(fⁿ))
    lhs_expressions = ((A11, A12), (A21, 0))
    rhs_expressions = ((0,), (b21,))

    return lhs_expressions, rhs_expressions 
end

function L²_projection(inputs::Assemblers.WeakFormInputs)
    v⁰ = Assemblers.get_trial_form(inputs)
    u⁰ = Assemblers.get_test_form(inputs)
    f⁰ = Assemblers.get_forcing(inputs)
    A = ∫(v⁰ ∧ ★(u⁰))
    b = ∫(v⁰ ∧ ★(f⁰))
    lhs_expressions = ((A,),)
    rhs_expressions = ((b,),)

    return lhs_expressions, rhs_expressions
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
# f_expr(x) = @. [x[:, 1] * (1 - x[:, 1])]
# f² = Forms.AnalyticalFormField(2, f_expr, Forms.get_geometry(φ²), "f²")

ϕ², δϕ², f² = sinusoidal_solution(Forms.get_geometry(φ²))

wfi = Assemblers.WeakFormInputs((ε¹, φ²), (f²,))
# wfi = Assemblers.WeakFormInputs((σ⁰,), (f⁰,))
wf = Assemblers.WeakForm(wfi, n_form_mixed)

A, b = Assemblers.assemble(wf, q_rule)
coeffs = vec(A \ b)
α¹, β² = Forms.build_form_fields((ε¹, φ²), coeffs)
error = Analysis.L2_norm(β² - ϕ², q_rule)
println("Error: ", error)

end
