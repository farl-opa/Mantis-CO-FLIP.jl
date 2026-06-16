# Shared analysis helpers used by more than one CO-FLIP case.
#
# Safe to include standalone: it pulls in the CO-FLIP engine if it has not
# already been loaded, so a case file can `include` it after the engine
# without double-loading.

isdefined(Main, :SimulationConfig) || include(joinpath(@__DIR__, "..", "CO-FLIP_solver.jl"))

"""
L2 error of a CO-FLIP grid velocity field against the analytical decaying
Taylor–Green solution.

The exact Navier–Stokes solution on the 2π×2π box with U₀=1 and unit
wavenumbers is

    u_e(x,y,t) =  sin(x)·cos(y)·exp(-ν·(kx²+ky²)·t)
    v_e(x,y,t) = -cos(x)·sin(y)·exp(-ν·(kx²+ky²)·t)

with kx=ky=2π/L. The discrete R1 form is first L2-projected onto the
physical-velocity R1 space via `★`, matching the visualisation pipeline
used everywhere else in CO-FLIP. The error is then computed by
`Analysis.L2_norm(u_phys_form - u_exact, dΩ_err)` against an
`AnalyticalFormField` representing the physical Taylor–Green velocity,
using a finer Gauss–Legendre quadrature than the assembly `dΩ`
(`(p + nq_extra)` points per direction).

Returns `(l2_err, rel_l2_err, l2_norm_exact)`.
"""
function compute_decaying_tg_l2_error(
        domain::Domain, u_coeffs::AbstractVector{Float64},
        t::Float64, viscosity::Float64;
        nq_extra::Int=2,
    )
    Lx, Ly = domain.box_size
    kx = 2π / Lx
    ky = 2π / Ly
    decay = exp(-viscosity * (kx^2 + ky^2) * t)
    U0    = 1.0

    u_form      = Forms.build_form_field(domain.R1, u_coeffs)
    u_phys_expr = ★(u_form)
    u_phys_form = Assemblers.solve_L2_projection(domain.R1, u_phys_expr, domain.dΩ)

    expr = function (X::Matrix{Float64})
        xs = @view X[:, 1]
        ys = @view X[:, 2]
        c1 = @.  decay * U0 * sin(kx * xs) * cos(ky * ys)   # physical u
        c2 = @. -decay * U0 * cos(kx * xs) * sin(ky * ys)   # physical v
        return [c1, c2]
    end
    u_exact = Forms.AnalyticalFormField(1, expr, domain.geo, "u_TG")

    p1, p2 = domain.p
    nq_err = (p1 + nq_extra, p2 + nq_extra)
    dΩ_err = Quadrature.StandardQuadrature(
        Quadrature.tensor_product_rule(nq_err, Quadrature.gauss_legendre),
        prod(domain.nel),
    )

    l2_err     = Analysis.L2_norm(u_phys_form - u_exact, dΩ_err)
    l2_norm_ex = Analysis.L2_norm(u_exact, dΩ_err)
    rel_l2_err = l2_err / max(l2_norm_ex, eps())

    return (l2_err=l2_err, rel_l2_err=rel_l2_err, l2_norm_exact=l2_norm_ex)
end
