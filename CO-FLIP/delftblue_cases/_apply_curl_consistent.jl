#!/usr/bin/env julia
# Adds the curl-consistent (FEEC Poisson) pressure-kick path to each delftblue case file.
# Run from anywhere: `julia CO-FLIP/delftblue_cases/_apply_curl_consistent.jl`
# Idempotent: re-runs are no-ops once all patches are applied.

const CASE_DIR = @__DIR__

const CASE_FILES = [
    "CO-FLIP_periodic_taylor_green.jl",
    "CO-FLIP_periodic_dipole.jl",
    "CO-FLIP_periodic_four_vortex.jl",
    "CO-FLIP_periodic_lamb_orseen.jl",
    "CO-FLIP_periodic_leapfrog.jl",
    "CO-FLIP_periodic_merging.jl",
    "CO-FLIP_periodic_shear.jl",
    "CO-FLIP_periodic_stuart.jl",
    "CO-FLIP_periodic_convecting.jl",
]

const EDITS = Tuple{String,String}[]

# ─── Edit 1: Domain struct gains K_R0/K_R0_fact/G_R0_R1/Mass1_fact ─────────────
push!(EDITS, (
"""
struct Domain{F0, F1, F2, G, Q, MH, NH, MM, HF}
    R0::F0
    R1::F1
    R2::F2
    geo::G
    dΩ::Q
    nel::NTuple{2,Int}
    p::NTuple{2,Int}
    k::NTuple{2,Int}
    LHS_Hodge::MH
    LHS_Hodge_fact::HF
    N_Hodge::NH
    Mass_matrix::MM
    eval_cache_size::Int
    box_size::NTuple{2,Float64}
    R1_basis_indices::Vector{Vector{Int}}
end
""",
"""
struct Domain{F0, F1, F2, G, Q, MH, NH, MM, HF, KR0, KR0F, GG, M1F}
    R0::F0
    R1::F1
    R2::F2
    geo::G
    dΩ::Q
    nel::NTuple{2,Int}
    p::NTuple{2,Int}
    k::NTuple{2,Int}
    LHS_Hodge::MH
    LHS_Hodge_fact::HF
    N_Hodge::NH
    Mass_matrix::MM
    K_R0::KR0
    K_R0_fact::KR0F
    G_R0_R1::GG
    Mass1_fact::M1F
    eval_cache_size::Int
    box_size::NTuple{2,Float64}
    R1_basis_indices::Vector{Vector{Int}}
end
"""
))

# ─── Edit 2: GenerateDomain — assemble K_R0, G_R0_R1, factorize M and K_R0 ─────
push!(EDITS, (
"""
    A, N = assemble_hodge_laplacian_matrices(R[2], R[3], dΩ)
    A_fact = lu(A)
    M = assemble_1form_mass_matrix(R[2], dΩ)

    num_elements_R1 = prod(nel)
    R1_basis_indices = Vector{Vector{Int}}(undef, num_elements_R1)
    for eid in 1:num_elements_R1
        R1_basis_indices[eid] = collect(FunctionSpaces.get_basis_indices(R[2].fem_space, eid))
    end
    eval_cache_size = maximum(length(bi) for bi in R1_basis_indices)

    return Domain(R[1], R[2], R[3], Forms.get_geometry(R[2]), dΩ, nel, p, k, A, A_fact, N, M, eval_cache_size, box_size, R1_basis_indices)
end
""",
"""
    A, N = assemble_hodge_laplacian_matrices(R[2], R[3], dΩ)
    A_fact = lu(A)
    M = assemble_1form_mass_matrix(R[2], dΩ)
    M_fact = lu(M)

    K_R0 = assemble_R0_stiffness_matrix(R[1], dΩ)
    G_R0_R1 = assemble_R0_R1_weak_grad_matrix(R[1], R[2], dΩ)

    n0 = size(K_R0, 1)
    ridge_R0 = 1e-10 * (sum(abs, diag(K_R0)) / max(n0, 1) + 1.0)
    K_R0_reg = K_R0 + ridge_R0 * sparse(I, n0, n0)
    K_R0_fact = lu(K_R0_reg)

    num_elements_R1 = prod(nel)
    R1_basis_indices = Vector{Vector{Int}}(undef, num_elements_R1)
    for eid in 1:num_elements_R1
        R1_basis_indices[eid] = collect(FunctionSpaces.get_basis_indices(R[2].fem_space, eid))
    end
    eval_cache_size = maximum(length(bi) for bi in R1_basis_indices)

    return Domain(R[1], R[2], R[3], Forms.get_geometry(R[2]), dΩ, nel, p, k,
                  A, A_fact, N, M, K_R0, K_R0_fact, G_R0_R1, M_fact,
                  eval_cache_size, box_size, R1_basis_indices)
end
"""
))

# ─── Edit 3: Add R0 stiffness + R0-R1 weak gradient assemblers after Mass1 ─────
push!(EDITS, (
"""
function assemble_1form_mass_matrix(R1_space::F, dΩ::Q) where {F, Q}
    weak_form_inputs = Assemblers.WeakFormInputs(R1_space)
    vᵏ = Assemblers.get_test_form(weak_form_inputs)
    uᵏ = Assemblers.get_trial_form(weak_form_inputs)

    A = ∫(vᵏ ∧ ★(uᵏ), dΩ)

    lhs_expressions = ((A,),)
    rhs_expressions = ((0,),)

    weak_form = Assemblers.WeakForm(lhs_expressions, rhs_expressions, weak_form_inputs)
    M, _ = Assemblers.assemble(weak_form; rhs_type=SparseArrays.SparseMatrixCSC{Float64, Int})
    return M
end
""",
"""
function assemble_1form_mass_matrix(R1_space::F, dΩ::Q) where {F, Q}
    weak_form_inputs = Assemblers.WeakFormInputs(R1_space)
    vᵏ = Assemblers.get_test_form(weak_form_inputs)
    uᵏ = Assemblers.get_trial_form(weak_form_inputs)

    A = ∫(vᵏ ∧ ★(uᵏ), dΩ)

    lhs_expressions = ((A,),)
    rhs_expressions = ((0,),)

    weak_form = Assemblers.WeakForm(lhs_expressions, rhs_expressions, weak_form_inputs)
    M, _ = Assemblers.assemble(weak_form; rhs_type=SparseArrays.SparseMatrixCSC{Float64, Int})
    return M
end

\"\"\"Assemble scalar Laplacian (R0 stiffness): K[i,j] = ∫(d(ψᵢ) ∧ ★ d(ψⱼ)).\"\"\"
function assemble_R0_stiffness_matrix(R0_space::F, dΩ::Q) where {F, Q}
    weak_form_inputs = Assemblers.WeakFormInputs(R0_space)
    ε⁰ = Assemblers.get_test_form(weak_form_inputs)
    u⁰ = Assemblers.get_trial_form(weak_form_inputs)

    A = ∫(d(ε⁰) ∧ ★(d(u⁰)), dΩ)

    lhs_expressions = ((A,),)
    rhs_expressions = ((0,),)

    weak_form = Assemblers.WeakForm(lhs_expressions, rhs_expressions, weak_form_inputs)
    K, _ = Assemblers.assemble(weak_form; rhs_type=SparseArrays.SparseMatrixCSC{Float64, Int})
    return K
end

\"\"\"
Assemble rectangular weak-gradient block G[i,j] = ∫(d(ψᵢ⁰) ∧ ★ φⱼ¹) coupling R0 (rows)
with R1 (cols). Provides the weak-divergence operator τ̃ ↦ G·τ̃ ∈ R0 and, via its
transpose, satisfies M₁·d₀ = Gᵀ for the FEEC complex.
\"\"\"
function assemble_R0_R1_weak_grad_matrix(R0_space::F0, R1_space::F1, dΩ::Q) where {F0, F1, Q}
    weak_form_inputs = Assemblers.WeakFormInputs((R0_space, R1_space))
    ε⁰, _ε¹ = Assemblers.get_test_forms(weak_form_inputs)
    _u⁰, u¹ = Assemblers.get_trial_forms(weak_form_inputs)

    A_12 = ∫(d(ε⁰) ∧ ★(u¹), dΩ)
    lhs_expressions = ((0, A_12), (0, 0))
    rhs_expressions = ((0, 0), (0, 0))

    weak_form = Assemblers.WeakForm(lhs_expressions, rhs_expressions, weak_form_inputs)
    sys, _ = Assemblers.assemble(weak_form; rhs_type=SparseArrays.SparseMatrixCSC{Float64, Int})

    n0 = FunctionSpaces.get_num_basis(R0_space.fem_space)
    n1 = FunctionSpaces.get_num_basis(R1_space.fem_space)
    return sys[1:n0, (n0 + 1):(n0 + n1)]
end
"""
))

# ─── Edit 4: Add apply_pressure_correction_curl! after div-consistent variant ──
push!(EDITS, (
"""
\"\"\"Apply FLIP-style pressure gradient correction to particles.\"\"\"
function apply_pressure_correction!(
        p::Particles,
        f_projected::AbstractVector{Tp},
        f_pre_projection::AbstractVector{Tq},
        d::Domain,
        thread_caches::Vector{EvaluationCache};
        delta_scale::Float64=1.0,
    ) where {Tp<:Real, Tq<:Real}
    tau_coeffs = compute_pressure_delta(f_projected, f_pre_projection)

    if isempty(thread_caches)
        throw(ArgumentError("thread_caches must not be empty"))
    end
    n_thread_slots = Threads.maxthreadid()
    if length(thread_caches) < n_thread_slots
        throw(ArgumentError("thread_caches length must be at least Threads.maxthreadid()"))
    end

    num_p = length(p.x)
    Threads.@threads :static for i in 1:num_p
        cache = thread_caches[Threads.threadid()]
        @inbounds begin
            val, _ = probe_field_at_point(p.x[i], p.y[i], tau_coeffs, d, cache)
            p.my[i] += delta_scale * val[1]
            p.mx[i] -= delta_scale * val[2]
        end
    end
end
""",
"""
\"\"\"Apply FLIP-style pressure gradient correction to particles.\"\"\"
function apply_pressure_correction!(
        p::Particles,
        f_projected::AbstractVector{Tp},
        f_pre_projection::AbstractVector{Tq},
        d::Domain,
        thread_caches::Vector{EvaluationCache};
        delta_scale::Float64=1.0,
    ) where {Tp<:Real, Tq<:Real}
    tau_coeffs = compute_pressure_delta(f_projected, f_pre_projection)

    if isempty(thread_caches)
        throw(ArgumentError("thread_caches must not be empty"))
    end
    n_thread_slots = Threads.maxthreadid()
    if length(thread_caches) < n_thread_slots
        throw(ArgumentError("thread_caches length must be at least Threads.maxthreadid()"))
    end

    num_p = length(p.x)
    Threads.@threads :static for i in 1:num_p
        cache = thread_caches[Threads.threadid()]
        @inbounds begin
            val, _ = probe_field_at_point(p.x[i], p.y[i], tau_coeffs, d, cache)
            p.my[i] += delta_scale * val[1]
            p.mx[i] -= delta_scale * val[2]
        end
    end
end

\"\"\"
Curl-consistent pressure-force kick (Section 6.2.2 of the CO-FLIP paper).

Given τ = f_projected − f_pre_projection on the grid (stored in R1 with flux-form
convention), this routine:
  1. interpolates τ to the particles as a flux form (same read-out as the
     div-consistent path), producing per-particle physical force vectors;
  2. inverse-interpolates those particle force vectors back to a discrete 1-form τ̃
     ∈ R1 (1-form storage) via the LSQR P2G operator B;
  3. solves the weak Poisson reconstruction
         K_R0 · p̃ = G_R0_R1 · τ̃
     (regularised with a tiny diagonal ridge and mean-pinned) to find the closest
     exact 1-form;
  4. computes d₀p̃ in R1 via the FEEC identity M₁ · d₀ = (G_R0_R1)ᵀ and probes the
     result at every particle, adding the curl-free physical gradient to (mx, my).
\"\"\"
function apply_pressure_correction_curl!(
        p::Particles,
        f_projected::AbstractVector{Tp},
        f_pre_projection::AbstractVector{Tq},
        d::Domain,
        buf::SimulationBuffers,
        thread_caches::Vector{EvaluationCache};
        delta_scale::Float64=1.0,
        atol::Float64=1e-9,
        btol::Float64=1e-9,
        maxiter::Int=2000,
        error_on_nonconvergence::Bool=true,
    ) where {Tp<:Real, Tq<:Real}
    if isempty(thread_caches)
        throw(ArgumentError("thread_caches must not be empty"))
    end
    n_thread_slots = Threads.maxthreadid()
    if length(thread_caches) < n_thread_slots
        throw(ArgumentError("thread_caches length must be at least Threads.maxthreadid()"))
    end

    tau_coeffs = compute_pressure_delta(f_projected, f_pre_projection)
    num_p = length(p.x)

    Fx = Vector{Float64}(undef, num_p)
    Fy = Vector{Float64}(undef, num_p)

    Threads.@threads :static for i in 1:num_p
        cache = thread_caches[Threads.threadid()]
        @inbounds begin
            val, _ = probe_field_at_point(p.x[i], p.y[i], tau_coeffs, d, cache)
            Fx[i] = -val[2]
            Fy[i] =  val[1]
        end
    end

    B = build_B_matrix(p, d, buf)

    V_p = @view buf.V_p[1:(2 * num_p)]
    @inbounds for i in 1:num_p
        w = sqrt(p.volume[i])
        V_p[2i - 1] = Fx[i] * w
        V_p[2i]     = Fy[i] * w
    end

    tilde_tau = zeros(Float64, size(B, 2))
    _, ch = IterativeSolvers.lsqr!(tilde_tau, B, V_p; atol=atol, btol=btol,
                                   maxiter=maxiter, log=true)
    if !ch.isconverged && error_on_nonconvergence
        @error "Curl-consistent P2G LSQR did not converge" iters=ch.iters atol=atol btol=btol maxiter=maxiter
    end

    rhs_R0 = d.G_R0_R1 * tilde_tau
    rhs_mean = sum(rhs_R0) / length(rhs_R0)
    @. rhs_R0 -= rhs_mean

    p_tilde = d.K_R0_fact \\ rhs_R0
    p_mean = sum(p_tilde) / length(p_tilde)
    @. p_tilde -= p_mean

    rhs1 = transpose(d.G_R0_R1) * p_tilde
    g_R1 = d.Mass1_fact \\ rhs1

    Threads.@threads :static for i in 1:num_p
        cache = thread_caches[Threads.threadid()]
        @inbounds begin
            val, _ = probe_field_at_point(p.x[i], p.y[i], g_R1, d, cache)
            p.mx[i] += delta_scale * val[1]
            p.my[i] += delta_scale * val[2]
        end
    end
    return nothing
end
"""
))

# ─── Edit 5: SimulationConfig gains pressure_kick_method ───────────────────────
push!(EDITS, (
"""
    enable_pressure_kick::Bool        = true
""",
"""
    enable_pressure_kick::Bool        = true
    pressure_kick_method::Symbol      = :curl_consistent
"""
))

# ─── Edit 6: step_co_flip! pressure-kick dispatch ──────────────────────────────
push!(EDITS, (
"""
    if cfg.enable_pressure_kick
        t0 = time()
        apply_pressure_correction!(
            p, f_np1_proj, f_np1, d, thread_caches; delta_scale=1.0,
        )
        println("  [pressure kick enabled — FLIP/hybrid] t=\$(round(time()-t0,digits=2))s")
    end
""",
"""
    if cfg.enable_pressure_kick
        t0 = time()
        if cfg.pressure_kick_method === :div_consistent
            apply_pressure_correction!(
                p, f_np1_proj, f_np1, d, thread_caches; delta_scale=1.0,
            )
            println("  [pressure kick enabled — div-consistent (FLIP/hybrid)] t=\$(round(time()-t0,digits=2))s")
        elseif cfg.pressure_kick_method === :curl_consistent
            apply_pressure_correction_curl!(
                p, f_np1_proj, f_np1, d, buf, thread_caches;
                delta_scale=1.0,
                atol=cfg.lsqr_atol, btol=cfg.lsqr_btol,
                maxiter=cfg.lsqr_maxiter,
                error_on_nonconvergence=cfg.lsqr_error_on_nonconvergence,
            )
            println("  [pressure kick enabled — curl-consistent (FEEC Poisson)] t=\$(round(time()-t0,digits=2))s")
        else
            throw(ArgumentError(
                "pressure_kick_method must be :div_consistent or :curl_consistent, got \$(cfg.pressure_kick_method)"
            ))
        end
    end
"""
))

function apply_edits(path::String)
    src = read(path, String)
    skipped = String[]
    applied = 0
    for (idx, (old, new)) in enumerate(EDITS)
        if !occursin(old, src)
            push!(skipped, "edit#$idx (already applied or not present)")
            continue
        end
        occurrences = length(split(src, old)) - 1
        if occurrences > 1
            error("Edit $idx in $(basename(path)): old_string matches $occurrences times — needs more context to disambiguate.")
        end
        src = replace(src, old => new; count = 1)
        applied += 1
    end
    write(path, src)
    return applied, skipped
end

println("Applying curl-consistent pressure kick to $(length(CASE_FILES)) case files...")
for fn in CASE_FILES
    path = joinpath(CASE_DIR, fn)
    if !isfile(path)
        println("  SKIP $fn (not found)")
        continue
    end
    try
        applied, skipped = apply_edits(path)
        println("  $fn: applied=$applied, skipped=$(length(skipped))")
        for s in skipped
            println("      - $s")
        end
    catch e
        println("  ERROR in $fn: $e")
        rethrow(e)
    end
end
println("Done.")
