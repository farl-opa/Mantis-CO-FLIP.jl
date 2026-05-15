#!/usr/bin/env julia
# Applies the optimization patches from the main solver to each delftblue case file.
# Run from the repo root: `julia CO-FLIP/delftblue_cases/_apply_optimizations.jl`
# Idempotent: re-runs are no-ops if all patches already applied.

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

# Each entry is (old_string, new_string). old_string MUST be unique in the file.
const EDITS = Tuple{String,String}[]

# ─── Edit 1: Domain struct gains R1_basis_indices field ────────────────────────
push!(EDITS, (
"""
    eval_cache_size::Int
    box_size::NTuple{2,Float64}
end
""",
"""
    eval_cache_size::Int
    box_size::NTuple{2,Float64}
    R1_basis_indices::Vector{Vector{Int}}
end
"""
))

# ─── Edit 2: EvaluationCache restructured with fast-path scratch buffers ───────
push!(EDITS, (
"""
struct EvaluationCache
    temp_mat::Matrix{Float64}
    results::Vector{Vector{Vector{Matrix{Float64}}}}
    xi_buf::Vector{Float64}
    eta_buf::Vector{Float64}

    \"\"\"Allocates basis-evaluation storage for fast local element evaluations.\"\"\"
    function EvaluationCache(max_basis_size::Int)
        v0  = [[zeros(1, max_basis_size) for _ in 1:2]]
        v1  = [[zeros(1, max_basis_size) for _ in 1:2] for _ in 1:2]
        res = Vector{Vector{Vector{Matrix{Float64}}}}(undef, 2)
        res[1] = v0; res[2] = v1
        return new(zeros(1, max_basis_size), res, zeros(1), zeros(1))
    end
end
""",
"""
struct EvaluationCache
    results::Vector{Vector{Vector{Matrix{Float64}}}}
    # Scratch buffers for the allocation-free 2D fast-path (per dimension):
    # bern_buf[d]  is (p+1, nder+1)  — Bernstein values
    # bsp_vals[d]  is (nder+1, n_bsp) — after BSpline extraction
    # gtb_vals[d]  is (nder+1, n_gtb) — after GTBSpline extraction
    bern_buf::NTuple{2, Matrix{Float64}}
    bsp_vals::NTuple{2, Matrix{Float64}}
    gtb_vals::NTuple{2, Matrix{Float64}}

    \"\"\"Allocates basis-evaluation storage for fast local element evaluations.\"\"\"
    function EvaluationCache(max_basis_size::Int)
        v0  = [[zeros(1, max_basis_size) for _ in 1:2]]
        v1  = [[zeros(1, max_basis_size) for _ in 1:2] for _ in 1:2]
        res = Vector{Vector{Vector{Matrix{Float64}}}}(undef, 2)
        res[1] = v0; res[2] = v1

        # Oversized to handle p ≤ 11 and nderivatives ≤ 3 (current use is p=3, nder=1)
        MAX_P1  = 12
        MAX_ND1 = 4
        bern = (zeros(MAX_P1, MAX_ND1), zeros(MAX_P1, MAX_ND1))
        bsp  = (zeros(MAX_ND1, MAX_P1), zeros(MAX_ND1, MAX_P1))
        gtb  = (zeros(MAX_ND1, MAX_P1), zeros(MAX_ND1, MAX_P1))

        return new(res, bern, bsp, gtb)
    end
end
"""
))

# ─── Edit 3: GenerateDomain precomputes R1_basis_indices ────────────────────────
push!(EDITS, (
"""
    eval_cache_size = maximum(
        length(FunctionSpaces.get_basis_indices(R[2].fem_space, eid)) for eid in 1:prod(nel)
    )

    return Domain(R[1], R[2], R[3], Forms.get_geometry(R[2]), dΩ, nel, p, k, A, A_fact, N, M, eval_cache_size, box_size)
end
""",
"""
    num_elements_R1 = prod(nel)
    R1_basis_indices = Vector{Vector{Int}}(undef, num_elements_R1)
    for eid in 1:num_elements_R1
        R1_basis_indices[eid] = collect(FunctionSpaces.get_basis_indices(R[2].fem_space, eid))
    end
    eval_cache_size = maximum(length(bi) for bi in R1_basis_indices)

    return Domain(R[1], R[2], R[3], Forms.get_geometry(R[2]), dΩ, nel, p, k, A, A_fact, N, M, eval_cache_size, box_size, R1_basis_indices)
end
"""
))

# ─── Edit 4: evaluate_fast! gets @inbounds + new evaluate_fast_2d! function ────
push!(EDITS, (
"""
function evaluate_fast!(cache::EvaluationCache, space::S, element_id::Int, xi::P, nderivatives::Int) where {S, P}
    for ord in 1:(nderivatives+1)
        for der in 1:length(cache.results[ord])
            for comp in 1:2
                fill!(cache.results[ord][der][comp], 0.0)
            end
        end
    end

    num_components = 2

    for component_idx in 1:num_components
        extraction_coefficients, J = FunctionSpaces.get_extraction(space, element_id, component_idx)
        component_basis = FunctionSpaces.get_local_basis(space, element_id, xi, nderivatives, component_idx)

        for der_order in 1:(nderivatives+1)
             for der_idx in 1:length(cache.results[der_order])
                basis_val = component_basis[der_order][der_idx][1]
                out_mat   = cache.results[der_order][der_idx][component_idx]

                LinearAlgebra.mul!(view(out_mat, :, J), basis_val, extraction_coefficients)
            end
        end
    end
    return cache.results
end
""",
"""
function evaluate_fast!(cache::EvaluationCache, space::S, element_id::Int, xi::P, nderivatives::Int) where {S, P}
    @inbounds for ord in 1:(nderivatives+1)
        for der in 1:length(cache.results[ord])
            for comp in 1:2
                fill!(cache.results[ord][der][comp], 0.0)
            end
        end
    end

    num_components = 2

    @inbounds for component_idx in 1:num_components
        extraction_coefficients, J = FunctionSpaces.get_extraction(space, element_id, component_idx)
        component_basis = FunctionSpaces.get_local_basis(space, element_id, xi, nderivatives, component_idx)

        for der_order in 1:(nderivatives+1)
             for der_idx in 1:length(cache.results[der_order])
                basis_val = component_basis[der_order][der_idx][1]
                out_mat   = cache.results[der_order][der_idx][component_idx]

                LinearAlgebra.mul!(view(out_mat, :, J), basis_val, extraction_coefficients)
            end
        end
    end
    return cache.results
end

\"\"\"
Allocation-free, thread-safe 2D fast-path evaluator that bypasses Mantis's allocating
`evaluate` chain for the CO-FLIP setup. Expects `space` to be a DirectSumSpace with 2
TensorProductSpace components, each having 2 1D GTBSplineSpace constituents (1 patch
each, single BSplineSpace inside). Writes into `cache.results` in the same layout as
`evaluate_fast!`.

Caller passes (xi, eta) ∈ [0,1] reference coords for the element directly — no
CartesianPoints wrapping. nderivatives ≤ 1 is supported in this implementation.
\"\"\"
@inline function evaluate_fast_2d!(
    cache::EvaluationCache,
    space,
    element_id::Int,
    xi_d1::Float64,
    xi_d2::Float64,
    nderivatives::Int,
)
    @inbounds for ord in 1:(nderivatives + 1)
        for der in 1:length(cache.results[ord])
            for comp in 1:2
                fill!(cache.results[ord][der][comp], 0.0)
            end
        end
    end

    J_offset = 0
    @inbounds for c in 1:2
        tp_space    = space.component_spaces[c]
        gtb_space_1 = tp_space.constituent_spaces[1]
        gtb_space_2 = tp_space.constituent_spaces[2]

        cart_idx = tp_space.cart_num_elements[element_id]
        e1 = cart_idx[1]
        e2 = cart_idx[2]

        # Single-patch GTBSpline assumed (CO-FLIP setup): patch_id = 1
        bsp_space_1 = gtb_space_1.patch_spaces[1]
        bsp_space_2 = gtb_space_2.patch_spaces[1]

        bsp_ext_1 = bsp_space_1.extraction_op.extraction_coefficients[e1][1]
        bsp_ext_2 = bsp_space_2.extraction_op.extraction_coefficients[e2][1]
        gtb_ext_1 = gtb_space_1.extraction_op.extraction_coefficients[e1][1]
        gtb_ext_2 = gtb_space_2.extraction_op.extraction_coefficients[e2][1]

        polynomial_1 = bsp_space_1.polynomials
        polynomial_2 = bsp_space_2.polynomials
        p1 = polynomial_1.p
        p2 = polynomial_2.p
        nd1 = nderivatives + 1
        n_bsp_1 = size(bsp_ext_1, 2)
        n_bsp_2 = size(bsp_ext_2, 2)
        n_gtb_1 = size(gtb_ext_1, 2)
        n_gtb_2 = size(gtb_ext_2, 2)

        # Leaf Bernstein evals — allocation-free
        bern_1 = @view cache.bern_buf[1][1:(p1 + 1), 1:nd1]
        bern_2 = @view cache.bern_buf[2][1:(p2 + 1), 1:nd1]
        Mantis.FunctionSpaces.evaluate_at!(bern_1, polynomial_1, xi_d1, nderivatives)
        Mantis.FunctionSpaces.evaluate_at!(bern_2, polynomial_2, xi_d2, nderivatives)

        # Apply BSpline extraction: (nd1, p+1) × (p+1, n_bsp) → (nd1, n_bsp)
        bsp_vals_1 = @view cache.bsp_vals[1][1:nd1, 1:n_bsp_1]
        bsp_vals_2 = @view cache.bsp_vals[2][1:nd1, 1:n_bsp_2]
        mul!(bsp_vals_1, transpose(bern_1), bsp_ext_1)
        mul!(bsp_vals_2, transpose(bern_2), bsp_ext_2)

        # Apply GTBSpline extraction: (nd1, n_bsp) × (n_bsp, n_gtb) → (nd1, n_gtb)
        gtb_vals_1 = @view cache.gtb_vals[1][1:nd1, 1:n_gtb_1]
        gtb_vals_2 = @view cache.gtb_vals[2][1:nd1, 1:n_gtb_2]
        mul!(gtb_vals_1, bsp_vals_1, gtb_ext_1)
        mul!(gtb_vals_2, bsp_vals_2, gtb_ext_2)

        # 2D kron pattern (matches Mantis): result_2d[(i-1)*n_gtb_1 + j] = vals_2[i] * vals_1[j]
        # Value key (0,0):
        out_val = cache.results[1][1][c]
        for i in 1:n_gtb_2
            v2 = gtb_vals_2[1, i]
            base = J_offset + (i - 1) * n_gtb_1
            for j in 1:n_gtb_1
                out_val[1, base + j] = v2 * gtb_vals_1[1, j]
            end
        end

        if nderivatives >= 1
            # d/dxi key (1,0): k_1=1, k_2=0
            out_dxi = cache.results[2][1][c]
            for i in 1:n_gtb_2
                v2 = gtb_vals_2[1, i]
                base = J_offset + (i - 1) * n_gtb_1
                for j in 1:n_gtb_1
                    out_dxi[1, base + j] = v2 * gtb_vals_1[2, j]
                end
            end
            # d/deta key (0,1): k_1=0, k_2=1
            out_deta = cache.results[2][2][c]
            for i in 1:n_gtb_2
                v2 = gtb_vals_2[2, i]
                base = J_offset + (i - 1) * n_gtb_1
                for j in 1:n_gtb_1
                    out_deta[1, base + j] = v2 * gtb_vals_1[1, j]
                end
            end
        end

        J_offset += n_gtb_1 * n_gtb_2
    end

    return cache.results
end
"""
))

# ─── Edit 5: probe_field_at_point uses fast path + precomputed basis indices ───
push!(EDITS, (
"""
    points = Mantis.Points.CartesianPoints((SVector{1,Float64}(xi), SVector{1,Float64}(eta)))

    eval_out     = evaluate_fast!(cache, d.R1.fem_space, elem_idx, points, 1)
    dof_indices  = FunctionSpaces.get_basis_indices(d.R1.fem_space, elem_idx)
    local_coeffs = @view u_coeffs[dof_indices]
    n_loc        = length(local_coeffs)

    val_x  = @view eval_out[1][1][1][1:n_loc]
    val_y  = @view eval_out[1][1][2][1:n_loc]
    u      = dot(val_x, local_coeffs)
    v      = dot(val_y, local_coeffs)

    dxi_u  = @view eval_out[2][1][1][1:n_loc]
    deta_u = @view eval_out[2][2][1][1:n_loc]
    dxi_v  = @view eval_out[2][1][2][1:n_loc]
    deta_v = @view eval_out[2][2][2][1:n_loc]

    du_dx = dot(dxi_u,  local_coeffs) * inv_dx
    du_dy = dot(deta_u, local_coeffs) * inv_dy
    dv_dx = dot(dxi_v,  local_coeffs) * inv_dx
    dv_dy = dot(deta_v, local_coeffs) * inv_dy

    return SVector{2, Tout}(u, v), SMatrix{2,2,Tout,4}(du_dx, dv_dx, du_dy, dv_dy)
end

\"\"\"Warm up memoization cache with initial basis evaluation.\"\"\"
function warmup_evaluation_memo!(d::Domain)
    cache  = EvaluationCache(evaluation_cache_size(d))
    points = Mantis.Points.CartesianPoints(([0.5], [0.5]))
    evaluate_fast!(cache, d.R1.fem_space, 1, points, 1)
    return nothing
end
""",
"""
    eval_out     = evaluate_fast_2d!(cache, d.R1.fem_space, elem_idx, xi, eta, 1)
    dof_indices  = d.R1_basis_indices[elem_idx]
    local_coeffs = @view u_coeffs[dof_indices]
    n_loc        = length(local_coeffs)

    val_x  = @view eval_out[1][1][1][1:n_loc]
    val_y  = @view eval_out[1][1][2][1:n_loc]
    u      = dot(val_x, local_coeffs)
    v      = dot(val_y, local_coeffs)

    dxi_u  = @view eval_out[2][1][1][1:n_loc]
    deta_u = @view eval_out[2][2][1][1:n_loc]
    dxi_v  = @view eval_out[2][1][2][1:n_loc]
    deta_v = @view eval_out[2][2][2][1:n_loc]

    du_dx = dot(dxi_u,  local_coeffs) * inv_dx
    du_dy = dot(deta_u, local_coeffs) * inv_dy
    dv_dx = dot(dxi_v,  local_coeffs) * inv_dx
    dv_dy = dot(deta_v, local_coeffs) * inv_dy

    return SVector{2, Tout}(u, v), SMatrix{2,2,Tout,4}(du_dx, dv_dx, du_dy, dv_dy)
end

\"\"\"Warm up basis-evaluation paths (compilation, memoization).\"\"\"
function warmup_evaluation_memo!(d::Domain)
    cache  = EvaluationCache(evaluation_cache_size(d))
    points = Mantis.Points.CartesianPoints(([0.5], [0.5]))
    evaluate_fast!(cache, d.R1.fem_space, 1, points, 1)
    evaluate_fast_2d!(cache, d.R1.fem_space, 1, 0.5, 0.5, 1)
    evaluate_fast_2d!(cache, d.R1.fem_space, 1, 0.5, 0.5, 0)
    return nothing
end
"""
))

# ─── Edit 6: probe_field_at_point_with_hint — drop dead branch + use fast path ─
push!(EDITS, (
"""
@inline function probe_field_at_point_with_hint(
        x::Real, y::Real,
        u_coeffs::AbstractVector{T},
        d::Domain,
        cache::EvaluationCache,
        hint_elem::Int,
    ) where {T<:Real}
    Tout   = probe_output_eltype(T)
    Lx, Ly = d.box_size
    nx, ny = d.nel

    x_wrapped = mod(x, Lx)
    y_wrapped = mod(y, Ly)

    dx = Lx / nx;  dy = Ly / ny
    inv_dx = 1.0 / dx;  inv_dy = 1.0 / dy

    ei = clamp(floor(Int, x_wrapped * inv_dx) + 1, 1, nx)
    ej = clamp(floor(Int, y_wrapped * inv_dy) + 1, 1, ny)
    elem_idx = (ej - 1) * nx + ei

    xi  = (x_wrapped - (ei - 1) * dx) * inv_dx
    eta = (y_wrapped - (ej - 1) * dy) * inv_dy

    # If point is still in hinted element and reference coords in [0,1], skip element lookup
    if hint_elem != elem_idx || xi < 0.0 || xi > 1.0 || eta < 0.0 || eta > 1.0
        # Point moved to a different element; use standard lookup
        elem_idx = (ej - 1) * nx + ei
    end

    points = Mantis.Points.CartesianPoints((SVector{1,Float64}(xi), SVector{1,Float64}(eta)))

    eval_out     = evaluate_fast!(cache, d.R1.fem_space, elem_idx, points, 1)
    dof_indices  = FunctionSpaces.get_basis_indices(d.R1.fem_space, elem_idx)
    local_coeffs = @view u_coeffs[dof_indices]
    n_loc        = length(local_coeffs)
""",
"""
@inline function probe_field_at_point_with_hint(
        x::Real, y::Real,
        u_coeffs::AbstractVector{T},
        d::Domain,
        cache::EvaluationCache,
        _hint_elem::Int,  # reserved for future hint optimization
    ) where {T<:Real}
    Tout   = probe_output_eltype(T)
    Lx, Ly = d.box_size
    nx, ny = d.nel

    x_wrapped = mod(x, Lx)
    y_wrapped = mod(y, Ly)

    dx = Lx / nx;  dy = Ly / ny
    inv_dx = 1.0 / dx;  inv_dy = 1.0 / dy

    ei = clamp(floor(Int, x_wrapped * inv_dx) + 1, 1, nx)
    ej = clamp(floor(Int, y_wrapped * inv_dy) + 1, 1, ny)
    elem_idx = (ej - 1) * nx + ei

    xi  = (x_wrapped - (ei - 1) * dx) * inv_dx
    eta = (y_wrapped - (ej - 1) * dy) * inv_dy

    eval_out     = evaluate_fast_2d!(cache, d.R1.fem_space, elem_idx, xi, eta, 1)
    dof_indices  = d.R1_basis_indices[elem_idx]
    local_coeffs = @view u_coeffs[dof_indices]
    n_loc        = length(local_coeffs)
"""
))

# ─── Edit 7: build_B_matrix uses fast path ─────────────────────────────────────
push!(EDITS, (
"""
        while pid != 0
            points = Mantis.Points.CartesianPoints((
                SVector{1,Float64}(p.can_x[pid]),
                SVector{1,Float64}(p.can_y[pid]),
            ))
            eval_out = evaluate_fast!(cache, fes, eid, points, 0)
""",
"""
        while pid != 0
            eval_out = evaluate_fast_2d!(cache, fes, eid, p.can_x[pid], p.can_y[pid], 0)
"""
))

# ─── Edit 8: apply_pressure_correction! threaded ───────────────────────────────
push!(EDITS, (
"""
    tau_coeffs = compute_pressure_delta(f_projected, f_pre_projection)

    if isempty(thread_caches)
        throw(ArgumentError("thread_caches must not be empty"))
    end

    cache = thread_caches[1]
    @inbounds for i in 1:length(p.x)
        val, _ = probe_field_at_point(p.x[i], p.y[i], tau_coeffs, d, cache)
        p.my[i] += delta_scale * val[1]
        p.mx[i] -= delta_scale * val[2]
    end
end
""",
"""
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
"""
))

# ─── Edit 9: apply_pic_blend! threaded ─────────────────────────────────────────
push!(EDITS, (
"""
    alpha <= 0.0 && return nothing

    if isempty(thread_caches)
        throw(ArgumentError("thread_caches must not be empty"))
    end

    cache = thread_caches[1]
    @inbounds for i in 1:length(p.x)
        raw_vel, _ = probe_field_at_point(p.x[i], p.y[i], f_coeffs, d, cache)

        u_g = -raw_vel[2]
        v_g =  raw_vel[1]

        p.mx[i] = (1.0 - alpha) * p.mx[i] + alpha * u_g
        p.my[i] = (1.0 - alpha) * p.my[i] + alpha * v_g
    end
    return nothing
end
""",
"""
    alpha <= 0.0 && return nothing

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
            raw_vel, _ = probe_field_at_point(p.x[i], p.y[i], f_coeffs, d, cache)

            u_g = -raw_vel[2]
            v_g =  raw_vel[1]

            p.mx[i] = (1.0 - alpha) * p.mx[i] + alpha * u_g
            p.my[i] = (1.0 - alpha) * p.my[i] + alpha * v_g
        end
    end
    return nothing
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

println("Applying optimizations to $(length(CASE_FILES)) case files...")
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
