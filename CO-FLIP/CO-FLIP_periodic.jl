using Mantis
using Random
using LinearAlgebra
using Plots
using SparseArrays
using IterativeSolvers
using LinearMaps
using StaticArrays
using CUDA
using DelimitedFiles
using Printf
using Memoization
using WriteVTK
import ReadVTK

gr()

struct Domain{F0, F1, F2, G, Q, MH, NH, MM, HF, KR0, KR0F, GG, M1F, M0M, OBS}
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
    R0_Mass_matrix::M0M
    K_R0::KR0
    K_R0_fact::KR0F
    G_R0_R1::GG
    Mass1_fact::M1F
    eval_cache_size::Int
    box_size::NTuple{2,Float64}
    R1_basis_indices::Vector{Vector{Int}}
    bc_sides::NTuple{4,Symbol}
    homogeneous_dofs_R1::Vector{Int}
    dirichlet_dofs_R1::Vector{Int}
    lift_g_R1::Vector{Float64}
    obstacle::OBS
end

mutable struct Particles
    x::Vector{Float64}
    y::Vector{Float64}
    mx::Vector{Float64}
    my::Vector{Float64}
    volume::Vector{Float64}
    can_x::Vector{Float64}
    can_y::Vector{Float64}
    head::Vector{Int}
    next::Vector{Int}
    elem_ids::Vector{Int}
    P11::Vector{Float64}
    P12::Vector{Float64}
    P21::Vector{Float64}
    P22::Vector{Float64}
    delta_t::Vector{Float64}
end

struct EvaluationCache
    results::Vector{Vector{Vector{Matrix{Float64}}}}
    bern_buf::NTuple{2, Matrix{Float64}}
    bsp_vals::NTuple{2, Matrix{Float64}}
    gtb_vals::NTuple{2, Matrix{Float64}}

    """Allocates basis-evaluation storage for fast local element evaluations."""
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

mutable struct SimulationBuffers
    x_n::Vector{Float64}
    y_n::Vector{Float64}
    mx_n::Vector{Float64}
    my_n::Vector{Float64}
    x_np1::Vector{Float64}
    y_np1::Vector{Float64}
    mx_np1::Vector{Float64}
    my_np1::Vector{Float64}

    short_P11::Vector{Float64}
    short_P12::Vector{Float64}
    short_P21::Vector{Float64}
    short_P22::Vector{Float64}

    f_n_saved::Vector{Float64}
    f_star::Vector{Float64}
    f_np1::Vector{Float64}
    f_np1_raw::Vector{Float64}
    f_np1_proj::Vector{Float64}
    f_np1_proj_prev::Vector{Float64}

    lsqr_warm::Vector{Float64}

    thread_caches::Vector{EvaluationCache}

    quarter_pids::Vector{Vector{Int}}
    add_quarter::Matrix{Int}
    counts_elem::Vector{Int}
    counts_quarter::Matrix{Int}
    keep_mask::Vector{Bool}

    V_p::Vector{Float64}

    B_I::Vector{Int}
    B_J::Vector{Int}
    B_V::Vector{Float64}
    elem_basis_indices::Vector{Vector{Int}}

    max_err_per_thread::Vector{Float64}

    v_coeffs_buf::Vector{Float64}
    b_buf::Vector{Float64}
    sol_buf::Vector{Float64}
end

"""Allocates scratch buffers for particle-to-grid transfer and time stepping."""
function SimulationBuffers(num_particles_initial::Int, ndofs::Int, num_elements::Int, d::Domain)
    n_threads  = Threads.maxthreadid()
    cache_size = evaluation_cache_size(d)
    cap        = num_particles_initial * 2
    max_nnz    = 2 * cap * d.eval_cache_size

    return SimulationBuffers(
        Vector{Float64}(undef, cap), Vector{Float64}(undef, cap),
        Vector{Float64}(undef, cap), Vector{Float64}(undef, cap),
        Vector{Float64}(undef, cap), Vector{Float64}(undef, cap),
        Vector{Float64}(undef, cap), Vector{Float64}(undef, cap),
        Vector{Float64}(undef, cap), Vector{Float64}(undef, cap),
        Vector{Float64}(undef, cap), Vector{Float64}(undef, cap),
        Vector{Float64}(undef, ndofs),
        Vector{Float64}(undef, ndofs),
        Vector{Float64}(undef, ndofs),
        Vector{Float64}(undef, ndofs),
        Vector{Float64}(undef, ndofs),
        zeros(Float64, ndofs),
        zeros(Float64, ndofs),
        [EvaluationCache(cache_size) for _ in 1:n_threads],
        [Int[] for _ in 1:num_elements * 4],
        zeros(Int, num_elements, 4),
        zeros(Int, num_elements),
        zeros(Int, num_elements, 4),
        trues(cap),
        Vector{Float64}(undef, 2 * cap),
        Vector{Int}(undef, max_nnz),
        Vector{Int}(undef, max_nnz),
        Vector{Float64}(undef, max_nnz),
        [Int[] for _ in 1:num_elements],
        zeros(Float64, n_threads),
        zeros(Float64, size(d.N_Hodge, 1)),
        zeros(Float64, size(d.N_Hodge, 1)),
        zeros(Float64, size(d.N_Hodge, 1)),
    )
end

Base.@kwdef struct SimulationConfig
    nel::NTuple{2,Int}                = (96, 64)
    p::NTuple{2,Int}                  = (3, 3)
    k::NTuple{2,Int}                  = (1, 1)
    box_size::NTuple{2,Float64}       = (40, 20)
    starting_point::NTuple{2,Float64} = (0.0, 0.0)
    boundary_condition::Union{Symbol, NTuple{2,Symbol}, NTuple{4,Symbol}} = (:inlet, :outlet, :periodic, :periodic)
    inlet_U_inf::Float64              = 1.0
    obstacle::Union{Nothing,NamedTuple} = (kind=:cylinder, center=(10.0, 10.0), radius=1.0)
    particles_per_cell::Int           = 10
    stratified_seeding::Bool          = true
    volume_convention::Symbol         = :physical
    rng_seed::Union{Int,Nothing}      = nothing
    flow_type::Symbol                 = :cylinder
    target_cfl::Float64               = 0.5
    T_final::Float64                  = 40.0
    viscosity::Float64                = 0.02
    max_fp_iter::Int                  = 6
    fp_tol::Float64                   = 5e-9
    enable_energy_correction::Bool    = true
    enable_pressure_kick::Bool        = true
    pic_blend_alpha::Float64          = 0.02
    min_particles_per_element::Int    = 4
    max_particles_per_element::Int    = 16
    min_particles_per_quarter::Int    = 1
    ftle_threshold::Float64           = 1.0
    max_longterm_delta_t::Float64     = 1.0
    ftle_use_rate::Bool               = true
    global_ftle_gate::Float64         = -Inf
    delayed_reinit_frequency::Int     = 0
    output_every::Int                 = 1
    clear_memo_every::Int             = 0
    clear_memo_every_fp_iter::Int     = 1
    lsqr_atol::Float64                = 1e-9
    lsqr_btol::Float64                = 1e-9
    lsqr_maxiter::Int                 = 2000
    lsqr_error_on_nonconvergence::Bool = true
    cfl_recheck_tolerance::Float64    = 0.5
    cfl_adaptive::Bool                = true
    projection_mean_subtract::Bool    = true
    advection_time_integrator::Symbol = :rk4
end

"""Promote coefficient type for field probe output."""
probe_output_eltype(::Type{T}) where {T<:Real} = promote_type(Float64, T)

"""Get maximum local basis size for evaluation cache."""
evaluation_cache_size(d::Domain) = d.eval_cache_size

"""Per-side BC tags currently understood by the parser. Inlet/outlet are
accepted here so future stages can ship without changing the parser, but
this stage classifies them as homogeneous walls (zero u·n) — the inlet
lift and outlet free-DOF handling live in later stages."""
const VALID_BC_SIDES = (:periodic, :wall, :inlet, :outlet)

"""
Normalize a user-supplied boundary_condition to a per-side `NTuple{4,Symbol}`
in the order `(left, right, bottom, top)` (i.e. (x-min, x-max, y-min, y-max)).

Accepted input forms:
- `Symbol`: `:periodic` or `:wall` applied to all four sides.
- `NTuple{2,Symbol}`: legacy per-axis `(axis_x, axis_y)`; each axis value is
  broadcast to both of its sides.
- `NTuple{4,Symbol}`: explicit per-side specification.

Axis-consistency rule for periodicity: an axis must be `:periodic` on both
sides or non-periodic on both sides. Periodic B-splines wrap a whole
univariate axis, so mixing `:periodic` with anything else on the same axis
is ill-defined.
"""
function normalize_bc(bc)::NTuple{4,Symbol}
    if bc isa Symbol
        if bc === :periodic
            return (:periodic, :periodic, :periodic, :periodic)
        elseif bc === :wall
            return (:wall, :wall, :wall, :wall)
        elseif bc === :neumann
            @warn "boundary_condition :neumann is a legacy stub; treating as :periodic."
            return (:periodic, :periodic, :periodic, :periodic)
        else
            throw(ArgumentError(
                "boundary_condition Symbol must be :periodic or :wall, got $bc"
            ))
        end
    elseif bc isa NTuple{2,Symbol}
        for s in bc
            s in VALID_BC_SIDES || throw(ArgumentError(
                "per-axis boundary_condition entries must be in $(VALID_BC_SIDES), got $s"
            ))
        end
        return (bc[1], bc[1], bc[2], bc[2])
    elseif bc isa NTuple{4,Symbol}
        for s in bc
            s in VALID_BC_SIDES || throw(ArgumentError(
                "per-side boundary_condition entries must be in $(VALID_BC_SIDES), got $s"
            ))
        end
        (bc[1] === :periodic) == (bc[2] === :periodic) || throw(ArgumentError(
            "x-axis (sides 1,2) must be both :periodic or both non-periodic, got $((bc[1], bc[2]))"
        ))
        (bc[3] === :periodic) == (bc[4] === :periodic) || throw(ArgumentError(
            "y-axis (sides 3,4) must be both :periodic or both non-periodic, got $((bc[3], bc[4]))"
        ))
        return bc
    else
        throw(ArgumentError(
            "boundary_condition must be Symbol, NTuple{2,Symbol}, or NTuple{4,Symbol}, got $(typeof(bc))"
        ))
    end
end

"""Per-axis summary `(axis_x, axis_y)` collapsing a per-side BC. An axis is
`:periodic` iff both of its sides are `:periodic`, else `:wall` (the catch-all
non-periodic tag the FE-space construction needs)."""
@inline function bc_axes(bc_sides::NTuple{4,Symbol})::NTuple{2,Symbol}
    ax = bc_sides[1] === :periodic && bc_sides[2] === :periodic ? :periodic : :wall
    ay = bc_sides[3] === :periodic && bc_sides[4] === :periodic ? :periodic : :wall
    return (ax, ay)
end

"""True if BC has any non-periodic side."""
has_walls(bc_sides::NTuple{4,Symbol}) = any(s -> s !== :periodic, bc_sides)

"""Per-axis position policy `(mode_x, mode_y)` derived from per-side BC.
An axis is `:clamp` when its sides are non-periodic and `:periodic` otherwise.
This summary is sufficient for the position-arithmetic wrap/clamp choice;
per-side asymmetry (e.g. inlet vs outlet) is handled where it matters."""
@inline function bc_to_position_modes(bc_sides::NTuple{4,Symbol})
    axes = bc_axes(bc_sides)
    return (axes[1] === :periodic ? :periodic : :clamp,
            axes[2] === :periodic ? :periodic : :clamp)
end

"""
Classify R1 wall-normal-trace DOFs into a homogeneous set (`u·n = 0`) and a
prescribed-Dirichlet set (used for non-zero inlet conditions). DOFs on
`:outlet` sides are intentionally left out of both sets — those DOFs remain
**free** so the Hodge-Laplace projection produces a natural zero-gradient /
do-nothing outflow there.

R1 has two TensorProductSpace components (rotated R1 storage; ★u_h = phys vel):
- comp1 = D_x ⊗ P_y → physical v, wall-normal trace on y-sides (bottom=3, top=4).
- comp2 = P_x ⊗ D_y → physical -u, wall-normal trace on x-sides (left=1, right=2).

Per-side classification:
- `:inlet`    → DOF joins the Dirichlet set; coefficient set from `lift_g_R1`.
- `:wall`     → DOF joins the homogeneous set; coefficient pinned to 0.
- `:outlet`   → DOF is skipped (free). Operator factorisations don't pin it,
                LSQR fits it from particles, and the projection produces the
                natural `∂p/∂n = 0` outflow on those DOFs automatically.
- `:periodic` → DOF is skipped (no boundary trace on a wrapped axis).

Returns `(homogeneous_dofs_R1, dirichlet_dofs_R1)` as two disjoint sorted
`Vector{Int}` indices.
"""
function classify_R1_boundary_dofs(R1_space, bc_sides::NTuple{4,Symbol})
    homog = Set{Int}()
    dir   = Set{Int}()
    has_walls(bc_sides) || return (Int[], Int[])

    components  = FunctionSpaces.get_component_spaces(R1_space.fem_space)
    dof_offsets = FunctionSpaces.get_dof_offsets(R1_space.fem_space)

    # Assign one trace DOF to its right bucket based on the side type. Returns
    # nothing for :outlet / :periodic so those DOFs stay unconstrained.
    @inline function dispatch!(kind::Symbol, gdof::Int)
        if kind === :inlet
            push!(dir, gdof)
        elseif kind === :wall
            push!(homog, gdof)
        end
        # :outlet, :periodic → free
        return nothing
    end

    # Component 1 — y-side traces (sides 3=bottom, 4=top)
    bot, top = bc_sides[3], bc_sides[4]
    if bot !== :periodic || top !== :periodic
        comp1   = components[1]
        nbasis1 = FunctionSpaces.get_constituent_num_basis(comp1)  # (n_dx, n_py)
        n_py    = nbasis1[2]
        for bid in 1:FunctionSpaces.get_num_basis(comp1)
            cb   = FunctionSpaces.get_constituent_basis_id(comp1, bid)
            gdof = dof_offsets[1] + bid
            if cb[2] == 1
                dispatch!(bot, gdof)
            elseif cb[2] == n_py
                dispatch!(top, gdof)
            end
        end
    end

    # Component 2 — x-side traces (sides 1=left, 2=right)
    lf, rt = bc_sides[1], bc_sides[2]
    if lf !== :periodic || rt !== :periodic
        comp2   = components[2]
        nbasis2 = FunctionSpaces.get_constituent_num_basis(comp2)  # (n_px, n_dy)
        n_px    = nbasis2[1]
        for bid in 1:FunctionSpaces.get_num_basis(comp2)
            cb   = FunctionSpaces.get_constituent_basis_id(comp2, bid)
            gdof = dof_offsets[2] + bid
            if cb[1] == 1
                dispatch!(lf, gdof)
            elseif cb[1] == n_px
                dispatch!(rt, gdof)
            end
        end
    end

    return (sort!(collect(homog)), sort!(collect(dir)))
end

"""Backwards-compatible alias returning the union of homogeneous and
Dirichlet wall-trace DOFs (stage-1 behaviour). Kept for any external
callers; internal code should use `classify_R1_boundary_dofs`."""
function compute_wall_dofs_R1(R1_space, bc_sides::NTuple{4,Symbol})::Vector{Int}
    h, d = classify_R1_boundary_dofs(R1_space, bc_sides)
    return sort!(vcat(h, d))
end

"""
Snapshot the inlet lift `lift_g_R1` from a freshly-fitted unconstrained R1
coefficient vector. No-op when the Dirichlet set is empty or the lift has
already been populated. The "already populated" check looks at any nonzero
entry on the Dirichlet indices — the lift starts as all zeros, so the first
P2G with a non-trivial inlet velocity flips at least one entry, after which
this call becomes a no-op for the rest of the run.

This approach guarantees `lift_g_R1[i]` matches whatever value the LSQR
particle-to-grid fit produces for the inlet velocity in the same FEM
normalisation the rest of the solver uses — without an extra L2 projection,
Greville interpolation, or mass-matrix solve at construction time.
"""
function maybe_initialise_inlet_lift!(d::Domain, u_grid_unconstrained::AbstractVector{Float64})
    isempty(d.dirichlet_dofs_R1) && return d
    already_set = false
    @inbounds for i in d.dirichlet_dofs_R1
        if d.lift_g_R1[i] != 0.0
            already_set = true
            break
        end
    end
    already_set && return d
    @inbounds for i in d.dirichlet_dofs_R1
        d.lift_g_R1[i] = u_grid_unconstrained[i]
    end
    return d
end

"""Stamp boundary-trace coefficients onto an R1 coefficient vector:
homogeneous DOFs go to 0, Dirichlet DOFs go to `lift_g_R1[i]`."""
@inline function enforce_boundary_dofs!(coeffs::AbstractVector{Float64}, d::Domain)
    @inbounds for i in d.homogeneous_dofs_R1
        coeffs[i] = 0.0
    end
    @inbounds for i in d.dirichlet_dofs_R1
        coeffs[i] = d.lift_g_R1[i]
    end
    return coeffs
end

"""True iff point `(x, y)` lies strictly inside the obstacle. `obs === nothing`
always returns false. Currently supports `obs.kind === :cylinder`."""
@inline function point_inside_obstacle(x::Float64, y::Float64, obs)::Bool
    obs === nothing && return false
    if obs.kind === :cylinder
        cx, cy = obs.center
        dx = x - cx;  dy = y - cy
        return dx * dx + dy * dy < obs.radius * obs.radius
    end
    return false
end

"""True iff every corner of the rectangular element `eid` lies strictly
inside the obstacle. For a convex obstacle (e.g. cylinder) this is equivalent
to the whole element being inside; for non-convex obstacles a finer sample
would be needed but `:cylinder` is the only supported kind in stage 5."""
@inline function element_inside_obstacle_box(
        eid::Int, nel::NTuple{2,Int}, box_size::NTuple{2,Float64}, obs,
    )::Bool
    obs === nothing && return false
    nx, ny = nel
    Lx, Ly = box_size
    dx = Lx / nx;  dy = Ly / ny
    ej = (eid - 1) ÷ nx + 1
    ei = eid - (ej - 1) * nx
    x0 = (ei - 1) * dx
    y0 = (ej - 1) * dy
    return point_inside_obstacle(x0,      y0,      obs) &&
           point_inside_obstacle(x0 + dx, y0,      obs) &&
           point_inside_obstacle(x0,      y0 + dy, obs) &&
           point_inside_obstacle(x0 + dx, y0 + dy, obs)
end

"""
Compute the R1 DOFs whose entire basis support lies in elements that are
fully inside the obstacle. Pinning these to zero exactly zeros the velocity
in the interior of the obstacle without polluting the surrounding fluid:
any DOF that is also supported on a partially-fluid element is excluded.

Returns a sorted `Vector{Int}` of global R1 DOF indices. Empty when
`obs === nothing`.

**Not currently called by `GenerateDomain`.** Strong-Dirichlet pinning of
*interior* R1 DOFs in the Hodge-Laplace saddle-point operator produces
stranded R2 rows (all-zero rows from pinned R1 columns + zero A_22 block),
which makes `ldiv!` return `O(10^54)` entries and the solver explode.
Kept here as a building block for a future enforcement stage that also
pins the stranded R2 DOFs. Today the no-slip surface is established
purely via `apply_obstacle_brinkman!`.
"""
function compute_solid_dofs_R1(
        R1_basis_indices::Vector{Vector{Int}},
        nel::NTuple{2,Int}, box_size::NTuple{2,Float64},
        obs, n_R1::Int,
    )::Vector{Int}
    obs === nothing && return Int[]
    num_elements = prod(nel)

    # Tag each DOF as "touched by an outside element" and "touched by an
    # inside element". Solid DOFs are those touched only by inside elements.
    touched_outside = falses(n_R1)
    touched_inside  = falses(n_R1)
    @inbounds for eid in 1:num_elements
        inside = element_inside_obstacle_box(eid, nel, box_size, obs)
        flags  = inside ? touched_inside : touched_outside
        for dof in R1_basis_indices[eid]
            flags[dof] = true
        end
    end

    solid = Int[]
    @inbounds for dof in 1:n_R1
        if touched_inside[dof] && !touched_outside[dof]
            push!(solid, dof)
        end
    end
    return solid
end

"""
Apply symmetric strong-Dirichlet zeroing to a sparse matrix in place: zero
rows and columns indexed by `dofs`, then place 1.0 on the diagonal. Preserves
symmetry; the resulting linear system pins those DOFs to zero when the RHS
also has zeros at those rows.
"""
function apply_dirichlet_zero!(A::SparseMatrixCSC, dofs::AbstractVector{Int})
    isempty(dofs) && return A
    mask = falses(size(A, 1))
    @inbounds for i in dofs; mask[i] = true; end

    # Walk CSC columns and zero out rows in `dofs`; if the column itself is in
    # `dofs`, zero the entire column.
    @inbounds for col in 1:size(A, 2)
        col_in_dofs = mask[col]
        for k in A.colptr[col]:(A.colptr[col + 1] - 1)
            row = A.rowval[k]
            if col_in_dofs || mask[row]
                A.nzval[k] = 0.0
            end
        end
    end
    @inbounds for i in dofs
        A[i, i] = 1.0
    end
    dropzeros!(A)
    return A
end

"""Zero the rows of `A` indexed by `dofs`."""
function zero_rows!(A::SparseMatrixCSC, dofs::AbstractVector{Int})
    isempty(dofs) && return A
    mask = falses(size(A, 1))
    @inbounds for i in dofs; mask[i] = true; end
    @inbounds for col in 1:size(A, 2)
        for k in A.colptr[col]:(A.colptr[col + 1] - 1)
            if mask[A.rowval[k]]
                A.nzval[k] = 0.0
            end
        end
    end
    dropzeros!(A)
    return A
end

"""Zero entire columns of `A` indexed by `dofs`. Used to make a rectangular
operator (e.g. weak-gradient G[R0,R1]) ignore the wall-pinned R1 DOFs in its
column space."""
function zero_cols!(A::SparseMatrixCSC, dofs::AbstractVector{Int})
    isempty(dofs) && return A
    @inbounds for col in dofs
        (col < 1 || col > size(A, 2)) && continue
        for k in A.colptr[col]:(A.colptr[col + 1] - 1)
            A.nzval[k] = 0.0
        end
    end
    dropzeros!(A)
    return A
end

"""Build De Rham complex with quadrature and assembled Hodge-Laplace/mass operators."""
function GenerateDomain(
        nel::NTuple{2,Int},
        p::NTuple{2,Int},
        k::NTuple{2,Int};
        box_size::NTuple{2,Float64}=(1.0, 1.0),
        starting_point::NTuple{2,Float64}=(0.0, 0.0),
        boundary_condition=:periodic,
        obstacle=nothing,
    )
    bc_sides = normalize_bc(boundary_condition)

    nq_assembly    = p .+ 1
    nq_error       = nq_assembly .* 2
    ∫ₐ, ∫ₑ = Quadrature.get_canonical_quadrature_rules(Quadrature.gauss_legendre, nq_assembly, nq_error)
    dΩ = Quadrature.StandardQuadrature(∫ₐ, prod(nel))

    bc_ax = bc_axes(bc_sides)
    periodic_flags = (bc_ax[1] === :periodic, bc_ax[2] === :periodic)
    R = Forms.create_tensor_product_bspline_de_rham_complex(
        starting_point, box_size, nel, p, k; periodic=periodic_flags,
    )

    # Build the element→DOF map early so the obstacle solid-DOF classification
    # can run before the matrix-level Dirichlet pinning. The result is also
    # stored on Domain at the end.
    num_elements = prod(nel)
    R1_basis_indices = Vector{Vector{Int}}(undef, num_elements)
    for eid in 1:num_elements
        R1_basis_indices[eid] = collect(FunctionSpaces.get_basis_indices(R[2].fem_space, eid))
    end
    eval_cache_size = maximum(length(bi) for bi in R1_basis_indices)
    n_R1 = FunctionSpaces.get_num_basis(R[2].fem_space)

    homogeneous_dofs_R1, dirichlet_dofs_R1 =
        classify_R1_boundary_dofs(R[2], bc_sides)

    # Immersed-boundary obstacle: we do *not* fold interior-cylinder R1 DOFs
    # into the matrix pinning. Strong-Dirichlet-pinning an *interior* R1 DOF
    # (one that's not on a domain boundary) decouples it from the discrete
    # exterior derivative `d:R1→R2` — but every R2 DOF whose support lies in
    # cells coupled exclusively to pinned R1 columns then has an all-zero
    # row in the Hodge-Laplace saddle-point operator. LU produces a tiny
    # pivot, `ldiv!` returns O(10^54) entries, the next advection blows up.
    # The cylinder no-slip is enforced via particle Brinkman penalisation
    # instead (see `apply_obstacle_brinkman!`), which is sufficient for the
    # qualitative-demonstration target. A future stage that wants strict
    # operator-level enforcement must also pin the corresponding stranded
    # R2 DOFs to keep the saddle-point system well-posed.
    pinned_dofs_R1 = sort!(vcat(homogeneous_dofs_R1, dirichlet_dofs_R1))

    A, N = assemble_hodge_laplacian_matrices(R[2], R[3], dΩ)
    M    = assemble_1form_mass_matrix(R[2], dΩ)
    M_R0 = assemble_R0_mass_matrix(R[1], dΩ)

    if !isempty(pinned_dofs_R1)
        # Both homogeneous and Dirichlet wall-trace DOFs are symmetrically
        # pinned in the operator factorisations (rows/cols zeroed, diag=1).
        # The RHS at homogeneous DOFs is enforced to 0 and at Dirichlet DOFs
        # to lift_g_R1[i] at the call sites where the solver is invoked, so
        # the resulting field satisfies u·n = 0 on walls/outlet/obstacle and
        # u·n = g on the inlet exactly.
        apply_dirichlet_zero!(A, pinned_dofs_R1)
        apply_dirichlet_zero!(M, pinned_dofs_R1)
    end

    A_fact = lu(A)
    M_fact = lu(M)

    K_R0    = assemble_R0_stiffness_matrix(R[1], dΩ)
    G_R0_R1 = assemble_R0_R1_weak_grad_matrix(R[1], R[2], dΩ)

    if !isempty(pinned_dofs_R1)
        # Pinned DOFs hold prescribed values; the weak-gradient operator's R1
        # column space should not couple to them. Zeroing G_R0_R1 columns
        # means (a) the discrete divergence G·u ignores wall normal
        # components, and (b) Gᵀ·p (used to map pressure back to R1 in the
        # projection / curl-consistent kick) has zero rows at the pinned
        # DOFs, so the projection-induced update preserves the prescribed
        # boundary trace.
        zero_cols!(G_R0_R1, pinned_dofs_R1)
    end

    n0 = size(K_R0, 1)
    ridge_R0 = 1e-10 * (sum(abs, diag(K_R0)) / max(n0, 1) + 1.0)
    K_R0_reg = K_R0 + ridge_R0 * sparse(I, n0, n0)
    K_R0_fact = lu(K_R0_reg)

    lift_g_R1 = zeros(Float64, n_R1)

    if !isnothing(obstacle)
        println("  Obstacle: $(obstacle.kind) at center=$(obstacle.center), radius=$(obstacle.radius). " *
                "No-slip enforced via particle Brinkman penalisation (no operator-level DOF pinning).")
    end

    return Domain(R[1], R[2], R[3], Forms.get_geometry(R[2]), dΩ, nel, p, k,
                  A, A_fact, N, M, M_R0, K_R0, K_R0_fact, G_R0_R1, M_fact,
                  eval_cache_size, box_size, R1_basis_indices,
                  bc_sides, homogeneous_dofs_R1, dirichlet_dofs_R1, lift_g_R1,
                  obstacle)
end

"""Evaluate analytic velocity field at specified point. `bc` selects per-side
boundary type so flows that depend on it (e.g. leapfrog with wall images) can
produce a BC-consistent velocity. `obstacle`/`U_inf` are needed only by the
`:cylinder` flow (uniform far-field with zero velocity inside the obstacle)."""
function initial_velocity(
        flow_type::Symbol, px::Float64, py::Float64, Lx::Float64, Ly::Float64;
        bc::NTuple{4,Symbol}=(:periodic, :periodic, :periodic, :periodic),
        obstacle=nothing,
        U_inf::Float64=0.0,
    )
    if flow_type == :tg;             return flow_taylor_green(px, py, Lx, Ly)
    elseif flow_type == :decaying_tg; return flow_decaying_tg(px, py, Lx, Ly)
    elseif flow_type == :vortex;     return flow_lamb_oseen(px, py, Lx, Ly)
    elseif flow_type == :gyre;       return flow_double_gyre(px, py, Lx, Ly)
    elseif flow_type == :decay;      return flow_decay(px, py, Lx, Ly)
    elseif flow_type == :convecting; return flow_convecting_vortex(px, py, Lx, Ly)
    elseif flow_type == :merging;    return flow_merging_vortices(px, py, Lx, Ly)
    elseif flow_type == :uniform;     return flow_uniform(px, py, Lx, Ly)
    elseif flow_type == :zero;        return flow_zero(px, py, Lx, Ly)
    elseif flow_type == :shear;       return flow_shear(px, py, Lx, Ly)
    elseif flow_type == :kh;          return flow_kelvin_helmholtz(px, py, Lx, Ly)
    elseif flow_type == :dipole;      return flow_dipole(px, py, Lx, Ly)
    elseif flow_type == :leapfrog;    return flow_leapfrog(px, py, Lx, Ly; bc=bc)
    elseif flow_type == :four_vortex; return flow_four_vortex(px, py, Lx, Ly)
    elseif flow_type == :stuart;      return flow_stuart(px, py, Lx, Ly)
    elseif flow_type == :cylinder;   return flow_cylinder(px, py, Lx, Ly, obstacle, U_inf)
    else; error("Unknown flow type: $flow_type")
    end
end

"""Tensor-product Gauss-Legendre nodes and weights on `[0, 1]` for `n = 1..6`.
Returned weights sum to 1. Use ``x_i·dx + x0`` to map to a cell and
``w_i·dx`` to get the physical 1D weight."""
function gauss_legendre_01(n::Int)
    n == 1 && return ([0.5], [1.0])
    n == 2 && return ([0.21132486540518713, 0.7886751345948129],
                      [0.5, 0.5])
    n == 3 && return ([0.1127016653792583,  0.5, 0.8872983346207417],
                      [0.2777777777777778, 0.4444444444444444, 0.2777777777777778])
    n == 4 && return ([0.06943184420297371, 0.33000947820757187,
                       0.6699905217924281,  0.9305681557970263],
                      [0.17392742256872692, 0.32607257743127305,
                       0.32607257743127305, 0.17392742256872692])
    n == 5 && return ([0.04691007703066802, 0.23076534494715845, 0.5,
                       0.7692346550528415,  0.9530899229693319],
                      [0.11846344251309455, 0.23931433524968326,
                       0.28444444444444444, 0.23931433524968326,
                       0.11846344251309455])
    n == 6 && return ([0.03376524289842398, 0.16939530676686775,
                       0.38069040695840156, 0.6193095930415985,
                       0.8306046932331323,  0.9662347571015761],
                      [0.0856622461895852,  0.18038078652406930,
                       0.23395696728634552, 0.23395696728634552,
                       0.18038078652406930, 0.0856622461895852])
    error("gauss_legendre_01: only n=1..6 supported, got n=$n")
end

"""Generate initial particle distribution with stratified or random seeding.

`U_inf` is forwarded to `initial_velocity` and is consumed by `:cylinder` /
similar flow types that need the far-field uniform speed. The obstacle (if
any) is read from `domain.obstacle`.

`seeding_mode`:
- `:auto` (default): legacy behaviour driven by `stratified_seeding` —
  jittered stratified sub-cell placement or pure uniform random.
- `:gauss_legendre`: deterministic tensor-product Gauss-Legendre nodes
  inside each element, with `vol_p` set to the GL weight (×dx·dy). The
  LSQR P2G fit then becomes an *exact* polynomial quadrature up to
  total degree `2n−1` per direction (with `n = floor(sqrt(ppc))`),
  exposing the true spline-space approximation order in convergence
  studies. `volume_convention` is ignored in this mode. Excess
  particles requested by `num_particles ≠ n²·num_elements` are
  dropped (return count is `n²·num_elements`)."""
function generate_particles(
        num_particles::Int,
        domain::Domain,
        flow_type::Symbol=:vortex;
        stratified_seeding::Bool=true,
        seeding_mode::Symbol=:auto,
        rng_seed::Union{Int,Nothing}=nothing,
        volume_convention::Symbol=:physical,
        boundary_condition=:periodic,
        U_inf::Float64=0.0,
    )
    if seeding_mode === :gauss_legendre
        return _generate_particles_gauss_legendre(num_particles, domain, flow_type;
            boundary_condition=boundary_condition, U_inf=U_inf)
    end
    seeding_mode === :auto ||
        throw(ArgumentError("seeding_mode must be :auto or :gauss_legendre, got $seeding_mode"))
    if rng_seed !== nothing
        Random.seed!(rng_seed)
    end

    if !(volume_convention in (:physical, :cell_fraction))
        throw(ArgumentError(
            "volume_convention must be :physical or :cell_fraction, got $(volume_convention)"
        ))
    end

    bc       = normalize_bc(boundary_condition)
    modes    = bc_to_position_modes(bc)
    obstacle = domain.obstacle
    Lx, Ly = domain.box_size
    nx, ny = domain.nel
    dx     = Lx / nx
    dy     = Ly / ny

    num_elements = nx * ny
    base_ppc     = num_particles ÷ num_elements
    extra        = num_particles - base_ppc * num_elements

    x   = Vector{Float64}(undef, num_particles)
    y   = Vector{Float64}(undef, num_particles)
    mx  = Vector{Float64}(undef, num_particles)
    my  = Vector{Float64}(undef, num_particles)
    vol = ones(Float64, num_particles)

    can_x    = zeros(Float64, num_particles)
    can_y    = zeros(Float64, num_particles)
    head     = zeros(Int, num_elements)
    next     = zeros(Int, num_particles)
    elem_ids = zeros(Int, num_particles)

    P11 = ones(Float64, num_particles)
    P12 = zeros(Float64, num_particles)
    P21 = zeros(Float64, num_particles)
    P22 = ones(Float64, num_particles)
    delta_t = zeros(Float64, num_particles)

    pid = 1
    if stratified_seeding && base_ppc > 0
        used_N = max(1, floor(Int, sqrt(base_ppc) + 1e-12))
        for ej in 1:ny, ei in 1:nx
            x0 = (ei - 1) * dx
            y0 = (ej - 1) * dy
            for jj in 0:(used_N - 1), ii in 0:(used_N - 1)
                px = wrap_axis(x0 + ((ii + rand()) / used_N) * dx, Lx, modes[1], 1e-12)
                py = wrap_axis(y0 + ((jj + rand()) / used_N) * dy, Ly, modes[2], 1e-12)
                x[pid]  = px
                y[pid]  = py
                u, v    = initial_velocity(flow_type, px, py, Lx, Ly; bc=bc, obstacle=obstacle, U_inf=U_inf)
                mx[pid] = u
                my[pid] = v
                pid += 1
            end
            remainder = base_ppc - used_N * used_N
            for _ in 1:remainder
                px = wrap_axis(x0 + rand() * dx, Lx, modes[1], 1e-12)
                py = wrap_axis(y0 + rand() * dy, Ly, modes[2], 1e-12)
                x[pid]  = px
                y[pid]  = py
                u, v    = initial_velocity(flow_type, px, py, Lx, Ly; bc=bc, obstacle=obstacle, U_inf=U_inf)
                mx[pid] = u
                my[pid] = v
                pid += 1
            end
        end
        for _ in 1:extra
            px = rand() * Lx
            py = rand() * Ly
            x[pid]  = px
            y[pid]  = py
            u, v    = initial_velocity(flow_type, px, py, Lx, Ly; bc=bc, obstacle=obstacle, U_inf=U_inf)
            mx[pid] = u
            my[pid] = v
            pid += 1
        end
    else
        for i in 1:num_particles
            px = rand() * Lx
            py = rand() * Ly
            x[i]  = px
            y[i]  = py
            u, v  = initial_velocity(flow_type, px, py, Lx, Ly; bc=bc, obstacle=obstacle, U_inf=U_inf)
            mx[i] = u
            my[i] = v
        end
    end

    if volume_convention === :physical
        fill!(vol, (Lx * Ly) / num_particles)
    else
        ppc_for_volume = base_ppc > 0 ? base_ppc : num_particles
        fill!(vol, 1.0 / ppc_for_volume)
    end

    return Particles(x, y, mx, my, vol, can_x, can_y, head, next, elem_ids,
                     P11, P12, P21, P22, delta_t)
end

"""Deterministic tensor-product Gauss-Legendre particle seeding (helper
for `generate_particles(...; seeding_mode=:gauss_legendre)`).

Places `n² = floor(√(num_particles ÷ num_elements))² ` particles per
element at GL nodes. Particle volume is the GL weight ×dx·dy (sums to
cell area). With this volume convention the LSQR P2G discrete inner
product `Σ vol_p · f(x_p) g(x_p)` equals the continuous L2 inner
product exactly when `f·g` is a polynomial of degree ≤ `2n−1` per
direction inside each cell, which is what allows clean
spline-order convergence in `test_p2g_taylor_green_convergence`."""
function _generate_particles_gauss_legendre(
        num_particles::Int,
        domain::Domain,
        flow_type::Symbol;
        boundary_condition=:periodic,
        U_inf::Float64=0.0,
    )
    bc       = normalize_bc(boundary_condition)
    modes    = bc_to_position_modes(bc)
    obstacle = domain.obstacle
    Lx, Ly = domain.box_size
    nx, ny = domain.nel
    dx     = Lx / nx
    dy     = Ly / ny

    num_elements = nx * ny
    base_ppc     = max(1, num_particles ÷ num_elements)
    n_gl         = max(1, floor(Int, sqrt(base_ppc) + 1e-12))
    nodes_01, weights_01 = gauss_legendre_01(n_gl)

    actual_ppc = n_gl * n_gl
    total      = actual_ppc * num_elements

    if actual_ppc != base_ppc
        @info "GL seeding: requested $(base_ppc) ppc rounded down to $(actual_ppc) (n_gl=$n_gl per direction)"
    end

    x   = Vector{Float64}(undef, total)
    y   = Vector{Float64}(undef, total)
    mx  = Vector{Float64}(undef, total)
    my  = Vector{Float64}(undef, total)
    vol = Vector{Float64}(undef, total)

    can_x    = zeros(Float64, total)
    can_y    = zeros(Float64, total)
    head     = zeros(Int, num_elements)
    next     = zeros(Int, total)
    elem_ids = zeros(Int, total)

    P11 = ones(Float64,  total)
    P12 = zeros(Float64, total)
    P21 = zeros(Float64, total)
    P22 = ones(Float64,  total)
    delta_t = zeros(Float64, total)

    pid = 1
    @inbounds for ej in 1:ny, ei in 1:nx
        x0 = (ei - 1) * dx
        y0 = (ej - 1) * dy
        for jj in 1:n_gl, ii in 1:n_gl
            px = wrap_axis(x0 + nodes_01[ii] * dx, Lx, modes[1], 1e-12)
            py = wrap_axis(y0 + nodes_01[jj] * dy, Ly, modes[2], 1e-12)
            w  = weights_01[ii] * weights_01[jj] * dx * dy
            x[pid]  = px
            y[pid]  = py
            u, v    = initial_velocity(flow_type, px, py, Lx, Ly;
                                       bc=bc, obstacle=obstacle, U_inf=U_inf)
            mx[pid]  = u
            my[pid]  = v
            vol[pid] = w
            pid += 1
        end
    end

    return Particles(x, y, mx, my, vol, can_x, can_y, head, next, elem_ids,
                     P11, P12, P21, P22, delta_t)
end

"""Build per-element linked lists and canonical reference coordinates for particles."""
function particle_sorter!(p::Particles, d::Domain; cull_outlets::Bool=true)
    fill!(p.head, 0)

    nx, ny = d.nel
    Lx, Ly = d.box_size
    dx = Lx / nx;  dy = Ly / ny
    inv_dx = 1.0 / dx;  inv_dy = 1.0 / dy
    modes  = bc_to_position_modes(d.bc_sides)
    sides  = d.bc_sides
    tol    = 1e-12
    # Slab thickness in which wall-normal momentum is zeroed. Anything in
    # the outermost ~5% of a cell is treated as a wall straggler, otherwise
    # round-off-clamped particles never get cleaned up.
    near_wall_tol_x = max(2 * tol, 0.05 * dx)
    near_wall_tol_y = max(2 * tol, 0.05 * dy)

    # Per-side flags. Only `:wall` sides zero the wall-normal momentum slab —
    # `:inlet` must preserve its prescribed inflow velocity, and `:outlet`
    # particles are culled in this same pass so their velocity is immaterial.
    wall_left  = sides[1] === :wall
    wall_right = sides[2] === :wall
    wall_bot   = sides[3] === :wall
    wall_top   = sides[4] === :wall

    # `cull_outlets=false` is used inside the step_co_flip! fixed-point loop
    # so the particle array doesn't shrink mid-iter (the advection buffers
    # x_n, P11..P22, etc. are sized to the initial num_p; shrinking p.x
    # under them would corrupt long-term pullback bookkeeping). The outer
    # post-iter sort and the per-step rebalance both cull as normal.
    out_left   = cull_outlets && sides[1] === :outlet
    out_right  = cull_outlets && sides[2] === :outlet
    out_bot    = cull_outlets && sides[3] === :outlet
    out_top    = cull_outlets && sides[4] === :outlet
    any_outlet = out_left | out_right | out_bot | out_top

    n_particles = length(p.x)
    # `trues` allocates a BitVector (~n/8 bytes); only paid when outlets exist
    # in the BC. For non-outlet configs the sorter is bit-identical to before.
    keep_mask = any_outlet ? trues(n_particles) : Bool[]

    @inbounds for i in 1:n_particles
        # Outlet culling — done on the raw post-advection position, before
        # wrap/clamp, so a particle that drifted past an outlet edge is
        # recognised as exited rather than clamped back inside the domain.
        if any_outlet
            if (out_left  && p.x[i] <= 0.0) || (out_right && p.x[i] >= Lx) ||
               (out_bot   && p.y[i] <= 0.0) || (out_top   && p.y[i] >= Ly)
                keep_mask[i] = false
                continue
            end
        end

        p.x[i] = wrap_axis(p.x[i], Lx, modes[1], tol)
        p.y[i] = wrap_axis(p.y[i], Ly, modes[2], tol)

        if (wall_left && p.x[i] <= near_wall_tol_x) ||
           (wall_right && p.x[i] >= Lx - near_wall_tol_x)
            p.mx[i] = 0.0
        end
        if (wall_bot && p.y[i] <= near_wall_tol_y) ||
           (wall_top && p.y[i] >= Ly - near_wall_tol_y)
            p.my[i] = 0.0
        end

        ei = clamp(floor(Int, p.x[i] * inv_dx) + 1, 1, nx)
        ej = clamp(floor(Int, p.y[i] * inv_dy) + 1, 1, ny)
        eid = (ej - 1) * nx + ei
        p.elem_ids[i] = eid

        p.can_x[i] = (p.x[i] - (ei - 1) * dx) * inv_dx
        p.can_y[i] = (p.y[i] - (ej - 1) * dy) * inv_dy

        p.next[i]   = p.head[eid]
        p.head[eid] = i
    end

    if any_outlet
        any_dead = false
        @inbounds for i in 1:n_particles
            if !keep_mask[i]
                any_dead = true
                break
            end
        end
        if any_dead
            compact_particles_inplace!(p, keep_mask, n_particles)
            # head/next are stale after compaction (some indices removed);
            # rebuild from the surviving particles' preserved elem_ids.
            fill!(p.head, 0)
            @inbounds for i in 1:length(p.x)
                eid = p.elem_ids[i]
                p.next[i] = p.head[eid]
                p.head[eid] = i
            end
        end
    end

    return nothing
end

"""Set particle position and velocity by sampling grid 1-form."""
function set_g2p_velocity(
        p::Particles, pid::Int,
        px::Float64, py::Float64, eid::Int,
        x0::Float64, y0::Float64, dx::Float64, dy::Float64,
        ref_phys_form, Lx::Float64, Ly::Float64;
        modes::NTuple{2,Symbol}=(:periodic, :periodic),
    )
    pxw = wrap_axis(px, Lx, modes[1], 1e-12)
    pyw = wrap_axis(py, Ly, modes[2], 1e-12)

    xi  = (pxw - x0) / dx
    eta = (pyw - y0) / dy
    sample_points = Mantis.Points.CartesianPoints(([xi], [eta]))
    pushfwd_eval, _ = Forms.evaluate_sharp_pushforward(ref_phys_form, eid, sample_points)

    p.x[pid]  = pxw
    p.y[pid]  = pyw
    p.mx[pid] = reduce(+, pushfwd_eval[1], dims=2)[1]
    p.my[pid] = reduce(+, pushfwd_eval[2], dims=2)[1]

    return nothing
end

"""
Physically reorder particle arrays so particles in the same element are contiguous in memory.
This significantly improves cache locality during RK advection loops by ensuring threads
access the same grid coefficients repeatedly rather than scattered lookups.
Call this periodically (e.g., every few timesteps) for maximum benefit.
"""
function physical_spatial_sort!(p::Particles, d::Domain)
    num_p    = length(p.x)
    num_elem = prod(d.nel)
    
    # Create a permutation array based on element IDs
    # Elements are already computed in particle_sorter
    perm = sortperm(p.elem_ids[1:num_p])
    
    # Create temporary storage for reordered data
    x_new     = similar(p.x, num_p)
    y_new     = similar(p.y, num_p)
    mx_new    = similar(p.mx, num_p)
    my_new    = similar(p.my, num_p)
    vol_new   = similar(p.volume, num_p)
    can_x_new = similar(p.can_x, num_p)
    can_y_new = similar(p.can_y, num_p)
    P11_new   = similar(p.P11, num_p)
    P12_new   = similar(p.P12, num_p)
    P21_new   = similar(p.P21, num_p)
    P22_new   = similar(p.P22, num_p)
    elem_ids_new = similar(p.elem_ids, num_p)
    
    # Reorder all particle data according to element locality
    @inbounds for i in 1:num_p
        old_idx = perm[i]
        x_new[i]     = p.x[old_idx]
        y_new[i]     = p.y[old_idx]
        mx_new[i]    = p.mx[old_idx]
        my_new[i]    = p.my[old_idx]
        vol_new[i]   = p.volume[old_idx]
        can_x_new[i] = p.can_x[old_idx]
        can_y_new[i] = p.can_y[old_idx]
        P11_new[i]   = p.P11[old_idx]
        P12_new[i]   = p.P12[old_idx]
        P21_new[i]   = p.P21[old_idx]
        P22_new[i]   = p.P22[old_idx]
        elem_ids_new[i] = p.elem_ids[old_idx]
    end
    
    # Copy reordered data back into particle arrays
    copyto!(p.x, 1, x_new, 1, num_p)
    copyto!(p.y, 1, y_new, 1, num_p)
    copyto!(p.mx, 1, mx_new, 1, num_p)
    copyto!(p.my, 1, my_new, 1, num_p)
    copyto!(p.volume, 1, vol_new, 1, num_p)
    copyto!(p.can_x, 1, can_x_new, 1, num_p)
    copyto!(p.can_y, 1, can_y_new, 1, num_p)
    copyto!(p.P11, 1, P11_new, 1, num_p)
    copyto!(p.P12, 1, P12_new, 1, num_p)
    copyto!(p.P21, 1, P21_new, 1, num_p)
    copyto!(p.P22, 1, P22_new, 1, num_p)
    copyto!(p.elem_ids, 1, elem_ids_new, 1, num_p)
    
    # After physical reordering, rebuild the head/next linked list structure
    particle_sorter!(p, d)
    
    return nothing
end

"""Spawn and cull particles to maintain per-element and per-quarter count bounds."""
function enforce_min_particles_per_element!(
        p::Particles, d::Domain,
        min_particles_per_element::Int,
        max_particles_per_element::Int,
        min_particles_per_quarter::Int,
        ref_grid_coeffs::AbstractVector{T},
        buf::SimulationBuffers,
    ) where {T<:Real}

    if min_particles_per_element <= 0 && max_particles_per_element <= 0 && min_particles_per_quarter <= 0
        return 0, 0
    end
    if max_particles_per_element > 0 && min_particles_per_element > max_particles_per_element
        throw(ArgumentError("min_particles_per_element must be <= max_particles_per_element"))
    end
    if max_particles_per_element > 0 && min_particles_per_quarter > 0 &&
            max_particles_per_element < 4 * min_particles_per_quarter
        throw(ArgumentError("max_particles_per_element must be >= 4 * min_particles_per_quarter"))
    end

    particle_sorter!(p, d)

    num_elements       = prod(d.nel)
    min_elem_effective = max(min_particles_per_element, 4 * max(min_particles_per_quarter, 0))

    counts_elem    = buf.counts_elem
    counts_quarter = buf.counts_quarter
    add_quarter    = buf.add_quarter

    fill!(counts_elem,    0)
    fill!(counts_quarter, 0)
    fill!(add_quarter,    0)

    @inbounds for pid in eachindex(p.x)
        eid = p.elem_ids[pid]
        qx  = p.can_x[pid] >= 0.5 ? 2 : 1
        qy  = p.can_y[pid] >= 0.5 ? 1 : 0
        qid = qx + 2 * qy
        counts_elem[eid]         += 1
        counts_quarter[eid, qid] += 1
    end

    if min_particles_per_quarter > 0
        @inbounds for eid in 1:num_elements
            for qid in 1:4
                deficit = min_particles_per_quarter - counts_quarter[eid, qid]
                if deficit > 0
                    add_quarter[eid, qid] += deficit
                end
            end
        end
    end

    if min_elem_effective > 0
        @inbounds for eid in 1:num_elements
            base_after_quarter = counts_elem[eid] + sum(@view add_quarter[eid, :])
            extra_needed = min_elem_effective - base_after_quarter
            while extra_needed > 0
                qbest = 1
                cbest = counts_quarter[eid, 1] + add_quarter[eid, 1]
                for qid in 2:4
                    c = counts_quarter[eid, qid] + add_quarter[eid, qid]
                    if c < cbest; cbest = c; qbest = qid; end
                end
                add_quarter[eid, qbest] += 1
                extra_needed -= 1
            end
        end
    end

    total_to_add = sum(add_quarter)

    if total_to_add == 0 && max_particles_per_element <= 0
        rebind_volumes_per_element!(p, d, counts_elem)
        return 0, 0
    end

    nx, ny = d.nel
    Lx, Ly = d.box_size
    dx = Lx / nx;  dy = Ly / ny
    rng = Random.default_rng()

    if total_to_add > 0
        ref_coeffs_f64 = collect(Float64, ref_grid_coeffs)
        ref_form       = Forms.build_form_field(d.R1, ref_coeffs_f64)
        ref_phys_expr  = ★(ref_form)
        ref_phys_form  = Assemblers.solve_L2_projection(d.R1, ref_phys_expr, d.dΩ)

        old_count = length(p.x)
        new_count = old_count + total_to_add

        resize!(p.x,       new_count)
        resize!(p.y,       new_count)
        resize!(p.mx,      new_count)
        resize!(p.my,      new_count)
        resize!(p.volume,  new_count)
        resize!(p.can_x,   new_count)
        resize!(p.can_y,   new_count)
        resize!(p.next,    new_count)
        resize!(p.elem_ids, new_count)
        resize!(p.P11,     new_count)
        resize!(p.P12,     new_count)
        resize!(p.P21,     new_count)
        resize!(p.P22,     new_count)
        resize!(p.delta_t, new_count)

        ensure_particle_capacity!(buf, new_count)

        pid = old_count + 1
        @inbounds for eid in 1:num_elements
            ej = ((eid - 1) ÷ nx) + 1
            ei = eid - (ej - 1) * nx
            x0 = (ei - 1) * dx
            y0 = (ej - 1) * dy

            for qid in 1:4
                n_add_q = add_quarter[eid, qid]
                n_add_q <= 0 && continue

                qx  = ((qid - 1) % 2)
                qy  = ((qid - 1) ÷ 2)
                qx0 = x0 + qx * 0.5 * dx
                qy0 = y0 + qy * 0.5 * dy
                qdx = 0.5 * dx
                qdy = 0.5 * dy

                modes = bc_to_position_modes(d.bc_sides)
                for _ in 1:n_add_q
                    px = qx0 + rand(rng) * qdx
                    py = qy0 + rand(rng) * qdy
                    set_g2p_velocity(p, pid, px, py, eid, x0, y0, dx, dy, ref_phys_form, Lx, Ly; modes=modes)
                    p.P11[pid]     = 1.0
                    p.P12[pid]     = 0.0
                    p.P21[pid]     = 0.0
                    p.P22[pid]     = 1.0
                    p.delta_t[pid] = 0.0
                    pid += 1
                end
            end
        end

        particle_sorter!(p, d)
    end

    total_to_remove = 0
    if max_particles_per_element > 0 && length(p.x) > 0
        n_particles = length(p.x)

        ensure_particle_capacity!(buf, n_particles)
        keep_mask    = buf.keep_mask
        quarter_pids = buf.quarter_pids

        fill!(view(keep_mask, 1:n_particles), true)
        @inbounds for s in eachindex(quarter_pids); empty!(quarter_pids[s]); end

        fill!(counts_elem,    0)
        fill!(counts_quarter, 0)

        @inbounds for pid in 1:n_particles
            eid = p.elem_ids[pid]
            qx  = p.can_x[pid] >= 0.5 ? 2 : 1
            qy  = p.can_y[pid] >= 0.5 ? 1 : 0
            qid = qx + 2 * qy
            counts_elem[eid]         += 1
            counts_quarter[eid, qid] += 1
            push!(quarter_pids[(eid - 1) * 4 + qid], pid)
        end

        @inbounds for eid in 1:num_elements
            local_excess = counts_elem[eid] - max_particles_per_element
            while local_excess > 0
                qbest       = 0
                excess_best = 0
                for qid in 1:4
                    removable = counts_quarter[eid, qid] - max(min_particles_per_quarter, 0)
                    if removable > excess_best
                        excess_best = removable
                        qbest = qid
                    end
                end
                qbest == 0 && break

                qlist = quarter_pids[(eid - 1) * 4 + qbest]
                while !isempty(qlist)
                    victim = pop!(qlist)
                    if keep_mask[victim]
                        keep_mask[victim]            = false
                        counts_quarter[eid, qbest]  -= 1
                        counts_elem[eid]            -= 1
                        total_to_remove             += 1
                        local_excess                -= 1
                        break
                    end
                end
            end
        end

        if total_to_remove > 0
            compact_particles_inplace!(p, keep_mask, n_particles)
            particle_sorter!(p, d)
        end
    end

    fill!(counts_elem, 0)
    @inbounds for pid in 1:length(p.x)
        counts_elem[p.elem_ids[pid]] += 1
    end
    rebind_volumes_per_element!(p, d, counts_elem)

    return total_to_add, total_to_remove
end

"""Compact particle arrays in-place using keep mask."""
function compact_particles_inplace!(p::Particles, keep_mask::AbstractVector{Bool}, n::Int)
    dst = 0
    @inbounds for src in 1:n
        keep_mask[src] || continue
        dst += 1
        if dst != src
            p.x[dst]        = p.x[src]
            p.y[dst]        = p.y[src]
            p.mx[dst]       = p.mx[src]
            p.my[dst]       = p.my[src]
            p.volume[dst]   = p.volume[src]
            p.can_x[dst]    = p.can_x[src]
            p.can_y[dst]    = p.can_y[src]
            p.next[dst]     = p.next[src]
            p.elem_ids[dst] = p.elem_ids[src]
            p.P11[dst]      = p.P11[src]
            p.P12[dst]      = p.P12[src]
            p.P21[dst]      = p.P21[src]
            p.P22[dst]      = p.P22[src]
            p.delta_t[dst]  = p.delta_t[src]
        end
    end
    resize!(p.x,        dst)
    resize!(p.y,        dst)
    resize!(p.mx,       dst)
    resize!(p.my,       dst)
    resize!(p.volume,   dst)
    resize!(p.can_x,    dst)
    resize!(p.can_y,    dst)
    resize!(p.next,     dst)
    resize!(p.elem_ids, dst)
    resize!(p.P11,      dst)
    resize!(p.P12,      dst)
    resize!(p.P21,      dst)
    resize!(p.P22,      dst)
    resize!(p.delta_t,  dst)
    return nothing
end

"""Distribute cell volume equally among particles in each element."""
function rebind_volumes_per_element!(p::Particles, d::Domain, counts_elem::AbstractVector{Int})
    nx, ny = d.nel
    Lx, Ly = d.box_size
    cell_area = (Lx / nx) * (Ly / ny)
    @inbounds for pid in 1:length(p.x)
        c = counts_elem[p.elem_ids[pid]]
        p.volume[pid] = c > 0 ? cell_area / c : cell_area
    end
    return nothing
end

"""Expand scratch buffers to accommodate at least n particles."""
function ensure_particle_capacity!(buf::SimulationBuffers, n::Int)
    if length(buf.x_n) >= n
        return nothing
    end
    new_cap = max(n, length(buf.x_n) * 2)
    for arr in (buf.x_n, buf.y_n, buf.mx_n, buf.my_n,
                buf.x_np1, buf.y_np1, buf.mx_np1, buf.my_np1,
                buf.short_P11, buf.short_P12, buf.short_P21, buf.short_P22,
                buf.keep_mask)
        resize!(arr, new_cap)
    end
    if length(buf.V_p) < 2 * new_cap
        resize!(buf.V_p, 2 * new_cap)
    end
    return nothing
end

"""
Per-side analytical inlet velocity, pointing into the domain along the
inward normal of side `s` with magnitude `U`. Returns `(0.0, 0.0)` when
the side is not an inlet.
"""
@inline function inlet_inward_velocity(sides::NTuple{4,Symbol}, s::Int, U::Float64)
    sides[s] === :inlet || return (0.0, 0.0)
    return s == 1 ? ( U, 0.0) :       # left  → +x
           s == 2 ? (-U, 0.0) :       # right → -x
           s == 3 ? ( 0.0,  U) :      # bot   → +y
                    ( 0.0, -U)        # top   → -y
end

"""Stamp one freshly-injected inlet particle into slot `pid` at random
sub-position within element `eid` (which spans `[x0, x0+dx] × [y0, y0+dy]`).
Identity pullback, zero accumulated `delta_t`, volume = cell area (rebound
to per-particle later by the rebalance pass)."""
@inline function _seed_inlet_particle!(
        p::Particles, pid::Int, eid::Int,
        x0::Float64, y0::Float64, dx::Float64, dy::Float64,
        u_in::Float64, v_in::Float64, cell_area::Float64, rng,
    )
    px = x0 + rand(rng) * dx
    py = y0 + rand(rng) * dy
    @inbounds begin
        p.x[pid]        = px
        p.y[pid]        = py
        p.mx[pid]       = u_in
        p.my[pid]       = v_in
        p.volume[pid]   = cell_area
        p.can_x[pid]    = (px - x0) / dx
        p.can_y[pid]    = (py - y0) / dy
        p.elem_ids[pid] = eid
        p.P11[pid]      = 1.0
        p.P12[pid]      = 0.0
        p.P21[pid]      = 0.0
        p.P22[pid]      = 1.0
        p.delta_t[pid]  = 0.0
    end
    return nothing
end

"""
Replenish and re-stamp particles in elements adjacent to any `:inlet` side
so the prescribed inflow is maintained throughout the run:

1. Patch the velocity of every particle currently in an inlet-row element
   to the analytical inlet value (independent of any L2-projection smearing
   in the grid representation that would otherwise feed velocity through
   `set_g2p_velocity`).
2. Where an inlet element holds fewer than `cfg.particles_per_cell`
   particles, seed new ones at uniform random sub-positions, give them the
   analytical inlet velocity, and start them with identity pullback.

No-op for configs with no inlet sides or `cfg.inlet_U_inf == 0`. Call once
per outer step *before* the fixed-point loop so the inflow is established
before P2G.
"""
function enforce_inlet_particles!(
        p::Particles, d::Domain, buf::SimulationBuffers, cfg::SimulationConfig,
    )
    sides = d.bc_sides
    has_inlet = sides[1] === :inlet || sides[2] === :inlet ||
                sides[3] === :inlet || sides[4] === :inlet
    has_inlet || return 0
    cfg.inlet_U_inf > 0.0 || return 0

    target = max(cfg.particles_per_cell, cfg.min_particles_per_element)
    target > 0 || return 0

    nx, ny = d.nel
    Lx, Ly = d.box_size
    dx = Lx / nx;  dy = Ly / ny
    cell_area = dx * dy
    U = cfg.inlet_U_inf

    in_left  = sides[1] === :inlet
    in_right = sides[2] === :inlet
    in_bot   = sides[3] === :inlet
    in_top   = sides[4] === :inlet

    particle_sorter!(p, d)

    # Pass 1: tally per-element counts and patch velocities of existing
    # particles in inlet-row elements (analytical inlet velocity).
    counts_elem = buf.counts_elem
    fill!(counts_elem, 0)
    @inbounds for pid in 1:length(p.x)
        eid = p.elem_ids[pid]
        counts_elem[eid] += 1
        ej = (eid - 1) ÷ nx + 1
        ei = eid - (ej - 1) * nx
        if in_left && ei == 1
            p.mx[pid] = U;    p.my[pid] = 0.0
        elseif in_right && ei == nx
            p.mx[pid] = -U;   p.my[pid] = 0.0
        elseif in_bot && ej == 1
            p.mx[pid] = 0.0;  p.my[pid] = U
        elseif in_top && ej == ny
            p.mx[pid] = 0.0;  p.my[pid] = -U
        end
    end

    # Pass 2: count how many fresh particles to inject across all inlet rows.
    deficit_total = 0
    if in_left
        @inbounds for ej in 1:ny
            deficit_total += max(0, target - counts_elem[(ej - 1) * nx + 1])
        end
    end
    if in_right
        @inbounds for ej in 1:ny
            deficit_total += max(0, target - counts_elem[(ej - 1) * nx + nx])
        end
    end
    if in_bot
        @inbounds for ei in 1:nx
            deficit_total += max(0, target - counts_elem[ei])
        end
    end
    if in_top
        @inbounds for ei in 1:nx
            deficit_total += max(0, target - counts_elem[(ny - 1) * nx + ei])
        end
    end
    deficit_total > 0 || return 0

    # Pass 3: allocate slots and inject.
    old_count = length(p.x)
    new_count = old_count + deficit_total

    resize!(p.x,        new_count)
    resize!(p.y,        new_count)
    resize!(p.mx,       new_count)
    resize!(p.my,       new_count)
    resize!(p.volume,   new_count)
    resize!(p.can_x,    new_count)
    resize!(p.can_y,    new_count)
    resize!(p.next,     new_count)
    resize!(p.elem_ids, new_count)
    resize!(p.P11,      new_count)
    resize!(p.P12,      new_count)
    resize!(p.P21,      new_count)
    resize!(p.P22,      new_count)
    resize!(p.delta_t,  new_count)
    ensure_particle_capacity!(buf, new_count)

    rng = Random.default_rng()
    pid = old_count + 1

    if in_left
        u_in, v_in = inlet_inward_velocity(sides, 1, U)
        @inbounds for ej in 1:ny
            eid = (ej - 1) * nx + 1
            deficit = target - counts_elem[eid]
            deficit > 0 || continue
            x0 = 0.0
            y0 = (ej - 1) * dy
            for _ in 1:deficit
                _seed_inlet_particle!(p, pid, eid, x0, y0, dx, dy,
                                      u_in, v_in, cell_area, rng)
                pid += 1
            end
        end
    end
    if in_right
        u_in, v_in = inlet_inward_velocity(sides, 2, U)
        @inbounds for ej in 1:ny
            eid = (ej - 1) * nx + nx
            deficit = target - counts_elem[eid]
            deficit > 0 || continue
            x0 = (nx - 1) * dx
            y0 = (ej - 1) * dy
            for _ in 1:deficit
                _seed_inlet_particle!(p, pid, eid, x0, y0, dx, dy,
                                      u_in, v_in, cell_area, rng)
                pid += 1
            end
        end
    end
    if in_bot
        u_in, v_in = inlet_inward_velocity(sides, 3, U)
        @inbounds for ei in 1:nx
            eid = ei
            deficit = target - counts_elem[eid]
            deficit > 0 || continue
            x0 = (ei - 1) * dx
            y0 = 0.0
            for _ in 1:deficit
                _seed_inlet_particle!(p, pid, eid, x0, y0, dx, dy,
                                      u_in, v_in, cell_area, rng)
                pid += 1
            end
        end
    end
    if in_top
        u_in, v_in = inlet_inward_velocity(sides, 4, U)
        @inbounds for ei in 1:nx
            eid = (ny - 1) * nx + ei
            deficit = target - counts_elem[eid]
            deficit > 0 || continue
            x0 = (ei - 1) * dx
            y0 = (ny - 1) * dy
            for _ in 1:deficit
                _seed_inlet_particle!(p, pid, eid, x0, y0, dx, dy,
                                      u_in, v_in, cell_area, rng)
                pid += 1
            end
        end
    end

    particle_sorter!(p, d)
    return deficit_total
end

"""
Immersed-boundary penalisation (Brinkman, τ→0 limit) for any obstacle. For
each particle currently inside the obstacle, zero its velocity **in place**
(position is left unchanged). LSQR then sees a smooth distribution of
v=0 samples filling the cylinder interior, and fits a clean U∞-to-0
transition over ~1-2 cells at the boundary.

We *don't* project the position onto a thin offset annulus outside the
surface. That formulation concentrates every interior particle into a
band of width ~1e-4·rc; after a handful of steps tens of thousands of
zero-velocity particles share that band, the per-step rebalance churns
whole-population fractions just from the resulting density spike, and the
LSQR B-spline fit develops singular gradients near the surface that
amplify under advection (10^19 grid-vel errors observed in step 1 even
with no operator-level cylinder pinning). In-place zeroing keeps the
distribution diffuse and the gradients bounded.

Returns the number of particles modified. No-op when `d.obstacle === nothing`.
"""
function apply_obstacle_brinkman!(p::Particles, d::Domain)
    obs = d.obstacle
    obs === nothing && return 0
    obs.kind === :cylinder || return 0

    cx, cy = obs.center
    rc     = obs.radius
    rc2    = rc * rc

    damped = 0
    @inbounds for pid in 1:length(p.x)
        dx = p.x[pid] - cx
        dy = p.y[pid] - cy
        r2 = dx * dx + dy * dy
        if r2 < rc2
            p.mx[pid] = 0.0
            p.my[pid] = 0.0
            damped   += 1
        end
    end
    return damped
end

"""
Brinkman-style tangential-velocity enforcement for a lid-driven cavity.

Pins the particle velocity to `(U_lid, 0)` in a band of thickness
`lid_band_cells*dy` immediately below the top wall, and to `(0, 0)` in
bands of thickness `wall_band_cells * dx (or dy)` along the bottom, left,
and right walls. The `:wall` BC already pins the wall-normal trace
`u·n = 0` strongly via DOF zeroing, so this hook only adds the
*tangential* component the operator doesn't constrain — a Brinkman-style
approximation of full no-slip + moving lid. Apply once per step (after
`step_co_flip!`) so the next step's P2G sees the prescribed boundary
tangential velocity.

Why the lid band is wider than the wall bands: the LSQR particle→grid
fit gives the top R1 coefficient roughly the *fraction* of in-support
particles that are stamped. For p=3 splines the boundary DOF has support
over ~p+1=4 cells; a 1-cell band leaves ~75% of those particles unmarked
and the LSQR fit therefore reports a top u of only ≈ 0.25·U_lid. Setting
`lid_band_cells ≈ p+1` lets the fit converge to U_lid. The wall bands
are kept thin to avoid artificially thickening the boundary layer.
"""
function apply_lid_cavity_brinkman!(p::Particles, d::Domain, U_lid::Float64;
        lid_band_cells::Float64=4.0,
        wall_band_cells::Float64=1.0,
        # Back-compat: callers passing the old `band_cells` kwarg get the
        # same value applied to both the lid and the walls.
        band_cells::Union{Nothing,Float64}=nothing,
    )
    if band_cells !== nothing
        lid_band_cells  = band_cells
        wall_band_cells = band_cells
    end
    nx, ny = d.nel
    Lx, Ly = d.box_size
    dx = Lx / nx;  dy = Ly / ny
    δy_lid  = lid_band_cells  * dy
    δy_wall = wall_band_cells * dy
    δx_wall = wall_band_cells * dx
    pinned = 0
    @inbounds for pid in 1:length(p.x)
        x = p.x[pid];  y = p.y[pid]
        if y >= Ly - δy_lid
            p.mx[pid] = U_lid
            p.my[pid] = 0.0
            pinned += 1
        elseif y <= δy_wall
            p.mx[pid] = 0.0
            p.my[pid] = 0.0
            pinned += 1
        elseif x <= δx_wall || x >= Lx - δx_wall
            p.mx[pid] = 0.0
            p.my[pid] = 0.0
            pinned += 1
        end
    end
    return pinned
end

"""Taylor-Green vortex on periodic domain."""
function flow_taylor_green(x::Float64, y::Float64, Lx::Float64, Ly::Float64)
    U0 = 1.0
    kx = 2*π * x / Lx
    ky = 2*π * y / Ly
    u =  U0 * sin(kx) * cos(ky)
    v = -U0 * cos(kx) * sin(ky)
    return u, v
end

"""Initial condition for the decaying (viscous) Taylor–Green vortex.

t=0 snapshot of the analytical Navier–Stokes solution
  u(x,y,t) =  U₀·sin(kx·x)·cos(ky·y)·exp(-ν·(kx²+ky²)·t)
  v(x,y,t) = -U₀·cos(kx·x)·sin(ky·y)·exp(-ν·(kx²+ky²)·t)
Energy decays as E(t) = E₀·exp(-2ν·(kx²+ky²)·t); use as validation when
cfg.viscosity > 0. Requires periodic boundaries."""
function flow_decaying_tg(x::Float64, y::Float64, Lx::Float64, Ly::Float64)
    return flow_taylor_green(x, y, Lx, Ly)
end

"""Lamb-Oseen vortex centered in domain."""
function flow_lamb_oseen(x::Float64, y::Float64, Lx::Float64, Ly::Float64)
    cx = Lx / 2.0;  cy = Ly / 2.0
    Gamma = 2.0
    r_core = min(Lx, Ly) / 8.0

    dx = x - cx;  dy = y - cy
    r2 = dx^2 + dy^2
    r  = sqrt(r2)

    if r < 1e-8; return 0.0, 0.0; end

    v_theta = (Gamma / (2 * π * r)) * (1.0 - exp(-r2 / (r_core^2)))
    u = -v_theta * (dy / r)
    v =  v_theta * (dx / r)
    return u, v
end

"""Double-gyre flow field."""
function flow_double_gyre(x::Float64, y::Float64, Lx::Float64, Ly::Float64)
    U0 = 1.0
    X = x / Lx;  Y = y / Ly
    u =  U0 * sin(π * X) * cos(π * Y)
    v = -U0 * cos(π * X) * sin(π * Y)
    return u, v
end

"""Compactly-supported decaying vortex."""
function flow_decay(x::Float64, y::Float64, Lx::Float64, Ly::Float64)
    cx = Lx / 2.0;  cy = Ly / 2.0
    Gamma = 10.0
    r_core = min(Lx, Ly) / 6.0

    dx = x - cx;  dy = y - cy
    r2 = dx^2 + dy^2
    r  = sqrt(r2)

    if r < 1e-8; return 0.0, 0.0; end

    sin_theta = dy/r
    cos_theta = dx/r
    u = -sin_theta*exp(-r2/(0.2*r_core^2))
    v =  cos_theta*exp(-r2/(0.2*r_core^2))
    return u, v
end

"""Gaussian vortex with uniform advection."""
function flow_convecting_vortex(x::Float64, y::Float64, Lx::Float64, Ly::Float64)
    U0 = 1.0
    Γ  = 5.0
    σ  = 0.1 * min(Lx, Ly)
    x0 = 0.5 * Lx;  y0 = 0.5 * Ly

    dx = x - x0;  dy = y - y0
    r2 = dx^2 + dy^2

    factor = (Γ / (2π)) * exp(-r2 / σ^2)
    u_vortex = -factor * dy / σ^2
    v_vortex =  factor * dx / σ^2

    u = u_vortex + U0
    v = v_vortex
    return u, v
end

"""Two co-rotating vortices configured to merge."""
function flow_merging_vortices(x::Float64, y::Float64, Lx::Float64, Ly::Float64)
    cx = Lx / 2.0;  cy = Ly / 2.0
    Gamma = 3.0
    r_core = min(Lx, Ly) / 15.0
    separation = r_core * 3.5

    cx1 = cx - separation / 2.0;  cy1 = cy
    cx2 = cx + separation / 2.0;  cy2 = cy

    dx1 = x - cx1;  dy1 = y - cy1
    r1_2 = dx1^2 + dy1^2;  r1 = sqrt(r1_2)
    dx2 = x - cx2;  dy2 = y - cy2
    r2_2 = dx2^2 + dy2^2;  r2 = sqrt(r2_2)

    u = 0.0;  v = 0.0
    if r1 > 1e-8
        v_theta1 = (Gamma / (2 * π * r1)) * (1.0 - exp(-r1_2 / (r_core^2)))
        u += -v_theta1 * (dy1 / r1)
        v +=  v_theta1 * (dx1 / r1)
    end
    if r2 > 1e-8
        v_theta2 = (Gamma / (2 * π * r2)) * (1.0 - exp(-r2_2 / (r_core^2)))
        u += -v_theta2 * (dy2 / r2)
        v +=  v_theta2 * (dx2 / r2)
    end
    return u, v
end

"""Uniform translating flow — sanity check for advection and projection bugs."""
function flow_uniform(x::Float64, y::Float64, Lx::Float64, Ly::Float64)
    _ = (x, y, Lx, Ly)
    return 1.0, 0.0
end

"""Zero velocity field. Used as the initial condition for cases where the
flow is driven entirely by boundary conditions (lid-driven cavity)."""
function flow_zero(x::Float64, y::Float64, Lx::Float64, Ly::Float64)
    _ = (x, y, Lx, Ly)
    return 0.0, 0.0
end

"""Initial condition for flow past an obstacle: uniform far-field
`(U_inf, 0)` outside the obstacle, zero velocity inside. Used together with
the immersed-boundary penalisation (Step 5) to bootstrap a flow-past-cylinder
simulation. `obstacle === nothing` falls back to a pure uniform field."""
function flow_cylinder(
        x::Float64, y::Float64, Lx::Float64, Ly::Float64,
        obstacle, U_inf::Float64,
    )
    _ = (Lx, Ly)
    point_inside_obstacle(x, y, obstacle) && return 0.0, 0.0
    return U_inf, 0.0
end

"""Bell-Colella-Glaz double shear layer — vortex roll-up benchmark on periodic box."""
function flow_shear(x::Float64, y::Float64, Lx::Float64, Ly::Float64)
    ρ = 30.0
    δ = 0.05
    yh = y / Ly
    u = (yh ≤ 0.5) ? tanh(ρ * (yh - 0.25)) : tanh(ρ * (0.75 - yh))
    v = δ * sin(2π * x / Lx)
    return u, v
end

"""Kelvin-Helmholtz: two smoothed anti-parallel shear layers + single-mode perturbation."""
function flow_kelvin_helmholtz(x::Float64, y::Float64, Lx::Float64, Ly::Float64)
    U0 = 1.0
    δ  = 0.025 * Ly
    A  = 0.01 * U0
    base_u = (y ≤ 0.5*Ly) ?  U0 * tanh((y - 0.25*Ly) / δ) :
                             U0 * tanh((0.75*Ly - y) / δ)
    u = base_u
    v = A * sin(2π * x / Lx)
    return u, v
end

"""Counter-rotating Lamb-Oseen pair forming a self-propagating dipole (translates in +x)."""
function flow_dipole(x::Float64, y::Float64, Lx::Float64, Ly::Float64)
    cx = Lx / 4.0;  cy = Ly / 2.0
    Γ = 3.0
    r_core = min(Lx, Ly) / 25.0
    sep = min(Lx, Ly) / 10.0

    centers = ((cx, cy + sep/2, +Γ),
               (cx, cy - sep/2, -Γ))

    u = 0.0;  v = 0.0
    for (xc, yc, Γi) in centers
        dx = x - xc;  dy = y - yc
        r2 = dx^2 + dy^2;  r = sqrt(r2)
        if r > 1e-8
            v_theta = (Γi / (2π * r)) * (1.0 - exp(-r2 / r_core^2))
            u += -v_theta * dy / r
            v +=  v_theta * dx / r
        end
    end
    return u, v
end

"""Velocity field induced at (x, y) by a single Gaussian vortex blob centered
at (xc, yc) with circulation Γ and core radius `r_core` (Gaussian σ on the
vorticity profile)."""
@inline function _gauss_vortex_velocity(x::Float64, y::Float64,
                                       xc::Float64, yc::Float64,
                                       Γ::Float64, r_core::Float64)
    dx = x - xc;  dy = y - yc
    r2 = dx^2 + dy^2
    r  = sqrt(r2)
    r > 1e-8 || return 0.0, 0.0
    v_theta = (Γ / (2π * r)) * (1.0 - exp(-r2 / r_core^2))
    return -v_theta * dy / r, v_theta * dx / r
end

"""
Leapfrog: four coaxial vortices in two stacked dipoles that leapfrog through
each other (mirrors C++ COFLIPSolver2D::sampleLeapfrog). Defaults match the
C++ leapfrog experiment on an L=2π box: vortex y-position Ly/4, separations
0.239·Lx and 0.477·Lx (= 1.5 and 3.0 for Lx=2π).

When `bc` declares any axis as a wall, image vortices are added across each
wall (with sign-flipped circulation) so the resulting velocity satisfies
u·n=0 on those walls — same free-slip / no-penetration outcome as the C++
streamfunction-Dirichlet-on-wall construction, just expressed analytically
via the method of images. `n_images` controls how many image rings are
summed (1 ≈ exact for vortices far from walls)."""
function flow_leapfrog(x::Float64, y::Float64, Lx::Float64, Ly::Float64;
        bc::NTuple{4,Symbol}=(:periodic, :periodic, :periodic, :periodic),
        n_images::Int=2,
    )
    # Real vortex centers + circulations. C++: dist_a=1.5, dist_b=3.0 on L=2π.
    cx = Lx / 2.0;  cy = Ly / 4.0
    Γ      = 2.0
    r_core = min(Lx, Ly) / 35.0
    dist_a = 0.2387324 * min(Lx, Ly)   # ≈ 1.5 / (2π)
    dist_b = 0.4774648 * min(Lx, Ly)   # ≈ 3.0 / (2π)
    real_vortices = ((cx - 0.5*dist_a, cy, +Γ),
                     (cx + 0.5*dist_a, cy, -Γ),
                     (cx - 0.5*dist_b, cy, +Γ),
                     (cx + 0.5*dist_b, cy, -Γ))

    # Image lattice keys on per-axis wall presence; the method-of-images
    # construction reflects across both sides of any non-periodic axis, so
    # per-axis collapse is the right summary here.
    axes_summary = bc_axes(bc)
    wall_x = axes_summary[1] === :wall
    wall_y = axes_summary[2] === :wall

    # Image lattice for free-slip on a rectangle: for each real vortex at
    # (xc, yc, Γ), the image positions are (sx·xc + 2k·Lx, sy·yc + 2m·Ly) with
    # sx, sy ∈ {+1, -1} and k, m ∈ ℤ, carrying circulation sx·sy·Γ. The
    # (sx=+1, sy=+1, k=m=0) entry is the real vortex itself. When an axis is
    # periodic instead of walled, we suppress its image factor by restricting
    # the corresponding sx (or sy) to +1 and k (or m) to 0.
    sx_choices = wall_x ? (1, -1) : (1,)
    sy_choices = wall_y ? (1, -1) : (1,)
    k_range    = wall_x ? (-n_images:n_images) : 0:0
    m_range    = wall_y ? (-n_images:n_images) : 0:0

    u = 0.0;  v = 0.0
    for (xc, yc, Γi) in real_vortices
        for sy in sy_choices, sx in sx_choices, m in m_range, k in k_range
            xim = sx * xc + 2 * k * Lx
            yim = sy * yc + 2 * m * Ly
            Γim = sx * sy * Γi
            du, dv = _gauss_vortex_velocity(x, y, xim, yim, Γim, r_core)
            u += du;  v += dv
        end
    end
    return u, v
end

"""Four alternating-sign Lamb-Oseen vortices in checkerboard square — symmetry-preservation test (net Γ = 0)."""
function flow_four_vortex(x::Float64, y::Float64, Lx::Float64, Ly::Float64)
    cx = Lx / 2.0;  cy = Ly / 2.0
    Γ = 2.0
    r_core = min(Lx, Ly) / 20.0
    s = min(Lx, Ly) / 6.0

    centers = ((cx - s, cy - s, +Γ),
               (cx + s, cy + s, +Γ),
               (cx + s, cy - s, -Γ),
               (cx - s, cy + s, -Γ))

    u = 0.0;  v = 0.0
    for (xc, yc, Γi) in centers
        dx = x - xc;  dy = y - yc
        r2 = dx^2 + dy^2;  r = sqrt(r2)
        if r > 1e-8
            v_theta = (Γi / (2π * r)) * (1.0 - exp(-r2 / r_core^2))
            u += -v_theta * dy / r
            v +=  v_theta * dx / r
        end
    end
    return u, v
end

"""Stuart vortex — exact traveling-wave solution of 2D Euler. Periodic in x; in y, decays to a uniform shear (use a tall enough box)."""
function flow_stuart(x::Float64, y::Float64, Lx::Float64, Ly::Float64)
    ρ  = 0.5
    cy = Ly / 2.0
    kx = 2π / Lx
    ky = kx
    tx = kx * x
    ty = ky * (y - cy)
    D  = cosh(ty) - ρ * cos(tx)
    u  = -sinh(ty) * ky / D
    v  =  ρ * sin(tx) * kx / D
    return u, v
end

"""Evaluate basis and derivatives at sample point with caching."""
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

"""
Allocation-free, thread-safe 2D fast-path evaluator that bypasses Mantis's allocating
`evaluate` chain for the CO-FLIP setup. Expects `space` to be a DirectSumSpace with 2
TensorProductSpace components, each having 2 1D GTBSplineSpace constituents (1 patch
each, single BSplineSpace inside). Writes into `cache.results` in the same layout as
`evaluate_fast!`.

Caller passes (xi, eta) ∈ [0,1] reference coords for the element directly — no
CartesianPoints wrapping. nderivatives ≤ 1 is supported in this implementation.
"""
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
        tp_space  = space.component_spaces[c]
        leaf_1    = tp_space.constituent_spaces[1]
        leaf_2    = tp_space.constituent_spaces[2]

        cart_idx = tp_space.cart_num_elements[element_id]
        e1 = cart_idx[1]
        e2 = cart_idx[2]

        # Periodic axes wrap a BSplineSpace inside a GTBSplineSpace (the GTB
        # extraction identifies endpoints). Non-periodic axes use the
        # BSplineSpace directly with no outer extraction layer. Detect both.
        is_gtb_1 = leaf_1 isa Mantis.FunctionSpaces.GTBSplineSpace
        is_gtb_2 = leaf_2 isa Mantis.FunctionSpaces.GTBSplineSpace

        bsp_space_1 = is_gtb_1 ? leaf_1.patch_spaces[1] : leaf_1
        bsp_space_2 = is_gtb_2 ? leaf_2.patch_spaces[1] : leaf_2

        bsp_ext_1 = bsp_space_1.extraction_op.extraction_coefficients[e1][1]
        bsp_ext_2 = bsp_space_2.extraction_op.extraction_coefficients[e2][1]

        polynomial_1 = bsp_space_1.polynomials
        polynomial_2 = bsp_space_2.polynomials
        p1 = polynomial_1.p
        p2 = polynomial_2.p
        nd1 = nderivatives + 1
        n_bsp_1 = size(bsp_ext_1, 2)
        n_bsp_2 = size(bsp_ext_2, 2)

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

        # When the axis is periodic, apply the GTBSpline extraction layer on top
        # to produce the final (n_gtb) basis values; otherwise the BSpline values
        # are themselves the final basis values.
        if is_gtb_1
            gtb_ext_1  = leaf_1.extraction_op.extraction_coefficients[e1][1]
            n_gtb_1    = size(gtb_ext_1, 2)
            gtb_vals_1 = @view cache.gtb_vals[1][1:nd1, 1:n_gtb_1]
            mul!(gtb_vals_1, bsp_vals_1, gtb_ext_1)
        else
            n_gtb_1    = n_bsp_1
            gtb_vals_1 = bsp_vals_1
        end
        if is_gtb_2
            gtb_ext_2  = leaf_2.extraction_op.extraction_coefficients[e2][1]
            n_gtb_2    = size(gtb_ext_2, 2)
            gtb_vals_2 = @view cache.gtb_vals[2][1:nd1, 1:n_gtb_2]
            mul!(gtb_vals_2, bsp_vals_2, gtb_ext_2)
        else
            n_gtb_2    = n_bsp_2
            gtb_vals_2 = bsp_vals_2
        end

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

"""Evaluate velocity and Jacobian at point from grid coefficients."""
function probe_field_at_point(
        x::Real, y::Real,
        u_coeffs::AbstractVector{T},
        d::Domain,
        cache::EvaluationCache,
    ) where {T<:Real}
    Tout   = probe_output_eltype(T)
    Lx, Ly = d.box_size
    nx, ny = d.nel
    modes  = bc_to_position_modes(d.bc_sides)

    x_wrapped = wrap_axis(Float64(x), Lx, modes[1], 1e-12)
    y_wrapped = wrap_axis(Float64(y), Ly, modes[2], 1e-12)

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

"""
Evaluate velocity and Jacobian at point with element hint for faster lookup.
If the point remains in the hinted element (xi, eta both in [0,1]), avoid element lookup.
Otherwise, fall back to standard element lookup.
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
    modes  = bc_to_position_modes(d.bc_sides)

    x_wrapped = wrap_axis(Float64(x), Lx, modes[1], 1e-12)
    y_wrapped = wrap_axis(Float64(y), Ly, modes[2], 1e-12)

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

"""Warm up basis-evaluation paths (compilation, memoization)."""
function warmup_evaluation_memo!(d::Domain)
    cache  = EvaluationCache(evaluation_cache_size(d))
    points = Mantis.Points.CartesianPoints(([0.5], [0.5]))
    evaluate_fast!(cache, d.R1.fem_space, 1, points, 1)
    evaluate_fast_2d!(cache, d.R1.fem_space, 1, 0.5, 0.5, 1)
    evaluate_fast_2d!(cache, d.R1.fem_space, 1, 0.5, 0.5, 0)
    return nothing
end

"""Export particle data to VTK format."""
function export_particles_to_vtk(particles::Particles, output_path::String)
    n_particles = length(particles.x)

    points = zeros(3, n_particles)
    points[1, :] .= particles.x
    points[2, :] .= particles.y

    cells = [MeshCell(VTKCellTypes.VTK_VERTEX, vec([i])) for i in 1:n_particles]

    vtkfile = vtk_grid(output_path, points, cells)

    u_part     = copy(particles.mx)
    v_part     = copy(particles.my)
    speed_part = sqrt.(u_part.^2 .+ v_part.^2)

    vtkfile["mx",     VTKPointData()] = u_part
    vtkfile["my",     VTKPointData()] = v_part
    vtkfile["speed",  VTKPointData()] = speed_part
    vtkfile["volume", VTKPointData()] = particles.volume

    outfiles = vtk_save(vtkfile)
    return outfiles
end
"""
Read particle state from a `.vtu` file written by `export_particles_to_vtk`.
Returns `(x, y, mx, my, volume)` as `Vector{Float64}`.
"""
function read_particles_from_vtu(path::String)
    vtk = ReadVTK.VTKFile(path)

    points = ReadVTK.get_data(ReadVTK.get_data_section(vtk, "Points")["Points"])
    coords = reshape(points, 3, :)

    pd = ReadVTK.get_point_data(vtk)
    mx = ReadVTK.get_data(pd["mx"])
    my = ReadVTK.get_data(pd["my"])
    vol = ReadVTK.get_data(pd["volume"])

    n = size(coords, 2)
    x = Vector{Float64}(undef, n)
    y = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        x[i] = coords[1, i]
        y[i] = coords[2, i]
    end

    return x, y, Vector{Float64}(mx), Vector{Float64}(my), Vector{Float64}(vol)
end

"""
Build a fresh `Particles` from raw vectors. P-tensor and `delta_t` are
initialised to identity / zero — the .vtu export does not store them, so
the FTLE accumulator restarts cold.
"""
function particles_from_vtu_state(
        x::Vector{Float64}, y::Vector{Float64},
        mx::Vector{Float64}, my::Vector{Float64},
        vol::Vector{Float64},
        num_elements::Int,
    )
    n = length(x)
    @assert length(y)  == n
    @assert length(mx) == n
    @assert length(my) == n
    @assert length(vol) == n

    can_x    = zeros(Float64, n)
    can_y    = zeros(Float64, n)
    head     = zeros(Int, num_elements)
    next     = zeros(Int, n)
    elem_ids = zeros(Int, n)

    P11 = ones(Float64, n);  P12 = zeros(Float64, n)
    P21 = zeros(Float64, n); P22 = ones(Float64, n)
    delta_t = zeros(Float64, n)

    return Particles(x, y, mx, my, vol,
                     can_x, can_y, head, next, elem_ids,
                     P11, P12, P21, P22, delta_t)
end

"""
Continue a CO-FLIP run from a saved particle snapshot.

Arguments:
- `cfg`              — `SimulationConfig` matching the original run (same nel, p, k,
                       box_size, BCs, obstacle, flow_type, viscosity, …). The
                       `T_final` field is ignored on restart; `additional_steps`
                       controls the run length instead.
- `particles_vtu`    — path to the `particles_NNNN.vtu` to resume from.

Keyword arguments:
- `restart_step`     — step index of the snapshot (used only for output naming;
                       next file written is `*_{restart_step+output_every}`).
- `additional_steps` — number of CO-FLIP steps to run after the restart.
- `output_dir`       — where VTK snapshots and particles are written. Defaults
                       to the parent directory of `particles_vtu`.
- `dt`               — optional override; if `nothing`, recomputed from CFL on
                       the reconstructed velocity field.
"""
function restart_main(
        cfg::SimulationConfig,
        particles_vtu::String;
        restart_step::Int,
        additional_steps::Int,
        dt::Float64,
        output_dir::Union{String,Nothing} = nothing,
    )
    println("Restart adapter: loading $(particles_vtu)")
    LinearAlgebra.BLAS.set_num_threads(1)

    out_dir = output_dir === nothing ? dirname(abspath(particles_vtu)) : output_dir
    mkpath(out_dir)

    domain = GenerateDomain(cfg.nel, cfg.p, cfg.k;
                            box_size           = cfg.box_size,
                            starting_point     = cfg.starting_point,
                            boundary_condition = cfg.boundary_condition,
                            obstacle           = cfg.obstacle)

    num_elements = prod(domain.nel)

    x, y, mx, my, vol = read_particles_from_vtu(particles_vtu)
    particles = particles_from_vtu_state(x, y, mx, my, vol, num_elements)
    println("  Loaded $(length(particles.x)) particles")

    apply_obstacle_brinkman!(particles, domain)
    particle_sorter!(particles, domain)

    ndofs   = FunctionSpaces.get_num_basis(domain.R1.fem_space)
    sim_buf = SimulationBuffers(length(particles.x), ndofs, num_elements, domain)

    println("Reconstructing u_coeffs from particle impulses (coadjoint P2G + projection)...")
    u_coeffs = coadjoint_step!(particles, domain, sim_buf;
        lsqr_atol                    = cfg.lsqr_atol,
        lsqr_btol                    = cfg.lsqr_btol,
        lsqr_maxiter                 = cfg.lsqr_maxiter,
        lsqr_error_on_nonconvergence = cfg.lsqr_error_on_nonconvergence,
        projection_mean_subtract     = cfg.projection_mean_subtract,
    )

    u_form      = Forms.build_form_field(domain.R1, u_coeffs; label="u_h")
    u_phys_expr = ★(u_form)
    u_phys_form = Assemblers.solve_L2_projection(domain.R1, u_phys_expr, domain.dΩ)

    nx, ny = domain.nel
    Lx, Ly = domain.box_size
    dx = Lx / nx;  dy = Ly / ny
    @inbounds for pid in eachindex(particles.x)
        eid = particles.elem_ids[pid]
        ej  = ((eid - 1) ÷ nx) + 1
        ei  = eid - (ej - 1) * nx
        set_g2p_velocity(
            particles, pid,
            particles.x[pid], particles.y[pid],
            eid, (ei - 1) * dx, (ej - 1) * dy, dx, dy,
            u_phys_form, Lx, Ly,
        )
    end
    apply_obstacle_brinkman!(particles, domain)

    isfinite(dt) && dt > 0 ||
        error("restart_main: dt must be a finite positive Float64, got dt=$(dt)")

    # Informational CFL gauge against the loaded particle + reconstructed grid
    # state — just printed, not enforced. The user's `dt` is used as-is; the
    # in-loop `recheck_cfl_dt` / `enforce_cfl_recheck!` will still adapt it if
    # `cfg.cfl_adaptive=true`.
    dt_particles = compute_cfl_dt_from_particles(particles, domain, cfg.target_cfl)
    dt_grid      = recheck_cfl_dt(u_coeffs, domain, cfg.target_cfl, Inf)
    println(@sprintf("CFL gauge at restart: dt_particles=%.6g, dt_grid=%.6g (user dt=%.6g)",
                     dt_particles, dt_grid, dt))
    println("Restarting from step $(restart_step) for $(additional_steps) more steps (dt = $(dt))")

    warmup_evaluation_memo!(domain)

    step = restart_step + 1
    final_step = restart_step + additional_steps
    while step <= final_step
        println("Step $step / $final_step (t ≈ $(round(step * dt, digits=3)))...")

        if mod(step, 5) == 0
            physical_spatial_sort!(particles, domain)
        end

        u_coeffs = step_co_flip!(particles, domain, dt, u_coeffs, sim_buf, cfg)

        u_form      = Forms.build_form_field(domain.R1, u_coeffs; label="u_h")
        u_phys_expr = ★(u_form)
        u_phys_form = Assemblers.solve_L2_projection(domain.R1, u_phys_expr, domain.dΩ)

        if cfg.delayed_reinit_frequency > 0 &&
           step % cfg.delayed_reinit_frequency == 0
            new_count = global_reseed_from_grid!(particles, domain, cfg, u_phys_form;
                                                 rng_seed = cfg.rng_seed)
            println("  Delayed re-seed at step $step: $(new_count) particles re-generated.")
        end

        reset_count = apply_ftle_reset!(particles, domain, u_phys_form, cfg)
        reset_count > 0 && println("  FTLE reset: $reset_count particles reseeded.")

        dt_check = recheck_cfl_dt(u_coeffs, domain, cfg.target_cfl, dt)
        new_dt   = enforce_cfl_recheck!(dt, dt_check, cfg)
        if new_dt != dt
            println("  Adaptive CFL: dt=$(new_dt) (was $(dt))")
            dt = new_dt
        end

        if cfg.output_every > 0 && step % cfg.output_every == 0
            d_u_phys = Forms.d(u_phys_form)
            ω_h      = ★(d_u_phys)
            Plot.export_form_fields_to_vtk((u_form,), @sprintf("u_h_%04d", step);
                                           output_directory_tree=[out_dir])
            Plot.export_form_fields_to_vtk((ω_h,),    @sprintf("w_h_%04d", step);
                                           output_directory_tree=[out_dir])

            diagnostics = compute_conservation_diagnostics(u_coeffs, domain)
            print_conservation_diagnostics(step, step * dt, diagnostics)

            export_particles_to_vtk(
                particles,
                joinpath(out_dir, @sprintf("particles_%04d", step)),
            )
            println("  Saved step $step visualization and particle data.")
        end

        maybe_clear_memo_tables!(step, cfg.clear_memo_every, domain)
        step += 1
    end

    println("\nRestart complete. Files written to '$(out_dir)'.")
    return nothing
end

"""Ensure triplet buffers can hold at least `needed_nnz` B-matrix nonzeros."""
function ensure_B_triplet_capacity!(buf::SimulationBuffers, needed_nnz::Int)
    if length(buf.B_I) < needed_nnz
        resize!(buf.B_I, needed_nnz)
        resize!(buf.B_J, needed_nnz)
        resize!(buf.B_V, needed_nnz)
    end
    return nothing
end

"""Count non-zero particles in each element (used for analysis and diagnostics)."""
function count_particles_per_element!(counts_elem::Vector{Int}, p::Particles, d::Domain)
    fill!(counts_elem, 0)
    @inbounds for eid in 1:length(counts_elem)
        pid = p.head[eid]
        while pid != 0
            counts_elem[eid] += 1
            pid = p.next[pid]
        end
    end
    return counts_elem
end

"""Build particle-to-grid least-squares matrix with parallelization."""
function build_B_matrix(p::Particles, d::Domain, buf::SimulationBuffers)
    fes      = d.R1.fem_space
    num_p    = length(p.x)
    num_dofs = FunctionSpaces.get_num_basis(fes)
    num_elem = prod(d.nel)

    ensure_B_triplet_capacity!(buf, 2 * num_p * d.eval_cache_size)
    I_idx  = buf.B_I
    J_idx  = buf.B_J
    V_val  = buf.B_V

    if length(buf.elem_basis_indices) != num_elem
        resize!(buf.elem_basis_indices, num_elem)
        for eid in 1:num_elem
            buf.elem_basis_indices[eid] = Int[]
        end
    end

    # Pre-compute basis indices for all elements
    @inbounds for eid in 1:num_elem
        dof_indices = buf.elem_basis_indices[eid]
        if isempty(dof_indices)
            dof_indices = collect(FunctionSpaces.get_basis_indices(fes, eid))
            buf.elem_basis_indices[eid] = dof_indices
        end
    end

    # STEP 1: Count particles per element and compute offsets
    counts_elem = buf.counts_elem
    count_particles_per_element!(counts_elem, p, d)
    
    offsets = Vector{Int}(undef, num_elem + 1)
    offsets[1] = 1
    
    @inbounds for eid in 1:num_elem
        n_loc = length(buf.elem_basis_indices[eid])
        num_particles_in_elem = counts_elem[eid]
        nnz_this_elem = num_particles_in_elem * n_loc * 2
        offsets[eid + 1] = offsets[eid] + nnz_this_elem
    end
    
    upper_bound_nnz = offsets[end] - 1
    ensure_B_triplet_capacity!(buf, upper_bound_nnz)

    # STEP 2: Parallel element loop with thread-local writes
    # Track actual cursor position for each element
    final_cursors = Vector{Int}(undef, num_elem)
    
    Threads.@threads :static for eid in 1:num_elem
        tid   = Threads.threadid()
        cache = buf.thread_caches[tid]
        
        pid = p.head[eid]
        if pid == 0
            final_cursors[eid] = offsets[eid]
            continue
        end

        dof_indices = buf.elem_basis_indices[eid]
        n_loc = length(dof_indices)
        
        cursor = offsets[eid]

        while pid != 0
            eval_out = evaluate_fast_2d!(cache, fes, eid, p.can_x[pid], p.can_y[pid], 0)

            vals_x = @view eval_out[1][1][1][1, 1:n_loc]
            vals_y = @view eval_out[1][1][2][1, 1:n_loc]
            w      = sqrt(p.volume[pid])
            row_x  = 2pid - 1
            row_y  = 2pid

            for k in 1:n_loc
                vx = vals_x[k] * w
                vy = vals_y[k] * w

                if abs(vx) > 1e-15
                    I_idx[cursor] = row_x
                    J_idx[cursor] = dof_indices[k]
                    V_val[cursor] = vx
                    cursor += 1
                end

                if abs(vy) > 1e-15
                    I_idx[cursor] = row_y
                    J_idx[cursor] = dof_indices[k]
                    V_val[cursor] = vy
                    cursor += 1
                end
            end

            pid = p.next[pid]
        end
        
        final_cursors[eid] = cursor
    end

    # STEP 3: Compact triplet arrays to remove gaps created by filtering
    # After parallel assembly, element eid has valid entries from offsets[eid] to final_cursors[eid]-1
    # We need to compact these into a contiguous block
    write_pos = 1
    @inbounds for eid in 1:num_elem
        read_start = offsets[eid]
        read_end   = final_cursors[eid] - 1
        num_to_copy = max(0, read_end - read_start + 1)
        
        if num_to_copy > 0
            # Copy valid entries from this element to the next contiguous write position
            copyto!(I_idx, write_pos, I_idx, read_start, num_to_copy)
            copyto!(J_idx, write_pos, J_idx, read_start, num_to_copy)
            copyto!(V_val, write_pos, V_val, read_start, num_to_copy)
            write_pos += num_to_copy
        end
    end
    
    actual_nnz = write_pos - 1

    return sparse(
        @view(I_idx[1:actual_nnz]),
        @view(J_idx[1:actual_nnz]),
        @view(V_val[1:actual_nnz]),
        2num_p, num_dofs,
    )
end

"""Build LSQR right-hand side from particle impulses."""
function build_lsqr_rhs!(V_p::AbstractVector{Float64}, p::Particles)
    @inbounds for i in 1:length(p.x)
        w = sqrt(p.volume[i])
        V_p[2i-1] =  p.my[i] * w
        V_p[2i]   = -p.mx[i] * w
    end
    return V_p
end

"""
Solve `min ‖B·x − b‖₂` via LSQR with Jacobi (column-norm) right preconditioning.

Equivalent to scaling B's columns to unit ℓ²-norm before the iteration: we solve
`min ‖(B·D)·y − b‖₂` with `D = diag(1/‖B[:,j]‖₂)`, then recover `x = D·y`. This change
of variables typically halves or quarters the LSQR iteration count on FEM P2G systems
where column norms vary by 10×–100× depending on how many particles touch each DOF.

`B` is modified **in place** (columns are scaled). `x` carries the warm-start in
original x-space on entry and the converged solution in original x-space on exit.
Returns the LSQR convergence info (`ConvergenceHistory`).
"""
function solve_lsqr_jacobi!(
        x::AbstractVector{Float64},
        B::SparseMatrixCSC{Float64,Int},
        b::AbstractVector{Float64};
        atol::Float64=1e-9,
        btol::Float64=1e-9,
        maxiter::Int=2000,
    )
    ncols = size(B, 2)
    length(x) == ncols || throw(DimensionMismatch("x length $(length(x)) ≠ ncols $ncols"))

    # Column 2-norms (and clamp empty columns to 1.0 to avoid divide-by-zero)
    col_norms = Vector{Float64}(undef, ncols)
    colptr = B.colptr
    nzval  = B.nzval
    @inbounds for j in 1:ncols
        s = 0.0
        for k in colptr[j]:(colptr[j + 1] - 1)
            v = nzval[k]
            s += v * v
        end
        cn = sqrt(s)
        col_norms[j] = cn > eps() ? cn : 1.0
    end

    # Scale B in place: B := B · diag(1/col_norms)
    @inbounds for j in 1:ncols
        inv_n = 1.0 / col_norms[j]
        for k in colptr[j]:(colptr[j + 1] - 1)
            nzval[k] *= inv_n
        end
    end

    # Warm-start change of variables: y = D⁻¹·x = x .* col_norms
    @inbounds for j in 1:ncols
        x[j] *= col_norms[j]
    end

    _, ch = IterativeSolvers.lsqr!(x, B, b; atol=atol, btol=btol,
                                   maxiter=maxiter, log=true)

    # Recover x = D·y = y ./ col_norms
    @inbounds for j in 1:ncols
        x[j] /= col_norms[j]
    end

    return ch
end

"""Solve P2G least-squares system with warm-start for grid velocity."""
function solve_grid_velocity_lsqr(
        B::SparseMatrixCSC{Float64,Int},
        p::Particles,
        V_p_buf::AbstractVector{Float64},
        warm::AbstractVector{Float64};
        atol::Float64=1e-8,
        btol::Float64=1e-8,
        maxiter::Int=2000,
        error_on_nonconvergence::Bool=true,
    )
    n = 2 * length(p.x)
    V_p = @view V_p_buf[1:n]
    build_lsqr_rhs!(V_p, p)

    x0 = copy(warm)
    if length(x0) != size(B, 2)
        x0 = zeros(size(B, 2))
    end
    ch = solve_lsqr_jacobi!(x0, B, V_p; atol=atol, btol=btol, maxiter=maxiter)
    if !ch.isconverged && error_on_nonconvergence
        @error "P2G LSQR did not converge" iters=ch.iters atol=atol btol=btol maxiter=maxiter
    end
    return x0
end

"""Project to divergence-free space via Hodge-Laplace and return pressure."""
function project_and_get_pressure(dom::Domain, v_h::V, buf::SimulationBuffers; subtract_mean::Bool=true) where {V}
    fill!(buf.v_coeffs_buf, 0.0)
    copyto!(buf.v_coeffs_buf, 1, v_h.coefficients, 1, length(v_h.coefficients))

    mul!(buf.b_buf, dom.N_Hodge, buf.v_coeffs_buf)
    copyto!(buf.sol_buf, buf.b_buf)
    ldiv!(dom.LHS_Hodge_fact, buf.sol_buf)
    u_corr, _ = Forms.build_form_fields((dom.R1, dom.R2), buf.sol_buf; labels=("u¹ₕ", "ϕ²ₕ"))

    div_free_coeffs = v_h.coefficients - u_corr.coefficients

    return div_free_coeffs
end

"""
Apply one implicit FEEC viscous diffusion step to grid 1-form coefficients.

In the rotated R1 storage used by this code (★u_h = physical velocity), the
Hodge-Laplace projection enforces d(u_h) = 0, i.e. *physical* divergence-free.
The viscous Laplacian on physically-div-free fields is therefore dδ on u_h, not
δd. Its weak-form matrix is L = Gᵀ·M_R0⁻¹·G, where G = G_R0_R1 and M_R0 is the
R0 mass. Forming L explicitly is dense, so we solve the equivalent sparse
saddle-point system

    ⎡ M_R1        ν·Δt·Gᵀ ⎤ ⎡f_new⎤   ⎡ M_R1·f_in ⎤
    ⎣  G         -M_R0    ⎦ ⎣  q  ⎦ = ⎣     0     ⎦

where q ≈ M_R0⁻¹·G·f_new is the discrete physical vorticity at the new step.
Eliminating q yields (M_R1 + ν·Δt·Gᵀ·M_R0⁻¹·G)·f_new = M_R1·f_in.

The system is reassembled each call because ν·Δt changes under adaptive CFL,
then factorised with `lu` and solved. Wall DOFs (R1 only) are pinned to zero to
preserve free-slip BCs. Returns f_out, overwriting it.
"""
function apply_viscous_diffusion_feec!(
        f_out::AbstractVector{Float64},
        f_in::AbstractVector{Float64},
        domain::Domain,
        ν::Float64,
        dt::Float64,
    )
    ν <= 0.0 && (copyto!(f_out, f_in); return f_out)

    M_R1 = domain.Mass_matrix
    M_R0 = domain.R0_Mass_matrix
    G    = domain.G_R0_R1

    n_R1 = size(M_R1, 1)
    n_R0 = size(M_R0, 1)
    νdt  = ν * dt

    A_sys = [ M_R1                 νdt .* transpose(G);
              G                    -M_R0               ]

    rhs = vcat(M_R1 * f_in, zeros(n_R0))

    homog = domain.homogeneous_dofs_R1
    dir   = domain.dirichlet_dofs_R1
    pinned = isempty(homog) ? dir : (isempty(dir) ? homog : sort!(vcat(homog, dir)))
    lift   = domain.lift_g_R1

    if !isempty(pinned)
        apply_dirichlet_zero!(A_sys, pinned)
        @inbounds for i in homog
            rhs[i] = 0.0
        end
        @inbounds for i in dir
            rhs[i] = lift[i]
        end
    end

    sol = lu(A_sys) \ rhs
    copyto!(f_out, view(sol, 1:n_R1))

    @inbounds for i in homog
        f_out[i] = 0.0
    end
    @inbounds for i in dir
        f_out[i] = lift[i]
    end

    return f_out
end

"""Assemble Hodge-Laplace system matrices."""
function assemble_hodge_laplacian_matrices(R1, R2, dΩ)
    weak_form_inputs = Assemblers.WeakFormInputs((R1, R2))
    lhs_expressions, rhs_expressions = get_lhs_rhs_hodge_laplacian(weak_form_inputs, dΩ)
    weak_form = Assemblers.WeakForm(lhs_expressions, rhs_expressions, weak_form_inputs)
    A, N = Assemblers.assemble(weak_form; rhs_type=SparseArrays.SparseMatrixCSC{Float64, Int})

    return A, N

end

"""Build weak-form expressions for Hodge-Laplace system."""
function get_lhs_rhs_hodge_laplacian(
    inputs::Assemblers.WeakFormInputs, dΩ::Quadrature.AbstractGlobalQuadratureRule
)
    ϵ¹, ε² = Assemblers.get_test_forms(inputs)
    u¹, ϕ² = Assemblers.get_trial_forms(inputs)
    deps1 = d(ϵ¹)
    A_11 = ∫(ϵ¹ ∧ ★(u¹), dΩ)
    A_12 = -∫(d(ϵ¹) ∧ ★(ϕ²), dΩ)
    A_21 = ∫(ε² ∧ ★(d(u¹)), dΩ)
    lhs_expressions = ((A_11, A_12), (A_21, 0))
    b_21 = ∫(ε² ∧ ★(deps1), dΩ)
    rhs_expressions = ((0,0), (b_21,0))
    return lhs_expressions, rhs_expressions
end

"""Assemble Galerkin 1-form mass matrix."""
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

"""
Assemble the Galerkin R0 mass matrix (Hodge star on 0-forms).

M0[i,j] = ∫(ψᵢ ∧ ★(ψⱼ)), dΩ)

Required by the implicit viscous solve. In the rotated R1 storage convention used
here (★u_h = physical velocity), the discrete d:R1→R2 produces the physical
divergence, so the viscous Laplacian on divergence-free fields is dδ, not δd. Its
saddle-point representation needs M_R0 to map between R1 and the R0 codifferential
multiplier (q = M_R0⁻¹·G·f acts as the physical vorticity).
"""
function assemble_R0_mass_matrix(R0_space::F, dΩ::Q) where {F, Q}
    weak_form_inputs = Assemblers.WeakFormInputs(R0_space)
    vᵏ = Assemblers.get_test_form(weak_form_inputs)
    uᵏ = Assemblers.get_trial_form(weak_form_inputs)

    A = ∫(vᵏ ∧ ★(uᵏ), dΩ)

    lhs_expressions = ((A,),)
    rhs_expressions = ((0,),)

    weak_form = Assemblers.WeakForm(lhs_expressions, rhs_expressions, weak_form_inputs)
    M0, _ = Assemblers.assemble(weak_form; rhs_type=SparseArrays.SparseMatrixCSC{Float64, Int})
    return M0
end

"""Assemble scalar Laplacian (R0 stiffness): K[i,j] = ∫(d(ψᵢ) ∧ ★ d(ψⱼ))."""
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

"""
Assemble rectangular weak-gradient block G[i,j] = ∫(d(ψᵢ⁰) ∧ ★ φⱼ¹) coupling R0 (rows)
with R1 (cols). Provides the weak-divergence operator τ̃ ↦ G·τ̃ ∈ R0 and, via its
transpose, satisfies M₁·d₀ = Gᵀ for the FEEC complex.
"""
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

"""Apply energy-conserving correction orthogonal to midpoint velocity."""
function apply_energy_correction!(
        f_next::AbstractVector{Tn},
        f_original::AbstractVector{To},
        f_from_particles::AbstractVector{Tp},
        f_midpoint::AbstractVector{Tm};
        star_apply::Function,
        tol::Float64=1e-9,
        project_non_orthogonal::Bool=true,
    ) where {Tn<:Real, To<:Real, Tp<:Real, Tm<:Real}

    if project_non_orthogonal
        circulations_original = star_apply(f_midpoint)
        original_energy = dot(f_midpoint, circulations_original)

        if sqrt(abs(original_energy)) > tol
            raw_diff_dot = dot(f_from_particles .- f_original, circulations_original)
            scale = raw_diff_dot / original_energy

            @inbounds for i in eachindex(f_next)
                fd = (f_from_particles[i] - f_original[i]) - scale * f_midpoint[i]
                f_next[i] = f_original[i] + fd
            end
            return nothing
        end
    end

    @inbounds for i in eachindex(f_next)
        f_next[i] = f_from_particles[i]
    end
    return nothing
end

"""Compute pressure-gradient correction."""
@inline function compute_pressure_delta(
        f_projected::AbstractVector{Tp},
        f_pre_projection::AbstractVector{Tq},
    ) where {Tp<:Real, Tq<:Real}
    return f_projected .- f_pre_projection
end

"""Apply FLIP-style pressure gradient correction to particles."""
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

"""Blend particle velocity toward grid velocity."""
function apply_pic_blend!(
        p::Particles,
        f_coeffs::AbstractVector{Float64},
        d::Domain,
        thread_caches::Vector{EvaluationCache},
        alpha::Float64,
    )
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

"""Integrate form expression over all elements."""
function integrate_form_expression(expr, dom::Domain)
    total = 0.0
    for element_id in 1:Quadrature.get_num_base_elements(dom.dΩ)
        total += sum(Forms.evaluate(expr, element_id)[1])
    end
    return total
end

"""Compute energy, circulation, and enstrophy diagnostics."""
function compute_conservation_diagnostics(u_coeffs::AbstractVector{T}, dom::Domain) where {T<:Real}
    u_h          = Forms.build_form_field(dom.R1, u_coeffs; label="u_h")
    u_phys_expr  = ★(u_h)
    u_phys_form  = Assemblers.solve_L2_projection(dom.R1, u_phys_expr, dom.dΩ)
    d_u_phys     = Forms.d(u_phys_form)
    ω_h          = ★(d_u_phys)

    energy      = 0.5 * integrate_form_expression(∫(u_h ∧ ★(u_h), dom.dΩ), dom)
    circulation =       integrate_form_expression(∫(d_u_phys,       dom.dΩ), dom)
    enstrophy   = 0.5 * integrate_form_expression(∫(ω_h ∧ ★(ω_h), dom.dΩ), dom)

    return (; energy, circulation, enstrophy)
end

"""Print conservation diagnostics."""
function print_conservation_diagnostics(step::Int, time_value::Real, diagnostics)
    @printf(
        "  Diagnostics [step=%d, t=%.6f]: energy=%.12e, circulation=%.12e, enstrophy=%.12e\n",
        step, time_value,
        diagnostics.energy, diagnostics.circulation, diagnostics.enstrophy,
    )
end

"""Apply periodic wrapping or clamping to a single axis position."""
@inline function wrap_axis(x::T, L::T, mode::Symbol, tol::T) where {T<:AbstractFloat}
    if mode === :periodic
        return mod(x, L)
    elseif mode === :clamp
        return clamp(x, tol, L - tol)
    else
        throw(ArgumentError("Unknown position mode: $mode. Use :periodic or :clamp"))
    end
end

"""Apply position policy uniformly to both axes (legacy two-axis form)."""
@inline function apply_position_policy(x::T, y::T, Lx::T, Ly::T, mode::Symbol, tol::T) where {T<:AbstractFloat}
    return wrap_axis(x, Lx, mode, tol), wrap_axis(y, Ly, mode, tol)
end

"""Apply per-axis position policy."""
@inline function apply_position_policy(
        x::T, y::T, Lx::T, Ly::T,
        modes::NTuple{2,Symbol}, tol::T,
    ) where {T<:AbstractFloat}
    return wrap_axis(x, Lx, modes[1], tol), wrap_axis(y, Ly, modes[2], tol)
end

"""Evaluate physical velocity and Jacobian at point."""
@inline function physical_velocity_and_jacobian(
        x::T, y::T,
        u_coeffs::AbstractVector{U},
        d::Domain, cache::EvaluationCache,
    ) where {T<:AbstractFloat, U<:Real}
    raw_vel, raw_grad = probe_field_at_point(x, y, u_coeffs, d, cache)

    u = -raw_vel[2]
    v =  raw_vel[1]

    ux = -raw_grad[2, 1]
    uy = -raw_grad[2, 2]
    vx =  raw_grad[1, 1]
    vy =  raw_grad[1, 2]

    vel = SVector{2, T}(u, v)
    jac = SMatrix{2, 2, T, 4}(ux, vx, uy, vy)
    return vel, jac
end

"""
Evaluate physical velocity and Jacobian at point with element hint for optimization.
During RK sub-steps, particles move minimally, so the element hint reduces redundant lookups.
"""
@inline function physical_velocity_and_jacobian_hint(
        x::T, y::T,
        u_coeffs::AbstractVector{U},
        d::Domain, cache::EvaluationCache,
        hint_elem::Int,
    ) where {T<:AbstractFloat, U<:Real}
    raw_vel, raw_grad = probe_field_at_point_with_hint(x, y, u_coeffs, d, cache, hint_elem)

    u = -raw_vel[2]
    v =  raw_vel[1]

    ux = -raw_grad[2, 1]
    uy = -raw_grad[2, 2]
    vx =  raw_grad[1, 1]
    vy =  raw_grad[1, 2]

    vel = SVector{2, T}(u, v)
    jac = SMatrix{2, 2, T, 4}(ux, vx, uy, vy)
    return vel, jac
end

"""Iteratively rescale pullback to unit determinant."""
@inline function clamp_pullback(
        pullback::SMatrix{2, 2, T, 4};
        tol::T,
        max_iter::Int,
    ) where {T<:AbstractFloat}

    d_pb = det(pullback)
    if !isfinite(d_pb) || abs(d_pb) < eps(T)
        return pullback
    end

    Pt    = transpose(pullback)
    PtP   = Pt * pullback
    d_PtP = det(PtP)
    if !isfinite(d_PtP) || d_PtP < eps(T)
        return pullback
    end

    inv_Pt   = inv(Pt)
    tr_mixed = tr(inv(PtP))
    if !isfinite(tr_mixed) || abs(tr_mixed) < eps(T)
        return pullback
    end

    prev_mu         = zero(T)
    used_mu         = zero(T)
    output_pullback = pullback
    mixed_det       = d_pb

    iter = 0
    while abs(mixed_det - one(T)) > tol && iter < max_iter
        used_mu = prev_mu + ((one(T) / mixed_det) - one(T)) / tr_mixed
        !isfinite(used_mu) && break
        output_pullback = pullback + used_mu * inv_Pt
        prev_mu   = used_mu
        mixed_det = det(output_pullback)
        iter += 1
    end

    mixed_det > zero(T) && (output_pullback /= sqrt(mixed_det))



    return output_pullback
end

"""Advance position and pullback by one RK4 step."""
@inline function pullback_rk4_step(
        pos::SVector{2, T},
        input_pullback::SMatrix{2, 2, T, 4},
        dt::T,
        u_coeffs::AbstractVector{U},
        d::Domain,
        cache::EvaluationCache;
        modes::NTuple{2,Symbol},
        pos_tol::T,
        pullback_tol::T,
        pullback_max_iter::Int,
    ) where {T<:AbstractFloat, U<:Real}
    Lx, Ly = d.box_size
    nx, ny = d.nel
    c1 = dt / 6
    c2 = dt / 3
    c3 = dt / 3
    c4 = dt / 6

    # Calculate initial element as hint for RK stages
    x_wrapped = wrap_axis(pos[1], Lx, modes[1], pos_tol)
    y_wrapped = wrap_axis(pos[2], Ly, modes[2], pos_tol)
    dx = Lx / nx;  dy = Ly / ny
    inv_dx = 1.0 / dx;  inv_dy = 1.0 / dy
    ei = clamp(floor(Int, x_wrapped * inv_dx) + 1, 1, nx)
    ej = clamp(floor(Int, y_wrapped * inv_dy) + 1, 1, ny)
    hint_elem = (ej - 1) * nx + ei

    v1, J1 = physical_velocity_and_jacobian_hint(pos[1], pos[2], u_coeffs, d, cache, hint_elem)
    Pdot1 = -(transpose(J1) * input_pullback)

    midp1_raw = pos + (dt * 0.5) * v1
    mx, my = apply_position_policy(midp1_raw[1], midp1_raw[2], Lx, Ly, modes, pos_tol)
    midp1 = SVector{2, T}(mx, my)
    midP1 = input_pullback + (dt * 0.5) * Pdot1

    v2, J2 = physical_velocity_and_jacobian_hint(midp1[1], midp1[2], u_coeffs, d, cache, hint_elem)
    Pdot2 = -(transpose(J2) * midP1)

    midp2_raw = pos + (dt * 0.5) * v2
    mx, my = apply_position_policy(midp2_raw[1], midp2_raw[2], Lx, Ly, modes, pos_tol)
    midp2 = SVector{2, T}(mx, my)
    midP2 = input_pullback + (dt * 0.5) * Pdot2

    v3, J3 = physical_velocity_and_jacobian_hint(midp2[1], midp2[2], u_coeffs, d, cache, hint_elem)
    Pdot3 = -(transpose(J3) * midP2)

    midp3_raw = pos + dt * v3
    mx, my = apply_position_policy(midp3_raw[1], midp3_raw[2], Lx, Ly, modes, pos_tol)
    midp3 = SVector{2, T}(mx, my)
    midP3 = input_pullback + dt * Pdot3

    v4, J4 = physical_velocity_and_jacobian_hint(midp3[1], midp3[2], u_coeffs, d, cache, hint_elem)
    Pdot4 = -(transpose(J4) * midP3)

    pullback = input_pullback + c1 * Pdot1 + c2 * Pdot2 + c3 * Pdot3 + c4 * Pdot4
    pullback = clamp_pullback(pullback; tol=pullback_tol, max_iter=pullback_max_iter)

    pos_raw = pos + c1 * v1 + c2 * v2 + c3 * v3 + c4 * v4
    px, py = apply_position_policy(pos_raw[1], pos_raw[2], Lx, Ly, modes, pos_tol)

    return SVector{2, T}(px, py), pullback
end

"""Advance position and pullback by one RK2 step with element hint optimization."""
@inline function pullback_rk2_step(
        pos::SVector{2, T},
        input_pullback::SMatrix{2, 2, T, 4},
        dt::T,
        u_coeffs::AbstractVector{U},
        d::Domain,
        cache::EvaluationCache;
        modes::NTuple{2,Symbol},
        pos_tol::T,
        pullback_tol::T,
        pullback_max_iter::Int,
    ) where {T<:AbstractFloat, U<:Real}
    Lx, Ly = d.box_size
    nx, ny = d.nel

    # Calculate initial element as hint for RK stages
    x_wrapped = wrap_axis(pos[1], Lx, modes[1], pos_tol)
    y_wrapped = wrap_axis(pos[2], Ly, modes[2], pos_tol)
    dx = Lx / nx;  dy = Ly / ny
    inv_dx = 1.0 / dx;  inv_dy = 1.0 / dy
    ei = clamp(floor(Int, x_wrapped * inv_dx) + 1, 1, nx)
    ej = clamp(floor(Int, y_wrapped * inv_dy) + 1, 1, ny)
    hint_elem = (ej - 1) * nx + ei

    v1, J1 = physical_velocity_and_jacobian_hint(pos[1], pos[2], u_coeffs, d, cache, hint_elem)
    Pdot1 = -(transpose(J1) * input_pullback)

    midp_raw = pos + (dt * 0.5) * v1
    mx, my = apply_position_policy(midp_raw[1], midp_raw[2], Lx, Ly, modes, pos_tol)
    midp = SVector{2, T}(mx, my)
    midP = input_pullback + (dt * 0.5) * Pdot1

    v2, J2 = physical_velocity_and_jacobian_hint(midp[1], midp[2], u_coeffs, d, cache, hint_elem)
    Pdot2 = -(transpose(J2) * midP)

    pullback = input_pullback + dt * Pdot2
    pullback = clamp_pullback(pullback; tol=pullback_tol, max_iter=pullback_max_iter)

    pos_raw = pos + dt * v2
    px, py = apply_position_policy(pos_raw[1], pos_raw[2], Lx, Ly, modes, pos_tol)

    return SVector{2, T}(px, py), pullback
end

"""Advect all particles using RK4 integration (threaded)."""
function advect_particles_pullback_rk4!(
        p::Particles,
        x0::AbstractVector{T}, y0::AbstractVector{T},
        mx0::AbstractVector{T}, my0::AbstractVector{T},
        u_coeffs::AbstractVector{U},
        d::Domain, dt::T,
        x_out::AbstractVector{T}, y_out::AbstractVector{T},
        mx_out::AbstractVector{T}, my_out::AbstractVector{T},
        short_P11::AbstractVector{T}, short_P12::AbstractVector{T},
        short_P21::AbstractVector{T}, short_P22::AbstractVector{T},
        thread_caches::Vector{EvaluationCache},
        max_err_per_thread::Vector{T};
        position_modes::NTuple{2,Symbol}=(:periodic, :periodic),
        position_tol::T=T(1e-12),
        pullback_tol::T=T(1e-9),
        pullback_max_iter::Int=200,
    ) where {T<:AbstractFloat, U<:Real}

    num_p          = length(x0)
    Lx, Ly         = d.box_size
    n_thread_slots = Threads.maxthreadid()
    is_per_x       = position_modes[1] === :periodic
    is_per_y       = position_modes[2] === :periodic

    if length(thread_caches) < n_thread_slots
        throw(ArgumentError("thread_caches length must be at least Threads.maxthreadid()"))
    end
    if length(max_err_per_thread) < n_thread_slots
        throw(ArgumentError("max_err_per_thread length must be at least Threads.maxthreadid()"))
    end
    @inbounds for i in 1:n_thread_slots
        max_err_per_thread[i] = zero(T)
    end

    Threads.@threads :static for i in 1:num_p
        tid   = Threads.threadid()
        cache = thread_caches[tid]

        sx, sy = apply_position_policy(x0[i], y0[i], Lx, Ly, position_modes, position_tol)
        pos    = SVector{2, T}(sx, sy)
        input_pullback = SMatrix{2, 2, T, 4}(one(T), zero(T), zero(T), one(T))

        pos_new, pullback = pullback_rk4_step(
            pos, input_pullback, dt, u_coeffs, d, cache;
            modes=position_modes, pos_tol=position_tol,
            pullback_tol=pullback_tol, pullback_max_iter=pullback_max_iter,
        )

        m_old = SVector{2, T}(mx0[i], my0[i])
        m_new = pullback * m_old

        ox, oy = x_out[i], y_out[i]
        ex = abs(pos_new[1] - ox)
        ey = abs(pos_new[2] - oy)
        is_per_x && ex > 0.5 * Lx && (ex = Lx - ex)
        is_per_y && ey > 0.5 * Ly && (ey = Ly - ey)
        local_err = max(ex, ey)
        local_err > max_err_per_thread[tid] && (max_err_per_thread[tid] = local_err)

        x_out[i]  = pos_new[1]
        y_out[i]  = pos_new[2]
        mx_out[i] = m_new[1]
        my_out[i] = m_new[2]

        short_P11[i] = pullback[1, 1]
        short_P12[i] = pullback[1, 2]
        short_P21[i] = pullback[2, 1]
        short_P22[i] = pullback[2, 2]
    end

    return maximum(@view max_err_per_thread[1:n_thread_slots])
end

"""Advect all particles using RK2 integration (threaded)."""
function advect_particles_pullback_rk2!(
        p::Particles,
        x0::AbstractVector{T}, y0::AbstractVector{T},
        mx0::AbstractVector{T}, my0::AbstractVector{T},
        u_coeffs::AbstractVector{U},
        d::Domain, dt::T,
        x_out::AbstractVector{T}, y_out::AbstractVector{T},
        mx_out::AbstractVector{T}, my_out::AbstractVector{T},
        short_P11::AbstractVector{T}, short_P12::AbstractVector{T},
        short_P21::AbstractVector{T}, short_P22::AbstractVector{T},
        thread_caches::Vector{EvaluationCache},
        max_err_per_thread::Vector{T};
        position_modes::NTuple{2,Symbol}=(:periodic, :periodic),
        position_tol::T=T(1e-12),
        pullback_tol::T=T(1e-9),
        pullback_max_iter::Int=200,
    ) where {T<:AbstractFloat, U<:Real}

    num_p          = length(x0)
    Lx, Ly         = d.box_size
    n_thread_slots = Threads.maxthreadid()
    is_per_x       = position_modes[1] === :periodic
    is_per_y       = position_modes[2] === :periodic

    if length(thread_caches) < n_thread_slots
        throw(ArgumentError("thread_caches length must be at least Threads.maxthreadid()"))
    end
    if length(max_err_per_thread) < n_thread_slots
        throw(ArgumentError("max_err_per_thread length must be at least Threads.maxthreadid()"))
    end
    @inbounds for i in 1:n_thread_slots
        max_err_per_thread[i] = zero(T)
    end

    Threads.@threads :static for i in 1:num_p
        tid   = Threads.threadid()
        cache = thread_caches[tid]

        sx, sy = apply_position_policy(x0[i], y0[i], Lx, Ly, position_modes, position_tol)
        pos    = SVector{2, T}(sx, sy)
        input_pullback = SMatrix{2, 2, T, 4}(one(T), zero(T), zero(T), one(T))

        pos_new, pullback = pullback_rk2_step(
            pos, input_pullback, dt, u_coeffs, d, cache;
            modes=position_modes, pos_tol=position_tol,
            pullback_tol=pullback_tol, pullback_max_iter=pullback_max_iter,
        )

        m_old = SVector{2, T}(mx0[i], my0[i])
        m_new = pullback * m_old

        ox, oy = x_out[i], y_out[i]
        ex = abs(pos_new[1] - ox)
        ey = abs(pos_new[2] - oy)
        is_per_x && ex > 0.5 * Lx && (ex = Lx - ex)
        is_per_y && ey > 0.5 * Ly && (ey = Ly - ey)
        local_err = max(ex, ey)
        local_err > max_err_per_thread[tid] && (max_err_per_thread[tid] = local_err)

        x_out[i]  = pos_new[1]
        y_out[i]  = pos_new[2]
        mx_out[i] = m_new[1]
        my_out[i] = m_new[2]

        short_P11[i] = pullback[1, 1]
        short_P12[i] = pullback[1, 2]
        short_P21[i] = pullback[2, 1]
        short_P22[i] = pullback[2, 2]
    end

    return maximum(@view max_err_per_thread[1:n_thread_slots])
end

"""Perform single P2G and projection pass."""
function coadjoint_step!(p::Particles, d::Domain, buf::SimulationBuffers;
        lsqr_atol::Float64=1e-9, lsqr_btol::Float64=1e-9, lsqr_maxiter::Int=2000,
        lsqr_error_on_nonconvergence::Bool=true,
        projection_mean_subtract::Bool=true,
    )
    step_t0 = time()

    ensure_particle_capacity!(buf, length(p.x))

    t0 = time()
    B = build_B_matrix(p, d, buf)
    t_build_B_ms = time() - t0

    if length(buf.lsqr_warm) != size(B, 2)
        buf.lsqr_warm = zeros(size(B, 2))
    end

    t0 = time()
    u_grid_n = solve_grid_velocity_lsqr(B, p, buf.V_p, buf.lsqr_warm;
                                        atol=lsqr_atol, btol=lsqr_btol,
                                        maxiter=lsqr_maxiter,
                                        error_on_nonconvergence=lsqr_error_on_nonconvergence)
    # Lazy lift initialisation: on the first P2G that sees a non-empty
    # Dirichlet set, snapshot the inlet-DOF values from the unconstrained
    # LSQR fit and freeze them as the lift `lift_g_R1`. This anchors the
    # lift in the same FEM normalisation the rest of the solver uses
    # (no separate L2-projection / Greville interpolation needed).
    maybe_initialise_inlet_lift!(d, u_grid_n)
    # LSQR is BC-unaware; enforce homogeneous walls/outlet to zero and
    # Dirichlet (inlet) entries to the lift value. After this the field is
    # consistent with the strong-Dirichlet operator factorisations.
    enforce_boundary_dofs!(u_grid_n, d)
    copy!(buf.lsqr_warm, u_grid_n)
    t_solve_raw_ms = time() - t0

    t0 = time()
    u_grid_n_h             = Forms.build_form_field(d.R1, u_grid_n)
    u_div_free_coeffs_n = project_and_get_pressure(d, u_grid_n_h, buf;
                                                      subtract_mean=projection_mean_subtract)
    # Defensive: the projection preserves the prescribed trace mathematically
    # but a single re-stamp keeps round-off from drifting the inlet value.
    enforce_boundary_dofs!(u_div_free_coeffs_n, d)
    t_project_ms = time() - t0

    t_total_ms = time() - step_t0

    println(
        "  Coadjoint Timings [s]: " *
        "build_B=$(round(t_build_B_ms, digits=2)), " *
        "solve_raw=$(round(t_solve_raw_ms, digits=2)), " *
        "project=$(round(t_project_ms, digits=2)), " *
        "total=$(round(t_total_ms, digits=2))",
    )

    return u_div_free_coeffs_n
end

"""Advance one CO-FLIP timestep with fixed-point iteration."""
function step_co_flip!(
        p::Particles, d::Domain, dt::Float64,
        f_n::AbstractVector{Tf},
        buf::SimulationBuffers,
        cfg::SimulationConfig,
    ) where {Tf<:Real}

    @assert all(isfinite, f_n) "step_co_flip!: f_n contains non-finite values"

    step_t0       = time()
    thread_caches = buf.thread_caches

    mass_matrix_1form = d.Mass_matrix

    copy!(buf.f_n_saved, f_n)
    f_n_saved = buf.f_n_saved

    eff_min_pe = cfg.min_particles_per_element
    eff_max_pe = cfg.max_particles_per_element
    eff_min_pq = cfg.min_particles_per_quarter

    t0 = time()
    added_start, removed_start = enforce_min_particles_per_element!(
        p, d,
        eff_min_pe, eff_max_pe, eff_min_pq,
        f_n_saved, buf,
    )
    t_seed_start = time() - t0
    if added_start > 0 || removed_start > 0
        println("  Particle rebalance before solve: +$added_start / -$removed_start")
    end
    println("  Initial seeding time: $(round(t_seed_start, digits=2)) s")

    # Inlet replenishment: top up inlet-adjacent elements to
    # `cfg.particles_per_cell` and stamp the analytical inlet velocity onto
    # every particle currently in an inlet row. Runs after the generic
    # rebalance above so the inlet pass sees a freshly-sorted population and
    # its injection is preserved in the captured num_p below. No-op when no
    # `:inlet` side is configured or `cfg.inlet_U_inf == 0`.
    t0 = time()
    n_inlet = enforce_inlet_particles!(p, d, buf, cfg)
    t_inlet = time() - t0
    if n_inlet > 0
        println("  Inlet injection: +$n_inlet particles  (t=$(round(t_inlet, digits=2)) s)")
    end

    num_p = length(p.x)
    ensure_particle_capacity!(buf, num_p)

    x_n  = buf.x_n;   y_n  = buf.y_n
    mx_n = buf.mx_n;  my_n = buf.my_n
    @inbounds for i in 1:num_p
        x_n[i]  = p.x[i];   y_n[i]  = p.y[i]
        mx_n[i] = p.mx[i];  my_n[i] = p.my[i]
    end

    x_np1  = buf.x_np1;   y_np1  = buf.y_np1
    mx_np1 = buf.mx_np1;  my_np1 = buf.my_np1

    f_star          = buf.f_star
    f_np1           = buf.f_np1
    f_np1_raw       = buf.f_np1_raw
    f_np1_proj      = buf.f_np1_proj
    f_np1_proj_prev = buf.f_np1_proj_prev

    copy!(f_star, f_n_saved)
    copy!(f_np1_proj_prev, f_star)

    f0_norm = norm(f_n_saved)

    prev_err = Inf

    for iter in 1:cfg.max_fp_iter
        iter_t0 = time()

        maybe_clear_memo_tables_fp_iter!(iter, cfg.clear_memo_every_fp_iter, d)

        t0 = time()
        if cfg.advection_time_integrator === :rk2
            advect_particles_pullback_rk2!(
                p,
                @view(x_n[1:num_p]),  @view(y_n[1:num_p]),
                @view(mx_n[1:num_p]), @view(my_n[1:num_p]),
                f_star, d, dt,
                @view(x_np1[1:num_p]),  @view(y_np1[1:num_p]),
                @view(mx_np1[1:num_p]), @view(my_np1[1:num_p]),
                @view(buf.short_P11[1:num_p]), @view(buf.short_P12[1:num_p]),
                @view(buf.short_P21[1:num_p]), @view(buf.short_P22[1:num_p]),
                thread_caches, buf.max_err_per_thread;
                position_modes=bc_to_position_modes(d.bc_sides),
            )
        elseif cfg.advection_time_integrator === :rk4
            advect_particles_pullback_rk4!(
                p,
                @view(x_n[1:num_p]),  @view(y_n[1:num_p]),
                @view(mx_n[1:num_p]), @view(my_n[1:num_p]),
                f_star, d, dt,
                @view(x_np1[1:num_p]),  @view(y_np1[1:num_p]),
                @view(mx_np1[1:num_p]), @view(my_np1[1:num_p]),
                @view(buf.short_P11[1:num_p]), @view(buf.short_P12[1:num_p]),
                @view(buf.short_P21[1:num_p]), @view(buf.short_P22[1:num_p]),
                thread_caches, buf.max_err_per_thread;
                position_modes=bc_to_position_modes(d.bc_sides),
            )
        else
            throw(ArgumentError("Unknown advection_time_integrator=$(cfg.advection_time_integrator). Use :rk2 or :rk4"))
        end
        t_advect = time() - t0

        t0 = time()
        @inbounds for i in 1:num_p
            p.x[i]  = x_np1[i];   p.y[i]  = y_np1[i]
            p.mx[i] = mx_np1[i];  p.my[i] = my_np1[i]
        end
        # Immersed-boundary penalisation: any particle whose advected
        # position landed inside the obstacle gets projected to the surface
        # with zero velocity. Done *before* the sort so elem_ids/can_x/can_y
        # are computed from the corrected positions and LSQR sees clean,
        # no-slip-like data in the cells near the surface.
        apply_obstacle_brinkman!(p, d)
        # Skip outlet culling here: the fixed-point loop re-advects every
        # iter from the captured `x_n` anchor, and the advection / pullback
        # buffers are sized to the original num_p. Shrinking p.x mid-iter
        # would desync those. Outlet escapees are culled in the final post-
        # iter sort below.
        particle_sorter!(p, d; cull_outlets=false)
        t_sort = time() - t0

        t0    = time()
        B_np1 = build_B_matrix(p, d, buf)
        if length(buf.lsqr_warm) != size(B_np1, 2)
            buf.lsqr_warm = zeros(size(B_np1, 2))
        end
        t_build_B = time() - t0

        t0      = time()
        raw_sol = solve_grid_velocity_lsqr(B_np1, p, buf.V_p, buf.lsqr_warm;
                                           atol=cfg.lsqr_atol, btol=cfg.lsqr_btol,
                                           maxiter=cfg.lsqr_maxiter,
                                           error_on_nonconvergence=cfg.lsqr_error_on_nonconvergence)
        # First-time lift initialisation (no-op in subsequent iterations) and
        # then stamp the BC trace: homogeneous=0, Dirichlet=lift_g_R1.
        maybe_initialise_inlet_lift!(d, raw_sol)
        enforce_boundary_dofs!(raw_sol, d)
        copy!(f_np1_raw, raw_sol)
        copy!(buf.lsqr_warm, raw_sol)
        t_solve = time() - t0

        t0 = time()
        if cfg.enable_energy_correction
            star_apply = v -> mass_matrix_1form * v
            apply_energy_correction!(
                f_np1, f_n_saved, f_np1_raw, f_star;
                star_apply=star_apply,
                tol=1e-14,
                project_non_orthogonal=true,
            )
            # Energy correction blends three R1 vectors; the BC trace can drift
            # to a slightly different value if the boundary slice of those
            # vectors disagrees. Re-stamp before projection.
            enforce_boundary_dofs!(f_np1, d)
        else
            copy!(f_np1, f_np1_raw)
        end
        t_energy = time() - t0

        t0             = time()
        f_np1_h        = Forms.build_form_field(d.R1, f_np1)
        proj_result = project_and_get_pressure(d, f_np1_h, buf;
                               subtract_mean=cfg.projection_mean_subtract)
        copy!(f_np1_proj, proj_result)
        # Re-stamp the BC trace after projection; the pinned operator should
        # preserve it exactly, but defensive enforcement keeps round-off out
        # of the long-running fixed-point iteration.
        enforce_boundary_dofs!(f_np1_proj, d)
        t_project = time() - t0

        err = norm(@view(f_np1_proj[1:end]) .- @view(f_np1_proj_prev[1:end]))
        err /= (f0_norm + eps(Float64))

        t_iter = time() - iter_t0
        println("  Iter $iter: Grid-vel error = $(Printf.@sprintf("%.6e", err))")
        println(
            "    Timings [s]: advect=$(round(t_advect, digits=2)), " *
            "sort=$(round(t_sort, digits=2)), " *
            "build_B=$(round(t_build_B, digits=2)), " *
            "solve_raw=$(round(t_solve, digits=2)), " *
            "energy=$(round(t_energy, digits=2)), " *
            "project=$(round(t_project, digits=2)), " *
            "iter_total=$(round(t_iter, digits=2))",
        )

        copy!(f_np1_proj_prev, f_np1_proj)
        @. f_star = 0.5 * (f_n_saved + f_np1_proj)

        if err < cfg.fp_tol
            break
        end
        if iter > 1 && err >= 0.99 * prev_err
            println("  Fixed-point stalled (err=$err >= 0.99 * prev_err=$prev_err) — break.")
            break
        end
        prev_err = err
    end

    # --- Viscous split-step diffusion (FEEC) ---
    # f_np1_proj holds the inviscid divergence-free field f*. Apply one implicit
    # (M + ν·Δt·L)·f = M·f* step, then re-project to restore divergence-free.
    if cfg.viscosity > 0.0
        t0 = time()
        apply_viscous_diffusion_feec!(
            f_np1_proj,
            f_np1_proj,
            d,
            cfg.viscosity,
            dt,
        )
        f_np1_proj_h = Forms.build_form_field(d.R1, f_np1_proj)
        proj_after_diff = project_and_get_pressure(d, f_np1_proj_h, buf;
                              subtract_mean=cfg.projection_mean_subtract)
        copy!(f_np1_proj, proj_after_diff)
        println("  Viscous diffusion step (ν=$(cfg.viscosity)): t=$(round(time()-t0, digits=2))s")
    end

    # --- [STUB] Metriplectic viscosity (Approach 2) ---
    # The metriplectic formulation embeds dissipation directly into the geometric
    # structure via the evolution law:
    #   dF/dt = {F, G} + (F, G)
    # where {·,·} is the antisymmetric Lie–Poisson bracket (handled by CO-FLIP)
    # and (·,·) is a symmetric metric bracket generating entropy production.
    #
    # Discrete implementation sketch:
    #   1. At each timestep, compute the dissipation bracket contribution:
    #          Δf_metriplectic = -dt · M⁻¹ · L · f_np1_proj
    #      where L is the curl-curl stiffness (same as FEEC approach).
    #   2. Apply the update:
    #          f_np1_proj += Δf_metriplectic
    #   3. Project to divergence-free subspace.
    #
    # The key difference from split-step: instead of solving (M + ν·Δt·L)·f = M·f*
    # implicitly, the metric bracket drives an *explicit* dissipative update on f,
    # maintaining thermodynamic consistency (dS/dt ≥ 0, dH/dt = 0 in the bracket
    # decomposition). Long-time stability benefits require implicit or Runge–Kutta
    # discretization of the metric bracket.
    #
    # Reference: Kraus & Hirvijoki (2017), "Metriplectic integrators for the
    # Landau collision operator", PoP. Morrison (1984), Physica D.
    #
    # TODO: implement cfg.viscosity_method = :metriplectic branch here.

    compose_longterm_pullback!(p, buf, dt, num_p)

    if cfg.enable_pressure_kick
        t0 = time()
        apply_pressure_correction!(
            p, f_np1_proj, f_np1, d, thread_caches; delta_scale=1.0,
        )
        println("  [pressure kick enabled — div-consistent (FLIP/hybrid)] t=$(round(time()-t0,digits=2))s")
    end

    apply_pic_blend!(p, f_np1_proj, d, thread_caches, cfg.pic_blend_alpha)

    # End-of-step immersed-boundary clean-up: the pressure kick and PIC blend
    # above may have given particles inside the obstacle a small non-zero
    # velocity; re-stamp them to surface positions with zero velocity so the
    # next step starts from a clean no-slip state.
    apply_obstacle_brinkman!(p, d)

    t0 = time()
    particle_sorter!(p, d)
    t_final_sort = time() - t0

    t0 = time()
    added_end, removed_end = enforce_min_particles_per_element!(
        p, d,
        eff_min_pe, eff_max_pe, eff_min_pq,
        f_np1_proj, buf,
    )
    t_seed_end = time() - t0
    if added_end > 0 || removed_end > 0
        println("  Particle rebalance after solve: +$added_end / -$removed_end")
    end

    t_total = time() - step_t0
    println(
        "  Post Timings [s]: " *
        "final_sort=$(round(t_final_sort, digits=2)), " *
        "final_rebalance=$(round(t_seed_end, digits=2)), " *
        "step_total=$(round(t_total, digits=2))",
    )

    return f_np1_proj
end

"""Compose long-term pullback from short-term step and accumulate time."""
function compose_longterm_pullback!(p::Particles, buf::SimulationBuffers, dt::Float64, num_p::Int)
    @inbounds for i in 1:num_p
        sp11 = buf.short_P11[i]; sp12 = buf.short_P12[i]
        sp21 = buf.short_P21[i]; sp22 = buf.short_P22[i]
        p11 = p.P11[i]; p12 = p.P12[i]
        p21 = p.P21[i]; p22 = p.P22[i]
        p.P11[i]     = sp11 * p11 + sp12 * p21
        p.P12[i]     = sp11 * p12 + sp12 * p22
        p.P21[i]     = sp21 * p11 + sp22 * p21
        p.P22[i]     = sp21 * p12 + sp22 * p22
        p.delta_t[i] += dt
    end
    return nothing
end

"""Reset particles exceeding FTLE or time-accumulated thresholds."""
function apply_ftle_reset!(
        p::Particles, d::Domain, u_phys_form,
        cfg::SimulationConfig,
    )
    Lx, Ly = d.box_size
    nx, ny = d.nel
    dx = Lx / nx;  dy = Ly / ny

    if isfinite(cfg.global_ftle_gate)
        max_ftle_global = -Inf
        @inbounds for pid in eachindex(p.x)
            dt_acc = p.delta_t[pid]
            dt_acc <= 0.0 && continue
            P11 = p.P11[pid]; P12 = p.P12[pid]
            P21 = p.P21[pid]; P22 = p.P22[pid]
            a11 = P11*P11 + P21*P21
            a22 = P12*P12 + P22*P22
            a12 = P11*P12 + P21*P22
            tra = a11 + a22
            det_a = a11*a22 - a12*a12
            disc  = max(tra*tra - 4.0 * det_a, 0.0)
            σ2    = 0.5 * (tra + sqrt(disc))
            σ_max = sqrt(max(σ2, 0.0))
            ftle  = σ_max > 0 ?
                (cfg.ftle_use_rate ? log(σ_max) / dt_acc : log(σ_max)) :
                0.0
            if ftle > max_ftle_global
                max_ftle_global = ftle
            end
        end
        if max_ftle_global < cfg.global_ftle_gate
            return 0
        end
    end

    reset_count = 0
    @inbounds for pid in eachindex(p.x)
        dt_acc = p.delta_t[pid]
        dt_acc <= 0.0 && continue

        P11 = p.P11[pid]; P12 = p.P12[pid]
        P21 = p.P21[pid]; P22 = p.P22[pid]

        a11 = P11*P11 + P21*P21
        a22 = P12*P12 + P22*P22
        a12 = P11*P12 + P21*P22
        tra = a11 + a22
        det_a = a11*a22 - a12*a12
        disc  = max(tra*tra - 4.0 * det_a, 0.0)
        σ2    = 0.5 * (tra + sqrt(disc))
        σ_max = sqrt(max(σ2, 0.0))
        ftle  = σ_max > 0 ?
            (cfg.ftle_use_rate ? log(σ_max) / dt_acc : log(σ_max)) :
            0.0

        if ftle > cfg.ftle_threshold || dt_acc > cfg.max_longterm_delta_t
            eid = p.elem_ids[pid]
            ej = ((eid - 1) ÷ nx) + 1
            ei = eid - (ej - 1) * nx
            set_g2p_velocity(
                p, pid, p.x[pid], p.y[pid], eid,
                (ei - 1) * dx, (ej - 1) * dy, dx, dy,
                u_phys_form, Lx, Ly;
                modes=bc_to_position_modes(d.bc_sides),
            )
            p.P11[pid] = 1.0;  p.P12[pid] = 0.0
            p.P21[pid] = 0.0;  p.P22[pid] = 1.0
            p.delta_t[pid] = 0.0
            reset_count += 1
        end
    end
    return reset_count
end

"""Discard and re-seed all particles from current grid state."""
function global_reseed_from_grid!(
        p::Particles, dom::Domain, cfg::SimulationConfig,
        u_phys_form;
        rng_seed::Union{Int,Nothing}=nothing,
    )
    Lx, Ly = dom.box_size
    nx, ny = dom.nel
    dx     = Lx / nx
    dy     = Ly / ny

    target_total = nx * ny * cfg.particles_per_cell
    base_ppc     = target_total ÷ (nx * ny)
    extra        = target_total - base_ppc * (nx * ny)
    new_count    = target_total

    resize!(p.x,        new_count)
    resize!(p.y,        new_count)
    resize!(p.mx,       new_count)
    resize!(p.my,       new_count)
    resize!(p.volume,   new_count)
    resize!(p.can_x,    new_count)
    resize!(p.can_y,    new_count)
    resize!(p.next,     new_count)
    resize!(p.elem_ids, new_count)
    resize!(p.P11,      new_count)
    resize!(p.P12,      new_count)
    resize!(p.P21,      new_count)
    resize!(p.P22,      new_count)
    resize!(p.delta_t,  new_count)

    if rng_seed !== nothing
        Random.seed!(rng_seed)
    end

    modes = bc_to_position_modes(dom.bc_sides)
    pid = 1
    if cfg.particles_per_cell > 0
        used_N = max(1, floor(Int, sqrt(base_ppc) + 1e-12))
        for ej in 1:ny, ei in 1:nx
            x0 = (ei - 1) * dx
            y0 = (ej - 1) * dy
            eid = (ej - 1) * nx + ei
            for jj in 0:(used_N - 1), ii in 0:(used_N - 1)
                px = wrap_axis(x0 + ((ii + rand()) / used_N) * dx, Lx, modes[1], 1e-12)
                py = wrap_axis(y0 + ((jj + rand()) / used_N) * dy, Ly, modes[2], 1e-12)
                p.x[pid] = px
                p.y[pid] = py
                set_g2p_velocity(p, pid, px, py, eid, x0, y0, dx, dy,
                                 u_phys_form, Lx, Ly; modes=modes)
                p.P11[pid] = 1.0; p.P12[pid] = 0.0
                p.P21[pid] = 0.0; p.P22[pid] = 1.0
                p.delta_t[pid] = 0.0
                p.volume[pid] = (cfg.volume_convention === :physical) ?
                    (Lx * Ly) / new_count : 1.0 / max(base_ppc, 1)
                pid += 1
            end
            remainder = base_ppc - used_N * used_N
            for _ in 1:remainder
                px = wrap_axis(x0 + rand() * dx, Lx, modes[1], 1e-12)
                py = wrap_axis(y0 + rand() * dy, Ly, modes[2], 1e-12)
                p.x[pid] = px
                p.y[pid] = py
                set_g2p_velocity(p, pid, px, py, eid, x0, y0, dx, dy,
                                 u_phys_form, Lx, Ly; modes=modes)
                p.P11[pid] = 1.0; p.P12[pid] = 0.0
                p.P21[pid] = 0.0; p.P22[pid] = 1.0
                p.delta_t[pid] = 0.0
                p.volume[pid] = (cfg.volume_convention === :physical) ?
                    (Lx * Ly) / new_count : 1.0 / max(base_ppc, 1)
                pid += 1
            end
        end
        for _ in 1:extra
            px = rand() * Lx
            py = rand() * Ly
            ei = clamp(floor(Int, px * (nx / Lx)) + 1, 1, nx)
            ej = clamp(floor(Int, py * (ny / Ly)) + 1, 1, ny)
            eid = (ej - 1) * nx + ei
            x0 = (ei - 1) * dx
            y0 = (ej - 1) * dy
            p.x[pid] = px
            p.y[pid] = py
            set_g2p_velocity(p, pid, px, py, eid, x0, y0, dx, dy,
                             u_phys_form, Lx, Ly; modes=modes)
            p.P11[pid] = 1.0; p.P12[pid] = 0.0
            p.P21[pid] = 0.0; p.P22[pid] = 1.0
            p.delta_t[pid] = 0.0
            p.volume[pid] = (cfg.volume_convention === :physical) ?
                (Lx * Ly) / new_count : 1.0 / max(base_ppc, 1)
            pid += 1
        end
    end

    particle_sorter!(p, dom)

    return new_count
end

"""Clear and rewarm memoization cache periodically. Also runs an explicit
`GC.gc()` after the cache reset so the freed memo tables — plus any
throwaway sparse matrices the L2-projection / LSQR paths allocated in
the preceding steps — actually get released back to the OS. Without
this, Julia's heuristic GC sometimes lets sparse-matrix churn build up
to multi-GB before triggering, which is what causes the OOM at step
~256 on the lid-driven cavity."""
function maybe_clear_memo_tables!(step::Int, clear_every::Int, d::Domain)
    if clear_every > 0 && (step % clear_every == 0)
        Memoization.empty_all_caches!()
        warmup_evaluation_memo!(d)
        GC.gc()
    end
end

"""Clear and rewarm memoization cache periodically during fixed-point iterations."""
function maybe_clear_memo_tables_fp_iter!(iter::Int, clear_every::Int, d::Domain)
    if clear_every > 0 && (iter % clear_every == 0)
        Memoization.empty_all_caches!()
        warmup_evaluation_memo!(d)
    end
end

"""Compute maximum dt satisfying CFL criterion from particles."""
function compute_cfl_dt_from_particles(p::Particles, d::Domain, target_cfl::T) where {T<:AbstractFloat}
    nx, ny = d.nel
    Lx, Ly = d.box_size
    dx = Lx / nx;  dy = Ly / ny

    max_rate = zero(T)
    @inbounds for i in 1:length(p.mx)
        u = p.mx[i];  v = p.my[i]
        rate = abs(u) / dx + abs(v) / dy
        rate > max_rate && (max_rate = rate)
    end
    max_rate <= eps(Float64) && return Inf
    return target_cfl / max_rate
end

"""Recheck CFL condition from grid velocity and update dt if needed."""
function recheck_cfl_dt(u_coeffs::AbstractVector{Float64}, d::Domain, target_cfl::Float64, cur_dt::Float64)
    nx, ny = d.nel
    Lx, Ly = d.box_size
    dx = Lx / nx;  dy = Ly / ny

    max_abs = 0.0
    @inbounds for c in u_coeffs
        a = abs(c)
        a > max_abs && (max_abs = a)
    end
    max_abs <= eps(Float64) && return cur_dt
    candidate = target_cfl * min(dx, dy) / max_abs
    return min(cur_dt, candidate)
end

"""Enforce CFL recheck and adapt dt if necessary."""
function enforce_cfl_recheck!(cur_dt::Float64, dt_check::Float64, cfg::SimulationConfig)
    threshold = cur_dt * (1.0 - cfg.cfl_recheck_tolerance)
    if dt_check >= threshold
        return cur_dt
    end
    if cfg.cfl_adaptive
        @info "CFL recheck shrinking dt: $(cur_dt) → $(dt_check) (tolerance=$(cfg.cfl_recheck_tolerance))"
        return dt_check
    else
        @error "CFL recheck failed: grid velocity suggests dt=$(dt_check) " *
               "but cur_dt=$(cur_dt) (threshold=$(threshold), " *
               "tolerance=$(cfg.cfl_recheck_tolerance)). Set " *
               "cfg.cfl_adaptive=true to allow the simulation to shrink dt automatically."
        return cur_dt
    end
end

"""Run startup sanity checks on De Rham complex."""
function run_startup_smoke_tests(d::Domain)
    ndof_R0 = FunctionSpaces.get_num_basis(d.R0.fem_space)
    ndof_R1 = FunctionSpaces.get_num_basis(d.R1.fem_space)
    ndof_R2 = FunctionSpaces.get_num_basis(d.R2.fem_space)
    println("  Smoke test DOFs: R0=$ndof_R0, R1=$ndof_R1, R2=$ndof_R2")
    @assert ndof_R0 > 0 && ndof_R1 > 0 && ndof_R2 > 0 "Smoke test: empty DOF space"

    c0  = fill(1.0, ndof_R0)
    f0  = Forms.build_form_field(d.R0, c0)
    df0 = Forms.d(f0)
    qrule = Quadrature.get_canonical_quadrature_rule(d.dΩ)
    df0_norm = 0.0
    for elem_id in 1:Geometry.get_num_elements(d.geo)
        df0_eval, _ = Forms.evaluate(df0, elem_id, Quadrature.get_nodes(qrule))
        for component in df0_eval
            df0_norm = max(df0_norm, maximum(abs.(component)))
        end
    end
    println("  Smoke test ||d(1_R0)||_max = $(Printf.@sprintf("%.3e", df0_norm))")
    @assert df0_norm < 1e-8 "Smoke test: d(constant) not zero (got $df0_norm)"

    c2       = fill(1.0, ndof_R2)
    f2       = Forms.build_form_field(d.R2, c2)
    integral = integrate_form_expression(∫(f2, d.dΩ), d)
    expected = d.box_size[1] * d.box_size[2]
    err = abs(integral - expected)
    println("  Smoke test ∫1 dΩ = $integral (expected $expected, err=$(Printf.@sprintf("%.3e", err)))")
    @assert err < 1e-6 "Smoke test: integral consistency failed"
    return nothing
end

"""
Lightweight simulation runner used by the diagnostic / convergence /
shedding / cavity tests. Returns the conservation history and the final
grid coefficients, plus exposes the running `domain` and `particles` to
the caller via the named tuple.

Compared to `main`, this version:
  * supports a user-supplied `fixed_dt` (required for convergence sweeps —
    CFL adaptation must be disabled there so dt is constant across runs);
  * fires three optional callbacks per step:
      - `step_hook(particles, domain, cfg, step, t)` — runs *after* the
        CO-FLIP advance and any FTLE reset, before the CFL recheck. Used
        by the lid-cavity case to re-stamp the tangential lid velocity.
      - `probe_callback(t, u_coeffs, domain, step)` — runs every step.
        Used by the Von Kármán case to sample velocity at downstream
        probe points each step (FFT later → Strouhal).
      - `sample_callback(t, u_coeffs, domain, particles, step)` — runs
        the first time `t` crosses each requested `sample_times[k]`.
        Used by the TG convergence study to log L2 error at t=2, t=5.
  * VTK frequency controlled by `save_vtk_every` independently of the
    history `record_every`.

The runner mirrors `main`'s structure exactly (initial coadjoint P2G,
particle resampling from grid, identical step pipeline) so behaviour
matches the production simulation.
"""
function run_diagnostic_simulation(
        cfg::SimulationConfig;
        save_vtk::Bool=true,
        save_vtk_every::Int=1,
        output_dir::String=joinpath(get(ENV, "OUTPUT_DIR", pwd()), "diag_output"),
        record_every::Int=1,
        case_name::String="",
        fixed_dt::Union{Float64,Nothing}=nothing,
        step_hook::Union{Nothing,Function}=nothing,
        probe_callback::Union{Nothing,Function}=nothing,
        sample_times::Vector{Float64}=Float64[],
        sample_callback::Union{Nothing,Function}=nothing,
        save_particles::Bool=false,
        verbose::Bool=true,
    )
    LinearAlgebra.BLAS.set_num_threads(1)
    save_vtk && mkpath(output_dir)

    num_particles = cfg.nel[1] * cfg.nel[2] * cfg.particles_per_cell
    domain    = GenerateDomain(cfg.nel, cfg.p, cfg.k;
                               box_size=cfg.box_size,
                               starting_point=cfg.starting_point,
                               boundary_condition=cfg.boundary_condition,
                               obstacle=cfg.obstacle)
    particles = generate_particles(num_particles, domain, cfg.flow_type;
                                   stratified_seeding=cfg.stratified_seeding,
                                   rng_seed=cfg.rng_seed,
                                   volume_convention=cfg.volume_convention,
                                   boundary_condition=cfg.boundary_condition,
                                   U_inf=cfg.inlet_U_inf)
    apply_obstacle_brinkman!(particles, domain)
    if step_hook !== nothing
        step_hook(particles, domain, cfg, 0, 0.0)
    end
    particle_sorter!(particles, domain)

    dt = NaN
    n_steps = 0
    if fixed_dt !== nothing
        isfinite(fixed_dt) && fixed_dt > 0 ||
            error("run_diagnostic_simulation: fixed_dt must be a finite positive Float64")
        dt = fixed_dt
        n_steps = max(1, ceil(Int, cfg.T_final / dt))
    else
        dt_cfl = compute_cfl_dt_from_particles(particles, domain, cfg.target_cfl)
        if !isfinite(dt_cfl)
            n_steps = 1;  dt = cfg.T_final
        else
            n_steps = max(1, ceil(Int, cfg.T_final / dt_cfl))
            dt      = cfg.T_final / n_steps
        end
    end

    verbose && println(@sprintf("  [%s] T_final=%.4f  dt=%.6f  n_steps=%d  ν=%.4g",
                                case_name, cfg.T_final, dt, n_steps, cfg.viscosity))

    ndofs        = FunctionSpaces.get_num_basis(domain.R1.fem_space)
    num_elements = prod(domain.nel)
    sim_buf      = SimulationBuffers(num_particles, ndofs, num_elements, domain)

    u_coeffs = coadjoint_step!(particles, domain, sim_buf;
        lsqr_atol                    = cfg.lsqr_atol,
        lsqr_btol                    = cfg.lsqr_btol,
        lsqr_maxiter                 = cfg.lsqr_maxiter,
        lsqr_error_on_nonconvergence = cfg.lsqr_error_on_nonconvergence,
        projection_mean_subtract     = cfg.projection_mean_subtract)
    u_initial = copy(u_coeffs)

    u_form_init      = Forms.build_form_field(domain.R1, u_coeffs; label="u_h")
    u_phys_expr_init = ★(u_form_init)
    u_phys_form_init = Assemblers.solve_L2_projection(domain.R1, u_phys_expr_init, domain.dΩ)

    particle_sorter!(particles, domain)
    nx, ny = domain.nel
    Lx, Ly = domain.box_size
    dx_dom = Lx / nx;  dy_dom = Ly / ny
    @inbounds for pid in eachindex(particles.x)
        eid = particles.elem_ids[pid]
        ej  = ((eid - 1) ÷ nx) + 1
        ei  = eid - (ej - 1) * nx
        set_g2p_velocity(particles, pid, particles.x[pid], particles.y[pid],
                         eid, (ei - 1) * dx_dom, (ej - 1) * dy_dom, dx_dom, dy_dom,
                         u_phys_form_init, Lx, Ly)
    end
    apply_obstacle_brinkman!(particles, domain)
    if step_hook !== nothing
        step_hook(particles, domain, cfg, 0, 0.0)
    end

    if save_vtk
        d_u_phys_0 = Forms.d(u_phys_form_init)
        ω_h_0      = ★(d_u_phys_0)
        Plot.export_form_fields_to_vtk((u_form_init,), "u_h_0000"; output_directory_tree=[output_dir])
        Plot.export_form_fields_to_vtk((ω_h_0,),       "w_h_0000"; output_directory_tree=[output_dir])
        save_particles && export_particles_to_vtk(particles, joinpath(output_dir, "particles_0000"))
    end

    warmup_evaluation_memo!(domain)

    d0 = compute_conservation_diagnostics(u_coeffs, domain)
    history = [(t=0.0, energy=d0.energy, enstrophy=d0.enstrophy, circulation=d0.circulation)]

    if probe_callback !== nothing
        probe_callback(0.0, u_coeffs, domain, 0)
    end

    sample_done = falses(length(sample_times))

    t_now = 0.0
    for step in 1:n_steps
        verbose && println(@sprintf("Step %d/%d  t≈%.4f  case=%s",
                                    step, n_steps, step*dt, case_name))

        if mod(step, 5) == 0
            physical_spatial_sort!(particles, domain)
        end

        u_coeffs = step_co_flip!(particles, domain, dt, u_coeffs, sim_buf, cfg)
        t_now    = step * dt

        u_form      = Forms.build_form_field(domain.R1, u_coeffs; label="u_h")
        u_phys_expr = ★(u_form)
        u_phys_form = Assemblers.solve_L2_projection(domain.R1, u_phys_expr, domain.dΩ)

        if cfg.delayed_reinit_frequency > 0 && step % cfg.delayed_reinit_frequency == 0
            global_reseed_from_grid!(particles, domain, cfg, u_phys_form; rng_seed=cfg.rng_seed)
        end
        apply_ftle_reset!(particles, domain, u_phys_form, cfg)

        if step_hook !== nothing
            step_hook(particles, domain, cfg, step, t_now)
        end

        # CFL recheck is suppressed when dt is user-fixed (convergence sweeps).
        if fixed_dt === nothing
            dt_check = recheck_cfl_dt(u_coeffs, domain, cfg.target_cfl, dt)
            new_dt   = enforce_cfl_recheck!(dt, dt_check, cfg)
            if new_dt != dt
                dt = new_dt
            end
        end

        if probe_callback !== nothing
            probe_callback(t_now, u_coeffs, domain, step)
        end

        if sample_callback !== nothing && !isempty(sample_times)
            @inbounds for s in eachindex(sample_times)
                if !sample_done[s] && t_now >= sample_times[s]
                    sample_callback(t_now, u_coeffs, domain, particles, step)
                    sample_done[s] = true
                end
            end
        end

        if step % record_every == 0
            d = compute_conservation_diagnostics(u_coeffs, domain)
            push!(history, (t=t_now, energy=d.energy,
                            enstrophy=d.enstrophy, circulation=d.circulation))

            if save_vtk && save_vtk_every > 0 && step % save_vtk_every == 0
                d_u_phys = Forms.d(u_phys_form)
                ω_h      = ★(d_u_phys)
                Plot.export_form_fields_to_vtk((u_form,),
                    @sprintf("u_h_%04d", step); output_directory_tree=[output_dir])
                Plot.export_form_fields_to_vtk((ω_h,),
                    @sprintf("w_h_%04d", step); output_directory_tree=[output_dir])
                save_particles && export_particles_to_vtk(particles,
                    joinpath(output_dir, @sprintf("particles_%04d", step)))
            end
        end

        maybe_clear_memo_tables!(step, cfg.clear_memo_every, domain)
    end

    return (history=history, u_initial=u_initial, u_final=u_coeffs,
            domain=domain, particles=particles, dt=dt, n_steps=n_steps,
            sample_times=sample_times, sample_done=sample_done)
end

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

"""
Estimate the dominant frequency of a probe v(t) signal via a brute-force
periodogram over candidate frequencies. Returns `(f, St, power)` where
`St = f·D/U_inf` is the Strouhal number. Uses the second half of the
record by default to skip the start-up transient.

This is intentionally simple (no FFTW dependency). For a thesis-quality
spectrum the user should re-process the saved probe CSV in
Python/MATLAB; this estimate is meant as an in-run sanity check.
"""
function estimate_strouhal_from_v_signal(
        ts::Vector{Float64}, vs::Vector{Float64};
        U_inf::Float64=1.0, D::Float64=2.0,
        skip_fraction::Float64=0.5, n_freq::Int=4096,
    )
    n  = length(ts)
    n  >= 16 || return (f=NaN, St=NaN, power=0.0)
    n1 = max(1, floor(Int, skip_fraction * n))
    t_seg = ts[n1:end]
    v_seg = vs[n1:end]
    Nu = length(t_seg)
    t0, tf = first(t_seg), last(t_seg)
    window = tf - t0
    window > 0 || return (f=NaN, St=NaN, power=0.0)
    dt_avg = window / max(Nu - 1, 1)

    v_mean = sum(v_seg) / Nu
    v_c    = v_seg .- v_mean

    f_min = 1.0 / window
    f_max = 0.5 / dt_avg
    f_max <= f_min && return (f=NaN, St=NaN, power=0.0)

    df    = (f_max - f_min) / (n_freq - 1)
    best_power = 0.0
    best_f     = f_min
    for k in 0:(n_freq - 1)
        f = f_min + k * df
        ω = 2π * f
        a = 0.0;  b = 0.0
        @inbounds for j in 1:Nu
            tj = t_seg[j]
            a += v_c[j] * cos(ω * tj)
            b += v_c[j] * sin(ω * tj)
        end
        power = a*a + b*b
        if power > best_power
            best_power = power
            best_f     = f
        end
    end
    return (f=best_f, St=best_f * D / U_inf, power=best_power)
end

"""
Comprehensive validation of the FEEC viscous diffusion implementation against
the decaying Taylor–Green vortex.

For the 2π×2π periodic box with U₀=1, kx=ky=1 the exact Navier–Stokes solution is

    u(x,y,t) =  sin(x)·cos(y)·exp(-2νt)
    v(x,y,t) = -cos(x)·sin(y)·exp(-2νt)
    ω(x,y,t) =  2·sin(x)·sin(y)·exp(-2νt)

with conserved-but-decaying integrals

    E(t) = E₀ · exp(-2ν·(kx²+ky²)·t)         (E₀ = π² for these constants)
    Z(t) = Z₀ · exp(-2ν·(kx²+ky²)·t)         (Z₀ = 2π²)
    Γ    ≡ 0                                  (periodic ⇒ zero circulation)

Runs:
  1. Viscous case (cfg.viscosity = `viscosity`, decaying_tg IC, periodic BCs)
  2. Inviscid baseline (ν = 0) — should preserve energy within CO-FLIP's usual
     conservation tolerance, isolating numerical drift from viscous dissipation.

Reports:
  • Initial E and Z vs analytical π² and 2π².
  • Per-step relative error of E(t), Z(t) vs the analytical exponential.
  • Best-fit decay rate from log-linear regression of E and Z, compared to
    the analytical 2ν(kx²+ky²).
  • Circulation drift (should remain ≈ 0).
  • Inviscid control drift (sanity check that ν=0 → no extra decay).

Saves a CSV of (t, E_num, E_exact, Z_num, Z_exact, Γ) and a 3-panel plot of
E, Z, and log(E) vs time in `output_dir`.

Returns a NamedTuple with all numerics for downstream automated checks.
"""
function test_decaying_taylor_green(;
        nel::NTuple{2,Int}=(96, 96),
        particles_per_cell::Int=16,
        viscosity::Float64=0.2,
        T_final::Float64=10.0,
        target_cfl::Float64=0.5,
        output_dir::String="decaying_tg_test",
        save_vtk::Bool=true,
        run_inviscid_baseline::Bool=true,
    )
    mkpath(output_dir)

    println("\n========== DECAYING TAYLOR-GREEN VISCOUS VALIDATION ==========")
    @printf("  nel = %s, particles/cell = %d\n", string(nel), particles_per_cell)
    @printf("  ν   = %.6g, T_final = %.3f, target_cfl = %.3f\n",
            viscosity, T_final, target_cfl)

    box_size = (2π, 2π)
    kx, ky   = 1.0, 1.0
    decay_rate_exact = 2 * viscosity * (kx^2 + ky^2)      # exponent for E and Z
    E_exact_init     = π^2                                # ½∫|u₀|² over 2π×2π
    Z_exact_init     = 2 * π^2                            # ½∫ω₀² (ω₀ = 2 sin x sin y)

    println("\n--- Running viscous case (ν=$(viscosity)) ---")
    cfg_v = SimulationConfig(;
        flow_type           = :decaying_tg,
        boundary_condition  = :periodic,
        obstacle            = nothing,
        viscosity           = viscosity,
        T_final             = T_final,
        nel                 = nel,
        particles_per_cell  = particles_per_cell,
        target_cfl          = target_cfl,
        box_size            = box_size,
        output_every        = 0,
    )
    result_v = run_diagnostic_simulation(cfg_v;
        save_vtk     = save_vtk,
        output_dir   = joinpath(output_dir, "viscous"),
        record_every = 1,
        case_name    = "decaying_tg_viscous",
    )

    result_inv = nothing
    if run_inviscid_baseline
        println("\n--- Running inviscid baseline (ν=0.0) ---")
        cfg_inv = SimulationConfig(;
            flow_type           = :decaying_tg,
            boundary_condition  = :periodic,
            obstacle            = nothing,
            viscosity           = 0.0,
            T_final             = T_final,
            nel                 = nel,
            particles_per_cell  = particles_per_cell,
            target_cfl          = target_cfl,
            box_size            = box_size,
            output_every        = 0,
        )
        result_inv = run_diagnostic_simulation(cfg_inv;
            save_vtk     = true,
            output_dir   = joinpath(output_dir, "inviscid"),
            record_every = 1,
            case_name    = "decaying_tg_inviscid",
        )
    end

    ts      = [hi.t           for hi in result_v.history]
    E_num   = [hi.energy      for hi in result_v.history]
    Z_num   = [hi.enstrophy   for hi in result_v.history]
    Γ_num   = [hi.circulation for hi in result_v.history]
    E_exact = [E_num[1] * exp(-decay_rate_exact * t) for t in ts]
    Z_exact = [Z_num[1] * exp(-decay_rate_exact * t) for t in ts]

    E0_err       = abs(E_num[1] - E_exact_init) / E_exact_init
    Z0_err       = abs(Z_num[1] - Z_exact_init) / Z_exact_init
    E_final_err  = abs(E_num[end] - E_exact[end]) / E_exact[end]
    Z_final_err  = abs(Z_num[end] - Z_exact[end]) / Z_exact[end]
    rel_err_E    = [abs(E_num[i] - E_exact[i]) / E_exact[i] for i in eachindex(ts)]
    rel_err_Z    = [abs(Z_num[i] - Z_exact[i]) / Z_exact[i] for i in eachindex(ts)]
    mean_err_E   = sum(rel_err_E) / length(rel_err_E)
    mean_err_Z   = sum(rel_err_Z) / length(rel_err_Z)
    max_err_E    = maximum(rel_err_E)
    max_err_Z    = maximum(rel_err_Z)
    Γ_max_abs    = maximum(abs.(Γ_num))

    # Best-fit decay rate from log-linear regression (slope of log(y) vs t).
    fit_log_slope = function (t, y)
        n     = length(t)
        ly    = log.(max.(y, 1e-300))
        tbar  = sum(t) / n
        ybar  = sum(ly) / n
        num   = sum((t .- tbar) .* (ly .- ybar))
        den   = sum((t .- tbar).^2)
        return num / den
    end
    slope_E_fit  = -fit_log_slope(ts, E_num)
    slope_Z_fit  = -fit_log_slope(ts, Z_num)
    slope_err_E  = abs(slope_E_fit - decay_rate_exact) / max(decay_rate_exact, eps())
    slope_err_Z  = abs(slope_Z_fit - decay_rate_exact) / max(decay_rate_exact, eps())

    inviscid_E_drift = nothing
    inviscid_Z_drift = nothing
    if !isnothing(result_inv)
        E_inv = [hi.energy    for hi in result_inv.history]
        Z_inv = [hi.enstrophy for hi in result_inv.history]
        inviscid_E_drift = abs(E_inv[end] - E_inv[1]) / max(abs(E_inv[1]), eps())
        inviscid_Z_drift = abs(Z_inv[end] - Z_inv[1]) / max(abs(Z_inv[1]), eps())
    end

    println("\n========== VALIDATION REPORT ==========")
    @printf("Analytical decay rate (energy & enstrophy): %.6f  (half-life %.3f)\n",
            decay_rate_exact, decay_rate_exact > 0 ? log(2)/decay_rate_exact : Inf)
    println()
    @printf("Initial energy:      num=%.6e | exact=%.6e | rel err=%.3e\n",
            E_num[1], E_exact_init, E0_err)
    @printf("Initial enstrophy:   num=%.6e | exact=%.6e | rel err=%.3e\n",
            Z_num[1], Z_exact_init, Z0_err)
    println()
    @printf("Final energy:        num=%.6e | exact=%.6e | rel err=%.3e\n",
            E_num[end], E_exact[end], E_final_err)
    @printf("Final enstrophy:     num=%.6e | exact=%.6e | rel err=%.3e\n",
            Z_num[end], Z_exact[end], Z_final_err)
    println()
    @printf("Fit decay rate E:    %.6f  (target %.6f, rel err %.3e)\n",
            slope_E_fit, decay_rate_exact, slope_err_E)
    @printf("Fit decay rate Z:    %.6f  (target %.6f, rel err %.3e)\n",
            slope_Z_fit, decay_rate_exact, slope_err_Z)
    println()
    @printf("Mean rel err E(t):   %.3e   (max %.3e)\n", mean_err_E, max_err_E)
    @printf("Mean rel err Z(t):   %.3e   (max %.3e)\n", mean_err_Z, max_err_Z)
    println()
    if !isnothing(inviscid_E_drift)
        @printf("Inviscid ΔE/E₀:      %.3e   (should be ≪ %.3e from viscosity)\n",
                inviscid_E_drift, 1 - exp(-decay_rate_exact * T_final))
        @printf("Inviscid ΔZ/Z₀:      %.3e\n", inviscid_Z_drift)
    end
    @printf("max |Γ| over run:    %.3e   (should be ≈ 0)\n", Γ_max_abs)

    csv_path = joinpath(output_dir, "decaying_tg_diagnostics.csv")
    open(csv_path, "w") do io
        println(io, "t,E_num,E_exact,Z_num,Z_exact,circulation")
        for i in eachindex(ts)
            @printf(io, "%.10e,%.16e,%.16e,%.16e,%.16e,%.16e\n",
                    ts[i], E_num[i], E_exact[i], Z_num[i], Z_exact[i], Γ_num[i])
        end
    end
    println("\nSaved per-step CSV: $csv_path")

    try
        p_E = plot(ts, E_num, label="numerical", lw=2,
                   xlabel="t", ylabel="energy",
                   title="Decaying TG (ν=$(viscosity)): energy")
        plot!(p_E, ts, E_exact, label="analytical exp(-$(round(decay_rate_exact,digits=4))·t)",
              ls=:dash, lw=2)
        if !isnothing(result_inv)
            ts_i = [hi.t for hi in result_inv.history]
            E_i  = [hi.energy for hi in result_inv.history]
            plot!(p_E, ts_i, E_i, label="inviscid (ν=0)", ls=:dot, lw=1.5)
        end

        p_Z = plot(ts, Z_num, label="numerical", lw=2,
                   xlabel="t", ylabel="enstrophy",
                   title="Decaying TG: enstrophy")
        plot!(p_Z, ts, Z_exact, label="analytical", ls=:dash, lw=2)

        p_logE = plot(ts, log.(max.(E_num, 1e-300)), label="log(E_num)", lw=2,
                      xlabel="t", ylabel="log(E)",
                      title="log-energy slope ≈ -$(round(slope_E_fit,digits=4)) (target -$(round(decay_rate_exact,digits=4)))")
        plot!(p_logE, ts, log.(E_exact), label="log(E_exact)", ls=:dash, lw=2)

        layout = plot(p_E, p_Z, p_logE, layout=(3, 1), size=(800, 1000))
        plot_path = joinpath(output_dir, "decaying_tg_decay.png")
        savefig(layout, plot_path)
        println("Saved plot:         $plot_path")
    catch err
        @warn "Plot save failed (continuing)" exception=err
    end

    return (
        history          = result_v.history,
        ts               = ts,
        E_num            = E_num,
        E_exact          = E_exact,
        Z_num            = Z_num,
        Z_exact          = Z_exact,
        circulation      = Γ_num,
        decay_rate_exact = decay_rate_exact,
        decay_rate_fit_E = slope_E_fit,
        decay_rate_fit_Z = slope_Z_fit,
        slope_err_E      = slope_err_E,
        slope_err_Z      = slope_err_Z,
        E0_err           = E0_err,
        Z0_err           = Z0_err,
        E_final_err      = E_final_err,
        Z_final_err      = Z_final_err,
        mean_err_E       = mean_err_E,
        max_err_E        = max_err_E,
        mean_err_Z       = mean_err_Z,
        max_err_Z        = max_err_Z,
        Γ_max_abs        = Γ_max_abs,
        inviscid_E_drift = inviscid_E_drift,
        inviscid_Z_drift = inviscid_Z_drift,
        result_viscous   = result_v,
        result_inviscid  = result_inv,
    )
end

# ============================================================================
# THESIS RESULT CASES
# ============================================================================
# Three thesis-oriented test functions plus an env-driven dispatcher at the
# bottom of the file. Each function runs **one** independent simulation
# point and writes its CSV/VTK output under `output_dir`; the user submits
# multiple cluster jobs (typically a SLURM job array) to fill out a sweep.

const TG_NEL_SPACE_SWEEP    = (32, 48, 64, 96, 128)
const TG_DT_TIME_SWEEP      = (0.04, 0.02, 0.01, 0.005, 0.0025)
const TG_FIXED_NEL_FOR_TIME = 128
const TG_FIXED_DT_FOR_SPACE = 0.005

"""
Run ONE point of the decaying Taylor-Green convergence study.

`mode` selects whether this run is part of the spatial sweep (fixed dt,
varying nel) or the temporal sweep (fixed nel, varying dt). `sweep_idx`
picks the point within the corresponding sweep array
(`TG_NEL_SPACE_SWEEP` / `TG_DT_TIME_SWEEP`).

Each run samples the L2 velocity error against the analytical viscous
Navier-Stokes solution at `t=2` and `t=5`, and writes a CSV with one
row per sample plus a per-step history CSV. Cluster post-processing
should gather every `*_errors.csv` from one sweep and plot the resulting
convergence curve.

Defaults: ν=0.05 on the 2π×2π periodic box, T_final=5.0, p=(3,3) splines,
CFL adaptation off (so dt is constant across all spatial-sweep points).
"""
function test_decaying_tg_convergence(;
        mode::Symbol = :space,
        sweep_idx::Int = 1,
        output_dir::String = joinpath(get(ENV, "OUTPUT_DIR", pwd()), "tg_convergence"),
        viscosity::Float64 = 0.05,
        T_final::Float64 = 5.0,
        sample_times::Vector{Float64} = [2.0, 5.0],
        nel_space_sweep = TG_NEL_SPACE_SWEEP,
        dt_time_sweep   = TG_DT_TIME_SWEEP,
        fixed_nel_for_time::Int   = TG_FIXED_NEL_FOR_TIME,
        fixed_dt_for_space::Float64 = TG_FIXED_DT_FOR_SPACE,
        particles_per_cell::Int = 16,
        save_vtk::Bool = false,
        save_vtk_at_samples::Bool = true,
        nq_extra::Int = 2,
    )
    mkpath(output_dir)
    box_size = (2π, 2π)

    if mode === :space
        1 <= sweep_idx <= length(nel_space_sweep) ||
            error("sweep_idx=$sweep_idx out of range for space sweep (1:$(length(nel_space_sweep)))")
        nel_val = nel_space_sweep[sweep_idx]
        dt_val  = fixed_dt_for_space
        run_label = @sprintf("space_nel%03d_dt%.5f", nel_val, dt_val)
    elseif mode === :time
        1 <= sweep_idx <= length(dt_time_sweep) ||
            error("sweep_idx=$sweep_idx out of range for time sweep (1:$(length(dt_time_sweep)))")
        nel_val = fixed_nel_for_time
        dt_val  = dt_time_sweep[sweep_idx]
        run_label = @sprintf("time_nel%03d_dt%.5f", nel_val, dt_val)
    else
        error("mode must be :space or :time, got $mode")
    end

    println("\n========== DECAYING-TG CONVERGENCE ==========")
    println(@sprintf("  mode=%s  sweep_idx=%d  →  nel=(%d,%d)  dt=%.5f  ν=%.4f  T=%.2f",
                     mode, sweep_idx, nel_val, nel_val, dt_val, viscosity, T_final))
    println("  Output dir: $output_dir")

    cfg = SimulationConfig(;
        flow_type           = :decaying_tg,
        boundary_condition  = :periodic,
        obstacle            = nothing,
        viscosity           = viscosity,
        T_final             = T_final,
        nel                 = (nel_val, nel_val),
        particles_per_cell  = particles_per_cell,
        box_size            = box_size,
        target_cfl          = 0.5,
        cfl_adaptive        = false,
        projection_mean_subtract = true,
        output_every        = 0,
    )

    sample_records = NamedTuple[]
    sample_cb = function (t, u_coeffs, domain, particles, step)
        err  = compute_decaying_tg_l2_error(domain, u_coeffs, t, viscosity;
                                            nq_extra=nq_extra)
        diag = compute_conservation_diagnostics(u_coeffs, domain)
        rec  = (t=t, step=step, nel=nel_val, dt=dt_val,
                l2_err=err.l2_err, rel_l2_err=err.rel_l2_err,
                l2_norm_exact=err.l2_norm_exact,
                energy=diag.energy, enstrophy=diag.enstrophy,
                circulation=diag.circulation)
        push!(sample_records, rec)
        @printf("  [SAMPLE] t=%.3f  L2_err=%.6e  rel_L2_err=%.6e  ||u_exact||=%.6e  E=%.6e  Z=%.6e\n",
                t, err.l2_err, err.rel_l2_err, err.l2_norm_exact,
                diag.energy, diag.enstrophy)

        if save_vtk_at_samples
            u_form      = Forms.build_form_field(domain.R1, u_coeffs; label="u_h")
            u_phys_form = Assemblers.solve_L2_projection(domain.R1, ★(u_form), domain.dΩ)
            ω_h         = ★(Forms.d(u_phys_form))
            t_tag = @sprintf("t%05.2f", t)
            Plot.export_form_fields_to_vtk((u_form,),
                run_label * "_u_" * t_tag; output_directory_tree=[output_dir])
            Plot.export_form_fields_to_vtk((ω_h,),
                run_label * "_w_" * t_tag; output_directory_tree=[output_dir])
        end
    end

    result = run_diagnostic_simulation(cfg;
        save_vtk        = save_vtk,
        save_vtk_every  = 0,
        output_dir      = joinpath(output_dir, run_label * "_history"),
        record_every    = 5,
        case_name       = run_label,
        fixed_dt        = dt_val,
        sample_times    = sample_times,
        sample_callback = sample_cb,
    )

    csv_path = joinpath(output_dir, run_label * "_errors.csv")
    open(csv_path, "w") do io
        println(io, "mode,sweep_idx,nel,dt,viscosity,t,step,l2_err,rel_l2_err,l2_norm_exact,energy,enstrophy,circulation")
        for r in sample_records
            @printf(io, "%s,%d,%d,%.10e,%.10e,%.6f,%d,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e\n",
                    String(mode), sweep_idx, r.nel, r.dt, viscosity,
                    r.t, r.step, r.l2_err, r.rel_l2_err, r.l2_norm_exact,
                    r.energy, r.enstrophy, r.circulation)
        end
    end
    println("Saved per-sample errors: $csv_path")

    history_csv = joinpath(output_dir, run_label * "_history.csv")
    open(history_csv, "w") do io
        println(io, "t,energy,enstrophy,circulation")
        for hi in result.history
            @printf(io, "%.10e,%.16e,%.16e,%.16e\n",
                    hi.t, hi.energy, hi.enstrophy, hi.circulation)
        end
    end
    println("Saved per-step history: $history_csv")

    return (result=result, records=sample_records, run_label=run_label,
            csv_path=csv_path, history_csv=history_csv)
end

"""
Spatial convergence study of the P2G (particle-to-grid) least-squares
reconstruction at the **initial step**, on the periodic Taylor-Green
vortex.

For each `(p_order, nel)` pair this routine:
  1. Builds a periodic 2π×2π domain with `nel × nel` elements and B-spline
     order `p_order` (continuity k=1).
  2. Seeds particles via `seeding_mode`. Default is `:gauss_legendre`
     (tensor-product GL nodes per element with weights set as particle
     volumes), which turns the LSQR fit into an exact polynomial
     quadrature so the discretisation error is dominated by the spline
     approximation rather than particle-quadrature noise.
  3. Assembles the P2G matrix `B` (rows = 2·num_particles) and solves the
     least-squares system `min ‖B·u_grid − m_p‖₂` via LSQR.
  4. Measures the L2 error via `compute_decaying_tg_l2_error` (at t=0,
     ν=0), which uses Mantis's `Analysis.L2_norm` on the difference
     `u_phys_form − u_TG_analytical` over a high-order Gauss–Legendre
     quadrature.

The least-squares fit alone is tested — no Hodge projection / pressure
solve — so the resulting curve isolates the P2G reconstruction error.

For the rotated R1 storage on tensor-product B-splines of order `p`, the
mixed-degree components `(p-1, p)` and `(p, p-1)` make the L2-best-
approximation rate of a smooth physical-velocity field `O(h^p)`. The
reference slope drawn in the plot is `h^p`.

Returns the list of per-run `(p, nel, h, l2_err, rel_l2_err, lsqr_resid_rel)`
NamedTuples.
"""
function test_p2g_taylor_green_convergence(;
        nel_sweep::NTuple{N,Int}=(32, 48, 64, 96, 128),
        p_orders::NTuple{M,Int}=(2, 3),
        particles_per_cell::Int=16,
        seeding_mode::Symbol=:gauss_legendre,
        rng_seed::Union{Int,Nothing}=42,
        nq_extra::Int=2,
        lsqr_atol::Float64=1e-12,
        lsqr_btol::Float64=1e-12,
        lsqr_maxiter::Int=5000,
        output_dir::String=joinpath(get(ENV, "OUTPUT_DIR", pwd()), "p2g_tg_convergence"),
        save_plot::Bool=true,
    ) where {N,M}
    mkpath(output_dir)

    box_size = (2π, 2π)
    results  = NamedTuple[]

    println("\n========== P2G TAYLOR-GREEN CONVERGENCE ==========")
    @printf("  box=%s  ppc=%d  seeding_mode=%s  rng_seed=%s  nq_extra=%d\n",
            string(box_size), particles_per_cell, seeding_mode,
            string(rng_seed), nq_extra)

    for p_order in p_orders, n in nel_sweep
        nel = (n, n)
        p   = (p_order, p_order)
        k   = (1, 1)

        println(@sprintf("\n--- p=%d  nel=%d ---", p_order, n))

        domain = GenerateDomain(nel, p, k;
                                box_size=box_size,
                                starting_point=(0.0, 0.0),
                                boundary_condition=:periodic,
                                obstacle=nothing)

        num_particles = nel[1] * nel[2] * particles_per_cell
        particles = generate_particles(num_particles, domain, :tg;
                                       stratified_seeding=true,
                                       seeding_mode=seeding_mode,
                                       rng_seed=rng_seed,
                                       volume_convention=:physical,
                                       boundary_condition=:periodic,
                                       U_inf=0.0)
        # GL seeding may round the actual count down; use the resulting array.
        num_particles = length(particles.x)
        particle_sorter!(particles, domain)

        ndofs        = FunctionSpaces.get_num_basis(domain.R1.fem_space)
        num_elements = prod(domain.nel)
        buf          = SimulationBuffers(num_particles, ndofs, num_elements, domain)

        t0 = time()
        B  = build_B_matrix(particles, domain, buf)
        t_build_B = time() - t0

        if length(buf.lsqr_warm) != size(B, 2)
            buf.lsqr_warm = zeros(size(B, 2))
        end

        t0 = time()
        u_grid = solve_grid_velocity_lsqr(B, particles, buf.V_p, buf.lsqr_warm;
                                          atol=lsqr_atol, btol=lsqr_btol,
                                          maxiter=lsqr_maxiter,
                                          error_on_nonconvergence=true)
        t_solve = time() - t0

        # LSQR residual on the actual fit problem — quantifies whether the
        # least-squares optimum was reached. Build RHS afresh because the
        # LSQR call above scaled B in place via column-Jacobi preconditioner,
        # which leaves B with unit-norm columns rather than the original B.
        m_rhs = similar(buf.V_p, 2 * length(particles.x))
        build_lsqr_rhs!(m_rhs, particles)
        B_fresh = build_B_matrix(particles, domain, buf)
        residual = B_fresh * u_grid .- m_rhs
        lsqr_resid_rel = norm(residual) / max(norm(m_rhs), eps())

        err = compute_decaying_tg_l2_error(domain, u_grid, 0.0, 0.0;
                                           nq_extra=nq_extra)

        h   = box_size[1] / n
        rec = (p=p_order, nel=n, h=h,
               l2_err=err.l2_err,
               rel_l2_err=err.rel_l2_err,
               l2_norm_exact=err.l2_norm_exact,
               lsqr_resid_rel=lsqr_resid_rel,
               num_particles=num_particles,
               t_build_B=t_build_B, t_solve=t_solve)
        push!(results, rec)
        @printf("  nel=%d h=%.5f  L2=%.4e  rel_L2=%.4e  LSQR_resid_rel=%.2e  (build_B=%.2fs solve=%.2fs)\n",
                n, h, err.l2_err, err.rel_l2_err,
                lsqr_resid_rel, t_build_B, t_solve)
    end

    println("\n========== OBSERVED CONVERGENCE ORDERS ==========")
    for p_order in p_orders
        subset = filter(r -> r.p == p_order, results)
        length(subset) >= 2 || continue
        println(@sprintf("\n  p=%d  (expected order %d)", p_order, p_order))
        @printf("  %-6s %-10s  %-12s %-8s %-8s  %-10s\n",
                "nel", "h", "L2_err", "ratio", "order", "lsqr_res")
        prev = nothing
        for r in subset
            if prev === nothing
                @printf("  %-6d %-10.5e  %-12.4e %-8s %-8s  %-10.2e\n",
                        r.nel, r.h, r.l2_err, "—", "—", r.lsqr_resid_rel)
            else
                h_ratio = prev.h / r.h
                ratio   = prev.l2_err / max(r.l2_err, eps())
                order   = log(ratio) / log(h_ratio)
                @printf("  %-6d %-10.5e  %-12.4e %-8.3f %-8.3f  %-10.2e\n",
                        r.nel, r.h, r.l2_err, ratio, order, r.lsqr_resid_rel)
            end
            prev = r
        end
    end

    csv_path = joinpath(output_dir, "p2g_tg_convergence.csv")
    open(csv_path, "w") do io
        println(io, "p,nel,h,num_particles,l2_err,rel_l2_err,l2_norm_exact,lsqr_resid_rel,t_build_B,t_solve")
        for r in results
            @printf(io, "%d,%d,%.10e,%d,%.16e,%.16e,%.16e,%.16e,%.6f,%.6f\n",
                    r.p, r.nel, r.h, r.num_particles,
                    r.l2_err, r.rel_l2_err, r.l2_norm_exact,
                    r.lsqr_resid_rel,
                    r.t_build_B, r.t_solve)
        end
    end
    println("\nSaved per-run CSV: $csv_path")

    if save_plot
        try
            plt = plot(xscale=:log10, yscale=:log10,
                       xlabel="h", ylabel="L2 error",
                       title="P2G LSQR reconstruction error vs h\n(Taylor-Green, t=0)",
                       legend=:bottomright, minorgrid=true)
            for p_order in p_orders
                subset = filter(r -> r.p == p_order, results)
                isempty(subset) && continue
                hs   = [r.h for r in subset]
                errs = [r.l2_err for r in subset]
                plot!(plt, hs, errs, marker=:circle, lw=2,
                      label="p=$(p_order)  measured")
                refs = errs[end] .* (hs ./ hs[end]) .^ p_order
                plot!(plt, hs, refs, ls=:dash, lw=1,
                      label="∝ h^$(p_order)")
            end
            plot_path = joinpath(output_dir, "p2g_tg_convergence.png")
            savefig(plt, plot_path)
            println("Saved log-log plot: $plot_path")
        catch err
            @warn "Plot save failed (continuing)" exception=err
        end
    end

    return results
end

"""
Run a Von Kármán vortex-shedding case past a cylinder and write probe
velocity data for offline Strouhal estimation.

Defaults set up Re=100 (D=2, U_inf=1, ν=0.02) on a 50×30 box with
nel=(200,120) → dx=dy=0.25. Top and bottom are :wall (free-slip) to
avoid the periodic feedthrough of the older default cfg. Three probe
points are sampled every step and dumped to CSV. A coarse periodogram
estimate of the dominant frequency is printed at the end as a sanity
check; for thesis-grade spectra, post-process the CSV with FFTW in
Python/MATLAB.
"""
function test_von_karman_strouhal(;
        output_dir::String       = joinpath(get(ENV, "OUTPUT_DIR", pwd()), "von_karman"),
        box_size::NTuple{2,Float64}        = (50.0, 30.0),
        nel::NTuple{2,Int}                 = (200, 120),
        cylinder_center::NTuple{2,Float64} = (10.0, 15.0),
        cylinder_radius::Float64           = 1.0,
        U_inf::Float64           = 1.5,
        viscosity::Float64       = 0.01,
        T_final::Float64         = 50.0,
        bc_top_bottom::Symbol    = :wall,
        probe_points::Vector{NTuple{2,Float64}} =
            [(20.0, 15.0), (25.0, 15.0), (30.0, 15.0)],
        save_vtk_every::Int      = 1,
        particles_per_cell::Int  = 10,
        target_cfl::Float64      = 0.5,
        case_label::String       = "vk_re300",
    )
    mkpath(output_dir)
    bc = (:inlet, :outlet, bc_top_bottom, bc_top_bottom)
    Re = U_inf * (2 * cylinder_radius) / viscosity

    println("\n========== VON KARMAN VORTEX SHEDDING ==========")
    println(@sprintf("  box=%s  nel=%s  cyl=%s  r=%.3f  U=%.3f  ν=%.4f  Re=%.1f  T=%.1f  bc_y=%s",
                     box_size, nel, cylinder_center, cylinder_radius,
                     U_inf, viscosity, Re, T_final, bc_top_bottom))

    cfg = SimulationConfig(;
        nel                 = nel,
        box_size            = box_size,
        boundary_condition  = bc,
        inlet_U_inf         = U_inf,
        obstacle            = (kind=:cylinder, center=cylinder_center, radius=cylinder_radius),
        flow_type           = :cylinder,
        viscosity           = viscosity,
        T_final             = T_final,
        target_cfl          = target_cfl,
        cfl_adaptive        = true,
        particles_per_cell  = particles_per_cell,
        output_every        = save_vtk_every,
    )

    n_probes = length(probe_points)
    probe_records = NamedTuple[]

    probe_cb = function (t, u_coeffs, domain, step)
        cache = EvaluationCache(evaluation_cache_size(domain))
        us = Vector{Float64}(undef, n_probes)
        vs = Vector{Float64}(undef, n_probes)
        @inbounds for k in 1:n_probes
            xp, yp = probe_points[k]
            (u, v), _ = probe_field_at_point(xp, yp, u_coeffs, domain, cache)
            us[k] = Float64(u);  vs[k] = Float64(v)
        end
        push!(probe_records, (t=t, step=step, us=us, vs=vs))
    end

    result = run_diagnostic_simulation(cfg;
        save_vtk        = true,
        save_vtk_every  = save_vtk_every,
        output_dir      = joinpath(output_dir, case_label * "_vtk"),
        record_every    = max(1, save_vtk_every),
        case_name       = case_label,
        probe_callback  = probe_cb,
        save_particles  = false,
    )

    probe_csv = joinpath(output_dir, case_label * "_probes.csv")
    open(probe_csv, "w") do io
        header = "t,step"
        for k in 1:n_probes
            header *= @sprintf(",u_p%d,v_p%d", k, k)
        end
        println(io, header)
        for r in probe_records
            line = @sprintf("%.10e,%d", r.t, r.step)
            for k in 1:n_probes
                line *= @sprintf(",%.16e,%.16e", r.us[k], r.vs[k])
            end
            println(io, line)
        end
    end
    println("Saved probe CSV: $probe_csv")

    history_csv = joinpath(output_dir, case_label * "_history.csv")
    open(history_csv, "w") do io
        println(io, "t,energy,enstrophy,circulation")
        for hi in result.history
            @printf(io, "%.10e,%.16e,%.16e,%.16e\n",
                    hi.t, hi.energy, hi.enstrophy, hi.circulation)
        end
    end
    println("Saved history CSV: $history_csv")

    if length(probe_records) > 16
        ts  = [r.t for r in probe_records]
        vs1 = [r.vs[1] for r in probe_records]
        st  = estimate_strouhal_from_v_signal(ts, vs1;
                                              U_inf=U_inf, D=2*cylinder_radius)
        println("\n--- In-run Strouhal estimate (probe 1, second half of record) ---")
        @printf("  Re=%.1f  U=%.3f  D=%.3f\n", Re, U_inf, 2*cylinder_radius)
        @printf("  f_dominant=%.6f  T_period=%.4f  St=f·D/U=%.4f\n",
                st.f, 1.0/max(st.f, eps()), st.St)

        open(joinpath(output_dir, case_label * "_strouhal.txt"), "w") do io
            println(io, @sprintf("Re=%.2f  U=%.4f  D=%.4f  ν=%.4f", Re, U_inf,
                                 2*cylinder_radius, viscosity))
            println(io, @sprintf("Probe 1 location = (%.3f, %.3f)",
                                 probe_points[1][1], probe_points[1][2]))
            println(io, @sprintf("Dominant freq f = %.6f", st.f))
            println(io, @sprintf("Period         T = %.4f", 1.0/max(st.f, eps())))
            println(io, @sprintf("Strouhal       St = %.6f", st.St))
        end
    end

    return (result=result, probes=probe_records,
            probe_csv=probe_csv, history_csv=history_csv)
end

"""
Run a lid-driven cavity case using a Brinkman-style tangential
enforcement to mimic no-slip side walls and a moving top lid.

The solver's `:wall` BC pins the wall-normal velocity trace to zero
strongly at the operator level. The tangential component is not
strongly constrained, so we re-stamp it onto the particles each step
via `apply_lid_cavity_brinkman!`. The result is a recognisable lid-
driven cavity flow but **not** strict no-slip — the boundary layer is
resolved over the Brinkman band rather than at the wall exactly. Note
this in the thesis when reporting results.

Default Re = U_lid·L/ν = 1·1/0.01 = 100.
"""

# ----------------------------------------------------------------------------
# Ghia, Ghia & Shin (1982) benchmark — lid-driven cavity centerline velocities
# ----------------------------------------------------------------------------
# Tables I & II, normalised by U_lid. Top-wall lid is +1, all other walls 0.
# Sampled along the vertical centerline (x = L/2, u(y)) and the horizontal
# centerline (y = L/2, v(x)). Only Re=100, 400, 1000 included here — the
# higher-Re columns are available in the paper and can be added later.
const GHIA_Y_OVER_L = (
    0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531,
    0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0000)
const GHIA_U_RE100 = (
     0.00000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150, -0.15662, -0.21090,
    -0.20581, -0.13641,  0.00332,  0.23151,  0.68717,  0.73722,  0.78871,  0.84123,  1.00000)
const GHIA_U_RE400 = (
     0.00000, -0.08186, -0.09266, -0.10338, -0.14612, -0.24299, -0.32726, -0.17119,
    -0.11477,  0.02135,  0.16256,  0.29093,  0.55892,  0.61756,  0.68439,  0.75837,  1.00000)
const GHIA_U_RE1000 = (
     0.00000, -0.18109, -0.20196, -0.22220, -0.29730, -0.38289, -0.27805, -0.10648,
    -0.06080,  0.05702,  0.18719,  0.33304,  0.46604,  0.51117,  0.57492,  0.65928,  1.00000)
const GHIA_X_OVER_L = (
    0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266, 0.2344,
    0.5000, 0.8047, 0.8594, 0.9063, 0.9453, 0.9531, 0.9609, 0.9688, 1.0000)
const GHIA_V_RE100 = (
    0.00000,  0.09233,  0.10091,  0.10890,  0.12317,  0.16077,  0.17507,  0.17527,
    0.05454, -0.24533, -0.22445, -0.16914, -0.10313, -0.08864, -0.07391, -0.05906,  0.00000)
const GHIA_V_RE400 = (
    0.00000,  0.18360,  0.19713,  0.20920,  0.22965,  0.28124,  0.30203,  0.30174,
    0.05186, -0.38598, -0.44993, -0.23827, -0.22847, -0.19254, -0.15663, -0.12146,  0.00000)
const GHIA_V_RE1000 = (
    0.00000,  0.27485,  0.29012,  0.30353,  0.32627,  0.37095,  0.33075,  0.32235,
    0.02526, -0.31966, -0.42665, -0.51550, -0.39188, -0.33714, -0.27669, -0.21388,  0.00000)

"""Return `(ys, us, xs, vs)` Ghia (1982) benchmark vectors for the given
Reynolds number, or `nothing` when no benchmark is tabulated. `ys`/`xs`
are in units of `L` (cavity side length); `us`/`vs` are in units of
`U_lid`."""
function ghia_reference_data(Re::Real)
    Re_int = Int(round(Re))
    if Re_int == 100
        return (collect(GHIA_Y_OVER_L), collect(GHIA_U_RE100),
                collect(GHIA_X_OVER_L), collect(GHIA_V_RE100))
    elseif Re_int == 400
        return (collect(GHIA_Y_OVER_L), collect(GHIA_U_RE400),
                collect(GHIA_X_OVER_L), collect(GHIA_V_RE400))
    elseif Re_int == 1000
        return (collect(GHIA_Y_OVER_L), collect(GHIA_U_RE1000),
                collect(GHIA_X_OVER_L), collect(GHIA_V_RE1000))
    end
    return nothing
end

"""Sample the *physical* velocity `(u, v)` at point `(x, y)` from an
already-projected `u_phys_form`. Cheaper than `probe_field_at_point`
when the caller has the L2-projected form already (and unlike the raw
R1 probe, this returns the physical-velocity components rather than the
rotated R1-form components)."""
function sample_phys_velocity_at_point(
        u_phys_form, domain::Domain, x::Float64, y::Float64,
    )
    Lx, Ly = domain.box_size
    nx, ny = domain.nel
    dx = Lx / nx;  dy = Ly / ny
    modes = bc_to_position_modes(domain.bc_sides)
    x_w = wrap_axis(Float64(x), Lx, modes[1], 1e-12)
    y_w = wrap_axis(Float64(y), Ly, modes[2], 1e-12)
    ei  = clamp(floor(Int, x_w / dx) + 1, 1, nx)
    ej  = clamp(floor(Int, y_w / dy) + 1, 1, ny)
    eid = (ej - 1) * nx + ei
    xi  = (x_w - (ei - 1) * dx) / dx
    eta = (y_w - (ej - 1) * dy) / dy
    sample_points = Mantis.Points.CartesianPoints(([xi], [eta]))
    pushfwd_eval, _ = Forms.evaluate_sharp_pushforward(u_phys_form, eid, sample_points)
    u = reduce(+, pushfwd_eval[1], dims=2)[1]
    v = reduce(+, pushfwd_eval[2], dims=2)[1]
    return u, v
end

"""Extract `u(y)` along the vertical centerline `x = L/2` and `v(x)` along
the horizontal centerline `y = L/2`, sampled uniformly with `n_samples`
points. Used both for full-resolution centerline CSVs and for Ghia-point
sampling. Returns `(ys, us, xs, vs)`."""
function extract_centerline_profiles(
        u_coeffs::AbstractVector{Float64}, domain::Domain;
        n_samples::Int = 401,
    )
    Lx, Ly = domain.box_size
    x_mid = 0.5 * Lx
    y_mid = 0.5 * Ly

    u_form      = Forms.build_form_field(domain.R1, u_coeffs)
    u_phys_form = Assemblers.solve_L2_projection(domain.R1, ★(u_form), domain.dΩ)

    ys = collect(range(0.0, Ly, length=n_samples))
    xs = collect(range(0.0, Lx, length=n_samples))
    us = Vector{Float64}(undef, n_samples)
    vs = Vector{Float64}(undef, n_samples)

    @inbounds for k in 1:n_samples
        u, _ = sample_phys_velocity_at_point(u_phys_form, domain, x_mid, ys[k])
        us[k] = u
    end
    @inbounds for k in 1:n_samples
        _, v = sample_phys_velocity_at_point(u_phys_form, domain, xs[k], y_mid)
        vs[k] = v
    end
    return ys, us, xs, vs
end

"""
Validate a cavity simulation against the Ghia, Ghia, Shin (1982)
benchmark for a matching Reynolds number.

Steps:
  1. Build the L2-projected physical-velocity form from `u_coeffs`.
  2. Sample `u/U_lid` at the 17 Ghia vertical-centerline `y/L` points
     and `v/U_lid` at the 17 horizontal-centerline `x/L` points.
  3. Compute pointwise abs/rel differences and trapezoidal L2 / max
     errors against the tabulated reference.
  4. Also extract dense centerline profiles (`n_dense` samples per axis)
     for full-curve plots alongside the 17 reference points.
  5. Write three CSVs (`*_ghia_u_centerline.csv`, `*_ghia_v_centerline.csv`,
     `*_centerline_{u,v}_dense.csv`) and a `*_ghia_summary.txt` file
     with the summary errors.

Returns `nothing` when no Ghia data exists for the rounded `Re`.
"""
function validate_lid_cavity_against_ghia(
        u_coeffs::AbstractVector{Float64}, domain::Domain;
        Re::Real, U_lid::Float64,
        output_dir::String, case_label::String,
        n_dense::Int = 401,
    )
    ref = ghia_reference_data(Re)
    if ref === nothing
        @printf("  [Ghia] no benchmark tabulated for Re=%.0f; skipping centerline validation.\n", Re)
        return nothing
    end
    ys_g, us_g, xs_g, vs_g = ref
    Lx, Ly = domain.box_size

    u_form      = Forms.build_form_field(domain.R1, u_coeffs)
    u_phys_form = Assemblers.solve_L2_projection(domain.R1, ★(u_form), domain.dΩ)

    # FEEC artefact: the dy-form derivative space `D_y` has reduced boundary
    # multiplicity, so physical `u` (which lives in `comp2 = P_x ⊗ D_y`)
    # vanishes structurally at `y = 0` and `y = L_y`. Same for physical `v`
    # at `x = 0` and `x = L_x`. Sampling exactly on the wall therefore
    # returns 0 by construction, which has nothing to do with the lid
    # enforcement. We probe one cell inward so the comparison reflects the
    # solver's behaviour rather than the discretisation's boundary trace.
    nx, ny = domain.nel
    dx_dom = Lx / nx;  dy_dom = Ly / ny
    nudge_into_domain_y(y) = clamp(y, 0.5 * dy_dom, Ly - 0.5 * dy_dom)
    nudge_into_domain_x(x) = clamp(x, 0.5 * dx_dom, Lx - 0.5 * dx_dom)

    us_num = Vector{Float64}(undef, length(ys_g))
    @inbounds for k in eachindex(ys_g)
        y_sample = nudge_into_domain_y(ys_g[k] * Ly)
        u, _ = sample_phys_velocity_at_point(u_phys_form, domain, 0.5*Lx, y_sample)
        us_num[k] = u / U_lid
    end
    vs_num = Vector{Float64}(undef, length(xs_g))
    @inbounds for k in eachindex(xs_g)
        x_sample = nudge_into_domain_x(xs_g[k] * Lx)
        _, v = sample_phys_velocity_at_point(u_phys_form, domain, x_sample, 0.5*Ly)
        vs_num[k] = v / U_lid
    end

    # Build masks that exclude the wall endpoints (y/L=0 and y/L=1 etc.)
    # when computing L2 / max errors — those samples report a structural
    # zero from the FEEC space, not a real solver disagreement.
    interior_y = (ys_g .> 1e-6) .& (ys_g .< 1.0 - 1e-6)
    interior_x = (xs_g .> 1e-6) .& (xs_g .< 1.0 - 1e-6)

    # Trapezoidal L2 norm over the (irregular) reference grid, restricted
    # to the interior indices.
    function trapz_norm_interior(gs, vals, mask)
        s = 0.0
        idxs = findall(mask)
        @inbounds for k in 1:length(idxs)-1
            i, j = idxs[k], idxs[k+1]
            s += 0.5 * (gs[j] - gs[i]) * (vals[i]^2 + vals[j]^2)
        end
        sqrt(s)
    end

    eu       = us_num .- us_g
    ev       = vs_num .- vs_g
    l2_u     = trapz_norm_interior(ys_g, eu, interior_y)
    l2_v     = trapz_norm_interior(xs_g, ev, interior_x)
    den_u    = trapz_norm_interior(ys_g, us_g, interior_y)
    den_v    = trapz_norm_interior(xs_g, vs_g, interior_x)
    rel_l2_u = l2_u / max(den_u, eps())
    rel_l2_v = l2_v / max(den_v, eps())
    max_u    = maximum(abs.(eu[interior_y]))
    max_v    = maximum(abs.(ev[interior_x]))

    ghia_csv_u = joinpath(output_dir, case_label * "_ghia_u_centerline.csv")
    open(ghia_csv_u, "w") do io
        println(io, "y_over_L,u_ghia,u_num,abs_err,rel_err,is_wall_endpoint")
        for k in eachindex(ys_g)
            rel = abs(us_num[k] - us_g[k]) / max(abs(us_g[k]), 1e-12)
            is_wall = !interior_y[k]
            @printf(io, "%.6f,%.6f,%.6f,%.6e,%.6e,%d\n",
                    ys_g[k], us_g[k], us_num[k],
                    abs(us_num[k] - us_g[k]), rel, is_wall ? 1 : 0)
        end
    end
    ghia_csv_v = joinpath(output_dir, case_label * "_ghia_v_centerline.csv")
    open(ghia_csv_v, "w") do io
        println(io, "x_over_L,v_ghia,v_num,abs_err,rel_err,is_wall_endpoint")
        for k in eachindex(xs_g)
            rel = abs(vs_num[k] - vs_g[k]) / max(abs(vs_g[k]), 1e-12)
            is_wall = !interior_x[k]
            @printf(io, "%.6f,%.6f,%.6f,%.6e,%.6e,%d\n",
                    xs_g[k], vs_g[k], vs_num[k],
                    abs(vs_num[k] - vs_g[k]), rel, is_wall ? 1 : 0)
        end
    end

    ys_d, us_d, xs_d, vs_d = extract_centerline_profiles(u_coeffs, domain; n_samples=n_dense)
    dense_u = joinpath(output_dir, case_label * "_centerline_u_dense.csv")
    open(dense_u, "w") do io
        println(io, "y,u_phys,u_over_U_lid")
        for k in eachindex(ys_d)
            @printf(io, "%.10e,%.10e,%.10e\n", ys_d[k], us_d[k], us_d[k] / U_lid)
        end
    end
    dense_v = joinpath(output_dir, case_label * "_centerline_v_dense.csv")
    open(dense_v, "w") do io
        println(io, "x,v_phys,v_over_U_lid")
        for k in eachindex(xs_d)
            @printf(io, "%.10e,%.10e,%.10e\n", xs_d[k], vs_d[k], vs_d[k] / U_lid)
        end
    end

    summary_path = joinpath(output_dir, case_label * "_ghia_summary.txt")
    open(summary_path, "w") do io
        println(io, @sprintf("Ghia (1982) benchmark validation for cavity Re=%.0f", Re))
        println(io, "=" ^ 60)
        println(io, "Errors are reported over INTERIOR Ghia points only.")
        println(io, "The wall endpoints (y/L=0, y/L=1, x/L=0, x/L=1) are excluded")
        println(io, "because the FEEC dy-form derivative space `D_y` has reduced")
        println(io, "boundary multiplicity → physical u/v vanishes at y/x walls")
        println(io, "by construction, regardless of the lid Brinkman stamping.")
        println(io, "Endpoints are still listed in the per-point CSV with the")
        println(io, "`is_wall_endpoint=1` flag for completeness.")
        println(io, "")
        println(io, "u along vertical centerline (x = L/2):")
        println(io, @sprintf("  L2 error (relative) : %.4e", rel_l2_u))
        println(io, @sprintf("  L2 error (absolute) : %.4e", l2_u))
        println(io, @sprintf("  max pointwise error : %.4e", max_u))
        println(io, "")
        println(io, "v along horizontal centerline (y = L/2):")
        println(io, @sprintf("  L2 error (relative) : %.4e", rel_l2_v))
        println(io, @sprintf("  L2 error (absolute) : %.4e", l2_v))
        println(io, @sprintf("  max pointwise error : %.4e", max_v))
    end

    println("")
    @printf("  [Ghia Re=%.0f]  u centerline: rel_L2=%.3e  max=%.3e\n", Re, rel_l2_u, max_u)
    @printf("  [Ghia Re=%.0f]  v centerline: rel_L2=%.3e  max=%.3e\n", Re, rel_l2_v, max_v)
    println("  Saved: $ghia_csv_u")
    println("  Saved: $ghia_csv_v")
    println("  Saved: $dense_u")
    println("  Saved: $dense_v")
    println("  Saved: $summary_path")

    return (rel_l2_u=rel_l2_u, rel_l2_v=rel_l2_v, max_u=max_u, max_v=max_v,
            ys_g=ys_g, us_g=us_g, us_num=us_num,
            xs_g=xs_g, vs_g=vs_g, vs_num=vs_num,
            ghia_csv_u=ghia_csv_u, ghia_csv_v=ghia_csv_v,
            dense_u=dense_u, dense_v=dense_v, summary_path=summary_path)
end

function test_lid_driven_cavity(;
        output_dir::String       = joinpath(get(ENV, "OUTPUT_DIR", pwd()), "lid_cavity"),
        nel::NTuple{2,Int}                 = (96, 96),
        box_size::NTuple{2,Float64}        = (1.0, 1.0),
        U_lid::Float64           = 1.0,
        viscosity::Float64       = 0.01,
        T_final::Float64         = 10.0,
        lid_band_cells::Float64  = 2.0,
        wall_band_cells::Float64 = 1.0,
        particles_per_cell::Int  = 10,
        save_vtk_every::Int      = 1,
        target_cfl::Float64      = 0.5,
        ss_record_every::Int     = 5,
        validate_ghia::Bool      = true,
        # Cavity-specific physics overrides (see notes below for the why):
        enable_energy_correction::Bool = false,
        ftle_threshold::Float64        = Inf,
        max_longterm_delta_t::Float64  = 1.0e9,
        pic_blend_alpha::Float64       = 0.05,
        clear_memo_every::Int          = 5,
        case_label::String       = "lid_re100",
    )
    mkpath(output_dir)
    Re = U_lid * box_size[1] / viscosity
    println("\n========== LID DRIVEN CAVITY ==========")
    println(@sprintf("  nel=%s  L=%.3f  U_lid=%.3f  ν=%.4f  Re=%.1f  T=%.1f",
                     nel, box_size[1], U_lid, viscosity, Re, T_final))
    println(@sprintf("  Brinkman bands:  lid=%.2f cells  walls=%.2f cells",
                     lid_band_cells, wall_band_cells))
    println(@sprintf("  energy_correction=%s  ftle_threshold=%s  pic_blend=%.3f",
                     enable_energy_correction, ftle_threshold, pic_blend_alpha))

    cfg = SimulationConfig(;
        nel                 = nel,
        box_size            = box_size,
        boundary_condition  = (:wall, :wall, :wall, :wall),
        obstacle            = nothing,
        flow_type           = :zero,
        viscosity           = viscosity,
        T_final             = T_final,
        target_cfl          = target_cfl,
        cfl_adaptive        = true,
        particles_per_cell  = particles_per_cell,
        min_particles_per_element = max(8, particles_per_cell ÷ 2),
        max_particles_per_element = max(32, particles_per_cell * 2),
        min_particles_per_quarter = 1,
        output_every        = save_vtk_every,
        enable_energy_correction = enable_energy_correction,
        ftle_threshold      = ftle_threshold,
        max_longterm_delta_t = max_longterm_delta_t,
        pic_blend_alpha     = pic_blend_alpha,
        clear_memo_every    = clear_memo_every,
    )

    step_hook = function (particles, domain, _cfg, step, t)
        apply_lid_cavity_brinkman!(particles, domain, U_lid;
                                   lid_band_cells=lid_band_cells,
                                   wall_band_cells=wall_band_cells)
    end

    # Steady-state convergence tracking. probe_callback receives u_coeffs every
    # step; we sample every `ss_record_every` steps and record the
    # mass-weighted L2 norm of (u_h - u_h_prev). When this plateaus near
    # round-off the cavity has reached steady state.
    ss_records  = NamedTuple[]
    u_prev_ref  = Ref{Vector{Float64}}(Float64[])
    ss_probe_cb = function (t, u_coeffs, domain, step)
        step == 0 && return  # first call is the initial state — no diff yet
        step % max(1, ss_record_every) == 0 || return
        u_prev = u_prev_ref[]
        if length(u_prev) == length(u_coeffs)
            diff       = u_coeffs .- u_prev
            Mdiff      = domain.Mass_matrix * diff
            Mu         = domain.Mass_matrix * u_coeffs
            norm_diff  = sqrt(max(dot(diff,    Mdiff), 0.0))
            norm_u     = sqrt(max(dot(u_coeffs, Mu),    0.0))
            rel_diff   = norm_diff / max(norm_u, eps())
            push!(ss_records, (t=t, step=step, abs_diff=norm_diff,
                               norm_u=norm_u, rel_diff=rel_diff))
            @printf("  [SS] t=%.3f  ||Δu||_M=%.3e  ||u||_M=%.3e  rel=%.3e\n",
                    t, norm_diff, norm_u, rel_diff)
        end
        u_prev_ref[] = copy(u_coeffs)
    end

    result = run_diagnostic_simulation(cfg;
        save_vtk        = true,
        save_vtk_every  = save_vtk_every,
        output_dir      = joinpath(output_dir, case_label * "_vtk"),
        record_every    = max(1, save_vtk_every),
        case_name       = case_label,
        step_hook       = step_hook,
        probe_callback  = ss_probe_cb,
        save_particles  = false,
    )

    history_csv = joinpath(output_dir, case_label * "_history.csv")
    open(history_csv, "w") do io
        println(io, "t,energy,enstrophy,circulation")
        for hi in result.history
            @printf(io, "%.10e,%.16e,%.16e,%.16e\n",
                    hi.t, hi.energy, hi.enstrophy, hi.circulation)
        end
    end
    println("Saved history CSV: $history_csv")

    ss_csv = joinpath(output_dir, case_label * "_steady_state.csv")
    open(ss_csv, "w") do io
        println(io, "t,step,abs_diff_L2,norm_u_L2,rel_diff")
        for r in ss_records
            @printf(io, "%.10e,%d,%.16e,%.16e,%.16e\n",
                    r.t, r.step, r.abs_diff, r.norm_u, r.rel_diff)
        end
    end
    println("Saved steady-state CSV: $ss_csv")

    ghia = nothing
    if validate_ghia
        ghia = validate_lid_cavity_against_ghia(
            result.u_final, result.domain;
            Re=Re, U_lid=U_lid,
            output_dir=output_dir, case_label=case_label,
        )
    end

    return (result=result, history_csv=history_csv, ss_csv=ss_csv,
            ss_records=ss_records, ghia=ghia)
end

"""
Dispatcher used by the bottom of this file when run as a script.

Reads `COFLIP_CASE` from the environment and runs the matching test
function. SLURM scripts set `COFLIP_CASE` plus the case-specific extra
vars listed below. Returns `nothing`.

Supported cases:
  - `tg_convergence`   — uses `COFLIP_SWEEP_MODE` ∈ {`space`, `time`}
                         and `COFLIP_SWEEP_IDX` (1-based).
  - `von_karman`       — Strouhal probe run, no extra vars (override
                         defaults by editing the function call below).
  - `lid_cavity`       — lid-driven cavity, no extra vars.
  - `tg_validation`    — single-point viscous-decay validation (the
                         legacy `test_decaying_taylor_green`).
  - `main`             — the full-blown `main()` from this file.
  - `restart`          — `restart_main` from a saved particle .vtu;
                         requires `COFLIP_RESTART_VTU`,
                         `COFLIP_RESTART_STEP`, `COFLIP_ADDITIONAL_STEPS`,
                         `COFLIP_DT`.

If `COFLIP_CASE` is unset or empty, the dispatcher does nothing (so
`include(...)` from external scripts is safe). Use that for
interactive sessions; call a specific test function manually.
"""
function coflip_dispatch_from_env()
    case = get(ENV, "COFLIP_CASE", "")
    if case == ""
        println("(CO-FLIP_periodic.jl loaded; set ENV[\"COFLIP_CASE\"] to run a case.)")
        return nothing
    end

    if case == "tg_convergence"
        mode_str = get(ENV, "COFLIP_SWEEP_MODE", "space")
        idx_str  = get(ENV, "COFLIP_SWEEP_IDX",  "1")
        mode = Symbol(mode_str)
        idx  = parse(Int, idx_str)
        test_decaying_tg_convergence(; mode=mode, sweep_idx=idx)
    elseif case == "von_karman"
        test_von_karman_strouhal()
    elseif case == "lid_cavity"
        test_lid_driven_cavity()
    elseif case == "tg_validation"
        test_decaying_taylor_green()
    elseif case == "main"
        main()
    elseif case == "restart"
        vtu  = get(ENV, "COFLIP_RESTART_VTU", "")
        step = parse(Int, get(ENV, "COFLIP_RESTART_STEP",  "0"))
        more = parse(Int, get(ENV, "COFLIP_ADDITIONAL_STEPS", "1000"))
        dt   = parse(Float64, get(ENV, "COFLIP_DT", "0.05"))
        vtu == "" && error("COFLIP_CASE=restart requires COFLIP_RESTART_VTU env var")
        restart_main(SimulationConfig(), vtu;
                     restart_step=step, additional_steps=more, dt=dt)
    else
        error("Unknown COFLIP_CASE=$case. " *
              "Expected one of: tg_convergence, von_karman, lid_cavity, " *
              "tg_validation, main, restart")
    end
    return nothing
end

"""Run complete CO-FLIP simulation."""
function main(cfg::SimulationConfig=SimulationConfig())
    println("Initializing Domain and Particles...")
    LinearAlgebra.BLAS.set_num_threads(1)

    num_particles = cfg.nel[1] * cfg.nel[2] * cfg.particles_per_cell

    domain    = GenerateDomain(cfg.nel, cfg.p, cfg.k;
                               box_size=cfg.box_size,
                               starting_point=cfg.starting_point,
                               boundary_condition=cfg.boundary_condition,
                               obstacle=cfg.obstacle)
    particles = generate_particles(num_particles, domain, cfg.flow_type;
                                   stratified_seeding=cfg.stratified_seeding,
                                   rng_seed=cfg.rng_seed,
                                   volume_convention=cfg.volume_convention,
                                   boundary_condition=cfg.boundary_condition,
                                   U_inf=cfg.inlet_U_inf)
    # Clean up any particles seeded inside the obstacle: push them to the
    # surface with zero velocity before the first sort/P2G.
    apply_obstacle_brinkman!(particles, domain)
    particle_sorter!(particles, domain)

    run_startup_smoke_tests(domain)

    dt_cfl = compute_cfl_dt_from_particles(particles, domain, cfg.target_cfl)
    if !isfinite(dt_cfl)
        n_steps = 1;  dt = cfg.T_final
    else
        n_steps = max(1, ceil(Int, cfg.T_final / dt_cfl))
        dt      = cfg.T_final / n_steps
    end

    println("Starting Time Integration (T_final=$(cfg.T_final), dt=$dt, steps=$n_steps)...")

    output_dir          = "coflip_output";  mkpath(output_dir)
    particle_output_dir = "coflip_output";  mkpath(particle_output_dir)

    ndofs        = FunctionSpaces.get_num_basis(domain.R1.fem_space)
    num_elements = prod(domain.nel)
    sim_buf      = SimulationBuffers(num_particles, ndofs, num_elements, domain)

    println("Saving true t=0 snapshot...")
    u_coeffs = coadjoint_step!(particles, domain, sim_buf;
        lsqr_atol=cfg.lsqr_atol, lsqr_btol=cfg.lsqr_btol, lsqr_maxiter=cfg.lsqr_maxiter,
        lsqr_error_on_nonconvergence=cfg.lsqr_error_on_nonconvergence,
        projection_mean_subtract=cfg.projection_mean_subtract)

    u_form      = Forms.build_form_field(domain.R1, u_coeffs; label="u_h")
    u_phys_expr = ★(u_form)
    u_phys_form = Assemblers.solve_L2_projection(domain.R1, u_phys_expr, domain.dΩ)

    particle_sorter!(particles, domain)
    nx, ny = domain.nel
    Lx, Ly = domain.box_size
    dx = Lx / nx;  dy = Ly / ny

    @inbounds for pid in eachindex(particles.x)
        eid = particles.elem_ids[pid]
        ej  = ((eid - 1) ÷ nx) + 1
        ei  = eid - (ej - 1) * nx
        set_g2p_velocity(
            particles, pid,
            particles.x[pid], particles.y[pid],
            eid, (ei - 1) * dx, (ej - 1) * dy, dx, dy,
            u_phys_form, Lx, Ly,
        )
    end
    apply_obstacle_brinkman!(particles, domain)

    d_u_phys = Forms.d(u_phys_form)
    ω_h      = ★(d_u_phys)
    Plot.export_form_fields_to_vtk((u_form,), "u_h_0000"; output_directory_tree=["coflip_output"])
    Plot.export_form_fields_to_vtk((ω_h,),    "w_h_0000"; output_directory_tree=["coflip_output"])
    export_particles_to_vtk(particles, joinpath(particle_output_dir, "particles_0000"))
    println("  Saved step 0 particles to VTK")

    warmup_evaluation_memo!(domain)

    step = 1
    while step <= n_steps
        println("Step $step / $n_steps (t = $(round(step * dt, digits=3)))...")

        if mod(step, 5) == 0
            physical_spatial_sort!(particles, domain)  # Every 5 steps
        end

        u_coeffs = step_co_flip!(particles, domain, dt, u_coeffs, sim_buf, cfg)

        u_form      = Forms.build_form_field(domain.R1, u_coeffs; label="u_h")
        u_phys_expr = ★(u_form)
        u_phys_form = Assemblers.solve_L2_projection(domain.R1, u_phys_expr, domain.dΩ)

        if cfg.delayed_reinit_frequency > 0 &&
           step % cfg.delayed_reinit_frequency == 0
            new_count = global_reseed_from_grid!(particles, domain, cfg, u_phys_form;
                                                 rng_seed=cfg.rng_seed)
            println("  Delayed re-seed at step $step: $(new_count) particles re-generated from grid.")
        end

        reset_count = apply_ftle_reset!(particles, domain, u_phys_form, cfg)
        reset_count > 0 && println("  FTLE reset: $reset_count particles reseeded.")

        dt_check = recheck_cfl_dt(u_coeffs, domain, cfg.target_cfl, dt)
        new_dt   = enforce_cfl_recheck!(dt, dt_check, cfg)
        if new_dt != dt
            remaining_steps = max(1, ceil(Int, (cfg.T_final - step * dt) / new_dt))
            n_steps = step + remaining_steps  # Adjust total steps to keep T_final fixed
            println("  Adaptive CFL: dt=$(new_dt) (was $(dt)); remaining_steps≈$(remaining_steps)")
            dt = new_dt
        end

        if cfg.output_every > 0 && step % cfg.output_every == 0
            d_u_phys = Forms.d(u_phys_form)
            ω_h      = ★(d_u_phys)
            Plot.export_form_fields_to_vtk((u_form,), @sprintf("u_h_%04d", step); output_directory_tree=["coflip_output"])
            Plot.export_form_fields_to_vtk((ω_h,),    @sprintf("w_h_%04d", step); output_directory_tree=["coflip_output"])

            diagnostics = compute_conservation_diagnostics(u_coeffs, domain)
            print_conservation_diagnostics(step, step * dt, diagnostics)

            export_particles_to_vtk(
                particles,
                joinpath(particle_output_dir, @sprintf("particles_%04d", step)),
            )
            println("  Saved step $step visualization and particle data.")
        end

        maybe_clear_memo_tables!(step, cfg.clear_memo_every, domain)
        step += 1
    end

    println("\nSimulation complete! Step files written to '$(output_dir)'.")
end


#main()

#run_test_suite()

#result = test_decaying_taylor_green()

if abspath(PROGRAM_FILE) == @__FILE__
    coflip_dispatch_from_env()
    #test_lid_driven_cavity()
    #test_p2g_taylor_green_convergence()
end

# Legacy restart entry point (kept for reference; opt in via COFLIP_CASE=restart):
# cfg = SimulationConfig()
# restart_main(cfg, "coflip_output/particles_0627.vtu";
#              restart_step = 0627, additional_steps = 1000, dt = 0.048)