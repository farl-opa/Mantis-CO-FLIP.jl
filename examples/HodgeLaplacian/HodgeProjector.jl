module HodgeProjectorModule

using Mantis

"""
    HodgeProjector(nel, p, k)

Create a projector that removes the non–divergence-free part
of a discrete 1-form field using the Hodge–Laplace problem.
"""
struct HodgeProjector
    R0::Any
    R1::Any
    R2::Any
    geo::Any
    dΩ::Any

    nel::NTuple{2,Int}
    p::NTuple{2,Int}
    k::NTuple{2,Int}
end

# -------------------------------------------------------------
#                      Constructor
# -------------------------------------------------------------
function HodgeProjector(nel::NTuple{2,Int},
                        p::NTuple{2,Int},
                        k::NTuple{2,Int})

    # mesh description
    starting_point = (0.0, 0.0)
    box_size       = (1.0, 1.0)

    # quadrature rules
    nq_assembly = p .+ 1
    nq_error    = nq_assembly .* 2

    ∫ₐ, ∫ₑ = Quadrature.get_canonical_quadrature_rules(
        Quadrature.gauss_legendre, nq_assembly, nq_error
    )

    dΩ = Quadrature.StandardQuadrature(∫ₐ, prod(nel))

    # de Rham complex of B-splines
    R = Forms.create_tensor_product_bspline_de_rham_complex(
        starting_point, box_size, nel, p, k
    )
    R0, R1, R2 = R[1], R[2], R[3]

    # geometry of 1-forms (needed for evaluation)
    geo = Forms.get_geometry(R1)

    return HodgeProjector(R0, R1, R2, geo, dΩ, nel, p, k)
end

# -------------------------------------------------------------
#                      Projection method
# -------------------------------------------------------------
"""
    project(hp, v_h)

Given a discrete 1-form field `v_h`, return its divergence-free
component by removing the exact part δϕₕ obtained from the
Hodge–Laplace problem.
"""
function project(hp::HodgeProjector, v_h)

    # 1. Compute discrete divergence
    f = d(v_h)  

    # 3. Solve Hodge-Laplacian for δϕ_h
    u_corr, ϕ_corr = Assemblers.solve_volume_form_hodge_laplacian(hp.R1, hp.R2, f, hp.dΩ)

    # 4. Subtract non-div-free part
    return v_h - u_corr
end



# Initialize the projector
hp = HodgeProjector((8,8), (3,3), (2,2))

manifold_dim = 2

# -------------------------
# Example discrete 1-form
# -------------------------

function sinusoidal_solution(
    form_rank::Int, geo::Geometry.AbstractGeometry{manifold_dim}
) where {manifold_dim}
    n_form_components = binomial(manifold_dim, form_rank)
    ω = 2.0 * pi
    function my_sol(x::Matrix{Float64})
        # ∀i ∈ {1, 2, ..., n}: uᵢ = sin(ωx¹)sin(ωx²)...sin(ωxⁿ)
        y = @. sin(ω * x)
        return repeat([vec(prod(y; dims=2))], n_form_components)
    end
    return Mantis.Forms.AnalyticalFormField(form_rank, my_sol, geo, "f")
end

function moderately_divergent_field(
    form_rank::Int,
    geo::Geometry.AbstractGeometry{manifold_dim};
) where {manifold_dim}

    n_form_components = binomial(manifold_dim, form_rank)
    π = Base.MathConstants.pi

    function my_sol(x::Matrix{Float64})
        npts, ndim = size(x)
        u = zeros(npts, ndim)

        for i in 1:ndim
            # u_i = sin(π * x_i) * product of (1 + x_j^2) for j ≠ i
            prod_others = ones(npts)
            for j in 1:ndim
                if j != i
                    prod_others .= prod_others .* (1 .+ x[:, j].^2)
                end
            end
            u[:, i] .= sin.(π .* x[:, i]) .* prod_others
        end

        # Flatten to n_form_components (scalar pattern repeated)
        u_vec = vec(sum(u; dims=2))
        return repeat([u_vec], n_form_components)
    end

    return Mantis.Forms.AnalyticalFormField(form_rank, my_sol, geo, "f")
end


# Construct v_h
v = moderately_divergent_field(1, hp.geo)

v_h = Mantis.Assemblers.solve_L2_projection(hp.R1, v, hp.dΩ)

div_vh = d(v_h)

# -------------------------
# Project to divergence-free
# -------------------------
u_divfree = project(hp, v_h)

# compute discrete divergence and print its coefficient vector
div_f = d(u_divfree)

# Export to VTK
Plot.export_form_fields_to_vtk((v_h,), "Original Velocity")
Plot.export_form_fields_to_vtk((div_vh,), "Divergence of Original Velocity")
Plot.export_form_fields_to_vtk((u_divfree,), "Divergence Free Velocity")
Plot.export_form_fields_to_vtk((div_f,), "Divergence Field")


end 
