import Mantis

using Test

# Refer to the following file for method and variable definitions
include("../../examples/HelperFunctions.jl")

############################################################################################
#                                      Problem setup                                       #
############################################################################################
# Mesh
const starting_point = (0.0, 0.0)
const box_size = (fpi, 1.0) # This size is so that the eigenvalues are unique.
const num_runs = 5 # Number of problem runs. Each solve 2≤i≤num_runs will have 2^i elements.
num_elements = (2, 2) .^ 3 # Initial mesh size.

# B-spline parameters
p = (2, 2) # Polynomial degrees.
k = p .- 1 # Regularities. (Maximally smooth B-splines.)

# Hierarchical parameters.
truncate = false # true = THB, false = HB
simplified = true
num_steps = 3 # Number of refinement steps.
num_sub = (2, 2) # Number of subdivisions per dimension per step.
dorfler_parameter = 0.2
Lchains = true # Decide if Lchains are added to fix inexact refinements.
eigenfunc = 1 # Eigenfunction to use for adaptive refinement.

# Quadrature rules
nq_assembly = p .+ 1
nq_error = nq_assembly .* 2
qrule_assembly, qrule_error = Mantis.Quadrature.get_canonical_quadrature_rules(
    Mantis.Quadrature.gauss_legendre, nq_assembly, nq_error
)

# Number of eigenvalues to compute
const num_eig = 20
# Scaling form maxwell eigenfunctions.
scale_factors = ntuple(2) do k
    return pi /(box_size[k] - starting_point[k])
end

############################################################################################
#                                       Run problem                                        #
############################################################################################
function run_problems(
    complex::C,
    num_steps::Int,
    dorfler_parameter::Float64,
    Lchains::Bool,
    q_rule_assembly::Mantis.Quadrature.AbstractGlobalQuadratureRule{manifold_dim},
    q_rule_error::Mantis.Quadrature.AbstractGlobalQuadratureRule{manifold_dim},
    eigenfunction::Int,
    num_eig::Int,
    scale_factors::NTuple{manifold_dim, Float64};
    verbose::Bool=false,
) where {manifold_dim, num_forms, C <: NTuple{num_forms, Mantis.Forms.AbstractFormSpace}}

    eigval_errors = [Vector{Float64}(undef, num_steps+1) for _ in 1:num_eig]
    eigfunc_errors = [Vector{Float64}(undef, num_steps+1) for _ in 1:num_eig]

    # Exact solution on initial step
    exact_eigvals, exact_eigfuncs = Mantis.Assemblers.get_analytical_maxwell_eig(
        num_eig, Mantis.Forms.get_geometry(complex[1]), scale_factors
    )

    # Solve problem on initial step
    compt_eigvals, compt_eigfuncs = Mantis.Assemblers.solve_maxwell_eig(
        complex[1], complex[2], q_rule_assembly, num_eig
    )

    for eigid in 1:num_eig
        eigval_errors[eigid][1] = compt_eigvals[eigid] - exact_eigvals[eigid]
        eigfunc_errors[eigid][1] = Mantis.Analysis.compute_error_total(
            compt_eigfuncs[eigid], exact_eigfuncs[eigid], qrule_error, "L2"
        )
    end

    # Calculate element-wise error
    err_per_element = Mantis.Analysis.compute_error_per_element(
        compt_eigfuncs[eigenfunction], exact_eigfuncs[eigenfunction], q_rule_error
    )

    for step in 1:num_steps
        zero_form_space = Mantis.FunctionSpaces.get_component_spaces(complex[1].fem_space)[1]

        L = Mantis.FunctionSpaces.get_num_levels(zero_form_space)

        new_operator, new_space = Mantis.FunctionSpaces.build_two_scale_operator(
            Mantis.FunctionSpaces.get_space(zero_form_space, L),
            Mantis.FunctionSpaces.get_num_subdivisions(zero_form_space),
        )

        dorfler_marking = Mantis.FunctionSpaces.get_dorfler_marking(
            err_per_element, dorfler_parameter
        )

        # Get domains to be refined in current step
        marked_elements_per_level = Mantis.FunctionSpaces.get_padding_per_level(
            zero_form_space, dorfler_marking
        )

        # Add Lchains if needed
        if Lchains
            Mantis.FunctionSpaces.add_Lchains_supports!(
                marked_elements_per_level, zero_form_space, new_operator
            )
        end
        # Get children of marked elements
        refinement_domains = Mantis.FunctionSpaces.get_refinement_domains(
            zero_form_space, marked_elements_per_level, new_operator
        )

        # Update the hierarchical complex based on the refinement domains and the 0-form space
        complex = Mantis.Forms.update_hierarchical_de_rham_complex(
            complex, refinement_domains, new_operator, new_space
        )

        # Update exact solution
        exact_eigvals, exact_eigfuncs = Mantis.Assemblers.get_analytical_maxwell_eig(
            num_eig, Mantis.Forms.get_geometry(complex[1]), scale_factors
        )

        # Solve problem on current step
        compt_eigvals, compt_eigfuncs = Mantis.Assemblers.solve_maxwell_eig(
            complex[1], complex[2], q_rule_assembly, num_eig
        )

        err_per_element = Mantis.Analysis._compute_square_error_per_element(
            compt_eigfuncs[eigenfunction], exact_eigfuncs[eigenfunction], q_rule_error
        )

        for eigid in 1:num_eig
            eigval_errors[eigid][step + 1] = compt_eigvals[eigid] - exact_eigvals[eigid]
            eigfunc_errors[eigid][step + 1] = Mantis.Analysis.compute_error_total(
                compt_eigfuncs[eigid], exact_eigfuncs[eigid], qrule_error, "L2"
            )
        end

    end

    return eigval_errors, eigfunc_errors
end

function runt_tests(eigval_errors, num_steps, num_eig)
    for eigid in 1:num_eig
        @test isapprox(abs(eigval_errors[eigid][(num_steps + 1)]), 0.0; atol=2e-2)
    end

    return nothing
end


R_complex = Mantis.Forms.create_hierarchical_de_rham_complex(
    starting_point, box_size, num_elements, p, k, num_sub, truncate, simplified
)
eigval_errors, eigfunc_errors = run_problems(
    R_complex,
    num_steps,
    dorfler_parameter,
    Lchains,
    qrule_assembly,
    qrule_error,
    eigenfunc,
    num_eig,
    scale_factors,
)
runt_tests(eigval_errors, num_steps, num_eig)
