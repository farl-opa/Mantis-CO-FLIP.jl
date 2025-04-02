# # Hodge Laplacian

# ## Introduction

# In this example, ww solve the Hodge-Laplacian for $0$-forms with Dirichlet boundary
# conditions. The
# ``0``-form Hodge-Laplacian on a domain ``\Omega`` with boundary ``\partial \Omega`` is
# ```math
# \begin{alignat*}{2}
#     &\mathrm{d}^* \mathrm{d} \phi^0 = - f^0  \quad &&\text{on}\ \Omega \;, \\
#     &tr(\phi^0) = 0  \quad &&\text{on}\ \partial\Omega \;.
# \end{alignat*}
# ```
# The weak formulation is then as follows.
# ```math
# \begin{gather*}
# \text{Given}\ f^0 \in L^2 \Lambda^0 (\Omega),\ \text{find}\ \phi^0 \in \Lambda^0_h (\Omega)
# \ \text{such that} \\
# \int_{\Omega} \mathrm{d} \phi^0 \wedge \star \mathrm{d} \varphi^0 = \int_{\Omega} f^0
# \wedge \star \varphi^0 \quad  \forall \ \varphi^0 \in \Lambda^0_{h,0} (\Omega) \;.
# \end{gather*}
# ```

# ## Implementation

# We start by defining the geometry and the function space. The geometry is a simple unit
# square with ``4 \times 4`` elements. The function space is a B-spline space with degree
# ``p = (3, 3)`` and knot vector ``k = (2, 2)``.

import Mantis

starting_point = (0.0, 0.0)
box_size = (1.0, 1.0)
num_elements = (4, 4)
geometry = Mantis.Geometry.create_cartesian_box(
    starting_point, box_size, num_elements
)

p = (3, 3)
k = (2, 2)
B = Mantis.FunctionSpaces.create_bspline_space(
    starting_point,
    box_size,
    num_elements,
    p,
    k,
)

# Mantis works with forms, so we need to define the form space. In this case, we are
# working with $0$-forms, so we define the form space as follows.

Λ⁰ = Mantis.Forms.FormSpace(0, geometry, Mantis.FunctionSpaces.DirectSumSpace((B,)), "label")


# We define the weak form inputs. The weak form inputs contain the trial and test spaces,
# the forcing function, and the quadrature rule. We define the forcing function as a
# function of the coordinates. In this case, we define the forcing function as
# ``f^0 = 8 \pi^2 \sin(2 \pi x) \sin(2 \pi y)``.

function forcing_function(x::Matrix{Float64})
    return [@. 8.0 * pi^2 * sin(2.0 * pi * x[:,1]) * sin(2.0 * pi * x[:,2])]
end
f⁰ = Mantis.Forms.AnalyticalFormField(0, forcing_function, geometry, "f⁰")

# The quadrature rule is defined as a tensor product rule of the degree of the B-spline
# space plus one. In this case, we define the quadrature rule as a Gauss-Legendre rule.
qrule = Mantis.Quadrature.tensor_product_rule(
    p .+ 1, Mantis.Quadrature.gauss_legendre
)

wfi = Mantis.Assemblers.WeakFormInputs(f⁰, Λ⁰, Λ⁰, qrule)


# We define the weak form for the Hodge-Laplacian. The weak form is defined as a function
# that takes the weak form inputs and the element id as arguments. The weak form inputs
# contain the trial and test spaces, the forcing function, and the quadrature rule.

function zero_form_hodge_laplacian(
    inputs::Mantis.Assemblers.WeakFormInputs,
    element_id,
)
    dtrial = Mantis.Forms.exterior_derivative(inputs.space_trial[1])
    dtest = Mantis.Forms.exterior_derivative(inputs.space_test[1])

    Ar, Ac, Av = Mantis.Forms.evaluate_inner_product(
        dtest, dtrial, element_id, inputs.quad_rule
    )

    br, bc, bv = Mantis.Forms.evaluate_inner_product(
        inputs.space_test[1], inputs.forcing[1], element_id, inputs.quad_rule
    )

    return (Ar, Ac, Av), (br, bv)
end

# We can now assemble the linear system and solve it to obtain the solution. We define the
# Dirichlet boundary conditions as a dictionary with the boundary condition index as the
# key and the boundary condition value as the value.

bc_dirichlet = Dict(Mantis.Assemblers.get_trace_dofs(B) .=> 0.0)

# We assemble the linear system and solve it to obtain the solution.

lhs, rhs = Mantis.Assemblers.assemble(zero_form_hodge_laplacian, wfi, bc_dirichlet)
ϕ⁰ = Mantis.Forms.FormField(Λ⁰, "ϕ⁰")
ϕ⁰.coefficients .= lhs \ rhs


# We can now plot the solution using the `plot` function. This will write the output to a
# VTK file that can be visualized using a VTK viewer, such as Paraview.

data_folder = joinpath(dirname(dirname(pathof(Mantis))), "examples", "data")
output_data_folder = joinpath(data_folder, "output", "HodgeLaplacian")
output_filename = "HodgeLaplacian-0form-Dirichlet-$(length(p))D.vtu"
output_file = joinpath(output_data_folder, output_filename)
Mantis.Plot.plot(
    ϕ⁰;
    vtk_filename = output_file,
    n_subcells = 1,
    degree = maximum(p),
    ascii = false,
    compress = false
)


# ## Extensions

# ### 3D
# We can extend this example to 3D by defining the geometry as a unit cube with ``4 \times 4
# \times 4`` elements. We define the function space as a B-spline space with degree ``p = (3
# , 3, 3)`` and knot vector ``k = (2, 2, 2)``. We define the forcing function as ``f^0 = 12
# \pi^2 \sin(2 \pi x) \sin(2 \pi y) \sin(2 \pi z)``.
# Note that the new variables are renamed to avoid overwriting the previous variables and to
# avoid type instabilities.

starting_point_3D = (0.0, 0.0, 0.0)
box_size_3D = (1.0, 1.0, 1.0)
num_elements_3D = (4, 4, 4)
geometry_3D = Mantis.Geometry.create_cartesian_box(
    starting_point_3D, box_size_3D, num_elements_3D
)

p_3D = (3, 3, 3)
k_3D = (2, 2, 2)
B_3D = Mantis.FunctionSpaces.create_bspline_space(
    starting_point_3D,
    box_size_3D,
    num_elements_3D,
    p_3D,
    k_3D,
)

function forcing_function_3D(x::Matrix{Float64})
    return [
        @. 12.0 * pi^2 * sin(2.0 * pi * x[:,1]) * sin(2.0 * pi * x[:,2]) *
        sin(2.0 * pi * x[:,3])
    ]
end


# We don't need to change any of the previous code, so we can reuse it directly:

#
Λ⁰_3D = Mantis.Forms.FormSpace(0, geometry_3D, Mantis.FunctionSpaces.DirectSumSpace((B_3D,)), "label")

f⁰_3D = Mantis.Forms.AnalyticalFormField(0, forcing_function_3D, geometry_3D, "f⁰")

qrule_3D = Mantis.Quadrature.tensor_product_rule(
    p_3D .+ 1, Mantis.Quadrature.gauss_legendre
)

wfi_3D = Mantis.Assemblers.WeakFormInputs(f⁰_3D, Λ⁰_3D, Λ⁰_3D, qrule_3D)

bc_dirichlet_3D = Dict(Mantis.Assemblers.get_trace_dofs(B_3D) .=> 0.0)

lhs_3D, rhs_3D = Mantis.Assemblers.assemble(zero_form_hodge_laplacian, wfi_3D, bc_dirichlet_3D)
ϕ⁰_3D = Mantis.Forms.FormField(Λ⁰_3D, "ϕ⁰")
ϕ⁰_3D.coefficients .= lhs_3D \ rhs_3D

output_filename_3D = "HodgeLaplacian-0form-Dirichlet-$(length(p_3D))D.vtu"
output_file_3D = joinpath(output_data_folder, output_filename_3D)
Mantis.Plot.plot(
    ϕ⁰_3D;
    vtk_filename = output_file_3D,
    n_subcells = 1,
    degree = maximum(p_3D),
    ascii = false,
    compress = false
)
