# # Heat Equation

# ## Background knowledge

# ### A. The Heat Equation in 1D

# #### 1. The differential equation

# The heat equation in one dimension is given by:
# ```math
# \frac{\partial u}{\partial t} - \frac{\partial}{\partial x} \left( \alpha
# \frac{\partial u}{\partial x} \right) = f \quad \text{for} \quad x \in (0, L)
# \quad \text{and} \quad t > 0
# ```
# where:
# - ``x`` is the spatial coordinate,
# - ``t`` is the time,
# - ``\alpha(x)`` is the spatially-varying thermal diffusivity of the material,
# - ``u(x, t)`` is the temperature distribution function,
# - ``f(x, t)`` is a time dependent and spatially varying heat source.

# #### 2. Initial and boundary conditions

# To solve the heat equation, we must specify initial and boundary conditions:
# 1. **Initial Condition**: This refers to the temperature distribution at the initial time
#    ``t = 0``. We need to know this temperature distribution in our entire domain
#    ``(0, L)``.
#    ```math
#    u(x, 0) = u_0(x)
#    ```
# 2. **Boundary Conditions**: We can specify different types of boundary conditions, i.e.,
#    conditions on the boundaries of our domain, ``x = 0`` and ``x = L``.
#    - **Dirichlet Boundary Conditions**: This boundary condition means that the temperature
#      is specified at the boundaries of the domain.
#      ```math
#      u(0, t) = u_0 \quad \text{and} \quad u(L, t) = u_L
#      ```
#    - **Neumann Boundary Conditions**: This boundary condition means that, instead of the
#      temperature, the derivative of the temperature (which is called the heat flux) is
#      specified at the boundaries of the domain.
#      ```math
#      \alpha(0)\frac{\partial u}{\partial x}(0, t) = g_0 \quad \text{and} \quad \alpha(L)
#      \frac{\partial u}{\partial x}(L, t) = g_L
#      ```

# #### 3. Applications

# A simplified example where the heat equation can be used is to find out how the
# temperature is distributed through the outside, insulating walls of your apartment. Look
# at the image below
# ([source](https://www.anglian-building.co.uk/products/external-wall-insulation/)).
# ![wall-insulation](../assets/wall-insulation.jpg) #md
# ![wall-insulation](../../docs/src/assets/wall-insulation.jpg) #nb
#
# The insulation wall is made up of several materials, each with their own thermal
# diffusivities $\alpha_i(x)$. Imagine that the temperature outside is 0 degrees, and your
# heating system holds the temperature inside your house at 18 degrees. Then, these are the
# boundary conditions for the heat equation. Given an initial temperature distribution
# through the insulation wall, you could use the heat equation to find out how the
# temperature varies inside the insulation wall.

# ### B. Solving the Heat Equation

# #### 1. Disadvantages of the above formulation

# Now, if we want to compute the solution of the heat equation as stated above, we run into
# two difficulties:
# 1. **Lack of exact solutions**: For a general function ``f``, finding out the exact,
#    analytical solution ``u`` is not an easy task. Well, this is not entirely true:
#    finding the solution can be easy enough in one dimension on a domain as simple as
#    ``(0, L)``, but in higher dimensions and on more complicated geometries (e.g., imagine
#    the insulation wall of the
#    [Guggenheim museum](https://www.britannica.com/topic/Guggenheim-Museum-Bilbao)) it is
#    not possible.
# 2. **A restrictive set of solutions**: For the above form of the heat equation to make
#    sense, we must also assume that the second-derivatives of the solution ``u(x)`` should
#    exist, and that the first derivatives of the thermal diffusivity ``\alpha(x)`` should
#    exist. It turns out that this is too *strong* of a requirement that it not satisfied
#    by many physical systems.
#
# For example, think of the insulation wall - each material in the insulation wall has its
# own thermal diffusivity which is completely unrelated to the diffusivities of the other
# materials. As a result, ``\alpha(x)`` is a discontinuous function and its first
# derivatives do not make sense.

# #### 2. Tackling the above disadvantages using a discrete & weaker formulation

# The above disadvantages are the reason why, in practice, the above *strong* formulation
# of the heat equation is not useful. Instead, we formulate a *discrete, weak* version of
# the equation which is much more useful in practice. The motivation is:
# 1. **Discrete approximation of unknown exact solutions**: Since we don't know the exact
#    solution in general, we try to approximate it. This is the process called
#    discretization. In this process, we fix a finite-dimensional vector space of
#    *spatially-varying* functions ``V_n`` and say that, for any given time ``t``, we want
#    to find a function ``u_n(\cdot,t) \in V_n`` which approximates the exact solution
#    ``u(\cdot,t)``. Here, ``n`` denotes the dimension of the vector space ``V_n``. We
#    expect that as ``n \rightarrow \infty``, the solution
#    ``u_n(\cdot,t) \rightarrow u(\cdot,t)``.
# 2. **Weak version of the equation**: Since the original equation imposes too strong
#    requirements on the smoothness of ``u(x,t)`` and ``\alpha(x)``, we instead work with
#    an integral formulation where only the first derivatives of ``u(x,t)`` should make
#    sense, and where ``\alpha(x)`` is allowed to be discontinuous.
#
# > **_ASSUMPTION:_**  For the sake of simplifying the discussion, from now on we will
# > assume that we are imposing Dirichlet boundary conditions at ``x = 0`` and ``x = L``.
#
# This discrete, weak version of the problem at a *fixed time* ``t`` is stated as: find
# ``u_n(\cdot,t) \in S_n`` such that
# ```math
# \int_{0}^{L} w_n\frac{\partial u_n}{\partial t}\;\mathrm{d}x + \int_{0}^{L} \alpha
# \frac{\partial u_n}{\partial x}\frac{\partial w_n}{\partial x} \,\mathrm{d}x = \int_0^L
# f w_n \, dx\,,\qquad \forall w_n\in W_n\;,
# ```
# where:
# * ``S_n := \{v_n(x) \in V_n~:~v_n(0) = u_0\;,\;v_n(L) = u_L\}``,
# * ``W_n := \{v_n(x) \in V_n~:~v_n(0) = 0\;,\;v_n(L) = 0\}``.
#
# Note the following important things:
# 1. The above problem tries to find the solution ``u_n(\cdot,t)`` at the *fixed* time
#    instant ``t``.
# 2. ``S_n`` consists of spatially-varying functions in ``V_n`` that satisfy the boundary
#    conditions.
# 3. We want the integral equation above to be satisfied for all functions ``w_n \in W_n``,
#    where the function space ``W_n`` consists of spatially-varying functions in ``V_n``
#    that satisfy homogeneous (or, equivalently, zero) boundary conditions.


# ## The finite element method

# Now we are at a stage where, if we make a choice for ``V_n``, we can convert the discrete
# weak problem into a system of ODEs. This section will explain how.

# ### A. How to choose ``V_n``?

# #### 1. Choosing ``V_n``

# In *the finite element method*, we choose ``V_n`` as the space of piecewise-polynomial
# functions of degree ``p`` on a mesh of the domain ``(0, L)``.

# ##### Meshing the domain
# We choose a set of ``N+1`` points, ``0 = x_1 < x_1 < x_2 < \cdots < x_{N+1} = L``, and
# these points divide the domain ``(0, L)`` into smaller subdomains, ``(x_{i}, x_{i+1})``
# with ``x_{i}\in (0, L)``, called *elements*. That is, we assume that there are ``N``
# elements in our mesh.
#
# > **_ASSUMPTION:_** In the following code, we will always assume that the mesh is
# > *uniform*. In other words, ``x_{i+1}-x_i`` is equal to ``L/N`` for all ``i``.

# ##### Defining ``V_n``

# The space ``V_n`` is defined as the space of functions ``v_n`` such that:
# * on any element (i.e., on the interval ``(x_{i}, x_{i+1})`` ``with i = 1, \dots, N``) it
#   is a polynomial of degree ``p``,
# * at each ``x_i``, ``i = 2, \dots, N``, the function ``v_n`` is ``C^k`` smooth for some
#   ``k \geq 0``.
#
# Once we do this, the vector-space dimension of ``V_n`` can be related to the parameters
# ``N, p, k`` as follows:
# ```math
#  n = (p+1)N - (k+1)(N-1)\;.
# ```
# In other words, there are ``n`` basis functions ``\phi_{i}(x)``, ``i = 1, \dots, n``,
# such that any arbitrary ``v_n \in V_n`` can be represented as:
# ```math
# v_n(x) = \sum_{i=1}^n c_i \phi_{i}(x)\;,
# ```
# for some numbers ``c_i \in \mathbb{R}``.
#
# > **EXAMPLE:** ``V_n`` with ``(N,p,k) = (4, 1, 0)``.
# > Consider the space of functions that are linear polynomials over each mesh element, and
# > which are ``C^0`` smooth (or, equivalently, continuous) at the interfaces ``x_i``
# > between the elements. This space of functions has dimension:
# > ```math
# > n = (p+1)N - (k+1)(N-1) = 2\times 4 - 1 \times 3 = 5\;.
# > ```
# So, we can find 5 basis functions, ``\phi_{1}, \phi_{2}, \dots \phi_{5}``, that span the
# space ``V_n``. Run the code below to create such a ``V_n`` and look at one such choice of
# the basis functions called *hat functions*. Convince yourself that linear combinations of
# these functions can be used to represent any piecewise-linear polynomial function on the
# mesh. (Each function is plotted in a different color.)

import Mantis
import DisplayAs
using CairoMakie

## The size of the domain where to solve our problem
L = 1.0

## The degree of the piecewise-polynomial basis functions
p = 1
## The number of elements in the mesh
N = 4
## The smoothness of the basis functions (must be smaller than the polynomial degree, and
## larger than -1)
k = 0

## The number of basis functions in the piecewise-polynomial function space
n = N * (p + 1) - (k + 1) * (N - 1)

## Create the mesh and the function space
breakpoints = LinRange(0.0, L, N+1)
patch = Mantis.Mesh.Patch1D(collect(breakpoints))
B = Mantis.FunctionSpaces.BSplineSpace(patch, p, k)
line_geo = Mantis.Geometry.CartesianGeometry((breakpoints,))

## Create a Form Space.
BF = Mantis.Forms.FormSpace(0, line_geo, B, "b")

## Plot the basis functions.
n_plot_points_per_element = 25

fig = Figure()
ax = Axis(fig[1, 1],
    title = "Basis functions of V_n",
    xlabel = "x",
    ylabel = "b_i(x)",
)

n_elements = Mantis.Geometry.get_num_elements(line_geo)
xi = Mantis.Points.CartesianPoints((LinRange(0.0, 1.0, n_plot_points_per_element),))
BFF = Mantis.Forms.FormField(BF, " ")

dim_V = Mantis.Forms.get_num_basis(BF)
colors = [:blue, :green, :red, :purple, :orange]
for basis_idx in 1:dim_V

    BFF.coefficients[basis_idx] = 1.0
    if basis_idx > 1
        BFF.coefficients[basis_idx - 1] = 0.0
    end

    color_i = colors[basis_idx]

    for element_idx in 1:n_elements
        form_eval, _ = Mantis.Forms.evaluate(BFF, element_idx, xi)
        x = Mantis.Geometry.evaluate(Mantis.Forms.get_geometry(BF), element_idx, xi)

        lines!(ax, x[:], form_eval[1], color=color_i)

        scatter!(ax, x[:][[1, end]], [0.0, 0.0], color=:tomato)
    end
end

fig = DisplayAs.Text(DisplayAs.PNG(fig))
