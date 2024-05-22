

import Mantis

using Test

function forcing_function(x)
    return pi^2 * sinpi(x)
end

function bilinear_function(Ndi, Ndj)
    return Ndi * Ndj
end

function linear_function(Ni, f)
    return Ni * f
end


# Domain.
Lleft = 0.0
Lright = 2.0
# Number of elements.
m = 4
# Boundary conditions (Dirichlet).
bc_left = 0.0
bc_right = 1.0
# polynomial degree and inter-element continuity.
p = 1
k = 0

# Create Patch.
brk = collect(LinRange(Lleft, Lright, m+1))
patch = Mantis.Mesh.Patch1D(brk)
# Continuity vector for OPEN knot vector.
kvec = fill(k, (m+1,))
kvec[1] = -1
kvec[end] = -1
# Create function spaces (b-splines here).
trail_space = Mantis.FunctionSpaces.BSplineSpace(patch, p, kvec)
test_space = Mantis.FunctionSpaces.BSplineSpace(patch, p, kvec)

# Create the geometry.
geom = Mantis.Geometry.CartesianGeometry((brk,))

# Setup the quadrature rule.
q_nodes, q_weights = Mantis.Quadrature.gauss_legendre(p+1)

# Setup the element assembler.
element_assembler = Mantis.Assemblers.PoissonBilinearForm1D(forcing_function,
                                                            bilinear_function,
                                                            linear_function,
                                                            trail_space,
                                                            test_space,
                                                            geom,
                                                            q_nodes,
                                                            q_weights)

# Setup the global assembler.
global_assembler = Mantis.Assemblers.Assembler(bc_left, bc_right)

# Assemble.
A, b = global_assembler(element_assembler)

println("A:")
display(A)
println("b:")
println(Vector(b))


# Solve & add bcs.
sol = [bc_left, (A \ Vector(b))..., bc_right]

println("sol:")
println(sol)

if p == 1 && k == 0
    @test all(isapprox.(sol, [bc_left, 1.2548348693320486, 0.5, -0.25483486933204813, bc_right], atol=1e-14))
end




# # Example on more complicated geometry
# # Domain.
# Lleft = 0.0
# Lright = 2.0
# # Number of elements.
# m = 4
# # Boundary conditions (Dirichlet).
# bc_left = 0.0
# bc_right = 1.0
# # polynomial degree and inter-element continuity.
# p = 1
# k = 0

# # Create Patch.
# brk = collect(LinRange(Lleft, Lright, m+1))
# patch = Mantis.Mesh.Patch1D(brk)
# # Continuity vector for OPEN knot vector.
# kvec = fill(k, (m+1,))
# kvec[1] = -1
# kvec[end] = -1
# # Create function spaces (b-splines here).
# trail_space = Mantis.FunctionSpaces.BSplineSpace(patch, p, kvec)
# test_space = Mantis.FunctionSpaces.BSplineSpace(patch, p, kvec)

# # Create the geometry.
# geom = Mantis.Geometry.CartesianGeometry((brk,))

# # Setup the quadrature rule.
# q_nodes, q_weights = Mantis.Quadrature.gauss_legendre(p+1)

# # Setup the element assembler.
# element_assembler = Mantis.Assemblers.PoissonBilinearForm1D(forcing_function,
#                                                             bilinear_function,
#                                                             linear_function,
#                                                             trail_space,
#                                                             test_space,
#                                                             geom,
#                                                             q_nodes,
#                                                             q_weights)

# # Setup the global assembler.
# global_assembler = Mantis.Assemblers.Assembler(bc_left, bc_right)

# # Assemble.
# A, b = global_assembler(element_assembler)

# # println("A:")
# # display(A)
# # println("b:")
# # println(Vector(b))


# # Solve & add bcs.
# sol = [bc_left, (A \ Vector(b))..., bc_right]

# # println("sol:")
# # println(sol)

# if p == 1 && k == 0
#     @test all(isapprox.(sol, [bc_left, 1.2548348693320486, 0.5, -0.25483486933204813, bc_right], atol=1e-14))
# end

