import Mantis

# Setup the form spaces
# First the FEM spaces
breakpoints1 = [0.0, 0.5, 1.0]
patch1 = Mantis.Mesh.Patch1D(breakpoints1)
breakpoints2 = [0.0, 0.5, 0.6, 1.0]
patch2 = Mantis.Mesh.Patch1D(breakpoints2)

# first B-spline patch
deg1 = 2
deg2 = 2
B1 = Mantis.FunctionSpaces.BSplineSpace(patch1, deg1, [-1, deg1-1, -1])
# second B-spline patch
B2 = Mantis.FunctionSpaces.BSplineSpace(patch2, deg2, [-1, min(deg2-1,1),  deg2-1, -1])
# tensor-product B-spline patch
TP_Space = Mantis.FunctionSpaces.TensorProductSpace(B1, B2)

# Then the geometry 
# Line 1
line_1_geo = Mantis.Geometry.CartesianGeometry((breakpoints1,))

# Line 2
line_2_geo = Mantis.Geometry.CartesianGeometry((breakpoints2,))

# Tensor product geometry 
tensor_prod_geo = Mantis.Geometry.TensorProductGeometry(line_1_geo, line_2_geo)

# Then the form space 
zero_form_space = Mantis.Forms.FormSpace(0, tensor_prod_geo, TP_Space)
top_form_space = Mantis.Forms.FormSpace(2, tensor_prod_geo, TP_Space)

# Generate the form expressions
α⁰ = Mantis.Forms.FormField(zero_form_space, "α")
β² = Mantis.Forms.FormField(top_form_space, "β")
θ² = Mantis.Forms.FormExpression(α⁰, β², 2, "∧")
ζ² = Mantis.Forms.FormExpression(α⁰, θ², 2, "∧")

print(θ².label)
print("\n")
ζ² = Mantis.Forms.FormExpression(α⁰, θ², 2, "∧")
print(ζ².label)
print("\n")
σ⁰ = Mantis.Forms.FormExpression(α⁰, ζ², 0, "⋆")
print(σ⁰.label)
print("\n")

Mantis.Forms.evaluate(α⁰)
print("\n")

Mantis.Forms.evaluate(θ²)
print("\n")

Mantis.Forms.evaluate(σ⁰)
print("\n")


function (∧)(form_1::F_1, form_2::F_2) where {F_1 <: Mantis.Forms.AbstractFormExpression{manifold_dim_1, form_rank_1}, F_2 <: Mantis.Forms.AbstractFormExpression{manifold_dim_2, form_rank_2}} where {manifold_dim_1, form_rank_1, manifold_dim_2, form_rank_2}
    return Mantis.Forms.FormExpression(form_1, form_2, form_rank_1 + form_rank_2, "∧")
end

function (⋆)(form::F) where {F <: Mantis.Forms.AbstractFormExpression{manifold_dim, form_rank}} where {manifold_dim, form_rank}
    return Mantis.Forms.FormExpression(form_1, form_2, form_rank_1 + form_rank_2, "∧")
end

print("Evaluating the wedge operator\n")
γ = ∧(α⁰, β²)
print(γ.label)

γ2 = ∧(γ, β²)
print("\n")
print(γ2.label)
print("\n")
Mantis.Forms.evaluate(γ2)



# abstract type AbstractMyStruct{n} end
# abstract type AbstractOtherStruct{n} end

# struct OtherStruct{k} <: AbstractOtherStruct{k}
#     a::Int
# end

#  struct MyStruct{k, F} <: AbstractMyStruct{k}
#     fem_space::F

#     # 0- and n-form constructor
#     function MyStruct(form_rank::Int, fem_space::F) where {F <: AbstractOtherStruct{k}} where {k}
#         new{k, F}(fem_space)
#     end
# end