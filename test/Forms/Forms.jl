import Mantis

α⁰ = Mantis.Forms.FormField(2, 0, "α⁰")
β⁰ = Mantis.Forms.FormField(2, 0, "β⁰")
θ⁰ = Mantis.Forms.FormExpression(α⁰, β⁰, 0, "∧")
ζ⁰ = Mantis.Forms.FormExpression(α⁰, θ⁰, 0, "∧")

print(θ⁰.label)
print("\n")
ζ⁰ = Mantis.Forms.FormExpression(α⁰, θ⁰, 0, "∧")
print(ζ⁰.label)
print("\n")
σ⁰ = Mantis.Forms.FormExpression(θ⁰, ζ⁰, 0, "⋆")
print(σ⁰.label)
print("\n")

evaluate(α⁰)
print("\n")

evaluate(θ⁰)
print("\n")

evaluate(σ⁰)
print("\n")

γ = ∧(α⁰, β⁰)
print(γ.label)

γ2 = ∧(γ, β⁰)
print("\n")
print(γ2.label)
print("\n")
evaluate(γ2)



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