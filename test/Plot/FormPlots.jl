import Mantis

import Mantis

using Test

# Setup the form spaces
# First the FEM spaces
n_breakpoints1 = 10
breakpoints1 = collect(range(0,1,n_breakpoints1))
patch1 = Mantis.Mesh.Patch1D(breakpoints1)
n_breakpoints2 = 10
breakpoints2 = collect(range(0,1,n_breakpoints2))
patch2 = Mantis.Mesh.Patch1D(breakpoints2)

# first B-spline patch
deg1 = 2
deg2 = 2
reg1 = 1
reg2 = 1
B1 = Mantis.FunctionSpaces.BSplineSpace(patch1, deg1, [-1; repeat([reg1], n_breakpoints1-2); -1])
# second B-spline patch
B2 = Mantis.FunctionSpaces.BSplineSpace(patch2, deg2, [-1; repeat([reg2], n_breakpoints2-2); -1])
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
zero_form_space = Mantis.Forms.FormSpace(0, tensor_prod_geo, (TP_Space,), "ν")
one_form_space = Mantis.Forms.FormSpace(1, tensor_prod_geo, (TP_Space, TP_Space), "η")
top_form_space = Mantis.Forms.FormSpace(2, tensor_prod_geo, (TP_Space,), "σ")

# Generate the form expressions
α⁰ = Mantis.Forms.FormField(zero_form_space, "α")
ξ¹ = Mantis.Forms.FormField(one_form_space, "ξ")
β² = Mantis.Forms.FormField(top_form_space, "β")

num_basis = Mantis.FunctionSpaces.get_num_basis(TP_Space)

α⁰.coefficients .= collect(range(0,3,num_basis))
ξ¹.coefficients .= vcat(ones(num_basis), collect(range(0,3,num_basis)))
β².coefficients .= collect(range(0,3,num_basis)) 



# Compute base directories for data input and output
Mantis_folder =  dirname(dirname(pathof(Mantis)))
data_folder = joinpath(Mantis_folder, "test", "data")
output_data_folder = joinpath(data_folder, "output", "Forms")

out_deg = maximum([1, maximum((deg1,deg2))])

zero_form_filename = "zero-form-field-test.vtu"
zero_form_file = joinpath(output_data_folder, zero_form_filename)

one_form_filename = "one-form-field-test.vtu"
one_form_file = joinpath(output_data_folder, one_form_filename)

two_form_filename = "two-form-field-test.vtu"
two_form_file = joinpath(output_data_folder, two_form_filename)

#Mantis.Plot.plot(α⁰; vtk_filename = zero_form_file, n_subcells = 1, degree = out_deg, ascii = false, compress = false)
#Mantis.Plot.plot(ξ¹; vtk_filename = one_form_file, n_subcells = 1, degree = out_deg, ascii = false, compress = false)
Mantis.Plot.plot(β²; vtk_filename = two_form_file, n_subcells = 1, degree = out_deg, ascii = false, compress = false)

