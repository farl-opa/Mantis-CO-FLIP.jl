import Mantis

# include("HelperFunctions.jl")

deg = (2,2)
reg = (1,1)

num_els = (5,5)
brk = (collect(range(0,1,num_els[1]+1)).^2, collect(range(0,1,num_els[2]+1)).^2)
patch = (Mantis.Mesh.Patch1D(brk[1]), Mantis.Mesh.Patch1D(brk[2]))
reg_vec = ([-1; fill(reg[1], num_els[1]-1); -1], [-1; fill(reg[2],num_els[2]-1); -1])
bspline = (Mantis.FunctionSpaces.BSplineSpace(patch[1], deg[1], reg_vec[1]), Mantis.FunctionSpaces.BSplineSpace(patch[2], deg[2], reg_vec[2]))
tp = Mantis.FunctionSpaces.TensorProductSpace(bspline[1], bspline[2])
# for el in 1:Mantis.FunctionSpaces.get_num_elements(tp)
#     display(Mantis.FunctionSpaces.get_element_dimensions(tp,el))
# end

Mantis_folder =  dirname(dirname(pathof(Mantis)))
data_folder = joinpath(Mantis_folder, "test", "data")
output_data_folder = joinpath(data_folder, "output", "Geometry")

output_filename = "fem-test.vtu"
output_file = joinpath(output_data_folder, output_filename)

geo = Mantis.Geometry.compute_parametric_geometry(tp)

space = Mantis.Forms.FormSpace(2, geo, Mantis.FunctionSpaces.DirectSumSpace((tp,)), "t")
field = Mantis.Forms.FormField(space,"t")
coeffs = zeros(Mantis.Forms.get_num_basis(field))
coeffs[25] = 1
field.coefficients .= coeffs

Mantis.Plot.plot(field; vtk_filename = output_file[1:end-4] * "_v1", n_subcells = 1, degree = 4, ascii = false, compress = false)
