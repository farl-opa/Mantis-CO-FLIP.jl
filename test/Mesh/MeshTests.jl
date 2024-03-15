

import Mantis

using Test

const Patch = Mantis.Mesh.Patch

n1 = 5
n2 = 8
n3 = 6
n4 = 3
test_brk = collect(LinRange(0.0, 1.0, n1))
test_brk2 = collect(LinRange(-0.25, 2.25, n2))
test_brk3 = collect(LinRange(-1.0, 0.0, n3))
test_brk4 = collect(LinRange(-10.0, -8.0, n4))

test_brk_wrong_order = [0.0, -1.0, 2.0]
test_brk_wrong_order2 = [0.0, 0.0, 1.0]

# The break points are not strictly increasing, which should be checked 
# when creating the Patch and should work also if other breakpoints are 
# correct. Note that the 1D Patch constructor uses a convience 
# constructor, so both the convenience constructor and the general 
# version are used to make sure they work.
@test_throws ArgumentError Patch(test_brk_wrong_order)  # Convenience constructor
@test_throws ArgumentError Patch((test_brk_wrong_order2,))  # General constructor
@test_throws ArgumentError Patch((test_brk, test_brk_wrong_order))
@test_throws ArgumentError Patch((test_brk_wrong_order, test_brk_wrong_order2))
@test_throws ArgumentError Patch((test_brk, test_brk2, test_brk_wrong_order2))

# Test if the size of a Patch or element are correctly computed. Note 
# that these are redefinitions of Base.size so that one can simply call 
# size() on either an Element or Patch.
test_patch = Patch((test_brk, test_brk2, test_brk3, test_brk4))
@test size(test_patch) == (n1-1, n2-1, n3-1, n4-1)
@test size(Mantis.Mesh.get_element(test_patch, (4,1,3,1))) == 4

#test_patch = Patch(test_brk)
#test_patch = Patch((test_brk, test_brk2))
# test_patch = Mantis.Mesh.Patch((test_brk, test_brk2, test_brk3))
# #test_patch = Patch((test_brk, test_brk2, test_brk3, test_brk4))
# println("All breakpoints:")
# println(get_breakpoints(test_patch))
# println("Element IDs with the Element:")
# println("Size of the patch:")
# for elem_id in get_element_ids(test_patch)
#     elem = get_element(test_patch, elem_id)
#     println(elem_id, " ", elem, " ", size(elem))
# end
# println("Size of the patch:")
# println(size(test_patch))

