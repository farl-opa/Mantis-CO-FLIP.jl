"""
    module Mesh

Contains all mesh structs and functions. At the moment, these are simple 
versions to be able to start experimenting with the other parts of the 
code. This module makes extensive use of getters and setters, so the 
internals of a struct are considered 'hidden' details that **can change 
without warning**, even those not preceded with an underscore.
"""
module Mesh


"""
    struct Interval

1-dimensional interval. Note that `left` < `right`.

# Fields
- `left::Float64`: Coordinate of the left boundary of the interval.
- `right::Float64`: Coordinate of the right boundary of the interval.
"""
struct Interval 
    left::Float64
    right::Float64
end

"""
    struct Element{n}

n-dimensional tensor-product element.

# Fields
- `intervals::NTuple{n, Interval}`: Interval per dimension that indicate the boundaries of the element.
"""
struct Element{n}
    intervals::NTuple{n, Interval}
end

"""
    struct Patch{n}

Simple struct for an n-dimensional **structured** tensor-product patch.

# Fields
- `breakpoints::NTuple{n, Vector{Float64}}`: Breakpoints per dimension.

# Extended help

# Implementation
The patch is assumed to be a tensor-product patch of quadilaterals. This 
is also reflected in the implementation.
"""
struct Patch{n}
    breakpoints::NTuple{n, Vector{Float64}}
end

# 1D convenience constructor
function Patch(brk::Vector{Float64})
    return Patch((brk,))
end



# Getters (and setters) for an Element
function get_intervals(element::Element{n}) where {n}
    return element.intervals
end

function size(element::Element{n}) where {n}
    return n 
end

# Getters (and setters) for a Patch
function get_breakpoints(patch::Patch{n}) where {n}
    return patch.breakpoints
end

function size(patch::Patch{n}) where {n}
    return NTuple{n, Int}(length(get_breakpoints(patch)[i]) - 1 for i = 1:1:n)   
end

function get_element_ids(patch::Patch{n}) where {n}
    return Iterators.product(NTuple{n,StepRange{Int,Int}}(1:1:nei for nei in size(patch))...)
end

function get_element(patch::Patch{n}, element_id::NTuple{n, Int}) where {n}
    return Element(NTuple{n,Interval}(Interval(patch.breakpoints[d][element_id[d]], patch.breakpoints[d][element_id[d]+1]) for d in 1:1:n))
end

n1 = 5
n2 = 8
n3 = 6
#n4 = 3
test_brk = collect(LinRange(0.0, 1.0, n1))
test_brk2 = collect(LinRange(-0.25, 2.25, n2))
test_brk3 = collect(LinRange(-1.0, 0.0, n3))
#test_brk4 = collect(LinRange(-10.0, -8.0, n4))
#test_patch = Patch(test_brk)
#test_patch = Patch((test_brk, test_brk2))
test_patch = Patch((test_brk, test_brk2, test_brk3))
#test_patch = Patch((test_brk, test_brk2, test_brk3, test_brk4))
println("All breakpoints:")
println(get_breakpoints(test_patch))
println("Element IDs with the Element:")
println("Size of the patch:")
for elem_id in get_element_ids(test_patch)
    elem = get_element(test_patch, elem_id)
    println(elem_id, " ", elem, " ", size(elem))
end
println("Size of the patch:")
println(size(test_patch))

BSpline(test_patch)

end
