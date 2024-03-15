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
- `breakpoints::NTuple{n, Vector{Float64}}`: Breakpoints per dimension. Should be strictly increasing.

# Extended help

# Implementation
The patch is assumed to be a tensor-product patch of quadilaterals. This 
is also reflected in the implementation.

# Error behaviour
The strictly increasing property of the breakpoints is checked. An 
`ArgumentError` is raised if the breakpoints are not strictly increasing.
"""
struct Patch{n}
    breakpoints::NTuple{n, Vector{Float64}}
    function Patch(breakpoints::NTuple{n, Vector{Float64}}) where {n}
        for d in 1:1:n
            for i in eachindex(breakpoints[d])[begin:end-1]
                if breakpoints[d][i] >= breakpoints[d][i+1]
                    msg1 = "Breakpoints should be strictly increasing, "
                    msg2 = "but the breakpoints in dimension $d are not."
                    throw(ArgumentError(msg1*msg2))
                end
            end
        end
        new{n}(breakpoints)        
    end
end

# 1D convenience constructor
function Patch(brk::Vector{Float64})
    return Patch((brk,))
end



# Getters (and setters) for an Element
function get_intervals(element::Element{n}) where {n}
    return element.intervals
end

"""
    size(element::Element{n}) where {n}

Returns the number of Intervals in an Element. This is the same as the dimension of the Element.

# Arguments
- `element::Element{n}`: Element of which to get the size.

# Returns
- `n::Int`: Number of intervals in the element.
"""
function Base.size(element::Element{n}) where {n}
    return n 
end

# Getters (and setters) for a Patch
function get_breakpoints(patch::Patch{n}) where {n}
    return patch.breakpoints
end

"""
    size(patch::Patch{n}) where {n}

Returns the number of elements in a Patch.

Redefinition of Base.size for a Patch. This ensure that one can call 
size(patch) to get the size of the patch.

# Arguments
- `patch::Patch{n}`: Patch of which to get the size.

# Returns
- `::NTuple{n, Int}`: Tuple of the number of elements per dimension.
"""
function Base.size(patch::Patch{n}) where {n}
    return NTuple{n, Int}(length(get_breakpoints(patch)[i]) - 1 for i = 1:1:n)   
end

function get_element_ids(patch::Patch{n}) where {n}
    return Iterators.product(NTuple{n,StepRange{Int,Int}}(1:1:nei for nei in size(patch))...)
end

function get_element(patch::Patch{n}, element_id::NTuple{n, Int}) where {n}
    return Element(NTuple{n,Interval}(Interval(patch.breakpoints[d][element_id[d]], patch.breakpoints[d][element_id[d]+1]) for d in 1:1:n))
end


end
