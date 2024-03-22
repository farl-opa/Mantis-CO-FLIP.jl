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
    Patch{n}

Is an n-dimensional **structured** tensor-product patch.

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

    # Field(s)

    # Only the 1D breakpoints are stored, as a tensor-product structure 
    # is assumed.
    breakpoints::NTuple{n, Vector{Float64}}

    # Inner constructor(s)
    # The inner constructor only tests if the breakpoints in each 
    # dimension are strictly increasing.
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
"""
    get_intervals(element::Element{n}) where {n}

Returns the `Intervals` per dimension.

# Arguments
- `element::Element{n}`: Element of which to get the Intervals.

# Returns
- `::NTuple{n, Interval}`: Interval per dimension.
"""
function get_intervals(element::Element{n}) where {n}
    return element.intervals
end


# Getters (and setters) for a Patch
"""
    get_breakpoints(patch::Patch{n}) where {n}

Returns the breakpoints per dimension.

# Arguments
- `patch::Patch{n}`: Patch of which to get the breakpoints.

# Returns
- `::NTuple{n, Vector{Float64}}`: Breakpoints per dimension.
"""
function get_breakpoints(patch::Patch{n}) where {n}
    return patch.breakpoints
end

"""
    size(patch::Patch{n}) where {n}

Returns the number of elements in the `patch`.

Redefinition of `Base.size` for a `Patch`. This ensures that one can 
call `size(patch)` to get the size of the patch.

# Arguments
- `patch::Patch{n}`: Patch of which to get the size.

# Returns
- `::NTuple{n, Int}`: Tuple of the number of elements per dimension.
"""
function Base.size(patch::Patch{n}) where {n}
    return NTuple{n, Int}(length(get_breakpoints(patch)[i]) - 1 for i = 1:1:n)   
end

"""
    get_element_ids(patch::Patch{n}) where {n}

Returns an iterator of the valid element ids for this `patch`.

The element ids are computed on-the-fly by the iterator and are neither 
precomputed nor stored.

# Arguments
- `patch::Patch{n}`: Patch to get the element ids for.

# Returns
- `::Base.Iterators.ProductIterator{NTuple{n, StepRange{Int, Int}}}`: Iterator of NTuples of element ids.
"""
function get_element_ids(patch::Patch{n}) where {n}
    return Iterators.product(NTuple{n,StepRange{Int,Int}}(1:1:nei for nei in size(patch))...)
end

"""
    get_element(patch::Patch{n}, element_id::NTuple{n, Int}) where {n}

Returns the `Element` that is referred to by `element_id`.

Note that this function creates the `Element` on-the-fly, so it is 
neither precomputed not stored.

# Arguments
- `patch::Patch{n}`: Patch on which to look.
- `element_id::NTuple{n, Int}`: Element which is to be returned.

# Returns
- `::Element`: Element object containing the intervals per dimension.
"""
function get_element(patch::Patch{n}, element_id::NTuple{n, Int}) where {n}
    return Element(NTuple{n,Interval}(Interval(patch.breakpoints[d][element_id[d]], patch.breakpoints[d][element_id[d]+1]) for d in 1:1:n))
end


end
