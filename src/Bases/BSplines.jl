struct BSpline{n}<:AbstractBases
    patch::Patch{n}
    p::NTuple{n, Vector{Int64}}
    k::NTuple{n, Vector{Int64}}
    rank::Int
end