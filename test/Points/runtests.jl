module PointsTests

using Test

@testset "AbstractPoints" begin
    include("AbstractPoints.jl")
end

@testset "PointSet" begin
    include("PointSet.jl")
end

@testset "CartesianPoints" begin
    include("CartesianPoints.jl")
end

end
