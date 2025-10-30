module AbstractPointsTests

using Mantis
using Test

struct UnknownPoints{T} <: Points.AbstractPoints{1}
    points::T
end

unknown_points = UnknownPoints("something")

@test_throws MethodError Points.get_num_points(unknown_points)

end
