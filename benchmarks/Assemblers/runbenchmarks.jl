module AssemblersBenchmarks

include("../BenchmarkHelpers.jl")

verbose = false
show = false
save = false

group = BenchmarkGroup()

if verbose
    println("Generating benchmarkable objects...")
end

group = add_benchmark_group!(group, "HodgeLaplace", "HodgeLaplace/runbenchmarks.jl")

if verbose
    println("Done!")
end

if show || save
    file_name = "Assemblers"
    file_location = relpath(dirname(@__FILE__), pwd())
    run_benchmarks(group, file_location, file_name; show=show, save=save, rtol=rtol)
end

end
