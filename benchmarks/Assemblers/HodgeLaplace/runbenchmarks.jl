module HodgeLaplaceBenchmarks

include("../../BenchmarkHelpers.jl")

verbose = false
show = true
save = true

group = BenchmarkGroup()

if verbose
    println("Generating benchmarkable objects...")
end

group = add_benchmark_group!(group, "1-form", "1-form-TensorProduct.jl")

if verbose
    println("Done!")
end

if show || save
    file_name = "Assemblers-HodgeLaplace"
    file_location = relpath(dirname(@__FILE__), pwd())
    run_benchmarks(group, file_location, file_name; show=show, save=save, rtol=rtol)
end

end
