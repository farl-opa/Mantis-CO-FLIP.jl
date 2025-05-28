############################################################################################
#                                        Note                                              #
############################################################################################
#=
    This should be the structure for every benchmark file. To add a new set of benchmarks or
    groups, just copy the contents after this note; a few changes are needed:
        1. `include("BenchmarkHelpers.jl")` should be changed to correctly identify the
            helper file location relative to the benchmark file in question.
        2. Change `verbose`, `show`, and/or `save` to print runtime information, show the
            result of the benchmarks, and/or store them in a csv file, respectively.
        3. `add_benchmark_group!(...)` should be changed to account for the correct
            group/benchmark name and file location.
        4.  The `file_name` inside the `if show || save` block should be changed with the
            desired group/benchmark name. DO NOT change `file_location`, to ensure the
            structure of the stored csv files matches the benchmark's folder structure.
    
    For more information, read the documentation in `BenchmarkHelpers.jl`.
=#

module MantisBenchmarks

include("BenchmarkHelpers.jl")

verbose = true
show = false
save = false

# Global benchmark group
group = BenchmarkGroup()

# Module-level benchmark groups
if verbose
    println("Generating benchmarkable objects...")
end

group = add_benchmark_group!(group, "Assemblers", "Assemblers/runbenchmarks.jl")

if verbose
    println("Done!")
end

if show || save
    file_name = "All"
    file_location = relpath(dirname(@__FILE__), pwd())
    run_benchmarks(group, file_location, file_name; show=show, save=save, rtol=rtol)
end

end
