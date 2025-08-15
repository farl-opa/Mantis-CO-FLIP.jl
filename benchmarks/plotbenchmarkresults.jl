using CairoMakie
using CSV
using DataFrames
using Dates

data = Dict{Tuple{String, String}, DataFrame}()

benchmark = joinpath(
    pwd(),
    "benchmarks",
    "data",
    "Assemblers",
    "deRham",
    "1-FormHodgeLaplace",
    "TensorProduct-BSplines",
    "Bernstein"
)

file_location = relpath(dirname(@__FILE__), pwd())
for file in readdir(benchmark; join=true)
    println("Processing file: $file")
    if endswith(file, ".csv")
        df = DataFrame(CSV.File(file))
        hostname = split(basename(file), "-")[2]
        benchmarkname = join(split(basename(file), "-")[3:end], "-")[1:end-4]
        data[(hostname, benchmarkname)] = df
    end
end

colours = Dict(
    "JoeyDekker" => :orange,
    "DiogoCabanas" => :blue,
)

markers = Dict(
    "JoeyDekker" => :circle,
    "DiogoCabanas" => :cross,
)

benchmark_names = unique([name[2] for name in keys(data)])

figures = Dict{String, Figure}()
for name in benchmark_names
    fig = Figure()
    figures[name] = fig
    ax = Axis(fig[1, 1], xlabel = "Dates", ylabel = "Mimimum allocations")
    ax.title = name
    dates = String[]
    for (names, df) in data
        if names[2] == name
            push!(dates, Dates.format(df.date[1], "yyyy-mm-dd"))
            scatter!(ax, 1:1:length(df.min_allocs), df.min_allocs;
                label=names[1], color=colours[names[1]], marker=markers[names[1]])
        end
    end
    ax.xticks = 1:1:length(dates)
    ax.xtickformat = values -> [dates[Int(i)] for i in values]
end
