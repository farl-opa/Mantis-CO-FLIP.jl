using GLMakie
using CSV
using DataFrames
using Dates

include("BenchmarkHelpers.jl")

benchmark = joinpath(
    pwd(),
    "data",
    "Assemblers",
    "deRham",
    "1-FormHodgeLaplace",
    "TensorProduct-BSplines",
    "Bernstein"
)

# Stores all data for a given benchmark as (hostname, benchmarkname) => DataFrame
data = Dict{Tuple{String, String}, DataFrame}()
benchmark_names = String[]
function process_benchmark!(data, name::String)
    # file_location = relpath(dirname(@__FILE__), pwd())
    for file in readdir(name; join=true)
        println("Processing file: $file")
        if endswith(file, ".csv")
            df = DataFrame(CSV.File(file))
            hostname = split(basename(file), "-")[2]
            benchmarkname = join(split(basename(file), "-")[3:end], "-")[1:end-4]
            push!(benchmark_names, benchmarkname)
            data[(hostname, benchmarkname)] = df
        end
    end
end

process_benchmark!(data, benchmark)

println(benchmark_names)
unique!(sort!(benchmark_names))
println("Benchmark names: ", benchmark_names)


colours = Dict(
    "JoeyDekker" => :orange,
    "DiogoCabanas" => :blue,
    "ArturPalha" => :green,
    "DeepeshToshniwal" => :red,
)

markers = Dict(
    "JoeyDekker" => :circle,
    "DiogoCabanas" => :cross,
    "ArturPalha" => :diamond,
    "DeepeshToshniwal" => :star6,
)

name = "2D-num_dofs=341-p=3-k=2"

fig = Figure()

menu_unit = Menu(fig, options = ["ns", "μs", "ms", "s"], default = "ms")
menu_benchmark = Menu(fig, options = benchmark_names, default = name)
fig[1, 1] = vgrid!(
    Label(fig, "Unit", width = nothing),
    menu_unit,
    Label(fig, "Benchmark", width = nothing),
    menu_benchmark;
    tellheight = false, width = 200)

ax = Axis(fig[1, 2], xlabel = "Date", ylabel = "Mimimum time [ms]", xticklabelrotation=45.0)
ax.title = name
dates = Dict{String, Vector{String}}(
    "JoeyDekker" => String[],
    "DiogoCabanas" => String[],
    "ArturPalha" => String[],
    "DeepeshToshniwal" => String[],
)
for (names, df) in data
    if names[2] == name
        push!(dates[names[1]], Dates.format.(df.date, "yyyy-mm-dd")...)
        scatter!(ax, 1:1:length(df.min_time), get_number_in_string.(df.min_time, new_unit="ms");
            label=names[1], color=colours[names[1]], marker=markers[names[1]])
    end
end
all_dates = unique!(reduce(vcat, collect(values(dates))))
ax.xticks = 1:1:length(all_dates)
ax.xtickformat = values -> [all_dates[Int(i)] for i in values]
axislegend("Host names", position = :rt)

on(menu_unit.selection) do s
    println("Selected unit: $s")
    ax.ylabel = "Mimimum time [$s]"
    for (names, df) in data
        if names[2] == name
            scatter!(ax, 1:1:length(df.min_time), get_number_in_string.(df.min_time, new_unit=s);
                label=names[1], color=colours[names[1]], marker=markers[names[1]])
        end
    end
    autolimits!(ax)
end
notify(menu_unit.selection)

on(menu_benchmark.selection) do s
    println("Selected benchmark: $s")
    ax.title = s
end
notify(menu_benchmark.selection)

display(fig)