using GLMakie
using CSV
using DataFrames
using Dates

include("BenchmarkHelpers.jl")


const colours = Dict(
    "JoeyDekker" => :orange,
    "DiogoCabanas" => :blue,
    "ArturPalha" => :green,
    "DeepeshToshniwal" => :red,
)

const markers = Dict(
    "JoeyDekker" => :circle,
    "DiogoCabanas" => :cross,
    "ArturPalha" => :diamond,
    "DeepeshToshniwal" => :star6,
)

data = Dict{Tuple{String, String}, DataFrame}()


data_dir = joinpath(pwd(), "data")
benchmark_series_vec = String[]
for (dirpath, dirnames, filenames) in walkdir(data_dir)
    if length(filenames) > 0
        push!(benchmark_series_vec, dirpath)
    end
end
unique!(sort!(benchmark_series_vec))

function get_benchmarks(benchmark_series)
    benchmarks = String[]
    benchmarknames = String[]
    hostnames = String[]
    for (dirpath, dirnames, filenames) in walkdir(benchmark_series)
        for filename in filenames
            if endswith(filename, ".csv")
                # Get the benchmark name without the extension and host name
                name_parts = split(filename, "-")
                hostname = name_parts[2]
                benchmarkname = join(name_parts[3:end], "-")[1:end-4]
                push!(benchmarks, joinpath(dirpath, filename))
                push!(benchmarknames, benchmarkname)
                push!(hostnames, hostname)
            end
        end
    end
    return unique!(sort!(benchmarks)), unique!(sort!(benchmarknames)), unique!(sort!(hostnames))
end

function update_data!(data, name, dirname, hostnames)
    for hostname in hostnames
        filename = "host-$hostname-$name.csv"
        println("Updating data for file: $filename")
        df = DataFrame(CSV.File(joinpath(dirname, filename)))
        data[(hostname, name)] = df
    end
    return data
end


# Create a root observable with the benchmark series name. Whenever this changes, everything
# else updates automatically. Set the first benchmark as default.
benchmark_series_ob = Observable(benchmark_series_vec[1])


benchmark_names_ob = Observable(get_benchmarks(to_value(benchmark_series_ob))[2])
benchmark_name_ob = Observable(to_value(benchmark_names_ob)[1])

update_data!(data, to_value(benchmark_name_ob), to_value(benchmark_series_ob), get_benchmarks(to_value(benchmark_series_ob))[3])
benchmark_data_ob = Observable(data)



fig = Figure()
println(to_value(benchmark_series_ob))
# Menus:
# Benchmark series menu
menu_benchmark_series = Menu(fig, options = benchmark_series_vec, default = to_value(benchmark_series_ob))
# Benchmark menu
menu_benchmark_names = Menu(fig, options = benchmark_names_ob, default = to_value(benchmark_name_ob))

# Create a list of menus on the left side of the figure.
fig[1, 1] = vgrid!(
    Label(fig, "Benchmark Series", width = nothing),
    menu_benchmark_series,
    Label(fig, "Benchmark", width = nothing),
    menu_benchmark_names;
    tellheight = false, width = 200
)



ax = Axis(fig[1, 2], xlabel = "Hash date [yyyy-mm-dd]", ylabel = "Mimimum time [ms]", xticklabelrotation=45.0)

col_x = Observable(:commit_date)
col_y = Observable(:min_time)

x_data = lift(benchmark_data_ob, benchmark_name_ob, col_x) do data, name, col
    df = first(values(data))
    all_dates = Dates.format.(df.commit_date, "yyyy-mm-dd")
    ax.xticks = 1:1:length(all_dates)
    ax.xtickformat = values -> [all_dates[Int(i)] for i in eachindex(values)]
    return 1:1:length(all_dates)
end

y_data_joey = lift(benchmark_data_ob, benchmark_name_ob, col_y) do data, name, col
    return get_number_in_string.(data[("JoeyDekker", name)][:,col], new_unit="ms")
end
y_data_diogo = lift(benchmark_data_ob, benchmark_name_ob, col_y) do data, name, col
    return get_number_in_string.(data[("DiogoCabanas", name)].min_time, new_unit="ms")
end
y_data_artur = lift(benchmark_data_ob, benchmark_name_ob, col_y) do data, name, col
    return get_number_in_string.(data[("ArturPalha", name)].min_time, new_unit="ms")
end
y_data_deepesh = lift(benchmark_data_ob, benchmark_name_ob, col_y) do data, name, col
    return get_number_in_string.(data[("DeepeshToshniwal", name)].min_time, new_unit="ms")
end

scatter!(ax, x_data, y_data_joey;
    label="JoeyDekker", color=colours["JoeyDekker"], marker=markers["JoeyDekker"]
)
scatter!(ax, x_data, y_data_diogo;
    label="DiogoCabanas", color=colours["DiogoCabanas"], marker=markers["DiogoCabanas"]
)
scatter!(ax, x_data, y_data_artur;
    label="ArturPalha", color=colours["ArturPalha"], marker=markers["ArturPalha"]
)
scatter!(ax, x_data, y_data_deepesh;
    label="DeepeshToshniwal", color=colours["DeepeshToshniwal"], marker=markers["DeepeshToshniwal"]
)

axislegend("Host names", position = :rt)


# Define what happens when a menu selection changes.
on(menu_benchmark_series.selection) do s
    println("Selected benchmark series: $s")
    benchmark_names_ob[] = get_benchmarks(s)[2]
    fig.title = s
end

on(menu_benchmark_names.selection) do s
    println("Selected benchmark name: $s")
    benchmark_stringdata = get_benchmarks(benchmark_series_ob[])
    # benchmark_data_ob[] = update_data!(data, s, to_value(benchmark_series_ob), benchmark_stringdata[3])
    update_data!(data, s, to_value(benchmark_series_ob), benchmark_stringdata[3])
    benchmark_name_ob[] = s
    ax.title = s
    autolimits!(ax)
end

display(fig)
