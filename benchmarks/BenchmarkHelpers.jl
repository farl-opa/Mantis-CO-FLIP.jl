using BenchmarkTools
using CSV
using DataFrames

const rtol = 0.05

"""
    add_benchmark_group!(group::BenchmarkGroup, name::String, file_path::String)

Append a benchmark sub-group from `file_path` to `group` with a given `name`. Note, added
sub-group will be the variable named `group` inside `file_path`; the contents of `file_path`
should be a module.

# Arguments
- `group::BenchmarkGroup`: The benchmark group to which the sub-group will be appended.
- `name::String`: The name of the appended sub-group.
- `file_path::String`: The file containing the appended sub-group.
# Returns
- `group::BenchmarkGroup`: The given group with the appended sub-group.
"""
function add_benchmark_group!(group::BenchmarkGroup, name::String, file_path::String)
    mod = include(file_path)
    sub_group = getproperty(mod, :group)
    group[name] = sub_group

    return group
end

"""
    run_benchmarks(
        group::BenchmarkGroup, location::String, name::String;
        show=false, save=false, rtol=0.05
    )

Runs the benchmarks from all nested `@benchmarkable` objects found in `group`. Optionally,
the flags `show` and `save` can be set to `true` to display the ran benchmarks and/or store
in a CSV file, respectively. In the latter case, `location` will be use to determine where
the file is stored and `rtol` is the cutoff determining if a benchmark is worse than before.
Note that `location` should almost always be set automatically with
`relpath(dirname(@__FILE__), pwd())` in the file calling this function. This ensures that
the folder structure in `data/` matches the benchmarks' folder structure.

# Arguments
- `group::BenchmarkGroup`: The benchmark group being used.
- `location::String`: The file folder location.
- `name::String`: The name of the stored file. This will be appended to the hostname
    automatically.
- `show=false`: Flag to display benchmark results.
- `save=false`: Flag to store the benchmark results.
- `rtol=0.05`: Tolerance for benchmark performance comparison.

# Returns
- `::Nothing`.
"""
function run_benchmarks(
    group::BenchmarkGroup, location::String, name::String; show=false, save=false, rtol=0.05
)
    println("Running benchmarks for group $(name)...")
    if !show && !save
        printstyled(
            "Warning: Both `show` and `save` are set to `false`. " *
            "Nothing will be done.\n";
            color=:red,
        )

        return nothing
    elseif !save
        input = ""
        first = true
        cases = ("y", "n", "c")
        while !(input in cases)
            if !first
                println("Invalid input. Valid options are `y`, `n` or `c`.")
            end
            printstyled(
                "Warning: `save` is set to `false`. No benchmarks will be stored. " *
                "Do you wish to proceed? (y)es, (n)o, (c)hange save to `true`\n";
                color=:yellow,
            )
            input = readline()
            first = false
        end

        if input == "c"
            save = true
        elseif input == "n"
            println("Quitting...")

            return nothing
        end
    end

    if location != "."
        printstyled(
            "Warning: You are not running the benchmark(s) from within the benchmark " *
            "directory. Aborting.\n";
            color=:red,
        )

        return nothing
    end

    hostname, date, commit_hash, julia_version = get_metadata()
    dataframe = _run_benchmarks!(
        DataFrame(),
        group,
        String[],
        hostname,
        date,
        commit_hash,
        julia_version;
        show=show,
        save=save,
    )

    if save
        file_name = "host-$(hostname)-group-$(name).csv"
        save_results(dataframe, location * "/" * file_name; rtol=rtol)
    end

    return nothing
end

"""
    get_metadata()

Returns useful metadata for CSV storage.

# Returns
- `hostname<:AbstractString`: The hostname of the device used to run the code.
- `date<:AbstractString`: The current date in the format `YYYY-MM-DD`.
- `commit_hash<:AbstractString`: The hash of latest commit in the current branch.
- `julia_version<:AbstractString`: The current julia version.
"""
function get_metadata()
    date = string(readchomp(`date +%Y-%m-%d`))
    commit_hash = string(readchomp(`git rev-parse --short HEAD`))
    hostname_shasum = split(readchomp(`sh -c "hostname | sha256sum"`))[1]
    julia_version = string(VERSION)
    # Mapping of SHA-256 hashes to hostnames
    hostname_map = Dict(
        "56960c953864e4d3e91ebb242bb87fd03002c2c6e3a62c8da21c719226bd1860" => "DiogoCabanas",
        "2b14f204f7c65a42a64568df8dc0179305ff91886e023977afef7561811ea920" => "JoeyDekker",
        # Add more mappings here as needed
    )

    if haskey(hostname_map, hostname_shasum)
        hostname = hostname_map[hostname_shasum]
    else
        throw(
            ArgumentError(
                "Hostname sha256sum not recognized. " *
                "Run `hostname | sha256sum` in your terminal to get the sha256sum and " *
                "add it to the `hostname_map` dictionary with the desired hostname.",
            ),
        )
    end

    return hostname, date, commit_hash, julia_version
end

"""
    _run_benchmarks(
        dataframe,
        group::BenchmarkGroup,
        path::Vector{String},
        hostname::String,
        date::String,
        commit_hash::String,
        julia_version::String;
        show::Bool,
        save::Bool,
    )

Recursive method that runs `@benchmarkable` objects inside `group`. See
[`run_benchmarks`](@ref) for more details.

# Arguments
- `dataframe`: The dataframe storing the benchmarks.
- `group::BenchmarkGroup`: The benchmark group being used.
- `path::Vector{String}`: The nested folder names.
- `hostname<:AbstractString`: The hostname of the device used to run the code.
- `date<:AbstractString`: The current date in the format `YYYY-MM-DD`.
- `commit_hash<:AbstractString`: The hash of latest commit in the current branch.
- `julia_version<:AbstractString`: The current julia version.
- `show=false`: Flag to display benchmark results.
- `save=false`: Flag to store the benchmark results.

# Returns
- `dataframe`: Updated dataframe at current nest level.
"""
function _run_benchmarks!(
    dataframe,
    group::BenchmarkGroup,
    path::Vector{String},
    hostname::String,
    date::String,
    commit_hash::String,
    julia_version::String;
    show::Bool,
    save::Bool,
)
    for (name, val) in group
        full_path = push!(copy(path), string(name))
        indent_level = length(full_path) - 1
        indent = "  "^indent_level
        if val isa BenchmarkGroup
            println(indent * "Entering group: $(join(full_path, "/"))")
            _run_benchmarks!(
                dataframe,
                val,
                full_path,
                hostname,
                date,
                commit_hash,
                julia_version;
                show=show,
                save=save,
            )
        else
            println(indent * "Benchmark: $(join(full_path, "/"))")
            #= The tuning should be uncommented if the benchmarking parameters are not set.
            print(indent * "Tuning...")
            tune!(val)
            print("Done! ") =#
            print(indent * "Running...")
            result = run(val)
            print("Done! ")
            if show
                println()
                io = IOBuffer()
                Base.show(
                    IOContext(io, :compact => false, :color => true), "text/plain", result
                )
                output = String(take!(io))
                for line in split(output, '\n')
                    println(indent * line)
                end
            end

            if save
                if show
                    print(indent * "Storing...")
                else
                    print("Storing...")
                end
                tags = [tag for tag in group.tags]
                push!(
                    dataframe,
                    (
                        benchmark=join(full_path, "/"),
                        commit_hash=commit_hash,
                        julia_version=julia_version,
                        date=date,
                        med_time=BenchmarkTools.prettytime(median(result).time),
                        min_time=BenchmarkTools.prettytime(minimum(result).time),
                        min_gc=BenchmarkTools.prettytime(minimum(result).gctime),
                        min_memory=BenchmarkTools.prettymemory(minimum(result).memory),
                        min_allocs=minimum(result).allocs,
                        tags=tags,
                    ),
                )
                println("Done!")
            end

            println()
        end
    end

    return dataframe
end

"""
    save_results(new_dataframe::DataFrame, file_name::String; rtol=0.05)

Stores the benchmark results in `new_dataframe` in the CSV file `file_name`. The tolerance
`rtol` is the cutoff determining if a benchmark is worse than before.

# Arguments
- `new_dataframe::DataFrame`: The dataframe containing the benchmark results.
- `file_name::String`: String containing the full file path.
- `rtol=0.05`: The cutoff tolerance for worse benchmarks.

# Returns
- `::Nothing`.
"""
function save_results(new_dataframe::DataFrame, file_name::String; rtol=0.05)
    println("Exporting to CSV...")
    if !endswith(file_name, ".csv")
        throw(ArgumentError("Invalid file name. Must end with `.csv`."))
    end

    data_prefix = "data/"
    if isfile(data_prefix * file_name)
        csv_file = CSV.File(data_prefix * file_name; header=true)
        header_syms = collect(keys(csv_file))
        types_dict = Dict(col => String for col in header_syms)
        old_dataframe = CSV.read(data_prefix * file_name, DataFrame; types=types_dict)
        duplicate_rows, new_rows = get_duplicate_and_new_rows(old_dataframe, new_dataframe)
        num_duplicate = length(duplicate_rows)
        if num_duplicate != 0
            println("Found $(num_duplicate) duplicate benchmarks.")
        end

        cap = 1.0 + rtol
        for (old_id, new_row) in duplicate_rows
            old_row = old_dataframe[old_id, :]
            println("Comparing benchmark $(new_row.benchmark).")
            old_time = get_number_in_string(old_row.min_time)
            new_time = get_number_in_string(new_row.min_time)
            if new_time < old_time
                printstyled(
                    "Minimum time improved! " *
                    "Old time=$(old_time), new time=$(new_time), " *
                    "ratio=$(new_time/old_time).\n";
                    color=:green,
                )
                for col in keys(types_dict)
                    old_dataframe[old_id, col] = new_row[col]
                end
            elseif new_time / old_time <= cap
                printstyled(
                    "Minimum time increased within $(rtol) tolerance. " *
                    "Old time=$(old_time), new time=$(new_time), " *
                    "ratio=$(new_time/old_time).\n";
                    color=:yellow,
                )
            else
                printstyled(
                    "Minimum time increased outside $(rtol) tolerance! " *
                    "Old time=$(old_time), new time=$(new_time), " *
                    "ratio=$(new_time/old_time).\n";
                    color=:red,
                )
            end
        end

        if !isempty(new_rows)
            append!(old_dataframe, new_rows)
        end

        CSV.write(data_prefix * file_name, old_dataframe)
    else
        mkpath(data_prefix * dirname(file_name))
        CSV.write(data_prefix * file_name, new_dataframe)
    end

    println("Done!")

    return nothing
end

"""
    get_duplicate_and_new_rows(old_dataframe, new_dataframe)

Returns the new and duplicated rows in `new_dataframe` when compared to `old_dataframe`.

# Arguments
- `old_dataframe`: The old dataframe.
- `new_dataframe`: The new dataframe.

# Returns
- `duplicate_rows::Vector{Tuple{Int, DataFrameRow{DataFrame, DataFrames.Index}}}`: The row
    id of a duplicated rows in `old_dataframe` and the equivalent row in `new_dataframe`.
- `new_rows::DataFrameRow{DataFrame, DataFrames.Index}`: The new rows.
"""
function get_duplicate_and_new_rows(old_dataframe, new_dataframe)
    duplicate_rows = Tuple{Int, DataFrameRow{DataFrame, DataFrames.Index}}[]
    new_rows = DataFrameRow{DataFrame, DataFrames.Index}[]

    for new_row in eachrow(new_dataframe)
        found_duplicate = false
        for (id, old_row) in pairs(IndexLinear(), eachrow(old_dataframe))
            if is_duplicate(old_row, new_row)
                push!(duplicate_rows, (id, new_row))
                found_duplicate = true
                break
            end
        end

        if !found_duplicate
            push!(new_rows, new_row)
        end
    end

    return duplicate_rows, new_rows
end

"""
    is_duplicate(old_row, new_row)

Determines if `old_row` and `new_row` refer to the same benchmark by comparing the name and
commit hash.

# Arguments
- `old_row`: The old row.
- `new_row`: The new row.

# Returns
- `::Bolean`: Whether the new row is a duplicate or not.
"""
function is_duplicate(old_row, new_row)
    if old_row.benchmark != new_row.benchmark
        return false
    end

    if old_row.commit_hash != new_row.commit_hash
        return false
    end

    return true
end

"""
    get_number_in_string(str::AbstractString)

Returns the number found in `str` as a `Float64`. This method assumes the string is given in
the pretty-print, number and unit, format used in `BenchmarkTools`, for example, `12.345
ms`.

# Arguments
- `str::AbstractString`: The pretty-print string.

# Returns
- `::Float64`: The number found in `str`.
"""
function get_number_in_string(str::AbstractString)
    m = match(r"\d+(?:\.\d+)?", str)
    if isnothing(m)
        throw(ArgumentError("Number not found in string \"$(str)\"."))
    end

    return parse(Float64, m.match)
end
