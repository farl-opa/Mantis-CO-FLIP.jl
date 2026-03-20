using DelimitedFiles
using Plots

gr()

const STEP_FILE_REGEX = r"^step_(\d{4})\.txt$"

function parse_field(field)
    if field isa Symbol
        f = field
    else
        f = Symbol(lowercase(String(field)))
    end

    if f in (:u, :v, :speed)
        return f
    end

    throw(ArgumentError("Unsupported field '$field'. Use: u, v, or speed."))
end

function step_files(input_dir::AbstractString)
    if !isdir(input_dir)
        throw(ArgumentError("Input directory not found: $input_dir"))
    end

    files = String[]
    for name in readdir(input_dir)
        if occursin(STEP_FILE_REGEX, name)
            push!(files, joinpath(input_dir, name))
        end
    end

    if isempty(files)
        throw(ArgumentError("No files matching step_%04d.txt found in: $input_dir"))
    end

    sort!(files)
    return files
end

function read_particle_frame(path::AbstractString)
    raw = readdlm(path)
    data = ndims(raw) == 1 ? reshape(raw, 1, :) : raw

    ncols = size(data, 2)
    if ncols < 4
        throw(ArgumentError("Frame '$path' must have at least 4 columns: x y u v [speed]."))
    end

    x = Vector{Float64}(data[:, 1])
    y = Vector{Float64}(data[:, 2])
    u = Vector{Float64}(data[:, 3])
    v = Vector{Float64}(data[:, 4])

    speed = if ncols >= 5
        Vector{Float64}(data[:, 5])
    else
        sqrt.(u .^ 2 .+ v .^ 2)
    end

    return x, y, u, v, speed
end

function select_values(u, v, speed, field::Symbol)
    if field == :u
        return u, "u"
    elseif field == :v
        return v, "v"
    else
        return speed, "|u|"
    end
end

function field_colormap(field::Symbol)
    return field == :speed ? :viridis : :balance
end

function base_output_name(field::Symbol)
    return "particle_velocity_$(String(field))"
end

function compute_global_stats(files::Vector{String}, field::Symbol)
    xmin = Inf
    xmax = -Inf
    ymin = Inf
    ymax = -Inf
    cmin = Inf
    cmax = -Inf

    for file in files
        x, y, u, v, speed = read_particle_frame(file)
        vals, _ = select_values(u, v, speed, field)

        xmin = min(xmin, minimum(x))
        xmax = max(xmax, maximum(x))
        ymin = min(ymin, minimum(y))
        ymax = max(ymax, maximum(y))
        cmin = min(cmin, minimum(vals))
        cmax = max(cmax, maximum(vals))
    end

    if isapprox(cmin, cmax; atol=1e-14)
        cmin -= 1e-12
        cmax += 1e-12
    end

    return xmin, xmax, ymin, ymax, cmin, cmax
end

function step_index_from_name(path::AbstractString)
    name = splitpath(path)[end]
    m = match(STEP_FILE_REGEX, name)
    return m === nothing ? "?" : m.captures[1]
end

function render_particle_evolution(; input_dir::AbstractString="particle_output", field=:speed, output::Union{Nothing,String}=nothing, fps::Int=10, markersize::Real=2)
    field_sym = parse_field(field)
    files = step_files(input_dir)

    is_single_frame = length(files) == 1
    default_ext = is_single_frame ? ".png" : ".mp4"
    out_path = if output === nothing
        joinpath(input_dir, base_output_name(field_sym) * default_ext)
    else
        output
    end

    if is_single_frame && !endswith(lowercase(out_path), ".png")
        out_path = out_path * ".png"
    elseif !is_single_frame && !endswith(lowercase(out_path), ".mp4")
        out_path = out_path * ".mp4"
    end

    xmin, xmax, ymin, ymax, cmin, cmax = compute_global_stats(files, field_sym)
    cmap = field_colormap(field_sym)

    if is_single_frame
        x, y, u, v, speed = read_particle_frame(files[1])
        vals, label = select_values(u, v, speed, field_sym)
        step_str = step_index_from_name(files[1])

        plt = scatter(
            x,
            y,
            marker_z=vals,
            markersize=markersize,
            markerstrokewidth=0,
            c=cmap,
            clims=(cmin, cmax),
            legend=false,
            xlabel="x",
            ylabel="y",
            colorbar_title=label,
            aspect_ratio=:equal,
            xlims=(xmin, xmax),
            ylims=(ymin, ymax),
            title="Particles $(label) - step $(step_str)",
            dpi=150,
        )
        savefig(plt, out_path)
        println("Saved $(out_path)")
        return out_path
    end

    anim = @animate for file in files
        x, y, u, v, speed = read_particle_frame(file)
        vals, label = select_values(u, v, speed, field_sym)
        step_str = step_index_from_name(file)

        scatter(
            x,
            y,
            marker_z=vals,
            markersize=markersize,
            markerstrokewidth=0,
            c=cmap,
            clims=(cmin, cmax),
            legend=false,
            xlabel="x",
            ylabel="y",
            colorbar_title=label,
            aspect_ratio=:equal,
            xlims=(xmin, xmax),
            ylims=(ymin, ymax),
            title="Particles $(label) - step $(step_str)",
            dpi=150,
        )
    end

    mp4(anim, out_path, fps=fps)
    println("Saved $(out_path)")
    return out_path
end

function parse_cli_args(args::Vector{String})
    opts = Dict{String,String}(
        "input-dir" => "particle_output",
        "field" => "speed",
        "output" => "",
        "fps" => "10",
        "markersize" => "2",
    )

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg in ("--help", "-h")
            println("Usage: julia --project=. CO-FLIP/particle_evol.jl [options]")
            println("  --input-dir <dir>   Directory with step_%04d.txt files (default: particle_output)")
            println("  --field <u|v|speed> Field to render (default: speed)")
            println("  --output <path>     Output file path (auto extension by frame count if omitted)")
            println("  --fps <int>         FPS for mp4 when multiple frames (default: 10)")
            println("  --markersize <num>  Scatter marker size (default: 2)")
            return nothing
        end

        if startswith(arg, "--")
            key = replace(arg, "--" => "")
            if !haskey(opts, key)
                throw(ArgumentError("Unknown argument: $arg"))
            end
            if i == length(args)
                throw(ArgumentError("Missing value for argument: $arg"))
            end
            opts[key] = args[i + 1]
            i += 2
        else
            throw(ArgumentError("Unexpected argument: $arg"))
        end
    end

    output = isempty(opts["output"]) ? nothing : opts["output"]

    return (
        input_dir=opts["input-dir"],
        field=parse_field(opts["field"]),
        output=output,
        fps=parse(Int, opts["fps"]),
        markersize=parse(Float64, opts["markersize"]),
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    parsed = parse_cli_args(ARGS)
    if parsed !== nothing
        render_particle_evolution(
            input_dir=parsed.input_dir,
            field=parsed.field,
            output=parsed.output,
            fps=parsed.fps,
            markersize=parsed.markersize,
        )
    end
end
