# Example: Particle Sorting and Visualization
using Mantis
using CairoMakie
using Colors
include(joinpath(@__DIR__, "HodgeProjector.jl"))
include(joinpath(@__DIR__, "ParticleSorter.jl"))

using .HodgeProjectorModule: HodgeProjector
using .ParticleSorterModule: generate_particles, particle_sorter, Particles

# -------------------------------------------------------------
#                   Setup Parameters
# -------------------------------------------------------------

# Number of elements in each direction
nel = (4, 4)

# Polynomial degree in each direction
p = (2, 2)

# Form degree (k-form)
k = (0, 1)

# Number of particles to generate
num_particles = 100

# -------------------------------------------------------------
#                   Create Hodge Projector
# -------------------------------------------------------------

println("Creating Hodge Projector...")
hodge = HodgeProjector(nel, p, k)
println("Hodge Projector created successfully!")

# -------------------------------------------------------------
#                   Generate Particles
# -------------------------------------------------------------

println("\nGenerating $num_particles particles...")
particles = generate_particles(num_particles, hodge)
println("Particles generated successfully!")

# Extract positions and impulses from the particles struct
positions = particles.position
impulses = particles.impulse

# Get sorted particle assignments
particles_in_elements = particles.particles_in_elements
canonical_positions = particles.canonical_positions

# -------------------------------------------------------------
#                   Visualize Results
# -------------------------------------------------------------

println("\nCreating visualization...")

# Create a figure
fig = Figure(size = (800, 800))
ax = Axis(fig[1, 1], 
    xlabel = "x", 
    ylabel = "y",
    title = "Particle Sorting by Element",
    aspect = DataAspect())

# Domain size
domain_x = hodge.box_size[1]
domain_y = hodge.box_size[2]

# Element size
elem_x = domain_x / nel[1]
elem_y = domain_y / nel[2]

# Draw element boundaries
for i in 0:nel[1]
    x_pos = i * elem_x
    lines!(ax, [x_pos, x_pos], [0, domain_y], color = :black, linewidth = 2)
end

for j in 0:nel[2]
    y_pos = j * elem_y
    lines!(ax, [0, domain_x], [y_pos, y_pos], color = :black, linewidth = 2)
end

# Create a colormap with enough distinct colors for all elements
total_elements = nel[1] * nel[2]
colors = distinguishable_colors(total_elements, [RGB(1,1,1), RGB(0,0,0)], 
                                dropseed=true)

# Plot particles colored by their element
for (elem_idx, particle_indices) in enumerate(particles_in_elements)
    if !isempty(particle_indices)
        # Get color for this element
        particle_color = colors[elem_idx]
        
        for pid in particle_indices
            # Get particle position and impulse
            pos_x, pos_y = positions[pid]
            imp_x, imp_y = impulses[pid]
            
            # Normalize impulse for arrow visualization (scale down if needed)
            arrow_scale = 0.15  # Scale factor for arrow length
            norm_imp = sqrt(imp_x^2 + imp_y^2)
            if norm_imp > 0
                arrow_imp_x = (imp_x / norm_imp) * arrow_scale
                arrow_imp_y = (imp_y / norm_imp) * arrow_scale
            else
                arrow_imp_x = 0
                arrow_imp_y = 0
            end
            
            # Plot the particle
            scatter!(ax, [pos_x], [pos_y], 
                     color = particle_color, 
                     markersize = 10)
            
            # Draw arrow indicating impulse direction
            arrows!(ax, [pos_x], [pos_y], [arrow_imp_x], [arrow_imp_y],
                    color = particle_color,
                    linewidth = 2,
                    arrowsize = 15)
        end
    end
end

# Add legend showing element numbering
elem_text = "Elements numbered row-wise:\n"
elem_text *= "Bottom-left = 1, Bottom-right = $(nel[1])\n"
elem_text *= "Top-left = $((nel[2]-1)*nel[1]+1), Top-right = $(total_elements)"

text!(ax, 0.02, 0.98, 
      text = elem_text,
      align = (:left, :top),
      fontsize = 10,
      space = :relative)

# Save the figure
output_path = joinpath(@__DIR__, "..", "examples", "data", "output", "particle_sorting.png")
mkpath(dirname(output_path))
save(output_path, fig)

println("Visualization saved to: $output_path")
println("\nDone!")

# Display the figure
display(fig)

# -------------------------------------------------------------
#                   Print Summary Statistics
# -------------------------------------------------------------

println("\n" * "="^60)
println("SUMMARY STATISTICS")
println("="^60)
println("Total particles: $num_particles")
println("Domain size: $(domain_x) x $(domain_y)")
println("Number of elements: $(nel[1]) x $(nel[2]) = $total_elements")
println("Element size: $(elem_x) x $(elem_y)")

# Count particles per element
particles_count = [length(pids) for pids in particles_in_elements]

println("\nParticles per element:")
for i in 1:nel[2]
    for j in 1:nel[1]
        elem_linear = (i - 1) * nel[1] + j
        count = particles_count[elem_linear]
        println("  Element ($j, $i) [#$elem_linear]: $count particles")
    end
end
println("="^60)
