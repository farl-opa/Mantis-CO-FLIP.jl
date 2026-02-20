module ParticleSorterModule

using Mantis
using Random
using LinearAlgebra
import ..HodgeProjectorModule: HodgeProjector

struct Particles
    position::Vector{Tuple{Float64, Float64}}
    impulse::Vector{Tuple{Float64, Float64}}
    Ψ::Vector{Matrix{Float64}}
    particles_in_elements::Vector{Vector{Int}}
    canonical_positions::Vector{Tuple{Float64, Float64}}

    function Particles(position::Vector{Tuple{Float64, Float64}}, impulse::Vector{Tuple{Float64, Float64}}, Ψ::Vector{Matrix{Float64}}, hodge_projector::HodgeProjector)
        particles_in_elements, canonical_positions = particle_sorter(position, hodge_projector)
        return new(position, impulse, Ψ, particles_in_elements, canonical_positions)
    end
end

function update_particles!(new_position::Vector{Tuple{Float64, Float64}}, new_impulse::Vector{Tuple{Float64, Float64}}, new_Ψ::Vector{Matrix{Float64}}, Particles::Particles, hodge_projector::HodgeProjector)
    Particles.position = new_position
    Particles.impulse = new_impulse
    Particles.Ψ = new_Ψ
    Particles.particles_in_elements, Particles.canonical_positions = particle_sorter(Particles.position, hodge_projector)
end


"""
    generate_particles(num_particles::Int, hodge::HodgeProjector)

Generates random particles within the domain defined by the HodgeProjector.
"""
function generate_particles(num_particles::Int, hodge::HodgeProjector)
    positions = Vector{Tuple{Float64, Float64}}()
    impulses = Vector{Tuple{Float64, Float64}}()
    psis = Vector{Matrix{Float64}}()
    
    box_size = hodge.box_size
    for _ in 1:num_particles
        x = rand() * box_size[1]
        y = rand() * box_size[2]
        push!(positions, (x, y))
        
        px = rand()
        py = rand()
        push!(impulses, (px, py))
        
        push!(psis, Matrix{Float64}(I, 2, 2))
    end
    
    return Particles(positions, impulses, psis, hodge)
end

"""
    particle_sorter(particles::Particles, hodge::HodgeProjector)

Assigns each particle to its corresponding element and normalized coordinates.
"""
function particle_sorter(particles::Particles, hodge::HodgeProjector)
    nel = hodge.nel
    num_elements = nel[1] * nel[2]
    particles_in_elements = [Vector{Int}() for _ in 1:num_elements]
    canonical_positions = Vector{Tuple{Float64, Float64}}()
    
    elem_width_x = hodge.box_size[1] / nel[1]
    elem_width_y = hodge.box_size[2] / nel[2]
    
    for (idx, (x, y)) in enumerate(particles.position)
        # Determine the element index based on coordinates (using floor and clamping)
        elem_i = clamp(Int(floor(x / elem_width_x)) + 1, 1, nel[1])
        elem_j = clamp(Int(floor(y / elem_width_y)) + 1, 1, nel[2])
        element_num = (elem_j - 1) * nel[1] + elem_i
        
        # Add particle index to the corresponding element
        push!(particles_in_elements[element_num], idx)
        
        # Calculate normalized coordinates within the element
        norm_x = (x - (elem_i - 1) * elem_width_x) / elem_width_x
        norm_y = (y - (elem_j - 1) * elem_width_y) / elem_width_y
        push!(canonical_positions, (norm_x, norm_y))
    end

    particles.particles_in_elements = particles_in_elements
    particles.canonical_positions = canonical_positions

end

export generate_particles, particle_sorter

end