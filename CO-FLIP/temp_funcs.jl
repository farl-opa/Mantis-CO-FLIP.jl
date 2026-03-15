function generate_particles(num_particles::Int, domain::Domain, flow_type::Symbol=:vortex)
    Lx, Ly = domain.box_size
    nx, ny = domain.nel
    dx = Lx / nx
    dy = Ly / ny
    num_elements = nx * ny
    particles_per_element = fill(div(num_particles, num_elements), num_elements)
    for eid in 1:rem(num_particles, num_elements)
        particles_per_element[eid] += 1
    end
    
    x  = Vector{Float64}(undef, num_particles)
    y  = Vector{Float64}(undef, num_particles)
    mx = Vector{Float64}(undef, num_particles)
    my = Vector{Float64}(undef, num_particles)
    vol = ones(Float64, num_particles) # Volume is 1.0/density roughly
    
    # Pre-allocate arrays for sorting structures
    can_x = zeros(Float64, num_particles)
    can_y = zeros(Float64, num_particles)
    head = zeros(Int, num_elements)
    next = zeros(Int, num_particles)
    elem_ids = zeros(Int, num_particles)

    particle_idx = 1
    for ej in 1:ny
        for ei in 1:nx
            eid = (ej - 1) * nx + ei
            n_local = particles_per_element[eid]
            if n_local == 0
                continue
            end

            nx_local = ceil(Int, sqrt(n_local))
            ny_local = ceil(Int, n_local / nx_local)

            for local_idx in 0:(n_local - 1)
                ix = mod(local_idx, nx_local)
                iy = fld(local_idx, nx_local)

                ξ = (ix + rand()) / nx_local
                η = (iy + rand()) / ny_local

                px = (ei - 1 + ξ) * dx
                py = (ej - 1 + η) * dy

                x[particle_idx] = px
                y[particle_idx] = py
        
                # 2. Select Flow Type
                u, v = 0.0, 0.0
        
                if flow_type == :tg
                    u, v = flow_taylor_green(px, py, Lx, Ly)
                elseif flow_type == :vortex
                    u, v = flow_lamb_oseen(px, py, Lx, Ly)
                elseif flow_type == :gyre
                    u, v = flow_double_gyre(px, py, Lx, Ly)
                elseif flow_type == :decay
                    u, v = flow_decay(px, py, Lx, Ly)
                elseif flow_type == :convecting
                    u, v = flow_convecting_vortex(px, py, Lx, Ly)
                elseif flow_type == :merging
                    u, v = flow_merging_vortices(px, py, Lx, Ly)
                else
                    error("Unknown flow type: $flow_type")
                end
        
                # 3. Assign Impulse
                # In this metric (Cartesian), impulse (covector) components = velocity components
                mx[particle_idx] = u
                my[particle_idx] = v
                particle_idx += 1
            end
        end
    end

    if particle_idx != num_particles + 1
        error("Particle generator filled $(particle_idx - 1) particles, expected $num_particles")
    end
    
    # Estimate particle volume (Area / N) for integration weights
    # This helps the least-squares P2G solver converge better
    particle_vol = (Lx * Ly) / num_particles
    fill!(vol, particle_vol)
    
    return Particles(x, y, mx, my, vol, can_x, can_y, head, next, elem_ids)
end