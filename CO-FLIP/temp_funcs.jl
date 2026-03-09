function transfer_particles_to_grid_pcg(p::Particles, d::Domain)
    B = build_B_matrix(p, d)
    num_particles = length(p.x)
    num_dofs = size(B, 2)
    
    # Build RHS V_p
    V_p = Vector{Float64}(undef, 2 * num_particles)
    @inbounds for i in 1:num_particles
        w = sqrt(p.volume[i])
        V_p[2*i - 1] = p.my[i] * w
        V_p[2*i]     = -p.mx[i] * w
    end

    b_vec = Vector{Float64}(undef, num_dofs)
    mul!(b_vec, B', V_p)
    
    # Optimization: Pre-allocate Scratch Buffer
    temp_buffer = Vector{Float64}(undef, 2 * num_particles)

    function calc_Ax!(y, x)
        mul!(temp_buffer, B, x)
        mul!(y, B', temp_buffer)
        return y
    end

    A_map = LinearMap(calc_Ax!, num_dofs; ismutating=true, issymmetric=true, isposdef=true)

    # Jacobi Preconditioner
    diag_vals = zeros(Float64, num_dofs)
    vals = nonzeros(B)
    
    @inbounds for j in 1:num_dofs
        sum_sq = 0.0
        for k in nzrange(B, j)
            val = vals[k]
            sum_sq += val^2
        end
        diag_vals[j] = sum_sq + 1e-8
    end
    P = Diagonal(1.0 ./ diag_vals) 

    u_sol = cg(A_map, b_vec; Pl=P, reltol=1e-6, maxiter=5000)
    return u_sol
end

function solve_grid_velocity_from_B(B, p)
    num_particles = length(p.x)
    num_dofs = size(B, 2)
    
    V_p = Vector{Float64}(undef, 2 * num_particles)
    @inbounds for i in 1:num_particles
        w = sqrt(p.volume[i])
        V_p[2*i - 1] = p.my[i] * w
        V_p[2*i]     = -p.mx[i] * w
    end

    b_vec = Vector{Float64}(undef, num_dofs)
    mul!(b_vec, B', V_p)
    
    temp_buffer = Vector{Float64}(undef, 2 * num_particles)
    function calc_Ax!(y, x)
        mul!(temp_buffer, B, x)
        mul!(y, B', temp_buffer)
        return y
    end

    A_map = LinearMap(calc_Ax!, num_dofs; ismutating=true, issymmetric=true, isposdef=true)

    diag_vals = zeros(Float64, num_dofs)
    vals = nonzeros(B)
    @inbounds for j in 1:num_dofs
        sum_sq = 0.0
        for k in nzrange(B, j)
            val = vals[k]
            sum_sq += val^2
        end
        diag_vals[j] = sum_sq + 1e-6 # Regularization
    end
    P = Diagonal(1.0 ./ diag_vals) 

    return cg(A_map, b_vec; Pl=P, reltol=1e-6, maxiter=5000)
end