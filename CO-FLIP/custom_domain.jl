function create_unclamped_1d_space(start_pt, box_len, n_elem, p, k)
    breakpoints = collect(LinRange(start_pt, start_pt + box_len, n_elem + 1))
    patch = Mantis.Mesh.Patch1D(breakpoints)
    regularity_vector = fill(k, n_elem + 1)
    section_space = FunctionSpaces.Bernstein(p)
    return FunctionSpaces.BSplineSpace(patch, section_space, regularity_vector, 0, 0)
end

function CreatePeriodicComplex(starting_points::NTuple{2, Float64}, box_sizes::NTuple{2, Float64}, num_elements::NTuple{2, Int}, p::NTuple{2, Int}, k::NTuple{2, Int})
    # Create UNCLAMPED 1D spaces
    spaces_x = create_unclamped_1d_space(starting_points[1], box_sizes[1], num_elements[1], p[1], k[1])
    spaces_y = create_unclamped_1d_space(starting_points[2], box_sizes[2], num_elements[2], p[2], k[2])
    unclamped_spaces = (spaces_x, spaces_y)

    # Wrap in GTBSplineSpace to impose Periodicity
    periodic_spaces_1d = ntuple(2) do i
        FunctionSpaces.GTBSplineSpace((unclamped_spaces[i],), [k[i]])
    end
    
    # Level 0 (0-forms / Potential)
    S0 = periodic_spaces_1d
    Fem0 = FunctionSpaces.TensorProductSpace(S0)
    
    # Level 1 (1-forms / Velocity)
    S1 = map(FunctionSpaces.get_derivative_space, S0)
    Fem1_x = FunctionSpaces.TensorProductSpace((S1[1], S0[2]))
    Fem1_y = FunctionSpaces.TensorProductSpace((S0[1], S1[2]))
    Fem1 = FunctionSpaces.DirectSumSpace((Fem1_x, Fem1_y))
    
    # Level 2 (2-forms / Vorticity)
    Fem2 = FunctionSpaces.TensorProductSpace(S1)
    
    geo = Mantis.Geometry.create_cartesian_box(starting_points, box_sizes, num_elements)
    
    R0 = Forms.FormSpace(0, geo, Fem0, "ω_0")
    R1 = Forms.FormSpace(1, geo, Fem1, "ω_1")
    R2 = Forms.FormSpace(2, geo, Fem2, "ω_2")
    
    return (R0, R1, R2)
end

# --- USE CUSTOM PERIODIC GENERATOR ---
#R = CreatePeriodicComplex(starting_point, box_size, nel, p, k)