"""
Algorithms related with L-chain creation.

"""

using Combinatorics
using Graphs

function proper_range(start::Int, finish::Int)
    if start>finish
        return range(start, finish, step=:(-1))
    end

    return range(start, finish, step=:1)
end

function ordered_range(start::Int, finish::Int)
    if start>finish
        return proper_range(finish, start)
    end

    return proper_range(start, finish)
end

function check_problematic_intersection(fem_space::F, twoscale_operator::T, basis_idx_1::Int, basis_idx_2::Int, corner_type::String) where {F <: AbstractFiniteElementSpace{n} where {n}, T <: AbstractTwoScaleOperator}    
    n = get_n(twoscale_operator.coarse_space) 
    @assert(n==2, "Only implemented for 2 dimensions.")

    max_ind_basis = _get_dim_per_space(fem_space)
    basis_per_dim_1 = linear_to_ordered_index(basis_idx_1, max_ind_basis)

    supp_1_per_dim = _get_support_per_dim(twoscale_operator.coarse_space, basis_idx_1)
    supp_2_per_dim = _get_support_per_dim(twoscale_operator.coarse_space, basis_idx_2)

    supp_intersection_per_dim = StepRange.(intersect.(supp_1_per_dim, supp_2_per_dim))

    el_supp_intersect_per_dim = Vector{StepRange{Int, Int}}(undef, n)
    
    if corner_type=="UR"
        el_supp_intersect_per_dim[1] = proper_range(
            minimum(supp_2_per_dim[1]),
            maximum(supp_1_per_dim[1])
        )
        el_supp_intersect_per_dim[2] = proper_range(
            minimum(supp_2_per_dim[2]),
            maximum(supp_1_per_dim[2])
        )
    elseif corner_type=="UL"
        el_supp_intersect_per_dim[1] = proper_range(
            minimum(supp_1_per_dim[1]),
            maximum(supp_2_per_dim[1])
        )
        el_supp_intersect_per_dim[2] = proper_range(
            minimum(supp_2_per_dim[2]),
            maximum(supp_1_per_dim[2])
        )
    elseif corner_type=="LL"
        el_supp_intersect_per_dim[1] = proper_range(
            minimum(supp_1_per_dim[1]),
            maximum(supp_2_per_dim[1])
        )
        el_supp_intersect_per_dim[2] = proper_range(
            minimum(supp_1_per_dim[2]),
            maximum(supp_2_per_dim[2])
        )
    elseif corner_type=="LR"
        el_supp_intersect_per_dim[1] = proper_range(
            minimum(supp_2_per_dim[1]),
            maximum(supp_1_per_dim[1])
        )
        el_supp_intersect_per_dim[2] = proper_range(
            minimum(supp_1_per_dim[2]),
            maximum(supp_2_per_dim[2])
        )
    end

    for k ∈ 1:n
        if first(supp_2_per_dim[k]) - last(supp_1_per_dim[k]) > 1 || first(supp_1_per_dim[k]) - last(supp_2_per_dim[k]) > 1 
            return false
        end
    end

    p_l2 = get_polynomial_degree_per_dim(twoscale_operator.fine_space)

    length_flag = Vector{Bool}(undef, n)
    #coarse_length_flag = Vector{Bool}(undef, n)

    for k ∈ 1:n
        if k ==1
            coarse_space = twoscale_operator.coarse_space.function_space_1
            fine_space = twoscale_operator.fine_space.function_space_1
            ts = twoscale_operator.twoscale_operator_1
        elseif k==2 
            coarse_space = twoscale_operator.coarse_space.function_space_2
            fine_space = twoscale_operator.fine_space.function_space_2
            ts = twoscale_operator.twoscale_operator_2
        end

        I_k = get_contained_knot_vector(supp_intersection_per_dim[k], ts, fine_space)
        #I_k_coarse = get_contained_knot_vector(el_supp_intersect_per_dim[k], ts.coarse_space)
        
        length_flag[k] = get_knot_vector_length(I_k) > p_l2[k]
        #coarse_length_flag[k] = get_knot_vector_length(I_k_coarse) >= get_knot_vector_length(get_local_knot_vector(coarse_space, basis_per_dim_1[k]))
    end

    return sum(length_flag) >= n-1 && !any(coarse_length_flag) ? true : false
end

function get_corner_basis_ids(fe_space::AbstractFiniteElementSpace{n}, basis_ids::Vector{Int}) where {n}
    max_ind_basis = _get_dim_per_space(fe_space)
    lower_left_bspline = minimum(basis_ids)
    upper_right_bspline = maximum(basis_ids)

    lower_left_bspline_per_dim = linear_to_ordered_index(lower_left_bspline, max_ind_basis)
    upper_right_bspline_per_dim = linear_to_ordered_index(upper_right_bspline, max_ind_basis)

    lower_right_bspline = ordered_to_linear_index((upper_right_bspline_per_dim[1], lower_left_bspline_per_dim[2]), max_ind_basis)
    upper_left_bspline = ordered_to_linear_index((lower_left_bspline_per_dim[1], upper_right_bspline_per_dim[2]), max_ind_basis)

    return upper_right_bspline, lower_right_bspline, lower_left_bspline, upper_left_bspline
end

function build_L_chain(hspace::HierarchicalFiniteElementSpace{n}, level::Int, basis_pair::Tuple{Int, Int}, chain_type="LR") where {n}
    L_chain = Int[]
    max_ind_basis = _get_dim_per_space(get_space(hspace, level))
    basis_per_dim_1 = linear_to_ordered_index(basis_pair[1], max_ind_basis)
    basis_per_dim_2 = linear_to_ordered_index(basis_pair[2], max_ind_basis)

    # Lower right L-chain
    if chain_type=="LR"
        for first_index ∈ proper_range(basis_per_dim_1[1],basis_per_dim_2[1])
            append!(L_chain, ordered_to_linear_index( (first_index, basis_per_dim_1[2]), max_ind_basis))
        end
        for second_index ∈ proper_range(basis_per_dim_1[2], basis_per_dim_2[2])
            append!(L_chain, ordered_to_linear_index( (basis_per_dim_2[1], second_index), max_ind_basis))
        end
    # Upper left L-chain
    elseif chain_type=="UL" 
        for second_index ∈ proper_range(basis_per_dim_1[2], basis_per_dim_2[2])
            append!(L_chain, ordered_to_linear_index( (basis_per_dim_1[1], second_index), max_ind_basis))
        end
        for first_index ∈ proper_range(basis_per_dim_1[1], basis_per_dim_2[1])
            append!(L_chain, ordered_to_linear_index( (first_index, basis_per_dim_2[2]), max_ind_basis))
        end
    else
        throw(ArgumentError("Invalid chain type. Supported types are \"LR\" or \"UL\" and \"$chain_type\" was given."))
    end

    return L_chain[2:end-1] 
end

function initiate_basis_to_check(fe_space::AbstractFiniteElementSpace{n}, marked_elements::Vector{Int}) where {n}
    all_basis_ids = Int[]
    curr_basis_ids = Int[]

    all_corner_basis_ids = Int[]
    all_interior_basis_ids = Int[]

    for el ∈ marked_elements
        _, curr_basis_ids = get_extraction(fe_space, el) 
        curr_corner_basis_ids = get_corner_basis_ids(fe_space, curr_basis_ids)
        curr_interior_basis_ids = setdiff(curr_basis_ids, curr_corner_basis_ids)

        append!(all_corner_basis_ids, curr_corner_basis_ids)
        append!(all_interior_basis_ids, curr_interior_basis_ids)
        append!(all_basis_ids, curr_basis_ids)
    end

    return setdiff(all_corner_basis_ids, all_interior_basis_ids), unique!(all_basis_ids)
end

function check_nl_intersection(hspace::HierarchicalFiniteElementSpace{n}, level::Int, basis_pair, new_operator) where {n}
    
    basis_supp_per_dim = [_get_support_per_dim(hspace.spaces[level], basis_pair[k]) for k ∈ 1:2]
    basis_supp_intersection_per_dim = StepRange.(intersect.(basis_supp_per_dim...))

    for k ∈ 1:n
        if first(basis_supp_per_dim[2][k]) - last(basis_supp_per_dim[1][k]) > 1 || first(basis_supp_per_dim[1][k]) - last(basis_supp_per_dim[2][k]) > 1 
            return false
        end
    end

    if level == get_num_levels(hspace)
        operator = new_operator
    else
        operator = hspace.two_scale_operators[level]
    end

    p_fine = get_polynomial_degree_per_dim(operator.fine_space)

    length_flag = Vector{Bool}(undef, n)

    for k ∈ 1:n
        if k ==1
            coarse_space = operator.coarse_space.function_space_1
            fine_space = operator.fine_space.function_space_1
            ts = operator.twoscale_operator_1
        elseif k==2 
            coarse_space = operator.coarse_space.function_space_2
            fine_space = operator.fine_space.function_space_2
            ts = operator.twoscale_operator_2
        end

        I_k = get_contained_knot_vector(basis_supp_intersection_per_dim[k], ts, fine_space)
        
        length_flag[k] = get_knot_vector_length(I_k) > p_fine[k]
    end
    
    return sum(length_flag) >= n-1 ? true : false
end

function _get_basis_pair_graph(max_id_basis, basis_per_dim, diff_basis_per_dim, inactive_basis, n::Int)
    basis_between_pair_per_dim = [ordered_range(basis_per_dim[1][k], basis_per_dim[2][k]) for k ∈ 1:n]

    basis_pair_graph = Graphs.SimpleGraphs.grid(abs.(diff_basis_per_dim) .+ 1)
    
    basis_to_remove = Int[]
    for (local_idx, ordered_id) ∈ enumerate(Iterators.product(basis_between_pair_per_dim...))
        basis_id = ordered_to_linear_index(ordered_id, max_id_basis)
        if basis_id ∉ inactive_basis
            append!(basis_to_remove, local_idx)
        end
    end
    
    Graphs.SimpleGraphs.rem_vertices!(basis_pair_graph, basis_to_remove, keep_order=true)

    return basis_pair_graph
end

function check_shortest_chain(hspace::HierarchicalFiniteElementSpace{n}, level::Int, basis_pair, inactive_basis) where {n}
    max_id_basis = _get_dim_per_space(get_space(hspace, level))
    basis_per_dim = [linear_to_ordered_index(basis_pair[k], max_id_basis) for k ∈ 1:2]
    diff_basis_per_dim = -(basis_per_dim...)

    any(diff_basis_per_dim .== 0) ? (return true) : nothing # check for trivial shortest chain
    
    basis_pair_graph = _get_basis_pair_graph(max_id_basis, basis_per_dim, diff_basis_per_dim, inactive_basis, n)

    Graphs.has_path(basis_pair_graph, 1, Graphs.nv(basis_pair_graph)) ? shortest_chain = true : shortest_chain = false

    return shortest_chain
end

function check_problematic_pair(hspace::HierarchicalFiniteElementSpace{n}, level::Int, basis_pair, inactive_basis, new_operator) where {n} 
    nl_intersection = check_nl_intersection(hspace, level, basis_pair, new_operator)

    if !nl_intersection
        return false
    end

    shortest_chain = check_shortest_chain(hspace, level, basis_pair, inactive_basis)

    problematic_pair = nl_intersection && !shortest_chain

    return problematic_pair
end

function build_L_chain(hspace::HierarchicalFiniteElementSpace{n}, level::Int, basis_pair, chain_type="LR") where {n}
    L_chain = Int[]
    max_id_basis = _get_dim_per_space(get_space(hspace, level))
    basis_per_dim = [linear_to_ordered_index(basis_pair[k], max_id_basis) for k ∈ 1:2]

    # Lower right L-chain
    if chain_type=="LR"
        for first_index ∈ proper_range(basis_per_dim[1][1],basis_per_dim[2][1])
            append!(L_chain, ordered_to_linear_index( (first_index, basis_per_dim[1][2]), max_id_basis))
        end
        for second_index ∈ proper_range(basis_per_dim[1][2], basis_per_dim[2][2])[2:end]
            append!(L_chain, ordered_to_linear_index( (basis_per_dim[2][1], second_index), max_id_basis))
        end
        corner_basis = ordered_to_linear_index( (basis_per_dim[2][1], basis_per_dim[1][2]), max_id_basis)
    # Upper left L-chain
    elseif chain_type=="UL" 
        for second_index ∈ proper_range(basis_per_dim[1][2], basis_per_dim[2][2])
            append!(L_chain, ordered_to_linear_index( (basis_per_dim[1][1], second_index), max_id_basis))
        end
        for first_index ∈ proper_range(basis_per_dim[1][1], basis_per_dim[2][1])
            append!(L_chain, ordered_to_linear_index( (first_index, basis_per_dim[2][2]), max_id_basis))
        end
        corner_basis = ordered_to_linear_index( (basis_per_dim[1][1], basis_per_dim[2][2]), max_id_basis)
    else
        throw(ArgumentError("Invalid chain type. Supported types are \"LR\" or \"UL\" and \"$chain_type\" was given."))
    end

    return L_chain[2:end-1], corner_basis
end

function get_level_marked_basis(hspace::HierarchicalFiniteElementSpace{n}, level::Int, marked_elements_per_level::Vector{Vector{Int}}, new_operator) where {n} 
    checked_pairs = Vector{Int}[]
    new_basis_to_check, new_inactive_basis = initiate_basis_to_check(get_space(hspace, level), marked_elements_per_level[level])
    previous_inactive_basis = get_basis_contained_in_next_level(hspace, level)
    if length(marked_elements_per_level[level]) == 0
        return previous_inactive_basis
    end 
    basis_to_check = union(previous_inactive_basis, new_basis_to_check)
    inactive_basis = union(previous_inactive_basis, new_inactive_basis)
    #=
    combinations_to_check = Combinatorics.combinations(basis_to_check, 2)
    
    check_count = 1
    while check_count > 0 # Add L-chains until there are no more problematic intersections
        check_count = 0
        
        for basis_pair ∈ combinations_to_check # Loop over unchecked pairs of B-splines
            problematic_pair = check_problematic_pair(hspace, level, basis_pair, inactive_basis, new_operator)
            
            if problematic_pair
                L_chain, corner_basis = build_L_chain(hspace, level, basis_pair) # Choose chain type here default is "LR" 
                append!(basis_to_check, corner_basis)
                append!(inactive_basis, L_chain)
                check_count += 1
            end
        end

        append!(checked_pairs, combinations_to_check)
        combinations_to_check = setdiff(Combinatorics.combinations(basis_to_check, 2), checked_pairs)
    end
    =#

    return inactive_basis
end

function get_refinement_domain(hspace::HierarchicalFiniteElementSpace{n}, marked_elements::Vector{Int}, new_two_scale_operator::AbstractTwoScaleOperator) where {n}
    n == 2 || throw(ArgumentError("Only implemented for n=2 dimensions."))

    L = get_num_levels(hspace)
    marked_elements_per_level = get_marked_elements_per_level(hspace, marked_elements)
    if step == 3
        println(marked_elements_per_level)
    end

    refinement_domains = Vector{Vector{Int}}(undef, L+1)
    refinement_domains[1] = Int[]
    for level ∈ 1:L
        level_marked_basis = get_level_marked_basis(hspace, level, marked_elements_per_level, new_two_scale_operator)

        if level_marked_basis == Int[]
            refinement_domains[level+1] = Int[]
            continue
        end
        coarse_els = get_marked_basis_support(hspace, level, level_marked_basis)
        if level == L
            refinement_domains[level+1] = get_finer_elements(new_two_scale_operator, coarse_els)
        else
            refinement_domains[level+1] = get_finer_elements(hspace.two_scale_operators[level], coarse_els)
        end
    end 

    if refinement_domains[end] == Int[]
        deleteat!(refinement_domains, length(refinement_domains))
    end

    return refinement_domains
end

