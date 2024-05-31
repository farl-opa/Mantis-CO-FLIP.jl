"""
Algorithms related with adaptive refinement.

"""

using Combinatorics
using Graphs

function proper_range(start::Int, finish::Int)
    if start>finish
        return range(start, finish, step=:(-1))
    end

    return range(start, finish)
end

function basis_functions_in_marked_elements(marked_elements::Vector{Int}, fem_space::T) where {T <: AbstractFiniteElementSpace{n} where {n}}
    basis_indices = Vector{Int}(undef, 0)

    for element_idx ∈ marked_elements
        append!(basis_indices, get_extraction(fem_space, element_idx)[2])
    end

    return unique!(basis_indices)
end

function check_problematic_intersection(twoscale_operator::T, basis_idx_1::Int, basis_idx_2::Int) where {T <: AbstractTwoScaleOperator}    
    n = get_n(twoscale_operator.coarse_space) 
    @assert(n==2, "Only implemented for 2 dimensions.")

    supp_1_per_dim = _get_support_per_dim(twoscale_operator.coarse_space, basis_idx_1)
    supp_2_per_dim = _get_support_per_dim(twoscale_operator.coarse_space, basis_idx_2)

    supp_intersection_per_dim = intersect.(supp_1_per_dim, supp_2_per_dim)

    for k ∈ 1:n
        if first(supp_2_per_dim[k]) - last(supp_1_per_dim[k]) > 1 || first(supp_1_per_dim[k]) - last(supp_2_per_dim[k]) > 1 
            return false
        end
    end

    p_l2 = get_polynomial_degree_per_dim(twoscale_operator.fine_space)

    length_flag = Vector{Bool}(undef, n)
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

        length_flag[k] = get_knot_vector_length(I_k) > p_l2[k]
    end

    return sum(length_flag) >= n-1 ? true : false
end

# Needs to be updated, the check is very weak
function check_shortest_chain(fem_space::F, basis_idx_1::Int, basis_idx_2::Int, marked_bsplines::Vector{Int}) where {F <: AbstractFiniteElementSpace{n} where {n}}
    n = get_n(fem_space)

    max_ind_basis = _get_dim_per_space(fem_space)
    basis_per_dim_1 = linear_to_ordered_index(basis_idx_1, max_ind_basis)
    basis_per_dim_2 = linear_to_ordered_index(basis_idx_2, max_ind_basis)

    diff_basis_per_dim = basis_per_dim_2 .- basis_per_dim_1
    any(diff_basis_per_dim .== 0) ? (return true) : nothing
    
    ordered_flag_per_dim = map(x -> x>0, diff_basis_per_dim)

    if sum(ordered_flag_per_dim) == 1
        for k ∈ 1:n
            if diff_basis_per_dim[k]<0
                diff_basis_per_dim[k] += get_polynomial_degree_per_dim(fem_space)[k]+1
                basis_per_dim_2[k] += get_polynomial_degree_per_dim(fem_space)[k]+1
            end
        end
    end

    diff_basis_per_dim = abs.(diff_basis_per_dim)

    all_in_between_basis_per_dim = [proper_range(basis_per_dim_1[k], basis_per_dim_2[k]) for k ∈ 1:n]
    local_grid_graph = Graphs.SimpleGraphs.grid(diff_basis_per_dim .+ 1)

    basis_to_remove = Int[]
    for (local_idx, ordered_index) ∈ enumerate(Iterators.product(all_in_between_basis_per_dim...))
        basis_idx = ordered_to_linear_index(ordered_index, max_ind_basis)
        if basis_idx ∉ marked_bsplines
            append!(basis_to_remove, local_idx)
        end
    end
    
    Graphs.SimpleGraphs.rem_vertices!(local_grid_graph, basis_to_remove, keep_order=true)
    
    Graphs.has_path(local_grid_graph, 1, Graphs.nv(local_grid_graph)) ? (return true) : (return false)
end

function check_problematic(fem_space::F, twoscale_operator::T, basis_idx_1::Int, basis_idx_2::Int, marked_bsplines::Vector{Int}) where {F <: AbstractFiniteElementSpace{n} where {n}, T <: AbstractTwoScaleOperator}
    if basis_idx_1 == 220 && basis_idx_2 == 347
        nothing
    end
    intersection_check = check_problematic_intersection(twoscale_operator, basis_idx_1, basis_idx_2)
    no_chain_check = !check_shortest_chain(fem_space, basis_idx_1, basis_idx_2, marked_bsplines)
    if intersection_check && no_chain_check
        nothing
    end

    return intersection_check && no_chain_check
end

function build_L_chain(fem_space::F, basis_idx_1::Int, basis_idx_2::Int) where {F <: AbstractFiniteElementSpace{n} where {n}}
    @assert(get_n(fem_space)==2, "Only implemented for 2 dimensions.")
    L_chain = Int[]
    max_ind_basis = _get_dim_per_space(fem_space)
    basis_per_dim_1 = linear_to_ordered_index(basis_idx_1, max_ind_basis)
    basis_per_dim_2 = linear_to_ordered_index(basis_idx_2, max_ind_basis)

    # Lower right L-chain
    for first_index ∈ proper_range(basis_per_dim_1[1],basis_per_dim_2[1])
        append!(L_chain, ordered_to_linear_index( (first_index, basis_per_dim_1[2]), max_ind_basis))
    end
    for second_index ∈ proper_range(basis_per_dim_1[2], basis_per_dim_2[2])
        append!(L_chain, ordered_to_linear_index( (basis_per_dim_2[1], second_index), max_ind_basis))
    end

    # Upper left L-chain
    #=
    for second_index ∈ basis_per_dim_1[2]:basis_per_dim_2[2]
        append!(L_chain, ordered_to_linear_index( (basis_per_dim_1[1], second_index), max_ind_basis))
    end
    for first_index ∈ basis_per_dim_1[1]:basis_per_dim_2[1]
        append!(L_chain, ordered_to_linear_index( (first_index, basis_per_dim_2[2]), max_ind_basis))
    end
    =#

    return L_chain 
end

function get_refinement_domain(fem_space::F, marked_elements::Vector{Int}, twoscale_operator::T) where {F <: AbstractFiniteElementSpace{n} where {n}, T<:AbstractTwoScaleOperator}
    
    marked_basis_functions = basis_functions_in_marked_elements(marked_elements, fem_space)

    pair_combinations = Combinatorics.combinations(marked_basis_functions, 2) # unordered pairs of marked basis functions
    el_pair_combinations = Combinatorics.combinations(marked_elements, 2)
    
    problematic_count = 1
    checked_pairs = Vector{Int}[]

    while problematic_count>0
        problematic_count = 0
        for (basis1, basis2) ∈ pair_combinations
            if check_problematic(fem_space, twoscale_operator, basis1, basis2, marked_basis_functions)
                append!(marked_basis_functions, build_L_chain(fem_space, basis1, basis2))
                problematic_count += 1
            end
        end
        append!(checked_pairs, collect(pair_combinations))
        pair_combinations = setdiff(collect(Combinatorics.combinations(marked_basis_functions, 2)), checked_pairs) # unordered pairs of marked basis functions
    end

    refinement_domain = Int[]

    for basis_idx ∈ marked_basis_functions
        append!(refinement_domain, get_support(fem_space, basis_idx))
    end

    return unique!(refinement_domain)
end

