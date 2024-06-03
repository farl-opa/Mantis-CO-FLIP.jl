"""
Algorithms related with adaptive refinement.

"""

using Combinatorics
using Graphs

function proper_range(start::Int, finish::Int)
    if start>finish
        return range(start, finish, step=:(-1))
    end

    return range(start, finish, step=:1)
end

function basis_functions_in_marked_elements(marked_elements::Vector{Int}, fem_space::T) where {T <: AbstractFiniteElementSpace{n} where {n}}
    basis_indices = Vector{Int}(undef, 0)

    for element_idx ∈ marked_elements
        append!(basis_indices, get_extraction(fem_space, element_idx)[2])
    end

    return unique!(basis_indices)
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
    coarse_length_flag = Vector{Bool}(undef, n)

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
        I_k_coarse = get_contained_knot_vector(el_supp_intersect_per_dim[k], ts.coarse_space)
        
        length_flag[k] = get_knot_vector_length(I_k) > p_l2[k]
        coarse_length_flag[k] = get_knot_vector_length(I_k_coarse) >= get_knot_vector_length(get_local_knot_vector(coarse_space, basis_per_dim_1[k]))
    end

    return sum(length_flag) >= n-1 && !any(coarse_length_flag) ? true : false
end

# Needs to be updated, the check is very weak
#=
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
=#

function get_relevant_bsplines(fem_space::F, el1::Int, el2::Int, corner_type::String) where {F <: AbstractFiniteElementSpace{n} where {n}}
    _, el1_bsplines = get_extraction(fem_space, el1)
    _, el2_bsplines = get_extraction(fem_space, el2)

    if corner_type=="UR"
        return maximum(el1_bsplines), minimum(el2_bsplines)
    elseif corner_type=="LL"
        return minimum(el1_bsplines), maximum(el2_bsplines)
    elseif corner_type=="UL"
        dim_per_space = _get_dim_per_space(fem_space)
        bsplines1_per_dim = linear_to_ordered_index.(el1_bsplines, (dim_per_space,))
        bsplines2_per_dim = linear_to_ordered_index.(el2_bsplines, (dim_per_space,)) 

        min_dim_1 = Inf
        max_dim_1 = -Inf
        for basis_indxs ∈ bsplines1_per_dim
            basis_indxs[1] < min_dim_1 ? min_dim_1 = basis_indxs[1] : nothing
            basis_indxs[2] > max_dim_1 ? max_dim_1 = basis_indxs[2] : nothing  
        end

        min_dim_2 = Inf
        max_dim_2 = -Inf
        for basis_indxs ∈ bsplines2_per_dim
            basis_indxs[2] < min_dim_2 ? min_dim_2 = basis_indxs[2] : nothing
            basis_indxs[1] > max_dim_2 ? max_dim_2 = basis_indxs[1] : nothing  
        end

        return ordered_to_linear_index((min_dim_1, max_dim_1), dim_per_space), ordered_to_linear_index((max_dim_2, min_dim_2), dim_per_space)
    elseif corner_type=="LR"
        dim_per_space = _get_dim_per_space(fem_space)
        bsplines1_per_dim = linear_to_ordered_index.(el1_bsplines, (dim_per_space,))
        bsplines2_per_dim = linear_to_ordered_index.(el2_bsplines, (dim_per_space,)) 

        min_dim_1 = Inf
        max_dim_1 = -Inf
        for basis_indxs ∈ bsplines1_per_dim
            basis_indxs[2] < min_dim_1 ? min_dim_1 = basis_indxs[2] : nothing
            basis_indxs[1] > max_dim_1 ? max_dim_1 = basis_indxs[1] : nothing  
        end

        min_dim_2 = Inf
        max_dim_2 = -Inf
        for basis_indxs ∈ bsplines2_per_dim
            basis_indxs[1] < min_dim_2 ? min_dim_2 = basis_indxs[1] : nothing
            basis_indxs[2] > max_dim_2 ? max_dim_2 = basis_indxs[2] : nothing  
        end

        return ordered_to_linear_index((max_dim_1, min_dim_1), dim_per_space), ordered_to_linear_index((min_dim_2, max_dim_2), dim_per_space)
    end

    throw(ArgumentError("Corner type not valid."))
end

function check_problematic_bsplines(fem_space::F, twoscale_operator::T, el1::Int, el2::Int, corner_type::String) where {F <: AbstractFiniteElementSpace{n} where {n}, T <: AbstractTwoScaleOperator}
    basis1, basis2 = get_relevant_bsplines(fem_space, el1, el2, corner_type)
    #marked_bsplines = basis_functions_in_marked_elements([el1, el2], fem_space)

    intersection_check = check_problematic_intersection(fem_space, twoscale_operator, basis1, basis2, corner_type)
    #no_chain_check = !check_shortest_chain(fem_space, basis1, basis2, marked_bsplines)

    return intersection_check #&& no_chain_check
end

function get_marked_elements_check_need(fem_space::F, el1::Int, el2::Int) where {F <: AbstractFiniteElementSpace{n} where {n}}
    n = get_n(fem_space)
    n==2 || throw(ArgumentError("Only implemented for 2 dimensions."))

    deg_per_dim = get_polynomial_degree_per_dim(fem_space)

    max_ind_els = _get_num_elements_per_space(fem_space)
    el1_per_dim = linear_to_ordered_index(el1, max_ind_els)
    el2_per_dim = linear_to_ordered_index(el2, max_ind_els)
    diff_per_dim = el2_per_dim .- el1_per_dim

    dim_1_gap = 2*deg_per_dim[1]+1
    dim_2_gap = 2*deg_per_dim[2]+1

    if (0 < diff_per_dim[1] <= dim_1_gap) && (0 < diff_per_dim[2] <= dim_2_gap)
        return true, "UR"
    elseif (0 < diff_per_dim[1] <= dim_1_gap) && (-dim_2_gap <= diff_per_dim[2] < 0)
        return true, "LR"
    elseif (-dim_1_gap <= diff_per_dim[1] < 0) && (0 < diff_per_dim[2] <= dim_2_gap)
        return true, "UL"
    elseif (-dim_1_gap <= diff_per_dim[1] < 0) && (-dim_2_gap <= diff_per_dim[2] < 0)
        return true, "LL"
    end

    return false, "N"
end

#=

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

=#

function get_fixer_element(fem_space::F, el1::Int, el2::Int) where {F <: AbstractFiniteElementSpace{n} where {n}}
    n = get_n(fem_space)
    @assert(n==2, "Only implemented for 2 dimensions.")
    max_ind_els = _get_num_elements_per_space(fem_space)
    el1_per_dim = linear_to_ordered_index(el1, max_ind_els)
    el2_per_dim = linear_to_ordered_index(el2, max_ind_els)

    location = Vector{Int}(undef, 2)
    for k ∈ 1:n
        location[k] = round(Int, (el2_per_dim[k] + el1_per_dim[k])/2)
    end

    return ordered_to_linear_index(location, max_ind_els)
end

function get_refinement_domain(fem_space::F, marked_elements::Vector{Int}, twoscale_operator::T) where {F <: AbstractFiniteElementSpace{n} where {n}, T<:AbstractTwoScaleOperator}
    el_pair_combinations = Combinatorics.combinations(marked_elements, 2)
    
    problematic_count = 1
    checked_pairs = Vector{Int}[]

    while problematic_count>0
        problematic_count = 0
        for (el1, el2) ∈ el_pair_combinations
            el_pair_check, corner_type = get_marked_elements_check_need(fem_space, el1, el2) 

            if el_pair_check && check_problematic_bsplines(fem_space, twoscale_operator, el1, el2, corner_type)
                fixer_element = get_fixer_element(fem_space, el1, el2)
                append!(marked_elements, fixer_element)
                append!(checked_pairs, [[el1, fixer_element], [el2, fixer_element]])
                problematic_count += 1
            end
        end
        append!(checked_pairs, el_pair_combinations)
        el_pair_combinations = setdiff(Combinatorics.combinations(marked_elements, 2), checked_pairs) # unordered pairs of marked basis functions
    end

    marked_basis_functions = basis_functions_in_marked_elements(marked_elements, fem_space)
    refinement_domain = Int[]

    for basis_idx ∈ marked_basis_functions
        append!(refinement_domain, get_support(fem_space, basis_idx))
    end

    return unique!(refinement_domain)
end

