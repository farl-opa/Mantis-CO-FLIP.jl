function build_two_scale_operator(
    coarse_space::GTBSplineSpace{num_patches, T},
    fine_space::GTBSplineSpace{num_patches, T},
    nsubdivisions::NTuple{num_patches, NTuple{1, Int}},
) where {num_patches, T}
    # Build the two-scale operators for the individual function spaces that form the coarse
    # unstructured space.
    discontinuous_two_scale_ops = ntuple(
        i -> build_two_scale_operator(
            get_constituent_spaces(coarse_space)[i], nsubdivisions[i]
        ),
        num_patches,
    )

    ###
    ### PART 1: Build the global subdivision matrix for the coarse and fine spaces
    ###

    # Build the global extraction operators for the coarse and fine spaces
    coarse_extraction_mat = assemble_global_extraction_matrix(coarse_space)
    fine_extraction_mat = assemble_global_extraction_matrix(fine_space)

    # Next, concatenate the two_scale_operator subdivision matrices in a block diagonal
    # format
    discontinuous_subdivision_mat = SparseArrays.blockdiag(
        [discontinuous_two_scale_ops[i][1].global_subdiv_matrix for i in 1:num_patches]...
    )

    # Finally, compute the two-scale matrix by solving a least-squares problem
    global_subdiv_matrix = SparseArrays.sparse(
        fine_extraction_mat \ Array(discontinuous_subdivision_mat * coarse_extraction_mat)
    )
    SparseArrays.fkeep!((i, j, x) -> abs(x) > 1e-14, global_subdiv_matrix)

    ###
    ### PART 2: Build the coarse-fine element relationships
    ###
    coarse_to_fine_elements = Vector{Vector{Int}}(undef, get_num_elements(coarse_space))
    num_elements_per_patch = get_num_elements_per_patch(coarse_space)
    # loop over coarse elements, and find global fine element ids that are contained in each
    # coarse element
    for patch_id in 1:num_patches
        for loc_elem_id in 1:num_elements_per_patch[patch_id]
            global_coarse_el_id = get_global_element_id(coarse_space, patch_id, loc_elem_id)
            local_coarse_to_fine_elements = discontinuous_two_scale_ops[patch_id][1].coarse_to_fine_elements[loc_elem_id]
            coarse_to_fine_elements[global_coarse_el_id] = [
                get_global_element_id(fine_space, patch_id, local_coarse_to_fine_elements[k])
                for k in eachindex(local_coarse_to_fine_elements)
            ]
        end
    end

    fine_to_coarse_elements = Vector{Int}(undef, get_num_elements(fine_space))
    num_elements_per_patch = get_num_elements_per_patch(fine_space)
    # loop over fine elements, and find the global coarse element id that contains each
    # fine element
    for patch_id in 1:num_patches
        for loc_elem_id in 1:num_elements_per_patch[patch_id]
            global_fine_el_id = get_global_element_id(fine_space, patch_id, loc_elem_id)
            fine_to_coarse_elements[global_fine_el_id] = get_global_element_id(
                coarse_space,
                patch_id,
                discontinuous_two_scale_ops[patch_id][1].fine_to_coarse_elements[loc_elem_id],
            )
        end
    end

    ###
    ### PART 3: Build the two-scale operator and return
    ###
    return TwoScaleOperator(
        coarse_space,
        fine_space,
        global_subdiv_matrix,
        coarse_to_fine_elements,
        fine_to_coarse_elements,
    ),
    fine_space
end
