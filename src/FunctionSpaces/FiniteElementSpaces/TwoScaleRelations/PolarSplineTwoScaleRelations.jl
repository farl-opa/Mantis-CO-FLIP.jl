function build_two_scale_operator(
    parent_space::PolarSplineSpace{num_components, T},
    child_space::PolarSplineSpace{num_components, T},
    num_subdivisions::NTuple{1, NTuple{2, Int}},
) where {num_components, T}
    # Build the two-scale operators for the individual function spaces that form the parent
    # unstructured space.
    discontinuous_two_scale_op, _ = build_two_scale_operator(
        DirectSumSpace(get_patch_spaces(parent_space)), num_subdivisions[1]
    )

    ###
    ### PART 1: Build the global subdivision matrix for the parent and child spaces
    ###

    # Build the global extraction operators for the parent and child spaces
    parent_extraction_mat = assemble_global_extraction_matrix(parent_space)
    child_extraction_mat = assemble_global_extraction_matrix(child_space)

    # Next, concatenate the two_scale_operator subdivision matrices in a block diagonal
    # format
    discontinuous_subdivision_mat = get_global_subdiv_matrix(discontinuous_two_scale_op)

    # Finally, compute the two-scale matrix by solving a least-squares problem
    global_subdiv_matrix = SparseArrays.sparse(
        child_extraction_mat \ Array(discontinuous_subdivision_mat * parent_extraction_mat)
    )
    SparseArrays.fkeep!((i, j, x) -> abs(x) > 1e-14, global_subdiv_matrix)

    ###
    ### PART 2: Build the two-scale operator and return
    ###
    return TwoScaleOperator(
        parent_space,
        child_space,
        global_subdiv_matrix,
        get_parent_to_children_elements(discontinuous_two_scale_op),
        get_child_to_parent_elements(discontinuous_two_scale_op),
    ),
    child_space
end
