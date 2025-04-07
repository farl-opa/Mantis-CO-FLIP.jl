"""
    GTBSplineSpace{num_patches, T} <: AbstractFESpace{1, 1, num_patches}

Constructs a GTBSplineSpace from a tuple of NURBS or B-spline spaces and a vector of
regularity conditions.
Periodic spaces are a special case of GTBSplines for `num_patches = 1` and `regularity = [r]`
with `r > -1`.

# Fields
- `patch_spaces::NTuple{num_patches, T}`: A tuple of `num_patches` NURBS or B-spline spaces.
- `extraction_op::ExtractionOperator`: The extraction operator for the GTBSpline space.
- `dof_partition::Vector{Vector{Vector{Int}}}`: A vector of vectors of vectors of integers
    representing the degree of freedom partitioning.
- `regularity::Vector{Int}`: A vector of regularity conditions at the interfaces between
    the spaces.

# Arguments for constructor
- `patch_spaces::NTuple{num_patches, T}`: A tuple of `num_patches` NURBS or B-spline spaces.
- `regularity::Vector{Int}`: A vector of regularity conditions at the interfaces between
    the spaces.

# Throws
- `ArgumentError`: If the number of regularity conditions does not match the number of
    interfaces.
- `ArgumentError`: If the minimal polynomial degree of any pair of adjacent spaces is less
    than the corresponding regularity condition.
"""
struct GTBSplineSpace{num_patches, T} <: AbstractFESpace{1, 1, num_patches}
    patch_spaces::T
    extraction_op::ExtractionOperator
    dof_partition::Vector{Vector{Vector{Int}}}
    num_elements_per_patch::Vector{Int}
    regularity::Vector{Int}

    function GTBSplineSpace(
        patch_spaces::T, regularity::Vector{Int}
    ) where {num_patches, S <: Union{BSplineSpace, RationalFiniteElementSpace}, T <: NTuple{num_patches, S}}
        # Check if the number of regularity conditions matches the number of interfaces
        if length(regularity) != num_patches
            throw(ArgumentError("""\
                The number of regularity conditions should be equal to the number of bspline \
                interfaces. You have $(num_patches) interfaces and $(length(regularity)) regularity \
                conditions.\
                """
            ))
        end

        # Check if the polynomial degree of each pair of adjacent spaces is sufficient for the regularity condition
        for i in 1:num_patches
            j = i
            k = i + 1
            if i == num_patches
                k = 1
            end
            polynomial_degree = min(
                get_polynomial_degree(patch_spaces[j], 1),
                get_polynomial_degree(patch_spaces[k], 1),
            )
            if polynomial_degree < regularity[i]
                throw(ArgumentError("""\
                    The minimal polynomial degrees must be greater than or equal to the \
                    regularity. The minimal degree is $(polynomial_degree) and there is
                    regularity $(regularity[i]) at index $(i).\
                    """
                ))
            end
        end

        # Create the extraction operator
        extraction_op = extract_gtbspline_to_bspline(patch_spaces, regularity)

        # Allocate memory for degree of freedom partitioning: note, even if the 1D space is
        # implemented as a multipatch object, the dof partition treats it as a single patch!
        dof_partition = Vector{Vector{Vector{Int}}}(undef, 1)
        dof_partition[1] = Vector{Vector{Int}}(undef, 3)
        # get the number of boundary dofs
        n_dofs_left = 1
        n_dofs_right = 1
        if regularity[num_patches] > -1 # periodic space: non-default dof partitioning
            n_dofs_left = 0
            n_dofs_right = 0
        end
        # First, store the left dofs ...
        dof_partition[1][1] = collect(1:n_dofs_left)
        # ... then the interior dofs ...
        dof_partition[1][2] = collect(
            (n_dofs_left + 1):(get_num_basis(extraction_op) - n_dofs_right)
        )
        # ... and then finally the right dofs.
        dof_partition[1][3] = collect(
            (get_num_basis(extraction_op) - n_dofs_right + 1):get_num_basis(extraction_op)
        )

        new{num_patches, T}(
            patch_spaces,
            extraction_op,
            dof_partition,
            collect(map(get_num_elements, patch_spaces)),
            regularity)
    end
end

# Specialise for the single patch case, as the default get_patch_space for a single-patch
# space returns itself, leading to an infinite recursion.
function get_patch_space(space::GTBSplineSpace{1, T}, patch_id::Int) where {T}
    if patch_id == 1
        return space.patch_spaces[1]
    else
        throw(ArgumentError("Space on patch $patch_id requested, but $space only has 1 patch."))
    end
end

function get_local_basis(
    space::GTBSplineSpace,
    element_id::Int,
    xi::NTuple{1, Vector{Float64}},
    nderivatives::Int,
)
    # We need space ID and local element ID to know which function space to evaluate.
    patch_id, patch_element_id = get_patch_and_local_element_id(space, element_id)

    return evaluate(get_patch_space(space, patch_id), patch_element_id, xi, nderivatives)[1]
end
