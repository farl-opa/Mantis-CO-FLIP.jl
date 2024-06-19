
"""
    AbstractFormSpace

Supertype for all function spaces.
"""
abstract type AbstractFormSpace{n,k} end
# abstract type AbstractFormExpression{n,k} end

# function get_form_rank(::AbstractFormSpace{n,k})
#     return k
# end

# function get_spatial_dim(::AbstractFormSpace{n,k})
#     return n
# end

struct FormSpace{n, k, G, F} <: AbstractFormSpace{n,k}
    geometry::G
    fem_space::F

    # NOTE: FunctionSpaces.AbstractFunctionSpace does not have a parameter of dimension of manifold,
    # we need to add this, but it implies several changes (I started, but postponed it!)

    # 0- and n-form constructor
    function FormSpace(form_rank::Int, geometry::G, fem_space::Tuple{F}) where {domain_dim, codomain_dim, G <: Geometry.AbstractGeometry{domain_dim, codomain_dim}, F <: FunctionSpaces.AbstractFunctionSpace}
        # The dimension match is automatically verified because of the inputs types that are enforced
        new{domain_dim, form_rank, G, Tuple{F}}(geometry, fem_space)
    end

    # 1-form constructor in 2D
    function FormSpace(form_rank::Int, geometry::G, fem_space::Tuple{F_dξ, F_dη}) where {domain_dim, codomain_dim, G <: Geometry.AbstractGeometry{domain_dim, codomain_dim}, F_dξ <: FunctionSpaces.AbstractFunctionSpace, F_dη <: FunctionSpaces.AbstractFunctionSpace}
        form_components = binomial(domain_dim, form_rank)
        @assert form_components == length(fem_space)
        new{domain_dim, form_rank, G, Tuple{F_dξ, F_dη}}(geometry, fem_space)
    end

    # 1- and 2-form constructor in 3D
    function FormSpace(form_rank::Int, geometry::G, fem_space::Tuple{F_dξ, F_dη, F_dζ}) where {domain_dim, codomain_dim, G <: Geometry.AbstractGeometry{domain_dim, codomain_dim}, F_dξ <: FunctionSpaces.AbstractFunctionSpace, F_dη <: FunctionSpaces.AbstractFunctionSpace, F_dζ <: FunctionSpaces.AbstractFunctionSpace}
        form_components = binomial(domain_dim, form_rank)
        @assert form_components == length(fem_space)
        new{domain_dim, form_rank, G, Tuple{F_dξ, F_dη, F_dζ}}(geometry, fem_space)
    end
end

# function evaluate(form_space::FS, element_idx::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, FS <: AbstractFormSpace{manifold_dim, 0}}
#     local_form_basis, form_basis_indices = FunctionSpaces.evaluate(form_space.fem_space, element_idx, xi)
#     no_derivative_key = Tuple(0 for k in 1:manifold_dim)
#     return local_form_basis[no_derivative_key], form_basis_indices
# end

# function evaluate(form_space::FS, element_idx::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, FS <: AbstractFormSpace{manifold_dim, manifold_dim}}
#     local_form_basis, form_basis_indices = FunctionSpaces.evaluate(form_space.fem_space, element_idx, xi)
    
#     no_derivative_key = Tuple(0 for k in 1:manifold_dim)

#     return local_form_basis[no_derivative_key], form_basis_indices
# end

function evaluate(form_space::FS, element_idx::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, FS <: AbstractFormSpace{manifold_dim, form_rank}} where {form_rank}
    n_form_components = binomial(manifold_dim, form_rank)
    
    # Compute the offset in the numbering of the degrees of freedom 
    # of the form space. The form space is made up of components (dξ, dη, dζ)
    # each component has a numbering starting at 1, we stack the numbering 
    # so the dη numbering starts from the end of the dξ numbering, and so on.
    n_dofs_component = FunctionSpaces.get_dim.(form_space.fem_space)
    dof_offset_component = zeros(Int, n_form_components)
    dof_offset_component[2:end] .= cumsum(n_dofs_component[1:(n_form_components-1)])  # we skip the first one because the offset is 0

    # Allocate memory for the evaluation of all basis for all components
    local_form_basis = Vector{Array{Float64, 2}}(undef, n_form_components)
    form_basis_indices = Vector{Vector{Int}}(undef, n_form_components)

    # Loop over the spaces of the each component and evaluate them 
    # Note that the indices of the basis for each component are updated
    # by the offset defined above.
    no_derivative_key = Tuple(0 for k in 1:manifold_dim)
    for form_component_idx in 1:n_form_components
        # Evaluate the basis for the component
        local_basis, form_basis_indices[form_component_idx] = FunctionSpaces.evaluate(form_space.fem_space[form_component_idx], element_idx, xi)

        local_form_basis[form_component_idx] = local_basis[no_derivative_key]

        # Update the basis indices by adding the offset, since they are now 
        # a component among several, so the indexing needs to include all components
        form_basis_indices[form_component_idx] .+= dof_offset_component[form_component_idx]
    end

    return local_form_basis, form_basis_indices
end



function evaluate_exterior_derivative(form_space::FS, element_idx::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, FS <: AbstractFormSpace{manifold_dim, 0}}
#    local_form_basis, form_basis_indices = FunctionSpaces.evaluate(form_space.fem_space[1], element_idx, xi, 1)
    first_derivative_base_key = [0 for k in 1:manifold_dim]

    n_form_components_result = binomial(manifold_dim, 1)
    
    local_d_form_basis_eval = Vector{Array{Float64, 2}}(undef, n_form_components_result)
    form_basis_indices = Vector{Vector{Int}}(undef, n_form_components_result)
    
    for result_component_idx in 1:n_form_components_result
        # Select the derivative for this component
        first_derivative_base_key[result_component_idx] = 1
        first_derivative_key = Tuple(first_derivative_base_key)
        first_derivative_base_key[result_component_idx] = 0

        local_basis, form_basis_indices[result_component_idx] = FunctionSpaces.evaluate(form_space.fem_space[1], element_idx, xi, 1)

        local_d_form_basis_eval[result_component_idx] = local_basis[first_derivative_key]
    end

    return local_d_form_basis_eval, form_basis_indices
end


# function evaluate_exterior_derivative(form_space::FS, element_idx::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, FS <: AbstractFormSpace{manifold_dim, form_rank}} where {form_rank}
#     n_form_components = binomial(manifold_dim, form_rank)
    
#     # Compute the offset in the numbering of the degrees of freedom 
#     # of the form space. The form space is made up of components (dξ, dη, dζ)
#     # each component has a numbering starting at 1, we stack the numbering 
#     # so the dη numbering starts from the end of the dξ numbering, and so on.
#     n_dofs_component = FunctionSpaces.get_dim.(form_space.fem_space)
#     dof_offset_component = zeros(Int, n_form_components)
#     dof_offset_component[2:end] .= cumsum(n_dofs_component[1:(n_form_components-1)])  # we skip the first one because the offset is 0

#     # Allocate memory for the evaluation of all basis for all components
#     local_form_basis = Vector{Array{Float64, 2}}(undef, n_form_components)
#     form_basis_indices = Vector{Vector{Int}}(undef, n_form_components)

#     # Loop over the spaces of the each component and evaluate them 
#     # Note that the indices of the basis for each component are updated
#     # by the offset defined above.
#     no_derivative_key = Tuple(0 for k in 1:manifold_dim)
#     for form_component_idx in 1:n_form_components
#         # Evaluate the basis for the component
#         local_basis, form_basis_indices[form_component_idx] = FunctionSpaces.evaluate(form_space.fem_space[form_component_idx], element_idx, xi)

#         local_form_basis[form_component_idx] = local_basis[no_derivative_key]

#         # Update the basis indices by adding the offset, since they are now 
#         # a component among several, so the indexing needs to include all components
#         form_basis_indices[form_component_idx] .+= dof_offset_component[form_component_idx]
#     end

#     return local_form_basis, form_basis_indices
# end


"""
    get_dim(form_space::AbstractFormSpace)

Returns the number of degrees of freedom of the the FormSpace `form_space`.

# Arguments
- `form_space::AbstractFormSpace)`: the FormSpace to compute the number of degrees of freedom.
# Returns
- `::Int`: The number of degrees of freedom of the space.
"""
# function get_dim(form_space::FS) where {FS <: AbstractFormSpace{manifold_dim, 0}} where {manifold_dim}
#     # Get the total number of dofs for the form space by extracting 
#     # the number of dofs from the FEM space (0- and n-forms have only
#     # one component).
#     return FunctionSpaces.get_dim(form_space.fem_space)
# end

# function get_dim(form_space::FS) where {FS <: AbstractFormSpace{manifold_dim, manifold_dim}} where {manifold_dim}
#     # Get the total number of dofs for the form space by extracting 
#     # the number of dofs from the FEM space (0- and n-forms have only
#     # one component).
#     return FunctionSpaces.get_dim(form_space.fem_space)
# end

function get_dim(form_space::FS) where {FS <: AbstractFormSpace{manifold_dim, form_rank}} where {manifold_dim, form_rank}
    # Get the total number of dofs for the form space by looping 
    # over all the FEM spaces. Each FEM space is associated to a 
    # component of the form: 0- and n-forms have one component, and 
    # 1- and 2-forms have multiple components, depending on the 
    # dimension of the underlying manifold in which they are defined
    form_space_n_dofs = 0  # initialize the number of dofs to 0, to add the ones from each FEM space
    
    # Loop over the spaces and add their number of dofs
    for space in form_space.fem_space
        form_space_n_dofs += FunctionSpaces.get_dim(space)
    end

    return form_space_n_dofs
end