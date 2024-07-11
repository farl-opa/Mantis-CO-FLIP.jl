
"""
    AbstractFormSpace

Supertype for all function spaces.
"""
abstract type AbstractFormSpace{manifold_dim, form_rank} <: AbstractFormField{manifold_dim, form_rank} end


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
    # of the form space. The form space is made up of components 
    # e.g, 
    #   0-forms: single component
    #   1-forms:(dξ₁, dξ₂) (2D)
    #   1-forms:(dξ₁, dξ₂, dξ₃) (3D)
    #   2-forms:(dξ₁dξ₂) (2D)
    #   2-forms:(dξ₂dξ₃, dξ₃dξ₁, dξ₁dξ₂) (3D)
    #   3-forms: single component
    # each component has a numbering starting at 1, we stack the numbering 
    # so the dξ₂ basis numbering starts from the end of the dξ₁ numbering, and so on.
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

    n_derivative_form_components = binomial(manifold_dim, 1)
    
    local_d_form_basis_eval = Vector{Array{Float64, 2}}(undef, n_derivative_form_components)
    form_basis_indices = Vector{Vector{Int}}(undef, n_derivative_form_components)
    
    d_local_fem_basis, fem_basis_indices = FunctionSpaces.evaluate(form_space.fem_space[1], element_idx, xi, 1)

    for derivative_component_idx in 1:n_derivative_form_components
        # Select the derivative for this component
        first_derivative_base_key[derivative_component_idx] = 1
        first_derivative_key = Tuple(first_derivative_base_key)
        first_derivative_base_key[derivative_component_idx] = 0
        
        # I think deepcopy is needed  because someone may make changes in place
        # somewhere outside this function and this will affect several 
        # of the components because they reuse the same matrix if no deepcopy is used
        local_d_form_basis_eval[derivative_component_idx] = deepcopy(d_local_fem_basis[first_derivative_key])
        form_basis_indices[derivative_component_idx] = deepcopy(fem_basis_indices)
    end

    return local_d_form_basis_eval, form_basis_indices
end

function evaluate_exterior_derivative(form_space::FS, element_idx::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, FS <: AbstractFormSpace{n, n} where n}
    throw("Manifold dim == Form rank: Unable to compute exterior derivative of volume forms.")
end

function evaluate_exterior_derivative(form_space::FS, element_idx::Int, xi::NTuple{2, Vector{Float64}}) where {FS <: AbstractFormSpace{2, 1}}
    # manifold_dim = 2
    n_form_components = 2 # binomial(manifold_dim, 1)
    n_derivative_form_components = 1 # binomial(manifold_dim, 2)
    
    # Compute the offset in the numbering of the degrees of freedom 
    # of the form space. The form space is made up of components 
    #   1-forms:(dξ₁, dξ₂) (2D)
    # each component has a numbering starting at 1, we stack the numbering 
    # so the dξ₂ basis numbering starts from the end of the dξ₁ numbering, and so on.
    n_dofs_component = FunctionSpaces.get_dim.(form_space.fem_space)
    dof_offset_component = zeros(Int, n_form_components)
    dof_offset_component[2:end] .= cumsum(n_dofs_component[1:(n_form_components-1)])  # we skip the first one because the offset is 0

    local_d_form_basis_eval = Vector{Array{Float64, 2}}(undef, n_derivative_form_components)
    form_basis_indices = Vector{Vector{Int}}(undef, n_derivative_form_components)
    
    d_local_fem_basis_dξ_1, form_basis_indices_dξ_1 = FunctionSpaces.evaluate(form_space.fem_space[1], element_idx, xi, 1)
    d_local_fem_basis_dξ_2, form_basis_indices_dξ_2 = FunctionSpaces.evaluate(form_space.fem_space[2], element_idx, xi, 1)
    form_basis_indices_dξ_2 .+= dof_offset_component[2]  # the second component basis need to be offset to have the correct numbering
    # The exterior derivative is 
    # (∂α₂/∂ξ₁ - ∂α₁/∂ξ₂) dξ₁∧dξ₂
    local_d_form_basis_eval[1] = hcat(d_local_fem_basis_dξ_2[1, 0], -d_local_fem_basis_dξ_1[0, 1])
    form_basis_indices[1] = vcat(form_basis_indices_dξ_2, form_basis_indices_dξ_1)
    
    return local_d_form_basis_eval, form_basis_indices
end

function evaluate_exterior_derivative(form_space::FS, element_idx::Int, xi::NTuple{3, Vector{Float64}}) where {FS <: AbstractFormSpace{3, 1}}
    # manifold_dim = 3
    n_form_components = 3 # binomial(manifold_dim, 1)
    n_derivative_form_components = 3 # binomial(manifold_dim, 2)
    
    # Compute the offset in the numbering of the degrees of freedom 
    # of the form space. The form space is made up of components 
    #   1-forms:(dξ₁, dξ₂, dξ₃) (3D)
    # each component has a numbering starting at 1, we stack the numbering 
    # so the dξ₂ basis numbering starts from the end of the dξ₁ numbering, and so on.
    n_dofs_component = FunctionSpaces.get_dim.(form_space.fem_space)
    dof_offset_component = zeros(Int, n_form_components)
    dof_offset_component[2:end] .= cumsum(n_dofs_component[1:(n_form_components-1)])  # we skip the first one because the offset is 0
    
    local_d_form_basis_eval = Vector{Array{Float64, 2}}(undef, n_derivative_form_components)
    form_basis_indices = Vector{Vector{Int}}(undef, n_derivative_form_components)
    
    d_local_fem_basis_dξ_1, form_basis_indices_dξ_1 = FunctionSpaces.evaluate(form_space.fem_space[1], element_idx, xi, 1)
    
    d_local_fem_basis_dξ_2, form_basis_indices_dξ_2 = FunctionSpaces.evaluate(form_space.fem_space[2], element_idx, xi, 1)
    form_basis_indices_dξ_2 .+= dof_offset_component[2]
    
    d_local_fem_basis_dξ_3, form_basis_indices_dξ_3 = FunctionSpaces.evaluate(form_space.fem_space[3], element_idx, xi, 1)
    form_basis_indices_dξ_3 .+= dof_offset_component[3]

    # The exterior derivative is 
    # (∂α₃/∂ξ₂ - ∂α₂/∂ξ₃) dξ₂∧dξ₃ + (∂α₁/∂ξ₃ - ∂α₃/∂ξ₁) dξ₃∧dξ₁ + (∂α₂/∂ξ₁ - ∂α₁/∂ξ₂) dξ₁∧dξ₂
    # First: (∂α₃/∂ξ₂ - ∂α₂/∂ξ₃) dξ₂∧dξ₃
    local_d_form_basis_eval[1] = hcat(d_local_fem_basis_dξ_3[0, 1, 0], -d_local_fem_basis_dξ_2[0, 0, 1])
    form_basis_indices[1] = vcat(form_basis_indices_dξ_3, form_basis_indices_dξ_2)
    # Second: (∂α₁/∂ξ₃ - ∂α₃/∂ξ₁) dξ₃∧dξ₁
    local_d_form_basis_eval[2] = hcat(d_local_fem_basis_dξ_1[0, 0, 1], -d_local_fem_basis_dξ_3[1, 0, 0])
    form_basis_indices[2] = vcat(form_basis_indices_dξ_1, form_basis_indices_dξ_3)
    # Third: (∂α₂/∂ξ₁ - ∂α₁/∂ξ₂) dξ₁∧dξ₂
    local_d_form_basis_eval[3] = hcat(d_local_fem_basis_dξ_2[1, 0, 0], -d_local_fem_basis_dξ_1[0, 1, 0])
    form_basis_indices[3] = vcat(form_basis_indices_dξ_2, form_basis_indices_dξ_1)
    
    return local_d_form_basis_eval, form_basis_indices
end

function evaluate_exterior_derivative(form_space::FS, element_idx::Int, xi::NTuple{3, Vector{Float64}}) where {FS <: AbstractFormSpace{3, 2}}
    # manifold_dim = 3
    n_form_components = 3 # binomial(manifold_dim, 1)
    n_derivative_form_components = 1 # binomial(manifold_dim, 2)
    
    # Compute the offset in the numbering of the degrees of freedom 
    # of the form space. The form space is made up of components 
    #   2-forms:(dξ₂dξ₃, dξ₃dξ₁, dξ₁dξ₂) (3D)
    # each component has a numbering starting at 1, we stack the numbering 
    # so the dξ₂ basis numbering starts from the end of the dξ₁ numbering, and so on.
    n_dofs_component = FunctionSpaces.get_dim.(form_space.fem_space)
    dof_offset_component = zeros(Int, n_form_components)
    dof_offset_component[2:end] .= cumsum(n_dofs_component[1:(n_form_components-1)])  # we skip the first one because the offset is 0


    local_d_form_basis_eval = Vector{Array{Float64, 2}}(undef, n_derivative_form_components)
    form_basis_indices = Vector{Vector{Int}}(undef, n_derivative_form_components)
    
    d_local_fem_basis_dξ_2_dξ_3, form_basis_indices_dξ_2_dξ_3 = FunctionSpaces.evaluate(form_space.fem_space[1], element_idx, xi, 1)
    
    d_local_fem_basis_dξ_3_dξ_1, form_basis_indices_dξ_3_dξ_1 = FunctionSpaces.evaluate(form_space.fem_space[2], element_idx, xi, 1)
    form_basis_indices_dξ_3_dξ_1 .+= dof_offset_component[2]

    d_local_fem_basis_dξ_1_dξ_2, form_basis_indices_dξ_1_dξ_2 = FunctionSpaces.evaluate(form_space.fem_space[3], element_idx, xi, 1)
    form_basis_indices_dξ_1_dξ_2 .+= dof_offset_component[3]

    # The form is 
    # α₁ dξ₂∧dξ₃ + α₂ dξ₃∧dξ₁ + α₃ dξ₁∧dξ₂
    # The exterior derivative is 
    # (∂α₁/∂ξ₁ + ∂α₂/∂ξ₂ + ∂α₃/∂ξ₃) dξ₁∧dξ₂∧dξ₃
    local_d_form_basis_eval[1] = hcat(d_local_fem_basis_dξ_2_dξ_3[1, 0, 0], d_local_fem_basis_dξ_3_dξ_1[0, 1, 0], d_local_fem_basis_dξ_1_dξ_2[0, 0, 1])
    form_basis_indices[1] = vcat(form_basis_indices_dξ_2_dξ_3, form_basis_indices_dξ_3_dξ_1, form_basis_indices_dξ_1_dξ_2)
    
    return local_d_form_basis_eval, form_basis_indices
end


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