

@doc raw"""
    FormSpace{manifold_dim, form_rank, G, F} <: AbstractFormSpace{manifold_dim, form_rank, G}

Concrete implementation of a function space for differential forms.

# Fields
- `geometry::G`: The geometry of the manifold
- `fem_space::F`: The finite element space(s) used for the form components
- `label::String`: Label for the form space

# Type parameters
- `manifold_dim`: Dimension of the manifold
- `form_rank`: Rank of the differential form
- `G`: Type of the geometry
- `F`: Type of the finite element space
"""
struct FormSpace{manifold_dim, form_rank, G, F} <: AbstractFormSpace{manifold_dim, form_rank, G}
    geometry::G
    fem_space::F
    label::String

    # NOTE: FunctionSpaces.AbstractFunctionSpace does not have a parameter of dimension of manifold,
    # we need to add this, but it implies several changes (I started, but postponed it!)

    @doc raw"""
        FormSpace(form_rank::Int, geometry::G, fem_space::Tuple{F}) where {manifold_dim, G <: Geometry.AbstractGeometry{manifold_dim}, F <: FunctionSpaces.AbstractFunctionSpace}

    Constructor for 0-forms and n-forms.

    # Arguments
    - `form_rank::Int`: Rank of the differential form (should be `0` or `manifold_dim``).
    - `geometry::G`: The geometry of the manifold.
    - `fem_space::Tuple{F}`: A tuple containing a single finite element space, since 0- and n-forms 
            contain only one component.
    - `label::String`: Label for the form space.
    """
    function FormSpace(form_rank::Int, geometry::G, fem_space::Tuple{F}, label::String) where {manifold_dim, G <: Geometry.AbstractGeometry{manifold_dim}, F <: FunctionSpaces.AbstractFunctionSpace}
        @assert form_rank in Set([0, manifold_dim])
        new{manifold_dim, form_rank, G, Tuple{F}}(geometry, fem_space, label)
    end

    @doc raw"""
        FormSpace(form_rank::Int, geometry::G, fem_space::Tuple{F_dξ, F_dη}) where {manifold_dim, G <: Geometry.AbstractGeometry{manifold_dim}, F_dξ <: FunctionSpaces.AbstractFunctionSpace, F_dη <: FunctionSpaces.AbstractFunctionSpace}

    Constructor for 1-forms in 2D.

    # Arguments
    - `form_rank::Int`: Rank of the differential form (must be 1)
    - `geometry::G`: The geometry of the space
    - `fem_space::Tuple{F_dξ, F_dη}`: A tuple containing two finite element spaces for dξ and dη components
    - `label::String`: Label for the form space.
    """
    function FormSpace(form_rank::Int, geometry::G, fem_space::Tuple{F_dξ, F_dη}, label::String) where {manifold_dim, G <: Geometry.AbstractGeometry{manifold_dim}, F_dξ <: FunctionSpaces.AbstractFunctionSpace, F_dη <: FunctionSpaces.AbstractFunctionSpace}
        @assert form_rank == 1
        form_components = binomial(manifold_dim, form_rank)
        @assert form_components == length(fem_space)
        new{manifold_dim, form_rank, G, Tuple{F_dξ, F_dη}}(geometry, fem_space, label)
    end

    @doc raw"""
        FormSpace(form_rank::Int, geometry::G, fem_space::Tuple{F_dξ, F_dη, F_dζ}) where {domain_dim, codomain_dim, G <: Geometry.AbstractGeometry{domain_dim, codomain_dim}, F_dξ <: FunctionSpaces.AbstractFunctionSpace, F_dη <: FunctionSpaces.AbstractFunctionSpace, F_dζ <: FunctionSpaces.AbstractFunctionSpace}

    Constructor for 1-forms and 2-forms in 3D.

    # Arguments
    - `form_rank::Int`: Rank of the differential form (must be 1 or 2)
    - `geometry::G`: The geometry of the manifold
    - `fem_space::Tuple{F_dξ, F_dη, F_dζ}`: A tuple containing three finite element spaces for dξ, dη, and dζ components
    - `label::String`: Label for the form space.
    """
    function FormSpace(form_rank::Int, geometry::G, fem_space::Tuple{F_dξ, F_dη, F_dζ}, label::String) where {manifold_dim, G <: Geometry.AbstractGeometry{manifold_dim}, F_dξ <: FunctionSpaces.AbstractFunctionSpace, F_dη <: FunctionSpaces.AbstractFunctionSpace, F_dζ <: FunctionSpaces.AbstractFunctionSpace}
        @assert form_rank in Set([1, 2])
        form_components = binomial(manifold_dim, form_rank)
        @assert form_components == length(fem_space)
        new{manifold_dim, form_rank, G, Tuple{F_dξ, F_dη, F_dζ}}(geometry, fem_space, label)
    end
end

@doc raw"""
    evaluate(form_space::FS, element_idx::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, FS <: AbstractFormSpace{manifold_dim, form_rank, G}} where {form_rank, G <: Geometry.AbstractGeometry}

Evaluate the basis functions of a differential form space at given points `xi`.

# Arguments
- `form_space::FS`: The differential form space.
- `element_idx::Int`: Index of the element to evaluate.
- `xi::NTuple{manifold_dim, Vector{Float64}}`: Tuple of tensor-product coordinate vectors for evaluation points. 
        Points are the tensor product of the coordinates per dimension, therefore there will be
        ``n_{1} \times \dots \times n_{\texttt{manifold_dim}}`` points where to evaluate.

# Returns
- `local_form_basis`: Vector of arrays containing evaluated basis functions for each component evaluated at each point.
        `local_form_basis[i][j,k]` is the evaluation at the tensor product point `j` of the component `i` of global basis
        `form_basis_indices[i][k]` of component `i` evaluated. Note that basis do not have values in all components. 
        Typically each basis is associated only to one component, e.g., ``\mathrm{d}\xi_{1}``. To avoid populating 
        the evaluation with zeros on the other components, we only return, per component, the basis that have nonzero values 
        on that component.
- `form_basis_indices`: Vector of vectors containing the global indices of the basis forms.
        `form_basis_indices[i][k]` is the global index of the returned basis `k` of this element
        from component `i`.

# Sizes
- `local_form_basis`: Vector of length `n_form_components`, where each element is an Array{Float64, 2} of size (n_evaluation_points, n_basis_functions)
- `form_basis_indices`: Vector of length `n_form_components`, where each element is a Vector{Int} of length n_basis_functions
"""
function evaluate(form_space::FS, element_idx::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, FS <: AbstractFormSpace{manifold_dim, form_rank, G}} where {form_rank, G <: Geometry.AbstractGeometry}
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
    n_dofs_component = FunctionSpaces.get_num_basis.(form_space.fem_space)
    dof_offset_component = zeros(Int, n_form_components)
    dof_offset_component[2:end] .= cumsum(n_dofs_component[1:(n_form_components-1)])  # we skip the first one because the offset is 0

    # Allocate memory for the evaluation of all basis for all components
    local_form_basis = Vector{Array{Float64, 2}}(undef, n_form_components)
    form_basis_indices = Vector{Vector{Int}}(undef, n_form_components)

    # Loop over the spaces of the each component and evaluate them 
    # Note that the indices of the basis for each component are updated
    # by the offset defined above.
    for form_component_idx in 1:n_form_components
        # Evaluate the basis for the component
        local_basis, form_basis_indices[form_component_idx] = FunctionSpaces.evaluate(form_space.fem_space[form_component_idx], element_idx, xi)

        local_form_basis[form_component_idx] = local_basis[1][1]

        # Update the basis indices by adding the offset, since they are now 
        # a component among several, so the indexing needs to include all components
        form_basis_indices[form_component_idx] .+= dof_offset_component[form_component_idx]
    end

    return local_form_basis, form_basis_indices
end


@doc raw"""
    evaluate_exterior_derivative(form_space::FS, element_idx::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, FS <: AbstractFormSpace{manifold_dim, 0, G}} where {G <: Geometry.AbstractGeometry}

Evaluate the exterior derivative of a 0-form (scalar function) at given points.

# Arguments
- `form_space::FS`: The 0-form space.
- `element_idx::Int`: Index of the element to evaluate.
- `xi::NTuple{manifold_dim, Vector{Float64}}`: Tuple of tensor-product coordinate vectors for evaluation points. 
        Points are the tensor product of the coordinates per dimension, therefore there will be
        ``n_{1} \times \dots \times n_{\texttt{manifold_dim}}`` points where to evaluate.

# Returns
- `local_d_form_basis_eval`: Vector of arrays containing evaluated exterior derivative basis functions
        `local_d_form_basis_eval[i][j, k]` is the component `i` of the exterior derivative of global basis 
        `form_basis_indices[i][k]` evaluated at the tensor product point `j`. 
        See [`(form_space::FS, element_idx::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, FS <: AbstractFormSpace{manifold_dim, form_rank, G}} where {form_rank, G <: Geometry.AbstractGeometry}`](@ref)
        for more details: the formats are identical.
- `form_basis_indices`: Vector of vectors containing indices of the basis functions
        `form_basis_indices[i][k]` is the global index of the  exterior derivative of the 
        returned basis `k` of this element from component `i`.

# Sizes
- `local_d_form_basis_eval`: Vector of length `manifold_dim`, where each element is an Array{Float64, 2} of size (n_evaluation_points, n_basis_functions).
- `form_basis_indices`: Vector of length `manifold_dim`, where each element is a Vector{Int} of length n_basis_functions
"""
function evaluate_exterior_derivative(form_space::FS, element_idx::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, FS <: AbstractFormSpace{manifold_dim, 0, G}} where {G <: Geometry.AbstractGeometry}
#    local_form_basis, form_basis_indices = FunctionSpaces.evaluate(form_space.fem_space[1], element_idx, xi, 1)
    first_derivative_base_key = [0 for k in 1:manifold_dim]

    n_derivative_form_components = binomial(manifold_dim, 1)
    
    local_d_form_basis_eval = Vector{Array{Float64, 2}}(undef, n_derivative_form_components)
    form_basis_indices = Vector{Vector{Int}}(undef, n_derivative_form_components)
    
    d_local_fem_basis, fem_basis_indices = FunctionSpaces.evaluate(form_space.fem_space[1], element_idx, xi, 1)

    for derivative_component_idx in 1:n_derivative_form_components
        # Select the derivative for this component
        first_derivative_base_key[derivative_component_idx] = 1
        first_derivative_idx = FunctionSpaces._get_derivative_idx(first_derivative_base_key)
        first_derivative_base_key[derivative_component_idx] = 0
        
        # I think deepcopy is needed  because someone may make changes in place
        # somewhere outside this function and this will affect several 
        # of the components because they reuse the same matrix if no deepcopy is used
        local_d_form_basis_eval[derivative_component_idx] = deepcopy(d_local_fem_basis[2][first_derivative_idx])
        form_basis_indices[derivative_component_idx] = deepcopy(fem_basis_indices)
    end

    return local_d_form_basis_eval, form_basis_indices
end

function evaluate_exterior_derivative(form_space::FS, element_idx::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, FS <: AbstractFormSpace{n, n, G}} where {n, G <: Geometry.AbstractGeometry}
    throw("Manifold dim == Form rank: Unable to compute exterior derivative of volume forms.")
end

@doc raw"""
    evaluate_exterior_derivative(form_space::FS, element_idx::Int, xi::NTuple{2, Vector{Float64}}) where {FS <: AbstractFormSpace{2, 1, G}} where {G <: Geometry.AbstractGeometry}

Evaluate the exterior derivative of a 1-form in 2D.

# Arguments
- `form_space::FS`: The 1-form space in 2D.
- `element_idx::Int`: Index of the element to evaluate.
- `xi::NTuple{2, Vector{Float64}}`: Tuple of tensor-product coordinate vectors for evaluation points. 
        Points are the tensor product of the coordinates per dimension, therefore there will be
        ``n_{1} \times \dots \times n_{\texttt{manifold_dim}}`` points where to evaluate.

# Returns
- `local_d_form_basis_eval`: Vector of arrays containing evaluated exterior derivative basis functions
        `local_d_form_basis_eval[i][j, k]` is the component `i` of the exterior derivative of global basis 
        `form_basis_indices[i][k]` evaluated at the tensor product point `j`. 
        See [`(form_space::FS, element_idx::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, FS <: AbstractFormSpace{manifold_dim, form_rank, G}} where {form_rank, G <: Geometry.AbstractGeometry}`](@ref)
        for more details: the formats are identical.
- `form_basis_indices`: Vector of vectors containing indices of the basis functions
        `form_basis_indices[i][k]` is the global index of the  exterior derivative of the 
        returned basis `k` of this element from component `i`.

# Sizes
- `local_d_form_basis_eval`: Vector of length 1, where the element is an Array{Float64, 2} of size (n_evaluation_points, n_basis_functions_dξ_1 + n_basis_functions_dξ_2)
- `form_basis_indices`: Vector of length 1, where the element is a Vector{Int} of length (n_basis_functions_dξ_1 + n_basis_functions_dξ_2)
"""
function evaluate_exterior_derivative(form_space::FS, element_idx::Int, xi::NTuple{2, Vector{Float64}}) where {FS <: AbstractFormSpace{2, 1, G}} where {G <: Geometry.AbstractGeometry}
    # manifold_dim = 2
    n_form_components = 2 # binomial(manifold_dim, 1)
    n_derivative_form_components = 1 # binomial(manifold_dim, 2)
    
    # Compute the offset in the numbering of the degrees of freedom 
    # of the form space. The form space is made up of components 
    #   1-forms:(dξ₁, dξ₂) (2D)
    # each component has a numbering starting at 1, we stack the numbering 
    # so the dξ₂ basis numbering starts from the end of the dξ₁ numbering, and so on.
    n_dofs_component = FunctionSpaces.get_num_basis.(form_space.fem_space)
    dof_offset_component = zeros(Int, n_form_components)
    dof_offset_component[2:end] .= cumsum(n_dofs_component[1:(n_form_components-1)])  # we skip the first one because the offset is 0

    local_d_form_basis_eval = Vector{Array{Float64, 2}}(undef, n_derivative_form_components)
    form_basis_indices = Vector{Vector{Int}}(undef, n_derivative_form_components)
    
    d_local_fem_basis_dξ_1, form_basis_indices_dξ_1 = FunctionSpaces.evaluate(form_space.fem_space[1], element_idx, xi, 1)
    d_local_fem_basis_dξ_2, form_basis_indices_dξ_2 = FunctionSpaces.evaluate(form_space.fem_space[2], element_idx, xi, 1)
    form_basis_indices_dξ_2 .+= dof_offset_component[2]  # the second component basis need to be offset to have the correct numbering
    # The exterior derivative is 
    # (∂α₂/∂ξ₁ - ∂α₁/∂ξ₂) dξ₁∧dξ₂
    der_idx_dξ_1_2 = FunctionSpaces._get_derivative_idx([0, 1])
    der_idx_dξ_2_1 = FunctionSpaces._get_derivative_idx([1, 0])
    local_d_form_basis_eval[1] = hcat(d_local_fem_basis_dξ_2[2][der_idx_dξ_2_1], -d_local_fem_basis_dξ_1[2][der_idx_dξ_1_2])
    form_basis_indices[1] = vcat(form_basis_indices_dξ_2, form_basis_indices_dξ_1)
    
    return local_d_form_basis_eval, form_basis_indices
end

@doc raw"""
    evaluate_exterior_derivative(form_space::FS, element_idx::Int, xi::NTuple{3, Vector{Float64}}) where {FS <: AbstractFormSpace{3, 1, G}} where {G <: Geometry.AbstractGeometry}

Evaluate the exterior derivative of a 1-form in 3D.

# Arguments
- `form_space::FS`: The 1-form space in 3D
- `element_idx::Int`: Index of the element to evaluate
- `xi::NTuple{3, Vector{Float64}}`: Tuple of tensor-product coordinate vectors for evaluation points. 
        Points are the tensor product of the coordinates per dimension, therefore there will be
        ``n_{1} \times \dots \times n_{\texttt{manifold_dim}}`` points where to evaluate.

# Returns
- `local_d_form_basis_eval`: Vector of arrays containing evaluated exterior derivative basis functions
        `local_d_form_basis_eval[i][j, k]` is the component `i` of the exterior derivative of global basis 
        `form_basis_indices[i][k]` evaluated at the tensor product point `j`. 
        See [`(form_space::FS, element_idx::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, FS <: AbstractFormSpace{manifold_dim, form_rank, G}} where {form_rank, G <: Geometry.AbstractGeometry}`](@ref)
        for more details: the formats are identical.
- `form_basis_indices`: Vector of vectors containing indices of the basis functions
        `form_basis_indices[i][k]` is the global index of the  exterior derivative of the 
        returned basis `k` of this element from component `i`.

# Sizes
- `local_d_form_basis_eval`: Vector of length 3, where each element is an Array{Float64, 2} of size (n_evaluation_points, n_basis_functions_dξ_i + n_basis_functions_dξ_j)
- `form_basis_indices`: Vector of length 3, where each element is a Vector{Int} of length (n_basis_functions_dξ_i + n_basis_functions_dξ_j)
"""
function evaluate_exterior_derivative(form_space::FS, element_idx::Int, xi::NTuple{3, Vector{Float64}}) where {FS <: AbstractFormSpace{3, 1, G}} where {G <: Geometry.AbstractGeometry}
    # manifold_dim = 3
    n_form_components = 3 # binomial(manifold_dim, 1)
    n_derivative_form_components = 3 # binomial(manifold_dim, 2)
    
    # Compute the offset in the numbering of the degrees of freedom 
    # of the form space. The form space is made up of components 
    #   1-forms:(dξ₁, dξ₂, dξ₃) (3D)
    # each component has a numbering starting at 1, we stack the numbering 
    # so the dξ₂ basis numbering starts from the end of the dξ₁ numbering, and so on.
    n_dofs_component = FunctionSpaces.get_num_basis.(form_space.fem_space)
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
    der_idx_dξ_3_2 = FunctionSpaces._get_derivative_idx([0,1,0])
    der_idx_dξ_2_3 = FunctionSpaces._get_derivative_idx([0,0,1])
    local_d_form_basis_eval[1] = hcat(d_local_fem_basis_dξ_3[2][der_idx_dξ_3_2], -d_local_fem_basis_dξ_2[2][der_idx_dξ_2_3])
    form_basis_indices[1] = vcat(form_basis_indices_dξ_3, form_basis_indices_dξ_2)
    # Second: (∂α₁/∂ξ₃ - ∂α₃/∂ξ₁) dξ₃∧dξ₁
    der_idx_dξ_1_3 = FunctionSpaces._get_derivative_idx([0,0,1])
    der_idx_dξ_3_1 = FunctionSpaces._get_derivative_idx([1,0,0])
    local_d_form_basis_eval[2] = hcat(d_local_fem_basis_dξ_1[2][der_idx_dξ_1_3], -d_local_fem_basis_dξ_3[2][der_idx_dξ_3_1])
    form_basis_indices[2] = vcat(form_basis_indices_dξ_1, form_basis_indices_dξ_3)
    # Third: (∂α₂/∂ξ₁ - ∂α₁/∂ξ₂) dξ₁∧dξ₂
    der_idx_dξ_2_1 = FunctionSpaces._get_derivative_idx([1,0,0])
    der_idx_dξ_1_2 = FunctionSpaces._get_derivative_idx([0,1,0])
    local_d_form_basis_eval[3] = hcat(d_local_fem_basis_dξ_2[2][der_idx_dξ_2_1], -d_local_fem_basis_dξ_1[2][der_idx_dξ_1_2])
    form_basis_indices[3] = vcat(form_basis_indices_dξ_2, form_basis_indices_dξ_1)
    
    return local_d_form_basis_eval, form_basis_indices
end

@doc raw"""
    evaluate_exterior_derivative(form_space::FS, element_idx::Int, xi::NTuple{3, Vector{Float64}}) where {FS <: AbstractFormSpace{3, 2, G}} where {G <: Geometry.AbstractGeometry}

Evaluate the exterior derivative of a 2-form in 3D.

# Arguments
- `form_space::FS`: The 2-form space in 3D
- `element_idx::Int`: Index of the element to evaluate
- `xi::NTuple{3, Vector{Float64}}`: Tuple of tensor-product coordinate vectors for evaluation points. 
        Points are the tensor product of the coordinates per dimension, therefore there will be
        ``n_{1} \times \dots \times n_{\texttt{manifold_dim}}`` points where to evaluate.

# Returns
- `local_d_form_basis_eval`: Vector of arrays containing evaluated exterior derivative basis functions
        `local_d_form_basis_eval[i][j, k]` is the component `i` of the exterior derivative of global basis 
        `form_basis_indices[i][k]` evaluated at the tensor product point `j`. 
        See [`(form_space::FS, element_idx::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, FS <: AbstractFormSpace{manifold_dim, form_rank, G}} where {form_rank, G <: Geometry.AbstractGeometry}`](@ref)
        for more details: the formats are identical.
- `form_basis_indices`: Vector of vectors containing indices of the basis functions
        `form_basis_indices[i][k]` is the global index of the  exterior derivative of the 
        returned basis `k` of this element from component `i`.

# Sizes
- `local_d_form_basis_eval`: Vector of length 1, where the element is an Array{Float64, 2} of size (n_evaluation_points, n_basis_functions_dξ_2_dξ_3 + n_basis_functions_dξ_3_dξ_1 + n_basis_functions_dξ_1_dξ_2)
- `form_basis_indices`: Vector of length 1, where the element is a Vector{Int} of length (n_basis_functions_dξ_2_dξ_3 + n_basis_functions_dξ_3_dξ_1 + n_basis_functions_dξ_1_dξ_2)
"""
function evaluate_exterior_derivative(form_space::FS, element_idx::Int, xi::NTuple{3, Vector{Float64}}) where {FS <: AbstractFormSpace{3, 2, G}} where {G <: Geometry.AbstractGeometry}
    # manifold_dim = 3
    n_form_components = 3 # binomial(manifold_dim, 1)
    n_derivative_form_components = 1 # binomial(manifold_dim, 2)
    
    # Compute the offset in the numbering of the degrees of freedom 
    # of the form space. The form space is made up of components 
    #   2-forms:(dξ₂dξ₃, dξ₃dξ₁, dξ₁dξ₂) (3D)
    # each component has a numbering starting at 1, we stack the numbering 
    # so the dξ₂ basis numbering starts from the end of the dξ₁ numbering, and so on.
    n_dofs_component = FunctionSpaces.get_num_basis.(form_space.fem_space)
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
    der_idx_dξ_2_dξ_3_1 = FunctionSpaces._get_derivative_idx([1,0,0])
    der_idx_dξ_3_dξ_1_2 = FunctionSpaces._get_derivative_idx([0,1,0])
    der_idx_dξ_1_dξ_2_3 = FunctionSpaces._get_derivative_idx([0,0,1])
    local_d_form_basis_eval[1] = hcat(d_local_fem_basis_dξ_2_dξ_3[1][der_idx_dξ_2_dξ_3_1], d_local_fem_basis_dξ_3_dξ_1[2][der_idx_dξ_3_dξ_1_2], d_local_fem_basis_dξ_1_dξ_2[2][der_idx_dξ_1_dξ_2_3])
    form_basis_indices[1] = vcat(form_basis_indices_dξ_2_dξ_3, form_basis_indices_dξ_3_dξ_1, form_basis_indices_dξ_1_dξ_2)
    
    return local_d_form_basis_eval, form_basis_indices
end

# Hodge star Operator (⋆) evaluation methods

# ωⁿ = √det(gᵢⱼ) dx¹∧…∧dxⁿ denotes the volume form of a Riemannian manifold.
# αⱼᵏ denotes the j-th component of a k-form αᵏ. 

# 0-forms (n dimensions)
@doc raw"""
    evaluate_hodge_star(form_space::FS, element_id::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, FS <: AbstractFormSpace{manifold_dim, 0}}

Evaluates the Hodge star operator ⋆ of a 0-form in n dimensions.

# Arguments
- `form_space::FS`: The 0-form space in n dimensions.
- `element_idx::Int`: Index of the element to evaluate
- `xi::NTuple{manifold_dim, Vector{Float64}}`: Tuple of tensor-product coordinate vectors for evaluation points. 
        Points are the tensor product of the coordinates per dimension, therefore there will be
        ``n_{1} \times \dots \times n_{\texttt{manifold_dim}}`` points where to evaluate.

# Returns
- `local_star_form_basis_eval`: Vector of arrays containing evaluated hodge star of the basis functions.
- `local_star_form_basis_indices`: Vector of vectors containing indices of the basis functions.
"""
function evaluate_hodge_star(form_space::FS, element_id::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, FS <: AbstractFormSpace{manifold_dim, 0}}
    _, sqrt_g = Geometry.metric(form_space.geometry, element_id, xi)

    local_star_form_basis_eval, local_star_form_basis_indices = evaluate(form_space, element_id, xi)

    # ⋆α₁⁰ = α₁⁰√det(gᵢⱼ) dξ₁∧…∧dξₙ. 
    local_star_form_basis_eval[1] .= local_star_form_basis_eval[1] .* sqrt_g
 
    return local_star_form_basis_eval, local_star_form_basis_indices
end

# n-forms (n dimensions)
 @doc raw"""
    evaluate_hodge_star(form_space::FS, element_id::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, FS <: AbstractFormSpace{manifold_dim, manifold_dim}}

Evaluates the Hodge star operator ⋆ of a n-form in n dimensions.

# Arguments
- `form_space::FS`: The n-form space in n dimensions.
- `element_idx::Int`: Index of the element to evaluate
- `xi::NTuple{manifold_dim, Vector{Float64}}`: Tuple of tensor-product coordinate vectors for evaluation points. 
        Points are the tensor product of the coordinates per dimension, therefore there will be
        ``n_{1} \times \dots \times n_{\texttt{manifold_dim}}`` points where to evaluate.

# Returns
- `local_star_form_basis_eval`: Vector of arrays containing evaluated hodge star of the basis functions.
- `local_star_form_basis_indices`: Vector of vectors containing indices of the basis functions.
"""
function evaluate_hodge_star(form_space::FS, element_id::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, FS <: AbstractFormSpace{manifold_dim, manifold_dim}}
    _, sqrt_g = Geometry.metric(form_space.geometry, element_id, xi)

    local_star_form_basis_eval, local_star_form_basis_indices = evaluate(form_space, element_id, xi)

    # ⋆α₁ⁿdξ₁∧…∧dξₙ = α₁ⁿ(√det(gᵢⱼ))⁻¹. 
    local_star_form_basis_eval[1] .= local_star_form_basis_eval[1] .* (sqrt_g.^(-1))
 
    return local_star_form_basis_eval, local_star_form_basis_indices
end

# 1-forms (2 dimensions)
@doc raw"""
evaluate_hodge_star(form_space::FS, element_id::Int, xi::NTuple{2, Vector{Float64}}) where {FS <: AbstractFormSpace{2, 1}}

Evaluates the Hodge star operator ⋆ of a 1-form in 2 dimensions.

# Arguments
- `form_space::FS`: The 1-form space in 2 dimensions.
- `element_idx::Int`: Index of the element to evaluate
- `xi::NTuple{2, Vector{Float64}}`: Tuple of tensor-product coordinate vectors for evaluation points. 
    Points are the tensor product of the coordinates per dimension, therefore there will be
    ``n_{1} \times \dots \times n_{\texttt{manifold_dim}}`` points where to evaluate.

# Returns
- `local_star_form_basis_eval`: Vector of arrays containing evaluated hodge star of the basis functions.
- `local_star_form_basis_indices`: Vector of vectors containing indices of the basis functions.
"""
function evaluate_hodge_star(form_space::FS, element_id::Int, xi::NTuple{2, Vector{Float64}}) where {FS <: AbstractFormSpace{2, 1}}
    g, sqrt_g = Geometry.metric(form_space.geometry, element_id, xi)
    inv_g = mapslices(inv, g, dims=(2,3))

    local_star_form_basis_eval, local_star_form_basis_indices = evaluate(form_space, element_id, xi)
    star_form_basis_eval, star_form_basis_indices = similar(local_star_form_basis_eval), similar(local_star_form_basis_indices)

    # ⋆(α₁¹dξ₁+α₂¹dξ₂) = [-(α₁¹g²¹+α₂¹g²²)dξ₁ + (α₁¹g¹¹+α₂¹g¹²)dξ₂]√det(gᵢⱼ). 
    # First: -(α₁¹g²¹+α₂¹g²²)dξ₁
    star_form_basis_eval[1] = @views hcat(-local_star_form_basis_eval[1].*inv_g[:,2,1], -local_star_form_basis_eval[2].*inv_g[:,2,2]).*sqrt_g
    star_form_basis_indices[1] = vcat(local_star_form_basis_indices[1], local_star_form_basis_indices[2])

    # Second: (α₁¹g¹¹+α₂¹g¹²)dξ₂
    star_form_basis_eval[2] = @views hcat(local_star_form_basis_eval[1].*inv_g[:,1,1], local_star_form_basis_eval[2].*inv_g[:,1,2]).*sqrt_g
    star_form_basis_indices[2] = vcat(local_star_form_basis_indices[1], local_star_form_basis_indices[2])

    return star_form_basis_eval, star_form_basis_indices
end

# 1-forms (3 dimensions)
@doc raw"""
evaluate_hodge_star(form_space::FS, element_id::Int, xi::NTuple{3, Vector{Float64}}) where {FS <: AbstractFormSpace{3, 1}}

Evaluates the Hodge star operator ⋆ of a 1-form in 3 dimensions.

# Arguments
- `form_space::FS`: The 1-form space in 3 dimensions.
- `element_idx::Int`: Index of the element to evaluate
- `xi::NTuple{3, Vector{Float64}}`: Tuple of tensor-product coordinate vectors for evaluation points. 
    Points are the tensor product of the coordinates per dimension, therefore there will be
    ``n_{1} \times \dots \times n_{\texttt{manifold_dim}}`` points where to evaluate.

# Returns
- `local_star_form_basis_eval`: Vector of arrays containing evaluated hodge star of the basis functions.
- `local_star_form_basis_indices`: Vector of vectors containing indices of the basis functions.
"""
function evaluate_hodge_star(form_space::FS, element_id::Int, xi::NTuple{3, Vector{Float64}}) where {FS <: AbstractFormSpace{3, 1}}
    g, sqrt_g = Geometry.metric(form_space.geometry, element_id, xi)
    inv_g = mapslices(inv, g, dims=(2,3))

    local_star_form_basis_eval, local_star_form_basis_indices = evaluate(form_space, element_id, xi)
    star_form_basis_eval, star_form_basis_indices = similar(local_star_form_basis_eval), similar(local_star_form_basis_indices)

    # ⋆(α₁¹dξ₁+α₂¹dξ₂+α₃¹dξ₃) = [(α₁¹g¹¹+α₂¹g¹²+α₃¹g¹³)dξ₂∧dξ₃ + (α₁¹g²¹+α₂¹g²²+α₃¹g²³)dξ₃∧dξ₁ + (α₁¹g³¹+α₂¹g³²+α₃¹g³³)dξ₁∧dξ₂]√det(gᵢⱼ). 
    # First: (α₁¹g¹¹+α₂¹g¹²+α₃¹g¹³)dξ₂∧dξ₃
    star_form_basis_eval[1] = @views hcat(local_star_form_basis_eval[1].*inv_g[:,1,1], local_star_form_basis_eval[2].*inv_g[:,1,2], local_star_form_basis_eval[3].*inv_g[:,1,3]).*sqrt_g
    star_form_basis_indices[1] = vcat(local_star_form_basis_indices[1], local_star_form_basis_indices[2], local_star_form_basis_indices[3])

    # Second: (α₁¹g²¹+α₂¹g²²+α₃¹g²³)dξ₃∧dξ₁
    star_form_basis_eval[2] = @views hcat(local_star_form_basis_eval[1].*inv_g[:,2,1], local_star_form_basis_eval[2].*inv_g[:,2,2], local_star_form_basis_eval[3].*inv_g[:,2,3]).*sqrt_g
    star_form_basis_indices[2] = vcat(local_star_form_basis_indices[1], local_star_form_basis_indices[2], local_star_form_basis_indices[3])

    # Third: (α₁¹g³¹+α₂¹g³²+α₃¹g³³)dξ₁∧dξ₂
    star_form_basis_eval[3] = @views hcat(local_star_form_basis_eval[1].*inv_g[:,3,1], local_star_form_basis_eval[2].*inv_g[:,3,2], local_star_form_basis_eval[3].*inv_g[:,3,3]).*sqrt_g
    star_form_basis_indices[3] = vcat(local_star_form_basis_indices[1], local_star_form_basis_indices[2], local_star_form_basis_indices[3])

    return star_form_basis_eval, star_form_basis_indices
end

# 2-forms (3 dimensions)
@doc raw"""
evaluate_hodge_star(form_space::FS, element_id::Int, xi::NTuple{3, Vector{Float64}}) where {FS <: AbstractFormSpace{3, 2}}

Evaluates the Hodge star operator ⋆ of a 2-form in 3 dimensions.

# Arguments
- `form_space::FS`: The 2-form space in 3 dimensions.
- `element_idx::Int`: Index of the element to evaluate
- `xi::NTuple{3, Vector{Float64}}`: Tuple of tensor-product coordinate vectors for evaluation points. 
    Points are the tensor product of the coordinates per dimension, therefore there will be
    ``n_{1} \times \dots \times n_{\texttt{manifold_dim}}`` points where to evaluate.

# Returns
- `local_star_form_basis_eval`: Vector of arrays containing evaluated hodge star of the basis functions.
- `local_star_form_basis_indices`: Vector of vectors containing indices of the basis functions.
"""
function evaluate_hodge_star(form_space::FS, element_id::Int, xi::NTuple{3, Vector{Float64}}) where {FS <: AbstractFormSpace{3, 2}}
    g, sqrt_g = Geometry.metric(form_space.geometry, element_id, xi)
    inv_g = mapslices(inv, g, dims=(2,3))
    
    local_star_form_basis_eval, local_star_form_basis_indices = evaluate(form_space, element_id, xi)
    star_form_basis_eval, star_form_basis_indices = similar(local_star_form_basis_eval), similar(local_star_form_basis_indices)

    # ⋆(α₁²dξ₂∧dξ₃ + α₂²dξ₃∧dξ₁ + α₃²dξ₁∧dξ₂) = [(α₁²(g²²g³³-g²³g³²) + α₂²(g²³g³¹-g²¹g³³) + α₃²(g²¹g³²-g²²g³¹))dξ¹ +
    #                                            (α₁²(g³²g¹³-g³³g¹²) + α₂²(g³³g¹¹-g³¹g¹³) + α₃²(g³¹g¹²-g³²g¹¹))dξ² +
    #                                            (α₁²(g¹²g²³-g¹³g²²) + α₂²(g¹³g²¹-g¹¹g²³) + α₃²(g¹¹g²²-g¹²g²¹))dξ³ ]√det(gᵢⱼ). 
    # First: (α₁²(g²²g³³-g²³g³²) + α₂²(g²³g³¹-g²¹g³³) + α₃²(g²¹g³²-g²²g³¹))dξ¹
    star_form_basis_eval[1] = @views hcat(local_star_form_basis_eval[1].*(inv_g[:,2,2].*inv_g[:,3,3]-inv_g[:,2,3].*inv_g[:,3,2]), local_star_form_basis_eval[2].*(inv_g[:,2,3].*inv_g[:,3,1]-inv_g[:,2,1].*inv_g[:,3,3]), local_star_form_basis_eval[3].*(inv_g[:,2,1].*inv_g[:,3,2]-inv_g[:,2,2].*inv_g[:,3,1])).*sqrt_g
    star_form_basis_indices[1] = vcat(local_star_form_basis_indices[1], local_star_form_basis_indices[2], local_star_form_basis_indices[3])

    # Second: (α₁²(g³²g¹³-g³³g¹²) + α₂²(g³³g¹¹-g³¹g¹³) + α₃²(g³¹g¹²-g³²g¹¹))dξ²
    star_form_basis_eval[2] = @views hcat(local_star_form_basis_eval[1].*(inv_g[:,3,2].*inv_g[:,1,3]-inv_g[:,3,3].*inv_g[:,1,2]), local_star_form_basis_eval[2].*(inv_g[:,3,3].*inv_g[:,1,1]-inv_g[:,3,1].*inv_g[:,1,3]), local_star_form_basis_eval[3].*(inv_g[:,3,1].*inv_g[:,1,2]-inv_g[:,3,2].*inv_g[:,1,1])).*sqrt_g
    star_form_basis_indices[2] = vcat(local_star_form_basis_indices[1], local_star_form_basis_indices[2], local_star_form_basis_indices[3])

    # Third: (α₁²(g¹²g²³-g¹³g²²) + α₂²(g¹³g²¹-g¹¹g²³) + α₃²(g¹¹g²²-g¹²g²¹))dξ³
    star_form_basis_eval[3] = @views hcat(local_star_form_basis_eval[1].*(inv_g[:,1,2].*inv_g[:,2,3]-inv_g[:,1,3].*inv_g[:,2,2]), local_star_form_basis_eval[2].*(inv_g[:,1,3].*inv_g[:,2,1]-inv_g[:,1,1].*inv_g[:,2,3]), local_star_form_basis_eval[3].*(inv_g[:,1,1].*inv_g[:,2,2]-inv_g[:,1,2].*inv_g[:,2,1])).*sqrt_g
    star_form_basis_indices[3] = vcat(local_star_form_basis_indices[1], local_star_form_basis_indices[2], local_star_form_basis_indices[3])

    return star_form_basis_eval, star_form_basis_indices
end

@doc raw"""
    get_num_basis(form_space::FS) where {FS <: AbstractFormSpace{manifold_dim, form_rank, G}} where {manifold_dim, form_rank, G <: Geometry.AbstractGeometry}

Returns the number of degrees of freedom of the FormSpace `form_space`.

# Arguments
- `form_space::AbstractFormSpace`: The FormSpace to compute the number of degrees of freedom.

# Returns
- `::Int`: The total number of degrees of freedom of the space.
"""
function get_num_basis(form_space::FS) where {FS <: AbstractFormSpace{manifold_dim, form_rank, G}} where {manifold_dim, form_rank, G <: Geometry.AbstractGeometry}
    # Get the total number of dofs for the form space by looping 
    # over all the FEM spaces. Each FEM space is associated to a 
    # component of the form: 0- and n-forms have one component, and 
    # 1- and 2-forms have multiple components, depending on the 
    # dimension of the underlying manifold in which they are defined
    form_space_n_dofs = 0  # initialize the number of dofs to 0, to add the ones from each FEM space
    
    # Loop over the spaces and add their number of dofs
    for space in form_space.fem_space
        form_space_n_dofs += FunctionSpaces.get_num_basis(space)
    end

    return form_space_n_dofs
end