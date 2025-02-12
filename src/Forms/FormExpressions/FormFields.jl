############################################################################################
#                                        Structures                                        #
############################################################################################

@doc raw"""
    FormField{manifold_dim, form_rank, G, FS} <: AbstractFormField{manifold_dim, form_rank, 0, G}

Represents a differential form field.

# Fields
- `form_space::FS`: The form space associated with this field
- `coefficients::Vector{Float64}`: Coefficients of the form field
- `label::String`: Label for the form field

# Type parameters
- `manifold_dim`: Dimension of the manifold
- `form_rank`: Rank of the differential form
- `G`: Type of the geometry
- `FS`: Type of the form space

# Inner Constructors
- `FormSpace(form_rank::Int, geometry::G, fem_space::Tuple{F})`: Constructor for 0-forms and n-forms
- `FormSpace(form_rank::Int, geometry::G, fem_space::Tuple{F_dξ, F_dη})`: Constructor for 1-forms in 2D
- `FormSpace(form_rank::Int, geometry::G, fem_space::Tuple{F_dξ, F_dη, F_dζ})`: Constructor for 1-forms and 2-forms in 3D
"""
struct FormField{manifold_dim, form_rank, G, FS} <: AbstractFormField{manifold_dim, form_rank, G}
    geometry::G
    form_space::FS
    coefficients::Vector{Float64}
    label::String

    """
        FormField(form_space::FS, label::String) where {FS <: AbstractFormSpace{manifold_dim, form_rank, 1, G}} where {manifold_dim, form_rank, G <: Geometry.AbstractGeometry{manifold_dim}}

    Construct a FormField with zero coefficients.

    # Arguments
    - `form_space::FS`: The form space
    - `label::String`: Label for the form field

    # Returns
    - `FormField`: A new FormField instance with zero coefficients
    """
    function FormField(form_space::FS, label::String) where {manifold_dim, form_rank, G <: Geometry.AbstractGeometry{manifold_dim}, FS <: AbstractFormSpace{manifold_dim, form_rank, G}}
        # Here we just generate the coefficients as zeros, i.e., initialize the coefficients
        n_dofs = get_num_basis(form_space)
        coefficients = zeros(Float64, n_dofs)
        label = label * Subscripts.super("$form_rank")
        new{manifold_dim, form_rank, G, FS}(get_geometry(form_space), form_space, coefficients, label)
    end
end

struct AnalyticalFormField{manifold_dim, form_rank, G, E} <: AbstractFormField{manifold_dim, form_rank, G}
    geometry::G
    expression::E
    label::String

    @doc raw"""
        AnalyticalFormField(form_rank::Int, expression::E, geometry::G, label::String) where {E <: Function, G <: Geometry.AbstractGeometry{manifold_dim}} where {manifold_dim}

    Construct a FormField with zero coefficients.

    # Arguments
    - `geometry::G`: The underlying geometry that defines the manifold where this form is defined.
    - `expression:E`: The expression that defined the forms. These expressions must be evaluated at points with physical coordinates `x`.
                      The dimension, $n$, of the physical domain must be the same as the dimension of the image of the geometry. The 
                      Input to expression must must have the same format as the ouput of the evaluation of geometries, i.e, an $m \times n$ matrix
                      containing the $m$ points with their $n$ coordinates.
                      The components must be returned in a vector of vectors. Each element of the outer vector is associated to a component  
                      of the form. The elements of the inner vector are associated to the evaluation points. The components must, as all 
                      other forms, be given for the following form basis
                            0-forms: single component.
                            1-forms:
                                2D: ``[\mathrm{d}x_{1}, \mathrm{d}x_{2}]``
                                3D: ``[\mathrm{d}x_{1}, \mathrm{d}x_{2}, \mathrm{d}x_{3}]``
                            2-forms:
                                3D: ``[\mathrm{d}x_{2}\wedge\mathrm{d}x_{3}, \mathrm{d}x_{3}\wedge\mathrm{d}x_{1}, \mathrm{d}x_{1}\wedge\mathrm{d}x_{2}]``
                            n-forms: single component
    - `label::String`: Label for the form field

    # Returns
    - `AnalyticalFormField`: A new FormField instance with zero coefficients
    """
    function AnalyticalFormField(form_rank::Int, expression::E, geometry::G, label::String) where {E <: Function, G <: Geometry.AbstractGeometry{manifold_dim}} where {manifold_dim}
        label = label * Subscripts.super("$form_rank")
        new{manifold_dim, form_rank, G, E}(geometry, expression, label)
    end
end

############################################################################################
#                                    Evaluation methods                                    #
############################################################################################

@doc raw"""
    evaluate(form::FormField{manifold_dim, form_rank, G, FS}, element_idx::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, form_rank, expression_rank, G <: Geometry.AbstractGeometry{manifold_dim}, FS <: AbstractFormSpace{manifold_dim, form_rank, G}}

Evaluate a FormField at given points.

# Arguments
- `form::FormField`: The form field to evaluate.
- `element_idx::Int`: Index of the element to evaluate.
- `xi::NTuple{manifold_dim, Vector{Float64}}`: Tuple of tensor-product coordinate vectors for evaluation points. 
        Points are the tensor product of the coordinates per dimension, therefore there will be
        ``n_{1} \times \dots \times n_{\texttt{manifold_dim}}`` points where to evaluate.

# Returns
- `form_eval::Vector{Vector{Float64}}`: Evaluated form field.
        `form_eval[i][j]` is the component `i`` of the form evaluated at the tensor
        product point `j`. The components are, by convention:
        0-forms: single component.
        1-forms:
            2D: ``[\mathrm{d}\xi_{1}, \mathrm{d}\xi_{2}]``
            3D: ``[\mathrm{d}\xi_{1}, \mathrm{d}\xi_{2}, \mathrm{d}\xi_{3}]``
        2-forms:
            3D: ``[\mathrm{d}\xi_{2}\wedge\mathrm{d}\xi_{3}, \mathrm{d}\xi_{3}\wedge\mathrm{d}\xi_{1}, \mathrm{d}\xi_{1}\wedge\mathrm{d}\xi_{2}]``
        n-forms: single component
- `form_indices`: Vector of length `n_form_components`, where each element is a Vector{Int} of length 1 with value 1, since for a 
        `FormField` there is only one `basis`. This is done for consistency with `FormBasis`.

# Sizes
- `form_eval`: Vector of length `n_form_components`, where each element is a Vector{Float64} of length `n_evaluation_points`
        `n_evaluation_points = n_1 * ... * n_n`, where `n_i` is the number of points in the component `i` of the tensor product Tuple.
"""
function evaluate(form::FormField{manifold_dim, form_rank, G, FS}, element_idx::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, form_rank, G <: Geometry.AbstractGeometry{manifold_dim}, FS <: AbstractFormSpace{manifold_dim, form_rank, G}}
    n_form_components = binomial(manifold_dim, form_rank)
    form_basis_eval, form_basis_indices = evaluate(form.form_space, element_idx, xi)

    form_eval = Vector{Vector{Float64}}(undef, n_form_components)

    for form_component_idx in 1:n_form_components
        form_eval[form_component_idx] = form_basis_eval[form_component_idx] * form.coefficients[form_basis_indices[1]]  # note that FormSpace has only one set of basis indices
    end

    # We need to wrap form_basis_indices in [] to return a vector of vector to allow multi-indexed expressions, like wedges
    return form_eval, [[1]]
end


@doc raw"""
    evaluate(form::FormField{manifold_dim, form_rank, G, FS}, element_idx::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, form_rank, G <: Geometry.AbstractGeometry{manifold_dim}, FS <: AbstractFormSpace{manifold_dim, form_rank, G}}

Evaluate a FormField at given points.

# Arguments
- `form::AbstractFormField`: The form field to evaluate.
- `element_idx::Int`: Index of the element to evaluate.
- `xi::NTuple{manifold_dim, Vector{Float64}}`: Tuple of tensor-product coordinate vectors for evaluation points. 
        Points are the tensor product of the coordinates per dimension, therefore there will be
        ``n_{1} \times \dots \times n_{\texttt{manifold_dim}}`` points where to evaluate.

# Returns
- `form_eval::Vector{Vector{Float64}}`: Evaluated analytical form field.
        `form_eval[i][j]` is the component `i`` of the form evaluated at the tensor
        product point `j`. The components are, by convention:
        0-forms: single component.
        1-forms:
            2D: ``[\mathrm{d}\xi_{1}, \mathrm{d}\xi_{2}]``
            3D: ``[\mathrm{d}\xi_{1}, \mathrm{d}\xi_{2}, \mathrm{d}\xi_{3}]``
        2-forms:
            3D: ``[\mathrm{d}\xi_{2}\wedge\mathrm{d}\xi_{3}, \mathrm{d}\xi_{3}\wedge\mathrm{d}\xi_{1}, \mathrm{d}\xi_{1}\wedge\mathrm{d}\xi_{2}]``
        n-forms: single component

        Note: The forms are pulledback via the geometry mapping, therefore they are evaluated on the computational manifold.
        
- `form_indices`: Vector of length `n_form_components`, where each element is a Vector{Int} of length 1 with value 1, since for an 
        `AnalyticalFormField` there is only one `basis`. This is done for consistency with `FormBasis`.

# Sizes
- `form_eval`: Vector of length `n_form_components`, where each element is a Vector{Float64} of length `n_evaluation_points`
        `n_evaluation_points = n_1 * ... * n_n`, where `n_i` is the number of points in the component `i` of the tensor product Tuple.
"""
function evaluate(form::AnalyticalFormField{manifold_dim, 0, G, E}, element_idx::Int, ξ::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, G <: Geometry.AbstractGeometry{manifold_dim}, E <: Function}
    x = Geometry.evaluate(form.geometry, element_idx, ξ)
    
    form_eval = form.expression(x)
    
    # We need to wrap form_basis_indices in [] to return a vector of vector to allow multi-indexed expressions, like wedges
    return form_eval, [[1]]
end

function evaluate(form::AnalyticalFormField{manifold_dim, manifold_dim, G, E}, element_idx::Int, ξ::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, G <: Geometry.AbstractGeometry{manifold_dim}, E <: Function}
    if Geometry.get_image_dim(form.geometry) != manifold_dim
        throw("Image manifold must have the same dimension as the domain manifold.")
    end
    
    x = Geometry.evaluate(form.geometry, element_idx, ξ)
    J = Geometry.jacobian(form.geometry, element_idx, ξ)  # Jₖⱼ = ∂Φᵏ\∂ξⱼ
    form_eval = form.expression(x)

    form_eval[1][:] .*= LinearAlgebra.det.(eachslice(J; dims=1))
    
    # We need to wrap form_basis_indices in [] to return a vector of vector to allow multi-indexed expressions, like wedges
    return form_eval, [[1]]
end

function evaluate(form::AnalyticalFormField{manifold_dim, 1, G, E}, element_idx::Int, ξ::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, G <: Geometry.AbstractGeometry{manifold_dim}, E <: Function}
    
    x = Geometry.evaluate(form.geometry, element_idx, ξ)
    J = Geometry.jacobian(form.geometry, element_idx, ξ)  # Jₖⱼ = ∂Φᵏ\∂ξⱼ
    form_eval = form.expression(x) # size: num_points x image_dim

    num_eval_points = size(x,1)
    image_dim = length(form_eval)
    form_pullback = Vector{Vector{Float64}}(undef, manifold_dim)
    for j = 1:manifold_dim
        form_pullback[j] = zeros(num_eval_points)
    end
    a = zeros(num_eval_points, manifold_dim)
    for j = 1:image_dim
        for i = 1:num_eval_points
            a[i,:] .+= form_eval[j][i] .* J[i,j,:]
        end
    end
    for j= 1:manifold_dim
        form_pullback[j] = a[:,j]
    end
    
    # We need to wrap form_basis_indices in [] to return a vector of vector to allow multi-indexed expressions, like wedges
    return form_pullback, [[1]]
end


