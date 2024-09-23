"""
    MixedFormField{num_forms, F}

A container for multiple form fields. This is useful for mixed finite element solutions and forcings.
"""
struct MixedFormField{num_forms, F}
    form_fields::F

    function MixedFormField(form_fields::F) where {num_forms, F <: NTuple{num_forms, Union{Nothing,AbstractFormField}}}
        new{num_forms, F}(form_fields)
    end
end

function Base.getindex(mixed_field::MixedFormField{num_forms, F}, idx::Int) where {num_forms, F}
    return mixed_field.form_fields[idx]
end

function Base.setindex(_::MixedFormField{num_forms, F}, idx::Int) where {num_forms, F}
    throw(ArgumentError("Cannot set a form space in a mixed form space."))
end

"""
    MixedFormSpace{num_forms, F}

A container for multiple form spaces. This is useful for mixed finite element methods.
"""
struct MixedFormSpace{num_forms, F}
    form_spaces::F

    function MixedFormSpace(form_spaces::F) where {num_forms, F <: NTuple{num_forms, FormSpace}}
        new{num_forms, F}(form_spaces)
    end
end

function Base.getindex(mixed_space::MixedFormSpace{num_forms, F}, idx::Int) where {num_forms, F}
    return mixed_space.form_spaces[idx]
end

function Base.setindex(_::MixedFormSpace{num_forms, F}, idx::Int) where {num_forms, F}
    throw(ArgumentError("Cannot set a form space in a mixed form space."))
end

"""
    get_num_elements(mixed_space::MixedFormSpace)

Get the number of elements for the geometry underlying all input form spaces. Assumes that all form spaces have the same geometry and therefore only accesses the geometry of the first form space. A check will be added once we have a proper struct for meshes.

# Arguments
- mixed_space::MixedFormSpace: The mixed form space.

# Returns
- num_elements::Int: The number of elements in the geometry of the first form space.
"""
function get_num_elements(mixed_space::MixedFormSpace)
    geo = Forms.get_geometry(mixed_space.form_spaces[1])
    return Geometry.get_num_elements(geo)
end

"""
    get_num_basis(mixed_space::MixedFormSpace)

Get the number of basis functions for the mixed form space. This is the sum of the number of basis functions for all component spaces.

# Arguments
- mixed_space::MixedFormSpace: The mixed form space.

# Returns
- num_basis::Int: The number of basis functions for the trial space.
"""
function get_num_basis(mixed_space::MixedFormSpace)
    num_basis = 0
    for form_space in mixed_space.form_spaces
        num_basis += Forms.get_num_basis(form_space)
    end
    return num_basis
end

"""
    get_max_local_dim(mixed_space::MixedFormSpace)

Get the maximum local dimension for the mixed form space. This is the sum of the maximum local dimensions for the component spaces.

# Arguments
- mixed_space::MixedFormSpace: The mixed form space.

# Returns
- max_local_dim::Int: The maximum local dimension.
"""
function get_max_local_dim(mixed_space::MixedFormSpace)
    max_local_dim = 0
    for form_space in mixed_space.form_spaces
        max_local_dim += Forms.get_max_local_dim(form_space)
    end
    return max_local_dim
end

"""
    build_form_fields(mixed_space::MixedFormSpace, coeffs::Vector{Float64})

Build the form fields for the mixed form space given a set of coefficients. This is done by assigning the right set of coefficients to the right form space.
"""
function build_form_fields(mixed_space::MixedFormSpace{num_forms, F}, coeffs::Vector{Float64}; labels::Union{Nothing,NTuple{num_forms,String}}=nothing) where {num_forms, F <: NTuple{num_forms, FormSpace}}
    if length(coeffs) != get_num_basis(mixed_space)
        throw(ArgumentError("The number of coefficients does not match the number of basis functions in the mixed form space."))
    end
    
    form_fields = Vector{FormFields}(undef, num_forms)
    start_idx = 1
    for form_idx ∈ 1:num_forms
        num_coeffs = Forms.get_num_basis(mixed_space.form_spaces[form_idx])
        if isnothing(labels)
            form_fields[form_idx] = Mantis.Forms.FormField(mixed_space.form_spaces[form_idx], "ζ" * string(form_idx))
        else
            form_fields[form_idx] = Mantis.Forms.FormField(mixed_space.form_spaces[form_idx], labels[form_idx])
        end
        form_fields[form_idx].coefficients = coeffs[start_idx:start_idx+num_coeffs-1]
        start_idx += num_coeffs
    end
    return MixedFormField(Tuple(form_fields))
end