"""
    SumSpace{manifold_dim, num_components, F} <:
       AbstractFESpace{manifold_dim, num_components, num_patches}

A multi-valued space that is the sum of `num_components` input scalar function spaces.
Each scalar function space contributes to each component of the multi-valued space.

# Fields
- `component_spaces::F`: Tuple of `num_components` scalar function spaces
"""
struct SumSpace{manifold_dim, num_components, num_patches, F} <:
       AbstractFESpace{manifold_dim, num_components, num_patches}
    component_spaces::F
    space_dim::Int

    function SumSpace(
        component_spaces::F, space_dim::Int
    ) where {
        manifold_dim,
        num_components,
        num_patches,
        F <: NTuple{num_components, AbstractFESpace{manifold_dim, 1, num_patches}},
    }
        return new{manifold_dim, num_components, num_patches, F}(component_spaces, space_dim)
    end
end


get_num_basis(space::SumSpace) = space.space_dim
