include("HelperFunctions.jl")
# AbstractFiniteElementSpace: Common functionality accross FEM spaces
include("AbstractFiniteElementSpace.jl")
# Extraction operator: functionaly for extraction operator
include("ExtractionOperator.jl")
# Two scale relations
include("TwoScaleRelations/TwoScaleRelations.jl")

# Univariate space implementations
include("UnivariateSplineSpaces.jl")

# General finite element space categories that can be composed to build more complex spaces:
# 1. rational version of finite element spaces
include("RationalFiniteElementSpaces.jl")
# 2. multi-patch function spaces
include("UnstructuredSpaces.jl")
# 3. tensor-product spaces
include("TensorProductSpaces.jl")
# 4. hierarchical function spaces
include("HierarchicalFiniteElementSpaces.jl")

