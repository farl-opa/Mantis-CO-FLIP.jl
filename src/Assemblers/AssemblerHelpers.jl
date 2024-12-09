function L2_norm(u, ∫)
    norm = 0.0
    for el_id ∈ 1:Geometry.get_num_elements(u.geometry)
        inner_prod = SparseArrays.sparse(Forms.evaluate_inner_product(u, u, el_id, ∫)...)
        norm += inner_prod[1,1]
    end
    return sqrt(norm)
end