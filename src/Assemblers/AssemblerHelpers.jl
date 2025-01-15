function L2_norm(u, ∫)
    norm = 0.0
    for el_id ∈ 1:Geometry.get_num_elements(u.geometry)
        norm += Forms.evaluate_inner_product(u, u, el_id, ∫)[3][1]
    end
    return sqrt(norm)
end