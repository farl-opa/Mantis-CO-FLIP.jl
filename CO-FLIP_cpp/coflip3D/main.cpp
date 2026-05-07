#include <cmath>
#include "../include/array.h"
#include <iostream>
#include "COFLIPSolver.h"
#include <boost/filesystem.hpp>

void makeCylinderUp(openvdb::Grid<openvdb::tree::Tree4<MyReal>::Type>::Ptr grid, MyReal radius, MyReal thickness, const Vec<3, MyReal>& center, const openvdb::CoordBBox& indexBB, double h)
{
    typename openvdb::Grid<openvdb::tree::Tree4<MyReal>::Type>::Accessor accessor = grid->getAccessor();

    for (openvdb::Int32 i = indexBB.min().x(); i < indexBB.max().x(); ++i) {
        for (openvdb::Int32 j = indexBB.min().y(); j < indexBB.max().y(); ++j) {
            for (openvdb::Int32 k = indexBB.min().z(); k < indexBB.max().z(); ++k) {
                // transform point (i, j, k) of index space into world space
                openvdb::Vec3d p(i * h - center[0], j * h - center[1], k * h - center[2]);

                Vec<2, MyReal> d = Vec<2, MyReal>(std::sqrt(p.x() * p.x() + p.z() * p.z()), std::abs(p.y())) - Vec<2, MyReal>(radius, thickness);
                Vec<2, MyReal> d_max(std::max(d[0], (MyReal)0.0f), std::max(d[1], (MyReal)0.0f));
                MyReal distance = std::min(std::max(d[0], d[1]), (MyReal)0.0f) + std::sqrt(d_max[0] * d_max[0] + d_max[1] * d_max[1]);

                accessor.setValue(openvdb::Coord(i, j, k), distance);
            }
        }
    }

    grid->setTransform(openvdb::math::Transform::createLinearTransform(h));
}

void makeCylinder(openvdb::Grid<openvdb::tree::Tree4<MyReal>::Type>::Ptr grid, MyReal radius, MyReal thickness, const Vec<3, MyReal>& center, const openvdb::CoordBBox& indexBB, double h)
{
    typename openvdb::Grid<openvdb::tree::Tree4<MyReal>::Type>::Accessor accessor = grid->getAccessor();

    for (openvdb::Int32 i = indexBB.min().x(); i < indexBB.max().x(); ++i) {
        for (openvdb::Int32 j = indexBB.min().y(); j < indexBB.max().y(); ++j) {
            for (openvdb::Int32 k = indexBB.min().z(); k < indexBB.max().z(); ++k) {
                // transform point (i, j, k) of index space into world space
                openvdb::Vec3d p(i * h - center[0], j * h - center[1], k * h - center[2]);

                Vec<2, MyReal> d = Vec<2, MyReal>(std::sqrt(p.z() * p.z() + p.y() * p.y()), std::abs(p.x())) - Vec<2, MyReal>(radius, thickness);
                Vec<2, MyReal> d_max(std::max(d[0], (MyReal)0.0f), std::max(d[1], (MyReal)0.0f));
                MyReal distance = std::min(std::max(d[0], d[1]), (MyReal)0.0f) + std::sqrt(d_max[0]*d_max[0]+d_max[1]*d_max[1]);
                // compute level set function value
                // MyReal distance = sqrt(p.z() * p.z() + p.y() * p.y()) - radius;

                accessor.setValue(openvdb::Coord(i, j, k), distance);
            }
        }
    }

    grid->setTransform(openvdb::math::Transform::createLinearTransform(h));
}

void makeSlab(openvdb::Grid<openvdb::tree::Tree4<MyReal>::Type>::Ptr grid, MyReal radius, MyReal innerRadius, MyReal thickness, const Vec<3, MyReal>& center, const openvdb::CoordBBox& indexBB, double h)
{
    typename openvdb::Grid<openvdb::tree::Tree4<MyReal>::Type>::Accessor accessor = grid->getAccessor();

    for (openvdb::Int32 i = indexBB.min().x(); i < indexBB.max().x(); ++i) {
        for (openvdb::Int32 j = indexBB.min().y(); j < indexBB.max().y(); ++j) {
            for (openvdb::Int32 k = indexBB.min().z(); k < indexBB.max().z(); ++k) {
                // transform point (i, j, k) of index space into world space
                openvdb::Vec3d p(i * h - center[0], j * h - center[1], k * h - center[2]);

                Vec<2, MyReal> d = Vec<2, MyReal>(std::max(std::abs(p.z()),std::abs(p.y())), std::abs(p.x())) - Vec<2, MyReal>(radius, thickness);
                Vec<2, MyReal> d2 = Vec<2, MyReal>(-1.0f);
                if (innerRadius > 0.0f)
                    d2 = Vec<2, MyReal>(innerRadius - std::sqrt(p.z() * p.z() + p.y() * p.y()), abs(p.x()) - thickness);
                
                Vec<2, MyReal> d_max = Vec<2, MyReal>(std::max(d[0], std::max(d2[0], (MyReal)0.0f)),
                                    std::max(d[1], std::max(d2[1], (MyReal)0.0f)));
                MyReal distance = std::min(std::max(std::max(d[0], d[1]),d2[0]), (MyReal)0.0f) + std::sqrt(d_max[0] * d_max[0] + d_max[1] * d_max[1]);
                // compute level set function value
                // MyReal distance = std::sqrt(p.z() * p.z() + p.y() * p.y()) - radius;

                accessor.setValue(openvdb::Coord(i, j, k), distance);
            }
        }
    }

    grid->setTransform(openvdb::math::Transform::createLinearTransform(h));
}

int main(int argc, char** argv) {
    uint ni = 0;
    uint nj = 0;
    uint nk = 0;
    uint total_frame = 0;
    int _baseres = 0;
    MyReal L = 0;
    MyReal h = 0;
    MyReal dt = 0;
    MyReal viscosity = 0;
    MyReal half_width = 0;
    MyReal smoke_rise = 0;
    MyReal smoke_drop = 0;
    MyReal framerate = 24.f;
    MyReal substeps = 1.f;
    Scheme sim_scheme;
    Experiment sim_experiment;
    bool do_vel_advection_only;
    int delayed_reinit_num = 1;
    std::string filepath = "../Out_3D/";
    int sim_name = 0;
    int experiment_name;
    if (argc != 3)
    {
        std::cout << "Please specify correct parameters!" << std::endl;
        std::cout << "inputs: [Method] [Experiment]" << std::endl;
        std::cout << "Valid method numbers are [0-4] for 0: POLYPIC, 1: POLYFLIP, 2: R_POLYFLIP, 3: CF_POLYFLIP, 4: CO_FLIP" << std::endl;
        std::cout << "Valid experiment numbers are [0-8] for 0: trefoil knot, 1: leapfrogging, 2: unkot, 3: twisted torus, 4: smoke plume, 5: pyroclastic, 6: ink jet, 7: rocket, 8: spot obstacle" << std::endl;
        exit(0);
    }
    sim_name = atoi(argv[1]);
    experiment_name = atoi(argv[2]);
    if (sim_name >= 5)
    {
        std::cout << "Please enter valid method number!" << std::endl;
        std::cout << "inputs: [Method] [Experiment]" << std::endl;
        std::cout << "Valid method numbers are [0-4] for 0: POLYPIC, 1: POLYFLIP, 2: R_POLYFLIP, 3: CF_POLYFLIP, 4: CO_FLIP" << std::endl;
        std::cout << "Valid experiment numbers are [0-8] for 0: trefoil knot, 1: leapfrogging, 2: unkot, 3: twisted torus, 4: smoke plume, 5: pyroclastic, 6: ink jet, 7: rocket, 8: spot obstacle" << std::endl;
        exit(0);
    }
    if (experiment_name >= 9)
    {
        std::cout << "Please enter valid experiment number!" << std::endl;
        std::cout << "inputs: [Method] [Experiment]" << std::endl;
        std::cout << "Valid method numbers are [0-4] for 0: POLYPIC, 1: POLYFLIP, 2: R_POLYFLIP, 3: CF_POLYFLIP, 4: CO_FLIP" << std::endl;
        std::cout << "Valid experiment numbers are [0-8] for 0: trefoil knot, 1: leapfrogging, 2: unkot, 3: twisted torus, 4: smoke plume, 5: pyroclastic, 6: ink jet, 7: rocket, 8: spot obstacle" << std::endl;
        exit(0);
    }
    omp_set_num_threads(24);
    Eigen::setNbThreads(24);
    Eigen::initParallel();
    std::cout << "Eigen thread count: " << Eigen::nbThreads() << std::endl;

    sim_scheme = static_cast<Scheme>(sim_name);
    sim_experiment = static_cast<Experiment>(experiment_name);
    delayed_reinit_num = (sim_scheme == Scheme::CO_FLIP) ? 50 : 
                        ((sim_scheme == Scheme::R_POLYFLIP || 
                          sim_scheme == Scheme::CF_POLYFLIP || 
                          sim_scheme == Scheme::POLYFLIP) ? 5 : 1);
    do_vel_advection_only = (sim_experiment == Experiment::INK_JET || 
                             sim_experiment == Experiment::PYROCLASTIC ||
                             sim_experiment == Experiment::SMOKE_PLUME ||
                             sim_experiment == Experiment::ROCKET) ? false : true;
    _baseres = 64;
    framerate = 24;

    std::vector<Emitter> emitter_list;
    std::vector<Boundary> boundary_list;

    MyReal _theta = M_PI_2, _phi = 0.f;
    std::string filepathVelField = "", filepathDensityRhoField = "", filepathDensityTempField = "";
    int pp_repeat_count = 1;
    bool set_velocity_inflow = false;
    bool set_init_vel = false;
    bool set_init_vel_from_emitter = false;
    bool is_fixed_domain = true;
    bool use_DEC_diagonal_hodge_star = false;
    MyReal inflow_vel = 1.f;
    MyReal adaptive_reset_cutoff = 3.;
    int N = 16;
    bool do_uniform_particle_seeding = false;
    bool do_particle_sample_after_first = false;
    bool is_matrix_small_enough = _baseres <= 75;
    switch(experiment_name)
    {
        case 0:
        {
            adaptive_reset_cutoff = 1;
            delayed_reinit_num = (sim_scheme == Scheme::CO_FLIP) ? 200 : delayed_reinit_num;
            // simulation resolution
            int baseres = _baseres;
            ni = baseres;
            nj = baseres;
            nk = baseres;
            substeps = 3;
            total_frame = 171; total_frame *= substeps;//sim_name != 2 ? 2 : 1;
            // length in x direction
            L = 5.0f;
            // grid size for simulation
            h = L / ni;
            // time step
            MyReal deltaT = 1.f / framerate;
            dt = deltaT / substeps;
            // smoke properties
            smoke_rise = 0.f;
            smoke_drop = 0.f;
            viscosity = 0.f;// 1e-6;
            // levelset half width, used when blending semi-lagrangian result near the boundary
            half_width = 3.f;
            filepathVelField = std::string("../modelData/TrefoilKnot/trefoilKnotStaggeredVelField_") + std::to_string(_baseres) + std::string("cubed.vdb");
        }
        break;
        case 1:
        {
            adaptive_reset_cutoff = 1.5;
            N = 27;
            delayed_reinit_num = (sim_scheme == Scheme::CO_FLIP) ? 200 : delayed_reinit_num;
            // simulation resolution
            int baseres = _baseres;
            ni = baseres * 2;
            nj = baseres;
            nk = baseres;
            substeps = 3;
            total_frame = 801; total_frame *= substeps;
            // length in x direction
            L = 10.0f;
            // grid size for simulation
            h = L / ni;
            // time step
            MyReal deltaT = 1.f / framerate;
            dt = deltaT / substeps;
            // smoke properties
            smoke_rise = 0.f;
            smoke_drop = 0.f;
            viscosity = 0.f;// 1e-6;
            // levelset half width, used when blending semi-lagrangian result near the boundary
            half_width = 3.f;
            filepathVelField = std::string("../modelData/Leapfrog/leapfrogStaggeredVelField_") + std::to_string(_baseres) + std::string("cubed.vdb");
        }
        break;
        case 2:
        {
            adaptive_reset_cutoff = 1;
            delayed_reinit_num = (sim_scheme == Scheme::CO_FLIP) ? 250 : delayed_reinit_num;
            // simulation resolution
            int baseres = _baseres;
            ni = baseres;
            nj = baseres;
            nk = baseres;
            substeps = 2;
            total_frame = 250; total_frame *= substeps;//sim_name != 2 ? 2 : 1;
            // length in x direction
            L = 5.0f;
            // grid size for simulation
            h = L / ni;
            // time step
            MyReal deltaT = 1.f / framerate;
            dt = deltaT / substeps;
            // smoke properties
            smoke_rise = 0.f;
            smoke_drop = 0.f;
            viscosity = 0.f;// 1e-6;
            // levelset half width, used when blending semi-lagrangian result near the boundary
            half_width = 3.f;
            filepathVelField = std::string("../modelData/Unknot1_5/unknot1_5StaggeredVelField_") + std::to_string(_baseres) + std::string("cubed.vdb");
        }
        break;
        case 3:
        {
            adaptive_reset_cutoff = 1;
            delayed_reinit_num = (sim_scheme == Scheme::CO_FLIP) ? 250 : delayed_reinit_num;
            // simulation resolution
            int baseres = _baseres;
            ni = baseres;
            nj = baseres;
            nk = baseres;
            substeps = 2;
            total_frame = 351; total_frame *= substeps;
            // length in x direction
            L = 5.0f;
            // grid size for simulation
            h = L / ni;
            // time step
            MyReal deltaT = 1.f / framerate;
            dt = deltaT / substeps;
            // smoke properties
            smoke_rise = 0.f;
            smoke_drop = 0.f;
            viscosity = 0.f;// 1e-6;
            // levelset half width, used when blending semi-lagrangian result near the boundary
            half_width = 3.f;
            filepathVelField = std::string("../modelData/TwistedTorus/twistedTorusStaggeredVelField_") + std::to_string(_baseres) + std::string("cubed.vdb");
        }
        break;
        case 4:
        {
            do_uniform_particle_seeding = true;
            do_particle_sample_after_first = true;
            adaptive_reset_cutoff = 1;
            N = 8;
            delayed_reinit_num = (sim_scheme == Scheme::CO_FLIP) ? 50 : delayed_reinit_num;
            // simulation resolution
            int baseres = _baseres;
            ni = baseres;
            nj = baseres * 2;
            nk = baseres;
            substeps = 8;
            total_frame = 91; total_frame *= substeps;
            // length in y direction
            L = 5.0f;
            // grid size for simulation
            h = L / ni;
            // time step
            MyReal deltaT = 1.f / framerate;
            dt = deltaT / substeps;
            // smoke properties
            smoke_rise = 1.0f;
            smoke_drop = 0.f;
            viscosity = 1e-4;
            // levelset half width, used when blending semi-lagrangian result near the boundary
            half_width = 3.f;
            auto vel_func_a = [&](Vec<3, MyReal> pos)
            {
                return Vec<3, MyReal>(0.0);
            };

            MyReal radius = 10.f * L / 128.f;
            Vec<2, MyReal> offset(-20.f * L / 128.f);
            MyReal height = 25.f * L / 128.f;
            MyReal density_to_add = 1.f / substeps;
            MyReal fixed_h = L/64.;
            openvdb::Grid<openvdb::tree::Tree4<MyReal>::Type>::Ptr sphere_sdf_a = openvdb::tools::createLevelSetSphere<openvdb::Grid<openvdb::tree::Tree4<MyReal>::Type>>(radius, openvdb::math::Vec3<MyReal>(L / 2.0f + offset[0], height, L / 2.0f + offset[1]), fixed_h, half_width);
            Emitter e_sphere_a(total_frame, density_to_add, density_to_add, Vec<3, MyReal>(0.f, 0.f, 0.f), sphere_sdf_a, [](MyReal frame)->Vec<3, MyReal> {return Vec<3, MyReal>(0.f, 0.f, 0.f); }, vel_func_a, false);
            emitter_list.push_back(e_sphere_a);

            _theta = M_PI_2 * 0.9f;
            _phi = M_PI_2 * 0.6f;
        }
        break;
        case 5:
        {
            do_uniform_particle_seeding = true;
            do_particle_sample_after_first = true;
            adaptive_reset_cutoff = 1;
            N = 8;
            delayed_reinit_num = (sim_scheme == Scheme::CO_FLIP) ? 50 : delayed_reinit_num;
            // simulation resolution
            int baseres = _baseres;
            ni = baseres;
            nj = baseres * 2;
            nk = baseres;
            substeps = 8;
            total_frame = 106; total_frame *= substeps;
            // length in y direction
            L = 5.0f;
            // grid size for simulation
            h = L / ni;
            // time step
            MyReal deltaT = 1.f / framerate;
            dt = deltaT / substeps;
            // smoke properties
            smoke_rise = 0.5f;
            smoke_drop = 0.f;
            viscosity = 1e-4;
            // levelset half width, used when blending semi-lagrangian result near the boundary
            half_width = 3.f;
            auto vel_func_a = [&](Vec<3, MyReal> pos)
            {
                return Vec<3, MyReal>(0.f);
            };

            MyReal radius = 40.f * L / 128.f;
            Vec<2, MyReal> offset(-5.f * L / 128.f);
            MyReal height = 5.f * L / 128.f;
            MyReal density_to_add = 1.f / substeps;

            MyReal fixed_h = L/64.;
            openvdb::Grid<openvdb::tree::Tree4<MyReal>::Type>::Ptr cylinder_sdf = openvdb::Grid<openvdb::tree::Tree4<MyReal>::Type>::create(20.0);
            openvdb::CoordBBox indexBB(openvdb::Coord(0, 0, 0), openvdb::Coord(ni, nj, nk));
            makeCylinderUp(cylinder_sdf, radius, height, Vec<3, MyReal>(L / 2.0f + offset[0], height, L / 2.0f + offset[1]), indexBB, fixed_h);
            Emitter e_cylinder_a(total_frame, density_to_add, density_to_add, Vec<3, MyReal>(0.f), cylinder_sdf, [](MyReal frame)->Vec<3, MyReal> {return Vec<3, MyReal>(0.f, 0.f, 0.f); }, vel_func_a, false, true);
            emitter_list.push_back(e_cylinder_a);

            _theta = M_PI_2 * 0.9f;
            _phi = M_PI_2 * 0.6f;
        }
        break;
        case 6:
        {
            do_uniform_particle_seeding = true;
            do_particle_sample_after_first = true;
            adaptive_reset_cutoff = 1;
            N = 8;
            delayed_reinit_num = (sim_scheme == Scheme::CO_FLIP) ? 50 : delayed_reinit_num;
            // simulation resolution
            int baseres = _baseres;
            ni = baseres;
            nj = baseres * 2;
            nk = baseres;
            substeps = 2;
            total_frame = 401; total_frame *= substeps;
            // length in y direction
            L = 5.0f;
            // grid size for simulation
            h = L / ni;
            // time step
            MyReal deltaT = 1.f / framerate;
            dt = deltaT / substeps;
            // smoke properties
            smoke_rise = 0.f;
            smoke_drop = 0.1f;
            viscosity = 1e-4;
            // levelset half width, used when blending semi-lagrangian result near the boundary
            half_width = 3.f;

            _theta = M_PI_2 * 0.85f;
            _phi = M_PI_2 * 0.6f;
            MyReal radius = 5.f * L / 128.f;
            MyReal height = 12.f * L / 128.f;
            MyReal density_to_add = 1.f / 8.f;
            MyReal speed = 2.f;

            auto vel_func_a = [&](Vec<3, MyReal> pos)
            {
                return Vec<3, MyReal>(-cos(_theta) * cos(_phi) * speed, -sin(_theta) * speed, -cos(_theta) * sin(_phi) * speed);
            };

            MyReal fixed_h = L/64.;
            Vec<2, MyReal> offset(30.f * L / 128.f);
            openvdb::Grid<openvdb::tree::Tree4<MyReal>::Type>::Ptr sphere_sdf_a = openvdb::tools::createLevelSetSphere<openvdb::Grid<openvdb::tree::Tree4<MyReal>::Type>>(radius, openvdb::math::Vec3<MyReal>(L / 2.0f + offset[0], 2 * L - height, L / 2.0f + offset[1]), fixed_h, half_width);
            Emitter e_sphere_a(total_frame, density_to_add, 0, Vec<3, MyReal>(0.f, 0.f, 0.f), sphere_sdf_a, [](MyReal frame)->Vec<3, MyReal> {return Vec<3, MyReal>(0.f, 0.f, 0.f); }, vel_func_a, true);
            emitter_list.push_back(e_sphere_a);

            auto vel_func_b = [&](Vec<3, MyReal> pos)
            {
                return Vec<3, MyReal>(-cos(_theta) * cos(_phi) * speed, -sin(_theta) * speed, cos(_theta) * sin(_phi) * speed);
            };

            Vec<2, MyReal> offset2(-30.f * L / 128.f);
            openvdb::Grid<openvdb::tree::Tree4<MyReal>::Type>::Ptr sphere_sdf_b = openvdb::tools::createLevelSetSphere<openvdb::Grid<openvdb::tree::Tree4<MyReal>::Type>>(radius, openvdb::math::Vec3<MyReal>(L / 2.0f + offset[0], 2 * L - height, L / 2.0f + offset2[1]), fixed_h, half_width);
            Emitter e_sphere_b(total_frame, 0, density_to_add, Vec<3, MyReal>(0.f, 0.f, 0.f), sphere_sdf_b, [](MyReal frame)->Vec<3, MyReal> {return Vec<3, MyReal>(0.f, 0.f, 0.f); }, vel_func_b, true);
            emitter_list.push_back(e_sphere_b);
        }
        break;
        case 7:
        {
            do_uniform_particle_seeding = true;
            do_particle_sample_after_first = true;
            adaptive_reset_cutoff = 1;
            N = 8;
            delayed_reinit_num = (sim_scheme == Scheme::CO_FLIP) ? 50 : delayed_reinit_num;
            // simulation resolution
            int baseres = _baseres;
            ni = baseres;
            nj = baseres;
            nk = baseres;
            substeps = 2;
            total_frame = 289; total_frame *= substeps; // 12 seconds
            // length in y direction
            L = 5.0f;
            // grid size for simulation
            h = L / ni;
            // time step
            MyReal deltaT = 1.f / framerate;
            dt = deltaT / substeps;
            // smoke properties
            smoke_rise = 0.55f;
            smoke_drop = 0.f;
            viscosity = 1e-4;
            // levelset half width, used when blending semi-lagrangian result near the boundary
            half_width = 3.f;

            _theta = M_PI_2 * 1.;
            _phi = M_PI_2 * 0.0f;
            MyReal radius = 5.f * L / 128.f;
            MyReal height = 12.f * L / 128.f;
            MyReal density_to_add = 3.f / 8.f;
            MyReal speed = 0.9f;

            auto vel_func_a = [&](Vec<3, MyReal> pos)
            {
                return Vec<3, MyReal>(-cos(_theta) * cos(_phi) * speed, -sin(_theta) * speed, -cos(_theta) * sin(_phi) * speed);
            };

            auto vel_emitter_pos_func_a = [&](MyReal framenum)
            {
                if ((framenum/(int)substeps) >= 72) { // 2 seconds
                    return Vec<3, MyReal>(0., 3.5/9., 0.);
                } else {
                    return Vec<3, MyReal>(0.);
                }
            };

            MyReal fixed_h = L/64.;
            openvdb::Grid<openvdb::tree::Tree4<MyReal>::Type>::Ptr sphere_sdf_a = openvdb::tools::createLevelSetSphere<openvdb::Grid<openvdb::tree::Tree4<MyReal>::Type>>(radius, openvdb::math::Vec3<MyReal>(L / 2.0f, height, L / 2.0f), fixed_h, half_width);
            Emitter e_sphere_a(total_frame, 0, density_to_add, Vec<3, MyReal>(0.f, 0.f, 0.f), sphere_sdf_a, vel_emitter_pos_func_a, vel_func_a, true);
            emitter_list.push_back(e_sphere_a);
        }
        break;
        case 8:
        {
            // simulation resolution
            int baseres = _baseres;
            ni = baseres*2;
            nj = baseres;
            nk = baseres;
            substeps = 4;
            total_frame = 231; total_frame *= substeps;
            // length in y direction
            L = 10.0f;
            // grid size for simulation
            h = L / ni;
            // time step
            MyReal deltaT = 1.f / framerate;
            dt = deltaT / substeps;
            // smoke properties
            smoke_rise = 0.0f;
            smoke_drop = 0.f;
            viscosity = 1e-4;
            // levelset half width, used when blending semi-lagrangian result near the boundary
            half_width = 3.f;
            MyReal speed = 2.f;
            auto vel_func = [&](Vec<3, MyReal> pos)
            {
                return Vec<3, MyReal>(speed, 0.f, 0.f);
            };

            MyReal fixed_h = L/64.;
            openvdb::Grid<openvdb::tree::Tree4<MyReal>::Type>::Ptr slab_sdf = openvdb::Grid<openvdb::tree::Tree4<MyReal>::Type>::create(20.0);
            openvdb::CoordBBox indexBB(openvdb::Coord(0, 0, 0), openvdb::Coord(ni, nj, nk));
            makeCylinder(slab_sdf, 2.5*L/128., fixed_h, Vec<3, MyReal>(.5, L / 4.0f, L / 4.0f), indexBB, fixed_h);
            Emitter e_slab(total_frame, 0.f, 0.f, Vec<3, MyReal>(0.f, 0.0f, 0.0f), slab_sdf,
                [](MyReal frame)->Vec<3, MyReal> {return Vec<3, MyReal>(0.f, 0.f, 0.f); }, vel_func, true);
            emitter_list.push_back(e_slab);

            openvdb::Grid<openvdb::tree::Tree4<float>::Type>::Ptr obstacle_sdf;
            std::string path_to_sdf = std::string("../modelData/spotObstacle/spotSDF_") + std::to_string(_baseres) + std::string("cubed.vdb");
            readVDBSDF<MyReal>(path_to_sdf, obstacle_sdf, "sdf");
            Boundary bdy_obstacle(Vec<3, MyReal>(0.f), obstacle_sdf, [](MyReal frame)->Vec<3, MyReal> {return Vec<3, MyReal>(0.f, 0.f, 0.f); });
            boundary_list.push_back(bdy_obstacle);
        }
        break;
        default:
        {

        }
    }
    std::cout << "[Experiment setup complete]" << std::endl;

    delayed_reinit_num *= (sim_scheme == Scheme::CO_FLIP) ? substeps : 1;
    
    filepath += enumToString(sim_experiment) + "/" + enumToString(sim_scheme) + ("_" + std::to_string(_baseres)) + "/";
    boost::filesystem::create_directories(filepath);

	COFLIPSolver mysolver(N, ni, nj, nk, L, viscosity, sim_scheme);
    mysolver.use_DEC_diagonal_hodge_star = use_DEC_diagonal_hodge_star;
    mysolver.use_pressure_solver = use_DEC_diagonal_hodge_star;
    mysolver.adaptive_reset_cutoff = adaptive_reset_cutoff;
    mysolver.delayed_reinit_num = delayed_reinit_num;
    if (do_vel_advection_only && (smoke_drop != 0.0f || smoke_rise != 0.0f))
    {
        std::cout << "There is bouyancy in this experiment, so density fields must also be advected!!!" << std::endl;
        do_vel_advection_only = false;
    }
    mysolver.do_vel_advection_only = do_vel_advection_only;
    mysolver.theta = _theta;
    mysolver.phi = _phi;
    mysolver.pp_repeat_count = pp_repeat_count;
    mysolver.set_velocity_inflow = set_velocity_inflow;
    mysolver.is_fixed_domain = is_fixed_domain;
    mysolver.do_uniform_particle_seeding = do_uniform_particle_seeding;
    mysolver.do_particle_sample_after_first = do_particle_sample_after_first;
    mysolver.is_matrix_small_enough = is_matrix_small_enough;
	mysolver.setSmoke(smoke_drop, smoke_rise, emitter_list);
    mysolver.setBoundary(boundary_list);
    mysolver.updateBoundary(0, dt);
    mysolver.setupPressureProjection(dt);
    mysolver.setupFromVDBFiles(filepathVelField, filepathDensityRhoField, filepathDensityTempField);
    if (set_init_vel && set_velocity_inflow)
        mysolver.setInitialVelocity(inflow_vel);
    if (set_init_vel_from_emitter)
        mysolver.setVelocityFromEmitter();
    mysolver.pressureProjectVelField();
    if (sim_scheme == Scheme::CO_FLIP || sim_scheme == Scheme::R_POLYFLIP || sim_scheme == Scheme::CF_POLYFLIP || sim_scheme == Scheme::POLYFLIP || sim_scheme == Scheme::POLYPIC) {
        mysolver.seedParticles();
        mysolver.sampleParticlesFromGrid();
    }
    mysolver.outputVorticityIntegral(filepath, 0);
    mysolver.outputEnergy(filepath, 0);
    mysolver.outputResult(0, filepath);
    std::cout << "[Solver setup complete]" << std::endl;
    for (uint i = 1; i < total_frame; i++)
	{
        std::cout << "Iteration " << i << " Starts !!!" << std::endl;
	    mysolver.updateBoundary(i, dt);
		mysolver.advance(i, dt);
        if ((experiment_name == 0 || experiment_name == 2) || i % int(substeps) == 0)
        {
            int framenum = (experiment_name == 0 || experiment_name == 2) ? i : i / int(substeps);
            mysolver.outputVorticityIntegral(filepath, dt*i);
            mysolver.outputEnergy(filepath, dt*i);
            mysolver.outputResult(framenum, filepath);
            std::cout << "Frame " << framenum << " Done !!!" << std::endl;
        }
    }
	return 0;
}
