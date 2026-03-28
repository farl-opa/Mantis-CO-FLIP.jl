#include <iostream>
#include <cstdlib>
#include <string>
#include "COFLIPSolver2D.h"
#include "../utils/visualize.h"

int main(int argc, char** argv)
{
    // resolution
    int nx;
    int ny;
    // time step
    double dt;
    // simulation domain length in x-direction
    double L;
    // particle per cell
    int N;
    int total_frame;
    double vorticity_distance;
    // smoke property
    double smoke_rise;
    double smoke_drop;
    // viscosity propoerty
    double viscosity;
    // use Neumann boundary or not
    bool PURE_NEUMANN;
    Scheme sim_scheme;
    int sim_name = 0;
    int experiment_name = 0;
    if (argc < 3)
    {
        std::cout << "Please specify correct parameters!" << std::endl;
        std::cout << "inputs: [Method] [Experiment]" << std::endl;
        std::cout << "Valid method numbers are [0-4] for 0: POLYPIC, 1: POLYFLIP, 2: R_POLYFLIP, 3: CF_POLYFLIP, 4: CO_FLIP" << std::endl;
        std::cout << "Valid experiment numbers are [0-4] for 0: Vortex leapfrogging marathon, 1: Rayleigh-Taylor Instability, 2: Smoke Plume, 3: Vortex sheet, 4: Convecting vortex periodic box" << std::endl;
        exit(0);
    }
    sim_name = atoi(argv[1]);
    experiment_name = atoi(argv[2]);
    if (sim_name > 4 || sim_name < 0)
    {
        std::cout << "Please enter valid method number!" << std::endl;
        std::cout << "inputs: [Method] [Experiment]" << std::endl;
        std::cout << "Valid method numbers are [0-4] for 0: POLYPIC, 1: POLYFLIP, 2: R_POLYFLIP, 3: CF_POLYFLIP, 4: CO_FLIP" << std::endl;
        std::cout << "Valid experiment numbers are [0-4] for 0: Vortex leapfrogging marathon, 1: Rayleigh-Taylor Instability, 2: Smoke Plume, 3: Vortex sheet, 4: Convecting vortex periodic box" << std::endl;
        exit(0);
    }
    if (experiment_name > 4 || experiment_name < 0)
    {
        std::cout << "Please enter valid experiment number!" << std::endl;
        std::cout << "inputs: [Method] [Experiment]" << std::endl;
        std::cout << "Valid method numbers are [0-4] for 0: POLYPIC, 1: POLYFLIP, 2: R_POLYFLIP, 3: CF_POLYFLIP, 4: CO_FLIP" << std::endl;
        std::cout << "Valid experiment numbers are [0-4] for 0: Vortex leapfrogging marathon, 1: Rayleigh-Taylor Instability, 2: Smoke Plume, 3: Vortex sheet, 4: Convecting vortex periodic box" << std::endl;
        exit(0);
    }
    sim_scheme = static_cast<Scheme>(sim_name);
    int delayed_reinit_frequency = 1;
    omp_set_num_threads(24);
    Eigen::setNbThreads(24);
    Eigen::initParallel();
    std::cout << "Eigen thread count: " << Eigen::nbThreads() << std::endl;

    std::string base_path = "../Out_2D";
    switch(experiment_name)
    {
        // Vortex leapfrogging pairs
        case 0:
        {
            std::cout << GREEN << "Start running 2D Vortex Leapfrogging experiment_name!!!" << RESET << std::endl;
            int extendFrames = 20;
            double substep = argc >= 4 ? atoi(argv[3]) : 1.0;
            int baseres = 256;
            nx = baseres;
            ny = baseres;
            dt = 1./48./substep;
            L = 2.*M_PI;
            N = argc >= 5 ? atoi(argv[4]) : 6;
            total_frame = 1200*extendFrames*substep;
            smoke_rise = 0.;
            smoke_drop = 0.;
            viscosity = 0.;
            PURE_NEUMANN = true;
            bool use_pressure_solver = false;
            TimeIntegration timeIntOrder = argc >= 6 ? static_cast<TimeIntegration>(std::clamp(atoi(argv[5]),1,4)-1) : TimeIntegration::RK2;
            delayed_reinit_frequency = (sim_scheme == Scheme::CO_FLIP || sim_scheme == Scheme::R_POLYFLIP || sim_scheme == Scheme::CF_POLYFLIP || sim_scheme == Scheme::POLYFLIP) ? (argc >= 7 ? std::max(std::abs(atoi(argv[6])),1) : 150) : delayed_reinit_frequency;
            std::string resolutionString = nx == 256 ? "" : "_res" + std::to_string(nx);
            std::string NString = N == 6 ? "" : "N" + std::to_string(N);
            std::string filepath = base_path + "/2D_Leapfrog" + resolutionString + "_ss" + std::to_string((int)substep) + "_" + NString + "_RK" + std::to_string(timeIntOrder+1) + "_" + std::to_string(delayed_reinit_frequency) + "/" + enumToString(sim_scheme) + "/";
            std::string filename = enumToString(sim_scheme) + "_dt_" + std::to_string(dt).substr(0,5) +"_";
            COFLIPSolver2D smokeSimulator(nx, ny, L, N, PURE_NEUMANN, sim_scheme);
            smokeSimulator.use_pressure_solver = use_pressure_solver;
            smokeSimulator.timeIntOrder = timeIntOrder;
            smokeSimulator.substep = substep;
            smokeSimulator.adaptive_reset_cutoff = 1;
            smokeSimulator.do_implicit = true;
            smokeSimulator.setSmoke(smoke_rise, smoke_drop);
            smokeSimulator.setBoundaryMask();
            smokeSimulator.buildMultiGridWithVort();
            smokeSimulator.setInitLeapFrog(0.15, 1.5, 3.0, M_PI-1.2, 0.3);
            if (sim_scheme != Scheme::CO_FLIP || use_pressure_solver) {
                smokeSimulator.buildMultiGrid(PURE_NEUMANN);
                smokeSimulator.projection_repeat_count = 2;
                smokeSimulator.pressureProjectVelField();
            }
            smokeSimulator.applyVelocityBoundary();
            smokeSimulator.seedParticles(N, true);
            smokeSimulator.sampleParticlesFromGrid();
            smokeSimulator.outputVorticityIntegral(filepath, 0.);
            smokeSimulator.outputEnergy(filepath, 0.);
            smokeSimulator.outputVortVisualized(filepath, filename, 0);

            for (int i = 0; i < total_frame; i++)
            {
                smokeSimulator.advance(dt, i, delayed_reinit_frequency);
                int outputFrame = ((int)substep*extendFrames);
                int curr_i = i+1;
                if (curr_i%outputFrame==0) {
                    smokeSimulator.outputVortVisualized(filepath, filename, curr_i/outputFrame);
                    double curr_time = dt * double(curr_i);
                    smokeSimulator.outputVorticityIntegral(filepath, curr_time);
                    smokeSimulator.outputEnergy(filepath, curr_time);
                }
            }
        }
        break;
        // Rayleigh-Taylor Instability
        case 1:
        {
            std::cout << GREEN << "Start running 2D Ink drop experiment_name!!!" << RESET << std::endl;
            nx = 256;
            ny = 512;
            double substep = argc >= 4 ? atoi(argv[3]) : 1.0;
            dt = 1./96./substep;
            L = 0.2;
            N = argc >= 5 ? atoi(argv[4]) : 6;
            total_frame = 390*substep;
            smoke_rise = 0.;
            smoke_drop = 0.15;
            viscosity = 1e-7;
            PURE_NEUMANN = true;
            bool use_pressure_solver = false;
            TimeIntegration timeIntOrder = argc >= 6 ? static_cast<TimeIntegration>(std::clamp(atoi(argv[5]),1,4)-1) : TimeIntegration::RK2;
            delayed_reinit_frequency = (sim_scheme == Scheme::CO_FLIP || sim_scheme == Scheme::R_POLYFLIP || sim_scheme == Scheme::CF_POLYFLIP || sim_scheme == Scheme::POLYFLIP) ? (argc >= 7 ? std::max(std::abs(atoi(argv[6])),1) : 100) : delayed_reinit_frequency;
            std::string resolutionString = nx == 256 ? "" : "_res" + std::to_string(nx);
            std::string NString = N == 6 ? "" : "N" + std::to_string(N);
            std::string filepath = base_path + "/2D_RayleighTaylor" + resolutionString +  "_ss" + std::to_string((int)substep) + "_" + NString + "_RK" + std::to_string(timeIntOrder+1) + "_" + std::to_string(delayed_reinit_frequency) + "/" + enumToString(sim_scheme) + "/";
            std::string filename = enumToString(sim_scheme) + "_dt_" + std::to_string(dt).substr(0,5);
            COFLIPSolver2D smokeSimulator(nx, ny, L, N, PURE_NEUMANN, sim_scheme);
            smokeSimulator.use_pressure_solver = use_pressure_solver;
            smokeSimulator.timeIntOrder = timeIntOrder;
            smokeSimulator.substep = substep;
            smokeSimulator.adaptive_reset_cutoff = 3;
            smokeSimulator.do_uniform_particle_seeding = true;
            smokeSimulator.precond_reset_frequency = 1;
            smokeSimulator.do_particle_sample_after_first = true;
            smokeSimulator.viscosity = viscosity;
            smokeSimulator.do_implicit = true;
            smokeSimulator.min_PPC_count = (sim_scheme == Scheme::R_POLYFLIP || sim_scheme == Scheme::CF_POLYFLIP || sim_scheme == Scheme::POLYFLIP) ? N : 10;
            smokeSimulator.setSmoke(smoke_rise, smoke_drop);
			smokeSimulator.setBoundaryMask();
            smokeSimulator.buildMultiGridWithVort();
            if (sim_scheme != Scheme::CO_FLIP || use_pressure_solver) {
                smokeSimulator.buildMultiGrid(PURE_NEUMANN);
                smokeSimulator.projection_repeat_count = 2;
            }
            smokeSimulator.setInitReyleighTaylor(0.05 * L);
            smokeSimulator.seedParticles(N, true);
            smokeSimulator.sampleParticlesFromGrid();
            for (int i = 0; i < total_frame; i++)
            {
                smokeSimulator.advance(dt, i, delayed_reinit_frequency);
                int curr_i = i+1;
                if (curr_i%(int)substep==0) {
                    smokeSimulator.outputVortVisualized(filepath, filename, curr_i/(int)substep);
                    smokeSimulator.outputDensity(filepath, "density", curr_i/(int)substep, true);
                    double curr_time = dt * double(curr_i);
                    smokeSimulator.outputVorticityIntegral(filepath, curr_time);
                    smokeSimulator.outputEnergy(filepath, curr_time);
                }
            }
        }
        break;
        // Smoke Plume
        case 2:
        {
            std::cout << GREEN << "Start running 2D ink drop siggraph logo experiment_name!!!" << RESET << std::endl;
            nx = 256;
            ny = 512;
            double substep = argc >= 4 ? atoi(argv[3]) : 1.0;
            dt = 1./96./substep;
            L = 0.5;
            N = argc >= 5 ? atoi(argv[4]) : 6;
            total_frame = 1000*substep;
            smoke_rise = 0.;
            smoke_drop = -0.1;
            viscosity = 1e-6;
            PURE_NEUMANN = true;
            bool use_pressure_solver = false;
            TimeIntegration timeIntOrder = argc >= 6 ? static_cast<TimeIntegration>(std::clamp(atoi(argv[5]),1,4)-1) : TimeIntegration::RK2;
            delayed_reinit_frequency = (sim_scheme == Scheme::CO_FLIP || sim_scheme == Scheme::R_POLYFLIP || sim_scheme == Scheme::CF_POLYFLIP || sim_scheme == Scheme::POLYFLIP) ? (argc >= 7 ? std::max(std::abs(atoi(argv[6])),1) : 100) : delayed_reinit_frequency;
            std::string resolutionString = nx == 256 ? "" : "_res" + std::to_string(nx);
            std::string NString = N == 6 ? "" : "N" + std::to_string(N);
            std::string filepath = base_path + "/2D_smokeplume" + resolutionString +  "_ss" + std::to_string((int)substep) + "_" + NString + "_RK" + std::to_string(timeIntOrder+1) + "_" + std::to_string(delayed_reinit_frequency) + "/" + enumToString(sim_scheme) + "/";
            std::string filename = enumToString(sim_scheme) + "_dt_" + std::to_string(dt).substr(0,5);
            COFLIPSolver2D smokeSimulator(nx, ny, L, N, PURE_NEUMANN, sim_scheme);
            smokeSimulator.use_pressure_solver = use_pressure_solver;
            smokeSimulator.timeIntOrder = timeIntOrder;
            smokeSimulator.substep = substep;
            smokeSimulator.adaptive_reset_cutoff = 3;
            smokeSimulator.do_uniform_particle_seeding = true;
            smokeSimulator.precond_reset_frequency = 1;
            smokeSimulator.do_particle_sample_after_first = true;
            smokeSimulator.viscosity = viscosity;
            smokeSimulator.do_implicit = true;
            smokeSimulator.min_PPC_count = (sim_scheme == Scheme::R_POLYFLIP || sim_scheme == Scheme::CF_POLYFLIP || sim_scheme == Scheme::POLYFLIP) ? N : 10;
            smokeSimulator.setSmoke(smoke_rise, smoke_drop);
			smokeSimulator.setBoundaryMask();
            smokeSimulator.buildMultiGridWithVort();
            if (sim_scheme != Scheme::CO_FLIP || use_pressure_solver) {
                smokeSimulator.buildMultiGrid(PURE_NEUMANN);
                smokeSimulator.projection_repeat_count = 2;
            }
            smokeSimulator.initSmokePlume();
            smokeSimulator.seedParticles(N, true);
            smokeSimulator.sampleParticlesFromGrid();
            for (int i = 0; i < total_frame; i++)
            {
                smokeSimulator.advance(dt, i, delayed_reinit_frequency);
                int curr_i = i+1;
                if (curr_i%(int)substep==0) {
                    smokeSimulator.outputVortVisualized(filepath, filename, curr_i/(int)substep);
                    smokeSimulator.outputDensity(filepath, "density", curr_i/(int)substep, true, false, true);
                    double curr_time = dt * double(curr_i);
                    smokeSimulator.outputVorticityIntegral(filepath, curr_time);
                    smokeSimulator.outputEnergy(filepath, curr_time);
                }
            }
        }
        break;
        // 2D Vortex sheet
        case 3:
        {
            std::cout << GREEN << "Start running 2D Vortex sheet experiment_name!!!" << RESET << std::endl;
            double substep = argc >= 4 ? atof(argv[3]) : 1.0;
            int baseres = argc >= 8 ? atoi(argv[7]) : 256;
            nx = baseres;
            ny = baseres;
            dt = 1./3./substep;
            L = 2.*M_PI;
            N = argc >= 5 ? atoi(argv[4]) : 6;
            total_frame = 40*substep;
            smoke_rise = 0.;
            smoke_drop = 0.;
            viscosity = 0.;
            PURE_NEUMANN = true;
            bool use_pressure_solver = false;
            TimeIntegration timeIntOrder = argc >= 6 ? static_cast<TimeIntegration>(std::clamp(atoi(argv[5]),1,4)-1) : TimeIntegration::RK2;
            delayed_reinit_frequency = (sim_scheme == Scheme::CO_FLIP || sim_scheme == Scheme::R_POLYFLIP || sim_scheme == Scheme::CF_POLYFLIP || sim_scheme == Scheme::POLYFLIP) ? (argc >= 7 ? std::max(std::abs(atoi(argv[6])),1) : 50) : delayed_reinit_frequency;
            std::string resolutionString = nx == 256 ? "" : "_res" + std::to_string(nx);
            std::string NString = N == 6 ? "" : "N" + std::to_string(N);
            std::string filepath = base_path + "/2D_vortex_sheet" + resolutionString + "_ss" + std::to_string(substep) + "_" + NString + "_RK" + std::to_string(timeIntOrder+1) + "_" + std::to_string(delayed_reinit_frequency) + "/" + enumToString(sim_scheme) + "/";
            std::string filename = enumToString(sim_scheme) + "_dt_" + std::to_string(dt).substr(0,5) + "_dist_" + std::to_string(vorticity_distance).substr(0,4) +"_";
            COFLIPSolver2D smokeSimulator(nx, ny, L, N, PURE_NEUMANN, sim_scheme);
            smokeSimulator.use_pressure_solver = use_pressure_solver;
            smokeSimulator.substep = substep;
            smokeSimulator.timeIntOrder = timeIntOrder;
            smokeSimulator.do_delta_circulation = false;
            smokeSimulator.adaptive_reset_cutoff = 3;
            smokeSimulator.do_uniform_particle_seeding = true;
            smokeSimulator.precond_reset_frequency = 1;
            smokeSimulator.do_implicit = true;
            smokeSimulator.setSmoke(smoke_rise, smoke_drop);
			smokeSimulator.setBoundaryMask();
            smokeSimulator.buildMultiGridWithVort();
            smokeSimulator.setInitVelocityVortexSheet(0.25*L,0.25,1.25*L/256.);
            if (sim_scheme != Scheme::CO_FLIP || use_pressure_solver) {
                smokeSimulator.buildMultiGrid(PURE_NEUMANN);
                smokeSimulator.projection_repeat_count = 2;
            }
            smokeSimulator.pressureProjectVelField();
            smokeSimulator.seedParticles(N, true);
            smokeSimulator.sampleParticlesFromGrid();
            smokeSimulator.outputVortVisualized(filepath, filename, 0);
            smokeSimulator.outputVorticityIntegral(filepath, 0.);
            smokeSimulator.outputEnergy(filepath, 0.);

            for (int i = 0; i < total_frame; i++)
            {
                smokeSimulator.advance(dt, i, delayed_reinit_frequency);
                int curr_i = i+1;
                int int_substep = std::max(1, (int)substep);
                if (curr_i%int_substep==0) {
                    smokeSimulator.outputVortVisualized(filepath, filename, curr_i/int_substep);
                    double curr_time = dt * double(curr_i);
                    smokeSimulator.outputDensity(filepath, "density", curr_i/int_substep, true);
                    smokeSimulator.outputVorticityIntegral(filepath, curr_time);
                    smokeSimulator.outputEnergy(filepath, curr_time);
                }
            }

        }
        break;
        // 2D Convecting vortex in a periodic unit box (Julia parity setup)
        case 4:
        {
            std::cout << GREEN << "Start running 2D Convecting vortex periodic-box experiment!!!" << RESET << std::endl;
            nx = 64;
            ny = 64;
            double target_cfl = 0.5;
            L = 1.0;
            N = 20;
            double T_final = 1.0;
            smoke_rise = 0.0;
            smoke_drop = 0.0;
            viscosity = 0.0;
            PURE_NEUMANN = true;
            bool use_pressure_solver = false;
            TimeIntegration timeIntOrder = TimeIntegration::RK2;
            delayed_reinit_frequency = 20;

            std::string filepath = base_path + "/2D_convecting_vortex_periodic_res64" + "/" + enumToString(sim_scheme) + "/";
            std::string filename = enumToString(sim_scheme) + "_convecting_vortex_";

            COFLIPSolver2D smokeSimulator(nx, ny, L, N, PURE_NEUMANN, sim_scheme);
            smokeSimulator.use_pressure_solver = use_pressure_solver;
            smokeSimulator.timeIntOrder = timeIntOrder;
            smokeSimulator.substep = 1.0;
            smokeSimulator.adaptive_reset_cutoff = 3;
            smokeSimulator.do_uniform_particle_seeding = true;
            smokeSimulator.precond_reset_frequency = 1;
            smokeSimulator.do_particle_sample_after_first = false;
            smokeSimulator.viscosity = viscosity;
            smokeSimulator.do_implicit = true;
            smokeSimulator.min_PPC_count = 10;
            smokeSimulator.setSmoke(smoke_rise, smoke_drop);
            smokeSimulator.setBoundaryMask();
            smokeSimulator.buildMultiGridWithVort();
            if (sim_scheme != Scheme::CO_FLIP || use_pressure_solver) {
                smokeSimulator.buildMultiGrid(PURE_NEUMANN);
                smokeSimulator.projection_repeat_count = 2;
            }

            // Julia flow_convecting_vortex parameters.
            const double U0 = 1.0;
            const double Gamma = 5.0;
            const double sigma = 0.1 * L;
            const Vec2d center = Vec2d(0.5 * L, 0.5 * L);
            smokeSimulator.setInitVelocityConvectingVortex(U0, Gamma, sigma, center);
            smokeSimulator.pressureProjectVelField();

            smokeSimulator.seedParticles(N, true);
            smokeSimulator.sampleParticlesFromGrid();

            smokeSimulator.getCFL();
            double dt_cfl = target_cfl * smokeSimulator._cfl;
            int n_steps = std::max(1, (int)std::ceil(T_final / dt_cfl));
            dt = T_final / (double)n_steps;
            total_frame = n_steps;

            smokeSimulator.outputVortVisualized(filepath, filename, 0);
            smokeSimulator.outputVorticityIntegral(filepath, 0.0);
            smokeSimulator.outputEnergy(filepath, 0.0);

            for (int i = 0; i < total_frame; i++)
            {
                smokeSimulator.advance(dt, i, delayed_reinit_frequency);
                int curr_i = i + 1;
                smokeSimulator.outputVortVisualized(filepath, filename, curr_i);
                double curr_time = dt * double(curr_i);
                smokeSimulator.outputVorticityIntegral(filepath, curr_time);
                smokeSimulator.outputEnergy(filepath, curr_time);
            }
        }
        break;
    }

    return 0;
}
