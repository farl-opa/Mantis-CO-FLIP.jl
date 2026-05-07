#ifndef COFLIPSOLVER_H
#define COFLIPSOLVER_H
#include "../include/array.h"
#include "../include/tbb/tbb.h"
#include "../include/fluid_buffer3D.h"
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <iostream>
#include <cstdio>
#include <string>
#include "../include/vec.h"
#include "../utils/pcg_solver.h"
#include "../include/array3.h"
#include "../utils/GeometricLevelGen.h"
#include "../utils/color_macro.h"
#include "../utils/AlgebraicMultigrid.h"
#include "../utils/volumeMeshTools.h"
#include <boost/filesystem.hpp>
#define EIGEN_USE_BLAS
#define EIGEN_USE_LAPACKE_STRICT
#include "../utils/Eigen/IterativeLinearSolvers"
#include "../utils/Eigen/Sparse"
#include "../utils/Eigen/Dense"
#include "../utils/happly.h"

// AMGCL_USE_EIGEN_VECTORS_WITH_BUILTIN_BACKEND()

#define TOLERANCE 1e-7
#define MAX_ITERATIONS 200
#define TBB_GRAINSIZE 1000

typedef double MyReal;

enum Scheme {POLYPIC, POLYFLIP, R_POLYFLIP, CF_POLYFLIP, CO_FLIP};

enum TimeIntegration {RK1, RK2, RK3, RK4};

inline std::string enumToString(const Scheme& sim_scheme)
{
    switch (sim_scheme)
    {
    case R_POLYFLIP:
        return std::string("R_POLYFLIP");
    case CF_POLYFLIP:
        return std::string("CF_POLYFLIP");
    case CO_FLIP:
        return std::string("CO_FLIP");
    case POLYFLIP:
        return std::string("POLYFLIP");
    case POLYPIC:
        return std::string("POLYPIC");
    default:
        return std::string("unknown scheme");
    }
}

enum Experiment { TREFOIL_KNOT, LEAPFROG, UNKNOT, TWISTED_TORUS, TGV_ERROR_ANALYSIS, SMOKE_PLUME, PYROCLASTIC, INK_JET, ROCKET, SPOT_OBSTACLE };

inline std::string enumToString(const Experiment& sim_scheme)
{
    switch (sim_scheme)
    {
    case TREFOIL_KNOT:
        return std::string("TrefoilKnot");
    case LEAPFROG:
        return std::string("Leapfrog");
    case UNKNOT:
        return std::string("Unknot");
    case TWISTED_TORUS:
        return std::string("Twisted_Torus");
    case TGV_ERROR_ANALYSIS:
        return std::string("TGV_ErrorAnalysis");
    case SMOKE_PLUME:
        return std::string("SmokePlume");
    case PYROCLASTIC:
        return std::string("Pyroclastic");
    case INK_JET:
        return std::string("InkJet");
    case ROCKET:
        return std::string("Rocket");
    case SPOT_OBSTACLE:
        return std::string("SpotObstacle");
    default:
        return std::string("unknown experiment");
    }
}

class Emitter{
public:
    Emitter() : emitFrame(0), emit_density(0.f), emit_temperature(0.f), e_pos(Vec<3, MyReal>(0.f)), e_sdf(nullptr),
                vel_func([](MyReal frame)->Vec<3, MyReal>{return Vec<3, MyReal>(0.f);}),
                emit_velocity([](const Vec<3, MyReal>& pos)->Vec<3, MyReal>{return Vec<3, MyReal>(0.f);}),
                do_set_velocities(true), do_randomize_density(false), set_vel_everywhere_from_vel_func(false) {}
    Emitter(int frame, MyReal density, MyReal temperature, Vec<3, MyReal> position, openvdb::Grid<openvdb::tree::Tree4<MyReal>::Type>::Ptr sdf,
            std::function<Vec<3, MyReal>(MyReal framenum)> func,
            std::function<Vec<3, MyReal>(Vec<3, MyReal> pos)> emit_velfunc, bool set_velocities, bool randomize_density=false, bool set_vel_everywhere_from_vel_func=false)
        : emitFrame(frame), emit_density(density), emit_temperature(temperature), e_pos(position), e_sdf(sdf), vel_func(func), emit_velocity(emit_velfunc), do_set_velocities(set_velocities), do_randomize_density(randomize_density), set_vel_everywhere_from_vel_func(set_vel_everywhere_from_vel_func){}
    ~Emitter() = default;

    int emitFrame;
    MyReal emit_density;
    MyReal emit_temperature;
    Vec<3, MyReal> e_pos;
    openvdb::Grid<openvdb::tree::Tree4<MyReal>::Type>::Ptr e_sdf;
    std::function<Vec<3, MyReal>(MyReal framenum)> vel_func;
    std::function<Vec<3, MyReal>(const Vec<3, MyReal>& pos)> emit_velocity;
    bool do_set_velocities;
    bool do_randomize_density;
    bool set_vel_everywhere_from_vel_func;

    // update levelset position
    void update(MyReal framenum, MyReal voxel_size, MyReal dt)
    {
        e_pos += vel_func(framenum)*dt;
        openvdb::math::Mat4f transMat;
        transMat.setToScale(openvdb::math::Vec3<MyReal>(voxel_size));
        transMat.setTranslation(openvdb::math::Vec3<MyReal>(e_pos[0], e_pos[1], e_pos[2]));
        e_sdf->setTransform(openvdb::math::Transform::createLinearTransform(transMat));
    }
};

class Boundary{
public:
    Boundary(){};
    Boundary(Vec<3, MyReal> position, openvdb::Grid<openvdb::tree::Tree4<float>::Type>::Ptr sdf, std::function<Vec<3, MyReal>(MyReal framenum)> func): b_pos(position), b_sdf(sdf), vel_func(func) {}
    ~Boundary() = default;

    Vec<3, MyReal> b_pos;
    openvdb::Grid<openvdb::tree::Tree4<float>::Type>::Ptr b_sdf;
    std::function<Vec<3, MyReal>(MyReal framenum)> vel_func;

    // update levelset position
    void update(MyReal framenum, MyReal voxel_size, MyReal dt)
    {
        b_pos += vel_func(framenum)*dt;
        openvdb::math::Mat4f transMat;
        transMat.setToScale(openvdb::math::Vec3<MyReal>(voxel_size));
        transMat.setTranslation(openvdb::math::Vec3<MyReal>(b_pos[0], b_pos[1], b_pos[2]));
        b_sdf->setTransform(openvdb::math::Transform::createLinearTransform(transMat));
    }
};

template <typename T>
struct atomwrapper
{
    std::atomic<T> _a;

    atomwrapper()
        :_a()
    {}

    atomwrapper(const std::atomic<T> &a)
        :_a(a.load())
    {}

    atomwrapper(const atomwrapper &other)
        :_a(other._a.load())
    {}

    atomwrapper &operator=(const atomwrapper &other)
    {
        _a.store(other._a.load());
    }
 
    inline void atomic_fetch_add(T arg) {
        T desired, expected = _a.load(std::memory_order_relaxed);
        do {
            desired = expected + arg;
        } while(!_a.compare_exchange_weak(expected, desired));
    }
};

class Lagrangian_Particles
{
public:
    Lagrangian_Particles()
    {}
    Lagrangian_Particles(const Lagrangian_Particles &p)
    {
        volume = p.volume;
        vel = p.vel;
        vort = p.vort;
        vel_floatingPoint_error = p.vel_floatingPoint_error;
        vel_temp = p.vel_temp;
        pos_current = p.pos_current;
        pos_temp = p.pos_temp;
        rho = p.rho;
        T = p.T;
        C_rho = p.C_rho;
        C_T = p.C_T;
        C_u = p.C_u;
        C_v = p.C_v;
        C_w = p.C_w;
        longterm_pullback = p.longterm_pullback;
        shorterm_pullback = p.shorterm_pullback;
        delta_t = p.delta_t;
    }
    ~Lagrangian_Particles(){}

    void resize(uint size) {
        volume.resize(size);
        vel.resize(size);
        vort.resize(size);
        vel_floatingPoint_error.resize(size);
        vel_temp.resize(size);
        pos_current.resize(size);
        pos_temp.resize(size);
        rho.resize(size);
        T.resize(size);
        C_rho.resize(size);
        C_T.resize(size);
        C_u.resize(size);
        C_v.resize(size);
        C_w.resize(size);
        longterm_pullback.resize(size);
        shorterm_pullback.resize(size);
        delta_t.resize(size);
    }

    std::vector<MyReal> volume;
    std::vector<Vec<3, MyReal> > vel;
    std::vector<Vec<3, MyReal> > vort;
    std::vector<Vec<3, MyReal> > vel_floatingPoint_error;
    std::vector<Vec<3, MyReal> > vel_temp;
    std::vector<Vec<3, MyReal> > pos_current;
    std::vector<Vec<3, MyReal> > pos_temp;
    std::vector<MyReal> rho;
    std::vector<MyReal> T;
    std::vector<std::array<MyReal,8> > C_rho;
    std::vector<std::array<MyReal,8> > C_T;
    std::vector<std::array<MyReal,8> > C_u;
    std::vector<std::array<MyReal,8> > C_v;
    std::vector<std::array<MyReal,8> > C_w;
    std::vector<Eigen::Matrix3<MyReal> > longterm_pullback;
    std::vector<Eigen::Matrix3<MyReal> > shorterm_pullback;
    std::vector<MyReal> delta_t;

    inline double D10(const double& x, int bdy_version) {
        return (bdy_version == 0 ? 1.0 : 2.0) * (1.0 - x);
    }
    inline double D11(const double& x){
        return x;
    }
    //---------Quadratic bsplines------------
    inline double B20(const double& x, int bdy_version) {
        double s = 1.0 - x;
        return (bdy_version == 0 ? 0.5 : 1.0) * s * s;
    }
    inline double B21(const double& x, int bdy_version) {
        return bdy_version == 0 ? x*(1.0 - x) + 0.5 : x*(2.0 - 1.5*x);
    }
    inline double B22(const double& x){
        return 0.5 * x * x;
    }
    inline double D20(const double& x, int bdy_version) {
        double s = 1.0 - x;
        return (bdy_version == 0 ? 0.5 : (bdy_version == 1 ? 3.0/4.0 : 3.0)) * s * s;
    }
    inline double D21(const double& x, int bdy_version) {
        return (bdy_version == 0 || bdy_version == 1) ? x*(1.0 - x) + 0.5 : 3.0/2.0 * x*(2.0 - 1.5*x);
    }
    inline double D22(const double& x){
        return 0.5 * x * x;
    }
    //----------Cubic bsplines--------------------
    inline double B30(const double& x, int bdy_version) {
        double s = 1.0 - x;
        return (bdy_version == 0 ? 1.0 / 6.0 : (bdy_version == 1 ? 1.0 / 4.0 : 1.0)) * s * s * s;
    }
    inline double B31(const double& x, int bdy_version) {
        double x2 = x * x;
        double x3 = x2 * x;
        return bdy_version == 0 ? 1.0 / 6.0 * (3.0*x3 - 6.0*x2 + 4.0) : (bdy_version == 1 ? 1.0 / 12.0 * (7*x3 - 15.0*x2 + 3.0*x + 7.0) : x * (7.0 / 4.0 * x2 - 9.0 / 2.0 * x + 3.0));
    }
    inline double B32(const double& x, int bdy_version){
        double x2 = x * x;
        double x3 = x2 * x;
        return (bdy_version == 0 || bdy_version == 1) ? 1.0 / 6.0 * (-3.0*x3+ 3.0*x2+ 3.0*x + 1.0) : x2 * (3.0/2.0 - 11.0/12.0 * x);
    }
    inline double B33(const double& x){
        return 1.0/6.0 * x * x * x;
    }

    inline double kernel(double r)
    {
        return std::abs(r) < 1.0 ? 1 - std::abs(r) : 0;
    }
    inline double kernel1prime(double r, int index, int bdy_version=0)
    {
        if (index == 0) return D10(r, bdy_version);
        else return D11(r);
    }
    inline double kernel2(double r, int index, int bdy_version=0)
    {
        if (index == 0) return B20(r, bdy_version);
        else if (index == 1) return B21(r, bdy_version);
        else return B22(r);
    }
    inline double kernel2prime(double r, int index, int bdy_version=0)
    {
        if (index == 0) return D20(r, bdy_version);
        else if (index == 1) return D21(r, bdy_version);
        else return D22(r);
    }
    inline double kernel3(double r, int index, int bdy_version=0)
    {
        if (index == 0) return B30(r, bdy_version);
        else if (index == 1) return B31(r, bdy_version);
        else if (index == 2) return B32(r, bdy_version);
        else return B33(r);
    }
    inline double frand(double a, double b)
    {
        return a + (b-a) * rand()/(double)RAND_MAX;
    }
};

class COFLIPSolver {
public:
    COFLIPSolver() = default;
    COFLIPSolver(int N, uint nx, uint ny, uint nz, MyReal L, MyReal vis_coeff, Scheme myscheme);
    ~COFLIPSolver() = default;

    inline void clampPos(Vec<3, MyReal> &pos)
    {
        pos[0] = std::min(std::max((double)TOLERANCE, (double)pos[0]),double(_nxp*_h)-(double)TOLERANCE);
        pos[1] = std::min(std::max((double)TOLERANCE, (double)pos[1]),double(_nyp*_h)-(double)TOLERANCE);
        pos[2] = std::min(std::max((double)TOLERANCE, (double)pos[2]),double(_nzp*_h)-(double)TOLERANCE);
    }
    inline void clampPullback(Eigen::Matrix3<MyReal> &pullback)
    {
        double prev_mu = 0.0;
        double used_mu = 0.0;
        int iter = 0;
        Eigen::Matrix3<double> pullback_transpose_inverse = pullback.transpose().inverse().cast<double>();
        Eigen::Matrix3<double> output_pullback = pullback.cast<double>() + used_mu*pullback_transpose_inverse;
        prev_mu = used_mu;
        double mixed_det = output_pullback.determinant();
        double mixed_trace = ((pullback.transpose().cast<double>() * pullback.cast<double>()).inverse()).trace();
        // TOCHECK: some times nan shows up in the code here, so check and
        // and revert to no clamp if it happens.
        while (std::abs(mixed_det - 1.) > TOLERANCE && iter < MAX_ITERATIONS) {
            used_mu = prev_mu + ((1./mixed_det) - 1.)/mixed_trace;
            if (std::isnan(used_mu) || std::isinf(used_mu))
                break;
            output_pullback = pullback.cast<double>() + used_mu*pullback_transpose_inverse;
            prev_mu = used_mu;
            mixed_det = output_pullback.determinant();
            iter++;
        }
        output_pullback /= std::pow(mixed_det, 1.f/3.f);
        double test_norm = output_pullback.norm();
        if (!std::isnan(test_norm) && !std::isinf(test_norm)) {
            pullback = output_pullback.cast<MyReal>();
        }
    }

    void advance(int framenum, MyReal dt);
    void advanceReflectionPOLYFLIP(int framenum, MyReal dt);
    void advanceCovectorPOLYFLIP(int framenum, MyReal dt);
    void advanceCOFLIP(int framenum, MyReal dt);
    void addBuoyancy(Buffer3D<MyReal>& u_to_change, Buffer3D<MyReal>& v_to_change, Buffer3D<MyReal>& w_to_change, MyReal dt);
    void updateEmitters(int framenum, MyReal dt);
    void addEmitterForce(int framenum, MyReal dt);
    void emitSmoke(int framenum, MyReal dt);
    void setVelocityFromEmitter(bool do_only_x_dir_vel=false);
    void setSmoke(MyReal drop, MyReal raise, const std::vector<Emitter> &emitters);
    void outputResult(uint frame, std::string filepath);
    void outputEnergy(std::string filename, MyReal curr_time);
    void outputVorticityIntegral(std::string filename, MyReal curr_time);
    void setupFromVDBFiles(const std::string& filepathVelField,
                           const std::string& filepathDensityRhoField,
                           const std::string& filepathDensityTempField);
    void setBoundary(const std::vector<Boundary> &boundaries);
    void updateBoundary(int framenum, MyReal dt);
    void projection();
    void projectionWithVort();
    void clearBoundary(Buffer3D<MyReal>& field);
    MyReal getCFL();
    void pressureProjectVelField();
    MyReal computeFTLE(MyReal& det, const Eigen::Matrix3<MyReal>& pullback);
    void advectCOFLIPHelper(int stage_count, int framenum, MyReal dt, bool do_all=false);
    void advectCFFLIPHelper(int framenum, MyReal dt, bool do_all=false);
    void multiplyWithWedge(Eigen::VectorXd& result, const Eigen::VectorXd& RHS, int res_amp=1);
    void multiplyWithStarFlux(Eigen::VectorXd& result, const Eigen::VectorXd& RHS, int res_amp=1);
    void multiplyWithStarVort(Eigen::VectorXd& result, const Eigen::VectorXd& RHS, int res_amp=1);
    void multiplyWithInterp(Eigen::VectorXd& result, const Eigen::VectorXd& RHS, int res_amp=1);
    void multiplyWithInterpTranspose(std::vector<atomwrapper<double> >& result, const Eigen::VectorXd& RHS, bool do_norm2_squared=false, int res_amp=1);
    void multiplyWithD1Transpose(Eigen::VectorXd& result, const Eigen::VectorXd& RHS, int res_amp=1);
    void updateBackward(MyReal dt);

    void setInitialVelocity(MyReal inflow_vel);
    void setupPressureProjection(MyReal dt);
    void takeDualwrtStar(Buffer3D<MyReal> &u, Buffer3D<MyReal> &v, Buffer3D<MyReal> &w, bool update_uvw, bool flux2circulation_or_circulation2flux);
    bool edgeInFluid(int dir, int i, int j, int k, int res_amp=1);
    bool vertexInFluid(int i, int j, int k);
    void calculateCurl(bool do_star_fluxes=false, bool solve_for_dual_vorts=false);
    inline Vec<3, MyReal> traceRK4(MyReal dt, const Vec<3, MyReal> &pos, const Buffer3D<MyReal>& un, const Buffer3D<MyReal>& vn, const Buffer3D<MyReal>& wn);
    inline Eigen::Matrix3<MyReal> pullbackRK4(MyReal dt, Vec<3, MyReal> &inout_pos, const Eigen::Matrix3<MyReal>& input_pullback, const Buffer3D<MyReal>& un, const Buffer3D<MyReal>& vn, const Buffer3D<MyReal>& wn);
    inline Eigen::Matrix3<MyReal> solvePullbackODE(MyReal dt, Vec<3, MyReal> &inout_pos, const Eigen::Matrix3<MyReal>& input_pullback, const Buffer3D<MyReal>& un, const Buffer3D<MyReal>& vn, const Buffer3D<MyReal>& wn);
    void seedParticles();
    Vec<3, MyReal> getVelocity(const Vec<3, MyReal>& pos, const Buffer3D<MyReal>& un, const Buffer3D<MyReal>& vn, const Buffer3D<MyReal>& wn);
    Eigen::Matrix3<MyReal> getJacobianVelocity(const Vec<3, MyReal>& pos, const Buffer3D<MyReal>& un, const Buffer3D<MyReal>& vn, const Buffer3D<MyReal>& wn);
    MyReal sampleField(const Vec<3, MyReal>& pos, const Buffer3D<MyReal>& field, bool use_uniform=false, bool amp_res=false);
    Vec<3, MyReal> sampleGradientField(const Vec<3, MyReal>& pos, const Buffer3D<MyReal>& field, bool use_uniform=false, bool amp_res=false);
    Vec<4, MyReal> sampleCrossHessianField(const Vec<3, MyReal>& pos, const Buffer3D<MyReal>& field, bool use_uniform=false, bool amp_res=false);
    Vec<3, MyReal> getVelocityBSpline(const Vec<3, MyReal>& pos, const Buffer3D<MyReal>& un, const Buffer3D<MyReal>& vn, const Buffer3D<MyReal>& wn);
    Eigen::Matrix3<MyReal> getJacobianVelocityBSpline(const Vec<3, MyReal>& pos, const Buffer3D<MyReal>& un, const Buffer3D<MyReal>& vn, const Buffer3D<MyReal>& wn);
    MyReal sampleFieldBSpline(const Vec<3, MyReal>& pos, const Buffer3D<MyReal>& field, int selected_row);
    MyReal sampleGradientFieldBSpline(const Vec<3, MyReal>& pos, const Buffer3D<MyReal>& field, int selected_row, int selected_column);
    MyReal getPointwiseDiv(const Vec<3, MyReal>& pos, const Buffer3D<MyReal>& un, const Buffer3D<MyReal>& vn, const Buffer3D<MyReal>& wn);
    Vec<3, MyReal> getVorticity(const Vec<3, MyReal>& pos, const Buffer3D<MyReal>& vort_un, const Buffer3D<MyReal>& vort_vn, const Buffer3D<MyReal>& vort_wn);
    Vec<3, MyReal> getVorticityBSpline(const Vec<3, MyReal>& pos, const Buffer3D<MyReal>& vort_un, const Buffer3D<MyReal>& vort_vn, const Buffer3D<MyReal>& vort_wn);
    MyReal sampleFieldBSplineVort(const Vec<3, MyReal>& pos, const Buffer3D<MyReal>& field, int selected_row);
    inline MyReal lerp(const MyReal& v0, const MyReal& v1, MyReal c);
    inline MyReal bilerp(const MyReal& v00, const MyReal& v01, const MyReal& v10, const MyReal& v11, MyReal cx, MyReal cy);
    inline MyReal trilerp(
        const MyReal& v000, const MyReal& v001,
        const MyReal& v010, const MyReal& v011,
        const MyReal& v100, const MyReal& v101,
        const MyReal& v110, const MyReal& v111,
        MyReal cx, MyReal cy, MyReal cz);
    void sampleParticlesFromGrid();
    void solveInterpDagger(int stage_count, int framenum);

    // smoke parameter
    MyReal _alpha;
    MyReal _beta;

    std::vector<double> wmax_fullInterpMat;
    std::vector<Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> > A_L_fullInterpMat;
    std::vector<Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> > R_L_fullInterpMat;
    std::vector<Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> > P_L_fullInterpMat;
    std::vector<Vec3i> S_L_fullInterpMat;
    int total_level_fullInterpMat;
    Eigen::IncompleteCholesky<double, Eigen::Lower, Eigen::AMDOrdering<std::ptrdiff_t> > interp_dagger_precond;
    Eigen::IncompleteCholesky<double, Eigen::Lower, Eigen::AMDOrdering<std::ptrdiff_t> > starflux_precond;
    Eigen::IncompleteCholesky<double, Eigen::Lower, Eigen::AMDOrdering<std::ptrdiff_t> > starvort_precond;
    Eigen::VectorX<MyReal> fluxes, circulations, vorts, streamforms, pre_solve_fluxes, prev_vorts1form, 
                            fluxes_original, fluxes_midpoint;
    Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> invstarflux_matrix;
    Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> d1transpose_matrix;
    Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> d1_matrix;
    Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> d2_matrix;
    Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> gauge_laplacian_matrix;
    // AMGPCG solver data
    std::vector<double> wmax_pressure_laplacian;
    std::vector<Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> > A_L_pressure_laplacian;
    std::vector<Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> > R_L_pressure_laplacian;
    std::vector<Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> > P_L_pressure_laplacian;
    std::vector<Vec3i> S_L_pressure_laplacian;
    int total_level_pressure_laplacian;
    std::vector<double> wmax_streamform_laplacian;
    std::vector<Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> > A_L_streamform_laplacian;
    std::vector<Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> > R_L_streamform_laplacian;
    std::vector<Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> > P_L_streamform_laplacian;
    std::vector<Vec3i> S_L_streamform_laplacian;
    int total_level_streamform_laplacian;
    std::vector<double> wmax_viscosity_laplacian;
    std::vector<Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> > A_L_viscosity_laplacian;
    std::vector<Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> > R_L_viscosity_laplacian;
    std::vector<Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> > P_L_viscosity_laplacian;
    std::vector<Vec3i> S_L_viscosity_laplacian;
    int total_level_viscosity_laplacian;
    levelGen<MyReal> amg_levelGen;
    levelGen<double> amg_levelGen_double;
    buffer3Dc _b_desc;

    // simulation data
    int _N;
    uint _nx, _ny, _nz, _nC, _nF, _nE, _nV;
    uint _nxp, _nyp, _nzp;
    MyReal max_v;
    MyReal max_vort;
    MyReal _h, _h_uniform;
    MyReal _amped_h;
    MyReal _amped_nx, _amped_ny, _amped_nz;
    MyReal viscosity;
    Buffer3D<MyReal> _vort_un, _vort_vn, _vort_wn;
    Buffer3D<MyReal> _un, _vn, _wn;
    Buffer3D<MyReal> _u_weight, _v_weight, _w_weight;
    Buffer3D<MyReal> _u0, _v0, _w0;
    Buffer3D<MyReal> _u1, _v1, _w1;
    Buffer3D<MyReal> _u2, _v2, _w2;
    Buffer3D<MyReal> _utemp, _vtemp, _wtemp;
    Buffer3D<MyReal> _duproj, _dvproj, _dwproj;
    Buffer3D<MyReal> _duextern, _dvextern, _dwextern;
    Buffer3D<MyReal> _rho, _rhotemp, _rhoinit, _rhoprev, _drhoextern, _rho_weight, _rho_save, _rho_diff;
    Buffer3D<MyReal> _T, _T_weight, _T_save, _T_diff;
    Buffer3D<MyReal> _u_save, _v_save, _w_save;
    Buffer3D<MyReal> _u_diff, _v_diff, _w_diff;
    Buffer3D<MyReal> _back_x, _back_y, _back_z;
    Buffer3D<MyReal> _back_temp_x, _back_temp_y, _back_temp_z;

    Buffer3D<MyReal> _usolid, _vsolid, _wsolid;
    Scheme sim_scheme;

    Lagrangian_Particles lagrangian_particles;
    std::vector<Emitter> sim_emitter;
    std::vector<Boundary> sim_boundary;
    bool do_vel_advection_only;
    int delayed_reinit_num = 1;
    MyReal theta=M_PI_2, phi=0.f;
    int _extraPad = 2;
    int pp_repeat_count = 1;
    bool set_velocity_inflow = false;
    bool is_fixed_domain = true;
    int bs_p = 3;
    bool do_mass_lumping = false;
    TimeIntegration timeIntOrder = TimeIntegration::RK2;
    bool do_delta_circulation = false;
    bool do_uniform_particle_seeding = false;
    bool use_DEC_diagonal_hodge_star = false;
    MyReal adaptive_reset_cutoff = 3.;
    int precond_reset_frequency = 5;
    bool reset_precond = false;
    bool is_matrix_small_enough = false;
    bool use_pressure_solver = false;
    bool do_particle_sample_after_first = false;
};


#endif //COFLIPSOLVER_H
