#ifndef COFLIPSOLVER2D_H
#define COFLIPSOLVER2D_H
#include <stdlib.h>
#include <fcntl.h>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <random>
#include "tbb/tbb.h"
#include "../include/array2.h"
#include "../include/vec.h"
#include "../utils/AlgebraicMultigrid.h"
#include "../utils/GeometricLevelGen.h"
#include "../utils/writeBMP.h"
#include "../utils/visualize.h"
#include "../utils/color_macro.h"
#include <boost/filesystem.hpp>
#define EIGEN_USE_BLAS
#define EIGEN_USE_LAPACKE_STRICT
#include "../utils/Eigen/IterativeLinearSolvers"
#include "../utils/Eigen/Sparse"
#include "../utils/Eigen/Dense"
#include "../utils/happly.h"

#define TOLERANCE 1e-9
#define MAX_ITERATIONS 200
#define TBB_GRAINSIZE 1000

enum Scheme {POLYPIC, POLYFLIP, R_POLYFLIP, CF_POLYFLIP, CO_FLIP};

enum TimeIntegration {RK1, RK2, RK3, RK4};

inline std::string enumToString(const Scheme &sim_scheme)
{
    switch(sim_scheme)
    {
        case POLYPIC:
            return std::string("POLYPIC");
        case POLYFLIP:
            return std::string("POLYFLIP");
        case R_POLYFLIP:
            return std::string("R+POLYFLIP");
        case CF_POLYFLIP:
            return std::string("CF+POLYFLIP");
        case CO_FLIP:
            return std::string("CO_FLIP");
        default:
            return std::string("unknown scheme");
    }
}

class CmapParticles
{
public:
    CmapParticles()
    {
        vel=Vec2d();
        vel_temp=Vec2d();
        pos_current = Vec2d();
        pos_temp = Vec2d();
        rho = 0;
        temperature = 0;
        C_x = Vec4d(0.0);
        C_y = Vec4d(0.0);
        C_rho = Vec4d(0.0);
        C_temperature = Vec4d(0.0);
        longterm_pullback = Eigen::Matrix2d::Zero();
        shorterm_pullback = Eigen::Matrix2d::Zero();
        delta_t = 0.0;
        volume = 0.;
    }
    ~CmapParticles(){}
    Vec2d vel;
    Vec2d vel_temp;
    Vec2d pos_current;
    Vec2d pos_temp;
    double rho;
    double temperature;
    Vec4d C_x;
    Vec4d C_y;
    Vec4d C_rho;
    Vec4d C_temperature;
    Eigen::Matrix2d longterm_pullback;
    Eigen::Matrix2d shorterm_pullback;
    double delta_t;
    double volume;

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

    CmapParticles(const CmapParticles &p)
    {
        vel = p.vel;
        vel_temp = p.vel_temp;
        pos_current = p.pos_current;
        pos_temp = p.pos_temp;
        rho = p.rho;
        temperature = p.temperature;
        C_x = p.C_x;
        C_y = p.C_y;
        C_rho = p.C_rho;
        C_temperature = p.C_temperature;
        longterm_pullback = p.longterm_pullback;
        shorterm_pullback = p.shorterm_pullback;
        delta_t = p.delta_t;
        volume = p.volume;
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
    inline Vec4d calculateCp(Vec2d pos, const Array2d &field, double h, int ni, int nj, double offx, double offy)
    {
        Vec4d Cp = Vec4d(0.);
        Vec2d spos = pos - h*Vec2d(offx, offy);
        int i = std::floor(spos.v[0] / h), j = std::floor(spos.v[1] / h);
        double px = spos.v[0] - i*h;
        double py = spos.v[1] - j*h;
        if (offy > 0)
        {
            if (!(i >= 0 && i <= ni - 1 && j >= 0 && j <= nj - 2))
                return Cp;
            else{
                Cp[0] = ((h-px)*(h-py)*field(i,j) + px*(h-py)*field(i+1,j)
                         + px*py*field(i+1,j+1) + (h-px)*py*field(i,j+1)) / (h*h);
                Cp[1] = (-(h-py)*field(i,j) + (h-py)*field(i+1,j) + py*field(i+1,j+1)
                         -py*field(i,j+1))/(h*h);
                Cp[2] = (-(h-px)*field(i,j) - px*field(i+1,j) + px*field(i+1,j+1)
                         + (h-px)*field(i,j+1))/(h*h);
                Cp[3] = (field(i,j) - field(i+1,j) + field(i+1,j+1) - field(i,j+1))/(h*h);
                return Cp;
            }
        }
        else
        {
            if (!(i >= 0 && i <= ni - 2 && j >= 0 && j <= nj - 1))
                return Cp;
            else{
                Cp[0] = ((h-px)*(h-py)*field(i,j) + px*(h-py)*field(i+1,j)
                         + px*py*field(i+1,j+1) + (h-px)*py*field(i,j+1)) / (h*h);
                Cp[1] = (-(h-py)*field(i,j) + (h-py)*field(i+1,j) + py*field(i+1,j+1)
                         -py*field(i,j+1))/(h*h);
                Cp[2] = (-(h-px)*field(i,j) - px*field(i+1,j) + px*field(i+1,j+1)
                         + (h-px)*field(i,j+1))/(h*h);
                Cp[3] = (field(i,j) - field(i+1,j) + field(i+1,j+1) - field(i,j+1))/(h*h);
                return Cp;
            }
        }
    }
};

class COFLIPSolver2D {
public:
    inline void clampPos(Vec2d &pos)
    {
        pos[0] = std::min(std::max(TOLERANCE, pos[0]),double(nip*h)-TOLERANCE);
        pos[1] = std::min(std::max(TOLERANCE, pos[1]),double(njp*h)-TOLERANCE);
    }
    inline void clampPullback(Eigen::Matrix2d &pullback)
    {
        double prev_mu = 0.0;
        double used_mu = 0.0;
        int iter = 0;
        Eigen::Matrix2d pullback_transpose_inverse = pullback.transpose().inverse();
        Eigen::Matrix2d output_pullback = pullback + used_mu*pullback_transpose_inverse;
        prev_mu = used_mu;
        double mixed_det = output_pullback.determinant();
        double mixed_trace = ((pullback.transpose() * pullback).inverse()).trace();
        while (std::abs(mixed_det - 1.) > TOLERANCE && iter < MAX_ITERATIONS) {
            used_mu = prev_mu + ((1./mixed_det) - 1.)/mixed_trace;
            if (std::isnan(used_mu) || std::isinf(used_mu))
                break;
            output_pullback = pullback + used_mu*pullback_transpose_inverse;
            prev_mu = used_mu;
            mixed_det = output_pullback.determinant();
            iter++;
        }
        output_pullback /= mixed_det > 0.0 ? std::sqrt(mixed_det) : 1.0;
        pullback = output_pullback;
    }
    COFLIPSolver2D(int nx, int ny, double L, int N, bool bc, Scheme s_scheme);
    ~COFLIPSolver2D() {};
    int ni, nj, nip, njp, nV, nF, nC;
    inline double lerp(const double& v0, const double& v1, double c);
    inline double bilerp(const double& v00, const double& v01, const double& v10, const double& v11, double cx, double cy);
    void semiLagAdvect(const Array2d &src, Array2d & dst, double dt, int ni, int nj, double off_x, double off_y);
    void applyBuoyancyForce(Array2d &v, double dt);
    void calculateCurl(bool do_star_fluxes=false, bool calculate_starvort_inverse=false);
    void projection(double tol, bool PURE_NEUMANN);
    void projectionWithVort(double tol);
    void buildMultiGridWithVort();
    void pressureProjectVelField();
    void seedParticles(int N, bool set_intensity_from_density=false);
    void setSparseParticles(int N, double magn);

    void advance(double dt, int frame, int delayed_reinit_frequency=1);
    void advanceReflectionPOLYFLIP(double dt, int currentframe, int delayed_reinit_frequency=1);
    void advanceCovectorPOLYFLIP(double dt, int currentframe, int delayed_reinit_frequency=1);
    void advectCovectorPOLYFLIPHelper(int stage_count, int currentframe, int delayed_reinit_frequency, double dt, bool do_all=false);
    void advanceCOFLIP(double dt, int currentframe, int delayed_reinit_frequency=1);
    void advancePolyPIC(double dt, int currentframe);
    std::tuple<double, double> getCasimirsAtCustomRes(int res_amp);

    void updateBackward(double dt, Array2d &back_x, Array2d &back_y);

    double computeFTLE(double& det, const Eigen::Matrix2d& pullback);

    void buildMultiGrid(bool PURE_NEUMANN);
    void applyVelocityBoundary(bool do_set_obstacle_vel = true);
    void sampleParticlesFromGrid();

    void takeDualwrtStar(Array2d &_u, Array2d &_v, bool update_uv, bool flux2circulation_or_circulation2flux);

    void setInitReyleighTaylor(double layer_height);
    void setInitLeapFrog(double amp, double dist1, double dist2, double rho_h, double rho_w);
    void setSmoke(double smoke_rise, double smoke_drop);
    void setBoundaryMask(std::function<double(Vec2d pos)> sdf=nullptr);

    double maxVel();
	inline Vec2d traceFE(double dt, const Vec2d &pos, const Array2d& un, const Array2d& vn);
	inline Vec2d traceRK3(double dt, const Vec2d &pos, const Array2d& un, const Array2d& vn);
    inline Vec2d traceRK4(double dt, const Vec2d& pos, const Array2d& un, const Array2d& vn);
    inline Eigen::Matrix2d pullbackRK4(double dt, Vec2d &inout_pos, const Eigen::Matrix2d& input_pullback, const Array2d& un, const Array2d& vn);
    inline Eigen::Matrix2d pullbackRK2(double dt, Vec2d &inout_pos, const Eigen::Matrix2d& input_pullback, const Array2d& un, const Array2d& vn);
    inline Eigen::Matrix2d pullbackRK1(double dt, Vec2d &inout_pos, const Eigen::Matrix2d& input_pullback, const Array2d& un, const Array2d& vn);
    inline Eigen::Matrix2d solvePullbackODE(double dt, Vec2d &inout_pos, const Eigen::Matrix2d& input_pullback, const Array2d& un, const Array2d& vn);
    inline Vec2d solveODE(double dt, const Vec2d &pos, const Array2d& un, const Array2d& vn);
    void emitSmoke();
    void outputDensity(std::string folder, std::string file, int i, bool color_density, bool do_tonemapping=false, bool scaleDensity=false);
    void outputVortVisualized(std::string folder, std::string file, int i);
    void outputVellVisualized(std::string folder, std::string file, int i, bool do_y_comp=false);
    void outputLevelset(std::string sdfFilename, int i);
    void outputEnergy(std::string filename, double curr_time);
    void outputVorticityIntegral(std::string filename, double curr_time, bool do_highres=false);
    void outputErrorTGV(std::string filename, int refinement_level);
    void initSmokePlume();
    
    color_bar cBar;
    int total_resampleCount = 0;
    int total_scalar_resample = 0;
    int resampleCount = 0;
    int frameCount = 0;
    void getCFL();
    Vec2d getVelocity(const Vec2d& pos, const Array2d& un, const Array2d& vn);
    Eigen::Matrix2d getJacobianVelocity(const Vec2d& pos, const Array2d& un, const Array2d& vn);
    double sampleField(const Vec2d& pos, const Array2d& field, bool use_uniform=false);
    Vec2d sampleGradientField(const Vec2d& pos, const Array2d& field, bool use_uniform=false);
    double sampleCrossHessianField(const Vec2d& pos, const Array2d& field, bool use_uniform=false);
    Vec2d getVelocityBSpline(const Vec2d& pos, const Array2d& un, const Array2d& vn, bool do_curlFree=false);
    Eigen::Matrix2d getJacobianVelocityBSpline(const Vec2d& pos, const Array2d& un, const Array2d& vn, bool do_curlFree=false);
    double sampleFieldBSpline(const Vec2d& pos, const Array2d& field, int selected_row);
    double sampleFieldBSplineCurlFree(const Vec2d& pos, const Array2d& field, int selected_row);
    double sampleFieldBSpline0form(const Vec2d& pos, const Array2d& field);
    double sampleFieldBSpline2form(const Vec2d& pos, const Array2d& field);
    double sampleGradientFieldBSpline(const Vec2d& pos, const Array2d& field, int selected_row, int selected_column);
    double sampleGradientFieldBSplineCurlFree(const Vec2d& pos, const Array2d& field, int selected_row, int selected_column);
    Vec2d getPointwiseDivCurl(const Vec2d& pos, const Array2d& un, const Array2d& vn);
    void advectCOFLIPHelper(int stage_count, int currentframe, int delayed_reinit_frequency, double dt, bool do_all=false);
    void solveInterpDaggerVelocity(int stage_count, int currentframe, int delayed_reinit_frequency);
    void solveInterpDaggerVorticity(bool do_pressureSolve=false);
    void integratePressureForce(Array2d& un, Array2d& vn);
    void setInitVelocityVortexSheet(double radius, double rotational_speed, double eps_smooth_gap);
    void setInitVelocityConvectingVortex(double U0, double Gamma, double sigma, const Vec2d& center);

    double h, h_uniform;
    double alpha_buoyancy, beta_buoyancy;
    Array2d u, v, u_temp, v_temp;
    std::vector<Eigen::VectorXd> harmonics_fluxes;
    std::vector<double> harmonics_coeff;
    Array2d u_save;
    Array2d v_save;
    Array2d rho, temperature, s_temp;
    Array2d curl;
    Array2d curl_mass;
    Array2c emitterMask;
    Array2c boundaryMask;
    Array2c boundaryMask_nodes;
    Eigen::VectorXd streamfunction;
    Eigen::VectorXd pressure;
    Eigen::VectorXd rhs;
    Eigen::VectorXd solved_visc_fluxes;

    //linear solver data
    Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> invstarflux_matrix;
    Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> dtranspose_matrix;
    Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> almostIdentityMatrixCells;
    std::vector<Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> > A_L;
    std::vector<Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> > R_L;
    std::vector<Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> > P_L;
    std::vector<Vec2i>                S_L;
    int total_level;
    Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> starflux_matrix;
    Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> dtranspose_matrix_curl;
    std::vector<Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> > A_L_curl;
    std::vector<Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> > R_L_curl;
    std::vector<Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> > P_L_curl;
    std::vector<Vec2i>                S_L_curl;
    int total_level_curl;
    std::vector<Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> > A_L_viscosity;
    std::vector<Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> > R_L_viscosity;
    std::vector<Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> > P_L_viscosity;
    std::vector<Vec2i>                S_L_viscosity;
    int total_level_viscosity;
    Eigen::IncompleteCholesky<double, Eigen::Lower, Eigen::AMDOrdering<std::ptrdiff_t> > _precond;
    Eigen::ConjugateGradient<Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t>, Eigen::Lower|Eigen::Upper, Eigen::IncompleteCholesky<double, Eigen::Lower, Eigen::AMDOrdering<std::ptrdiff_t> > > starflux_cg;
    //solver
    levelGen<double> mgLevelGenerator;
    std::vector<Eigen::Triplet<double> > tripletList;
    Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> starvort_matrix;
    Eigen::ConjugateGradient<Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t>, Eigen::Lower|Eigen::Upper, Eigen::IncompleteCholesky<double, Eigen::Lower, Eigen::AMDOrdering<std::ptrdiff_t> > > starvort_cg;

    Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> starpressure_matrix;
    Eigen::ConjugateGradient<Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t>, Eigen::Lower|Eigen::Upper, Eigen::IncompleteCholesky<double, Eigen::Lower, Eigen::AMDOrdering<std::ptrdiff_t> > > starpressure_cg;

    double _cfl;
    std::vector<CmapParticles> cParticles;
    Eigen::VectorXd fluxes, pre_solve_fluxes, fluxes_original;
    Eigen::VectorXd circulations;
    Eigen::VectorXd vorts, prev_vorts0form;
    Eigen::VectorXd pressure_field_0form;

    Array2d backward_x;
    Array2d backward_y;
    Array2d map_tempx;
    Array2d map_tempy;

    // fluid buffers
    Array2d u_mass;
    Array2d v_mass;

    bool use_neumann_boundary;
    Scheme sim_scheme;

    int projection_repeat_count = 1;
    bool use_pressure_solver = true;
    bool do_mass_lumping = false;
    bool use_DEC_diagonal_hodge_star = false;
    bool do_delta_circulation = false;
    double adaptive_reset_cutoff = 3.;
    TimeIntegration timeIntOrder = TimeIntegration::RK2;

    Vec3d HSVtoRGB(double H, double S,double V)
    {
        H = std::clamp(H, 0., 360.);
        S = std::clamp(S, 0., 100.);
        V = std::clamp(V, 0., 100.);

        double s = S/100;
        double v = V/100;
        double C = s*v;
        double X = C*(1-abs(fmod(H/60.0, 2)-1));
        double m = v-C;
        double r,g,b;
        if(H >= 0 && H < 60){
            r = C,g = X,b = 0;
        }
        else if(H >= 60 && H < 120){
            r = X,g = C,b = 0;
        }
        else if(H >= 120 && H < 180){
            r = 0,g = C,b = X;
        }
        else if(H >= 180 && H < 240){
            r = 0,g = X,b = C;
        }
        else if(H >= 240 && H < 300){
            r = X,g = 0,b = C;
        }
        else{
            r = C,g = 0,b = X;
        }
        double R = r+m;
        double G = g+m;
        double B = b+m;

        return Vec3d(R,G,B);
    }

    double substep = 1.0;

    bool do_uniform_particle_seeding = false;
    bool do_particle_sample_after_first = false;
    int precond_reset_frequency = 5;
    double viscosity = 0.;
    bool do_implicit = true;
    int min_PPC_count = 4;
private:
    double m_max_curl = 0.;

    int m_N;
    int bs_p = 3;
    double max_div_on_grid = 0.0;
    double energy_prev = 0;
};

#endif //COFLIPSOLVER2D_H
