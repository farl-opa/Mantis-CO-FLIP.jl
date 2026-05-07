#include "COFLIPSolver.h"
#include "../utils/bsplines.h"

COFLIPSolver::COFLIPSolver(int N, uint nx, uint ny, uint nz, MyReal L, MyReal vis_coeff, Scheme myscheme)
{
    _N = N;
    _nx = nx;
    _ny = ny;
    _nz = nz;
    _nF = (_nx+1)*_ny*_nz + _nx*(_ny+1)*_nz + _nx*_ny*(_nz+1);
    _nE = _nx*(_ny+1)*(_nz+1) + (_nx+1)*_ny*(_nz+1) + (_nx+1)*(_ny+1)*_nz;
    _nC = _nx*_ny*_nz;
    _nV = (_nx+1)*(_ny+1)*(_nz+1);
    _h_uniform = L / (MyReal)(_nx);
    max_v = 0.f;
    viscosity = vis_coeff;
    sim_scheme = myscheme;
    if (sim_scheme != Scheme::CO_FLIP || do_mass_lumping) {
        bs_p = 1;
    }
    _nxp = _nx+1-bs_p;
    _nyp = _ny+1-bs_p;
    _nzp = _nz+1-bs_p;
    _h = L / (MyReal)(_nxp);

    _un.init(_nx+1, _ny, _nz, _h, 0.5f, 0.f, 0.f);
    _vn.init(_nx, _ny+1, _nz, _h, 0.f, 0.5f, 0.f);
    _wn.init(_nx, _ny, _nz+1, _h, 0.f, 0.f, 0.5f);
    _vort_un.init(_nx, _ny+1, _nz+1, _h, 0.0f, 0.5f, 0.5f);
    _vort_vn.init(_nx+1, _ny, _nz+1, _h, 0.5f, 0.0f, 0.5f);
    _vort_wn.init(_nx+1, _ny+1, _nz, _h, 0.5f, 0.5f, 0.0f);

    _b_desc.init(_nx,_ny,_nz);

    _u_save.init(_nx+1, _ny, _nz, _h, 0.5f, 0.f, 0.f);
    _v_save.init(_nx, _ny+1, _nz, _h, 0.f, 0.5f, 0.f);
    _w_save.init(_nx, _ny, _nz+1, _h, 0.f, 0.f, 0.5f);
    streamforms.resize(_nE);
    pre_solve_fluxes.resize(_nF);
    streamforms.setZero();
    pre_solve_fluxes.setZero();
    fluxes_original.resize(_nF);
    fluxes_original.setZero();
    fluxes_midpoint.resize(_nF);
    fluxes_midpoint.setZero();
    if (sim_scheme == Scheme::CF_POLYFLIP) {
        _back_x.init(_nx, _ny, _nz, _h_uniform, 0.f, 0.f, 0.f);
        _back_y.init(_nx, _ny, _nz, _h_uniform, 0.f, 0.f, 0.f);
        _back_z.init(_nx, _ny, _nz, _h_uniform, 0.f, 0.f, 0.f);
        _back_temp_x.init(_nx, _ny, _nz, _h_uniform, 0.f, 0.f, 0.f);
        _back_temp_y.init(_nx, _ny, _nz, _h_uniform, 0.f, 0.f, 0.f);
        _back_temp_z.init(_nx, _ny, _nz, _h_uniform, 0.f, 0.f, 0.f);
    }
    int amp_factor = 2;
    _amped_h = _h_uniform / (double)amp_factor;
    _amped_nx = _nx * amp_factor;
    _amped_ny = _ny * amp_factor;
    _amped_nz = _nz * amp_factor;
    _rho.init(_amped_nx, _amped_ny, _amped_nz, _amped_h, 0.f, 0.f, 0.f);
    _T.init(_amped_nx, _amped_ny, _amped_nz, _amped_h, 0.f, 0.f, 0.f);
    fluxes.resize(_nF);
    circulations.resize(_nF);
    vorts.resize(_nE);
    prev_vorts1form.resize(_nE);
    fluxes.setZero();
    circulations.setZero();
    vorts.setZero();
    prev_vorts1form.setZero();
}

void COFLIPSolver::advance(int framenum, MyReal dt)
{
    switch (sim_scheme)
    {
        case R_POLYFLIP:
            advanceReflectionPOLYFLIP(framenum, dt);
            break;
        case CF_POLYFLIP:
        case POLYFLIP:
        case POLYPIC:
            advanceCovectorPOLYFLIP(framenum, dt);
            break;
        case CO_FLIP:
            advanceCOFLIP(framenum, dt);
            break;
        default:
            break;
    }
}

inline MyReal COFLIPSolver::lerp(const MyReal& v0, const MyReal& v1, MyReal c)
{
    return (1.0-c)*v0+c*v1;
}

inline MyReal COFLIPSolver::bilerp(const MyReal& v00, const MyReal& v01, const MyReal& v10, const MyReal& v11, MyReal cx, MyReal cy)
{
    return lerp(lerp(v00,v01,cx), lerp(v10,v11,cx),cy);
}

inline MyReal COFLIPSolver::trilerp(
    const MyReal& v000, const MyReal& v001,
    const MyReal& v010, const MyReal& v011,
    const MyReal& v100, const MyReal& v101,
    const MyReal& v110, const MyReal& v111,
    MyReal cx, MyReal cy, MyReal cz)
{
    return lerp(bilerp(v000, v001, v010, v011, cx, cy),
                bilerp(v100, v101, v110, v111, cx, cy),
                cz);
}

Vec<3, MyReal> COFLIPSolver::getVorticity(const Vec<3, MyReal>& pos, const Buffer3D<MyReal>& vort_un, const Buffer3D<MyReal>& vort_vn, const Buffer3D<MyReal>& vort_wn)
{
    MyReal vort_u_sample, vort_v_sample, vort_w_sample;
    //offset of u, we are in a staggered grid
    Vec<3, MyReal> upos = pos - Vec<3, MyReal>(0.5f*_h_uniform, 0.f, 0.f);
    vort_u_sample = sampleField(upos, vort_un, true);

    //offset of v, we are in a staggered grid
    Vec<3, MyReal> vpos = pos - Vec<3, MyReal>(0.f, 0.5f*_h_uniform, 0.f);
    vort_v_sample = sampleField(vpos, vort_vn, true);

    //offset of v, we are in a staggered grid
    Vec<3, MyReal> wpos = pos - Vec<3, MyReal>(0.f, 0.f, 0.5f*_h_uniform);
    vort_w_sample = sampleField(wpos, vort_wn, true);

    return Vec<3, MyReal>(vort_u_sample, vort_v_sample, vort_w_sample);
}

Vec<3, MyReal> COFLIPSolver::getVelocity(const Vec<3, MyReal>& pos, const Buffer3D<MyReal>& un, const Buffer3D<MyReal>& vn, const Buffer3D<MyReal>& wn)
{
    MyReal u_sample, v_sample, w_sample;
    //offset of u, we are in a staggered grid
    Vec<3, MyReal> upos = pos - Vec<3, MyReal>(0.0f, 0.5f*_h_uniform, 0.5f*_h_uniform);
    u_sample = sampleField(upos, un, true);

    //offset of v, we are in a staggered grid
    Vec<3, MyReal> vpos = pos - Vec<3, MyReal>(0.5f*_h_uniform, 0.0f, 0.5f*_h_uniform);
    v_sample = sampleField(vpos, vn, true);

    //offset of v, we are in a staggered grid
    Vec<3, MyReal> wpos = pos - Vec<3, MyReal>(0.5f*_h_uniform, 0.5f*_h_uniform, 0.0f);
    w_sample = sampleField(wpos, wn, true);

    return Vec<3, MyReal>(u_sample, v_sample, w_sample);
}

Eigen::Matrix3<MyReal> COFLIPSolver::getJacobianVelocity(const Vec<3, MyReal>& pos, const Buffer3D<MyReal>& un, const Buffer3D<MyReal>& vn, const Buffer3D<MyReal>& wn)
{
    Vec<3, MyReal> grad_u_sample, grad_v_sample, grad_w_sample;
    //offset of u, we are in a staggered grid
    Vec<3, MyReal> upos = pos - Vec<3, MyReal>(0.0f, 0.5f*_h_uniform, 0.5f*_h_uniform);
    grad_u_sample = sampleGradientField(upos, un, true);

    //offset of v, we are in a staggered grid
    Vec<3, MyReal> vpos = pos - Vec<3, MyReal>(0.5f*_h_uniform, 0.0f, 0.5f*_h_uniform);
    grad_v_sample = sampleGradientField(vpos, vn, true);

    //offset of v, we are in a staggered grid
    Vec<3, MyReal> wpos = pos - Vec<3, MyReal>(0.5f*_h_uniform, 0.5f*_h_uniform, 0.0f);
    grad_w_sample = sampleGradientField(wpos, wn, true);

    Eigen::Matrix3<MyReal> jacobian;
    jacobian << grad_u_sample[0], grad_u_sample[1], grad_u_sample[2],
                grad_v_sample[0], grad_v_sample[1], grad_v_sample[2],
                grad_w_sample[0], grad_w_sample[1], grad_w_sample[2];
    return jacobian;
}

MyReal COFLIPSolver::getPointwiseDiv(const Vec<3, MyReal>& pos, const Buffer3D<MyReal>& un, const Buffer3D<MyReal>& vn, const Buffer3D<MyReal>& wn)
{
    Eigen::Matrix3d jac;
    if (bs_p == 1) {
        jac = (getJacobianVelocity(pos, un, vn, wn)).cast<double>();
    } else {
        jac = (getJacobianVelocityBSpline(pos, un, vn, wn)).cast<double>();
    }
    MyReal div = jac.trace();
    return div;
}

MyReal COFLIPSolver::sampleField(const Vec<3, MyReal>& pos, const Buffer3D<MyReal>& field, bool use_uniform, bool amp_res)
{
    Vec<3, MyReal> spos = pos;
    MyReal used_h = use_uniform ? _h_uniform : _h;
    if (amp_res && use_uniform) {
        int ratio = field._nx/_nx;
        used_h = _h_uniform/(double)ratio;
    }
    int i = std::floor(spos.v[0] / used_h),
        j = std::floor(spos.v[1] / used_h),
        k = std::floor(spos.v[2] / used_h);
    MyReal alpha = spos.v[0] / used_h - (MyReal)i, 
          beta = spos.v[1] / used_h - (MyReal)j,
          gamma = spos.v[2] / used_h - (MyReal)k;
    return trilerp(
            field.boundedAt(i, j, k), field.boundedAt(i + 1, j, k),
            field.boundedAt(i, j + 1, k), field.boundedAt(i + 1, j + 1, k),
            field.boundedAt(i, j, k + 1), field.boundedAt(i + 1, j, k + 1),
            field.boundedAt(i, j + 1, k + 1), field.boundedAt(i + 1, j + 1, k + 1),
            alpha, beta, gamma);
}

Vec<3, MyReal> COFLIPSolver::sampleGradientField(const Vec<3, MyReal>& pos, const Buffer3D<MyReal>& field, bool use_uniform, bool amp_res)
{
    Vec<3, MyReal> spos = pos;
    MyReal used_h = use_uniform ? _h_uniform : _h;
    if (amp_res && use_uniform) {
        int ratio = field._nx/_nx;
        used_h = _h_uniform/(double)ratio;
    }
    int i = std::floor(spos.v[0] / used_h), 
        j = std::floor(spos.v[1] / used_h),
        k = std::floor(spos.v[2] / used_h);
    MyReal alpha = spos.v[0] / used_h - (MyReal)i, 
          beta = spos.v[1] / used_h - (MyReal)j,
          gamma = spos.v[2] / used_h - (MyReal)k;
    return Vec<3, MyReal>(
            bilerp(field.boundedAt(i + 1, j, k) - field.boundedAt(i, j, k),
                   field.boundedAt(i + 1, j + 1, k) - field.boundedAt(i, j + 1, k),
                   field.boundedAt(i + 1, j, k + 1) - field.boundedAt(i, j, k + 1),
                   field.boundedAt(i + 1, j + 1, k + 1) - field.boundedAt(i, j + 1, k + 1),
                   beta, gamma),
            bilerp(field.boundedAt(i, j + 1, k) - field.boundedAt(i, j, k),
                   field.boundedAt(i, j + 1, k + 1) - field.boundedAt(i, j, k + 1),
                   field.boundedAt(i + 1, j + 1, k) - field.boundedAt(i + 1, j, k),
                   field.boundedAt(i + 1, j + 1, k + 1) - field.boundedAt(i + 1, j, k + 1),
                   gamma, alpha),
            bilerp(field.boundedAt(i, j, k + 1) - field.boundedAt(i, j, k),
                   field.boundedAt(i + 1, j, k + 1) - field.boundedAt(i + 1, j, k),
                   field.boundedAt(i, j + 1, k + 1) - field.boundedAt(i, j + 1, k),
                   field.boundedAt(i + 1, j + 1, k + 1) - field.boundedAt(i + 1, j + 1, k),
                   alpha, beta)) / used_h;
}

Vec<4, MyReal> COFLIPSolver::sampleCrossHessianField(const Vec<3, MyReal>& pos, const Buffer3D<MyReal>& field, bool use_uniform, bool amp_res)
{
    Vec<3, MyReal> spos = pos;
    MyReal used_h = use_uniform ? _h_uniform : _h;
    if (amp_res && use_uniform) {
        int ratio = field._nx/_nx;
        used_h = _h_uniform/(double)ratio;
    }
    int i = std::floor(spos.v[0] / used_h),
        j = std::floor(spos.v[1] / used_h),
        k = std::floor(spos.v[2] / used_h);
    MyReal alpha = spos.v[0] / used_h - (MyReal)i, 
          beta = spos.v[1] / used_h - (MyReal)j,
          gamma = spos.v[2] / used_h - (MyReal)k;
    return Vec<4, MyReal>(
            lerp(-(field.boundedAt(i + 1, j, k) - field.boundedAt(i, j, k)) +
                   field.boundedAt(i + 1, j + 1, k) - field.boundedAt(i, j + 1, k),
                 -(field.boundedAt(i + 1, j, k + 1) - field.boundedAt(i, j, k + 1)) +
                   field.boundedAt(i + 1, j + 1, k + 1) - field.boundedAt(i, j + 1, k + 1),
                   gamma) / (used_h*used_h),
            lerp(-(field.boundedAt(i, j + 1, k) - field.boundedAt(i, j, k)) +
                   field.boundedAt(i, j + 1, k + 1) - field.boundedAt(i, j, k + 1),
                 -(field.boundedAt(i + 1, j + 1, k) - field.boundedAt(i + 1, j, k)) +
                   field.boundedAt(i + 1, j + 1, k + 1) - field.boundedAt(i + 1, j, k + 1),
                   alpha) / (used_h*used_h),
            lerp(-(field.boundedAt(i, j, k + 1) - field.boundedAt(i, j, k)) +
                   field.boundedAt(i + 1, j, k + 1) - field.boundedAt(i + 1, j, k),
                 -(field.boundedAt(i, j + 1, k + 1) - field.boundedAt(i, j + 1, k)) +
                   field.boundedAt(i + 1, j + 1, k + 1) - field.boundedAt(i + 1, j + 1, k),
                   beta) / (used_h*used_h),
            (-(-(field.boundedAt(i + 1, j, k) - field.boundedAt(i, j, k)) +
                 field.boundedAt(i + 1, j + 1, k) - field.boundedAt(i, j + 1, k)) +
              (-(field.boundedAt(i + 1, j, k + 1) - field.boundedAt(i, j, k + 1)) +
                 field.boundedAt(i + 1, j + 1, k + 1) - field.boundedAt(i, j + 1, k + 1))) / (used_h*used_h*used_h)) ;
}

Vec<3, MyReal> COFLIPSolver::getVorticityBSpline(const Vec<3, MyReal>& pos, const Buffer3D<MyReal>& vort_un, const Buffer3D<MyReal>& vort_vn, const Buffer3D<MyReal>& vort_wn)
{
    if (bs_p == 1) {
        return getVorticity(pos, vort_un, vort_vn, vort_wn);
    }

    MyReal vort_u_sample, vort_v_sample, vort_w_sample;
    vort_u_sample = sampleFieldBSplineVort(pos, vort_un, 0);
    vort_v_sample = sampleFieldBSplineVort(pos, vort_vn, 1);
    vort_w_sample = sampleFieldBSplineVort(pos, vort_wn, 2);

    return Vec<3, MyReal>(vort_u_sample, vort_v_sample, vort_w_sample);
}

Vec<3, MyReal> COFLIPSolver::getVelocityBSpline(const Vec<3, MyReal>& pos, const Buffer3D<MyReal>& un, const Buffer3D<MyReal>& vn, const Buffer3D<MyReal>& wn)
{
    if (bs_p == 1) {
        return getVelocity(pos, un, vn, wn);
    }

    MyReal u_sample, v_sample, w_sample;
    u_sample = sampleFieldBSpline(pos, un, 0);
    v_sample = sampleFieldBSpline(pos, vn, 1);
    w_sample = sampleFieldBSpline(pos, wn, 2);

    return Vec<3, MyReal>(u_sample, v_sample, w_sample);
}

Eigen::Matrix3<MyReal> COFLIPSolver::getJacobianVelocityBSpline(const Vec<3, MyReal>& pos, const Buffer3D<MyReal>& un, const Buffer3D<MyReal>& vn, const Buffer3D<MyReal>& wn)
{
    if (bs_p == 1) {
        return getJacobianVelocity(pos, un, vn, wn);
    }

    Vec<3, MyReal> grad_u_sample, grad_v_sample, grad_w_sample;
    grad_u_sample = Vec<3, MyReal>(
        sampleGradientFieldBSpline(pos, un, 0, 0),
        sampleGradientFieldBSpline(pos, un, 0, 1),
        sampleGradientFieldBSpline(pos, un, 0, 2));
    grad_v_sample = Vec<3, MyReal>(
        sampleGradientFieldBSpline(pos, vn, 1, 0),
        sampleGradientFieldBSpline(pos, vn, 1, 1),
        sampleGradientFieldBSpline(pos, vn, 1, 2));
    grad_w_sample = Vec<3, MyReal>(
        sampleGradientFieldBSpline(pos, wn, 2, 0),
        sampleGradientFieldBSpline(pos, wn, 2, 1),
        sampleGradientFieldBSpline(pos, wn, 2, 2));

    Eigen::Matrix3<MyReal> jacobian;
    jacobian << grad_u_sample[0], grad_u_sample[1], grad_u_sample[2],
                grad_v_sample[0], grad_v_sample[1], grad_v_sample[2],
                grad_w_sample[0], grad_w_sample[1], grad_w_sample[2];
    return jacobian;
}

MyReal COFLIPSolver::sampleFieldBSplineVort(const Vec<3, MyReal>& pos, const Buffer3D<MyReal>& field, int selected_row)
{
    double xpos = pos.v[0];
    double ypos = pos.v[1];
    double zpos = pos.v[2];
    int i = std::floor(xpos / _h), 
        j = std::floor(ypos / _h),
        k = std::floor(zpos / _h);
    double alpha = xpos / _h - (double)i, 
           beta = ypos / _h - (double)j, 
           gamma = zpos / _h - (double)k;

    std::array<int, 4> xoffsets, yoffsets, zoffsets, 
                       xshifts1, yshifts1, zshifts1,
                       xshifts2, yshifts2, zshifts2;
    double primary_t, secondary_t, tertiary_t;
    int idx, n_idx, other1_idx, other1_nidx, other2_idx, other2_nidx;
    if (selected_row == 0) {
        xoffsets = {0, 1, 2, 3};
        yoffsets = {0, 0, 0, 0};
        zoffsets = {0, 0, 0, 0};
        //--
        xshifts1 = {0, 0, 0, 0};
        yshifts1 = {0, 1, 2, 3};
        zshifts1 = {0, 0, 0, 0};
        //--
        xshifts2 = {0, 0, 0, 0};
        yshifts2 = {0, 0, 0, 0};
        zshifts2 = {0, 1, 2, 3};
        primary_t = alpha;
        secondary_t = beta;
        tertiary_t = gamma;
        idx = i;
        other1_idx = j;
        other2_idx = k;
        n_idx = _nxp;
        other1_nidx = _nyp;
        other2_nidx = _nzp;
    } else if (selected_row == 1) {
        xoffsets = {0, 0, 0, 0};
        yoffsets = {0, 1, 2, 3};
        zoffsets = {0, 0, 0, 0};
        //--
        xshifts1 = {0, 0, 0, 0};
        yshifts1 = {0, 0, 0, 0};
        zshifts1 = {0, 1, 2, 3};
        //--
        xshifts2 = {0, 1, 2, 3};
        yshifts2 = {0, 0, 0, 0};
        zshifts2 = {0, 0, 0, 0};
        primary_t = beta;
        secondary_t = gamma;
        tertiary_t = alpha;
        idx = j;
        other1_idx = k;
        other2_idx = i;
        n_idx = _nyp;
        other1_nidx = _nzp;
        other2_nidx = _nxp;
    } else {
        xoffsets = {0, 0, 0, 0};
        yoffsets = {0, 0, 0, 0};
        zoffsets = {0, 1, 2, 3};
        //--
        xshifts1 = {0, 1, 2, 3};
        yshifts1 = {0, 0, 0, 0};
        zshifts1 = {0, 0, 0, 0};
        //--
        xshifts2 = {0, 0, 0, 0};
        yshifts2 = {0, 1, 2, 3};
        zshifts2 = {0, 0, 0, 0};
        primary_t = gamma;
        secondary_t = alpha;
        tertiary_t = beta;
        idx = k;
        other1_idx = i;
        other2_idx = j;
        n_idx = _nzp;
        other1_nidx = _nxp;
        other2_nidx = _nyp;
    }

    int primary_bdy_version = 0;
    if (idx == 0 || idx == (n_idx-1)) {
        primary_bdy_version = 2;
        if (idx == (n_idx-1)) {
            primary_t = std::clamp(1.0 - primary_t, 0.0, 1.0);
            std::reverse(xoffsets.begin(), xoffsets.begin()+bs_p);
            std::reverse(yoffsets.begin(), yoffsets.begin()+bs_p);
            std::reverse(zoffsets.begin(), zoffsets.begin()+bs_p);
        }
    } else if (bs_p != 2 && (idx == 1 || idx == (n_idx-2))) {
        primary_bdy_version = 1;
        if (idx == (n_idx-2)) {
            primary_t = std::clamp(1.0 - primary_t, 0.0, 1.0);
            std::reverse(xoffsets.begin(), xoffsets.begin()+bs_p);
            std::reverse(yoffsets.begin(), yoffsets.begin()+bs_p);
            std::reverse(zoffsets.begin(), zoffsets.begin()+bs_p);
        }
    }

    int secondary_bdy_version = 0;
    if (other1_idx == 0 || other1_idx == (other1_nidx-1)) {
        secondary_bdy_version = 2;
        if (other1_idx == (other1_nidx-1)) {
            secondary_t = std::clamp(1.0 - secondary_t, 0.0, 1.0);
            std::reverse(xshifts1.begin(), xshifts1.begin()+bs_p+1);
            std::reverse(yshifts1.begin(), yshifts1.begin()+bs_p+1);
            std::reverse(zshifts1.begin(), zshifts1.begin()+bs_p+1);
        }
    } else if (bs_p != 2 && (other1_idx == 1 || other1_idx == (other1_nidx-2))) {
        secondary_bdy_version = 1;
        if (other1_idx == (other1_nidx-2)) {
            secondary_t = std::clamp(1.0 - secondary_t, 0.0, 1.0);
            std::reverse(xshifts1.begin(), xshifts1.begin()+bs_p+1);
            std::reverse(yshifts1.begin(), yshifts1.begin()+bs_p+1);
            std::reverse(zshifts1.begin(), zshifts1.begin()+bs_p+1);
        }
    }

    int tertiary_bdy_version = 0;
    if (other2_idx == 0 || other2_idx == (other2_nidx-1)) {
        tertiary_bdy_version = 2;
        if (other2_idx == (other2_nidx-1)) {
            tertiary_t = std::clamp(1.0 - tertiary_t, 0.0, 1.0);
            std::reverse(xshifts2.begin(), xshifts2.begin()+bs_p+1);
            std::reverse(yshifts2.begin(), yshifts2.begin()+bs_p+1);
            std::reverse(zshifts2.begin(), zshifts2.begin()+bs_p+1);
        }
    } else if (bs_p != 2 && (other2_idx == 1 || other2_idx == (other2_nidx-2))) {
        tertiary_bdy_version = 1;
        if (other2_idx == (other2_nidx-2)) {
            tertiary_t = std::clamp(1.0 - tertiary_t, 0.0, 1.0);
            std::reverse(xshifts2.begin(), xshifts2.begin()+bs_p+1);
            std::reverse(yshifts2.begin(), yshifts2.begin()+bs_p+1);
            std::reverse(zshifts2.begin(), zshifts2.begin()+bs_p+1);
        }
    }

    std::array<double, 3> basesp;
    std::array<double, 4> basess1;
    std::array<double, 4> basess2;
    if (bs_p == 2) {
        basesp = {D10(primary_t, primary_bdy_version), D11(primary_t), 0.0};
        basess1 = {B20(secondary_t, secondary_bdy_version), B21(secondary_t, secondary_bdy_version), B22(secondary_t), 0.0};
        basess2 = {B20(tertiary_t, tertiary_bdy_version), B21(tertiary_t, tertiary_bdy_version), B22(tertiary_t), 0.0};
    } else {
        basesp = {D20(primary_t, primary_bdy_version), D21(primary_t, primary_bdy_version), D22(primary_t)};
        basess1 = {B30(secondary_t, secondary_bdy_version), B31(secondary_t, secondary_bdy_version), B32(secondary_t, secondary_bdy_version), B33(secondary_t)};
        basess2 = {B30(tertiary_t, tertiary_bdy_version), B31(tertiary_t, tertiary_bdy_version), B32(tertiary_t, tertiary_bdy_version), B33(tertiary_t)};
    }
    double result = 0.0;
    for (int l = 0; l < bs_p; ++l) {
        for (int m = 0; m < bs_p+1; ++m) {
            for (int q = 0; q < bs_p+1; ++q) {
                result += field(i + xoffsets[l] + xshifts1[m] + xshifts2[q], 
                                j + yoffsets[l] + yshifts1[m] + yshifts2[q], 
                                k + zoffsets[l] + zshifts1[m] + zshifts2[q]) * basesp[l] * basess1[m] * basess2[q];
            }
        }
    }
    return result;
}

MyReal COFLIPSolver::sampleFieldBSpline(const Vec<3, MyReal>& pos, const Buffer3D<MyReal> &field, int selected_row)
{
    double xpos = pos.v[0];
    double ypos = pos.v[1];
    double zpos = pos.v[2];
    int i = std::floor(xpos / _h), 
        j = std::floor(ypos / _h),
        k = std::floor(zpos / _h);
    double alpha = xpos / _h - (double)i, 
           beta = ypos / _h - (double)j, 
           gamma = zpos / _h - (double)k;

    std::array<int, 4> xoffsets, yoffsets, zoffsets, 
                       xshifts1, yshifts1, zshifts1,
                       xshifts2, yshifts2, zshifts2;
    double primary_t, secondary_t, tertiary_t;
    int idx, n_idx, other1_idx, other1_nidx, other2_idx, other2_nidx;
    if (selected_row == 0) {
        xoffsets = {0, 1, 2, 3};
        yoffsets = {0, 0, 0, 0};
        zoffsets = {0, 0, 0, 0};
        //--
        xshifts1 = {0, 0, 0, 0};
        yshifts1 = {0, 1, 2, 3};
        zshifts1 = {0, 0, 0, 0};
        //--
        xshifts2 = {0, 0, 0, 0};
        yshifts2 = {0, 0, 0, 0};
        zshifts2 = {0, 1, 2, 3};
        primary_t = alpha;
        secondary_t = beta;
        tertiary_t = gamma;
        idx = i;
        other1_idx = j;
        other2_idx = k;
        n_idx = _nxp;
        other1_nidx = _nyp;
        other2_nidx = _nzp;
    } else if (selected_row == 1) {
        xoffsets = {0, 0, 0, 0};
        yoffsets = {0, 1, 2, 3};
        zoffsets = {0, 0, 0, 0};
        //--
        xshifts1 = {0, 0, 0, 0};
        yshifts1 = {0, 0, 0, 0};
        zshifts1 = {0, 1, 2, 3};
        //--
        xshifts2 = {0, 1, 2, 3};
        yshifts2 = {0, 0, 0, 0};
        zshifts2 = {0, 0, 0, 0};
        primary_t = beta;
        secondary_t = gamma;
        tertiary_t = alpha;
        idx = j;
        other1_idx = k;
        other2_idx = i;
        n_idx = _nyp;
        other1_nidx = _nzp;
        other2_nidx = _nxp;
    } else {
        xoffsets = {0, 0, 0, 0};
        yoffsets = {0, 0, 0, 0};
        zoffsets = {0, 1, 2, 3};
        //--
        xshifts1 = {0, 1, 2, 3};
        yshifts1 = {0, 0, 0, 0};
        zshifts1 = {0, 0, 0, 0};
        //--
        xshifts2 = {0, 0, 0, 0};
        yshifts2 = {0, 1, 2, 3};
        zshifts2 = {0, 0, 0, 0};
        primary_t = gamma;
        secondary_t = alpha;
        tertiary_t = beta;
        idx = k;
        other1_idx = i;
        other2_idx = j;
        n_idx = _nzp;
        other1_nidx = _nxp;
        other2_nidx = _nyp;
    }

    int primary_bdy_version = 0;
    if (idx == 0 || idx == (n_idx-1)) {
        primary_bdy_version = 2;
        if (idx == (n_idx-1)) {
            primary_t = std::clamp(1.0 - primary_t, 0.0, 1.0);
            std::reverse(xoffsets.begin(), xoffsets.begin()+bs_p+1);
            std::reverse(yoffsets.begin(), yoffsets.begin()+bs_p+1);
            std::reverse(zoffsets.begin(), zoffsets.begin()+bs_p+1);
        }
    } else if (bs_p != 2 && (idx == 1 || idx == (n_idx-2))) {
        primary_bdy_version = 1;
        if (idx == (n_idx-2)) {
            primary_t = std::clamp(1.0 - primary_t, 0.0, 1.0);
            std::reverse(xoffsets.begin(), xoffsets.begin()+bs_p+1);
            std::reverse(yoffsets.begin(), yoffsets.begin()+bs_p+1);
            std::reverse(zoffsets.begin(), zoffsets.begin()+bs_p+1);
        }
    }

    int secondary_bdy_version = 0;
    if (other1_idx == 0 || other1_idx == (other1_nidx-1)) {
        secondary_bdy_version = 2;
        if (other1_idx == (other1_nidx-1)) {
            secondary_t = std::clamp(1.0 - secondary_t, 0.0, 1.0);
            std::reverse(xshifts1.begin(), xshifts1.begin()+bs_p);
            std::reverse(yshifts1.begin(), yshifts1.begin()+bs_p);
            std::reverse(zshifts1.begin(), zshifts1.begin()+bs_p);
        }
    } else if (bs_p != 2 && (other1_idx == 1 || other1_idx == (other1_nidx-2))) {
        secondary_bdy_version = 1;
        if (other1_idx == (other1_nidx-2)) {
            secondary_t = std::clamp(1.0 - secondary_t, 0.0, 1.0);
            std::reverse(xshifts1.begin(), xshifts1.begin()+bs_p);
            std::reverse(yshifts1.begin(), yshifts1.begin()+bs_p);
            std::reverse(zshifts1.begin(), zshifts1.begin()+bs_p);
        }
    }

    int tertiary_bdy_version = 0;
    if (other2_idx == 0 || other2_idx == (other2_nidx-1)) {
        tertiary_bdy_version = 2;
        if (other2_idx == (other2_nidx-1)) {
            tertiary_t = std::clamp(1.0 - tertiary_t, 0.0, 1.0);
            std::reverse(xshifts2.begin(), xshifts2.begin()+bs_p);
            std::reverse(yshifts2.begin(), yshifts2.begin()+bs_p);
            std::reverse(zshifts2.begin(), zshifts2.begin()+bs_p);
        }
    } else if (bs_p != 2 && (other2_idx == 1 || other2_idx == (other2_nidx-2))) {
        tertiary_bdy_version = 1;
        if (other2_idx == (other2_nidx-2)) {
            tertiary_t = std::clamp(1.0 - tertiary_t, 0.0, 1.0);
            std::reverse(xshifts2.begin(), xshifts2.begin()+bs_p);
            std::reverse(yshifts2.begin(), yshifts2.begin()+bs_p);
            std::reverse(zshifts2.begin(), zshifts2.begin()+bs_p);
        }
    }

    std::array<double, 4> basesp;
    std::array<double, 3> basess1;
    std::array<double, 3> basess2;
    if (bs_p == 2) {
        basesp = {B20(primary_t, primary_bdy_version), B21(primary_t, primary_bdy_version), B22(primary_t), 0.0};
        basess1 = {D10(secondary_t, secondary_bdy_version), D11(secondary_t), 0.0};
        basess2 = {D10(tertiary_t, tertiary_bdy_version), D11(tertiary_t), 0.0};
    } else {
        basesp = {B30(primary_t, primary_bdy_version), B31(primary_t, primary_bdy_version), B32(primary_t, primary_bdy_version), B33(primary_t)};
        basess1 = {D20(secondary_t, secondary_bdy_version), D21(secondary_t, secondary_bdy_version), D22(secondary_t)};
        basess2 = {D20(tertiary_t, tertiary_bdy_version), D21(tertiary_t, tertiary_bdy_version), D22(tertiary_t)};
    }
    double result = 0.0;
    for (int l = 0; l < bs_p+1; ++l) {
        for (int m = 0; m < bs_p; ++m) {
            for (int q = 0; q < bs_p; ++q) {
                result += field(i + xoffsets[l] + xshifts1[m] + xshifts2[q], 
                                j + yoffsets[l] + yshifts1[m] + yshifts2[q], 
                                k + zoffsets[l] + zshifts1[m] + zshifts2[q]) * basesp[l] * basess1[m] * basess2[q];
            }
        }
    }
    return result;
}

MyReal COFLIPSolver::sampleGradientFieldBSpline(const Vec<3, MyReal>& pos, const Buffer3D<MyReal> &field, int selected_row, int selected_column)
{
    double xpos = pos.v[0];
    double ypos = pos.v[1];
    double zpos = pos.v[2];
    int i = std::floor(xpos / _h), 
        j = std::floor(ypos / _h),
        k = std::floor(zpos / _h);
    double alpha = xpos / _h - (double)i, 
           beta = ypos / _h - (double)j, 
           gamma = zpos / _h - (double)k;

    std::array<int, 4> xoffsets, yoffsets, zoffsets, 
                       xshifts1, yshifts1, zshifts1,
                       xshifts2, yshifts2, zshifts2;
    double primary_t, secondary_t, tertiary_t;
    int idx, n_idx, other1_idx, other1_nidx, other2_idx, other2_nidx;
    if (selected_row == 0) {
        xoffsets = {0, 1, 2, 3};
        yoffsets = {0, 0, 0, 0};
        zoffsets = {0, 0, 0, 0};
        //--
        xshifts1 = {0, 0, 0, 0};
        yshifts1 = {0, 1, 2, 3};
        zshifts1 = {0, 0, 0, 0};
        //--
        xshifts2 = {0, 0, 0, 0};
        yshifts2 = {0, 0, 0, 0};
        zshifts2 = {0, 1, 2, 3};
        primary_t = alpha;
        secondary_t = beta;
        tertiary_t = gamma;
        idx = i;
        other1_idx = j;
        other2_idx = k;
        n_idx = _nxp;
        other1_nidx = _nyp;
        other2_nidx = _nzp;
    } else if (selected_row == 1) {
        xoffsets = {0, 0, 0, 0};
        yoffsets = {0, 1, 2, 3};
        zoffsets = {0, 0, 0, 0};
        //--
        xshifts1 = {0, 0, 0, 0};
        yshifts1 = {0, 0, 0, 0};
        zshifts1 = {0, 1, 2, 3};
        //--
        xshifts2 = {0, 1, 2, 3};
        yshifts2 = {0, 0, 0, 0};
        zshifts2 = {0, 0, 0, 0};
        primary_t = beta;
        secondary_t = gamma;
        tertiary_t = alpha;
        idx = j;
        other1_idx = k;
        other2_idx = i;
        n_idx = _nyp;
        other1_nidx = _nzp;
        other2_nidx = _nxp;
    } else {
        xoffsets = {0, 0, 0, 0};
        yoffsets = {0, 0, 0, 0};
        zoffsets = {0, 1, 2, 3};
        //--
        xshifts1 = {0, 1, 2, 3};
        yshifts1 = {0, 0, 0, 0};
        zshifts1 = {0, 0, 0, 0};
        //--
        xshifts2 = {0, 0, 0, 0};
        yshifts2 = {0, 1, 2, 3};
        zshifts2 = {0, 0, 0, 0};
        primary_t = gamma;
        secondary_t = alpha;
        tertiary_t = beta;
        idx = k;
        other1_idx = i;
        other2_idx = j;
        n_idx = _nzp;
        other1_nidx = _nxp;
        other2_nidx = _nyp;
    }

    int case_index = (selected_column-selected_row+3)%3;

    int primary_bdy_version = 0;
    double mult = 1.0;
    if (idx == 0 || idx == (n_idx-1)) {
        primary_bdy_version = 2;
        if (idx == (n_idx-1)) {
            primary_t = std::clamp(1.0 - primary_t, 0.0, 1.0);
            std::reverse(xoffsets.begin(), xoffsets.begin()+bs_p+1);
            std::reverse(yoffsets.begin(), yoffsets.begin()+bs_p+1);
            std::reverse(zoffsets.begin(), zoffsets.begin()+bs_p+1);
            mult = case_index == 0 ? -mult : mult;
        }
    } else if (bs_p != 2 && (idx == 1 || idx == (n_idx-2))) {
        primary_bdy_version = 1;
        if (idx == (n_idx-2)) {
            primary_t = std::clamp(1.0 - primary_t, 0.0, 1.0);
            std::reverse(xoffsets.begin(), xoffsets.begin()+bs_p+1);
            std::reverse(yoffsets.begin(), yoffsets.begin()+bs_p+1);
            std::reverse(zoffsets.begin(), zoffsets.begin()+bs_p+1);
            mult = case_index == 0 ? -mult : mult;
        }
    }

    int secondary_bdy_version = 0;
    if (other1_idx == 0 || other1_idx == (other1_nidx-1)) {
        secondary_bdy_version = 2;
        if (other1_idx == (other1_nidx-1)) {
            secondary_t = std::clamp(1.0 - secondary_t, 0.0, 1.0);
            std::reverse(xshifts1.begin(), xshifts1.begin()+bs_p);
            std::reverse(yshifts1.begin(), yshifts1.begin()+bs_p);
            std::reverse(zshifts1.begin(), zshifts1.begin()+bs_p);
            mult = case_index == 1 ? -mult : mult;
        }
    } else if (bs_p != 2 && (other1_idx == 1 || other1_idx == (other1_nidx-2))) {
        secondary_bdy_version = 1;
        if (other1_idx == (other1_nidx-2)) {
            secondary_t = std::clamp(1.0 - secondary_t, 0.0, 1.0);
            std::reverse(xshifts1.begin(), xshifts1.begin()+bs_p);
            std::reverse(yshifts1.begin(), yshifts1.begin()+bs_p);
            std::reverse(zshifts1.begin(), zshifts1.begin()+bs_p);
            mult = case_index == 1 ? -mult : mult;
        }
    }

    int tertiary_bdy_version = 0;
    if (other2_idx == 0 || other2_idx == (other2_nidx-1)) {
        tertiary_bdy_version = 2;
        if (other2_idx == (other2_nidx-1)) {
            tertiary_t = std::clamp(1.0 - tertiary_t, 0.0, 1.0);
            std::reverse(xshifts2.begin(), xshifts2.begin()+bs_p);
            std::reverse(yshifts2.begin(), yshifts2.begin()+bs_p);
            std::reverse(zshifts2.begin(), zshifts2.begin()+bs_p);
            mult = case_index == 2 ? -mult : mult;
        }
    } else if (bs_p != 2 && (other2_idx == 1 || other2_idx == (other2_nidx-2))) {
        tertiary_bdy_version = 1;
        if (other2_idx == (other2_nidx-2)) {
            tertiary_t = std::clamp(1.0 - tertiary_t, 0.0, 1.0);
            std::reverse(xshifts2.begin(), xshifts2.begin()+bs_p);
            std::reverse(yshifts2.begin(), yshifts2.begin()+bs_p);
            std::reverse(zshifts2.begin(), zshifts2.begin()+bs_p);
            mult = case_index == 2 ? -mult : mult;
        }
    }

    std::array<double, 4> basesp;
    std::array<double, 3> basess1;
    std::array<double, 3> basess2;
    if (bs_p == 2) {
        std::array<double, 4> bases2p;
        std::array<double, 3> bases1s1;
        std::array<double, 3> bases1s2;
        if (case_index == 0) {
            bases2p = {B20x(primary_t, primary_bdy_version), B21x(primary_t, primary_bdy_version), B22x(primary_t), 0.0};
            bases1s1 = {D10(secondary_t, secondary_bdy_version), D11(secondary_t), 0.0};
            bases1s2 = {D10(tertiary_t, tertiary_bdy_version), D11(tertiary_t), 0.0};
        } else if (case_index == 1) {
            bases2p = {B20(primary_t, primary_bdy_version), B21(primary_t, primary_bdy_version), B22(primary_t), 0.0};
            bases1s1 = {D10x(secondary_t, secondary_bdy_version), D11x(secondary_t), 0.0};
            bases1s2 = {D10(tertiary_t, tertiary_bdy_version), D11(tertiary_t), 0.0};
        } else {
            bases2p = {B20(primary_t, primary_bdy_version), B21(primary_t, primary_bdy_version), B22(primary_t), 0.0};
            bases1s1 = {D10(secondary_t, secondary_bdy_version), D11(secondary_t), 0.0};
            bases1s2 = {D10x(tertiary_t, tertiary_bdy_version), D11x(tertiary_t), 0.0};
        }
        basesp = bases2p;
        basess1 = bases1s1;
        basess2 = bases1s2;
    } else {
        std::array<double, 4> bases3p;
        std::array<double, 3> bases2s1;
        std::array<double, 3> bases2s2;
        if (case_index == 0) {
            bases3p = {B30x(primary_t, primary_bdy_version), B31x(primary_t, primary_bdy_version), B32x(primary_t, primary_bdy_version), B33x(primary_t)};
            bases2s1 = {D20(secondary_t, secondary_bdy_version), D21(secondary_t, secondary_bdy_version), D22(secondary_t)};
            bases2s2 = {D20(tertiary_t, tertiary_bdy_version), D21(tertiary_t, tertiary_bdy_version), D22(tertiary_t)};
        } else if (case_index == 1) {
            bases3p = {B30(primary_t, primary_bdy_version), B31(primary_t, primary_bdy_version), B32(primary_t, primary_bdy_version), B33(primary_t)};
            bases2s1 = {D20x(secondary_t, secondary_bdy_version), D21x(secondary_t, secondary_bdy_version), D22x(secondary_t)};
            bases2s2 = {D20(tertiary_t, tertiary_bdy_version), D21(tertiary_t, tertiary_bdy_version), D22(tertiary_t)};
        } else {
            bases3p = {B30(primary_t, primary_bdy_version), B31(primary_t, primary_bdy_version), B32(primary_t, primary_bdy_version), B33(primary_t)};
            bases2s1 = {D20(secondary_t, secondary_bdy_version), D21(secondary_t, secondary_bdy_version), D22(secondary_t)};
            bases2s2 = {D20x(tertiary_t, tertiary_bdy_version), D21x(tertiary_t, tertiary_bdy_version), D22x(tertiary_t)};
        }
        basesp = bases3p;
        basess1 = bases2s1;
        basess2 = bases2s2;
    }
    double result = 0.0;
    for (int l = 0; l < bs_p+1; ++l) {
        for (int m = 0; m < bs_p; ++m) {
            for (int q = 0; q < bs_p; ++q) {
                result += field(i + xoffsets[l] + xshifts1[m] + xshifts2[q], 
                                j + yoffsets[l] + yshifts1[m] + yshifts2[q], 
                                k + zoffsets[l] + zshifts1[m] + zshifts2[q]) * basesp[l] * basess1[m] * basess2[q];
            }
        }
    }
    return result * mult / (double)_h;
}

MyReal COFLIPSolver::computeFTLE(MyReal& det, const Eigen::Matrix3<MyReal>& pullback)
{
    det = pullback.determinant();
    Eigen::JacobiSVD<Eigen::Matrix3<MyReal>, Eigen::ComputeThinU | Eigen::ComputeThinV> svd(pullback);
    MyReal max_sv = svd.singularValues()[0];
    MyReal FTLE = log(max_sv);
    return FTLE;
}

inline Vec<3, MyReal> COFLIPSolver::traceRK4(MyReal dt, const Vec<3, MyReal> &pos, const Buffer3D<MyReal>& un, const Buffer3D<MyReal>& vn, const Buffer3D<MyReal>& wn)
{
    MyReal c1 = 1.0 / 6.0 * dt, c2 = 1.0 / 3.0 * dt, c3 = 1.0 / 3.0 * dt, c4 = 1.0 / 6.0 * dt;
    Vec<3, MyReal> input = pos;
    Vec<3, MyReal> velocity1 = getVelocityBSpline(input, un, vn, wn);
    Vec<3, MyReal> midp1 = input + (MyReal)(0.5 * dt) * velocity1;
    clampPos(midp1);
    Vec<3, MyReal> velocity2 = getVelocityBSpline(midp1, un, vn, wn);
    Vec<3, MyReal> midp2 = input + (MyReal)(0.5 * dt) * velocity2;
    clampPos(midp2);
    Vec<3, MyReal> velocity3 = getVelocityBSpline(midp2, un, vn, wn);
    Vec<3, MyReal> midp3 = input + dt * velocity3;
    clampPos(midp3);
    Vec<3, MyReal> velocity4 = getVelocityBSpline(midp3, un, vn, wn);
    Vec<3, MyReal> out_pos = input + c1 * velocity1 + c2 * velocity2 + c3 * velocity3 + c4 * velocity4;
    clampPos(out_pos);
    return out_pos;
}

inline Eigen::Matrix3<MyReal> COFLIPSolver::pullbackRK4(MyReal dt, Vec<3, MyReal> &inout_pos, const Eigen::Matrix3<MyReal>& input_pullback, const Buffer3D<MyReal>& un, const Buffer3D<MyReal>& vn, const Buffer3D<MyReal>& wn)
{
    MyReal c1 = 1.0 / 6.0 * dt, c2 = 1.0 / 3.0 * dt, c3 = 1.0 / 3.0 * dt, c4 = 1.0 / 6.0 * dt;
    Vec<3, MyReal> input = inout_pos;
    Vec<3, MyReal> velocity1 = getVelocityBSpline(input, un, vn, wn);
    Eigen::Matrix3<MyReal> jacobianVelocity1 = getJacobianVelocityBSpline(input, un, vn, wn);
    Eigen::Matrix3<MyReal> midpullbackdot1 = -jacobianVelocity1.transpose()*input_pullback;
    Vec<3, MyReal> midp1 = input + (MyReal)(0.5 * dt) * velocity1;
    clampPos(midp1);
    Eigen::Matrix3<MyReal> midpullback1 = input_pullback + (0.5 * dt) * midpullbackdot1;
    Vec<3, MyReal> velocity2 = getVelocityBSpline(midp1, un, vn, wn);
    Eigen::Matrix3<MyReal> jacobianVelocity2 = getJacobianVelocityBSpline(midp1, un, vn, wn);
    Eigen::Matrix3<MyReal> midpullbackdot2 = -jacobianVelocity2.transpose()*midpullback1;
    Vec<3, MyReal> midp2 = input + (MyReal)(0.5 * dt) * velocity2;
    clampPos(midp2);
    Eigen::Matrix3<MyReal> midpullback2 = input_pullback + (0.5 * dt) * midpullbackdot2;
    Vec<3, MyReal> velocity3 = getVelocityBSpline(midp2, un, vn, wn);
    Eigen::Matrix3<MyReal> jacobianVelocity3 = getJacobianVelocityBSpline(midp2, un, vn, wn);
    Eigen::Matrix3<MyReal> midpullbackdot3 = -jacobianVelocity3.transpose()*midpullback2;
    Vec<3, MyReal> midp3 = input + dt * velocity3;
    clampPos(midp3);
    Eigen::Matrix3<MyReal> midpullback3 = input_pullback + dt * midpullbackdot3;
    Vec<3, MyReal> velocity4 = getVelocityBSpline(midp3, un, vn, wn);
    Eigen::Matrix3<MyReal> jacobianVelocity4 = getJacobianVelocityBSpline(midp3, un, vn, wn);
    Eigen::Matrix3<MyReal> midpullbackdot4 = -jacobianVelocity4.transpose()*midpullback3;
    Eigen::Matrix3<MyReal> pullback = input_pullback + c1 * midpullbackdot1 + c2 * midpullbackdot2 + c3 * midpullbackdot3 + c4 * midpullbackdot4;
    clampPullback(pullback);
    inout_pos = input + c1 * velocity1 + c2 * velocity2 + c3 * velocity3 + c4 * velocity4;
    clampPos(inout_pos);
    return pullback;
}

void COFLIPSolver::seedParticles()
{
    std::cout << RED << "Seeding particles..." << RESET << std::endl;
    calculateCurl(true, true);
    double max_rho = 0;
    if (!do_uniform_particle_seeding) {
        int res = _h/(10./128./4.)+TOLERANCE;
        std::uniform_real_distribution<MyReal> unif(0, 1);
        max_vort = tbb::parallel_reduce( 
                    tbb::blocked_range<int>(0,_nzp*_nyp*_nxp),
                    TOLERANCE,
                    [&](tbb::blocked_range<int> range, MyReal running_max)
                    {
                        for (int tIdx=range.begin(); tIdx<range.end(); ++tIdx)
                        {
                            std::mt19937_64 rng;
                            rng.seed(tIdx);
                            int k = tIdx / (_nxp*_nyp);
                            int j = (tIdx % (_nxp*_nyp)) / _nxp;
                            int i = tIdx % _nxp;
                            MyReal x = (MyReal)i*_h;
                            MyReal y = (MyReal)j*_h;
                            MyReal z = (MyReal)k*_h;
                            for(int kk=0;kk<res;kk++)
                            {
                                for(int jj=0;jj<res;jj++)
                                {
                                    for(int ii=0;ii<res;ii++)
                                    {
                                        Vec<3, MyReal> spos(x+((MyReal)ii + unif(rng))/(MyReal)res*_h,
                                                            y+((MyReal)jj + unif(rng))/(MyReal)res*_h,
                                                            z+((MyReal)kk + unif(rng))/(MyReal)res*_h);
                                        Vec<3, MyReal> vort = getVorticityBSpline(spos, _vort_un, _vort_vn, _vort_wn);
                                        running_max = std::max(running_max, mag(vort));
                                    }
                                }
                            }
                        }

                        return running_max;
                    }, [](MyReal a, MyReal b) { return std::max(a,b); } );
        max_rho = tbb::parallel_reduce( 
                    tbb::blocked_range<int>(0,_nzp*_nyp*_nxp),
                    TOLERANCE,
                    [&](tbb::blocked_range<int> range, MyReal running_max)
                    {
                        for (int tIdx=range.begin(); tIdx<range.end(); ++tIdx)
                        {
                            std::mt19937_64 rng;
                            rng.seed(tIdx);
                            int k = tIdx / (_nxp*_nyp);
                            int j = (tIdx % (_nxp*_nyp)) / _nxp;
                            int i = tIdx % _nxp;
                            MyReal x = (MyReal)i*_h;
                            MyReal y = (MyReal)j*_h;
                            MyReal z = (MyReal)k*_h;
                            for(int kk=0;kk<res;kk++)
                            {
                                for(int jj=0;jj<res;jj++)
                                {
                                    for(int ii=0;ii<res;ii++)
                                    {
                                        Vec<3, MyReal> spos(x+((MyReal)ii + unif(rng))/(MyReal)res*_h,
                                                            y+((MyReal)jj + unif(rng))/(MyReal)res*_h,
                                                            z+((MyReal)kk + unif(rng))/(MyReal)res*_h);
                                        spos[0] = std::min(std::max(0.5*_amped_h+TOLERANCE, spos[0]), ((double)_amped_nx-0.5)*_amped_h-TOLERANCE);
                                        spos[1] = std::min(std::max(0.5*_amped_h+TOLERANCE, spos[1]), ((double)_amped_ny-0.5)*_amped_h-TOLERANCE);
                                        spos[2] = std::min(std::max(0.5*_amped_h+TOLERANCE, spos[2]), ((double)_amped_nz-0.5)*_amped_h-TOLERANCE);
                                        Vec<3, MyReal> zeroFormPos = spos - _amped_h*Vec<3, MyReal>(0.5f);
                                        double srho = sampleField(zeroFormPos, _rho, true, true);
                                        running_max = std::max(running_max, srho);
                                    }
                                }
                            }
                        }

                        return running_max;
                    }, [](MyReal a, MyReal b) { return std::max(a,b); } );
    }
    // initialize a uniform distribution between 0 and 1
    std::uniform_real_distribution<MyReal> unif(0, 1);
    lagrangian_particles.pos_current.resize(0);
    lagrangian_particles.pos_temp.resize(0);
    lagrangian_particles.volume.resize(0);
    for (int k = 0; k < _nzp; k++) for (int j = 0; j < _nyp; j++) for (int i = 0; i < _nxp; i++) 
    {
        int used_num_particles_per_cell = _N;
        if (!do_uniform_particle_seeding) {
            Vec<3, MyReal> pos = (Vec<3, MyReal>(i,j,k) + Vec<3, MyReal>(0.5)) * _h;
            MyReal alpha_curl = std::clamp(mag(getVorticityBSpline(pos, _vort_un, _vort_vn, _vort_wn)) / max_vort, (MyReal)0.f, (MyReal)1.f);
            MyReal alpha_rho = 0;
            if (!do_vel_advection_only) {
                pos[0] = std::min(std::max(0.5*_amped_h+TOLERANCE, pos[0]), ((double)_amped_nx-0.5)*_amped_h-TOLERANCE);
                pos[1] = std::min(std::max(0.5*_amped_h+TOLERANCE, pos[1]), ((double)_amped_ny-0.5)*_amped_h-TOLERANCE);
                pos[2] = std::min(std::max(0.5*_amped_h+TOLERANCE, pos[2]), ((double)_amped_nz-0.5)*_amped_h-TOLERANCE);
                Vec<3, MyReal> zeroFormPos = pos - _amped_h*Vec<3, MyReal>(0.5f);
                alpha_rho = std::clamp(sampleField(zeroFormPos, _rho, true, true) / max_rho, (MyReal)0.f, (MyReal)1.f);
            }
            MyReal total_rho = std::clamp(alpha_rho + alpha_curl, (MyReal)0.f, (MyReal)1.f);
            // +8 ensures at least a radius 2 times bigger than its standard deviation
            total_rho = std::clamp(std::exp(std::log(total_rho)+8.f), (MyReal)0.f, (MyReal)1.f);
            used_num_particles_per_cell = std::floor(lerp((_alpha != 0. || _beta != 0.) ? std::min(20,_N) : 8, _N, total_rho) + 0.5f);
        }
        int used_N = std::pow(used_num_particles_per_cell, 1./3.)+TOLERANCE;
        std::mt19937_64 rng;
        int tIdx = (k * _nyp + j) * _nxp + i;
        rng.seed(tIdx);
        MyReal x = (MyReal)i*_h;
        MyReal y = (MyReal)j*_h;
        MyReal z = (MyReal)k*_h;
        for(int kk=0;kk<used_N;kk++)
        {
            for(int jj=0;jj<used_N;jj++)
            {
                for(int ii=0;ii<used_N;ii++)
                {
                    Vec<3, MyReal> spos(x+((MyReal)ii + unif(rng))/(MyReal)used_N*_h,y+((MyReal)jj + unif(rng))/(MyReal)used_N*_h,z+((MyReal)kk + unif(rng))/(MyReal)used_N*_h);
                    lagrangian_particles.pos_current.push_back(spos);
                    lagrangian_particles.pos_temp.push_back(spos);
                    lagrangian_particles.volume.push_back(1./(double)used_num_particles_per_cell);
                }
            }
        }
        int remainder_particles_count = used_num_particles_per_cell - (used_N*used_N*used_N);
        for (int count = 0; count < remainder_particles_count; count++) {
            Vec<3, MyReal> spos(x+unif(rng)*_h,y+unif(rng)*_h,z+unif(rng)*_h);
            lagrangian_particles.pos_current.push_back(spos);
            lagrangian_particles.pos_temp.push_back(spos);
            lagrangian_particles.volume.push_back(1./(double)used_num_particles_per_cell);
        }
    }

    uint particle_count = lagrangian_particles.pos_current.size();
    lagrangian_particles.vel.resize(particle_count);
    lagrangian_particles.vort.resize(particle_count);
    lagrangian_particles.vel_floatingPoint_error.resize(particle_count);
    lagrangian_particles.vel_temp.resize(particle_count);
    lagrangian_particles.longterm_pullback.resize(particle_count);
    lagrangian_particles.shorterm_pullback.resize(particle_count);
    lagrangian_particles.delta_t.resize(particle_count);
    if (!do_vel_advection_only) {
        lagrangian_particles.rho.resize(particle_count);
        lagrangian_particles.T.resize(particle_count);
        lagrangian_particles.C_rho.resize(particle_count);
        lagrangian_particles.C_T.resize(particle_count);
    }
    if (sim_scheme == Scheme::CO_FLIP) {
    } else {
        lagrangian_particles.C_u.resize(particle_count);
        lagrangian_particles.C_v.resize(particle_count);
        lagrangian_particles.C_w.resize(particle_count);
    }
    std::cout << RED << "Seeding particles done. particle count: " << particle_count << RESET << std::endl;
}

void COFLIPSolver::sampleParticlesFromGrid()
{
    std::cout << RED << "Sampling particles from grid ..." << RESET << std::endl;
    tbb::parallel_for(tbb::blocked_range<int>(0,lagrangian_particles.pos_current.size(),TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
    {
        for (int i = r.begin(); i < r.end(); ++i) {
            Vec<3, MyReal> pos = lagrangian_particles.pos_current[i];
            lagrangian_particles.vel[i] = getVelocityBSpline(pos, _un, _vn, _wn);
            lagrangian_particles.vel_floatingPoint_error[i] = Vec<3, MyReal>(0);
            lagrangian_particles.vel_temp[i] = lagrangian_particles.vel[i];
            lagrangian_particles.longterm_pullback[i] = Eigen::Matrix3<MyReal>::Identity();
            lagrangian_particles.shorterm_pullback[i] = Eigen::Matrix3<MyReal>::Identity();
            lagrangian_particles.delta_t[i] = 0.0;
            // update Cp
            if (sim_scheme == Scheme::R_POLYFLIP || sim_scheme == Scheme::CF_POLYFLIP || sim_scheme == Scheme::POLYFLIP || sim_scheme == Scheme::POLYPIC) {
                Eigen::Matrix3<MyReal> dveldx = getJacobianVelocity(pos, _un, _vn, _wn);
                Vec<4, MyReal> dudxdy = sampleCrossHessianField(pos - Vec<3, MyReal>(0.0f, 0.5f*_h, 0.5f*_h), _un, true);
                Vec<4, MyReal> dvdxdy = sampleCrossHessianField(pos - Vec<3, MyReal>(0.5f*_h, 0.0f, 0.5f*_h), _vn, true);
                Vec<4, MyReal> dwdxdy = sampleCrossHessianField(pos - Vec<3, MyReal>(0.5f*_h, 0.5f*_h, 0.0f), _wn, true);
                lagrangian_particles.C_u[i] = {{lagrangian_particles.vel[i][0], 
                    dveldx(0,0), dveldx(0,1), dveldx(0,2),
                    dudxdy[0], dudxdy[1], dudxdy[2], dudxdy[3]}};
                lagrangian_particles.C_v[i] = {{lagrangian_particles.vel[i][1], 
                    dveldx(1,0), dveldx(1,1), dveldx(1,2),
                    dvdxdy[0], dvdxdy[1], dvdxdy[2], dvdxdy[3]}};
                lagrangian_particles.C_w[i] = {{lagrangian_particles.vel[i][2], 
                    dveldx(2,0), dveldx(2,1), dveldx(2,2),
                    dwdxdy[0], dwdxdy[1], dwdxdy[2], dwdxdy[3]}};
            }

            if (!do_vel_advection_only) {
                Vec<3, MyReal> zeroFormPos = pos - _amped_h*Vec<3, MyReal>(0.5f);
                lagrangian_particles.rho[i] = sampleField(zeroFormPos, _rho, true, true);
                lagrangian_particles.T[i] = sampleField(zeroFormPos, _T, true, true);
                {
                    Vec<3, MyReal> drhodx = sampleGradientField(zeroFormPos, _rho, true, true);
                    Vec<4, MyReal> drhodxdy = sampleCrossHessianField(zeroFormPos, _rho, true, true);
                    Vec<3, MyReal> dTdx = sampleGradientField(zeroFormPos, _T, true, true);
                    Vec<4, MyReal> dTdxdy = sampleCrossHessianField(zeroFormPos, _T, true, true);
                    lagrangian_particles.C_rho[i] = {{lagrangian_particles.rho[i], 
                        drhodx[0], drhodx[1], drhodx[2], 
                        drhodxdy[0], drhodxdy[1], drhodxdy[2], drhodxdy[3]}};
                    lagrangian_particles.C_T[i] = {{lagrangian_particles.T[i], 
                        dTdx[0], dTdx[1], dTdx[2], 
                        dTdxdy[0], dTdxdy[1], dTdxdy[2], dTdxdy[3]}};
                }
            }
        }
    });
    std::cout << RED << "Sampling particles from grid done." << RESET << std::endl;
}

void COFLIPSolver::solveInterpDagger(int stage_count, int framenum) {
    // splat particle properties to grid
    std::cout << BLUE << "Momentum map started..." << RESET << std::endl;

    int ni = _nx;
    int nj = _ny;
    int nk = _nz;
    int x_nF = (ni+1)*nj*nk;
    int y_nF = ni*(nj+1)*nk;
    uint numParticles = lagrangian_particles.pos_current.size();
    Eigen::VectorX<double> point_circ(3*numParticles);
    Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> fullInterpMatrix;
    bool use_interp_matrix = (is_matrix_small_enough && precond_reset_frequency != 1 && stage_count == 0 && (framenum == 1 || (framenum%precond_reset_frequency == 0 || framenum%delayed_reinit_num == 0))) || reset_precond;
    if (use_interp_matrix) {
        fullInterpMatrix.resize(_nF, _nF);
        fullInterpMatrix.reserve(Eigen::VectorXi::Constant(_nF,(2*bs_p+1)*(2*bs_p-1)*(2*bs_p-1)/2 + 1));
        tbb::parallel_for(tbb::blocked_range<int>(0,3), [&](const tbb::blocked_range<int> &r)
        {
            for (int work_idx = r.begin(); work_idx < r.end(); ++work_idx) {
                for (int i = 0; i < numParticles; ++i) {
                    Vec<3, MyReal> pos = lagrangian_particles.pos_temp[i];
                    int ii = std::floor(pos[0]/_h);
                    int jj = std::floor(pos[1]/_h);
                    int kk = std::floor(pos[2]/_h);
                    double alpha = (double)pos[0]/_h - (double)ii,
                        beta = (double)pos[1]/_h - (double)jj,
                        gamma = (double)pos[2]/_h - (double)kk;

                    if (work_idx == 0) {
                        for(int kkk=kk;kkk<=kk+bs_p-1;kkk++)for(int jjj=jj;jjj<=jj+bs_p-1;jjj++)for(int iii=ii;iii<=ii+bs_p;iii++)
                        {
                            double w = 0;
                            if (bs_p == 2) {
                                w = 
                                    lagrangian_particles.kernel2(ii == _nxp-1 ? 1.0 - alpha : alpha, 
                                                ii == _nxp-1 ? bs_p - (iii-ii) : iii-ii, 
                                                ii == 0 || ii == _nxp-1) *
                                    lagrangian_particles.kernel1prime(jj == _nyp-1 ? 1.0 - beta : beta, 
                                                jj == _nyp-1 ? bs_p-1 - (jjj-jj) : jjj-jj, 
                                                jj == 0 || jj == _nyp-1) *
                                    lagrangian_particles.kernel1prime(kk == _nzp-1 ? 1.0 - gamma : gamma, 
                                                kk == _nzp-1 ? bs_p-1 - (kkk-kk) : kkk-kk, 
                                                kk == 0 || kk == _nzp-1);
                            } else {
                                w =
                                    lagrangian_particles.kernel3((ii == _nxp-1 || ii == _nxp-2) ? 1.0 - alpha : alpha, 
                                                (ii == _nxp-1 || ii == _nxp-2) ? bs_p - (iii-ii) : iii-ii, 
                                                (ii == 0 || ii == _nxp-1) ? 2 : ((ii == 1 || ii == _nxp-2) ? 1 : 0)) *
                                    lagrangian_particles.kernel2prime((jj == _nyp-1 || jj == _nyp-2) ? 1.0 - beta : beta,
                                                (jj == _nyp-1 || jj == _nyp-2) ? bs_p-1 - (jjj-jj) : jjj-jj, 
                                                (jj == 0 || jj == _nyp-1) ? 2 : ((jj == 1 || jj == _nyp-2) ? 1 : 0)) *
                                    lagrangian_particles.kernel2prime((kk == _nzp-1 || kk == _nzp-2) ? 1.0 - gamma : gamma,
                                                (kk == _nzp-1 || kk == _nzp-2) ? bs_p-1 - (kkk-kk) : kkk-kk, 
                                                (kk == 0 || kk == _nzp-1) ? 2 : ((kk == 1 || kk == _nzp-2) ? 1 : 0));
                            }
                            int idx = kkk*nj*(ni+1) + jjj*(ni+1) + iii;

                            for(int kkk2=kk;kkk2<=kk+bs_p-1;kkk2++)for(int jjj2=jj;jjj2<=jj+bs_p-1;jjj2++)for(int iii2=ii;iii2<=ii+bs_p;iii2++)
                            {
                                int idx2 = kkk2*nj*(ni+1) + jjj2*(ni+1) + iii2;
                                if (idx2 <= idx) {
                                    double w2 = 0;
                                    if (bs_p == 2) {
                                        w2 = 
                                            lagrangian_particles.kernel2(ii == _nxp-1 ? 1.0 - alpha : alpha, 
                                                        ii == _nxp-1 ? bs_p - (iii2-ii) : iii2-ii, 
                                                        ii == 0 || ii == _nxp-1) *
                                            lagrangian_particles.kernel1prime(jj == _nyp-1 ? 1.0 - beta : beta, 
                                                        jj == _nyp-1 ? bs_p-1 - (jjj2-jj) : jjj2-jj, 
                                                        jj == 0 || jj == _nyp-1) *
                                            lagrangian_particles.kernel1prime(kk == _nzp-1 ? 1.0 - gamma : gamma, 
                                                        kk == _nzp-1 ? bs_p-1 - (kkk2-kk) : kkk2-kk, 
                                                        kk == 0 || kk == _nzp-1);
                                    } else {
                                        w2 =
                                            lagrangian_particles.kernel3((ii == _nxp-1 || ii == _nxp-2) ? 1.0 - alpha : alpha, 
                                                        (ii == _nxp-1 || ii == _nxp-2) ? bs_p - (iii2-ii) : iii2-ii, 
                                                        (ii == 0 || ii == _nxp-1) ? 2 : ((ii == 1 || ii == _nxp-2) ? 1 : 0)) *
                                            lagrangian_particles.kernel2prime((jj == _nyp-1 || jj == _nyp-2) ? 1.0 - beta : beta,
                                                        (jj == _nyp-1 || jj == _nyp-2) ? bs_p-1 - (jjj2-jj) : jjj2-jj, 
                                                        (jj == 0 || jj == _nyp-1) ? 2 : ((jj == 1 || jj == _nyp-2) ? 1 : 0)) *
                                            lagrangian_particles.kernel2prime((kk == _nzp-1 || kk == _nzp-2) ? 1.0 - gamma : gamma,
                                                        (kk == _nzp-1 || kk == _nzp-2) ? bs_p-1 - (kkk2-kk) : kkk2-kk, 
                                                        (kk == 0 || kk == _nzp-1) ? 2 : ((kk == 1 || kk == _nzp-2) ? 1 : 0));
                                    }
                                    fullInterpMatrix.coeffRef(idx,idx2) += w*w2*lagrangian_particles.volume[i];
                                }
                            }
                        }
                    }
                    if (work_idx == 1) {
                        for(int kkk=kk;kkk<=kk+bs_p-1;kkk++)for(int jjj=jj;jjj<=jj+bs_p;jjj++)for(int iii=ii;iii<=ii+bs_p-1;iii++)
                        {
                            double w = 0;
                            if (bs_p == 2) {
                                w = 
                                    lagrangian_particles.kernel1prime(ii == _nxp-1 ? 1.0 - alpha : alpha, 
                                                ii == _nxp-1 ? bs_p-1 - (iii-ii) : iii-ii,
                                                ii == 0 || ii == _nxp-1) *
                                    lagrangian_particles.kernel2(jj == _nyp-1 ? 1.0 - beta : beta,
                                                jj == _nyp-1 ? bs_p - (jjj-jj) : jjj-jj,
                                                jj == 0 || jj == _nyp-1) *
                                    lagrangian_particles.kernel1prime(kk == _nzp-1 ? 1.0 - gamma : gamma, 
                                                kk == _nzp-1 ? bs_p-1 - (kkk-kk) : kkk-kk,
                                                kk == 0 || kk == _nzp-1);
                            } else {
                                w =
                                    lagrangian_particles.kernel2prime((ii == _nxp-1 || ii == _nxp-2) ? 1.0 - alpha : alpha, 
                                                (ii == _nxp-1 || ii == _nxp-2) ? bs_p-1 - (iii-ii) : iii-ii, 
                                                (ii == 0 || ii == _nxp-1) ? 2 : ((ii == 1 || ii == _nxp-2) ? 1 : 0)) *
                                    lagrangian_particles.kernel3((jj == _nyp-1 || jj == _nyp-2) ? 1.0 - beta : beta,
                                                (jj == _nyp-1 || jj == _nyp-2) ? bs_p - (jjj-jj) : jjj-jj, 
                                                (jj == 0 || jj == _nyp-1) ? 2 : ((jj == 1 || jj == _nyp-2) ? 1 : 0)) *
                                    lagrangian_particles.kernel2prime((kk == _nzp-1 || kk == _nzp-2) ? 1.0 - gamma : gamma, 
                                                (kk == _nzp-1 || kk == _nzp-2) ? bs_p-1 - (kkk-kk) : kkk-kk, 
                                                (kk == 0 || kk == _nzp-1) ? 2 : ((kk == 1 || kk == _nzp-2) ? 1 : 0));
                            }
                            int idx = x_nF + kkk*(nj+1)*ni + jjj*ni + iii;

                            for(int kkk2=kk;kkk2<=kk+bs_p-1;kkk2++)for(int jjj2=jj;jjj2<=jj+bs_p;jjj2++)for(int iii2=ii;iii2<=ii+bs_p-1;iii2++)
                            {
                                int idx2 = x_nF + kkk2*(nj+1)*ni + jjj2*ni + iii2;
                                if (idx2 <= idx) {
                                    double w2 = 0;
                                    if (bs_p == 2) {
                                        w = 
                                            lagrangian_particles.kernel1prime(ii == _nxp-1 ? 1.0 - alpha : alpha, 
                                                        ii == _nxp-1 ? bs_p-1 - (iii2-ii) : iii2-ii,
                                                        ii == 0 || ii == _nxp-1) *
                                            lagrangian_particles.kernel2(jj == _nyp-1 ? 1.0 - beta : beta,
                                                        jj == _nyp-1 ? bs_p - (jjj2-jj) : jjj2-jj,
                                                        jj == 0 || jj == _nyp-1) *
                                            lagrangian_particles.kernel1prime(kk == _nzp-1 ? 1.0 - gamma : gamma, 
                                                        kk == _nzp-1 ? bs_p-1 - (kkk2-kk) : kkk2-kk,
                                                        kk == 0 || kk == _nzp-1);
                                    } else {
                                        w2 =
                                            lagrangian_particles.kernel2prime((ii == _nxp-1 || ii == _nxp-2) ? 1.0 - alpha : alpha, 
                                                        (ii == _nxp-1 || ii == _nxp-2) ? bs_p-1 - (iii2-ii) : iii2-ii, 
                                                        (ii == 0 || ii == _nxp-1) ? 2 : ((ii == 1 || ii == _nxp-2) ? 1 : 0)) *
                                            lagrangian_particles.kernel3((jj == _nyp-1 || jj == _nyp-2) ? 1.0 - beta : beta,
                                                        (jj == _nyp-1 || jj == _nyp-2) ? bs_p - (jjj2-jj) : jjj2-jj, 
                                                        (jj == 0 || jj == _nyp-1) ? 2 : ((jj == 1 || jj == _nyp-2) ? 1 : 0)) *
                                            lagrangian_particles.kernel2prime((kk == _nzp-1 || kk == _nzp-2) ? 1.0 - gamma : gamma, 
                                                        (kk == _nzp-1 || kk == _nzp-2) ? bs_p-1 - (kkk2-kk) : kkk2-kk, 
                                                        (kk == 0 || kk == _nzp-1) ? 2 : ((kk == 1 || kk == _nzp-2) ? 1 : 0));
                                    }
                                    fullInterpMatrix.coeffRef(idx,idx2) += w*w2*lagrangian_particles.volume[i];
                                }
                            }
                        }
                    }
                    if (work_idx == 2) {
                        for(int kkk=kk;kkk<=kk+bs_p;kkk++)for(int jjj=jj;jjj<=jj+bs_p-1;jjj++)for(int iii=ii;iii<=ii+bs_p-1;iii++)
                        {
                            double w = 0;
                            if (bs_p == 2) {
                                w = 
                                    lagrangian_particles.kernel1prime(ii == _nxp-1 ? 1.0 - alpha : alpha, 
                                                ii == _nxp-1 ? bs_p-1 - (iii-ii) : iii-ii,
                                                ii == 0 || ii == _nxp-1) *
                                    lagrangian_particles.kernel1prime(jj == _nyp-1 ? 1.0 - beta : beta,
                                                jj == _nyp-1 ? bs_p-1 - (jjj-jj) : jjj-jj,
                                                jj == 0 || jj == _nyp-1) *
                                    lagrangian_particles.kernel2(kk == _nzp-1 ? 1.0 - gamma : gamma, 
                                                kk == _nzp-1 ? bs_p - (kkk-kk) : kkk-kk,
                                                kk == 0 || kk == _nzp-1);
                            } else {
                                w =
                                    lagrangian_particles.kernel2prime((ii == _nxp-1 || ii == _nxp-2) ? 1.0 - alpha : alpha, 
                                                (ii == _nxp-1 || ii == _nxp-2) ? bs_p-1 - (iii-ii) : iii-ii, 
                                                (ii == 0 || ii == _nxp-1) ? 2 : ((ii == 1 || ii == _nxp-2) ? 1 : 0)) *
                                    lagrangian_particles.kernel2prime((jj == _nyp-1 || jj == _nyp-2) ? 1.0 - beta : beta,
                                                (jj == _nyp-1 || jj == _nyp-2) ? bs_p-1 - (jjj-jj) : jjj-jj, 
                                                (jj == 0 || jj == _nyp-1) ? 2 : ((jj == 1 || jj == _nyp-2) ? 1 : 0)) *
                                    lagrangian_particles.kernel3((kk == _nzp-1 || kk == _nzp-2) ? 1.0 - gamma : gamma, 
                                                (kk == _nzp-1 || kk == _nzp-2) ? bs_p - (kkk-kk) : kkk-kk, 
                                                (kk == 0 || kk == _nzp-1) ? 2 : ((kk == 1 || kk == _nzp-2) ? 1 : 0));
                            }
                            int idx = (x_nF + y_nF) + kkk*nj*ni + jjj*ni + iii;

                            for(int kkk2=kk;kkk2<=kk+bs_p;kkk2++)for(int jjj2=jj;jjj2<=jj+bs_p-1;jjj2++)for(int iii2=ii;iii2<=ii+bs_p-1;iii2++)
                            {
                                int idx2 = (x_nF + y_nF) + kkk2*nj*ni + jjj2*ni + iii2;
                                if (idx2 <= idx) {
                                    double w2 = 0;
                                    if (bs_p == 2) {
                                        w2 = 
                                            lagrangian_particles.kernel1prime(ii == _nxp-1 ? 1.0 - alpha : alpha, 
                                                        ii == _nxp-1 ? bs_p-1 - (iii2-ii) : iii2-ii,
                                                        ii == 0 || ii == _nxp-1) *
                                            lagrangian_particles.kernel1prime(jj == _nyp-1 ? 1.0 - beta : beta,
                                                        jj == _nyp-1 ? bs_p-1 - (jjj2-jj) : jjj2-jj,
                                                        jj == 0 || jj == _nyp-1) *
                                            lagrangian_particles.kernel2(kk == _nzp-1 ? 1.0 - gamma : gamma, 
                                                        kk == _nzp-1 ? bs_p - (kkk2-kk) : kkk2-kk,
                                                        kk == 0 || kk == _nzp-1);
                                    } else {
                                        w2 =
                                            lagrangian_particles.kernel2prime((ii == _nxp-1 || ii == _nxp-2) ? 1.0 - alpha : alpha, 
                                                        (ii == _nxp-1 || ii == _nxp-2) ? bs_p-1 - (iii2-ii) : iii2-ii, 
                                                        (ii == 0 || ii == _nxp-1) ? 2 : ((ii == 1 || ii == _nxp-2) ? 1 : 0)) *
                                            lagrangian_particles.kernel2prime((jj == _nyp-1 || jj == _nyp-2) ? 1.0 - beta : beta,
                                                        (jj == _nyp-1 || jj == _nyp-2) ? bs_p-1 - (jjj2-jj) : jjj2-jj, 
                                                        (jj == 0 || jj == _nyp-1) ? 2 : ((jj == 1 || jj == _nyp-2) ? 1 : 0)) *
                                            lagrangian_particles.kernel3((kk == _nzp-1 || kk == _nzp-2) ? 1.0 - gamma : gamma, 
                                                        (kk == _nzp-1 || kk == _nzp-2) ? bs_p - (kkk2-kk) : kkk2-kk, 
                                                        (kk == 0 || kk == _nzp-1) ? 2 : ((kk == 1 || kk == _nzp-2) ? 1 : 0));
                                    }
                                    fullInterpMatrix.coeffRef(idx,idx2) += w*w2*lagrangian_particles.volume[i];
                                }
                            }
                        }
                    }
                }
            }
        });
        fullInterpMatrix.makeCompressed();
    }
    tbb::parallel_for(tbb::blocked_range<int>(0,numParticles,TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
    {
        for (int i = r.begin(); i < r.end(); ++i) {
            point_circ(i) = lagrangian_particles.vel_temp[i][0] * std::sqrt(lagrangian_particles.volume[i]);
            point_circ(numParticles + i) = lagrangian_particles.vel_temp[i][1] * std::sqrt(lagrangian_particles.volume[i]);
            point_circ(2*numParticles + i) = lagrangian_particles.vel_temp[i][2] * std::sqrt(lagrangian_particles.volume[i]);
        }
    });
    std::cout << BLUE << "Momentum map processing done..." << RESET << std::endl;

    std::vector<atomwrapper<double> > atomic_flux_values;
    atomic_flux_values.resize(_nF);
    if (!do_delta_circulation)
    {
        std::cout << BLUE << "Momentum map least squares solve starting..." << RESET << std::endl;

        if (pre_solve_fluxes.norm() == 0) {
            pre_solve_fluxes = fluxes;
        }
        Eigen::VectorXd pre_solve_fluxes_double = pre_solve_fluxes.cast<double>();
        if (do_mass_lumping) {
            multiplyWithInterpTranspose(atomic_flux_values, point_circ);
            tbb::parallel_for(tbb::blocked_range<int>(0,_nF, TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
            {
                for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
                    circulations(tIdx) = atomic_flux_values[tIdx]._a.load();
                }
            });
            Eigen::VectorXd sum_of_columns(_nF);
            multiplyWithInterpTranspose(atomic_flux_values, Eigen::VectorXd::Constant(3*numParticles, 1.0));
            tbb::parallel_for(tbb::blocked_range<int>(0,_nF, TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
            {
                for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
                    sum_of_columns(tIdx) = atomic_flux_values[tIdx]._a.load();
                }
            });
            pre_solve_fluxes.array() = circulations.array() / sum_of_columns.cast<MyReal>().array();
        } else {
            if (false && bs_p == 2) {
                if (S_L_fullInterpMat.empty()) {
                    amg_levelGen_double.generateLevelsGalerkinCoarsening(A_L_fullInterpMat, R_L_fullInterpMat, P_L_fullInterpMat, wmax_fullInterpMat, S_L_fullInterpMat, total_level_fullInterpMat, fullInterpMatrix, ni, nj, nk, 1, GMG::FACES, 1.0f);
                } else {
                    bool update_wmax = stage_count == 0 && (framenum == 1 || framenum%precond_reset_frequency == 0);
                    if (update_wmax) {
                        amg_levelGen_double.generateLevelsGalerkinCoarseningRedoA_L(A_L_fullInterpMat, R_L_fullInterpMat, P_L_fullInterpMat, wmax_fullInterpMat, update_wmax, total_level_fullInterpMat, fullInterpMatrix, 1.0f);
                    }
                }
                double tolerance;
                int iterations;
                bool success;
                std::cout << "#iteration:      " << iterations << std::endl;
                std::cout << "estimated error: " << tolerance << std::endl;
                if (!success) {
                    printf("WARNING: Momentum map least squares solve failed!************************************************\n");
                }
            } else {
                if (use_interp_matrix) {
                    interp_dagger_precond.compute(fullInterpMatrix);
                    std::cout << "interp_dagger_precond.info(): " << interp_dagger_precond.info() << " 0: success, 1: numerical issue" << std::endl;
                    reset_precond = false;
                }
                double tolerance;
                int iterations;
                bool success;
                auto multiply_func = [&](Eigen::VectorXd& output, const Eigen::VectorXd& input) {
                    multiplyWithInterp(output, input);
                };
                auto multiply_transpose_func = [&](Eigen::VectorXd& output, const Eigen::VectorXd& input) {
                    multiplyWithInterpTranspose(atomic_flux_values, input);
                    tbb::parallel_for(tbb::blocked_range<int>(0,_nF, TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
                    {
                        for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
                            output(tIdx) = atomic_flux_values[tIdx]._a.load();
                        }
                    });
                };
                if (is_matrix_small_enough && precond_reset_frequency != 1 && interp_dagger_precond.info() == Eigen::Success) {
                    success = PLSCGSolve<double>(multiply_func, multiply_transpose_func, point_circ, pre_solve_fluxes_double, interp_dagger_precond, (double)TOLERANCE, MAX_ITERATIONS, tolerance, iterations);
                } else {
                    Eigen::VectorXd digonalOfFullMat(_nF);
                    multiplyWithInterpTranspose(atomic_flux_values, Eigen::VectorXd::Constant(3*numParticles, 1.0), true);
                    tbb::parallel_for(tbb::blocked_range<int>(0,_nF, TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
                    {
                        for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
                            digonalOfFullMat(tIdx) = atomic_flux_values[tIdx]._a.load();
                        }
                    });
                    Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> digonalMat(_nF,_nF);
                    digonalMat.reserve(Eigen::VectorXi::Constant(_nF, 1));
                    tbb::parallel_for(tbb::blocked_range<int>(0,_nF, TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
                    {
                        for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
                            digonalMat.insert(tIdx, tIdx) = digonalOfFullMat(tIdx);
                        }
                    });
                    digonalMat.makeCompressed();
                    Eigen::DiagonalPreconditioner<double> diagPrecond;
                    diagPrecond.compute(digonalMat);
                    std::cout << "diagPrecond.info(): " << diagPrecond.info() << std::endl;
                    success = PLSCGSolve<double>(multiply_func, multiply_transpose_func, point_circ, pre_solve_fluxes_double, diagPrecond, (double)TOLERANCE, MAX_ITERATIONS, tolerance, iterations, 0.2, 2);
                }
                if (is_matrix_small_enough && precond_reset_frequency != 1 && iterations > (MAX_ITERATIONS/5)) {
                    reset_precond = true;
                }
                std::cout << "#iteration:      " << iterations << std::endl;
                std::cout << "estimated error: " << tolerance << std::endl;
                if (!success) {
                    printf("WARNING: Momentum map least squares solve failed!************************************************\n");
                }
            }
        }
        pre_solve_fluxes = pre_solve_fluxes_double.cast<MyReal>();
        fluxes = pre_solve_fluxes;
        std::cout << GREEN << "fluxes= " << fluxes.norm() << RESET << std::endl;
        std::cout << BLUE << "Momentum map least squares solve done." << RESET << std::endl;
    } else {
        multiplyWithInterpTranspose(atomic_flux_values, point_circ);
        tbb::parallel_for(tbb::blocked_range<int>(0,_nF, TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
        {
            for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
                circulations(tIdx) = atomic_flux_values[tIdx]._a.load();
            }
        });

        tbb::parallel_for(tbb::blocked_range<int>(0,_nF, TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
        {
            for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
                bool is_x_dir = tIdx < x_nF;
                bool is_y_dir = !is_x_dir && tIdx < (x_nF + y_nF);
                int comp_slice = is_x_dir ? (ni+1)*nj : (is_y_dir ? ni*(nj+1) : ni*nj);
                int comp_n = is_x_dir ? (ni+1) : ni;
                int comp_tIdx = is_x_dir ? tIdx : (is_y_dir ? (tIdx - x_nF) : (tIdx - (x_nF + y_nF)));
                int k = comp_tIdx/comp_slice;
                int j = (comp_tIdx%comp_slice)/comp_n;
                int i = comp_tIdx%comp_n;
                if (is_x_dir) {
                    _un(i,j,k) = circulations[tIdx];
                } else if (is_y_dir) {
                    _vn(i,j,k) = circulations[tIdx];
                } else {
                    _wn(i,j,k) = circulations[tIdx];
                }
            }
        });

        takeDualwrtStar(_un, _vn, _wn, false, true);
    }
}

void COFLIPSolver::advectCOFLIPHelper(int stage_count, int framenum, MyReal dt, bool do_all)
{
    std::cout << RED << "Starting a cycle with dt= " << dt << RESET << std::endl;

    bool do_emit_force = false;
    for(auto &emitter : sim_emitter)
    {
        if(framenum < emitter.emitFrame)
        {
            if (emitter.do_set_velocities)
            {
                do_emit_force = true;
                break;
            }
        }
    }

    if (do_emit_force || !do_vel_advection_only) {
        tbb::parallel_for(tbb::blocked_range<int>(0,lagrangian_particles.pos_current.size(),TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
        {
            for (int p = r.begin(); p < r.end(); ++p) {
                Vec<3, MyReal> pos = lagrangian_particles.pos_temp[p];
                if (!do_vel_advection_only) {
                    Vec<3, MyReal> zeroFormPos = pos - _amped_h*Vec<3, MyReal>(0.5f);
                    MyReal density = sampleField(zeroFormPos, _rho, true, true);
                    MyReal temperature = sampleField(zeroFormPos, _T, true, true);
                    MyReal f = -dt*_alpha*density + dt*_beta*temperature;
                    lagrangian_particles.vel_temp[p] += Vec<3, MyReal>(f*cos(theta)*cos(phi), f*sin(theta), f*cos(theta)*sin(phi));
                }
                if (do_emit_force) {
                    for(auto &emitter : sim_emitter)
                    {
                        if(framenum < emitter.emitFrame)
                        {
                            if (emitter.do_set_velocities)
                            {
                                openvdb::tools::GridSampler<openvdb::Grid<openvdb::tree::Tree4<MyReal>::Type>, openvdb::tools::BoxSampler> box_sampler(*emitter.e_sdf);
                                Vec<3, MyReal> vdb_world_pos = pos - Vec<3, MyReal>(0.5*_h_uniform);
                                MyReal sdf_value = box_sampler.wsSample(openvdb::math::Vec3<MyReal>(vdb_world_pos[0], vdb_world_pos[1], vdb_world_pos[2]));
                                if (sdf_value <= 0)
                                {
                                    lagrangian_particles.vel_temp[p] = emitter.emit_velocity(pos);
                                }
                            }
                        }
                    }
                }
            }
        });
    }

    std::cout << BLUE << "Advection started..." << RESET << std::endl;
    tbb::parallel_for(tbb::blocked_range<int>(0,lagrangian_particles.pos_current.size(),TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
    {
        for (int p = r.begin(); p < r.end(); ++p) {
            Vec<3, MyReal> pos = lagrangian_particles.pos_temp[p];
            Eigen::Matrix3<MyReal> pullback = pullbackRK4(dt, pos, Eigen::Matrix3<MyReal>::Identity(), _un, _vn, _wn);
            Eigen::Vector3<MyReal> tempvel;
            tempvel << lagrangian_particles.vel_temp[p][0], lagrangian_particles.vel_temp[p][1], lagrangian_particles.vel_temp[p][2];
            Eigen::Vector3<MyReal> vel_temp_advected = pullback * tempvel;
            lagrangian_particles.vel_temp[p] = Vec<3, MyReal>(vel_temp_advected.data());
            lagrangian_particles.pos_temp[p] = pos;
            lagrangian_particles.shorterm_pullback[p] = pullback;

            if (do_all) {
                lagrangian_particles.pos_current[p] = lagrangian_particles.pos_temp[p];
                lagrangian_particles.vel[p] = lagrangian_particles.vel_temp[p];

                lagrangian_particles.longterm_pullback[p] = pullback*lagrangian_particles.longterm_pullback[p];
                lagrangian_particles.delta_t[p] += dt;
            }
        }
    });
    std::cout << BLUE << "Advection done." << RESET << std::endl;

    int ni = _nx;
    int nj = _ny;
    int nk = _nz;
    int x_nF = (ni+1)*nj*nk;
    int y_nF = ni*(nj+1)*nk;
    tbb::parallel_for(tbb::blocked_range<int>(0,_nF, TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
    {
        for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
            bool is_x_dir = tIdx < x_nF;
            bool is_y_dir = !is_x_dir && tIdx < (x_nF + y_nF);
            int comp_slice = is_x_dir ? (ni+1)*nj : (is_y_dir ? ni*(nj+1) : ni*nj);
            int comp_n = is_x_dir ? (ni+1) : ni;
            int comp_tIdx = is_x_dir ? tIdx : (is_y_dir ? (tIdx - x_nF) : (tIdx - (x_nF + y_nF)));
            int k = comp_tIdx/comp_slice;
            int j = (comp_tIdx%comp_slice)/comp_n;
            int i = comp_tIdx%comp_n;
            MyReal value = is_x_dir ? _un(i,j,k) : (is_y_dir ? _vn(i,j,k) : _wn(i,j,k));
            fluxes_midpoint[tIdx] = value;
        }
    });

    solveInterpDagger(stage_count, framenum);

    // project the advection term
    if (do_vel_advection_only && viscosity == 0. && !do_emit_force)
    {
        Eigen::VectorXd circulations_original(_nF);
        multiplyWithStarFlux(circulations_original, fluxes_midpoint);
        double original_energy_fluxes = fluxes_midpoint.transpose() * circulations_original;
        if (original_energy_fluxes != 0.) {
            Eigen::VectorXd fluxes_diff = fluxes - fluxes_original;
            std::cout << RED << "(projection on star flux) BEFORE: " << 2.*(fluxes_diff.transpose() * circulations_original)*std::pow(_h,3) << RESET << std::endl;
            double projected_fluxes_diff_to_fluxes = fluxes_diff.transpose() * circulations_original;
            fluxes_diff -= projected_fluxes_diff_to_fluxes * (fluxes_midpoint/original_energy_fluxes);
            std::cout << RED << "(projection on star flux) AFTER : " << 2.*(fluxes_diff.transpose() * circulations_original)*std::pow(_h,3) << RESET << std::endl;
            fluxes = fluxes_original + fluxes_diff;
        }
    }

    tbb::parallel_for(tbb::blocked_range<int>(0,_nF, TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
    {
        for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
            bool is_x_dir = tIdx < x_nF;
            bool is_y_dir = !is_x_dir && tIdx < (x_nF + y_nF);
            int comp_slice = is_x_dir ? (ni+1)*nj : (is_y_dir ? ni*(nj+1) : ni*nj);
            int comp_n = is_x_dir ? (ni+1) : ni;
            int comp_tIdx = is_x_dir ? tIdx : (is_y_dir ? (tIdx - x_nF) : (tIdx - (x_nF + y_nF)));
            int k = comp_tIdx/comp_slice;
            int j = (comp_tIdx%comp_slice)/comp_n;
            int i = comp_tIdx%comp_n;
            if (is_x_dir) {
                _u_save(i,j,k) = fluxes[tIdx];
                _un(i,j,k) = fluxes[tIdx];
            } else if (is_y_dir) {
                _v_save(i,j,k) = fluxes[tIdx];
                _vn(i,j,k) = fluxes[tIdx];
            } else {
                _w_save(i,j,k) = fluxes[tIdx];
                _wn(i,j,k) = fluxes[tIdx];
            }
        }
    });

    if (use_pressure_solver) {
        projection();
    } else {
        projectionWithVort();
    }
}

void COFLIPSolver::advanceCOFLIP(int framenum, MyReal dt)
{
    std::cout << BLUE << "bs_p= " << bs_p << RESET << std::endl;
    std::cout << BLUE <<  "CO-FLIP scheme frame " << framenum << " starts !" << RESET << std::endl;
    getCFL();
    std::cout << YELLOW << "[ CFL number is: " << max_v * dt / _h << " ] " << RESET << std::endl;

    int ni = _nx;
    int nj = _ny;
    int nk = _nz;
    int x_nF = (ni+1)*nj*nk;
    int y_nF = ni*(nj+1)*nk;
    tbb::parallel_for(tbb::blocked_range<int>(0,_nF, TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
    {
        for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
            bool is_x_dir = tIdx < x_nF;
            bool is_y_dir = !is_x_dir && tIdx < (x_nF + y_nF);
            int comp_slice = is_x_dir ? (ni+1)*nj : (is_y_dir ? ni*(nj+1) : ni*nj);
            int comp_n = is_x_dir ? (ni+1) : ni;
            int comp_tIdx = is_x_dir ? tIdx : (is_y_dir ? (tIdx - x_nF) : (tIdx - (x_nF + y_nF)));
            int k = comp_tIdx/comp_slice;
            int j = (comp_tIdx%comp_slice)/comp_n;
            int i = comp_tIdx%comp_n;
            MyReal value = is_x_dir ? _un(i,j,k) : (is_y_dir ? _vn(i,j,k) : _wn(i,j,k));
            fluxes_original[tIdx] = value;
        }
    });

    if (framenum == 1) {
        if (!do_vel_advection_only) {
            _rho_diff.init(_amped_nx, _amped_ny, _amped_nz, _amped_h, 0.f, 0.f, 0.f);
            _T_diff.init(_amped_nx, _amped_ny, _amped_nz, _amped_h, 0.f, 0.f, 0.f);
            _rho_weight.init(_amped_nx, _amped_ny, _amped_nz, _amped_h, 0.f, 0.f, 0.f);
            _T_weight.init(_amped_nx, _amped_ny, _amped_nz, _amped_h, 0.f, 0.f, 0.f);
            _T_save.init(_amped_nx, _amped_ny, _amped_nz, _amped_h, 0.f, 0.f, 0.f);
            _rho_save.init(_amped_nx, _amped_ny, _amped_nz, _amped_h, 0.f, 0.f, 0.f);
        }
        _u_diff.init(_nx+1, _ny, _nz, _h, 0.5f, 0.f, 0.f);
        _v_diff.init(_nx, _ny+1, _nz, _h, 0.f, 0.5f, 0.f);
        _w_diff.init(_nx, _ny, _nz+1, _h, 0.f, 0.f, 0.5f);
        if (timeIntOrder >= TimeIntegration::RK2) {
            _u0.init(_nx+1, _ny, _nz, _h, 0.5f, 0.f, 0.f);
            _v0.init(_nx, _ny+1, _nz, _h, 0.f, 0.5f, 0.f);
            _w0.init(_nx, _ny, _nz+1, _h, 0.f, 0.f, 0.5f);
            if (timeIntOrder == TimeIntegration::RK2) {
                _utemp.init(_nx+1, _ny, _nz, _h, 0.5f, 0.f, 0.f);
                _vtemp.init(_nx, _ny+1, _nz, _h, 0.f, 0.5f, 0.f);
                _wtemp.init(_nx, _ny, _nz+1, _h, 0.f, 0.f, 0.5f);
            }
            if (timeIntOrder >= TimeIntegration::RK3) {
                _u1.init(_nx+1, _ny, _nz, _h, 0.5f, 0.f, 0.f);
                _v1.init(_nx, _ny+1, _nz, _h, 0.f, 0.5f, 0.f);
                _w1.init(_nx, _ny, _nz+1, _h, 0.f, 0.f, 0.5f);
                if (timeIntOrder == TimeIntegration::RK4) {
                    _u2.init(_nx+1, _ny, _nz, _h, 0.5f, 0.f, 0.f);
                    _v2.init(_nx, _ny+1, _nz, _h, 0.f, 0.5f, 0.f);
                    _w2.init(_nx, _ny, _nz+1, _h, 0.f, 0.f, 0.5f);
                }
            }
        }
    }
    if (timeIntOrder == TimeIntegration::RK2) {
        _u0.copy(_un); _v0.copy(_vn); _w0.copy(_wn);
        tbb::parallel_for(tbb::blocked_range<int>(0,lagrangian_particles.pos_current.size(),TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
        {
            for (int i = r.begin(); i < r.end(); ++i) {
                lagrangian_particles.pos_temp[i] = lagrangian_particles.pos_current[i];
                lagrangian_particles.vel_temp[i] = lagrangian_particles.vel[i];
            }
        });
        advectCOFLIPHelper(0, framenum, dt);
        _utemp.copy(_un); _vtemp.copy(_vn); _wtemp.copy(_wn);
        std::cout << BLUE <<  "ODE stage 1 done..." << RESET << std::endl;
    } else if (timeIntOrder >= TimeIntegration::RK3) {
        _u0.copy(_un); _v0.copy(_vn); _w0.copy(_wn);
        tbb::parallel_for(tbb::blocked_range<int>(0,lagrangian_particles.pos_current.size(),TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
        {
            for (int i = r.begin(); i < r.end(); ++i) {
                lagrangian_particles.pos_temp[i] = lagrangian_particles.pos_current[i];
                lagrangian_particles.vel_temp[i] = lagrangian_particles.vel[i];
            }
        });
        advectCOFLIPHelper(0, framenum, 0.5*dt);
        std::cout << BLUE <<  "ODE stage 1 done..." << RESET << std::endl;

        if (timeIntOrder == TimeIntegration::RK3) {
            _u1.copy(_un); _v1.copy(_vn); _w1.copy(_wn);
            tbb::parallel_for(tbb::blocked_range<int>(0,lagrangian_particles.pos_current.size(),TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
            {
                for (int i = r.begin(); i < r.end(); ++i) {
                    lagrangian_particles.pos_temp[i] = lagrangian_particles.pos_current[i];
                    lagrangian_particles.vel_temp[i] = lagrangian_particles.vel[i];
                }
            });
            advectCOFLIPHelper(1, framenum, 0.75*dt);
            std::cout << BLUE <<  "ODE stage 2 done..." << RESET << std::endl;
        } else if (timeIntOrder == TimeIntegration::RK4) {
            _u1.copy(_un); _v1.copy(_vn); _w1.copy(_wn);
            tbb::parallel_for(tbb::blocked_range<int>(0,lagrangian_particles.pos_current.size(),TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
            {
                for (int i = r.begin(); i < r.end(); ++i) {
                    lagrangian_particles.pos_temp[i] = lagrangian_particles.pos_current[i];
                    lagrangian_particles.vel_temp[i] = lagrangian_particles.vel[i];
                }
            });
            advectCOFLIPHelper(1, framenum, 0.5*dt);
            std::cout << BLUE <<  "ODE stage 2 done..." << RESET << std::endl;
            _u2.copy(_un); _v2.copy(_vn); _w2.copy(_wn);
            tbb::parallel_for(tbb::blocked_range<int>(0,lagrangian_particles.pos_current.size(),TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
            {
                for (int i = r.begin(); i < r.end(); ++i) {
                    lagrangian_particles.pos_temp[i] = lagrangian_particles.pos_current[i];
                    lagrangian_particles.vel_temp[i] = lagrangian_particles.vel[i];
                }
            });
            advectCOFLIPHelper(2, framenum, dt);
            std::cout << BLUE <<  "ODE stage 3 done..." << RESET << std::endl;
        }
    }
    if (timeIntOrder == TimeIntegration::RK3) {
        tbb::parallel_for(tbb::blocked_range3d<int,int,int>{0,nk,8,0,ni+1,8,0,nj,8}, [&](const tbb::blocked_range3d<int,int,int> &r)
        {
            int ie = r.rows().end();
            int je = r.cols().end();
            int ke = r.pages().end();
            for (int k = r.pages().begin(); k < ke; ++k) {
                for (int j = r.cols().begin(); j < je; ++j) {
                    for (int i = r.rows().begin(); i < ie; ++i) {
                        _un(i,j,k) = 2./9.*_u0(i,j,k) + 3./9.*_u1(i,j,k) + 4./9.*_un(i,j,k);
                    }
                }
            }
        });
        tbb::parallel_for(tbb::blocked_range3d<int,int,int>{0,nk,8,0,ni,8,0,nj+1,8}, [&](const tbb::blocked_range3d<int,int,int> &r)
        {
            int ie = r.rows().end();
            int je = r.cols().end();
            int ke = r.pages().end();
            for (int k = r.pages().begin(); k < ke; ++k) {
                for (int j = r.cols().begin(); j < je; ++j) {
                    for (int i = r.rows().begin(); i < ie; ++i) {
                        _vn(i,j,k) = 2./9.*_v0(i,j,k) + 3./9.*_v1(i,j,k) + 4./9.*_vn(i,j,k);
                    }
                }
            }
        });
        tbb::parallel_for(tbb::blocked_range3d<int,int,int>{0,nk+1,8,0,ni,8,0,nj,8}, [&](const tbb::blocked_range3d<int,int,int> &r)
        {
            int ie = r.rows().end();
            int je = r.cols().end();
            int ke = r.pages().end();
            for (int k = r.pages().begin(); k < ke; ++k) {
                for (int j = r.cols().begin(); j < je; ++j) {
                    for (int i = r.rows().begin(); i < ie; ++i) {
                        _wn(i,j,k) = 2./9.*_w0(i,j,k) + 3./9.*_w1(i,j,k) + 4./9.*_wn(i,j,k);
                    }
                }
            }
        });
    } else if (timeIntOrder == TimeIntegration::RK4) {
        tbb::parallel_for(tbb::blocked_range3d<int,int,int>{0,nk,8,0,ni+1,8,0,nj,8}, [&](const tbb::blocked_range3d<int,int,int> &r)
        {
            int ie = r.rows().end();
            int je = r.cols().end();
            int ke = r.pages().end();
            for (int k = r.pages().begin(); k < ke; ++k) {
                for (int j = r.cols().begin(); j < je; ++j) {
                    for (int i = r.rows().begin(); i < ie; ++i) {
                        _un(i,j,k) = 1./6.*_u0(i,j,k) + 2./6.*_u1(i,j,k) + 2./6.*_u2(i,j,k) + 1./6.*_un(i,j,k);
                    }
                }
            }
        });
        tbb::parallel_for(tbb::blocked_range3d<int,int,int>{0,nk,8,0,ni,8,0,nj+1,8}, [&](const tbb::blocked_range3d<int,int,int> &r)
        {
            int ie = r.rows().end();
            int je = r.cols().end();
            int ke = r.pages().end();
            for (int k = r.pages().begin(); k < ke; ++k) {
                for (int j = r.cols().begin(); j < je; ++j) {
                    for (int i = r.rows().begin(); i < ie; ++i) {
                        _vn(i,j,k) = 1./6.*_v0(i,j,k) + 2./6.*_v1(i,j,k) + 2./6.*_v2(i,j,k) + 1./6.*_vn(i,j,k);
                    }
                }
            }
        });
        tbb::parallel_for(tbb::blocked_range3d<int,int,int>{0,nk+1,8,0,ni,8,0,nj,8}, [&](const tbb::blocked_range3d<int,int,int> &r)
        {
            int ie = r.rows().end();
            int je = r.cols().end();
            int ke = r.pages().end();
            for (int k = r.pages().begin(); k < ke; ++k) {
                for (int j = r.cols().begin(); j < je; ++j) {
                    for (int i = r.rows().begin(); i < ie; ++i) {
                        _wn(i,j,k) = 1./6.*_w0(i,j,k) + 2./6.*_w1(i,j,k) + 2./6.*_w2(i,j,k) + 1./6.*_wn(i,j,k);
                    }
                }
            }
        });
    }

    if (timeIntOrder == TimeIntegration::RK2) {
        MyReal error = 1.f;
        MyReal error_prev = 1.f;
        MyReal fluxes0_norm = 0.f;
        int x_nF = (ni+1)*nj*nk;
        int y_nF = ni*(nj+1)*nk;
        {
            Eigen::VectorX<MyReal> fluxes0(_nF);
            tbb::parallel_for(tbb::blocked_range<int>(0,_nF, TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
            {
                for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
                    bool is_x_dir = tIdx < x_nF;
                    bool is_y_dir = !is_x_dir && tIdx < (x_nF + y_nF);
                    int comp_slice = is_x_dir ? (ni+1)*nj : (is_y_dir ? ni*(nj+1) : ni*nj);
                    int comp_n = is_x_dir ? (ni+1) : ni;
                    int comp_tIdx = is_x_dir ? tIdx : (is_y_dir ? (tIdx - x_nF) : (tIdx - (x_nF + y_nF)));
                    int k = comp_tIdx/comp_slice;
                    int j = (comp_tIdx%comp_slice)/comp_n;
                    int i = comp_tIdx%comp_n;
                    MyReal value = is_x_dir ? _u0(i,j,k) : (is_y_dir ? _v0(i,j,k) : _w0(i,j,k));
                    fluxes0[tIdx] = value;
                }
            });
            fluxes0_norm = fluxes0.norm();
        }
        int iter = 0;
        do {
            tbb::parallel_for(tbb::blocked_range3d<int,int,int>{0,nk,8,0,ni+1,8,0,nj,8}, [&](const tbb::blocked_range3d<int,int,int> &r)
            {
                int ie = r.rows().end();
                int je = r.cols().end();
                int ke = r.pages().end();
                for (int k = r.pages().begin(); k < ke; ++k) {
                    for (int j = r.cols().begin(); j < je; ++j) {
                        for (int i = r.rows().begin(); i < ie; ++i) {
                            _un(i,j,k) = 0.5*_u0(i,j,k) + 0.5*_un(i,j,k);
                        }
                    }
                }
            });
            tbb::parallel_for(tbb::blocked_range3d<int,int,int>{0,nk,8,0,ni,8,0,nj+1,8}, [&](const tbb::blocked_range3d<int,int,int> &r)
            {
                int ie = r.rows().end();
                int je = r.cols().end();
                int ke = r.pages().end();
                for (int k = r.pages().begin(); k < ke; ++k) {
                    for (int j = r.cols().begin(); j < je; ++j) {
                        for (int i = r.rows().begin(); i < ie; ++i) {
                            _vn(i,j,k) = 0.5*_v0(i,j,k) + 0.5*_vn(i,j,k);
                        }
                    }
                }
            });
            tbb::parallel_for(tbb::blocked_range3d<int,int,int>{0,nk+1,8,0,ni,8,0,nj,8}, [&](const tbb::blocked_range3d<int,int,int> &r)
            {
                int ie = r.rows().end();
                int je = r.cols().end();
                int ke = r.pages().end();
                for (int k = r.pages().begin(); k < ke; ++k) {
                    for (int j = r.cols().begin(); j < je; ++j) {
                        for (int i = r.rows().begin(); i < ie; ++i) {
                            _wn(i,j,k) = 0.5*_w0(i,j,k) + 0.5*_wn(i,j,k);
                        }
                    }
                }
            });
            tbb::parallel_for(tbb::blocked_range<int>(0,lagrangian_particles.pos_current.size(),TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
            {
                for (int i = r.begin(); i < r.end(); ++i) {
                    lagrangian_particles.pos_temp[i] = lagrangian_particles.pos_current[i];
                    lagrangian_particles.vel_temp[i] = lagrangian_particles.vel[i];
                }
            });
            advectCOFLIPHelper(3, framenum, dt);
            if (do_mass_lumping) {
                break;
            }
            Eigen::VectorX<MyReal> fluxes_temp(_nF);
            tbb::parallel_for(tbb::blocked_range<int>(0,_nF, TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
            {
                for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
                    bool is_x_dir = tIdx < x_nF;
                    bool is_y_dir = !is_x_dir && tIdx < (x_nF + y_nF);
                    int comp_slice = is_x_dir ? (ni+1)*nj : (is_y_dir ? ni*(nj+1) : ni*nj);
                    int comp_n = is_x_dir ? (ni+1) : ni;
                    int comp_tIdx = is_x_dir ? tIdx : (is_y_dir ? (tIdx - x_nF) : (tIdx - (x_nF + y_nF)));
                    int k = comp_tIdx/comp_slice;
                    int j = (comp_tIdx%comp_slice)/comp_n;
                    int i = comp_tIdx%comp_n;
                    MyReal value = is_x_dir ? _utemp(i,j,k) : (is_y_dir ? _vtemp(i,j,k) : _wtemp(i,j,k));
                    fluxes_temp[tIdx] = value;
                }
            });
            _utemp.copy(_un); _vtemp.copy(_vn); _wtemp.copy(_wn);
            tbb::parallel_for(tbb::blocked_range<int>(0,_nF, TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
            {
                for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
                    bool is_x_dir = tIdx < x_nF;
                    bool is_y_dir = !is_x_dir && tIdx < (x_nF + y_nF);
                    int comp_slice = is_x_dir ? (ni+1)*nj : (is_y_dir ? ni*(nj+1) : ni*nj);
                    int comp_n = is_x_dir ? (ni+1) : ni;
                    int comp_tIdx = is_x_dir ? tIdx : (is_y_dir ? (tIdx - x_nF) : (tIdx - (x_nF + y_nF)));
                    int k = comp_tIdx/comp_slice;
                    int j = (comp_tIdx%comp_slice)/comp_n;
                    int i = comp_tIdx%comp_n;
                    MyReal value = is_x_dir ? _un(i,j,k) : (is_y_dir ? _vn(i,j,k) : _wn(i,j,k));
                    fluxes[tIdx] = value;
                }
            });
            error_prev = error;
            error = fluxes0_norm != 0. ? (fluxes-fluxes_temp).norm() / fluxes0_norm : (fluxes-fluxes_temp).norm();
            std::cout << YELLOW << "implicit fixed point error: " << error << RESET << std::endl;
            if (error < (5.*TOLERANCE)) {
                break;
            }
            iter++;
        } while (iter==1 || (iter < (MAX_ITERATIONS/20) && error_prev/error > 1. && !((error_prev/error) < 1.1f && error < (500.f*TOLERANCE))));

        tbb::parallel_for(tbb::blocked_range<int>(0,lagrangian_particles.pos_current.size(),TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
        {
            for (int p = r.begin(); p < r.end(); ++p) {
                lagrangian_particles.pos_current[p] = lagrangian_particles.pos_temp[p];
                lagrangian_particles.vel[p] = lagrangian_particles.vel_temp[p];
                lagrangian_particles.longterm_pullback[p] = lagrangian_particles.shorterm_pullback[p]*lagrangian_particles.longterm_pullback[p];
                lagrangian_particles.delta_t[p] += dt;
            }
        });
    } else {
        tbb::parallel_for(tbb::blocked_range<int>(0,lagrangian_particles.pos_current.size(),TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
        {
            for (int i = r.begin(); i < r.end(); ++i) {
                lagrangian_particles.pos_temp[i] = lagrangian_particles.pos_current[i];
                lagrangian_particles.vel_temp[i] = lagrangian_particles.vel[i];
            }
        });
        advectCOFLIPHelper(3, framenum, dt, true);
    }

    if (viscosity != 0.f) {
        std::cout << BLUE << "Viscosity started..." << RESET << std::endl;
        int iterations;
        MyReal tolerance;
        Eigen::VectorXd rhs = fluxes.cast<double>();
        Eigen::VectorXd fluxes_result = fluxes.cast<double>();
        bool success = AMGPCGSolve(A_L_viscosity_laplacian[0], rhs, fluxes_result,
            A_L_viscosity_laplacian, R_L_viscosity_laplacian, P_L_viscosity_laplacian, wmax_viscosity_laplacian, S_L_viscosity_laplacian, total_level_viscosity_laplacian, (MyReal)TOLERANCE, MAX_ITERATIONS, tolerance, iterations, _nx, _ny, _nz, GMG::FACES);
        std::cout << "fluxes ratio (after/before): " << fluxes_result.norm()/fluxes.norm() << std::endl;
        fluxes = fluxes_result.cast<MyReal>();
        std::cout << "#iteration:      " << iterations << std::endl;
        std::cout << "estimated error: " << tolerance << std::endl;
        if (!success) {
            printf("WARNING: Viscosity solve failed!************************************************\n");
        }
        tbb::parallel_for(tbb::blocked_range<int>(0,_nF, TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
        {
            for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
                bool is_x_dir = tIdx < x_nF;
                bool is_y_dir = !is_x_dir && tIdx < (x_nF + y_nF);
                int comp_slice = is_x_dir ? (ni+1)*nj : (is_y_dir ? ni*(nj+1) : ni*nj);
                int comp_n = is_x_dir ? (ni+1) : ni;
                int comp_tIdx = is_x_dir ? tIdx : (is_y_dir ? (tIdx - x_nF) : (tIdx - (x_nF + y_nF)));
                int k = comp_tIdx/comp_slice;
                int j = (comp_tIdx%comp_slice)/comp_n;
                int i = comp_tIdx%comp_n;
                if (is_x_dir) {
                    _un(i,j,k) = fluxes[tIdx];
                } else if (is_y_dir) {
                    _vn(i,j,k) = fluxes[tIdx];
                } else {
                    _wn(i,j,k) = fluxes[tIdx];
                }
            }
        });
        std::cout << BLUE << "Viscosity done." << RESET << std::endl;
    }

    std::cout << BLUE <<  "Final stage done..." << RESET << std::endl;
    std::cout << BLUE <<  "div-free flux data is on the grid faces." << RESET << std::endl;

    if (!do_vel_advection_only) {
        _rho.setZero();
        _rho_weight.setZero();
        _T.setZero();
        _T_weight.setZero();
        // splat particle properties to grid
        uint numParticles = lagrangian_particles.pos_current.size();
        for(int i=0;i<numParticles;i++)
        {
            Vec<3, MyReal> spos = lagrangian_particles.pos_current[i];
            spos[0] = std::min(std::max(0.5*_amped_h+TOLERANCE, spos[0]), ((double)_amped_nx-0.5)*_amped_h-TOLERANCE);
            spos[1] = std::min(std::max(0.5*_amped_h+TOLERANCE, spos[1]), ((double)_amped_ny-0.5)*_amped_h-TOLERANCE);
            spos[2] = std::min(std::max(0.5*_amped_h+TOLERANCE, spos[2]), ((double)_amped_nz-0.5)*_amped_h-TOLERANCE);
            const auto& sC_rho = lagrangian_particles.C_rho[i];
            const auto& sC_T = lagrangian_particles.C_T[i];
            int ii, jj, kk;
            ii = std::floor(spos[0]/_amped_h - 0.5);
            jj = std::floor(spos[1]/_amped_h - 0.5);
            kk = std::floor(spos[2]/_amped_h - 0.5);
            for(int kkk=kk;kkk<=kk+1;kkk++)for(int jjj=jj;jjj<=jj+1;jjj++)for(int iii=ii;iii<=ii+1;iii++)
            {
                Vec<3, MyReal> gpos = Vec<3, MyReal>(iii,jjj,kkk)*_amped_h + Vec<3, MyReal>(0.5)*_amped_h;
                MyReal w = lagrangian_particles.kernel((spos[0] - gpos[0])/_amped_h)*lagrangian_particles.kernel((spos[1] - gpos[1])/_amped_h)*lagrangian_particles.kernel((spos[2] - gpos[2])/_amped_h);

                MyReal c0 = sC_rho[0];
                MyReal c1 = sC_rho[1]*(gpos[0] - spos[0]);
                MyReal c2 = sC_rho[2]*(gpos[1] - spos[1]);
                MyReal c3 = sC_rho[3]*(gpos[2] - spos[2]);
                MyReal c4 = sC_rho[4]*(gpos[0] - spos[0])*(gpos[1] - spos[1]);
                MyReal c5 = sC_rho[5]*(gpos[1] - spos[1])*(gpos[2] - spos[2]);
                MyReal c6 = sC_rho[6]*(gpos[2] - spos[2])*(gpos[0] - spos[0]);
                MyReal c7 = sC_rho[7]*(gpos[0] - spos[0])*(gpos[1] - spos[1])*(gpos[2] - spos[2]);

                _rho(iii,jjj,kkk) += w*(c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7);
                _rho_weight(iii,jjj,kkk) += w;

                c0 = sC_T[0];
                c1 = sC_T[1]*(gpos[0] - spos[0]);
                c2 = sC_T[2]*(gpos[1] - spos[1]);
                c3 = sC_T[3]*(gpos[2] - spos[2]);
                c4 = sC_T[4]*(gpos[0] - spos[0])*(gpos[1] - spos[1]);
                c5 = sC_T[5]*(gpos[1] - spos[1])*(gpos[2] - spos[2]);
                c6 = sC_T[6]*(gpos[2] - spos[2])*(gpos[0] - spos[0]);
                c7 = sC_T[7]*(gpos[0] - spos[0])*(gpos[1] - spos[1])*(gpos[2] - spos[2]);

                _T(iii,jjj,kkk) += w*(c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7);
                _T_weight(iii,jjj,kkk) += w;
            }
        }
        _rho /= _rho_weight;
        _T /= _T_weight;

        _T_save.copy(_T);
        _rho_save.copy(_rho);
        // do something that changes _rho
        // do something that changes _T
        updateEmitters(framenum, dt);
        emitSmoke(framenum, dt);
        _rho_diff.copy(_rho);
        _rho_diff -= _rho_save;
        _T_diff.copy(_T);
        _T_diff -= _T_save;
    }

    _u_diff.copy(_un);
    _v_diff.copy(_vn);
    _w_diff.copy(_wn);

    _u_diff -= _u_save;
    _v_diff -= _v_save;
    _w_diff -= _w_save;

    MyReal flip = do_delta_circulation ? 1.0 : (do_particle_sample_after_first || framenum%delayed_reinit_num==0 ? 0.0 : 1.0);

    double max_FTLE = tbb::parallel_reduce( 
                tbb::blocked_range<int>(0,lagrangian_particles.pos_current.size()),
                TOLERANCE,
                [&](tbb::blocked_range<int> range, double running_max)
                {
                    for (int tIdx=range.begin(); tIdx<range.end(); ++tIdx)
                    {
                        MyReal det = 0.;
                        running_max = std::max(running_max, computeFTLE(det, lagrangian_particles.longterm_pullback[tIdx]));
                    }

                    return running_max;
                }, [](double a, double b) { return std::max(a,b); } );
    bool apply_cuttoff = do_delta_circulation ? false : (max_FTLE > adaptive_reset_cutoff ? true : false);

    if (flip == 0.0) {
        if (do_particle_sample_after_first) {
            do_particle_sample_after_first = false;
            do_uniform_particle_seeding = false;
            _N = 27;
        }
        seedParticles();
        apply_cuttoff = false;
    }

    std::atomic<int> adaptive_reset_counter = 0;
    tbb::parallel_for(tbb::blocked_range<int>(0,lagrangian_particles.pos_current.size(),TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
    {
        for (int i = r.begin(); i < r.end(); ++i) {
            Vec<3, MyReal> p_vel = lagrangian_particles.vel[i];
            Vec<3, MyReal> p_pos = lagrangian_particles.pos_current[i];
            Vec<3, MyReal> curr_dv = getVelocityBSpline(p_pos, _u_diff, _v_diff, _w_diff);
            if (flip == 0.0) {
                lagrangian_particles.vel_floatingPoint_error[i] = 0.;
            }
            Vec<3, MyReal> original = flip*(p_vel)
                              + (1-flip)*(getVelocityBSpline(p_pos, _u_save, _v_save, _w_save));
            Vec<3, MyReal> change = (do_delta_circulation || (flip != 0.0 && use_pressure_solver) ? Vec<3, MyReal>(0.0f) : curr_dv);
            lagrangian_particles.vel_floatingPoint_error[i] += change;
            p_vel = original + lagrangian_particles.vel_floatingPoint_error[i];
            lagrangian_particles.vel_floatingPoint_error[i] += original - p_vel;

            MyReal det = 0.;
            MyReal FTLE = computeFTLE(det, lagrangian_particles.longterm_pullback[i]);
            MyReal mag2_p_vel = mag2(p_vel);
            if ((std::isnan(mag2_p_vel) || std::isinf(mag2_p_vel)) || (apply_cuttoff && FTLE > 0.6*adaptive_reset_cutoff)) {
                p_vel = getVelocityBSpline(p_pos, _un, _vn, _wn);
                lagrangian_particles.vort[i] = getVorticityBSpline(p_pos, _vort_un, _vort_vn, _vort_wn);
                lagrangian_particles.longterm_pullback[i] = Eigen::Matrix3<MyReal>::Identity();
                lagrangian_particles.delta_t[i] = 0.0;
                adaptive_reset_counter++;
            }
            lagrangian_particles.vel[i] = p_vel;

            if (flip == 0.0) {
                lagrangian_particles.longterm_pullback[i] = Eigen::Matrix3<MyReal>::Identity();
                lagrangian_particles.delta_t[i] = 0.0;
            }

            if (!do_vel_advection_only)
            {
                MyReal p_rho = lagrangian_particles.rho[i];
                MyReal p_T = lagrangian_particles.T[i];
                Vec<3, MyReal> zeroFormPos = p_pos - _amped_h*Vec<3, MyReal>(0.5f);
                p_rho = flip*(p_rho + sampleField(zeroFormPos, _rho_diff, true, true))
                        + (1-flip)*sampleField(zeroFormPos, _rho, true, true);
                p_T = flip*(p_T + sampleField(zeroFormPos, _T_diff, true, true))
                        + (1-flip)*sampleField(zeroFormPos, _T, true, true);
                {
                    Vec<3, MyReal> drhodx = sampleGradientField(zeroFormPos, _rho, true, true);
                    Vec<4, MyReal> drhodxdy = sampleCrossHessianField(zeroFormPos, _rho, true, true);
                    Vec<3, MyReal> dTdx = sampleGradientField(zeroFormPos, _T, true, true);
                    Vec<4, MyReal> dTdxdy = sampleCrossHessianField(zeroFormPos, _T, true, true);
                    lagrangian_particles.C_rho[i] = {{p_rho, 
                        drhodx[0], drhodx[1], drhodx[2], 
                        drhodxdy[0], drhodxdy[1], drhodxdy[2], drhodxdy[3]}};
                    lagrangian_particles.C_T[i] = {{p_T, 
                        dTdx[0], dTdx[1], dTdx[2], 
                        dTdxdy[0], dTdxdy[1], dTdxdy[2], dTdxdy[3]}};
                }
                lagrangian_particles.rho[i] = p_rho;
                lagrangian_particles.T[i] = p_T;
            }
        }
    });
    std::cout << BLUE <<  "adaptive_reset_counter: " << adaptive_reset_counter << ", (%): " << ((MyReal)adaptive_reset_counter/(MyReal)lagrangian_particles.pos_current.size())*100. << RESET << std::endl;
    std::cout << BLUE <<  "Particle information sampled from the grid!" << RESET << std::endl;
    std::cout << BLUE <<  "end of frame!" << RESET << std::endl;
}

void COFLIPSolver::updateBackward(MyReal dt)
{
    int ni = _nx;
    int nj = _ny;
    int nk = _nz;
    // backward mapping
    double sub_dt = _h / max_v;
    double T = dt;
    double t = 0;
    while(t < T)
    {
        if (t + sub_dt > T) sub_dt = T - t;

        tbb::parallel_for(tbb::blocked_range3d<int,int,int>{0,nk,8,0,ni,8,0,nj,8}, [&](const tbb::blocked_range3d<int,int,int> &r)
        {
            int ie = r.rows().end();
            int je = r.cols().end();
            int ke = r.pages().end();
            for (int k = r.pages().begin(); k < ke; ++k) {
                for (int j = r.cols().begin(); j < je; ++j) {
                    for (int i = r.rows().begin(); i < ie; ++i) {
                        Vec3d pos = _h*Vec3d(i, j, k) + _h*Vec3d(0.5);
                        Vec3d back_pos = traceRK4(-dt, pos, _un, _vn, _wn);
                        _back_temp_x(i, j, k) = sampleField(back_pos - _h*Vec3d(0.5), _back_x);
                        _back_temp_y(i, j, k) = sampleField(back_pos - _h*Vec3d(0.5), _back_y);
                        _back_temp_z(i, j, k) = sampleField(back_pos - _h*Vec3d(0.5), _back_z);
                    }
                }
            }
        });
        _back_x.copy(_back_temp_x);
        _back_y.copy(_back_temp_y);
        _back_z.copy(_back_temp_z);

        t += sub_dt;
    }
}

void COFLIPSolver::advectCFFLIPHelper(int framenum, MyReal dt, bool do_all)
{
    bool do_emit_force = false;
    for(auto &emitter : sim_emitter)
    {
        if(framenum < emitter.emitFrame)
        {
            if (emitter.do_set_velocities)
            {
                do_emit_force = true;
                break;
            }
        }
    }

    if (do_emit_force || !do_vel_advection_only) {
        tbb::parallel_for(tbb::blocked_range<int>(0,lagrangian_particles.pos_current.size(),TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
        {
            for (int p = r.begin(); p < r.end(); ++p) {
                Vec<3, MyReal> pos = lagrangian_particles.pos_temp[p];
                if (!do_vel_advection_only) {
                    Vec<3, MyReal> zeroFormPos = pos - _amped_h*Vec<3, MyReal>(0.5f);
                    MyReal density = sampleField(zeroFormPos, _rho, true, true);
                    MyReal temperature = sampleField(zeroFormPos, _T, true, true);
                    MyReal f = -dt*_alpha*density + dt*_beta*temperature;
                    lagrangian_particles.vel_temp[p] += Vec<3, MyReal>(f*cos(theta)*cos(phi), f*sin(theta), f*cos(theta)*sin(phi));
                }
                if (do_emit_force) {
                    for(auto &emitter : sim_emitter)
                    {
                        if(framenum < emitter.emitFrame)
                        {
                            if (emitter.do_set_velocities)
                            {
                                openvdb::tools::GridSampler<openvdb::Grid<openvdb::tree::Tree4<MyReal>::Type>, openvdb::tools::BoxSampler> box_sampler(*emitter.e_sdf);
                                Vec<3, MyReal> vdb_world_pos = pos - Vec<3, MyReal>(0.5*_h_uniform);
                                MyReal sdf_value = box_sampler.wsSample(openvdb::math::Vec3<MyReal>(vdb_world_pos[0], vdb_world_pos[1], vdb_world_pos[2]));
                                if (sdf_value <= 0)
                                {
                                    lagrangian_particles.vel_temp[p] = emitter.emit_velocity(pos);
                                }
                            }
                        }
                    }
                }
            }
        });
    }

    tbb::parallel_for(tbb::blocked_range<int>(0,lagrangian_particles.pos_current.size(),TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
    {
        for (int p = r.begin(); p < r.end(); ++p) {
            Vec<3, MyReal> pos = lagrangian_particles.pos_temp[p];
            lagrangian_particles.pos_temp[p] = traceRK4(dt, pos, _un, _vn, _wn);

            if (do_all) {
                lagrangian_particles.pos_current[p] = lagrangian_particles.pos_temp[p];
                lagrangian_particles.vel[p] = lagrangian_particles.vel_temp[p];
            }
        }
    });

    int ni = _nx;
    int nj = _ny;
    int nk = _nz;
    if (sim_scheme != Scheme::POLYFLIP && sim_scheme != Scheme::POLYPIC) {
        tbb::parallel_for(tbb::blocked_range3d<int,int,int>{0,nk,8,0,ni,8,0,nj,8}, [&](const tbb::blocked_range3d<int,int,int> &r)
        {
            int ie = r.rows().end();
            int je = r.cols().end();
            int ke = r.pages().end();
            for (int k = r.pages().begin(); k < ke; ++k) {
                for (int j = r.cols().begin(); j < je; ++j) {
                    for (int i = r.rows().begin(); i < ie; ++i) {
                        _back_x(i, j, k) = _h*((MyReal)i + 0.5);
                        _back_y(i, j, k) = _h*((MyReal)j + 0.5);
                        _back_z(i, j, k) = _h*((MyReal)k + 0.5);
                    }
                }
            }
        });
        updateBackward(dt);
    }

    _u_weight.setZero();
    _v_weight.setZero();
    _w_weight.setZero();
    _un.setZero();
    _vn.setZero();
    _wn.setZero();
    // splat particle properties to grid
    uint numParticles = lagrangian_particles.pos_current.size();
    for(int i=0;i<numParticles;i++)
    {
        Vec<3, MyReal> spos = lagrangian_particles.pos_temp[i];
        spos[0] = std::min(std::max(0+TOLERANCE, spos[0]), ((double)ni)*_h-TOLERANCE);
        spos[1] = std::min(std::max(0.5*_h+TOLERANCE, spos[1]), ((double)nj-0.5)*_h-TOLERANCE);
        spos[2] = std::min(std::max(0.5*_h+TOLERANCE, spos[2]), ((double)nk-0.5)*_h-TOLERANCE);
        const auto& sC_u = lagrangian_particles.C_u[i];
        const auto& sC_v = lagrangian_particles.C_v[i];
        const auto& sC_w = lagrangian_particles.C_w[i];
        int ii, jj, kk;
        ii = std::floor(spos[0]/_h_uniform);
        jj = std::floor(spos[1]/_h_uniform - 0.5);
        kk = std::floor(spos[2]/_h_uniform - 0.5);
        for(int kkk=kk;kkk<=kk+1;kkk++)for(int jjj=jj;jjj<=jj+1;jjj++)for(int iii=ii;iii<=ii+1;iii++)
        {
            Vec<3, MyReal> gpos = Vec<3, MyReal>(iii,jjj,kkk)*_h_uniform + Vec<3, MyReal>(0.,0.5,0.5)*_h_uniform;
            MyReal w = lagrangian_particles.kernel((spos[0] - gpos[0])/_h_uniform)*lagrangian_particles.kernel((spos[1] - gpos[1])/_h_uniform)*lagrangian_particles.kernel((spos[2] - gpos[2])/_h_uniform);

            MyReal c0 = lagrangian_particles.vel_temp[i][0];
            MyReal c1 = sC_u[1]*(gpos[0] - spos[0]);
            MyReal c2 = sC_u[2]*(gpos[1] - spos[1]);
            MyReal c3 = sC_u[3]*(gpos[2] - spos[2]);
            MyReal c4 = sC_u[4]*(gpos[0] - spos[0])*(gpos[1] - spos[1]);
            MyReal c5 = sC_u[5]*(gpos[1] - spos[1])*(gpos[2] - spos[2]);
            MyReal c6 = sC_u[6]*(gpos[2] - spos[2])*(gpos[0] - spos[0]);
            MyReal c7 = sC_u[7]*(gpos[0] - spos[0])*(gpos[1] - spos[1])*(gpos[2] - spos[2]);

            _un(iii,jjj,kkk) += w*(c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7);
            _u_weight(iii,jjj,kkk) += w;
        }

        spos = lagrangian_particles.pos_temp[i];
        spos[0] = std::min(std::max(0.5*_h+TOLERANCE, spos[0]), ((double)ni-0.5)*_h-TOLERANCE);
        spos[1] = std::min(std::max(0+TOLERANCE, spos[1]), ((double)nj)*_h-TOLERANCE);
        spos[2] = std::min(std::max(0.5*_h+TOLERANCE, spos[2]), ((double)nk-0.5)*_h-TOLERANCE);
        ii = std::floor(spos[0]/_h_uniform - 0.5);
        jj = std::floor(spos[1]/_h_uniform);
        kk = std::floor(spos[2]/_h_uniform - 0.5);
        for(int kkk=kk;kkk<=kk+1;kkk++)for(int jjj=jj;jjj<=jj+1;jjj++)for(int iii=ii;iii<=ii+1;iii++)
        {
            Vec<3, MyReal> gpos = Vec<3, MyReal>(iii,jjj,kkk)*_h_uniform + Vec<3, MyReal>(0.5,0.,0.5)*_h_uniform;
            MyReal w = lagrangian_particles.kernel((spos[0] - gpos[0])/_h_uniform)*lagrangian_particles.kernel((spos[1] - gpos[1])/_h_uniform)*lagrangian_particles.kernel((spos[2] - gpos[2])/_h_uniform);

            MyReal c0 = lagrangian_particles.vel_temp[i][1];
            MyReal c1 = sC_v[1]*(gpos[0] - spos[0]);
            MyReal c2 = sC_v[2]*(gpos[1] - spos[1]);
            MyReal c3 = sC_v[3]*(gpos[2] - spos[2]);
            MyReal c4 = sC_v[4]*(gpos[0] - spos[0])*(gpos[1] - spos[1]);
            MyReal c5 = sC_v[5]*(gpos[1] - spos[1])*(gpos[2] - spos[2]);
            MyReal c6 = sC_v[6]*(gpos[2] - spos[2])*(gpos[0] - spos[0]);
            MyReal c7 = sC_v[7]*(gpos[0] - spos[0])*(gpos[1] - spos[1])*(gpos[2] - spos[2]);

            _vn(iii,jjj,kkk) += w*(c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7);
            _v_weight(iii,jjj,kkk) += w;
        }

        spos = lagrangian_particles.pos_temp[i];
        spos[0] = std::min(std::max(0.5*_h+TOLERANCE, spos[0]), ((double)ni-0.5)*_h-TOLERANCE);
        spos[1] = std::min(std::max(0.5*_h+TOLERANCE, spos[1]), ((double)nj-0.5)*_h-TOLERANCE);
        spos[2] = std::min(std::max(0+TOLERANCE, spos[2]), ((double)nk)*_h-TOLERANCE);
        ii = std::floor(spos[0]/_h_uniform - 0.5);
        jj = std::floor(spos[1]/_h_uniform - 0.5);
        kk = std::floor(spos[2]/_h_uniform);
        for(int kkk=kk;kkk<=kk+1;kkk++)for(int jjj=jj;jjj<=jj+1;jjj++)for(int iii=ii;iii<=ii+1;iii++)
        {
            Vec<3, MyReal> gpos = Vec<3, MyReal>(iii,jjj,kkk)*_h_uniform + Vec<3, MyReal>(0.5,0.5,0.)*_h_uniform;
            MyReal w = lagrangian_particles.kernel((spos[0] - gpos[0])/_h_uniform)*lagrangian_particles.kernel((spos[1] - gpos[1])/_h_uniform)*lagrangian_particles.kernel((spos[2] - gpos[2])/_h_uniform);

            MyReal c0 = lagrangian_particles.vel_temp[i][2];
            MyReal c1 = sC_w[1]*(gpos[0] - spos[0]);
            MyReal c2 = sC_w[2]*(gpos[1] - spos[1]);
            MyReal c3 = sC_w[3]*(gpos[2] - spos[2]);
            MyReal c4 = sC_w[4]*(gpos[0] - spos[0])*(gpos[1] - spos[1]);
            MyReal c5 = sC_w[5]*(gpos[1] - spos[1])*(gpos[2] - spos[2]);
            MyReal c6 = sC_w[6]*(gpos[2] - spos[2])*(gpos[0] - spos[0]);
            MyReal c7 = sC_w[7]*(gpos[0] - spos[0])*(gpos[1] - spos[1])*(gpos[2] - spos[2]);

            _wn(iii,jjj,kkk) += w*(c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7);
            _w_weight(iii,jjj,kkk) += w;
        }
    }
    _un /= _u_weight;
    _vn /= _v_weight;
    _wn /= _w_weight;

    _u_save.copy(_un);
    _v_save.copy(_vn);
    _w_save.copy(_wn);

    if (sim_scheme != Scheme::POLYFLIP && sim_scheme != Scheme::POLYPIC) {
        int x_nF = (ni+1)*nj*nk;
        int y_nF = ni*(nj+1)*nk;
        tbb::parallel_for(tbb::blocked_range<int>(0,_nF, TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
        {
            for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
                bool is_x_dir = tIdx < x_nF;
                bool is_y_dir = !is_x_dir && tIdx < (x_nF + y_nF);
                int comp_slice = is_x_dir ? (ni+1)*nj : (is_y_dir ? ni*(nj+1) : ni*nj);
                int comp_n = is_x_dir ? (ni+1) : ni;
                int comp_tIdx = is_x_dir ? tIdx : (is_y_dir ? (tIdx - x_nF) : (tIdx - (x_nF + y_nF)));
                int k = comp_tIdx/comp_slice;
                int j = (comp_tIdx%comp_slice)/comp_n;
                int i = comp_tIdx%comp_n;
                if (is_x_dir) {
                    if (i != 0 && i != ni)
                    {
                        Vec3d pos = _h * Vec3d(i, j, k) + _h * Vec3d(0.0, 0.5, 0.5);

                        Vec3d pos_front = pos + _h * Vec3d(0.5,0.0,0.0);
                        double x_front_init = sampleField(pos_front - _h * Vec3d(0.5), _back_x);
                        double y_front_init = sampleField(pos_front - _h * Vec3d(0.5), _back_y);
                        double z_front_init = sampleField(pos_front - _h * Vec3d(0.5), _back_z);
                        Vec3d pos1_front(x_front_init, y_front_init, z_front_init);
                        Vec3d pos_back  = pos - _h * Vec3d(0.5,0.0,0.0);
                        double x_back_init = sampleField(pos_back - _h * Vec3d(0.5), _back_x);
                        double y_back_init = sampleField(pos_back - _h * Vec3d(0.5), _back_y);
                        double z_back_init = sampleField(pos_back - _h * Vec3d(0.5), _back_z);
                        Vec3d pos1_back(x_back_init, y_back_init, z_back_init);
                        Vec3d diff = -pos1_back+pos1_front;
                        double distance = dist(pos_back,pos_front);
                        Vec3d vel_at_face = getVelocityBSpline(pos, _u_save, _v_save, _w_save);
                        _un(i,j,k) = dot(diff,vel_at_face) / distance;
                    }
                } else if (is_y_dir) {
                    if (j != 0 && j != nj)
                    {
                        Vec3d pos = _h * Vec3d(i, j, k) + _h * Vec3d(0.5, 0.0, 0.5);

                        Vec3d pos_front = pos + _h * Vec3d(0.0,0.5,0.0);
                        double x_front_init = sampleField(pos_front - _h * Vec3d(0.5), _back_x);
                        double y_front_init = sampleField(pos_front - _h * Vec3d(0.5), _back_y);
                        double z_front_init = sampleField(pos_front - _h * Vec3d(0.5), _back_z);
                        Vec3d pos1_front(x_front_init, y_front_init, z_front_init);
                        Vec3d pos_back  = pos - _h * Vec3d(0.0,0.5,0.0);
                        double x_back_init = sampleField(pos_back - _h * Vec3d(0.5), _back_x);
                        double y_back_init = sampleField(pos_back - _h * Vec3d(0.5), _back_y);
                        double z_back_init = sampleField(pos_back - _h * Vec3d(0.5), _back_z);
                        Vec3d pos1_back(x_back_init, y_back_init, z_back_init);
                        Vec3d diff = -pos1_back+pos1_front;
                        double distance = dist(pos_back,pos_front);
                        Vec3d vel_at_face = getVelocityBSpline(pos, _u_save, _v_save, _w_save);
                        _vn(i,j,k) = dot(diff,vel_at_face) / distance;
                    }
                } else {
                    if (k != 0 && k != nk)
                    {
                        Vec3d pos = _h * Vec3d(i, j, k) + _h * Vec3d(0.5, 0.5, 0.0);

                        Vec3d pos_front = pos + _h * Vec3d(0.0,0.0,0.5);
                        double x_front_init = sampleField(pos_front - _h * Vec3d(0.5), _back_x);
                        double y_front_init = sampleField(pos_front - _h * Vec3d(0.5), _back_y);
                        double z_front_init = sampleField(pos_front - _h * Vec3d(0.5), _back_z);
                        Vec3d pos1_front(x_front_init, y_front_init, z_front_init);
                        Vec3d pos_back  = pos - _h * Vec3d(0.0,0.0,0.5);
                        double x_back_init = sampleField(pos_back - _h * Vec3d(0.5), _back_x);
                        double y_back_init = sampleField(pos_back - _h * Vec3d(0.5), _back_y);
                        double z_back_init = sampleField(pos_back - _h * Vec3d(0.5), _back_z);
                        Vec3d pos1_back(x_back_init, y_back_init, z_back_init);
                        Vec3d diff = -pos1_back+pos1_front;
                        double distance = dist(pos_back,pos_front);
                        Vec3d vel_at_face = getVelocityBSpline(pos, _u_save, _v_save, _w_save);
                        _wn(i,j,k) = dot(diff,vel_at_face) / distance;
                    }
                }
            }
        });
    }

    projection();
}

void COFLIPSolver::advanceCovectorPOLYFLIP(int framenum, MyReal dt)
{
    std::cout << BLUE << "bs_p= " << bs_p << RESET << std::endl;
    std::cout << BLUE <<  "CO-FLIP scheme frame " << framenum << " starts !" << RESET << std::endl;
    getCFL();
    std::cout << YELLOW << "[ CFL number is: " << max_v * dt / _h << " ] " << RESET << std::endl;

    if (framenum == 1) {
        if (!do_vel_advection_only) {
            _rho_diff.init(_amped_nx, _amped_ny, _amped_nz, _amped_h, 0.f, 0.f, 0.f);
            _T_diff.init(_amped_nx, _amped_ny, _amped_nz, _amped_h, 0.f, 0.f, 0.f);
            _rho_weight.init(_amped_nx, _amped_ny, _amped_nz, _amped_h, 0.f, 0.f, 0.f);
            _T_weight.init(_amped_nx, _amped_ny, _amped_nz, _amped_h, 0.f, 0.f, 0.f);
            _T_save.init(_amped_nx, _amped_ny, _amped_nz, _amped_h, 0.f, 0.f, 0.f);
            _rho_save.init(_amped_nx, _amped_ny, _amped_nz, _amped_h, 0.f, 0.f, 0.f);
        }
        _u_diff.init(_nx+1, _ny, _nz, _h, 0.5f, 0.f, 0.f);
        _v_diff.init(_nx, _ny+1, _nz, _h, 0.f, 0.5f, 0.f);
        _w_diff.init(_nx, _ny, _nz+1, _h, 0.f, 0.f, 0.5f);
        _u_weight.init(_nx+1, _ny, _nz, _h, 0.5f, 0.f, 0.f);
        _v_weight.init(_nx, _ny+1, _nz, _h, 0.f, 0.5f, 0.f);
        _w_weight.init(_nx, _ny, _nz+1, _h, 0.f, 0.f, 0.5f);
        if (timeIntOrder >= TimeIntegration::RK2) {
            _u0.init(_nx+1, _ny, _nz, _h, 0.5f, 0.f, 0.f);
            _v0.init(_nx, _ny+1, _nz, _h, 0.f, 0.5f, 0.f);
            _w0.init(_nx, _ny, _nz+1, _h, 0.f, 0.f, 0.5f);
            if (timeIntOrder >= TimeIntegration::RK3) {
                _u1.init(_nx+1, _ny, _nz, _h, 0.5f, 0.f, 0.f);
                _v1.init(_nx, _ny+1, _nz, _h, 0.f, 0.5f, 0.f);
                _w1.init(_nx, _ny, _nz+1, _h, 0.f, 0.f, 0.5f);
                if (timeIntOrder == TimeIntegration::RK4) {
                    _u2.init(_nx+1, _ny, _nz, _h, 0.5f, 0.f, 0.f);
                    _v2.init(_nx, _ny+1, _nz, _h, 0.f, 0.5f, 0.f);
                    _w2.init(_nx, _ny, _nz+1, _h, 0.f, 0.f, 0.5f);
                }
            }
        }
    }

    int ni = _nx;
    int nj = _ny;
    int nk = _nz;
    if (timeIntOrder >= TimeIntegration::RK2) {
        _u0.copy(_un); _v0.copy(_vn); _w0.copy(_wn);
        tbb::parallel_for(tbb::blocked_range<int>(0,lagrangian_particles.pos_current.size(),TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
        {
            for (int i = r.begin(); i < r.end(); ++i) {
                lagrangian_particles.pos_temp[i] = lagrangian_particles.pos_current[i];
                lagrangian_particles.vel_temp[i] = lagrangian_particles.vel[i];
            }
        });
        advectCFFLIPHelper(framenum, 0.5*dt);
        std::cout << BLUE <<  "ODE stage 1 done..." << RESET << std::endl;

        if (timeIntOrder == TimeIntegration::RK3) {
            _u1.copy(_un); _v1.copy(_vn); _w1.copy(_wn);
            tbb::parallel_for(tbb::blocked_range<int>(0,lagrangian_particles.pos_current.size(),TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
            {
                for (int i = r.begin(); i < r.end(); ++i) {
                    lagrangian_particles.pos_temp[i] = lagrangian_particles.pos_current[i];
                    lagrangian_particles.vel_temp[i] = lagrangian_particles.vel[i];
                }
            });
            advectCFFLIPHelper(framenum, 0.75*dt);
            std::cout << BLUE <<  "ODE stage 2 done..." << RESET << std::endl;
        } else if (timeIntOrder == TimeIntegration::RK4) {
            _u1.copy(_un); _v1.copy(_vn); _w1.copy(_wn);
            tbb::parallel_for(tbb::blocked_range<int>(0,lagrangian_particles.pos_current.size(),TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
            {
                for (int i = r.begin(); i < r.end(); ++i) {
                    lagrangian_particles.pos_temp[i] = lagrangian_particles.pos_current[i];
                    lagrangian_particles.vel_temp[i] = lagrangian_particles.vel[i];
                }
            });
            advectCFFLIPHelper(framenum, 0.5*dt);
            std::cout << BLUE <<  "ODE stage 2 done..." << RESET << std::endl;
            _u2.copy(_un); _v2.copy(_vn); _w2.copy(_wn);
            tbb::parallel_for(tbb::blocked_range<int>(0,lagrangian_particles.pos_current.size(),TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
            {
                for (int i = r.begin(); i < r.end(); ++i) {
                    lagrangian_particles.pos_temp[i] = lagrangian_particles.pos_current[i];
                    lagrangian_particles.vel_temp[i] = lagrangian_particles.vel[i];
                }
            });
            advectCFFLIPHelper(framenum, dt);
            std::cout << BLUE <<  "ODE stage 3 done..." << RESET << std::endl;
        }
    }
    if (timeIntOrder == TimeIntegration::RK3) {
        tbb::parallel_for(tbb::blocked_range3d<int,int,int>{0,nk,8,0,ni+1,8,0,nj,8}, [&](const tbb::blocked_range3d<int,int,int> &r)
        {
            int ie = r.rows().end();
            int je = r.cols().end();
            int ke = r.pages().end();
            for (int k = r.pages().begin(); k < ke; ++k) {
                for (int j = r.cols().begin(); j < je; ++j) {
                    for (int i = r.rows().begin(); i < ie; ++i) {
                        _un(i,j,k) = 2./9.*_u0(i,j,k) + 3./9.*_u1(i,j,k) + 4./9.*_un(i,j,k);
                    }
                }
            }
        });
        tbb::parallel_for(tbb::blocked_range3d<int,int,int>{0,nk,8,0,ni,8,0,nj+1,8}, [&](const tbb::blocked_range3d<int,int,int> &r)
        {
            int ie = r.rows().end();
            int je = r.cols().end();
            int ke = r.pages().end();
            for (int k = r.pages().begin(); k < ke; ++k) {
                for (int j = r.cols().begin(); j < je; ++j) {
                    for (int i = r.rows().begin(); i < ie; ++i) {
                        _vn(i,j,k) = 2./9.*_v0(i,j,k) + 3./9.*_v1(i,j,k) + 4./9.*_vn(i,j,k);
                    }
                }
            }
        });
        tbb::parallel_for(tbb::blocked_range3d<int,int,int>{0,nk+1,8,0,ni,8,0,nj,8}, [&](const tbb::blocked_range3d<int,int,int> &r)
        {
            int ie = r.rows().end();
            int je = r.cols().end();
            int ke = r.pages().end();
            for (int k = r.pages().begin(); k < ke; ++k) {
                for (int j = r.cols().begin(); j < je; ++j) {
                    for (int i = r.rows().begin(); i < ie; ++i) {
                        _wn(i,j,k) = 2./9.*_w0(i,j,k) + 3./9.*_w1(i,j,k) + 4./9.*_wn(i,j,k);
                    }
                }
            }
        });
    } else if (timeIntOrder == TimeIntegration::RK4) {
        tbb::parallel_for(tbb::blocked_range3d<int,int,int>{0,nk,8,0,ni+1,8,0,nj,8}, [&](const tbb::blocked_range3d<int,int,int> &r)
        {
            int ie = r.rows().end();
            int je = r.cols().end();
            int ke = r.pages().end();
            for (int k = r.pages().begin(); k < ke; ++k) {
                for (int j = r.cols().begin(); j < je; ++j) {
                    for (int i = r.rows().begin(); i < ie; ++i) {
                        _un(i,j,k) = 1./6.*_u0(i,j,k) + 2./6.*_u1(i,j,k) + 2./6.*_u2(i,j,k) + 1./6.*_un(i,j,k);
                    }
                }
            }
        });
        tbb::parallel_for(tbb::blocked_range3d<int,int,int>{0,nk,8,0,ni,8,0,nj+1,8}, [&](const tbb::blocked_range3d<int,int,int> &r)
        {
            int ie = r.rows().end();
            int je = r.cols().end();
            int ke = r.pages().end();
            for (int k = r.pages().begin(); k < ke; ++k) {
                for (int j = r.cols().begin(); j < je; ++j) {
                    for (int i = r.rows().begin(); i < ie; ++i) {
                        _vn(i,j,k) = 1./6.*_v0(i,j,k) + 2./6.*_v1(i,j,k) + 2./6.*_v2(i,j,k) + 1./6.*_vn(i,j,k);
                    }
                }
            }
        });
        tbb::parallel_for(tbb::blocked_range3d<int,int,int>{0,nk+1,8,0,ni,8,0,nj,8}, [&](const tbb::blocked_range3d<int,int,int> &r)
        {
            int ie = r.rows().end();
            int je = r.cols().end();
            int ke = r.pages().end();
            for (int k = r.pages().begin(); k < ke; ++k) {
                for (int j = r.cols().begin(); j < je; ++j) {
                    for (int i = r.rows().begin(); i < ie; ++i) {
                        _wn(i,j,k) = 1./6.*_w0(i,j,k) + 2./6.*_w1(i,j,k) + 2./6.*_w2(i,j,k) + 1./6.*_wn(i,j,k);
                    }
                }
            }
        });
    }

    tbb::parallel_for(tbb::blocked_range<int>(0,lagrangian_particles.pos_current.size(),TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
    {
        for (int i = r.begin(); i < r.end(); ++i) {
            lagrangian_particles.pos_temp[i] = lagrangian_particles.pos_current[i];
            lagrangian_particles.vel_temp[i] = lagrangian_particles.vel[i];
        }
    });
    advectCFFLIPHelper(framenum, dt, true);

    if (viscosity != 0.f) {
        std::cout << BLUE << "Viscosity started..." << RESET << std::endl;
        int iterations;
        MyReal tolerance;
        Eigen::VectorXd rhs = fluxes.cast<double>();
        Eigen::VectorXd fluxes_result = fluxes.cast<double>();
        bool success = AMGPCGSolve(A_L_viscosity_laplacian[0], rhs, fluxes_result,
            A_L_viscosity_laplacian, R_L_viscosity_laplacian, P_L_viscosity_laplacian, wmax_viscosity_laplacian, S_L_viscosity_laplacian, total_level_viscosity_laplacian, (MyReal)TOLERANCE, MAX_ITERATIONS, tolerance, iterations, _nx, _ny, _nz, GMG::FACES);
        std::cout << "fluxes ratio (after/before): " << fluxes_result.norm()/fluxes.norm() << std::endl;
        fluxes = fluxes_result.cast<MyReal>();
        std::cout << "#iteration:      " << iterations << std::endl;
        std::cout << "estimated error: " << tolerance << std::endl;
        if (!success) {
            printf("WARNING: Viscosity solve failed!************************************************\n");
        }
        int x_nF = (ni+1)*nj*nk;
        int y_nF = ni*(nj+1)*nk;
        tbb::parallel_for(tbb::blocked_range<int>(0,_nF, TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
        {
            for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
                bool is_x_dir = tIdx < x_nF;
                bool is_y_dir = !is_x_dir && tIdx < (x_nF + y_nF);
                int comp_slice = is_x_dir ? (ni+1)*nj : (is_y_dir ? ni*(nj+1) : ni*nj);
                int comp_n = is_x_dir ? (ni+1) : ni;
                int comp_tIdx = is_x_dir ? tIdx : (is_y_dir ? (tIdx - x_nF) : (tIdx - (x_nF + y_nF)));
                int k = comp_tIdx/comp_slice;
                int j = (comp_tIdx%comp_slice)/comp_n;
                int i = comp_tIdx%comp_n;
                if (is_x_dir) {
                    _un(i,j,k) = fluxes[tIdx];
                } else if (is_y_dir) {
                    _vn(i,j,k) = fluxes[tIdx];
                } else {
                    _wn(i,j,k) = fluxes[tIdx];
                }
            }
        });
        std::cout << BLUE << "Viscosity done." << RESET << std::endl;
    }

    std::cout << BLUE <<  "Final stage done..." << RESET << std::endl;

    if (!do_vel_advection_only) {
        _rho.setZero();
        _rho_weight.setZero();
        _T.setZero();
        _T_weight.setZero();
        // splat particle properties to grid
        uint numParticles = lagrangian_particles.pos_current.size();
        for(int i=0;i<numParticles;i++)
        {
            Vec<3, MyReal> spos = lagrangian_particles.pos_current[i];
            spos[0] = std::min(std::max(0.5*_amped_h+TOLERANCE, spos[0]), ((double)_amped_nx-0.5)*_amped_h-TOLERANCE);
            spos[1] = std::min(std::max(0.5*_amped_h+TOLERANCE, spos[1]), ((double)_amped_ny-0.5)*_amped_h-TOLERANCE);
            spos[2] = std::min(std::max(0.5*_amped_h+TOLERANCE, spos[2]), ((double)_amped_nz-0.5)*_amped_h-TOLERANCE);
            const auto& sC_rho = lagrangian_particles.C_rho[i];
            const auto& sC_T = lagrangian_particles.C_T[i];
            int ii, jj, kk;
            ii = std::floor(spos[0]/_amped_h - 0.5);
            jj = std::floor(spos[1]/_amped_h - 0.5);
            kk = std::floor(spos[2]/_amped_h - 0.5);
            for(int kkk=kk;kkk<=kk+1;kkk++)for(int jjj=jj;jjj<=jj+1;jjj++)for(int iii=ii;iii<=ii+1;iii++)
            {
                Vec<3, MyReal> gpos = Vec<3, MyReal>(iii,jjj,kkk)*_amped_h + Vec<3, MyReal>(0.5)*_amped_h;
                MyReal w = lagrangian_particles.kernel((spos[0] - gpos[0])/_amped_h)*lagrangian_particles.kernel((spos[1] - gpos[1])/_amped_h)*lagrangian_particles.kernel((spos[2] - gpos[2])/_amped_h);

                MyReal c0 = sC_rho[0];
                MyReal c1 = sC_rho[1]*(gpos[0] - spos[0]);
                MyReal c2 = sC_rho[2]*(gpos[1] - spos[1]);
                MyReal c3 = sC_rho[3]*(gpos[2] - spos[2]);
                MyReal c4 = sC_rho[4]*(gpos[0] - spos[0])*(gpos[1] - spos[1]);
                MyReal c5 = sC_rho[5]*(gpos[1] - spos[1])*(gpos[2] - spos[2]);
                MyReal c6 = sC_rho[6]*(gpos[2] - spos[2])*(gpos[0] - spos[0]);
                MyReal c7 = sC_rho[7]*(gpos[0] - spos[0])*(gpos[1] - spos[1])*(gpos[2] - spos[2]);

                _rho(iii,jjj,kkk) += w*(c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7);
                _rho_weight(iii,jjj,kkk) += w;

                c0 = sC_T[0];
                c1 = sC_T[1]*(gpos[0] - spos[0]);
                c2 = sC_T[2]*(gpos[1] - spos[1]);
                c3 = sC_T[3]*(gpos[2] - spos[2]);
                c4 = sC_T[4]*(gpos[0] - spos[0])*(gpos[1] - spos[1]);
                c5 = sC_T[5]*(gpos[1] - spos[1])*(gpos[2] - spos[2]);
                c6 = sC_T[6]*(gpos[2] - spos[2])*(gpos[0] - spos[0]);
                c7 = sC_T[7]*(gpos[0] - spos[0])*(gpos[1] - spos[1])*(gpos[2] - spos[2]);

                _T(iii,jjj,kkk) += w*(c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7);
                _T_weight(iii,jjj,kkk) += w;
            }
        }
        _rho /= _rho_weight;
        _T /= _T_weight;

        _T_save.copy(_T);
        _rho_save.copy(_rho);
        // do something that changes _rho
        // do something that changes _T
        updateEmitters(framenum, dt);
        emitSmoke(framenum, dt);
        _rho_diff.copy(_rho);
        _rho_diff -= _rho_save;
        _T_diff.copy(_T);
        _T_diff -= _T_save;
    }

    int x_nF = (ni+1)*nj*nk;
    int y_nF = ni*(nj+1)*nk;
    tbb::parallel_for(tbb::blocked_range<int>(0,_nF, TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
    {
        for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
            bool is_x_dir = tIdx < x_nF;
            bool is_y_dir = !is_x_dir && tIdx < (x_nF + y_nF);
            int comp_slice = is_x_dir ? (ni+1)*nj : (is_y_dir ? ni*(nj+1) : ni*nj);
            int comp_n = is_x_dir ? (ni+1) : ni;
            int comp_tIdx = is_x_dir ? tIdx : (is_y_dir ? (tIdx - x_nF) : (tIdx - (x_nF + y_nF)));
            int k = comp_tIdx/comp_slice;
            int j = (comp_tIdx%comp_slice)/comp_n;
            int i = comp_tIdx%comp_n;
            if (is_x_dir) {
                _u_diff(i,j,k) = _un(i,j,k) - _u_save(i,j,k);
            } else if (is_y_dir) {
                _v_diff(i,j,k) = _vn(i,j,k) - _v_save(i,j,k);
            } else {
                _w_diff(i,j,k) = _wn(i,j,k) - _w_save(i,j,k);
            }
        }
    });

    MyReal flip = framenum%delayed_reinit_num==0 ? 0.0 : 1.0;

    if (flip == 0.0) {
        if (do_particle_sample_after_first) {
            do_particle_sample_after_first = false;
            do_uniform_particle_seeding = false;
            _N = 27;
        }
        seedParticles();
    }

    std::cout << "rho.norm: " << _rho.squared_norm() << std::endl;
    std::cout << "T.norm: " << _T.squared_norm() << std::endl;
    tbb::parallel_for(tbb::blocked_range<int>(0,lagrangian_particles.pos_current.size(),TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
    {
        for (int i = r.begin(); i < r.end(); ++i) {
            Vec<3, MyReal> p_vel = lagrangian_particles.vel[i];
            Vec<3, MyReal> p_pos = lagrangian_particles.pos_current[i];
            Vec<3, MyReal> curr_dv = getVelocityBSpline(p_pos, _u_diff, _v_diff, _w_diff);
            p_vel = (flip)*p_vel +
                    (1.-flip)*getVelocityBSpline(p_pos, _u_save, _v_save, _w_save) + 
                    curr_dv;

            lagrangian_particles.vel[i] = p_vel;
            lagrangian_particles.vel_temp[i] = p_vel;
            
            {
                Eigen::Matrix3<MyReal> dveldx = getJacobianVelocity(p_pos, _un, _vn, _wn);
                Vec<4, MyReal> dudxdy = sampleCrossHessianField(p_pos - Vec<3, MyReal>(0.0f, 0.5f*_h, 0.5f*_h), _un, true);
                Vec<4, MyReal> dvdxdy = sampleCrossHessianField(p_pos - Vec<3, MyReal>(0.5f*_h, 0.0f, 0.5f*_h), _vn, true);
                Vec<4, MyReal> dwdxdy = sampleCrossHessianField(p_pos - Vec<3, MyReal>(0.5f*_h, 0.5f*_h, 0.0f), _wn, true);
                lagrangian_particles.C_u[i] = {{lagrangian_particles.vel[i][0], 
                    dveldx(0,0), dveldx(0,1), dveldx(0,2),
                    dudxdy[0], dudxdy[1], dudxdy[2], dudxdy[3]}};
                lagrangian_particles.C_v[i] = {{lagrangian_particles.vel[i][1], 
                    dveldx(1,0), dveldx(1,1), dveldx(1,2),
                    dvdxdy[0], dvdxdy[1], dvdxdy[2], dvdxdy[3]}};
                lagrangian_particles.C_w[i] = {{lagrangian_particles.vel[i][2], 
                    dveldx(2,0), dveldx(2,1), dveldx(2,2),
                    dwdxdy[0], dwdxdy[1], dwdxdy[2], dwdxdy[3]}};
            }

            if (!do_vel_advection_only)
            {
                MyReal p_rho = lagrangian_particles.rho[i];
                MyReal p_T = lagrangian_particles.T[i];
                Vec<3, MyReal> zeroFormPos = p_pos - _amped_h*Vec<3, MyReal>(0.5f);
                p_rho = flip*(p_rho + sampleField(zeroFormPos, _rho_diff, true, true))
                        + (1-flip)*sampleField(zeroFormPos, _rho, true, true);
                p_T = flip*(p_T + sampleField(zeroFormPos, _T_diff, true, true))
                        + (1-flip)*sampleField(zeroFormPos, _T, true, true);
                {
                    Vec<3, MyReal> drhodx = sampleGradientField(zeroFormPos, _rho, true, true);
                    Vec<4, MyReal> drhodxdy = sampleCrossHessianField(zeroFormPos, _rho, true, true);
                    Vec<3, MyReal> dTdx = sampleGradientField(zeroFormPos, _T, true, true);
                    Vec<4, MyReal> dTdxdy = sampleCrossHessianField(zeroFormPos, _T, true, true);
                    lagrangian_particles.C_rho[i] = {{p_rho, 
                        drhodx[0], drhodx[1], drhodx[2], 
                        drhodxdy[0], drhodxdy[1], drhodxdy[2], drhodxdy[3]}};
                    lagrangian_particles.C_T[i] = {{p_T, 
                        dTdx[0], dTdx[1], dTdx[2], 
                        dTdxdy[0], dTdxdy[1], dTdxdy[2], dTdxdy[3]}};
                }
                lagrangian_particles.rho[i] = p_rho;
                lagrangian_particles.T[i] = p_T;
            }
        }
    });
    std::cout << BLUE <<  "Particle information sampled from the grid!" << RESET << std::endl;
    std::cout << BLUE <<  "end of frame!" << RESET << std::endl;
}

void COFLIPSolver::advanceReflectionPOLYFLIP(int framenum, MyReal dt)
{
    std::cout << BLUE << "bs_p= " << bs_p << RESET << std::endl;
    std::cout << BLUE <<  "CO-FLIP scheme frame " << framenum << " starts !" << RESET << std::endl;
    getCFL();
    std::cout << YELLOW << "[ CFL number is: " << max_v * dt / _h << " ] " << RESET << std::endl;

    if (framenum == 1) {
        if (!do_vel_advection_only) {
            _rho_diff.init(_amped_nx, _amped_ny, _amped_nz, _amped_h, 0.f, 0.f, 0.f);
            _T_diff.init(_amped_nx, _amped_ny, _amped_nz, _amped_h, 0.f, 0.f, 0.f);
            _rho_weight.init(_amped_nx, _amped_ny, _amped_nz, _amped_h, 0.f, 0.f, 0.f);
            _T_weight.init(_amped_nx, _amped_ny, _amped_nz, _amped_h, 0.f, 0.f, 0.f);
            _T_save.init(_amped_nx, _amped_ny, _amped_nz, _amped_h, 0.f, 0.f, 0.f);
            _rho_save.init(_amped_nx, _amped_ny, _amped_nz, _amped_h, 0.f, 0.f, 0.f);
        }
        _u_diff.init(_nx+1, _ny, _nz, _h, 0.5f, 0.f, 0.f);
        _v_diff.init(_nx, _ny+1, _nz, _h, 0.f, 0.5f, 0.f);
        _w_diff.init(_nx, _ny, _nz+1, _h, 0.f, 0.f, 0.5f);
        _u_weight.init(_nx+1, _ny, _nz, _h, 0.5f, 0.f, 0.f);
        _v_weight.init(_nx, _ny+1, _nz, _h, 0.f, 0.5f, 0.f);
        _w_weight.init(_nx, _ny, _nz+1, _h, 0.f, 0.f, 0.5f);
        _u0.init(_nx+1, _ny, _nz, _h, 0.5f, 0.f, 0.f);
        _v0.init(_nx, _ny+1, _nz, _h, 0.f, 0.5f, 0.f);
        _w0.init(_nx, _ny, _nz+1, _h, 0.f, 0.f, 0.5f);
    }

    _u0.copy(_un);
    _v0.copy(_vn);
    _w0.copy(_wn);

    tbb::parallel_for(tbb::blocked_range<int>(0,lagrangian_particles.pos_current.size(),TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
    {
        for (int p = r.begin(); p < r.end(); ++p) {
            Vec<3, MyReal> pos = lagrangian_particles.pos_current[p];
            lagrangian_particles.pos_current[p] = traceRK4(0.5*dt, pos, _un, _vn, _wn);
            lagrangian_particles.pos_temp[p] = lagrangian_particles.pos_current[p];
        }
    });

    std::cout << "advection is done." << std::endl;

    int ni = _nx;
    int nj = _ny;
    int nk = _nz;

    _u_weight.setZero();
    _v_weight.setZero();
    _w_weight.setZero();
    _un.setZero();
    _vn.setZero();
    _wn.setZero();
    // splat particle properties to grid
    uint numParticles = lagrangian_particles.pos_current.size();
    for(int i=0;i<numParticles;i++)
    {
        Vec<3, MyReal> spos = lagrangian_particles.pos_current[i];
        spos[0] = std::min(std::max(0+TOLERANCE, spos[0]), ((double)ni)*_h-TOLERANCE);
        spos[1] = std::min(std::max(0.5*_h+TOLERANCE, spos[1]), ((double)nj-0.5)*_h-TOLERANCE);
        spos[2] = std::min(std::max(0.5*_h+TOLERANCE, spos[2]), ((double)nk-0.5)*_h-TOLERANCE);
        const auto& sC_u = lagrangian_particles.C_u[i];
        const auto& sC_v = lagrangian_particles.C_v[i];
        const auto& sC_w = lagrangian_particles.C_w[i];
        int ii, jj, kk;
        ii = std::floor(spos[0]/_h_uniform);
        jj = std::floor(spos[1]/_h_uniform - 0.5);
        kk = std::floor(spos[2]/_h_uniform - 0.5);
        for(int kkk=kk;kkk<=kk+1;kkk++)for(int jjj=jj;jjj<=jj+1;jjj++)for(int iii=ii;iii<=ii+1;iii++)
        {
            Vec<3, MyReal> gpos = Vec<3, MyReal>(iii,jjj,kkk)*_h_uniform + Vec<3, MyReal>(0.,0.5,0.5)*_h_uniform;
            MyReal w = lagrangian_particles.kernel((spos[0] - gpos[0])/_h_uniform)*lagrangian_particles.kernel((spos[1] - gpos[1])/_h_uniform)*lagrangian_particles.kernel((spos[2] - gpos[2])/_h_uniform);

            MyReal c0 = sC_u[0];
            MyReal c1 = sC_u[1]*(gpos[0] - spos[0]);
            MyReal c2 = sC_u[2]*(gpos[1] - spos[1]);
            MyReal c3 = sC_u[3]*(gpos[2] - spos[2]);
            MyReal c4 = sC_u[4]*(gpos[0] - spos[0])*(gpos[1] - spos[1]);
            MyReal c5 = sC_u[5]*(gpos[1] - spos[1])*(gpos[2] - spos[2]);
            MyReal c6 = sC_u[6]*(gpos[2] - spos[2])*(gpos[0] - spos[0]);
            MyReal c7 = sC_u[7]*(gpos[0] - spos[0])*(gpos[1] - spos[1])*(gpos[2] - spos[2]);

            _un(iii,jjj,kkk) += w*(c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7);
            _u_weight(iii,jjj,kkk) += w;
        }

        spos = lagrangian_particles.pos_current[i];
        spos[0] = std::min(std::max(0.5*_h+TOLERANCE, spos[0]), ((double)ni-0.5)*_h-TOLERANCE);
        spos[1] = std::min(std::max(0+TOLERANCE, spos[1]), ((double)nj)*_h-TOLERANCE);
        spos[2] = std::min(std::max(0.5*_h+TOLERANCE, spos[2]), ((double)nk-0.5)*_h-TOLERANCE);
        ii = std::floor(spos[0]/_h_uniform - 0.5);
        jj = std::floor(spos[1]/_h_uniform);
        kk = std::floor(spos[2]/_h_uniform - 0.5);
        for(int kkk=kk;kkk<=kk+1;kkk++)for(int jjj=jj;jjj<=jj+1;jjj++)for(int iii=ii;iii<=ii+1;iii++)
        {
            Vec<3, MyReal> gpos = Vec<3, MyReal>(iii,jjj,kkk)*_h_uniform + Vec<3, MyReal>(0.5,0.,0.5)*_h_uniform;
            MyReal w = lagrangian_particles.kernel((spos[0] - gpos[0])/_h_uniform)*lagrangian_particles.kernel((spos[1] - gpos[1])/_h_uniform)*lagrangian_particles.kernel((spos[2] - gpos[2])/_h_uniform);

            MyReal c0 = sC_v[0];
            MyReal c1 = sC_v[1]*(gpos[0] - spos[0]);
            MyReal c2 = sC_v[2]*(gpos[1] - spos[1]);
            MyReal c3 = sC_v[3]*(gpos[2] - spos[2]);
            MyReal c4 = sC_v[4]*(gpos[0] - spos[0])*(gpos[1] - spos[1]);
            MyReal c5 = sC_v[5]*(gpos[1] - spos[1])*(gpos[2] - spos[2]);
            MyReal c6 = sC_v[6]*(gpos[2] - spos[2])*(gpos[0] - spos[0]);
            MyReal c7 = sC_v[7]*(gpos[0] - spos[0])*(gpos[1] - spos[1])*(gpos[2] - spos[2]);

            _vn(iii,jjj,kkk) += w*(c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7);
            _v_weight(iii,jjj,kkk) += w;
        }

        spos = lagrangian_particles.pos_current[i];
        spos[0] = std::min(std::max(0.5*_h+TOLERANCE, spos[0]), ((double)ni-0.5)*_h-TOLERANCE);
        spos[1] = std::min(std::max(0.5*_h+TOLERANCE, spos[1]), ((double)nj-0.5)*_h-TOLERANCE);
        spos[2] = std::min(std::max(0+TOLERANCE, spos[2]), ((double)nk)*_h-TOLERANCE);
        ii = std::floor(spos[0]/_h_uniform - 0.5);
        jj = std::floor(spos[1]/_h_uniform - 0.5);
        kk = std::floor(spos[2]/_h_uniform);
        for(int kkk=kk;kkk<=kk+1;kkk++)for(int jjj=jj;jjj<=jj+1;jjj++)for(int iii=ii;iii<=ii+1;iii++)
        {
            Vec<3, MyReal> gpos = Vec<3, MyReal>(iii,jjj,kkk)*_h_uniform + Vec<3, MyReal>(0.5,0.5,0.)*_h_uniform;
            MyReal w = lagrangian_particles.kernel((spos[0] - gpos[0])/_h_uniform)*lagrangian_particles.kernel((spos[1] - gpos[1])/_h_uniform)*lagrangian_particles.kernel((spos[2] - gpos[2])/_h_uniform);

            MyReal c0 = sC_w[0];
            MyReal c1 = sC_w[1]*(gpos[0] - spos[0]);
            MyReal c2 = sC_w[2]*(gpos[1] - spos[1]);
            MyReal c3 = sC_w[3]*(gpos[2] - spos[2]);
            MyReal c4 = sC_w[4]*(gpos[0] - spos[0])*(gpos[1] - spos[1]);
            MyReal c5 = sC_w[5]*(gpos[1] - spos[1])*(gpos[2] - spos[2]);
            MyReal c6 = sC_w[6]*(gpos[2] - spos[2])*(gpos[0] - spos[0]);
            MyReal c7 = sC_w[7]*(gpos[0] - spos[0])*(gpos[1] - spos[1])*(gpos[2] - spos[2]);

            _wn(iii,jjj,kkk) += w*(c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7);
            _w_weight(iii,jjj,kkk) += w;
        }
    }
    std::cout << "un.norm: " << _un.squared_norm() << " vn.norm: " << _vn.squared_norm() << " wn.norm: " << _wn.squared_norm() << std::endl;
    std::cout << "u_weight.norm: " << _u_weight.squared_norm() << " v_weight.norm: " << _v_weight.squared_norm() << " w_weight.norm: " << _w_weight.squared_norm() << std::endl;
    _un /= _u_weight;
    _vn /= _v_weight;
    _wn /= _w_weight;

    addEmitterForce(framenum, 0.5*dt);
    addBuoyancy(_un, _vn, _wn, 0.5*dt);
    _u_save.copy(_un);
    _v_save.copy(_vn);
    _w_save.copy(_wn);
    projection();

    int x_nF = (ni+1)*nj*nk;
    int y_nF = ni*(nj+1)*nk;
    if (viscosity != 0.f) {
        std::cout << BLUE << "Viscosity started..." << RESET << std::endl;
        int iterations;
        MyReal tolerance;
        Eigen::VectorXd rhs = fluxes.cast<double>();
        Eigen::VectorXd fluxes_result = fluxes.cast<double>();
        bool success = AMGPCGSolve(A_L_viscosity_laplacian[0], rhs, fluxes_result,
            A_L_viscosity_laplacian, R_L_viscosity_laplacian, P_L_viscosity_laplacian, wmax_viscosity_laplacian, S_L_viscosity_laplacian, total_level_viscosity_laplacian, (MyReal)TOLERANCE, MAX_ITERATIONS, tolerance, iterations, _nx, _ny, _nz, GMG::FACES);
        std::cout << "fluxes ratio (after/before): " << fluxes_result.norm()/fluxes.norm() << std::endl;
        fluxes = fluxes_result.cast<MyReal>();
        std::cout << "#iteration:      " << iterations << std::endl;
        std::cout << "estimated error: " << tolerance << std::endl;
        if (!success) {
            printf("WARNING: Viscosity solve failed!************************************************\n");
        }

        tbb::parallel_for(tbb::blocked_range<int>(0,_nF, TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
        {
            for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
                bool is_x_dir = tIdx < x_nF;
                bool is_y_dir = !is_x_dir && tIdx < (x_nF + y_nF);
                int comp_slice = is_x_dir ? (ni+1)*nj : (is_y_dir ? ni*(nj+1) : ni*nj);
                int comp_n = is_x_dir ? (ni+1) : ni;
                int comp_tIdx = is_x_dir ? tIdx : (is_y_dir ? (tIdx - x_nF) : (tIdx - (x_nF + y_nF)));
                int k = comp_tIdx/comp_slice;
                int j = (comp_tIdx%comp_slice)/comp_n;
                int i = comp_tIdx%comp_n;
                if (is_x_dir) {
                    _un(i,j,k) = fluxes[tIdx];
                } else if (is_y_dir) {
                    _vn(i,j,k) = fluxes[tIdx];
                } else {
                    _wn(i,j,k) = fluxes[tIdx];
                }
            }
        });
        std::cout << BLUE << "Viscosity done." << RESET << std::endl;
    }

    tbb::parallel_for(tbb::blocked_range<int>(0,_nF, TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
    {
        for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
            bool is_x_dir = tIdx < x_nF;
            bool is_y_dir = !is_x_dir && tIdx < (x_nF + y_nF);
            int comp_slice = is_x_dir ? (ni+1)*nj : (is_y_dir ? ni*(nj+1) : ni*nj);
            int comp_n = is_x_dir ? (ni+1) : ni;
            int comp_tIdx = is_x_dir ? tIdx : (is_y_dir ? (tIdx - x_nF) : (tIdx - (x_nF + y_nF)));
            int k = comp_tIdx/comp_slice;
            int j = (comp_tIdx%comp_slice)/comp_n;
            int i = comp_tIdx%comp_n;
            if (is_x_dir) {
                _u_diff(i,j,k) = _un(i,j,k) - _u_save(i,j,k);
            } else if (is_y_dir) {
                _v_diff(i,j,k) = _vn(i,j,k) - _v_save(i,j,k);
            } else {
                _w_diff(i,j,k) = _wn(i,j,k) - _w_save(i,j,k);
            }
        }
    });

    bool do_emit_force = false;
    for(auto &emitter : sim_emitter)
    {
        if(framenum < emitter.emitFrame)
        {
            if (emitter.do_set_velocities)
            {
                do_emit_force = true;
                break;
            }
        }
    }

    tbb::parallel_for(tbb::blocked_range<int>(0,lagrangian_particles.pos_current.size(),TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
    {
        for (int i = r.begin(); i < r.end(); ++i) {
            Vec<3, MyReal> p_vel = lagrangian_particles.vel[i];
            Vec<3, MyReal> p_pos = lagrangian_particles.pos_current[i];
            Vec<3, MyReal> curr_dv = getVelocityBSpline(p_pos, _u_diff, _v_diff, _w_diff);
            p_vel = p_vel + 2.*curr_dv;

            if (do_emit_force || !do_vel_advection_only) {
                if (!do_vel_advection_only) {
                    Vec<3, MyReal> zeroFormPos = p_pos - _amped_h*Vec<3, MyReal>(0.5f);
                    MyReal density = sampleField(zeroFormPos, _rho, true, true);
                    MyReal temperature = sampleField(zeroFormPos, _T, true, true);
                    MyReal f = -dt*_alpha*density + dt*_beta*temperature;
                    p_vel += Vec<3, MyReal>(f*cos(theta)*cos(phi), f*sin(theta), f*cos(theta)*sin(phi));
                }
                if (do_emit_force) {
                    for(auto &emitter : sim_emitter)
                    {
                        if(framenum < emitter.emitFrame)
                        {
                            if (emitter.do_set_velocities)
                            {
                                openvdb::tools::GridSampler<openvdb::Grid<openvdb::tree::Tree4<MyReal>::Type>, openvdb::tools::BoxSampler> box_sampler(*emitter.e_sdf);
                                Vec<3, MyReal> vdb_world_pos = p_pos - Vec<3, MyReal>(0.5*_h_uniform);
                                MyReal sdf_value = box_sampler.wsSample(openvdb::math::Vec3<MyReal>(vdb_world_pos[0], vdb_world_pos[1], vdb_world_pos[2]));
                                if (sdf_value <= 0)
                                {
                                    p_vel = emitter.emit_velocity(p_pos);
                                }
                            }
                        }
                    }
                }
            }

            lagrangian_particles.vel[i] = p_vel;
            lagrangian_particles.vel_temp[i] = p_vel;
            
            {
                Eigen::Matrix3<MyReal> dveldx = getJacobianVelocity(p_pos, _un, _vn, _wn);
                Vec<4, MyReal> dudxdy = sampleCrossHessianField(p_pos - Vec<3, MyReal>(0.0f, 0.5f*_h, 0.5f*_h), _un, true);
                Vec<4, MyReal> dvdxdy = sampleCrossHessianField(p_pos - Vec<3, MyReal>(0.5f*_h, 0.0f, 0.5f*_h), _vn, true);
                Vec<4, MyReal> dwdxdy = sampleCrossHessianField(p_pos - Vec<3, MyReal>(0.5f*_h, 0.5f*_h, 0.0f), _wn, true);
                lagrangian_particles.C_u[i] = {{lagrangian_particles.vel[i][0], 
                    dveldx(0,0), dveldx(0,1), dveldx(0,2),
                    dudxdy[0], dudxdy[1], dudxdy[2], dudxdy[3]}};
                lagrangian_particles.C_v[i] = {{lagrangian_particles.vel[i][1], 
                    dveldx(1,0), dveldx(1,1), dveldx(1,2),
                    dvdxdy[0], dvdxdy[1], dvdxdy[2], dvdxdy[3]}};
                lagrangian_particles.C_w[i] = {{lagrangian_particles.vel[i][2], 
                    dveldx(2,0), dveldx(2,1), dveldx(2,2),
                    dwdxdy[0], dwdxdy[1], dwdxdy[2], dwdxdy[3]}};
            }
        }
    });

    std::cout << BLUE <<  "First stage done..." << RESET << std::endl;

    _un *= 2;
    _un -= _u0;
    _vn *= 2;
    _vn -= _v0;
    _wn *= 2;
    _wn -= _w0;

    tbb::parallel_for(tbb::blocked_range<int>(0,lagrangian_particles.pos_current.size(),TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
    {
        for (int p = r.begin(); p < r.end(); ++p) {
            Vec<3, MyReal> pos = lagrangian_particles.pos_current[p];
            lagrangian_particles.pos_current[p] = traceRK4(0.5*dt, pos, _un, _vn, _wn);
            lagrangian_particles.pos_temp[p] = lagrangian_particles.pos_current[p];
        }
    });

    if (!do_vel_advection_only) {
        _rho.setZero();
        _rho_weight.setZero();
        _T.setZero();
        _T_weight.setZero();
        // splat particle properties to grid
        for(int i=0;i<numParticles;i++)
        {
            Vec<3, MyReal> spos = lagrangian_particles.pos_current[i];
            spos[0] = std::min(std::max(0.5*_amped_h+TOLERANCE, spos[0]), ((double)_amped_nx-0.5)*_amped_h-TOLERANCE);
            spos[1] = std::min(std::max(0.5*_amped_h+TOLERANCE, spos[1]), ((double)_amped_ny-0.5)*_amped_h-TOLERANCE);
            spos[2] = std::min(std::max(0.5*_amped_h+TOLERANCE, spos[2]), ((double)_amped_nz-0.5)*_amped_h-TOLERANCE);
            const auto& sC_rho = lagrangian_particles.C_rho[i];
            const auto& sC_T = lagrangian_particles.C_T[i];
            int ii, jj, kk;
            ii = std::floor(spos[0]/_amped_h - 0.5);
            jj = std::floor(spos[1]/_amped_h - 0.5);
            kk = std::floor(spos[2]/_amped_h - 0.5);
            for(int kkk=kk;kkk<=kk+1;kkk++)for(int jjj=jj;jjj<=jj+1;jjj++)for(int iii=ii;iii<=ii+1;iii++)
            {
                Vec<3, MyReal> gpos = Vec<3, MyReal>(iii,jjj,kkk)*_amped_h + Vec<3, MyReal>(0.5)*_amped_h;
                MyReal w = lagrangian_particles.kernel((spos[0] - gpos[0])/_amped_h)*lagrangian_particles.kernel((spos[1] - gpos[1])/_amped_h)*lagrangian_particles.kernel((spos[2] - gpos[2])/_amped_h);

                MyReal c0 = sC_rho[0];
                MyReal c1 = sC_rho[1]*(gpos[0] - spos[0]);
                MyReal c2 = sC_rho[2]*(gpos[1] - spos[1]);
                MyReal c3 = sC_rho[3]*(gpos[2] - spos[2]);
                MyReal c4 = sC_rho[4]*(gpos[0] - spos[0])*(gpos[1] - spos[1]);
                MyReal c5 = sC_rho[5]*(gpos[1] - spos[1])*(gpos[2] - spos[2]);
                MyReal c6 = sC_rho[6]*(gpos[2] - spos[2])*(gpos[0] - spos[0]);
                MyReal c7 = sC_rho[7]*(gpos[0] - spos[0])*(gpos[1] - spos[1])*(gpos[2] - spos[2]);

                _rho(iii,jjj,kkk) += w*(c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7);
                _rho_weight(iii,jjj,kkk) += w;

                c0 = sC_T[0];
                c1 = sC_T[1]*(gpos[0] - spos[0]);
                c2 = sC_T[2]*(gpos[1] - spos[1]);
                c3 = sC_T[3]*(gpos[2] - spos[2]);
                c4 = sC_T[4]*(gpos[0] - spos[0])*(gpos[1] - spos[1]);
                c5 = sC_T[5]*(gpos[1] - spos[1])*(gpos[2] - spos[2]);
                c6 = sC_T[6]*(gpos[2] - spos[2])*(gpos[0] - spos[0]);
                c7 = sC_T[7]*(gpos[0] - spos[0])*(gpos[1] - spos[1])*(gpos[2] - spos[2]);

                _T(iii,jjj,kkk) += w*(c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7);
                _T_weight(iii,jjj,kkk) += w;
            }
        }
        _rho /= _rho_weight;
        _T /= _T_weight;

        _T_save.copy(_T);
        _rho_save.copy(_rho);
        // do something that changes _rho
        // do something that changes _T
        updateEmitters(framenum, dt);
        emitSmoke(framenum, dt);
        _rho_diff.copy(_rho);
        _rho_diff -= _rho_save;
        _T_diff.copy(_T);
        _T_diff -= _T_save;
    }

    _u_weight.setZero();
    _v_weight.setZero();
    _w_weight.setZero();
    _un.setZero();
    _vn.setZero();
    _wn.setZero();
    // splat particle properties to grid
    for(int i=0;i<numParticles;i++)
    {
        Vec<3, MyReal> spos = lagrangian_particles.pos_current[i];
        spos[0] = std::min(std::max(0+TOLERANCE, spos[0]), ((double)ni)*_h-TOLERANCE);
        spos[1] = std::min(std::max(0.5*_h+TOLERANCE, spos[1]), ((double)nj-0.5)*_h-TOLERANCE);
        spos[2] = std::min(std::max(0.5*_h+TOLERANCE, spos[2]), ((double)nk-0.5)*_h-TOLERANCE);
        const auto& sC_u = lagrangian_particles.C_u[i];
        const auto& sC_v = lagrangian_particles.C_v[i];
        const auto& sC_w = lagrangian_particles.C_w[i];
        int ii, jj, kk;
        ii = std::floor(spos[0]/_h_uniform);
        jj = std::floor(spos[1]/_h_uniform - 0.5);
        kk = std::floor(spos[2]/_h_uniform - 0.5);
        for(int kkk=kk;kkk<=kk+1;kkk++)for(int jjj=jj;jjj<=jj+1;jjj++)for(int iii=ii;iii<=ii+1;iii++)
        {
            Vec<3, MyReal> gpos = Vec<3, MyReal>(iii,jjj,kkk)*_h_uniform + Vec<3, MyReal>(0.,0.5,0.5)*_h_uniform;
            MyReal w = lagrangian_particles.kernel((spos[0] - gpos[0])/_h_uniform)*lagrangian_particles.kernel((spos[1] - gpos[1])/_h_uniform)*lagrangian_particles.kernel((spos[2] - gpos[2])/_h_uniform);

            MyReal c0 = sC_u[0];
            MyReal c1 = sC_u[1]*(gpos[0] - spos[0]);
            MyReal c2 = sC_u[2]*(gpos[1] - spos[1]);
            MyReal c3 = sC_u[3]*(gpos[2] - spos[2]);
            MyReal c4 = sC_u[4]*(gpos[0] - spos[0])*(gpos[1] - spos[1]);
            MyReal c5 = sC_u[5]*(gpos[1] - spos[1])*(gpos[2] - spos[2]);
            MyReal c6 = sC_u[6]*(gpos[2] - spos[2])*(gpos[0] - spos[0]);
            MyReal c7 = sC_u[7]*(gpos[0] - spos[0])*(gpos[1] - spos[1])*(gpos[2] - spos[2]);

            _un(iii,jjj,kkk) += w*(c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7);
            _u_weight(iii,jjj,kkk) += w;
        }

        spos = lagrangian_particles.pos_current[i];
        spos[0] = std::min(std::max(0.5*_h+TOLERANCE, spos[0]), ((double)ni-0.5)*_h-TOLERANCE);
        spos[1] = std::min(std::max(0+TOLERANCE, spos[1]), ((double)nj)*_h-TOLERANCE);
        spos[2] = std::min(std::max(0.5*_h+TOLERANCE, spos[2]), ((double)nk-0.5)*_h-TOLERANCE);
        ii = std::floor(spos[0]/_h_uniform - 0.5);
        jj = std::floor(spos[1]/_h_uniform);
        kk = std::floor(spos[2]/_h_uniform - 0.5);
        for(int kkk=kk;kkk<=kk+1;kkk++)for(int jjj=jj;jjj<=jj+1;jjj++)for(int iii=ii;iii<=ii+1;iii++)
        {
            Vec<3, MyReal> gpos = Vec<3, MyReal>(iii,jjj,kkk)*_h_uniform + Vec<3, MyReal>(0.5,0.,0.5)*_h_uniform;
            MyReal w = lagrangian_particles.kernel((spos[0] - gpos[0])/_h_uniform)*lagrangian_particles.kernel((spos[1] - gpos[1])/_h_uniform)*lagrangian_particles.kernel((spos[2] - gpos[2])/_h_uniform);

            MyReal c0 = sC_v[0];
            MyReal c1 = sC_v[1]*(gpos[0] - spos[0]);
            MyReal c2 = sC_v[2]*(gpos[1] - spos[1]);
            MyReal c3 = sC_v[3]*(gpos[2] - spos[2]);
            MyReal c4 = sC_v[4]*(gpos[0] - spos[0])*(gpos[1] - spos[1]);
            MyReal c5 = sC_v[5]*(gpos[1] - spos[1])*(gpos[2] - spos[2]);
            MyReal c6 = sC_v[6]*(gpos[2] - spos[2])*(gpos[0] - spos[0]);
            MyReal c7 = sC_v[7]*(gpos[0] - spos[0])*(gpos[1] - spos[1])*(gpos[2] - spos[2]);

            _vn(iii,jjj,kkk) += w*(c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7);
            _v_weight(iii,jjj,kkk) += w;
        }

        spos = lagrangian_particles.pos_current[i];
        spos[0] = std::min(std::max(0.5*_h+TOLERANCE, spos[0]), ((double)ni-0.5)*_h-TOLERANCE);
        spos[1] = std::min(std::max(0.5*_h+TOLERANCE, spos[1]), ((double)nj-0.5)*_h-TOLERANCE);
        spos[2] = std::min(std::max(0+TOLERANCE, spos[2]), ((double)nk)*_h-TOLERANCE);
        ii = std::floor(spos[0]/_h_uniform - 0.5);
        jj = std::floor(spos[1]/_h_uniform - 0.5);
        kk = std::floor(spos[2]/_h_uniform);
        for(int kkk=kk;kkk<=kk+1;kkk++)for(int jjj=jj;jjj<=jj+1;jjj++)for(int iii=ii;iii<=ii+1;iii++)
        {
            Vec<3, MyReal> gpos = Vec<3, MyReal>(iii,jjj,kkk)*_h_uniform + Vec<3, MyReal>(0.5,0.5,0.)*_h_uniform;
            MyReal w = lagrangian_particles.kernel((spos[0] - gpos[0])/_h_uniform)*lagrangian_particles.kernel((spos[1] - gpos[1])/_h_uniform)*lagrangian_particles.kernel((spos[2] - gpos[2])/_h_uniform);

            MyReal c0 = sC_w[0];
            MyReal c1 = sC_w[1]*(gpos[0] - spos[0]);
            MyReal c2 = sC_w[2]*(gpos[1] - spos[1]);
            MyReal c3 = sC_w[3]*(gpos[2] - spos[2]);
            MyReal c4 = sC_w[4]*(gpos[0] - spos[0])*(gpos[1] - spos[1]);
            MyReal c5 = sC_w[5]*(gpos[1] - spos[1])*(gpos[2] - spos[2]);
            MyReal c6 = sC_w[6]*(gpos[2] - spos[2])*(gpos[0] - spos[0]);
            MyReal c7 = sC_w[7]*(gpos[0] - spos[0])*(gpos[1] - spos[1])*(gpos[2] - spos[2]);

            _wn(iii,jjj,kkk) += w*(c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7);
            _w_weight(iii,jjj,kkk) += w;
        }
    }
    _un /= _u_weight;
    _vn /= _v_weight;
    _wn /= _w_weight;

    _u_save.copy(_un);
    _v_save.copy(_vn);
    _w_save.copy(_wn);
    projection();
    tbb::parallel_for(tbb::blocked_range<int>(0,_nF, TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
    {
        for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
            bool is_x_dir = tIdx < x_nF;
            bool is_y_dir = !is_x_dir && tIdx < (x_nF + y_nF);
            int comp_slice = is_x_dir ? (ni+1)*nj : (is_y_dir ? ni*(nj+1) : ni*nj);
            int comp_n = is_x_dir ? (ni+1) : ni;
            int comp_tIdx = is_x_dir ? tIdx : (is_y_dir ? (tIdx - x_nF) : (tIdx - (x_nF + y_nF)));
            int k = comp_tIdx/comp_slice;
            int j = (comp_tIdx%comp_slice)/comp_n;
            int i = comp_tIdx%comp_n;
            if (is_x_dir) {
                _u_diff(i,j,k) = _un(i,j,k) - _u_save(i,j,k);
            } else if (is_y_dir) {
                _v_diff(i,j,k) = _vn(i,j,k) - _v_save(i,j,k);
            } else {
                _w_diff(i,j,k) = _wn(i,j,k) - _w_save(i,j,k);
            }
        }
    });

    MyReal flip = framenum%delayed_reinit_num==0 ? 0.0 : 1.0;

    if (flip == 0.0) {
        if (do_particle_sample_after_first) {
            do_particle_sample_after_first = false;
            do_uniform_particle_seeding = false;
            _N = 27;
        }
        seedParticles();
    }

    std::cout << "rho.norm: " << _rho.squared_norm() << std::endl;
    std::cout << "T.norm: " << _T.squared_norm() << std::endl;
    tbb::parallel_for(tbb::blocked_range<int>(0,lagrangian_particles.pos_current.size(),TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
    {
        for (int i = r.begin(); i < r.end(); ++i) {
            Vec<3, MyReal> p_vel = lagrangian_particles.vel[i];
            Vec<3, MyReal> p_pos = lagrangian_particles.pos_current[i];
            Vec<3, MyReal> curr_dv = getVelocityBSpline(p_pos, _u_diff, _v_diff, _w_diff);
            p_vel = (flip)*p_vel +
                    (1.-flip)*getVelocityBSpline(p_pos, _u_save, _v_save, _w_save) + 
                    curr_dv;

            lagrangian_particles.vel[i] = p_vel;
            lagrangian_particles.vel_temp[i] = p_vel;
            
            {
                Eigen::Matrix3<MyReal> dveldx = getJacobianVelocity(p_pos, _un, _vn, _wn);
                Vec<4, MyReal> dudxdy = sampleCrossHessianField(p_pos - Vec<3, MyReal>(0.0f, 0.5f*_h, 0.5f*_h), _un, true);
                Vec<4, MyReal> dvdxdy = sampleCrossHessianField(p_pos - Vec<3, MyReal>(0.5f*_h, 0.0f, 0.5f*_h), _vn, true);
                Vec<4, MyReal> dwdxdy = sampleCrossHessianField(p_pos - Vec<3, MyReal>(0.5f*_h, 0.5f*_h, 0.0f), _wn, true);
                lagrangian_particles.C_u[i] = {{lagrangian_particles.vel[i][0], 
                    dveldx(0,0), dveldx(0,1), dveldx(0,2),
                    dudxdy[0], dudxdy[1], dudxdy[2], dudxdy[3]}};
                lagrangian_particles.C_v[i] = {{lagrangian_particles.vel[i][1], 
                    dveldx(1,0), dveldx(1,1), dveldx(1,2),
                    dvdxdy[0], dvdxdy[1], dvdxdy[2], dvdxdy[3]}};
                lagrangian_particles.C_w[i] = {{lagrangian_particles.vel[i][2], 
                    dveldx(2,0), dveldx(2,1), dveldx(2,2),
                    dwdxdy[0], dwdxdy[1], dwdxdy[2], dwdxdy[3]}};
            }

            if (!do_vel_advection_only)
            {
                MyReal p_rho = lagrangian_particles.rho[i];
                MyReal p_T = lagrangian_particles.T[i];
                Vec<3, MyReal> zeroFormPos = p_pos - _amped_h*Vec<3, MyReal>(0.5f);
                p_rho = flip*(p_rho + sampleField(zeroFormPos, _rho_diff, true, true))
                        + (1-flip)*sampleField(zeroFormPos, _rho, true, true);
                p_T = flip*(p_T + sampleField(zeroFormPos, _T_diff, true, true))
                        + (1-flip)*sampleField(zeroFormPos, _T, true, true);
                {
                    Vec<3, MyReal> drhodx = sampleGradientField(zeroFormPos, _rho, true, true);
                    Vec<4, MyReal> drhodxdy = sampleCrossHessianField(zeroFormPos, _rho, true, true);
                    Vec<3, MyReal> dTdx = sampleGradientField(zeroFormPos, _T, true, true);
                    Vec<4, MyReal> dTdxdy = sampleCrossHessianField(zeroFormPos, _T, true, true);
                    lagrangian_particles.C_rho[i] = {{p_rho, 
                        drhodx[0], drhodx[1], drhodx[2], 
                        drhodxdy[0], drhodxdy[1], drhodxdy[2], drhodxdy[3]}};
                    lagrangian_particles.C_T[i] = {{p_T, 
                        dTdx[0], dTdx[1], dTdx[2], 
                        dTdxdy[0], dTdxdy[1], dTdxdy[2], dTdxdy[3]}};
                }
                lagrangian_particles.rho[i] = p_rho;
                lagrangian_particles.T[i] = p_T;
            }
        }
    });
    std::cout << BLUE <<  "Particle information sampled from the grid!" << RESET << std::endl;
    std::cout << BLUE <<  "end of frame!" << RESET << std::endl;
}

void COFLIPSolver::setInitialVelocity(MyReal inflow_vel)
{
    int compute_elements = _un._nx * _un._ny * _un._nz;
    int slice = _un._nx * _un._ny;

    tbb::parallel_for(0, compute_elements, 1, [&](int thread_idx) {

        uint k = thread_idx/slice;
        uint j = (thread_idx%slice)/_un._nx;
        uint i = thread_idx%(_un._nx);

        _un(i,j,k) = inflow_vel;
    });
}

void COFLIPSolver::setVelocityFromEmitter(bool do_only_x_dir_vel)
{
    for(auto &emitter : sim_emitter)
    {
        openvdb::tools::GridSampler<openvdb::Grid<openvdb::tree::Tree4<MyReal>::Type>, openvdb::tools::BoxSampler> box_sampler(*emitter.e_sdf);

        if (emitter.do_set_velocities)
        {
            if (sim_scheme != Scheme::CO_FLIP) {
                int compute_elements = _un._blockx * _un._blocky * _un._blockz;
                int slice = _un._blockx * _un._blocky;

                tbb::parallel_for(0, compute_elements, 1, [&](int thread_idx) {

                    uint bk = thread_idx/slice;
                    uint bj = (thread_idx%slice)/_un._blockx;
                    uint bi = thread_idx%(_un._blockx);

                    for (uint kk=0;kk<8;kk++)for(uint jj=0;jj<8;jj++)for(uint ii=0;ii<8;ii++)
                    {
                        uint i=bi*8+ii, j=bj*8+jj, k=bk*8+kk;

                        if(i<_un._nx && j<_un._ny && k<_un._nz)
                        {
                            MyReal w_x = ((MyReal)i-_un._ox)*_h_uniform;
                            MyReal w_y = ((MyReal)j-_un._oy)*_h_uniform;
                            MyReal w_z = ((MyReal)k-_un._oz)*_h_uniform;
                            Vec<3, MyReal> world_pos(w_x, w_y, w_z);
                            MyReal sdf_value = box_sampler.wsSample(openvdb::math::Vec3<MyReal>(w_x, w_y, w_z));
                            if (emitter.set_vel_everywhere_from_vel_func || sdf_value <= 0)
                            {
                                _un(i,j,k) = emitter.emit_velocity(world_pos+Vec<3, MyReal>(0.5*_h_uniform))[0];
                            }
                        }
                    }
                });

                if (!do_only_x_dir_vel)
                {
                    compute_elements = _vn._blockx*_vn._blocky*_vn._blockz;
                    slice = _vn._blockx*_vn._blocky;

                    tbb::parallel_for(0, compute_elements, 1, [&](int thread_idx) {

                        uint bk = thread_idx/slice;
                        uint bj = (thread_idx%slice)/_vn._blockx;
                        uint bi = thread_idx%(_vn._blockx);

                        for (uint kk=0;kk<8;kk++)for(uint jj=0;jj<8;jj++)for(uint ii=0;ii<8;ii++)
                        {
                            uint i=bi*8+ii, j=bj*8+jj, k=bk*8+kk;
                            if(i<_vn._nx && j<_vn._ny && k<_vn._nz)
                            {
                                MyReal w_x = ((MyReal)i-_vn._ox)*_h_uniform;
                                MyReal w_y = ((MyReal)j-_vn._oy)*_h_uniform;
                                MyReal w_z = ((MyReal)k-_vn._oz)*_h_uniform;
                                Vec<3, MyReal> world_pos(w_x, w_y, w_z);
                                MyReal sdf_value = box_sampler.wsSample(openvdb::math::Vec3<MyReal>(w_x, w_y, w_z));
                                if (emitter.set_vel_everywhere_from_vel_func || sdf_value <= 0)
                                {
                                    _vn(i,j,k) = emitter.emit_velocity(world_pos+Vec<3, MyReal>(0.5*_h_uniform))[1];
                                }
                            }
                        }
                    });

                    compute_elements = _wn._blockx*_wn._blocky*_wn._blockz;
                    slice = _wn._blockx*_wn._blocky;

                    tbb::parallel_for(0, compute_elements, 1, [&](int thread_idx) {

                        uint bk = thread_idx/slice;
                        uint bj = (thread_idx%slice)/_wn._blockx;
                        uint bi = thread_idx%(_wn._blockx);

                        for (uint kk=0;kk<8;kk++)for(uint jj=0;jj<8;jj++)for(uint ii=0;ii<8;ii++)
                        {
                            uint i=bi*8+ii, j=bj*8+jj, k=bk*8+kk;
                            if(i<_wn._nx && j<_wn._ny && k<_wn._nz)
                            {
                                MyReal w_x = ((MyReal)i-_wn._ox)*_h_uniform;
                                MyReal w_y = ((MyReal)j-_wn._oy)*_h_uniform;
                                MyReal w_z = ((MyReal)k-_wn._oz)*_h_uniform;
                                Vec<3, MyReal> world_pos(w_x, w_y, w_z);
                                MyReal sdf_value = box_sampler.wsSample(openvdb::math::Vec3<MyReal>(w_x, w_y, w_z));
                                if (emitter.set_vel_everywhere_from_vel_func || sdf_value <= 0)
                                {
                                    _wn(i,j,k) = emitter.emit_velocity(world_pos+Vec<3, MyReal>(0.5*_h_uniform))[2];
                                }
                            }
                        }
                    });
                }
            } else {
                seedParticles();
                tbb::parallel_for(tbb::blocked_range<int>(0,lagrangian_particles.pos_current.size(), TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
                {
                    for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
                        Vec<3, MyReal> spos = lagrangian_particles.pos_current[tIdx];
                        lagrangian_particles.vel_temp[tIdx] = emitter.emit_velocity(spos);
                        lagrangian_particles.vel[tIdx] = lagrangian_particles.vel_temp[tIdx];
                    }
                });
                solveInterpDagger(0, 1);
                int ni = _nx;
                int nj = _ny;
                int nk = _nz;
                int x_nF = (ni+1)*nj*nk;
                int y_nF = ni*(nj+1)*nk;
                tbb::parallel_for(tbb::blocked_range<int>(0,_nF, TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
                {
                    for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
                        bool is_x_dir = tIdx < x_nF;
                        bool is_y_dir = !is_x_dir && tIdx < (x_nF + y_nF);
                        int comp_slice = is_x_dir ? (ni+1)*nj : (is_y_dir ? ni*(nj+1) : ni*nj);
                        int comp_n = is_x_dir ? (ni+1) : ni;
                        int comp_tIdx = is_x_dir ? tIdx : (is_y_dir ? (tIdx - x_nF) : (tIdx - (x_nF + y_nF)));
                        int k = comp_tIdx/comp_slice;
                        int j = (comp_tIdx%comp_slice)/comp_n;
                        int i = comp_tIdx%comp_n;
                        if (is_x_dir) {
                            _un(i,j,k) = fluxes[tIdx];
                        } else if (is_y_dir) {
                            _vn(i,j,k) = fluxes[tIdx];
                        } else {
                            _wn(i,j,k) = fluxes[tIdx];
                        }
                    }
                });
                fluxes_midpoint = fluxes;
            }
        }
    }
}

void COFLIPSolver::updateEmitters(int framenum, MyReal dt) {
    for(auto &emitter : sim_emitter)
    {
        emitter.update(framenum, (_h_uniform*(double)_nx) / 64., dt);
    }
}

void COFLIPSolver::addEmitterForce(int framenum, MyReal dt)
{
    for(auto &emitter : sim_emitter)
    {
        openvdb::tools::GridSampler<openvdb::Grid<openvdb::tree::Tree4<MyReal>::Type>, openvdb::tools::BoxSampler> box_sampler(*emitter.e_sdf);
        if(framenum < emitter.emitFrame)
        {
            int compute_elements = 0;
            int slice = 0;
            if (emitter.do_set_velocities)
            {
                compute_elements = _un._blockx * _un._blocky * _un._blockz;
                slice = _un._blockx * _un._blocky;

                tbb::parallel_for(0, compute_elements, 1, [&](int thread_idx) {

                    uint bk = thread_idx/slice;
                    uint bj = (thread_idx%slice)/_un._blockx;
                    uint bi = thread_idx%(_un._blockx);

                    for (uint kk=0;kk<8;kk++)for(uint jj=0;jj<8;jj++)for(uint ii=0;ii<8;ii++)
                    {
                        uint i=bi*8+ii, j=bj*8+jj, k=bk*8+kk;

                        if(i<_un._nx && j<_un._ny && k<_un._nz)
                        {
                            MyReal w_x = ((MyReal)i-_un._ox)*_h_uniform;
                            MyReal w_y = ((MyReal)j-_un._oy)*_h_uniform;
                            MyReal w_z = ((MyReal)k-_un._oz)*_h_uniform;
                            Vec<3, MyReal> world_pos(w_x, w_y, w_z);
                            MyReal sdf_value = box_sampler.wsSample(openvdb::math::Vec3<MyReal>(w_x, w_y, w_z));
                            if (sdf_value <= 0)
                            {
                                _un(i,j,k) = emitter.emit_velocity(world_pos+Vec<3, MyReal>(0.5*_h_uniform))[0];
                            }
                        }
                    }
                });

                compute_elements = _vn._blockx*_vn._blocky*_vn._blockz;
                slice = _vn._blockx*_vn._blocky;

                tbb::parallel_for(0, compute_elements, 1, [&](int thread_idx) {

                    uint bk = thread_idx/slice;
                    uint bj = (thread_idx%slice)/_vn._blockx;
                    uint bi = thread_idx%(_vn._blockx);

                    for (uint kk=0;kk<8;kk++)for(uint jj=0;jj<8;jj++)for(uint ii=0;ii<8;ii++)
                    {
                        uint i=bi*8+ii, j=bj*8+jj, k=bk*8+kk;
                        if(i<_vn._nx && j<_vn._ny && k<_vn._nz)
                        {
                            MyReal w_x = ((MyReal)i-_vn._ox)*_h_uniform;
                            MyReal w_y = ((MyReal)j-_vn._oy)*_h_uniform;
                            MyReal w_z = ((MyReal)k-_vn._oz)*_h_uniform;
                            Vec<3, MyReal> world_pos(w_x, w_y, w_z);
                            MyReal sdf_value = box_sampler.wsSample(openvdb::math::Vec3<MyReal>(w_x, w_y, w_z));
                            if (sdf_value <= 0)
                            {
                                _vn(i,j,k) = emitter.emit_velocity(world_pos+Vec<3, MyReal>(0.5*_h_uniform))[1];
                            }
                        }
                    }
                });

                compute_elements = _wn._blockx*_wn._blocky*_wn._blockz;
                slice = _wn._blockx*_wn._blocky;

                tbb::parallel_for(0, compute_elements, 1, [&](int thread_idx) {

                    uint bk = thread_idx/slice;
                    uint bj = (thread_idx%slice)/_wn._blockx;
                    uint bi = thread_idx%(_wn._blockx);

                    for (uint kk=0;kk<8;kk++)for(uint jj=0;jj<8;jj++)for(uint ii=0;ii<8;ii++)
                    {
                        uint i=bi*8+ii, j=bj*8+jj, k=bk*8+kk;
                        if(i<_wn._nx && j<_wn._ny && k<_wn._nz)
                        {
                            MyReal w_x = ((MyReal)i-_wn._ox)*_h_uniform;
                            MyReal w_y = ((MyReal)j-_wn._oy)*_h_uniform;
                            MyReal w_z = ((MyReal)k-_wn._oz)*_h_uniform;
                            Vec<3, MyReal> world_pos(w_x, w_y, w_z);
                            MyReal sdf_value = box_sampler.wsSample(openvdb::math::Vec3<MyReal>(w_x, w_y, w_z));
                            if (sdf_value <= 0)
                            {
                                _wn(i,j,k) = emitter.emit_velocity(world_pos+Vec<3, MyReal>(0.5*_h_uniform))[2];
                            }
                        }
                    }
                });
            }
        }
    }
}

void COFLIPSolver::emitSmoke(int framenum, MyReal dt)
{
    for(auto &emitter : sim_emitter)
    {
        openvdb::tools::GridSampler<openvdb::Grid<openvdb::tree::Tree4<MyReal>::Type>, openvdb::tools::BoxSampler> box_sampler(*emitter.e_sdf);
        if(framenum < emitter.emitFrame)
        {
            if (!do_vel_advection_only)
            {
                int compute_elements = _rho._blockx * _rho._blocky * _rho._blockz;
                int slice = _rho._blockx * _rho._blocky;

                std::uniform_real_distribution<double> unif(0, 1);
                tbb::parallel_for(0, compute_elements, 1, [&](int thread_idx) {

                    uint bk = thread_idx / slice;
                    uint bj = (thread_idx % slice) / _rho._blockx;
                    uint bi = thread_idx % (_rho._blockx);

                    std::mt19937_64 rng;
                    rng.seed(thread_idx);
                    for (uint kk = 0; kk < 8; kk++)for (uint jj = 0; jj < 8; jj++)for (uint ii = 0; ii < 8; ii++)
                    {
                        uint i = bi * 8 + ii, j = bj * 8 + jj, k = bk * 8 + kk;
                        if (i < _rho._nx && j < _rho._ny && k < _rho._nz)
                        {
                            MyReal w_x = ((MyReal)i - _rho._ox) * _amped_h;
                            MyReal w_y = ((MyReal)j - _rho._oy) * _amped_h;
                            MyReal w_z = ((MyReal)k - _rho._oz) * _amped_h;
                            MyReal sdf_value = box_sampler.wsSample(openvdb::math::Vec3<MyReal>(w_x, w_y, w_z));
                            if (sdf_value <= 0)
                            {
                                MyReal multiplier = 1.0;
                                MyReal thickness = 2 * (_h_uniform*(double)_nx) / 64.;
                                multiplier = -sdf_value / thickness;
                                MyReal rand_mulitplier = 1.f;
                                if (emitter.do_randomize_density)
                                {
                                    rand_mulitplier += unif(rng);
                                }
                                _rho(i, j, k) += rand_mulitplier * multiplier * emitter.emit_density;
                                _T(i, j, k) += rand_mulitplier * multiplier * emitter.emit_temperature;
                            }
                        }
                    }
                });
            }
        }
    }
}

void COFLIPSolver::addBuoyancy(Buffer3D<MyReal>& u_to_change, Buffer3D<MyReal>& v_to_change, Buffer3D<MyReal>& w_to_change, MyReal dt)
{
    if (_alpha == 0.0f && _beta == 0.0f)
        return;

    int compute_elements = v_to_change._blockx*v_to_change._blocky*v_to_change._blockz;
    int slice = v_to_change._blockx*v_to_change._blocky;

    tbb::parallel_for(0, compute_elements, 1, [&](int thread_idx) {

        uint bk = thread_idx/slice;
        uint bj = (thread_idx%slice)/v_to_change._blockx;
        uint bi = thread_idx%(v_to_change._blockx);

        for (uint kk=0;kk<8;kk++)for(uint jj=0;jj<8;jj++)for(uint ii=0;ii<8;ii++)
        {
            uint i=bi*8+ii, j=bj*8+jj, k=bk*8+kk;
            if(i<v_to_change._nx && j<v_to_change._ny && k<v_to_change._nz)
            {
                Vec<3, MyReal> p_pos = _h_uniform * (Vec<3, MyReal>(i,j,k) + Vec<3, MyReal>(0.5f, 0.0f, 0.5f));
                Vec<3, MyReal> zeroFormPos = p_pos - _amped_h*Vec<3, MyReal>(0.5f);
                MyReal density = sampleField(zeroFormPos, _rho, true, true);
                MyReal temperature = sampleField(zeroFormPos, _T, true, true);
                MyReal f = -dt*_alpha*density + dt*_beta*temperature;

                v_to_change(i,j,k) += f*sin(theta);
            }
        }
    });

    compute_elements = u_to_change._blockx*u_to_change._blocky*u_to_change._blockz;
    slice = u_to_change._blockx*u_to_change._blocky;
    tbb::parallel_for(0, compute_elements, 1, [&](int thread_idx) {

        uint bk = thread_idx/slice;
        uint bj = (thread_idx%slice)/u_to_change._blockx;
        uint bi = thread_idx%(u_to_change._blockx);

        for (uint kk=0;kk<8;kk++)for(uint jj=0;jj<8;jj++)for(uint ii=0;ii<8;ii++)
        {
            uint i=bi*8+ii, j=bj*8+jj, k=bk*8+kk;
            if(i<u_to_change._nx && j<u_to_change._ny && k<u_to_change._nz)
            {
                Vec<3, MyReal> p_pos = _h_uniform * (Vec<3, MyReal>(i,j,k) + Vec<3, MyReal>(0.0f, 0.5f, 0.5f));
                Vec<3, MyReal> zeroFormPos = p_pos - _amped_h*Vec<3, MyReal>(0.5f);
                MyReal density = sampleField(zeroFormPos, _rho, true, true);
                MyReal temperature = sampleField(zeroFormPos, _T, true, true);
                MyReal f = -dt*_alpha*density + dt*_beta*temperature;

                u_to_change(i,j,k) += f*cos(theta)*cos(phi);
            }
        }
    });

    compute_elements = w_to_change._blockx*w_to_change._blocky*w_to_change._blockz;
    slice = w_to_change._blockx*w_to_change._blocky;
    tbb::parallel_for(0, compute_elements, 1, [&](int thread_idx) {

        uint bk = thread_idx/slice;
        uint bj = (thread_idx%slice)/w_to_change._blockx;
        uint bi = thread_idx%(w_to_change._blockx);

        for (uint kk=0;kk<8;kk++)for(uint jj=0;jj<8;jj++)for(uint ii=0;ii<8;ii++)
        {
            uint i=bi*8+ii, j=bj*8+jj, k=bk*8+kk;
            if(i<w_to_change._nx && j<w_to_change._ny && k<w_to_change._nz)
            {
                Vec<3, MyReal> p_pos = _h_uniform * (Vec<3, MyReal>(i,j,k) + Vec<3, MyReal>(0.5f, 0.5f, 0.0f));
                Vec<3, MyReal> zeroFormPos = p_pos - _amped_h*Vec<3, MyReal>(0.5f);
                MyReal density = sampleField(zeroFormPos, _rho, true, true);
                MyReal temperature = sampleField(zeroFormPos, _T, true, true);
                MyReal f = -dt*_alpha*density + dt*_beta*temperature;

                w_to_change(i,j,k+1) += f*cos(theta)*sin(phi);
            }
        }
    });
}

void COFLIPSolver::setSmoke(MyReal drop, MyReal raise, const std::vector<Emitter> &emitters)
{
    _alpha = drop;
    _beta = raise;
    sim_emitter = emitters;
}

void COFLIPSolver::clearBoundary(Buffer3D<MyReal>& field)
{
    int compute_elements = field._blockx* field._blocky* field._blockz;

    int slice = field._blockx* field._blocky;
    tbb::parallel_for(0, compute_elements, 1, [&](int thread_idx) {

        uint bk = thread_idx/slice;
        uint bj = (thread_idx%slice)/ field._blockx;
        uint bi = thread_idx%(field._blockx);

        for (uint kk=0;kk<8;kk++)for(uint jj=0;jj<8;jj++)for(uint ii=0;ii<8;ii++)
        {
            uint i=bi*8+ii, j=bj*8+jj, k=bk*8+kk;
            if(_b_desc(i,j,k)==1 || _b_desc(i,j,k)==3)
            {
                field(i,j,k) = 0;
            }
        }
    });
}

void COFLIPSolver::updateBoundary(int framenum, MyReal dt)
{
    _b_desc.setZero();
    if (!sim_boundary.empty()) {
        _usolid.init(_nx+1, _ny, _nz, _h, 0.5f, 0.f, 0.f);
        _vsolid.init(_nx, _ny+1, _nz, _h, 0.f, 0.5f, 0.f);
        _wsolid.init(_nx, _ny, _nz+1, _h, 0.f, 0.f, 0.5f);
    }
    for(auto &boundary : sim_boundary)
    {
        Vec<3, MyReal> boundary_vel = boundary.vel_func(framenum);
        boundary.update(framenum, _h, dt);
        openvdb::tools::GridSampler<openvdb::Grid<openvdb::tree::Tree4<float>::Type>, openvdb::tools::BoxSampler> box_sampler(*boundary.b_sdf);

        int compute_elements = _b_desc._blockx*_b_desc._blocky*_b_desc._blockz;
        int slice = _b_desc._blockx*_b_desc._blocky;

        tbb::parallel_for(0, compute_elements, 1, [&](int thread_idx) {

            uint bk = thread_idx/slice;
            uint bj = (thread_idx%slice)/_b_desc._blockx;
            uint bi = thread_idx%(_b_desc._blockx);

            for (uint kk=0;kk<8;kk++)for(uint jj=0;jj<8;jj++)for(uint ii=0;ii<8;ii++)
            {
                uint i=bi*8+ii, j=bj*8+jj, k=bk*8+kk;
                if(i<_b_desc._nx && j<_b_desc._ny && k<_b_desc._nz)
                {
                    MyReal w_x = ((MyReal)i-_b_desc._ox)*_h;
                    MyReal w_y = ((MyReal)j-_b_desc._oy)*_h;
                    MyReal w_z = ((MyReal)k-_b_desc._oz)*_h;
                    MyReal sdf_value = box_sampler.wsSample(openvdb::math::Vec3<MyReal>(w_x, w_y, w_z));
                    if (sdf_value <= 0.f)
                    {
                        _b_desc(i,j,k) = 3;
                    }
                }
            }
        });

        compute_elements = _un._blockx*_un._blocky*_un._blockz;
        slice = _un._blockx*_un._blocky;

        tbb::parallel_for(0, compute_elements, 1, [&](int thread_idx) {

            uint bk = thread_idx/slice;
            uint bj = (thread_idx%slice)/_un._blockx;
            uint bi = thread_idx%(_un._blockx);

            for (uint kk=0;kk<8;kk++)for(uint jj=0;jj<8;jj++)for(uint ii=0;ii<8;ii++)
            {
                uint i=bi*8+ii, j=bj*8+jj, k=bk*8+kk;
                if(i<_un._nx && j<_un._ny && k<_un._nz)
                {
                    MyReal w_x = ((MyReal)i-_un._ox)*_h;
                    MyReal w_y = ((MyReal)j-_un._oy)*_h;
                    MyReal w_z = ((MyReal)k-_un._oz)*_h;
                    MyReal sdf_value = box_sampler.wsSample(openvdb::math::Vec3<MyReal>(w_x, w_y, w_z));
                    if (sdf_value <= 0)
                    {
                        _usolid(i,j,k) = boundary_vel[0];
                    }
                }
            }
        });

        compute_elements = _vn._blockx*_vn._blocky*_vn._blockz;
        slice = _vn._blockx*_vn._blocky;

        tbb::parallel_for(0, compute_elements, 1, [&](int thread_idx) {

            uint bk = thread_idx/slice;
            uint bj = (thread_idx%slice)/_vn._blockx;
            uint bi = thread_idx%(_vn._blockx);

            for (uint kk=0;kk<8;kk++)for(uint jj=0;jj<8;jj++)for(uint ii=0;ii<8;ii++)
            {
                uint i=bi*8+ii, j=bj*8+jj, k=bk*8+kk;
                if(i<_vn._nx && j<_vn._ny && k<_vn._nz)
                {
                    MyReal w_x = ((MyReal)i-_vn._ox)*_h;
                    MyReal w_y = ((MyReal)j-_vn._oy)*_h;
                    MyReal w_z = ((MyReal)k-_vn._oz)*_h;
                    MyReal sdf_value = box_sampler.wsSample(openvdb::math::Vec3<MyReal>(w_x, w_y, w_z));
                    if (sdf_value <= 0)
                    {
                        _vsolid(i,j,k) = boundary_vel[1];
                    }
                }
            }
        });

        compute_elements = _wn._blockx*_wn._blocky*_wn._blockz;
        slice = _wn._blockx*_wn._blocky;

        tbb::parallel_for(0, compute_elements, 1, [&](int thread_idx) {

            uint bk = thread_idx/slice;
            uint bj = (thread_idx%slice)/_wn._blockx;
            uint bi = thread_idx%(_wn._blockx);

            for (uint kk=0;kk<8;kk++)for(uint jj=0;jj<8;jj++)for(uint ii=0;ii<8;ii++)
            {
                uint i=bi*8+ii, j=bj*8+jj, k=bk*8+kk;
                if(i<_wn._nx && j<_wn._ny && k<_wn._nz)
                {
                    MyReal w_x = ((MyReal)i-_wn._ox)*_h;
                    MyReal w_y = ((MyReal)j-_wn._oy)*_h;
                    MyReal w_z = ((MyReal)k-_wn._oz)*_h;
                    MyReal sdf_value = box_sampler.wsSample(openvdb::math::Vec3<MyReal>(w_x, w_y, w_z));
                    if (sdf_value <= 0)
                    {
                        _wsolid(i,j,k) = boundary_vel[2];
                    }
                }
            }
        });
    }
}

void COFLIPSolver::setBoundary(const std::vector<Boundary> &boundaries)
{
    sim_boundary = boundaries;
}

MyReal COFLIPSolver::getCFL()
{
    int res = _h/(10./128./4.)+TOLERANCE;
    std::uniform_real_distribution<double> unif(0, 1);
    max_v = tbb::parallel_reduce( 
                tbb::blocked_range<int>(0,_nzp*_nyp*_nxp),
                TOLERANCE,
                [&](tbb::blocked_range<int> range, double running_max)
                {
                    for (int tIdx=range.begin(); tIdx<range.end(); ++tIdx)
                    {
                        std::mt19937_64 rng;
                        rng.seed(tIdx);
                        int k = tIdx / (_nxp*_nyp);
                        int j = (tIdx % (_nxp*_nyp)) / _nxp;
                        int i = tIdx % _nxp;
                        double x = (double)i*_h;
                        double y = (double)j*_h;
                        double z = (double)k*_h;
                        for(int kk=0;kk<res;kk++)
                        {
                            for(int jj=0;jj<res;jj++)
                            {
                                for(int ii=0;ii<res;ii++)
                                {
                                    Vec<3, double> spos(x+((double)ii + unif(rng))/(double)res*_h,
                                                        y+((double)jj + unif(rng))/(double)res*_h,
                                                        z+((double)kk + unif(rng))/(double)res*_h);
                                    Vec<3, double> vel = getVelocityBSpline(spos, _un, _vn, _wn);
                                    running_max = std::max(running_max, mag(vel));
                                }
                            }
                        }
                    }

                    return running_max;
                }, [](double a, double b) { return std::max(a,b); } );
    return _h / max_v;
}

bool COFLIPSolver::edgeInFluid(int dir, int i, int j, int k, int res_amp)
{
    if (dir == 0)
    {
        // x
        // int ni = _nx;
        int nj = (_ny*res_amp)+1;
        int nk = (_nz*res_amp)+1;
        if (j == 0 || k == 0 || j == nj - 1 || k == nk - 1)
        {
            return false;
        } else if (res_amp != 1) {
            return true;
        }
        bool is_solid1 = _b_desc(i, j, k) == 3;
        bool is_solid2 = _b_desc(i, j - 1, k) == 3;
        bool is_solid3 = _b_desc(i, j, k - 1) == 3;
        bool is_solid4 = _b_desc(i, j - 1, k - 1) == 3;
        return !(is_solid1 || is_solid2 || is_solid3 || is_solid4);
    }
    else if (dir == 1)
    {
        // y
        int ni = (_nx*res_amp)+1;
        // int nj = _ny;
        int nk = (_nz*res_amp)+1;
        if (i == 0 || k == 0 || i == ni - 1 || k == nk - 1)
        {
            return false;
        } else if (res_amp != 1) {
            return true;
        }
        bool is_solid1 = _b_desc(i, j, k) == 3;
        bool is_solid2 = _b_desc(i - 1, j, k) == 3;
        bool is_solid3 = _b_desc(i, j, k - 1) == 3;
        bool is_solid4 = _b_desc(i - 1, j, k - 1) == 3;
        return !(is_solid1 || is_solid2 || is_solid3 || is_solid4);
    }
    else
    {
        // z
        int ni = (_nx*res_amp)+1;
        int nj = (_ny*res_amp)+1;
        // int nk = _nz;
        if (i == 0 || j == 0 || i == ni - 1 || j == nj - 1)
        {
            return false;
        } else if (res_amp != 1) {
            return true;
        }
        bool is_solid1 = _b_desc(i, j, k) == 3;
        bool is_solid2 = _b_desc(i - 1, j, k) == 3;
        bool is_solid3 = _b_desc(i, j - 1, k) == 3;
        bool is_solid4 = _b_desc(i - 1, j - 1, k) == 3;

        return !(is_solid1 || is_solid2 || is_solid3 || is_solid4);
    }
    return true;
}

bool COFLIPSolver::vertexInFluid(int i, int j, int k)
{
    int ni = _nx+1;
    int nj = _ny+1;
    int nk = _nz+1;
    if (i == 0 || j == 0 || k == 0 ||
        i == ni - 1 || j == nj - 1 || k == nk - 1)
    {
        return false;
    }
    bool is_solid1 = _b_desc(i, j, k) == 3;
    bool is_solid2 = _b_desc(i, j - 1, k) == 3;
    bool is_solid3 = _b_desc(i, j, k - 1) == 3;
    bool is_solid4 = _b_desc(i, j - 1, k - 1) == 3;
    bool is_solid5 = _b_desc(i - 1, j, k) == 3;
    bool is_solid6 = _b_desc(i - 1, j - 1, k) == 3;
    bool is_solid7 = _b_desc(i - 1, j, k - 1) == 3;
    bool is_solid8 = _b_desc(i - 1, j - 1, k - 1) == 3;
    return !(is_solid1 || is_solid2 || is_solid3 || is_solid4 ||
             is_solid5 || is_solid6 || is_solid7 || is_solid8);
}

void COFLIPSolver::setupPressureProjection(MyReal dt)
{
    std::cout << "[Setting projection required matrices...]" << std::endl;
    int ni = _nx;
    int nj = _ny;
    int nk = _nz;
    int x_nF = (ni+1)*nj*nk;
    int y_nF = ni*(nj+1)*nk;
    int x_nE = ni*(nj+1)*(nk+1);
    int y_nE = (ni+1)*nj*(nk+1);
    Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> lumped_starflux_matrix;
    {
        Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> d2transpose(_nF, _nC);
        d2transpose.reserve(Eigen::VectorXi::Constant(_nF, 2));
        lumped_starflux_matrix.resize(_nF, _nF);
        lumped_starflux_matrix.reserve(Eigen::VectorXi::Constant(_nF, 1));
        tbb::parallel_for(tbb::blocked_range<int>(0,_nF, TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
        {
            for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
                bool is_x_dir = tIdx < x_nF;
                bool is_y_dir = !is_x_dir && tIdx < (x_nF + y_nF);
                bool is_z_dir = !is_x_dir && !is_y_dir;
                int comp_slice = is_x_dir ? (ni+1)*nj : (is_y_dir ? ni*(nj+1) : ni*nj);
                int comp_n = is_x_dir ? (ni+1) : ni;
                int comp_tIdx = is_x_dir ? tIdx : (is_y_dir ? (tIdx - x_nF) : (tIdx - (x_nF + y_nF)));
                int k = comp_tIdx/comp_slice;
                int j = (comp_tIdx%comp_slice)/comp_n;
                int i = comp_tIdx%comp_n;
                int tIdx_threeForm = k*(ni*nj)+j*ni+i;
                if (is_x_dir && i == 0) {
                    if (_b_desc(i,j,k)==1) d2transpose.insert(tIdx, tIdx_threeForm) = 1.0;
                } else if (is_x_dir && i == ni) {
                    if (_b_desc(i-1,j,k)==1) d2transpose.insert(tIdx, tIdx_threeForm-1) = -1.0;
                } else if (is_y_dir && j == 0) {
                    if (_b_desc(i,j,k)==1) d2transpose.insert(tIdx, tIdx_threeForm) = 1.0;
                } else if (is_y_dir && j == nj) {
                    if (_b_desc(i,j-1,k)==1) d2transpose.insert(tIdx, tIdx_threeForm-ni) = -1.0;
                } else if (is_z_dir && k == 0) {
                    if (_b_desc(i,j,k)==1) d2transpose.insert(tIdx, tIdx_threeForm) = 1.0;
                } else if (is_z_dir && k == nk) {
                    if (_b_desc(i,j,k-1)==1) d2transpose.insert(tIdx, tIdx_threeForm-(ni*nj)) = -1.0;
                } else if ((is_x_dir && _b_desc(i,j,k)==0 && _b_desc(i-1,j,k)==0) ||
                        (is_y_dir && _b_desc(i,j,k)==0 && _b_desc(i,j-1,k)==0) ||
                        (is_z_dir && _b_desc(i,j,k)==0 && _b_desc(i,j,k-1)==0)) {
                    d2transpose.insert(tIdx, tIdx_threeForm - (is_x_dir ? 1 : (is_y_dir ? ni : (ni*nj)))) = -1.0;
                    d2transpose.insert(tIdx, tIdx_threeForm) = 1.0;
                }
                int idx = is_x_dir ? i : (is_y_dir ? j : k);
                int nidx = is_x_dir ? ni : (is_y_dir ? nj : nk);
                int other1_idx = is_x_dir ? j : (is_y_dir ? k : i);
                int other1_nidx = is_x_dir ? nj : (is_y_dir ? nk : ni);
                int other2_idx = is_x_dir ? k : (is_y_dir ? i : j);
                int other2_nidx = is_x_dir ? nk : (is_y_dir ? ni : nj);
                if (sim_scheme == Scheme::CO_FLIP) {
                    if (use_DEC_diagonal_hodge_star) {
                        double length_of_dualedge = bs_p == 3 ? ((idx == 0 || idx == nidx) ? 1./3. : ((idx == 1 || idx == (nidx-1)) ? 0.5 : ((idx == 2 || idx == (nidx-2)) ? 5./6. : 1.0))) : (bs_p == 2 ? ((idx == 0 || idx == nidx) ? 0.5 : ((idx == 1 || idx == (nidx-1)) ? 0.75 : 1.0)) : 1.0);
                        double length_of_edge1 = bs_p == 3 ? ((other1_idx == 0 || other1_idx == (other1_nidx-1)) ? 1./3. : ((other1_idx == 1 || other1_idx == (other1_nidx-2)) ? 2./3. : 1.0)) : (bs_p == 2 ? ((other1_idx == 0 || other1_idx == (other1_nidx-1)) ? 0.5 : 1.0) : 1.0);
                        double length_of_edge2 = bs_p == 3 ? ((other2_idx == 0 || other2_idx == (other2_nidx-1)) ? 1./3. : ((other2_idx == 1 || other2_idx == (other2_nidx-2)) ? 2./3. : 1.0)) : (bs_p == 2 ? ((other2_idx == 0 || other2_idx == (other2_nidx-1)) ? 0.5 : 1.0) : 1.0);
                        double area_of_primalface = length_of_edge1 * length_of_edge2;
                        double factor = length_of_dualedge / area_of_primalface;
                        lumped_starflux_matrix.insert(tIdx, tIdx) = factor;
                    } else {
                        std::vector<std::array<double, 7> > Bs = bs_p == 3 ? B3s : (bs_p == 2 ? B2s : B1s);
                        std::vector<std::array<double, 7> > Ds = bs_p == 3 ? D2s : (bs_p == 2 ? D1s : D0s);
                        std::array<double, 7> Barray = Bs[std::min(std::min(2*bs_p-1, idx), nidx-idx)];
                        std::array<double, 7> Darray1 = Ds[std::min(std::min(2*(bs_p-1), other1_idx), (other1_nidx-1)-other1_idx)];
                        std::array<double, 7> Darray2 = Ds[std::min(std::min(2*(bs_p-1), other2_idx), (other2_nidx-1)-other2_idx)];
                        if (nidx-idx < 2*bs_p-1) std::reverse(Barray.begin(), Barray.end());
                        if ((other1_nidx-1)-other1_idx < (2*(bs_p-1))) std::reverse(Darray1.begin(), Darray1.end());
                        if ((other2_nidx-1)-other2_idx < (2*(bs_p-1))) std::reverse(Darray2.begin(), Darray2.end());
                        double lumped_element = 0;
                        for (int l = -bs_p; l <= bs_p; ++l) {
                            for (int m = -(bs_p-1); m <= (bs_p-1); ++m) {
                                for (int q = -(bs_p-1); q <= (bs_p-1); ++q) {
                                    int jdx = idx + l;
                                    int other1_jdx = other1_idx + m;
                                    int other2_jdx = other2_idx + q;
                                    if (jdx >= 0 && jdx <= nidx && other1_jdx >=0 && other1_jdx < other1_nidx && other2_jdx >=0 && other2_jdx < other2_nidx) {
                                        double factor = Barray[l+3]*Darray1[m+3]*Darray2[q+3];
                                        lumped_element += factor;
                                    }
                                }
                            }
                        }
                        lumped_starflux_matrix.insert(tIdx, tIdx) = lumped_element;
                    }
                } else {
                    lumped_starflux_matrix.insert(tIdx, tIdx) = 1.0;
                }
            }
        });
        lumped_starflux_matrix.makeCompressed();
        d2transpose.makeCompressed();

        if (sim_scheme != Scheme::CO_FLIP || use_pressure_solver) {
            d2_matrix = d2transpose.transpose();
            invstarflux_matrix = lumped_starflux_matrix;
            invstarflux_matrix.diagonal().array() = 1. / invstarflux_matrix.diagonal().array();
            Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> pressure_laplacian_matrix = d2_matrix * invstarflux_matrix * d2transpose;
            pressure_laplacian_matrix.makeCompressed();
            amg_levelGen_double.generateLevelsGalerkinCoarsening(A_L_pressure_laplacian, R_L_pressure_laplacian, P_L_pressure_laplacian, wmax_pressure_laplacian, S_L_pressure_laplacian, total_level_pressure_laplacian, pressure_laplacian_matrix, ni, nj, nk, 1.0f);
        }
    }
    if (sim_scheme != Scheme::CO_FLIP || use_pressure_solver || do_delta_circulation) {
        Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> starflux_matrix(_nF, _nF);
        starflux_matrix.reserve(Eigen::VectorXi::Constant(_nF, sim_scheme == Scheme::CO_FLIP && !use_DEC_diagonal_hodge_star && is_matrix_small_enough ? (2*bs_p+1)*(2*bs_p-1)*(2*bs_p-1)/2 + 1 : 1));

        tbb::parallel_for(tbb::blocked_range<int>(0,_nF, TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
        {
            for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
                bool is_x_dir = tIdx < x_nF;
                bool is_y_dir = !is_x_dir && tIdx < (x_nF + y_nF);
                int comp_slice = is_x_dir ? (ni+1)*nj : (is_y_dir ? ni*(nj+1) : ni*nj);
                int comp_n = is_x_dir ? (ni+1) : ni;
                int comp_tIdx = is_x_dir ? tIdx : (is_y_dir ? (tIdx - x_nF) : (tIdx - (x_nF + y_nF)));
                int k = comp_tIdx/comp_slice;
                int j = (comp_tIdx%comp_slice)/comp_n;
                int i = comp_tIdx%comp_n;
                int idx = is_x_dir ? i : (is_y_dir ? j : k);
                int nidx = is_x_dir ? ni : (is_y_dir ? nj : nk);
                int other1_idx = is_x_dir ? j : (is_y_dir ? k : i);
                int other1_nidx = is_x_dir ? nj : (is_y_dir ? nk : ni);
                int other2_idx = is_x_dir ? k : (is_y_dir ? i : j);
                int other2_nidx = is_x_dir ? nk : (is_y_dir ? ni : nj);
                if (sim_scheme == Scheme::CO_FLIP) {
                    if (use_DEC_diagonal_hodge_star) {
                        double length_of_dualedge = bs_p == 3 ? ((idx == 0 || idx == nidx) ? 1./3. : ((idx == 1 || idx == (nidx-1)) ? 0.5 : ((idx == 2 || idx == (nidx-2)) ? 5./6. : 1.0))) : (bs_p == 2 ? ((idx == 0 || idx == nidx) ? 0.5 : ((idx == 1 || idx == (nidx-1)) ? 0.75 : 1.0)) : 1.0);
                        double length_of_edge1 = bs_p == 3 ? ((other1_idx == 0 || other1_idx == (other1_nidx-1)) ? 1./3. : ((other1_idx == 1 || other1_idx == (other1_nidx-2)) ? 2./3. : 1.0)) : (bs_p == 2 ? ((other1_idx == 0 || other1_idx == (other1_nidx-1)) ? 0.5 : 1.0) : 1.0);
                        double length_of_edge2 = bs_p == 3 ? ((other2_idx == 0 || other2_idx == (other2_nidx-1)) ? 1./3. : ((other2_idx == 1 || other2_idx == (other2_nidx-2)) ? 2./3. : 1.0)) : (bs_p == 2 ? ((other2_idx == 0 || other2_idx == (other2_nidx-1)) ? 0.5 : 1.0) : 1.0);
                        double area_of_primalface = length_of_edge1 * length_of_edge2;
                        double factor = length_of_dualedge / area_of_primalface;
                        starflux_matrix.insert(tIdx, tIdx) = factor;
                    } else {
                        std::vector<std::array<double, 7> > Bs = bs_p == 3 ? B3s : (bs_p == 2 ? B2s : B1s);
                        std::vector<std::array<double, 7> > Ds = bs_p == 3 ? D2s : (bs_p == 2 ? D1s : D0s);
                        std::array<double, 7> Barray = Bs[std::min(std::min(2*bs_p-1, idx), nidx-idx)];
                        std::array<double, 7> Darray1 = Ds[std::min(std::min(2*(bs_p-1), other1_idx), (other1_nidx-1)-other1_idx)];
                        std::array<double, 7> Darray2 = Ds[std::min(std::min(2*(bs_p-1), other2_idx), (other2_nidx-1)-other2_idx)];
                        if (nidx-idx < 2*bs_p-1) std::reverse(Barray.begin(), Barray.end());
                        if ((other1_nidx-1)-other1_idx < (2*(bs_p-1))) std::reverse(Darray1.begin(), Darray1.end());
                        if ((other2_nidx-1)-other2_idx < (2*(bs_p-1))) std::reverse(Darray2.begin(), Darray2.end());
                        for (int l = -bs_p; l <= bs_p; ++l) {
                            for (int m = -(bs_p-1); m <= (bs_p-1); ++m) {
                                for (int q = -(bs_p-1); q <= (bs_p-1); ++q) {
                                    int jdx = idx + l;
                                    int other1_jdx = other1_idx + m;
                                    int other2_jdx = other2_idx + q;
                                    if (jdx >= 0 && jdx <= nidx && other1_jdx >=0 && other1_jdx < other1_nidx && other2_jdx >=0 && other2_jdx < other2_nidx) {
                                        int tJdx = tIdx + l * (is_x_dir ? 1 : (is_y_dir ? ni : ni*nj)) + m * (is_x_dir ? (ni+1) : (is_y_dir ? ni*(nj+1) : 1)) + q * (is_x_dir ? (ni+1)*nj : (is_y_dir ? 1 : ni));
                                        double factor = Barray[l+3]*Darray1[m+3]*Darray2[q+3];
                                        if ((is_matrix_small_enough && tJdx <= tIdx) || (!is_matrix_small_enough && tJdx == tIdx)) {
                                            starflux_matrix.insert(tIdx, tJdx) = factor;
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else {
                    starflux_matrix.insert(tIdx, tIdx) = 1.0;
                }
            }
        });
        starflux_matrix.makeCompressed();
        starflux_precond.compute(starflux_matrix);
        std::cout << "did starflux_precond succeed? " << starflux_precond.info() << " ; 0<-success, 1<-not SPD" << std::endl;
    }
    std::cout << "done putting together starflux, wedge, and d2 matrices." << std::endl;

    {
        Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> d0_matrix(_nE, _nV);
        d0_matrix.reserve(Eigen::VectorXi::Constant(_nE, 2));
        d1transpose_matrix.resize(_nE, _nF);
        d1transpose_matrix.reserve(Eigen::VectorXi::Constant(_nE, 4));
        Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> almostIdentityMatrix(_nE,_nE);
        almostIdentityMatrix.reserve(Eigen::VectorXi::Constant(_nE,1));
        tbb::parallel_for(tbb::blocked_range<int>(0,_nE, TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
        {
            for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
                bool is_x_dir = tIdx < x_nE;
                bool is_y_dir = !is_x_dir && tIdx < (x_nE + y_nE);
                bool is_z_dir = !is_x_dir && !is_y_dir;
                int comp_slice = is_x_dir ? ni*(nj+1) : (is_y_dir ? (ni+1)*nj : (ni+1)*(nj+1));
                int comp_n = is_x_dir ? ni : (ni+1);
                int comp_tIdx = is_x_dir ? tIdx : (is_y_dir ? (tIdx - x_nE) : (tIdx - (x_nE + y_nE)));
                int k = comp_tIdx/comp_slice;
                int j = (comp_tIdx%comp_slice)/comp_n;
                int i = comp_tIdx%comp_n;
                if (edgeInFluid(is_x_dir ? 0 : (is_y_dir ? 1 : 2), i, j, k)) {
                    int tIdx_oneForm1 = (is_x_dir ? x_nF : (is_y_dir ? (x_nF + y_nF) : 0)) + 
                        k*((is_z_dir ? (ni+1) : ni)*(is_x_dir ? (nj+1) : nj))+j*(is_z_dir ? (ni+1) : ni)+i;
                    int tIdx_oneForm2 = (is_x_dir ? (x_nF + y_nF) : (is_y_dir ? 0 : x_nF)) + 
                        k*((is_y_dir ? (ni+1) : ni)*(is_z_dir ? (nj+1) : nj))+j*(is_y_dir ? (ni+1) : ni)+i;
                    d1transpose_matrix.insert(tIdx, tIdx_oneForm1-(is_x_dir ? ni*(nj+1) : (is_y_dir ? 1 : (ni+1)))) = 1.0;
                    d1transpose_matrix.insert(tIdx, tIdx_oneForm1) = -1.0;
                    d1transpose_matrix.insert(tIdx, tIdx_oneForm2-(is_x_dir ? ni : (is_y_dir ? (ni+1)*nj : 1))) = -1.0;
                    d1transpose_matrix.insert(tIdx, tIdx_oneForm2) = 1.0;

                    int tIdx_zeroForm = k*(ni+1)*(nj+1) + j*(ni+1) + i;
                    if (vertexInFluid(i,j,k)) {
                        d0_matrix.insert(tIdx, tIdx_zeroForm) = -1.0;
                    }
                    if (vertexInFluid(i + (is_x_dir ? 1 : 0),j + (is_y_dir ? 1 : 0),k + (is_z_dir ? 1 : 0))) {
                        d0_matrix.insert(tIdx, tIdx_zeroForm + (is_x_dir ? 1 : (is_y_dir ? (ni+1) : (ni+1)*(nj+1)))) = 1.0;
                    }
                } else {
                    almostIdentityMatrix.insert(tIdx, tIdx) = 1.0;
                }
            }
        });
        d1transpose_matrix.makeCompressed();
        d0_matrix.makeCompressed();
        almostIdentityMatrix.makeCompressed();
        gauge_laplacian_matrix = 0.1 * d0_matrix * d0_matrix.transpose();
        gauge_laplacian_matrix = gauge_laplacian_matrix + almostIdentityMatrix;
        gauge_laplacian_matrix.makeCompressed();
        d0_matrix.resize(0,0);
        d0_matrix.data().squeeze();
        d1_matrix = d1transpose_matrix.transpose();

        if (!(is_matrix_small_enough && bs_p != 1) && sim_scheme == Scheme::CO_FLIP && !use_pressure_solver) {
            Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> streamform_laplacian_matrix = d1transpose_matrix * lumped_starflux_matrix * d1_matrix;
            streamform_laplacian_matrix = streamform_laplacian_matrix + gauge_laplacian_matrix;
            streamform_laplacian_matrix.makeCompressed();
            amg_levelGen_double.generateLevelsGalerkinCoarsening(A_L_streamform_laplacian, R_L_streamform_laplacian, P_L_streamform_laplacian, wmax_streamform_laplacian, S_L_streamform_laplacian, total_level_streamform_laplacian, streamform_laplacian_matrix, ni, nj, nk, 1, GMG::EDGES, 1.0f);
        }

        if ((is_matrix_small_enough && bs_p != 1) && sim_scheme == Scheme::CO_FLIP && !use_pressure_solver) {
            Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> streamform_laplacian_matrix;
            {
                std::cout << "computing streamform-vorticity laplacian..." << std::endl;
                int bs_p_saved = bs_p;
                bs_p = 2;
                Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> starflux_matrix(_nF, _nF);
                starflux_matrix.reserve(Eigen::VectorXi::Constant(_nF, sim_scheme == Scheme::CO_FLIP && !use_DEC_diagonal_hodge_star ? (2*bs_p+1)*(2*bs_p-1)*(2*bs_p-1) : 1));

                tbb::parallel_for(tbb::blocked_range<int>(0,_nF, TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
                {
                    for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
                        bool is_x_dir = tIdx < x_nF;
                        bool is_y_dir = !is_x_dir && tIdx < (x_nF + y_nF);
                        int comp_slice = is_x_dir ? (ni+1)*nj : (is_y_dir ? ni*(nj+1) : ni*nj);
                        int comp_n = is_x_dir ? (ni+1) : ni;
                        int comp_tIdx = is_x_dir ? tIdx : (is_y_dir ? (tIdx - x_nF) : (tIdx - (x_nF + y_nF)));
                        int k = comp_tIdx/comp_slice;
                        int j = (comp_tIdx%comp_slice)/comp_n;
                        int i = comp_tIdx%comp_n;
                        int idx = is_x_dir ? i : (is_y_dir ? j : k);
                        int nidx = is_x_dir ? ni : (is_y_dir ? nj : nk);
                        int other1_idx = is_x_dir ? j : (is_y_dir ? k : i);
                        int other1_nidx = is_x_dir ? nj : (is_y_dir ? nk : ni);
                        int other2_idx = is_x_dir ? k : (is_y_dir ? i : j);
                        int other2_nidx = is_x_dir ? nk : (is_y_dir ? ni : nj);
                        if (sim_scheme == Scheme::CO_FLIP) {
                            if (use_DEC_diagonal_hodge_star) {
                                double length_of_dualedge = bs_p == 3 ? ((idx == 0 || idx == nidx) ? 1./3. : ((idx == 1 || idx == (nidx-1)) ? 0.5 : ((idx == 2 || idx == (nidx-2)) ? 5./6. : 1.0))) : (bs_p == 2 ? ((idx == 0 || idx == nidx) ? 0.5 : ((idx == 1 || idx == (nidx-1)) ? 0.75 : 1.0)) : 1.0);
                                double length_of_edge1 = bs_p == 3 ? ((other1_idx == 0 || other1_idx == (other1_nidx-1)) ? 1./3. : ((other1_idx == 1 || other1_idx == (other1_nidx-2)) ? 2./3. : 1.0)) : (bs_p == 2 ? ((other1_idx == 0 || other1_idx == (other1_nidx-1)) ? 0.5 : 1.0) : 1.0);
                                double length_of_edge2 = bs_p == 3 ? ((other2_idx == 0 || other2_idx == (other2_nidx-1)) ? 1./3. : ((other2_idx == 1 || other2_idx == (other2_nidx-2)) ? 2./3. : 1.0)) : (bs_p == 2 ? ((other2_idx == 0 || other2_idx == (other2_nidx-1)) ? 0.5 : 1.0) : 1.0);
                                double area_of_primalface = length_of_edge1 * length_of_edge2;
                                double factor = length_of_dualedge / area_of_primalface;
                                starflux_matrix.insert(tIdx, tIdx) = factor;
                            } else {
                                std::vector<std::array<double, 7> > Bs = bs_p == 3 ? B3s : (bs_p == 2 ? B2s : B1s);
                                std::vector<std::array<double, 7> > Ds = bs_p == 3 ? D2s : (bs_p == 2 ? D1s : D0s);
                                std::array<double, 7> Barray = Bs[std::min(std::min(2*bs_p-1, idx), nidx-idx)];
                                std::array<double, 7> Darray1 = Ds[std::min(std::min(2*(bs_p-1), other1_idx), (other1_nidx-1)-other1_idx)];
                                std::array<double, 7> Darray2 = Ds[std::min(std::min(2*(bs_p-1), other2_idx), (other2_nidx-1)-other2_idx)];
                                if (nidx-idx < 2*bs_p-1) std::reverse(Barray.begin(), Barray.end());
                                if ((other1_nidx-1)-other1_idx < (2*(bs_p-1))) std::reverse(Darray1.begin(), Darray1.end());
                                if ((other2_nidx-1)-other2_idx < (2*(bs_p-1))) std::reverse(Darray2.begin(), Darray2.end());
                                for (int l = -bs_p; l <= bs_p; ++l) {
                                    for (int m = -(bs_p-1); m <= (bs_p-1); ++m) {
                                        for (int q = -(bs_p-1); q <= (bs_p-1); ++q) {
                                            int jdx = idx + l;
                                            int other1_jdx = other1_idx + m;
                                            int other2_jdx = other2_idx + q;
                                            if (jdx >= 0 && jdx <= nidx && other1_jdx >=0 && other1_jdx < other1_nidx && other2_jdx >=0 && other2_jdx < other2_nidx) {
                                                int tJdx = tIdx + l * (is_x_dir ? 1 : (is_y_dir ? ni : ni*nj)) + m * (is_x_dir ? (ni+1) : (is_y_dir ? ni*(nj+1) : 1)) + q * (is_x_dir ? (ni+1)*nj : (is_y_dir ? 1 : ni));
                                                double factor = Barray[l+3]*Darray1[m+3]*Darray2[q+3];
                                                starflux_matrix.insert(tIdx, tJdx) = factor;
                                            }
                                        }
                                    }
                                }
                            }
                        } else {
                            starflux_matrix.insert(tIdx, tIdx) = 1.0;
                        }
                    }
                });
                starflux_matrix.makeCompressed();
                bs_p = bs_p_saved;
                streamform_laplacian_matrix = d1transpose_matrix * starflux_matrix * d1_matrix;
                streamform_laplacian_matrix = streamform_laplacian_matrix + gauge_laplacian_matrix;
                streamform_laplacian_matrix.makeCompressed();
                std::cout << "done with streamform-vorticity laplacian..." << std::endl;
            }
            //----------
            std::cout << "computing streamform-vorticity laplacian GMG preconditioner..." << std::endl;
            amg_levelGen_double.generateLevelsGalerkinCoarsening(A_L_streamform_laplacian, R_L_streamform_laplacian, P_L_streamform_laplacian, wmax_streamform_laplacian, S_L_streamform_laplacian, total_level_streamform_laplacian, streamform_laplacian_matrix, ni, nj, nk, 1, GMG::EDGES, 1.0f);
            std::cout << "done with streamform-vorticity laplacian GMG preconditioner..." << std::endl;
        }
    }
    {
        Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> starvort_matrix(_nE, _nE);
        starvort_matrix.reserve(Eigen::VectorXi::Constant(_nE, sim_scheme == Scheme::CO_FLIP && !use_DEC_diagonal_hodge_star && is_matrix_small_enough ? (2*bs_p+1)*(2*bs_p+1)*(2*bs_p-1)/2 + 1 : 1));
        Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> lumped_invstarvort_matrix(_nE, _nE);
        lumped_invstarvort_matrix.reserve(Eigen::VectorXi::Constant(_nE, 1));
        tbb::parallel_for(tbb::blocked_range<int>(0,_nE, TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
        {
            for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
                bool is_x_dir = tIdx < x_nE;
                bool is_y_dir = !is_x_dir && tIdx < (x_nE + y_nE);
                int comp_slice = is_x_dir ? ni*(nj+1) : (is_y_dir ? (ni+1)*nj : (ni+1)*(nj+1));
                int comp_n = is_x_dir ? ni : (ni+1);
                int comp_tIdx = is_x_dir ? tIdx : (is_y_dir ? (tIdx - x_nE) : (tIdx - (x_nE + y_nE)));
                int k = comp_tIdx/comp_slice;
                int j = (comp_tIdx%comp_slice)/comp_n;
                int i = comp_tIdx%comp_n;
                int idx = is_x_dir ? i : (is_y_dir ? j : k);
                int other1_idx = is_x_dir ? j : (is_y_dir ? k : i);
                int other2_idx = is_x_dir ? k : (is_y_dir ? i : j);
                int nidx = is_x_dir ? ni : (is_y_dir ? nj : nk);
                int other1_nidx = is_x_dir ? nj : (is_y_dir ? nk : ni);
                int other2_nidx = is_x_dir ? nk : (is_y_dir ? ni : nj);
                if (sim_scheme == Scheme::CO_FLIP) {
                    if (use_DEC_diagonal_hodge_star) {
                        double length_of_primaledge = bs_p == 3 ? ((idx == 0 || idx == (nidx-1)) ? 1./3. : ((idx == 1 || idx == (nidx-2)) ? 2./3. : 1.0)) : (bs_p == 2 ? ((idx == 0 || idx == (nidx-1)) ? 0.5 : 1.0) : 1.0);
                        double length_of_dualedge1 = bs_p == 3 ? ((other1_idx == 0 || other1_idx == other1_nidx) ? 1./3. : ((other1_idx == 1 || other1_idx == (other1_nidx-1)) ? 0.5 : ((other1_idx == 2 || other1_idx == (other1_nidx-2)) ? 5./6. : 1.0))) : (bs_p == 2 ? ((other1_idx == 0 || other1_idx == other1_nidx) ? 0.5 : ((other1_idx == 1 || other1_idx == (other1_nidx-1)) ? 0.75 : 1.0)) : 1.0);
                        double length_of_dualedge2 = bs_p == 3 ? ((other2_idx == 0 || other2_idx == other2_nidx) ? 1./3. : ((other2_idx == 1 || other2_idx == (other2_nidx-1)) ? 0.5 : ((other2_idx == 2 || other2_idx == (other2_nidx-2)) ? 5./6. : 1.0))) : (bs_p == 2 ? ((other2_idx == 0 || other2_idx == other2_nidx) ? 0.5 : ((other2_idx == 1 || other2_idx == (other2_nidx-1)) ? 0.75 : 1.0)) : 1.0);
                        double area_of_dualface = length_of_dualedge1 * length_of_dualedge2;
                        double factor = area_of_dualface / length_of_primaledge;
                        starvort_matrix.insert(tIdx, tIdx) = factor;
                        lumped_invstarvort_matrix.insert(tIdx, tIdx) = factor;
                    } else {
                        std::vector<std::array<double, 7> > Bs = bs_p == 3 ? B3s : (bs_p == 2 ? B2s : B1s);
                        std::vector<std::array<double, 7> > Ds = bs_p == 3 ? D2s : (bs_p == 2 ? D1s : D0s);
                        std::array<double, 7> Darray = Ds[std::min(std::min(2*(bs_p-1), idx), (nidx-1)-idx)];
                        std::array<double, 7> Barray1 = Bs[std::min(std::min(2*bs_p-1, other1_idx), other1_nidx-other1_idx)];
                        std::array<double, 7> Barray2 = Bs[std::min(std::min(2*bs_p-1, other2_idx), other2_nidx-other2_idx)];
                        if ((nidx-1)-idx < 2*(bs_p-1)) std::reverse(Darray.begin(), Darray.end());
                        if (other1_nidx-other1_idx < 2*bs_p-1) std::reverse(Barray1.begin(), Barray1.end());
                        if (other2_nidx-other2_idx < 2*bs_p-1) std::reverse(Barray2.begin(), Barray2.end());
                        double lumped_factor = 0;
                        for (int l = -(bs_p-1); l <= (bs_p-1); ++l) {
                            for (int m = -bs_p; m <= bs_p; ++m) {
                                for (int q = -bs_p; q <= bs_p; ++q) {
                                    int jdx = idx + l;
                                    int other1_jdx = other1_idx + m;
                                    int other2_jdx = other2_idx + q;
                                    if (jdx >= 0 && jdx < nidx && 
                                        other1_jdx >=0 && other1_jdx <= other1_nidx && 
                                        other2_jdx >=0 && other2_jdx <= other2_nidx) {
                                        int tJdx = tIdx + l * (is_x_dir ? 1 : (is_y_dir ? (ni+1) : (ni+1)*(nj+1))) + m * (is_x_dir ? ni : (is_y_dir ? (ni+1)*nj : 1)) + q * (is_x_dir ? ni*(nj+1) : (is_y_dir ? 1 : (ni+1)));
                                        double factor = Darray[l+3]*Barray1[m+3]*Barray2[q+3];
                                        if ((is_matrix_small_enough && tJdx <= tIdx) || (!is_matrix_small_enough && tJdx == tIdx)) {
                                            starvort_matrix.insert(tIdx, tJdx) = factor;
                                            lumped_factor += factor;
                                        }
                                    }
                                }
                            }
                        }
                        lumped_invstarvort_matrix.insert(tIdx, tIdx) = lumped_factor;
                    }
                } else {
                    starvort_matrix.insert(tIdx, tIdx) = 1.0;
                    lumped_invstarvort_matrix.insert(tIdx, tIdx) = 1.0;
                }
            }
        });
        starvort_matrix.makeCompressed();
        lumped_invstarvort_matrix.makeCompressed();
        starvort_precond.compute(starvort_matrix);
        std::cout << "did starvort_precond succeed? " << starvort_precond.info() << " ; 0<-success, 1<-not SPD" << std::endl;
        if (viscosity != 0.) {
            Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> identityMatrixEdges(_nF,_nF);
            identityMatrixEdges.reserve(Eigen::VectorXi::Constant(_nF,1));
            tbb::parallel_for(tbb::blocked_range<int>(0,_nF, TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
            {
                for (int i = r.begin(); i < r.end(); ++i) {
                    identityMatrixEdges.insert(i,i) = 1;
                }
            });
            identityMatrixEdges.makeCompressed();
            Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> LHS_mat = (viscosity * dt / (_h*_h)) * (d1_matrix * lumped_invstarvort_matrix * d1transpose_matrix * lumped_starflux_matrix);
            LHS_mat += identityMatrixEdges;
            amg_levelGen_double.generateLevelsGalerkinCoarsening(A_L_viscosity_laplacian, R_L_viscosity_laplacian, P_L_viscosity_laplacian, wmax_viscosity_laplacian, S_L_viscosity_laplacian, total_level_viscosity_laplacian, LHS_mat, ni, nj, nk, 1, GMG::FACES, 1.0f);
        }
    }
    
}

void COFLIPSolver::takeDualwrtStar(Buffer3D<MyReal> &u, Buffer3D<MyReal> &v, Buffer3D<MyReal> &w, bool update_uv, bool flux2circulation_or_circulation2flux) {
    int ni = _nx;
    int nj = _ny;
    int nk = _nz;
    int x_nF = (ni+1)*nj*nk;
    int y_nF = ni*(nj+1)*nk;
    tbb::parallel_for(tbb::blocked_range<int>(0,_nF, TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
    {
        for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
            bool is_x_dir = tIdx < x_nF;
            bool is_y_dir = !is_x_dir && tIdx < (x_nF + y_nF);
            int comp_slice = is_x_dir ? (ni+1)*nj : (is_y_dir ? ni*(nj+1) : ni*nj);
            int comp_n = is_x_dir ? (ni+1) : ni;
            int comp_tIdx = is_x_dir ? tIdx : (is_y_dir ? (tIdx - x_nF) : (tIdx - (x_nF + y_nF)));
            int k = comp_tIdx/comp_slice;
            int j = (comp_tIdx%comp_slice)/comp_n;
            int i = comp_tIdx%comp_n;
            MyReal value = is_x_dir ? u(i,j,k) : (is_y_dir ? v(i,j,k) : w(i,j,k));
            if (flux2circulation_or_circulation2flux == false)
                fluxes[tIdx] = value;
            else
                circulations[tIdx] = value;
        }
    });
    if (flux2circulation_or_circulation2flux == false) {
        Eigen::VectorXd circulations_double = circulations.cast<double>();
        Eigen::VectorXd fluxes_double = fluxes.cast<double>();
        multiplyWithStarFlux(circulations_double, fluxes_double);
        circulations = circulations_double.cast<MyReal>();
    } else {
        int iterations;
        double tolerance;
        auto multiply_func = [&](Eigen::VectorXd& output, const Eigen::VectorXd& input) {
            multiplyWithStarFlux(output, input);
        };
        Eigen::VectorXd circulations_double = circulations.cast<double>();
        Eigen::VectorXd fluxes_double = fluxes.cast<double>();
        bool success = AMGPCGSolve<double>(multiply_func, circulations_double, fluxes_double, starflux_precond, (double)TOLERANCE, MAX_ITERATIONS, tolerance, iterations);
        fluxes = fluxes_double.cast<MyReal>();
        if (!success) {
            printf("WARNING: inverse star-flux solve failed!************************************************\n");
        }
        std::cout << "#iteration:      " << iterations << std::endl;
        std::cout << "estimated error: " << tolerance << std::endl;
    }

    if (update_uv) {
        tbb::parallel_for(tbb::blocked_range<int>(0,_nF, TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
        {
            for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
                bool is_x_dir = tIdx < x_nF;
                bool is_y_dir = !is_x_dir && tIdx < (x_nF + y_nF);
                int comp_slice = is_x_dir ? (ni+1)*nj : (is_y_dir ? ni*(nj+1) : ni*nj);
                int comp_n = is_x_dir ? (ni+1) : ni;
                int comp_tIdx = is_x_dir ? tIdx : (is_y_dir ? (tIdx - x_nF) : (tIdx - (x_nF + y_nF)));
                int k = comp_tIdx/comp_slice;
                int j = (comp_tIdx%comp_slice)/comp_n;
                int i = comp_tIdx%comp_n;
                if (flux2circulation_or_circulation2flux == false) {
                    if (is_x_dir) {
                        _un(i,j,k) = circulations[tIdx];
                    } else if (is_y_dir) {
                        _vn(i,j,k) = circulations[tIdx];
                    } else {
                        _wn(i,j,k) = circulations[tIdx];
                    }
                } else {
                    if (is_x_dir) {
                        _un(i,j,k) = fluxes[tIdx];
                    } else if (is_y_dir) {
                        _vn(i,j,k) = fluxes[tIdx];
                    } else {
                        _wn(i,j,k) = fluxes[tIdx];
                    }
                }
            }
        });
    }
}

void COFLIPSolver::projectionWithVort()
{
    std::cout << BLUE << "Projection with vort started..." << RESET << std::endl;
    int ni = _nx;
    int nj = _ny;
    int nk = _nz;

    //write boundary velocity;
    int compute_num = _nC;
    int slice = ni*nj;
    
    takeDualwrtStar(_un, _vn, _wn, true, false);

    calculateCurl();
    int iterations;
    double tolerance;
    auto multiply_func = [&](Eigen::VectorXd& output, const Eigen::VectorXd& input) {
        Eigen::VectorXd input_fluxes = d1_matrix * input;
        Eigen::VectorXd output_circualtions(_nF);
        multiplyWithStarFlux(output_circualtions, input_fluxes);
        output = (d1transpose_matrix * output_circualtions) + (gauge_laplacian_matrix * input);
    };
    Eigen::VectorXd rhs = d1transpose_matrix * circulations.cast<double>();
    Eigen::VectorXd streamforms_double = streamforms.cast<double>();
    bool success = AMGPCGSolve<double>(multiply_func, rhs, streamforms_double,
        A_L_streamform_laplacian, R_L_streamform_laplacian, P_L_streamform_laplacian, wmax_streamform_laplacian, S_L_streamform_laplacian, total_level_streamform_laplacian, (double)TOLERANCE, MAX_ITERATIONS, tolerance, iterations, _nx, _ny, _nz, GMG::EDGES);
    streamforms = streamforms_double.cast<MyReal>();
    std::cout << "#iteration:      " << iterations << std::endl;
    std::cout << "estimated error: " << tolerance << std::endl;
    if (!success) {
        printf("WARNING: Streamform-vorticity solve failed!************************************************\n");
    }
    
    int x_nE = ni*(nj+1)*(nk+1);
    int y_nE = (ni+1)*nj*(nk+1);
    tbb::parallel_for(tbb::blocked_range<int>(0,_nE, TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
    {
        for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
            bool is_x_dir = tIdx < x_nE;
            bool is_y_dir = !is_x_dir && tIdx < (x_nE + y_nE);
            int comp_slice = is_x_dir ? ni*(nj+1) : (is_y_dir ? (ni+1)*nj : (ni+1)*(nj+1));
            int comp_n = is_x_dir ? ni : (ni+1);
            int comp_tIdx = is_x_dir ? tIdx : (is_y_dir ? (tIdx - x_nE) : (tIdx - (x_nE + y_nE)));
            int k = comp_tIdx/comp_slice;
            int j = (comp_tIdx%comp_slice)/comp_n;
            int i = comp_tIdx%comp_n;
            if(!edgeInFluid(is_x_dir ? 0 : (is_y_dir ? 1 : 2), i, j, k)) {
                streamforms(tIdx) = 0.;
            }
        }
    });

    fluxes = (d1_matrix * streamforms.cast<double>()).cast<MyReal>();
    int x_nF = (ni+1)*nj*nk;
    int y_nF = ni*(nj+1)*nk;
    tbb::parallel_for(tbb::blocked_range<int>(0,_nF, TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
    {
        for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
            bool is_x_dir = tIdx < x_nF;
            bool is_y_dir = !is_x_dir && tIdx < (x_nF + y_nF);
            int comp_slice = is_x_dir ? (ni+1)*nj : (is_y_dir ? ni*(nj+1) : ni*nj);
            int comp_n = is_x_dir ? (ni+1) : ni;
            int comp_tIdx = is_x_dir ? tIdx : (is_y_dir ? (tIdx - x_nF) : (tIdx - (x_nF + y_nF)));
            int k = comp_tIdx/comp_slice;
            int j = (comp_tIdx%comp_slice)/comp_n;
            int i = comp_tIdx%comp_n;
            if (is_x_dir) {
                _un(i,j,k) = fluxes[tIdx];
            } else if (is_y_dir) {
                _vn(i,j,k) = fluxes[tIdx];
            } else {
                _wn(i,j,k) = fluxes[tIdx];
            }
        }
    });

    //write boundary velocity
    tbb::parallel_for(0,compute_num,1,[&](int thread_idx)
    {
        uint k = thread_idx/slice;
        uint j = (thread_idx%slice)/ni;
        uint i = thread_idx%ni;
        if (_b_desc(i,j,k)==3)//solid
        {
            _un(i,j,k) = _usolid(i,j,k);
            _un(i+1,j,k) = _usolid(i+1,j,k);
            _vn(i,j,k) = _vsolid(i,j,k);
            _vn(i,j+1,k) = _vsolid(i,j+1,k);
            _wn(i,j,k) = _wsolid(i,j,k);
            _wn(i,j,k+1) = _wsolid(i,j,k+1);
        }

        if(i==0)
        {
            _un(i,j,k) = 0;
        }
        if(j==0)
        {
            _vn(i,j,k) = 0;
        }
        if(k==0)
        {
            _wn(i,j,k) = 0;
        }
        if(i==ni-1)
        {
            _un(i+1,j,k) = 0;
        }
        if(j==nj-1)
        {
            _vn(i,j+1,k) = 0;
        }
        if(k==nk-1)
        {
            _wn(i,j,k+1) = 0;
        }
    });

    if (set_velocity_inflow)
        setVelocityFromEmitter(true);

    takeDualwrtStar(_un, _vn, _wn, false, false);
    std::cout  << GREEN << "fluxes.norm = " << fluxes.norm() << ", circulations.norm = " << circulations.norm()  << ", vorts.norm = " << vorts.norm() << RESET << std::endl;
    std::cout << BLUE << "Projection with vort done." << RESET << std::endl;
}

void COFLIPSolver::projection()
{
    int ni = _nx;
    int nj = _ny;
    int nk = _nz;

    //write boundary velocity;
    int compute_num = _nC;
    int slice = ni*nj;
    tbb::parallel_for(0,compute_num,1,[&](int thread_idx)
    {
        int k = thread_idx/slice;
        int j = (thread_idx%slice)/ni;
        int i = thread_idx%ni;
        if (_b_desc(i,j,k)==3)//solid
        {
            _un(i,j,k) = _usolid(i,j,k);
            _un(i+1,j,k) = _usolid(i+1,j,k);
            _vn(i,j,k) = _vsolid(i,j,k);
            _vn(i,j+1,k) = _vsolid(i,j+1,k);
            _wn(i,j,k) = _wsolid(i,j,k);
            _wn(i,j,k+1) = _wsolid(i,j,k+1);
        }

        if(i==0)
        {
            _un(i,j,k) = 0;
        }
        if(j==0)
        {
            _vn(i,j,k) = 0;
        }
        if(k==0)
        {
            _wn(i,j,k) = 0;
        }
        if(i==ni-1)
        {
            _un(i+1,j,k) = 0;
        }
        if(j==nj-1)
        {
            _vn(i,j+1,k) = 0;
        }
        if(k==nk-1)
        {
            _wn(i,j,k+1) = 0;
        }
    });

    if (set_velocity_inflow)
        setVelocityFromEmitter(true);

    for (int count = 0; count < pp_repeat_count; count++)
    {
        int x_nF = (ni+1)*nj*nk;
        int y_nF = ni*(nj+1)*nk;
        tbb::parallel_for(tbb::blocked_range<int>(0,_nF, TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
        {
            for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
                bool is_x_dir = tIdx < x_nF;
                bool is_y_dir = !is_x_dir && tIdx < (x_nF + y_nF);
                int comp_slice = is_x_dir ? (ni+1)*nj : (is_y_dir ? ni*(nj+1) : ni*nj);
                int comp_n = is_x_dir ? (ni+1) : ni;
                int comp_tIdx = is_x_dir ? tIdx : (is_y_dir ? (tIdx - x_nF) : (tIdx - (x_nF + y_nF)));
                int k = comp_tIdx/comp_slice;
                int j = (comp_tIdx%comp_slice)/comp_n;
                int i = comp_tIdx%comp_n;
                fluxes[tIdx] = is_x_dir ? _un(i,j,k) : (is_y_dir ? _vn(i,j,k) : _wn(i,j,k));
            }
        });

        if (!is_fixed_domain)
        {
            Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> d2transpose(_nF, _nC);
            d2transpose.reserve(Eigen::VectorXi::Constant(_nF, 2));
            tbb::parallel_for(tbb::blocked_range<int>(0,_nF, TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
            {
                for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
                    bool is_x_dir = tIdx < x_nF;
                    bool is_y_dir = !is_x_dir && tIdx < (x_nF + y_nF);
                    bool is_z_dir = !is_x_dir && !is_y_dir;
                    int comp_slice = is_x_dir ? (ni+1)*nj : (is_y_dir ? ni*(nj+1) : ni*nj);
                    int comp_n = is_x_dir ? (ni+1) : ni;
                    int comp_tIdx = is_x_dir ? tIdx : (is_y_dir ? (tIdx - x_nF) : (tIdx - (x_nF + y_nF)));
                    int k = comp_tIdx/comp_slice;
                    int j = (comp_tIdx%comp_slice)/comp_n;
                    int i = comp_tIdx%comp_n;
                    int tIdx_threeForm = k*(ni*nj)+j*ni+i;
                    if (is_x_dir && i == 0) {
                        if (_b_desc(i,j,k)==1) d2transpose.insert(tIdx, tIdx_threeForm) = 1.0;
                    } else if (is_x_dir && i == ni) {
                        if (_b_desc(i-1,j,k)==1) d2transpose.insert(tIdx, tIdx_threeForm-1) = -1.0;
                    } else if (is_y_dir && j == 0) {
                        if (_b_desc(i,j,k)==1) d2transpose.insert(tIdx, tIdx_threeForm) = 1.0;
                    } else if (is_y_dir && j == nj) {
                        if (_b_desc(i,j-1,k)==1) d2transpose.insert(tIdx, tIdx_threeForm-ni) = -1.0;
                    } else if (is_z_dir && k == 0) {
                        if (_b_desc(i,j,k)==1) d2transpose.insert(tIdx, tIdx_threeForm) = 1.0;
                    } else if (is_z_dir && k == nk) {
                        if (_b_desc(i,j,k-1)==1) d2transpose.insert(tIdx, tIdx_threeForm-(ni*nj)) = -1.0;
                    } else if ((is_x_dir && _b_desc(i,j,k)==0 && _b_desc(i-1,j,k)==0) ||
                            (is_y_dir && _b_desc(i,j,k)==0 && _b_desc(i,j-1,k)==0) ||
                            (is_z_dir && _b_desc(i,j,k)==0 && _b_desc(i,j,k-1)==0)) {
                        d2transpose.insert(tIdx, tIdx_threeForm - (is_x_dir ? 1 : (is_y_dir ? ni : (ni*nj)))) = -1.0;
                        d2transpose.insert(tIdx, tIdx_threeForm) = 1.0;
                    }
                }
            });
            d2transpose.makeCompressed();
            d2_matrix = d2transpose.transpose();
            Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> pressure_laplacian_matrix = d2_matrix * invstarflux_matrix * d2transpose;
            pressure_laplacian_matrix.makeCompressed();
            amg_levelGen_double.generateLevelsGalerkinCoarsening(A_L_pressure_laplacian, R_L_pressure_laplacian, P_L_pressure_laplacian, wmax_pressure_laplacian, S_L_pressure_laplacian, total_level_pressure_laplacian, pressure_laplacian_matrix, ni, nj, nk, 1.0f);
        }
        double tolerance;
        int iterations;
        Eigen::VectorXd rhs_double = d2_matrix * fluxes.cast<double>();
        std::cout << GREEN<< "fluxes.norm = " << fluxes.norm() << ", circulations.norm = " << circulations.norm()  << ", rhs.norm = " << rhs_double.norm() << ", rhs.mean = " << rhs_double.mean() << RESET << std::endl;
        Eigen::VectorXd pressure_double(_nC);
        pressure_double.setZero();
        bool success = AMGPCGSolve(A_L_pressure_laplacian[0], rhs_double, pressure_double,
            A_L_pressure_laplacian, R_L_pressure_laplacian, P_L_pressure_laplacian, wmax_pressure_laplacian, S_L_pressure_laplacian, total_level_pressure_laplacian, TOLERANCE, MAX_ITERATIONS, tolerance, iterations, _nx, _ny, _nz);
        std::cout << "#iteration:      " << iterations << std::endl;
        std::cout << "estimated error: " << tolerance << std::endl;
        if (!success) {
            printf("WARNING: Pressure solve failed!************************************************\n");
        }
        std::cout << "pressure.norm = " << pressure_double.norm() << std::endl;

        Eigen::VectorXd delta_circulations = Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t>(d2_matrix.transpose()) * pressure_double;
        auto multiply_func = [&](Eigen::VectorXd& output, const Eigen::VectorXd& input) {
            multiplyWithStarFlux(output, input);
        };
        Eigen::VectorXd delta_fluxes(_nF);
        delta_fluxes.setZero();
        success = AMGPCGSolve<double>(multiply_func, delta_circulations, delta_fluxes, starflux_precond, (double)TOLERANCE, MAX_ITERATIONS, tolerance, iterations);
        if (!success) {
            printf("WARNING: inverse star-flux solve failed!************************************************\n");
        }
        std::cout << "#iteration:      " << iterations << std::endl;
        std::cout << "estimated error: " << tolerance << std::endl;
        
        tbb::parallel_for(tbb::blocked_range<int>(0,_nF, TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
        {
            for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
                bool is_x_dir = tIdx < x_nF;
                bool is_y_dir = !is_x_dir && tIdx < (x_nF + y_nF);
                int comp_slice = is_x_dir ? (ni+1)*nj : (is_y_dir ? ni*(nj+1) : ni*nj);
                int comp_n = is_x_dir ? (ni+1) : ni;
                int comp_tIdx = is_x_dir ? tIdx : (is_y_dir ? (tIdx - x_nF) : (tIdx - (x_nF + y_nF)));
                int k = comp_tIdx/comp_slice;
                int j = (comp_tIdx%comp_slice)/comp_n;
                int i = comp_tIdx%comp_n;
                if (is_x_dir) {
                    _un(i,j,k) -= delta_fluxes[tIdx];
                } else if (is_y_dir) {
                    _vn(i,j,k) -= delta_fluxes[tIdx];
                } else {
                    _wn(i,j,k) -= delta_fluxes[tIdx];
                }
            }
        });

        //write boundary velocity
        compute_num = _nC;
        slice = ni*nj;
        tbb::parallel_for(0,compute_num,1,[&](int thread_idx)
        {
            uint k = thread_idx/slice;
            uint j = (thread_idx%slice)/ni;
            uint i = thread_idx%ni;
            if (_b_desc(i,j,k)==3)//solid
            {
                _un(i,j,k) = _usolid(i,j,k);
                _un(i+1,j,k) = _usolid(i+1,j,k);
                _vn(i,j,k) = _vsolid(i,j,k);
                _vn(i,j+1,k) = _vsolid(i,j+1,k);
                _wn(i,j,k) = _wsolid(i,j,k);
                _wn(i,j,k+1) = _wsolid(i,j,k+1);
            }

            if(i==0)
            {
                _un(i,j,k) = 0;
            }
            if(j==0)
            {
                _vn(i,j,k) = 0;
            }
            if(k==0)
            {
                _wn(i,j,k) = 0;
            }
            if(i==ni-1)
            {
                _un(i+1,j,k) = 0;
            }
            if(j==nj-1)
            {
                _vn(i,j+1,k) = 0;
            }
            if(k==nk-1)
            {
                _wn(i,j,k+1) = 0;
            }
        });

        if (set_velocity_inflow)
            setVelocityFromEmitter(true);
    }

    calculateCurl(true);
    std::cout << GREEN << "fluxes.norm = " << fluxes.norm() << " circulations.norm = " << circulations.norm() << RESET << std::endl;
}

void COFLIPSolver::outputResult(uint frame, std::string filepath)
{
    if (!do_vel_advection_only)
    {
        writeVDB(frame, filepath, _amped_h, _rho, "density_1");
        writeVDB(frame, filepath, _amped_h, _T, "density_2");
    }

    if (bs_p == 1) {
        writeVDB<MyReal>(frame, filepath, _h, _un, "vel_x", true);
        writeVDB<MyReal>(frame, filepath, _h, _vn, "vel_y", true);
        writeVDB<MyReal>(frame, filepath, _h, _wn, "vel_z", true);
    } else {
        auto sampleFieldBSplineFunc = [&](const Vec<3, MyReal>& inpos, const Buffer3D<MyReal> &infield, int inselected_row)
        {
            Vec<3, MyReal> spos = inpos;
            clampPos(spos);
            return sampleFieldBSpline(spos, infield, inselected_row);
        };
        writeVDB<MyReal>(frame, filepath, _h_uniform, _un, "vel_x", true, sampleFieldBSplineFunc, 0, 2);
        writeVDB<MyReal>(frame, filepath, _h_uniform, _vn, "vel_y", true, sampleFieldBSplineFunc, 1, 2);
        writeVDB<MyReal>(frame, filepath, _h_uniform, _wn, "vel_z", true, sampleFieldBSplineFunc, 2, 2);
    }
    int boundary_index = 0;
    for (auto &b : sim_boundary)
    {
        char file_name[256];
        sprintf(file_name,"%s/sim_boundary%02d_%04d.obj", filepath.c_str(), boundary_index, frame);
        std::string objname(file_name);

        std::vector<openvdb::Vec3s> points;
        std::vector<openvdb::Vec4I> quads;
        openvdb::tools::volumeToMesh<openvdb::Grid<openvdb::tree::Tree4<float>::Type>>(*b.b_sdf, points, quads);
        writeObj(objname, points, quads);
        boundary_index += 1;
    }
}

void COFLIPSolver::multiplyWithInterp(Eigen::VectorXd& result, const Eigen::VectorXd& RHS, int res_amp) {
    result.setZero();
    uint numParticles = lagrangian_particles.pos_current.size();
    int ni = _nx*res_amp;
    int nj = _ny*res_amp;
    int nk = _nz*res_amp;
    int nip = ni+1-bs_p;
    int njp = nj+1-bs_p;
    int nkp = nk+1-bs_p;
    double used_h = (_h*(double)_nxp)/(double)(nip);
    int x_nF = (ni+1)*nj*nk;
    int y_nF = ni*(nj+1)*nk;
    tbb::parallel_for(tbb::blocked_range<int>(0,numParticles,TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
    {
        for (int i = r.begin(); i < r.end(); ++i) {
            Vec<3, MyReal> pos = lagrangian_particles.pos_temp[i];
            int ii = std::floor(pos[0]/used_h);
            int jj = std::floor(pos[1]/used_h);
            int kk = std::floor(pos[2]/used_h);
            double alpha = (double)pos[0]/used_h - (double)ii,
                  beta = (double)pos[1]/used_h - (double)jj,
                  gamma = (double)pos[2]/used_h - (double)kk;
            for(int kkk=kk;kkk<=kk+bs_p-1;kkk++)for(int jjj=jj;jjj<=jj+bs_p-1;jjj++)for(int iii=ii;iii<=ii+bs_p;iii++)
            {
                int idx = kkk*nj*(ni+1) + jjj*(ni+1) + iii;
                if (RHS(idx) == 0.){
                    continue;
                }

                double w = 0;
                if (bs_p == 2) {
                    w = 
                        lagrangian_particles.kernel2(ii == nip-1 ? 1.0 - alpha : alpha, 
                                    ii == nip-1 ? bs_p - (iii-ii) : iii-ii, 
                                    ii == 0 || ii == nip-1) *
                        lagrangian_particles.kernel1prime(jj == njp-1 ? 1.0 - beta : beta, 
                                    jj == njp-1 ? bs_p-1 - (jjj-jj) : jjj-jj, 
                                    jj == 0 || jj == njp-1) *
                        lagrangian_particles.kernel1prime(kk == nkp-1 ? 1.0 - gamma : gamma, 
                                    kk == nkp-1 ? bs_p-1 - (kkk-kk) : kkk-kk, 
                                    kk == 0 || kk == nkp-1);
                } else {
                    w =
                        lagrangian_particles.kernel3((ii == nip-1 || ii == nip-2) ? 1.0 - alpha : alpha, 
                                    (ii == nip-1 || ii == nip-2) ? bs_p - (iii-ii) : iii-ii, 
                                    (ii == 0 || ii == nip-1) ? 2 : ((ii == 1 || ii == nip-2) ? 1 : 0)) *
                        lagrangian_particles.kernel2prime((jj == njp-1 || jj == njp-2) ? 1.0 - beta : beta,
                                    (jj == njp-1 || jj == njp-2) ? bs_p-1 - (jjj-jj) : jjj-jj, 
                                    (jj == 0 || jj == njp-1) ? 2 : ((jj == 1 || jj == njp-2) ? 1 : 0)) *
                        lagrangian_particles.kernel2prime((kk == nkp-1 || kk == nkp-2) ? 1.0 - gamma : gamma,
                                    (kk == nkp-1 || kk == nkp-2) ? bs_p-1 - (kkk-kk) : kkk-kk, 
                                    (kk == 0 || kk == nkp-1) ? 2 : ((kk == 1 || kk == nkp-2) ? 1 : 0));
                }
                int particleCircIndex = i;
                result(particleCircIndex) += w * RHS(idx) * std::sqrt(lagrangian_particles.volume[i]*std::pow(res_amp,3));
            }

            for(int kkk=kk;kkk<=kk+bs_p-1;kkk++)for(int jjj=jj;jjj<=jj+bs_p;jjj++)for(int iii=ii;iii<=ii+bs_p-1;iii++)
            {
                int idx = x_nF + kkk*(nj+1)*ni + jjj*ni + iii;
                if (RHS(idx) == 0.){
                    continue;
                }

                double w = 0;
                if (bs_p == 2) {
                    w = 
                        lagrangian_particles.kernel1prime(ii == nip-1 ? 1.0 - alpha : alpha, 
                                    ii == nip-1 ? bs_p-1 - (iii-ii) : iii-ii,
                                    ii == 0 || ii == nip-1) *
                        lagrangian_particles.kernel2(jj == njp-1 ? 1.0 - beta : beta,
                                    jj == njp-1 ? bs_p - (jjj-jj) : jjj-jj,
                                    jj == 0 || jj == njp-1) *
                        lagrangian_particles.kernel1prime(kk == nkp-1 ? 1.0 - gamma : gamma, 
                                    kk == nkp-1 ? bs_p-1 - (kkk-kk) : kkk-kk,
                                    kk == 0 || kk == nkp-1);
                } else {
                    w =
                        lagrangian_particles.kernel2prime((ii == nip-1 || ii == nip-2) ? 1.0 - alpha : alpha, 
                                    (ii == nip-1 || ii == nip-2) ? bs_p-1 - (iii-ii) : iii-ii, 
                                    (ii == 0 || ii == nip-1) ? 2 : ((ii == 1 || ii == nip-2) ? 1 : 0)) *
                        lagrangian_particles.kernel3((jj == njp-1 || jj == njp-2) ? 1.0 - beta : beta,
                                    (jj == njp-1 || jj == njp-2) ? bs_p - (jjj-jj) : jjj-jj, 
                                    (jj == 0 || jj == njp-1) ? 2 : ((jj == 1 || jj == njp-2) ? 1 : 0)) *
                        lagrangian_particles.kernel2prime((kk == nkp-1 || kk == nkp-2) ? 1.0 - gamma : gamma, 
                                    (kk == nkp-1 || kk == nkp-2) ? bs_p-1 - (kkk-kk) : kkk-kk, 
                                    (kk == 0 || kk == nkp-1) ? 2 : ((kk == 1 || kk == nkp-2) ? 1 : 0));
                }
                int particleCircIndex = numParticles + i;
                result(particleCircIndex) += w * RHS(idx) * std::sqrt(lagrangian_particles.volume[i]*std::pow(res_amp,3));
            }

            for(int kkk=kk;kkk<=kk+bs_p;kkk++)for(int jjj=jj;jjj<=jj+bs_p-1;jjj++)for(int iii=ii;iii<=ii+bs_p-1;iii++)
            {
                int idx = (x_nF + y_nF) + kkk*nj*ni + jjj*ni + iii;
                if (RHS(idx) == 0.){
                    continue;
                }

                double w = 0;
                if (bs_p == 2) {
                    w = 
                        lagrangian_particles.kernel1prime(ii == nip-1 ? 1.0 - alpha : alpha, 
                                    ii == nip-1 ? bs_p-1 - (iii-ii) : iii-ii,
                                    ii == 0 || ii == nip-1) *
                        lagrangian_particles.kernel1prime(jj == njp-1 ? 1.0 - beta : beta,
                                    jj == njp-1 ? bs_p-1 - (jjj-jj) : jjj-jj,
                                    jj == 0 || jj == njp-1) *
                        lagrangian_particles.kernel2(kk == nkp-1 ? 1.0 - gamma : gamma, 
                                    kk == nkp-1 ? bs_p - (kkk-kk) : kkk-kk,
                                    kk == 0 || kk == nkp-1);
                } else {
                    w =
                        lagrangian_particles.kernel2prime((ii == nip-1 || ii == nip-2) ? 1.0 - alpha : alpha, 
                                    (ii == nip-1 || ii == nip-2) ? bs_p-1 - (iii-ii) : iii-ii, 
                                    (ii == 0 || ii == nip-1) ? 2 : ((ii == 1 || ii == nip-2) ? 1 : 0)) *
                        lagrangian_particles.kernel2prime((jj == njp-1 || jj == njp-2) ? 1.0 - beta : beta,
                                    (jj == njp-1 || jj == njp-2) ? bs_p-1 - (jjj-jj) : jjj-jj, 
                                    (jj == 0 || jj == njp-1) ? 2 : ((jj == 1 || jj == njp-2) ? 1 : 0)) *
                        lagrangian_particles.kernel3((kk == nkp-1 || kk == nkp-2) ? 1.0 - gamma : gamma, 
                                    (kk == nkp-1 || kk == nkp-2) ? bs_p - (kkk-kk) : kkk-kk, 
                                    (kk == 0 || kk == nkp-1) ? 2 : ((kk == 1 || kk == nkp-2) ? 1 : 0));
                }
                int particleCircIndex = 2*numParticles + i;
                result(particleCircIndex) += w * RHS(idx) * std::sqrt(lagrangian_particles.volume[i]*std::pow(res_amp,3));
            }
        }
    });
}

void COFLIPSolver::multiplyWithInterpTranspose(std::vector<atomwrapper<double> >& result, const Eigen::VectorXd& RHS, bool do_norm2_squared, int res_amp) {
    uint numParticles = lagrangian_particles.pos_current.size();
    int ni = _nx*res_amp;
    int nj = _ny*res_amp;
    int nk = _nz*res_amp;
    int nip = ni+1-bs_p;
    int njp = nj+1-bs_p;
    int nkp = nk+1-bs_p;
    double used_h = (_h*(double)_nxp)/(double)(nip);
    int nF = (ni+1)*nj*nk + ni*(nj+1)*nk + ni*nj*(nk+1);
    int x_nF = (ni+1)*nj*nk;
    int y_nF = ni*(nj+1)*nk;
    tbb::parallel_for(tbb::blocked_range<int>(0,nF, TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
    {
        for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
            result[tIdx]._a.store(0);
        }
    });
    tbb::parallel_for(tbb::blocked_range<int>(0,numParticles,TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
    {
        for (int i = r.begin(); i < r.end(); ++i) {
            Vec<3, MyReal> pos = lagrangian_particles.pos_temp[i];
            int ii = std::floor(pos[0]/used_h);
            int jj = std::floor(pos[1]/used_h);
            int kk = std::floor(pos[2]/used_h);
            double alpha = (double)pos[0]/used_h - (double)ii,
                    beta = (double)pos[1]/used_h - (double)jj,
                    gamma = (double)pos[2]/used_h - (double)kk;
            int particleCircIndex = i;
            if (RHS(particleCircIndex) != 0.) {
                for(int kkk=kk;kkk<=kk+bs_p-1;kkk++)for(int jjj=jj;jjj<=jj+bs_p-1;jjj++)for(int iii=ii;iii<=ii+bs_p;iii++)
                {
                    double w = 0;
                    if (bs_p == 2) {
                        w = 
                            lagrangian_particles.kernel2(ii == nip-1 ? 1.0 - alpha : alpha, 
                                        ii == nip-1 ? bs_p - (iii-ii) : iii-ii, 
                                        ii == 0 || ii == nip-1) *
                            lagrangian_particles.kernel1prime(jj == njp-1 ? 1.0 - beta : beta, 
                                        jj == njp-1 ? bs_p-1 - (jjj-jj) : jjj-jj, 
                                        jj == 0 || jj == njp-1) *
                            lagrangian_particles.kernel1prime(kk == nkp-1 ? 1.0 - gamma : gamma, 
                                        kk == nkp-1 ? bs_p-1 - (kkk-kk) : kkk-kk, 
                                        kk == 0 || kk == nkp-1);
                    } else {
                        w =
                            lagrangian_particles.kernel3((ii == nip-1 || ii == nip-2) ? 1.0 - alpha : alpha, 
                                        (ii == nip-1 || ii == nip-2) ? bs_p - (iii-ii) : iii-ii, 
                                        (ii == 0 || ii == nip-1) ? 2 : ((ii == 1 || ii == nip-2) ? 1 : 0)) *
                            lagrangian_particles.kernel2prime((jj == njp-1 || jj == njp-2) ? 1.0 - beta : beta,
                                        (jj == njp-1 || jj == njp-2) ? bs_p-1 - (jjj-jj) : jjj-jj, 
                                        (jj == 0 || jj == njp-1) ? 2 : ((jj == 1 || jj == njp-2) ? 1 : 0)) *
                            lagrangian_particles.kernel2prime((kk == nkp-1 || kk == nkp-2) ? 1.0 - gamma : gamma,
                                        (kk == nkp-1 || kk == nkp-2) ? bs_p-1 - (kkk-kk) : kkk-kk, 
                                        (kk == 0 || kk == nkp-1) ? 2 : ((kk == 1 || kk == nkp-2) ? 1 : 0));
                    }
                    int idx = kkk*nj*(ni+1) + jjj*(ni+1) + iii;
                    double value_to_add = w * RHS(particleCircIndex) * std::sqrt(lagrangian_particles.volume[i]*std::pow(res_amp,3));
                    if (do_norm2_squared) {
                        value_to_add = value_to_add*value_to_add;
                    }
                    result[idx].atomic_fetch_add(value_to_add);
                }
            }

            particleCircIndex = numParticles + i;
            if (RHS(particleCircIndex) != 0.) {
                for(int kkk=kk;kkk<=kk+bs_p-1;kkk++)for(int jjj=jj;jjj<=jj+bs_p;jjj++)for(int iii=ii;iii<=ii+bs_p-1;iii++)
                {
                    double w = 0;
                    if (bs_p == 2) {
                        w = 
                            lagrangian_particles.kernel1prime(ii == nip-1 ? 1.0 - alpha : alpha, 
                                        ii == nip-1 ? bs_p-1 - (iii-ii) : iii-ii,
                                        ii == 0 || ii == nip-1) *
                            lagrangian_particles.kernel2(jj == njp-1 ? 1.0 - beta : beta,
                                        jj == njp-1 ? bs_p - (jjj-jj) : jjj-jj,
                                        jj == 0 || jj == njp-1) *
                            lagrangian_particles.kernel1prime(kk == nkp-1 ? 1.0 - gamma : gamma, 
                                        kk == nkp-1 ? bs_p-1 - (kkk-kk) : kkk-kk,
                                        kk == 0 || kk == nkp-1);
                    } else {
                        w =
                            lagrangian_particles.kernel2prime((ii == nip-1 || ii == nip-2) ? 1.0 - alpha : alpha, 
                                        (ii == nip-1 || ii == nip-2) ? bs_p-1 - (iii-ii) : iii-ii, 
                                        (ii == 0 || ii == nip-1) ? 2 : ((ii == 1 || ii == nip-2) ? 1 : 0)) *
                            lagrangian_particles.kernel3((jj == njp-1 || jj == njp-2) ? 1.0 - beta : beta,
                                        (jj == njp-1 || jj == njp-2) ? bs_p - (jjj-jj) : jjj-jj, 
                                        (jj == 0 || jj == njp-1) ? 2 : ((jj == 1 || jj == njp-2) ? 1 : 0)) *
                            lagrangian_particles.kernel2prime((kk == nkp-1 || kk == nkp-2) ? 1.0 - gamma : gamma, 
                                        (kk == nkp-1 || kk == nkp-2) ? bs_p-1 - (kkk-kk) : kkk-kk, 
                                        (kk == 0 || kk == nkp-1) ? 2 : ((kk == 1 || kk == nkp-2) ? 1 : 0));
                    }
                    int idx = x_nF + kkk*(nj+1)*ni + jjj*ni + iii;
                    double value_to_add = w * RHS(particleCircIndex) * std::sqrt(lagrangian_particles.volume[i]*std::pow(res_amp,3));
                    if (do_norm2_squared) {
                        value_to_add = value_to_add*value_to_add;
                    }
                    result[idx].atomic_fetch_add(value_to_add);
                }
            }

            particleCircIndex = 2*numParticles + i;
            if (RHS(particleCircIndex) != 0.) {
                for(int kkk=kk;kkk<=kk+bs_p;kkk++)for(int jjj=jj;jjj<=jj+bs_p-1;jjj++)for(int iii=ii;iii<=ii+bs_p-1;iii++)
                {
                    double w = 0;
                    if (bs_p == 2) {
                        w = 
                            lagrangian_particles.kernel1prime(ii == nip-1 ? 1.0 - alpha : alpha, 
                                        ii == nip-1 ? bs_p-1 - (iii-ii) : iii-ii,
                                        ii == 0 || ii == nip-1) *
                            lagrangian_particles.kernel1prime(jj == njp-1 ? 1.0 - beta : beta,
                                        jj == njp-1 ? bs_p-1 - (jjj-jj) : jjj-jj,
                                        jj == 0 || jj == njp-1) *
                            lagrangian_particles.kernel2(kk == nkp-1 ? 1.0 - gamma : gamma, 
                                        kk == nkp-1 ? bs_p - (kkk-kk) : kkk-kk,
                                        kk == 0 || kk == nkp-1);
                    } else {
                        w =
                            lagrangian_particles.kernel2prime((ii == nip-1 || ii == nip-2) ? 1.0 - alpha : alpha, 
                                        (ii == nip-1 || ii == nip-2) ? bs_p-1 - (iii-ii) : iii-ii, 
                                        (ii == 0 || ii == nip-1) ? 2 : ((ii == 1 || ii == nip-2) ? 1 : 0)) *
                            lagrangian_particles.kernel2prime((jj == njp-1 || jj == njp-2) ? 1.0 - beta : beta,
                                        (jj == njp-1 || jj == njp-2) ? bs_p-1 - (jjj-jj) : jjj-jj, 
                                        (jj == 0 || jj == njp-1) ? 2 : ((jj == 1 || jj == njp-2) ? 1 : 0)) *
                            lagrangian_particles.kernel3((kk == nkp-1 || kk == nkp-2) ? 1.0 - gamma : gamma, 
                                        (kk == nkp-1 || kk == nkp-2) ? bs_p - (kkk-kk) : kkk-kk, 
                                        (kk == 0 || kk == nkp-1) ? 2 : ((kk == 1 || kk == nkp-2) ? 1 : 0));
                    }
                    int idx = (x_nF + y_nF) + kkk*nj*ni + jjj*ni + iii;
                    double value_to_add = w * RHS(particleCircIndex) * std::sqrt(lagrangian_particles.volume[i]*std::pow(res_amp,3));
                    if (do_norm2_squared) {
                        value_to_add = value_to_add*value_to_add;
                    }
                    result[idx].atomic_fetch_add(value_to_add);
                }
            }
        }
    });
}

void COFLIPSolver::multiplyWithStarVort(Eigen::VectorXd& result, const Eigen::VectorXd& RHS, int res_amp) {
    result.setZero();
    int ni = _nx*res_amp;
    int nj = _ny*res_amp;
    int nk = _nz*res_amp;
    int x_nE = ni*(nj+1)*(nk+1);
    int y_nE = (ni+1)*nj*(nk+1);
    int nE = ni*(nj+1)*(nk+1) + (ni+1)*nj*(nk+1) + (ni+1)*(nj+1)*nk;
    tbb::parallel_for(tbb::blocked_range<int>(0,nE, TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
        {
            for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
                bool is_x_dir = tIdx < x_nE;
                bool is_y_dir = !is_x_dir && tIdx < (x_nE + y_nE);
                int comp_slice = is_x_dir ? ni*(nj+1) : (is_y_dir ? (ni+1)*nj : (ni+1)*(nj+1));
                int comp_n = is_x_dir ? ni : (ni+1);
                int comp_tIdx = is_x_dir ? tIdx : (is_y_dir ? (tIdx - x_nE) : (tIdx - (x_nE + y_nE)));
                int k = comp_tIdx/comp_slice;
                int j = (comp_tIdx%comp_slice)/comp_n;
                int i = comp_tIdx%comp_n;
                int idx = is_x_dir ? i : (is_y_dir ? j : k);
                int other1_idx = is_x_dir ? j : (is_y_dir ? k : i);
                int other2_idx = is_x_dir ? k : (is_y_dir ? i : j);
                int nidx = is_x_dir ? ni : (is_y_dir ? nj : nk);
                int other1_nidx = is_x_dir ? nj : (is_y_dir ? nk : ni);
                int other2_nidx = is_x_dir ? nk : (is_y_dir ? ni : nj);
                if (sim_scheme == Scheme::CO_FLIP) {
                    if (use_DEC_diagonal_hodge_star) {
                        double length_of_primaledge = bs_p == 3 ? ((idx == 0 || idx == (nidx-1)) ? 1./3. : ((idx == 1 || idx == (nidx-2)) ? 2./3. : 1.0)) : (bs_p == 2 ? ((idx == 0 || idx == (nidx-1)) ? 0.5 : 1.0) : 1.0);
                        double length_of_dualedge1 = bs_p == 3 ? ((other1_idx == 0 || other1_idx == other1_nidx) ? 1./3. : ((other1_idx == 1 || other1_idx == (other1_nidx-1)) ? 0.5 : ((other1_idx == 2 || other1_idx == (other1_nidx-2)) ? 5./6. : 1.0))) : (bs_p == 2 ? ((other1_idx == 0 || other1_idx == other1_nidx) ? 0.5 : ((other1_idx == 1 || other1_idx == (other1_nidx-1)) ? 0.75 : 1.0)) : 1.0);
                        double length_of_dualedge2 = bs_p == 3 ? ((other2_idx == 0 || other2_idx == other2_nidx) ? 1./3. : ((other2_idx == 1 || other2_idx == (other2_nidx-1)) ? 0.5 : ((other2_idx == 2 || other2_idx == (other2_nidx-2)) ? 5./6. : 1.0))) : (bs_p == 2 ? ((other2_idx == 0 || other2_idx == other2_nidx) ? 0.5 : ((other2_idx == 1 || other2_idx == (other2_nidx-1)) ? 0.75 : 1.0)) : 1.0);
                        double area_of_dualface = length_of_dualedge1 * length_of_dualedge2;
                        double factor = area_of_dualface / length_of_primaledge;
                        result(tIdx) = factor * RHS(tIdx);
                    } else {
                        std::vector<std::array<double, 7> > Bs = bs_p == 3 ? B3s : (bs_p == 2 ? B2s : B1s);
                        std::vector<std::array<double, 7> > Ds = bs_p == 3 ? D2s : (bs_p == 2 ? D1s : D0s);
                        std::array<double, 7> Darray = Ds[std::min(std::min(2*(bs_p-1), idx), (nidx-1)-idx)];
                        std::array<double, 7> Barray1 = Bs[std::min(std::min(2*bs_p-1, other1_idx), other1_nidx-other1_idx)];
                        std::array<double, 7> Barray2 = Bs[std::min(std::min(2*bs_p-1, other2_idx), other2_nidx-other2_idx)];
                        if ((nidx-1)-idx < 2*(bs_p-1)) std::reverse(Darray.begin(), Darray.end());
                        if (other1_nidx-other1_idx < 2*bs_p-1) std::reverse(Barray1.begin(), Barray1.end());
                        if (other2_nidx-other2_idx < 2*bs_p-1) std::reverse(Barray2.begin(), Barray2.end());
                        for (int l = -(bs_p-1); l <= (bs_p-1); ++l) {
                            for (int m = -bs_p; m <= bs_p; ++m) {
                                for (int q = -bs_p; q <= bs_p; ++q) {
                                    int jdx = idx + l;
                                    int other1_jdx = other1_idx + m;
                                    int other2_jdx = other2_idx + q;
                                    if (jdx >= 0 && jdx < nidx && 
                                        other1_jdx >=0 && other1_jdx <= other1_nidx && 
                                        other2_jdx >=0 && other2_jdx <= other2_nidx) {
                                        int tJdx = tIdx + l * (is_x_dir ? 1 : (is_y_dir ? (ni+1) : (ni+1)*(nj+1))) + m * (is_x_dir ? ni : (is_y_dir ? (ni+1)*nj : 1)) + q * (is_x_dir ? ni*(nj+1) : (is_y_dir ? 1 : (ni+1)));
                                        double factor = Darray[l+3]*Barray1[m+3]*Barray2[q+3];
                                        result(tIdx) += factor * RHS(tJdx);
                                    }
                                }
                            }
                        }
                    }
                } else {
                    result(tIdx) = RHS(tIdx);
                }
            }
        });
}

void COFLIPSolver::multiplyWithStarFlux(Eigen::VectorXd& result, const Eigen::VectorXd& RHS, int res_amp) {
    result.setZero();
    int ni = _nx*res_amp;
    int nj = _ny*res_amp;
    int nk = _nz*res_amp;
    int x_nF = (ni+1)*nj*nk;
    int y_nF = ni*(nj+1)*nk;
    int nF = (ni+1)*nj*nk + ni*(nj+1)*nk + ni*nj*(nk+1);
    tbb::parallel_for(tbb::blocked_range<int>(0,nF, TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
    {
        for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
            bool is_x_dir = tIdx < x_nF;
            bool is_y_dir = !is_x_dir && tIdx < (x_nF + y_nF);
            int comp_slice = is_x_dir ? (ni+1)*nj : (is_y_dir ? ni*(nj+1) : ni*nj);
            int comp_n = is_x_dir ? (ni+1) : ni;
            int comp_tIdx = is_x_dir ? tIdx : (is_y_dir ? (tIdx - x_nF) : (tIdx - (x_nF + y_nF)));
            int k = comp_tIdx/comp_slice;
            int j = (comp_tIdx%comp_slice)/comp_n;
            int i = comp_tIdx%comp_n;
            int idx = is_x_dir ? i : (is_y_dir ? j : k);
            int nidx = is_x_dir ? ni : (is_y_dir ? nj : nk);
            int other1_idx = is_x_dir ? j : (is_y_dir ? k : i);
            int other1_nidx = is_x_dir ? nj : (is_y_dir ? nk : ni);
            int other2_idx = is_x_dir ? k : (is_y_dir ? i : j);
            int other2_nidx = is_x_dir ? nk : (is_y_dir ? ni : nj);
            if (sim_scheme == Scheme::CO_FLIP) {
                if (use_DEC_diagonal_hodge_star) {
                    double length_of_dualedge = bs_p == 3 ? ((idx == 0 || idx == nidx) ? 1./3. : ((idx == 1 || idx == (nidx-1)) ? 0.5 : ((idx == 2 || idx == (nidx-2)) ? 5./6. : 1.0))) : (bs_p == 2 ? ((idx == 0 || idx == nidx) ? 0.5 : ((idx == 1 || idx == (nidx-1)) ? 0.75 : 1.0)) : 1.0);
                    double length_of_edge1 = bs_p == 3 ? ((other1_idx == 0 || other1_idx == (other1_nidx-1)) ? 1./3. : ((other1_idx == 1 || other1_idx == (other1_nidx-2)) ? 2./3. : 1.0)) : (bs_p == 2 ? ((other1_idx == 0 || other1_idx == (other1_nidx-1)) ? 0.5 : 1.0) : 1.0);
                    double length_of_edge2 = bs_p == 3 ? ((other2_idx == 0 || other2_idx == (other2_nidx-1)) ? 1./3. : ((other2_idx == 1 || other2_idx == (other2_nidx-2)) ? 2./3. : 1.0)) : (bs_p == 2 ? ((other2_idx == 0 || other2_idx == (other2_nidx-1)) ? 0.5 : 1.0) : 1.0);
                    double area_of_primalface = length_of_edge1 * length_of_edge2;
                    double factor = length_of_dualedge / area_of_primalface;
                    result(tIdx) = factor * RHS(tIdx);
                } else {
                    std::vector<std::array<double, 7> > Bs = bs_p == 3 ? B3s : (bs_p == 2 ? B2s : B1s);
                    std::vector<std::array<double, 7> > Ds = bs_p == 3 ? D2s : (bs_p == 2 ? D1s : D0s);
                    std::array<double, 7> Barray = Bs[std::min(std::min(2*bs_p-1, idx), nidx-idx)];
                    std::array<double, 7> Darray1 = Ds[std::min(std::min(2*(bs_p-1), other1_idx), (other1_nidx-1)-other1_idx)];
                    std::array<double, 7> Darray2 = Ds[std::min(std::min(2*(bs_p-1), other2_idx), (other2_nidx-1)-other2_idx)];
                    if (nidx-idx < 2*bs_p-1) std::reverse(Barray.begin(), Barray.end());
                    if ((other1_nidx-1)-other1_idx < (2*(bs_p-1))) std::reverse(Darray1.begin(), Darray1.end());
                    if ((other2_nidx-1)-other2_idx < (2*(bs_p-1))) std::reverse(Darray2.begin(), Darray2.end());
                    for (int l = -bs_p; l <= bs_p; ++l) {
                        for (int m = -(bs_p-1); m <= (bs_p-1); ++m) {
                            for (int q = -(bs_p-1); q <= (bs_p-1); ++q) {
                                int jdx = idx + l;
                                int other1_jdx = other1_idx + m;
                                int other2_jdx = other2_idx + q;
                                if (jdx >= 0 && jdx <= nidx && other1_jdx >=0 && other1_jdx < other1_nidx && other2_jdx >=0 && other2_jdx < other2_nidx) {
                                    int tJdx = tIdx + l * (is_x_dir ? 1 : (is_y_dir ? ni : ni*nj)) + m * (is_x_dir ? (ni+1) : (is_y_dir ? ni*(nj+1) : 1)) + q * (is_x_dir ? (ni+1)*nj : (is_y_dir ? 1 : ni));
                                    double factor = Barray[l+3]*Darray1[m+3]*Darray2[q+3];
                                    result(tIdx) += factor * RHS(tJdx);
                                }
                            }
                        }
                    }
                }
            } else {
                result(tIdx) = RHS(tIdx);
            }
        }
    });
}

void COFLIPSolver::multiplyWithWedge(Eigen::VectorXd& result, const Eigen::VectorXd& RHS, int res_amp) {
    result.setZero();
    int ni = _nx*res_amp;
    int nj = _ny*res_amp;
    int nk = _nz*res_amp;
    int x_nF = (ni+1)*nj*nk;
    int y_nF = ni*(nj+1)*nk;
    int x_nE = ni*(nj+1)*(nk+1);
    int y_nE = (ni+1)*nj*(nk+1);
    int nF = (ni+1)*nj*nk + ni*(nj+1)*nk + ni*nj*(nk+1);
    tbb::parallel_for(tbb::blocked_range<int>(0,nF, TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
    {
        for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
            bool is_x_dir = tIdx < x_nF;
            bool is_y_dir = !is_x_dir && tIdx < (x_nF + y_nF);
            int comp_slice = is_x_dir ? (ni+1)*nj : (is_y_dir ? ni*(nj+1) : ni*nj);
            int comp_n = is_x_dir ? (ni+1) : ni;
            int comp_tIdx = is_x_dir ? tIdx : (is_y_dir ? (tIdx - x_nF) : (tIdx - (x_nF + y_nF)));
            int k = comp_tIdx/comp_slice;
            int j = (comp_tIdx%comp_slice)/comp_n;
            int i = comp_tIdx%comp_n;
            int idx = is_x_dir ? i : (is_y_dir ? j : k);
            int nidx = is_x_dir ? ni : (is_y_dir ? nj : nk);
            int other1_idx = is_x_dir ? j : (is_y_dir ? k : i);
            int other1_nidx = is_x_dir ? nj : (is_y_dir ? nk : ni);
            int other2_idx = is_x_dir ? k : (is_y_dir ? i : j);
            int other2_nidx = is_x_dir ? nk : (is_y_dir ? ni : nj);
            if (sim_scheme == Scheme::CO_FLIP) {
                std::vector<std::array<double, 6> > BDs = bs_p == 3 ? BD3s : (bs_p == 2 ? BD2s : BD1s);
                std::vector<std::array<double, 6> > DBs = bs_p == 3 ? DB3s : (bs_p == 2 ? DB2s : DB1s);
                std::array<double, 6> BDarray  = BDs[std::min(std::min(2*bs_p-1, idx), nidx-idx)];
                std::array<double, 6> DBarray1 = DBs[std::min(std::min(2*(bs_p-1), other1_idx), (other1_nidx-1)-other1_idx)];
                std::array<double, 6> DBarray2 = DBs[std::min(std::min(2*(bs_p-1), other2_idx), (other2_nidx-1)-other2_idx)];
                if (nidx-idx < 2*bs_p-1) std::reverse(BDarray.begin(), BDarray.end());
                if ((other1_nidx-1)-other1_idx < (2*(bs_p-1))) std::reverse(DBarray1.begin(), DBarray1.end());
                if ((other2_nidx-1)-other2_idx < (2*(bs_p-1))) std::reverse(DBarray2.begin(), DBarray2.end());
                for (int l = -bs_p; l <= (bs_p-1); ++l) {
                    for (int m = -(bs_p-1); m <= bs_p; ++m) {
                        for (int q = -(bs_p-1); q <= bs_p; ++q) {
                            int jdx = idx + l;
                            int other1_jdx = other1_idx + m;
                            int other2_jdx = other2_idx + q;
                            if (jdx >= 0 && jdx < nidx && other1_jdx >=0 && other1_jdx <= other1_nidx && other2_jdx >=0 && other2_jdx <= other2_nidx) {
                                int tJdx = is_x_dir ? (jdx + ni * other1_jdx + ni*(nj+1) * other2_jdx) :
                                            (is_y_dir ? (x_nE + other2_jdx + (ni+1) * jdx + (ni+1)*nj * other1_jdx) :
                                                        ((x_nE+y_nE) + other1_jdx + (ni+1) * other2_jdx + (ni+1)*(nj+1) * jdx));
                                double factor = BDarray[l+3]*DBarray1[m+2]*DBarray2[q+2];
                                result(tIdx) += factor * RHS(tJdx);
                            }
                        }
                    }
                }
            } else {
                std::array<double, 4> BDs = {1./48., 23./48., 23./48., 1./48.}; //sum=1
                for (int l = -2; l <= 1; ++l) {
                    for (int m = -1; m <= 2; ++m) {
                        for (int q = -1; q <= 2; ++q) {
                            int jdx = idx + l;
                            int other1_jdx = other1_idx + m;
                            int other2_jdx = other2_idx + q;
                            if (jdx >= 0 && jdx < nidx && other1_jdx >=0 && other1_jdx <= other1_nidx && other2_jdx >=0 && other2_jdx <= other2_nidx) {
                                int tJdx = is_x_dir ? (jdx + ni * other1_jdx + ni*(nj+1) * other2_jdx) :
                                            (is_y_dir ? (x_nE + other2_jdx + (ni+1) * jdx + (ni+1)*nj * other1_jdx) :
                                                        ((x_nE+y_nE) + other1_jdx + (ni+1) * other2_jdx + (ni+1)*(nj+1) * jdx));
                                double factor = BDs[l+2]*BDs[m+1]*BDs[q+1];
                                result(tIdx) += factor * RHS(tJdx);
                            }
                        }
                    }
                }
            }
        }
    });
}

void COFLIPSolver::multiplyWithD1Transpose(Eigen::VectorXd& result, const Eigen::VectorXd& RHS, int res_amp) {
    result.setZero();
    int ni = _nx*res_amp;
    int nj = _ny*res_amp;
    int nk = _nz*res_amp;
    int x_nF = (ni+1)*nj*nk;
    int y_nF = ni*(nj+1)*nk;
    int x_nE = ni*(nj+1)*(nk+1);
    int y_nE = (ni+1)*nj*(nk+1);
    int nE = ni*(nj+1)*(nk+1) + (ni+1)*nj*(nk+1) + (ni+1)*(nj+1)*nk;
    tbb::parallel_for(tbb::blocked_range<int>(0,nE, TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
    {
        for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
            bool is_x_dir = tIdx < x_nE;
            bool is_y_dir = !is_x_dir && tIdx < (x_nE + y_nE);
            bool is_z_dir = !is_x_dir && !is_y_dir;
            int comp_slice = is_x_dir ? ni*(nj+1) : (is_y_dir ? (ni+1)*nj : (ni+1)*(nj+1));
            int comp_n = is_x_dir ? ni : (ni+1);
            int comp_tIdx = is_x_dir ? tIdx : (is_y_dir ? (tIdx - x_nE) : (tIdx - (x_nE + y_nE)));
            int k = comp_tIdx/comp_slice;
            int j = (comp_tIdx%comp_slice)/comp_n;
            int i = comp_tIdx%comp_n;
            if (edgeInFluid(is_x_dir ? 0 : (is_y_dir ? 1 : 2), i, j, k, res_amp)) {
                int tIdx_oneForm1 = (is_x_dir ? x_nF : (is_y_dir ? (x_nF + y_nF) : 0)) + 
                    k*((is_z_dir ? (ni+1) : ni)*(is_x_dir ? (nj+1) : nj))+j*(is_z_dir ? (ni+1) : ni)+i;
                int tIdx_oneForm2 = (is_x_dir ? (x_nF + y_nF) : (is_y_dir ? 0 : x_nF)) + 
                    k*((is_y_dir ? (ni+1) : ni)*(is_z_dir ? (nj+1) : nj))+j*(is_y_dir ? (ni+1) : ni)+i;
                result(tIdx) += RHS(tIdx_oneForm1-(is_x_dir ? ni*(nj+1) : (is_y_dir ? 1 : (ni+1))));
                result(tIdx) -= RHS(tIdx_oneForm1);
                result(tIdx) -= RHS(tIdx_oneForm2-(is_x_dir ? ni : (is_y_dir ? (ni+1)*nj : 1)));
                result(tIdx) += RHS(tIdx_oneForm2);
            }
        }
    });
}

void COFLIPSolver::outputEnergy(std::string filename, MyReal curr_time)
{
    boost::filesystem::create_directories(filename);
    std::ofstream foutU, foutU2, foutU3, foutU4, foutU5, foutU6;
    std::string filenameU = filename + std::string("energy") + std::string(".txt");
    std::string filenameU2 = filename + std::string("helicity") + std::string(".txt");
    std::string filenameU3 = filename + std::string("super_helicity_density") + std::string(".txt");
    foutU.open(filenameU, curr_time == 0. ? std::ios_base::out : std::ios_base::app);
    foutU2.open(filenameU2, curr_time == 0. ? std::ios_base::out : std::ios_base::app);
    foutU3.open(filenameU3, curr_time == 0. ? std::ios_base::out : std::ios_base::app);
    
    calculateCurl(true, true);
    if (use_pressure_solver || sim_scheme != Scheme::CO_FLIP) {
        int res = _h/(10./128./4.)+TOLERANCE;
        std::uniform_real_distribution<double> unif(0, 1);
        double energy = tbb::parallel_reduce(
                    tbb::blocked_range<int>(0,_nzp*_nyp*_nxp),
                    0.0,
                    [&](tbb::blocked_range<int> range, double running_total)
                    {
                        for (int tIdx=range.begin(); tIdx<range.end(); ++tIdx)
                        {
                            std::mt19937_64 rng;
                            rng.seed(tIdx);
                            int k = tIdx / (_nxp*_nyp);
                            int j = (tIdx % (_nxp*_nyp)) / _nxp;
                            int i = tIdx % _nxp;
                            double x = (double)i*_h;
                            double y = (double)j*_h;
                            double z = (double)k*_h;
                            for(int kk=0;kk<res;kk++)
                            {
                                for(int jj=0;jj<res;jj++)
                                {
                                    for(int ii=0;ii<res;ii++)
                                    {
                                        Vec<3, double> spos(x+((double)ii + unif(rng))/(double)res*_h,
                                                            y+((double)jj + unif(rng))/(double)res*_h,
                                                            z+((double)kk + unif(rng))/(double)res*_h);
                                        Vec<3, double> vel = getVelocityBSpline(spos, _un, _vn, _wn);
                                        running_total += dot(vel, vel);
                                    }
                                }
                            }
                        }

                        return running_total;
                    }, std::plus<double>() );
        energy *= std::pow(_h/(double)res, 3);
        std::cout << RED << "Energy = " << energy << RESET <<  std::endl;
        foutU << energy << " " << curr_time << std::endl;

        double helicity = tbb::parallel_reduce( 
                    tbb::blocked_range<int>(0,_nzp*_nyp*_nxp),
                    0.0,
                    [&](tbb::blocked_range<int> range, double running_total)
                    {
                        for (int tIdx=range.begin(); tIdx<range.end(); ++tIdx)
                        {
                            std::mt19937_64 rng;
                            rng.seed(tIdx);
                            int k = tIdx / (_nxp*_nyp);
                            int j = (tIdx % (_nxp*_nyp)) / _nxp;
                            int i = tIdx % _nxp;
                            double x = (double)i*_h;
                            double y = (double)j*_h;
                            double z = (double)k*_h;
                            for(int kk=0;kk<res;kk++)
                            {
                                for(int jj=0;jj<res;jj++)
                                {
                                    for(int ii=0;ii<res;ii++)
                                    {
                                        Vec<3, double> spos(x+((double)ii + unif(rng))/(double)res*_h,
                                                            y+((double)jj + unif(rng))/(double)res*_h,
                                                            z+((double)kk + unif(rng))/(double)res*_h);
                                        Vec<3, double> vel = getVelocityBSpline(spos, _un, _vn, _wn);
                                        Vec<3, double> vort = getVorticityBSpline(spos, _vort_un, _vort_vn, _vort_wn);
                                        running_total += dot(vel, vort);
                                    }
                                }
                            }
                        }

                        return running_total;
                    }, std::plus<double>() );
        helicity *= std::pow(_h/(double)(res), 3);
        std::cout << RED << "Helicity = " << helicity << RESET <<  std::endl;
        foutU2 << helicity << " " << curr_time << std::endl;

        int ni = _nx;
        int nj = _ny;
        int nk = _nz;
        int x_nF = (ni+1)*nj*nk;
        int y_nF = ni*(nj+1)*nk;
        Eigen::VectorXd curlOfVorticity = (1.0 / _h) * d1_matrix * prev_vorts1form;
        Buffer3D<MyReal> _u_calc, _v_calc, _w_calc;
        _u_calc.init(_nx+1, _ny, _nz, _h, 0.5f, 0.f, 0.f);
        _v_calc.init(_nx, _ny+1, _nz, _h, 0.f, 0.5f, 0.f);
        _w_calc.init(_nx, _ny, _nz+1, _h, 0.f, 0.f, 0.5f);
        tbb::parallel_for(tbb::blocked_range<int>(0,_nF, TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
        {
            for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
                bool is_x_dir = tIdx < x_nF;
                bool is_y_dir = !is_x_dir && tIdx < (x_nF + y_nF);
                int comp_slice = is_x_dir ? (ni+1)*nj : (is_y_dir ? ni*(nj+1) : ni*nj);
                int comp_n = is_x_dir ? (ni+1) : ni;
                int comp_tIdx = is_x_dir ? tIdx : (is_y_dir ? (tIdx - x_nF) : (tIdx - (x_nF + y_nF)));
                int k = comp_tIdx/comp_slice;
                int j = (comp_tIdx%comp_slice)/comp_n;
                int i = comp_tIdx%comp_n;
                if (is_x_dir) {
                    _u_calc(i,j,k) = curlOfVorticity[tIdx];
                } else if (is_y_dir) {
                    _v_calc(i,j,k) = curlOfVorticity[tIdx];
                } else {
                    _w_calc(i,j,k) = curlOfVorticity[tIdx];
                }
            }
        });
        double super_helicity_density = tbb::parallel_reduce( 
                    tbb::blocked_range<int>(0,_nzp*_nyp*_nxp),
                    0.0,
                    [&](tbb::blocked_range<int> range, double running_total)
                    {
                        for (int tIdx=range.begin(); tIdx<range.end(); ++tIdx)
                        {
                            std::mt19937_64 rng;
                            rng.seed(tIdx);
                            int k = tIdx / (_nxp*_nyp);
                            int j = (tIdx % (_nxp*_nyp)) / _nxp;
                            int i = tIdx % _nxp;
                            double x = (double)i*_h;
                            double y = (double)j*_h;
                            double z = (double)k*_h;
                            for(int kk=0;kk<res;kk++)
                            {
                                for(int jj=0;jj<res;jj++)
                                {
                                    for(int ii=0;ii<res;ii++)
                                    {
                                        Vec<3, double> spos(x+((double)ii + unif(rng))/(double)res*_h,
                                                            y+((double)jj + unif(rng))/(double)res*_h,
                                                            z+((double)kk + unif(rng))/(double)res*_h);
                                        Vec<3, double> curlOfVort = getVelocityBSpline(spos, _u_calc, _v_calc, _w_calc);
                                        Vec<3, double> vort = getVorticityBSpline(spos, _vort_un, _vort_vn, _vort_wn);
                                        running_total += dot(curlOfVort, vort);
                                    }
                                }
                            }
                        }

                        return running_total;
                    }, std::plus<double>() );
        super_helicity_density *= std::pow(_h/(double)(res), 3);
        _u_calc.free();
        _v_calc.free();
        _w_calc.free();
        std::cout << RED << "Super-Helicity Density = " << super_helicity_density << RESET <<  std::endl;
        foutU3 << super_helicity_density << " " << curr_time << std::endl;
    } else {
        double energy = fluxes.transpose() * circulations;
        energy *= std::pow(_h, 3);
        std::cout << RED << "Energy = " << energy << RESET <<  std::endl;
        foutU << energy << " " << curr_time << std::endl;

        Eigen::VectorXd prev_vorts1form_double = prev_vorts1form.cast<double>();
        Eigen::VectorXd result(_nF);
        multiplyWithWedge(result, prev_vorts1form_double);

        double helicity = fluxes.transpose().cast<double>() * result;
        helicity *= std::pow(_h, 3);
        std::cout << RED << "Helicity = " << helicity << RESET <<  std::endl;
        foutU2 << helicity << " " << curr_time << std::endl;

        double super_helicity_density = ((1.0 / _h) * d1_matrix * prev_vorts1form_double).transpose() * result;
        super_helicity_density *= std::pow(_h, 3);
        std::cout << RED << "Super-Helicity Density = " << super_helicity_density << RESET <<  std::endl;
        foutU3 << super_helicity_density << " " << curr_time << std::endl;
    }
    foutU.close();
    foutU2.close();
    foutU3.close();
}

void COFLIPSolver::calculateCurl(bool do_star_fluxes, bool solve_for_dual_vorts) {
    int ni = _nx;
    int nj = _ny;
    int nk = _nz;
    if (do_star_fluxes) {
        takeDualwrtStar(_un, _vn, _wn, false, false);
    } else {
        int x_nF = (ni+1)*nj*nk;
        int y_nF = ni*(nj+1)*nk;
        tbb::parallel_for(tbb::blocked_range<int>(0,_nF, TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
        {
            for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
                bool is_x_dir = tIdx < x_nF;
                bool is_y_dir = !is_x_dir && tIdx < (x_nF + y_nF);
                int comp_slice = is_x_dir ? (ni+1)*nj : (is_y_dir ? ni*(nj+1) : ni*nj);
                int comp_n = is_x_dir ? (ni+1) : ni;
                int comp_tIdx = is_x_dir ? tIdx : (is_y_dir ? (tIdx - x_nF) : (tIdx - (x_nF + y_nF)));
                int k = comp_tIdx/comp_slice;
                int j = (comp_tIdx%comp_slice)/comp_n;
                int i = comp_tIdx%comp_n;
                circulations[tIdx] = is_x_dir ? _un(i,j,k) : (is_y_dir ? _vn(i,j,k) : _wn(i,j,k));
            }
        });
    }
    vorts = (d1transpose_matrix * circulations.cast<double>()).cast<MyReal>() / _h;

    if (solve_for_dual_vorts) {
        int iterations;
        double tolerance;
        auto multiply_func = [&](Eigen::VectorXd& output, const Eigen::VectorXd& input) {
            multiplyWithStarVort(output, input);
        };
        Eigen::VectorXd vorts_double = vorts.cast<double>();
        Eigen::VectorXd prev_vorts1form_double = prev_vorts1form.cast<double>();
        bool success = AMGPCGSolve<double>(multiply_func, vorts_double, prev_vorts1form_double, starvort_precond, (double)TOLERANCE, MAX_ITERATIONS, tolerance, iterations, !is_matrix_small_enough ? 0.2 : 1., !is_matrix_small_enough ? 2 : 1);
        prev_vorts1form = prev_vorts1form_double.cast<MyReal>();
        if (!success) {
            printf("WARNING: inverse star-vort solve failed!************************************************\n");
        }
        std::cout << "#iteration:      " << iterations << std::endl;
        std::cout << "estimated error: " << tolerance << std::endl;
        int x_nE = ni*(nj+1)*(nk+1);
        int y_nE = (ni+1)*nj*(nk+1);
        tbb::parallel_for(tbb::blocked_range<int>(0,_nE, TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
        {
            for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
                bool is_x_dir = tIdx < x_nE;
                bool is_y_dir = !is_x_dir && tIdx < (x_nE + y_nE);
                int comp_slice = is_x_dir ? ni*(nj+1) : (is_y_dir ? (ni+1)*nj : (ni+1)*(nj+1));
                int comp_n = is_x_dir ? ni : (ni+1);
                int comp_tIdx = is_x_dir ? tIdx : (is_y_dir ? (tIdx - x_nE) : (tIdx - (x_nE + y_nE)));
                int k = comp_tIdx/comp_slice;
                int j = (comp_tIdx%comp_slice)/comp_n;
                int i = comp_tIdx%comp_n;
                if (is_x_dir) {
                    _vort_un(i,j,k) = prev_vorts1form[tIdx];
                } else if (is_y_dir) {
                    _vort_vn(i,j,k) = prev_vorts1form[tIdx];
                } else {
                    _vort_wn(i,j,k) = prev_vorts1form[tIdx];
                }
            }
        });
    }
}

void COFLIPSolver::outputVorticityIntegral(std::string filename, MyReal curr_time)
{
    boost::filesystem::create_directories(filename);
    std::ofstream foutU, foutU2;
    std::string filenameU = filename + std::string("vort1") + std::string(".txt");
    std::string filenameU2 = filename + std::string("vort2") + std::string(".txt");
    foutU.open(filenameU, curr_time == 0. ? std::ios_base::out : std::ios_base::app);
    foutU2.open(filenameU2, curr_time == 0. ? std::ios_base::out : std::ios_base::app);

    calculateCurl(true, true);
    if (use_pressure_solver || sim_scheme != Scheme::CO_FLIP) {
        int res = _h/(10./128./4.)+TOLERANCE;
        std::uniform_real_distribution<double> unif(0, 1);
        double firstMomentVorticityIntegral = tbb::parallel_reduce( 
                    tbb::blocked_range<int>(0,_nzp*_nyp*_nxp),
                    0.0,
                    [&](tbb::blocked_range<int> range, double running_total)
                    {
                        for (int tIdx=range.begin(); tIdx<range.end(); ++tIdx)
                        {
                            std::mt19937_64 rng;
                            rng.seed(tIdx);
                            int k = tIdx / (_nxp*_nyp);
                            int j = (tIdx % (_nxp*_nyp)) / _nxp;
                            int i = tIdx % _nxp;
                            double x = (double)i*_h;
                            double y = (double)j*_h;
                            double z = (double)k*_h;
                            for(int kk=0;kk<res;kk++)
                            {
                                for(int jj=0;jj<res;jj++)
                                {
                                    for(int ii=0;ii<res;ii++)
                                    {
                                        Vec<3, double> spos(x+((double)ii + unif(rng))/(double)res*_h,
                                                            y+((double)jj + unif(rng))/(double)res*_h,
                                                            z+((double)kk + unif(rng))/(double)res*_h);
                                        Vec<3, double> vort = getVorticityBSpline(spos, _vort_un, _vort_vn, _vort_wn);
                                        running_total += vort[0] + vort[1] + vort[2];
                                    }
                                }
                            }
                        }

                        return running_total;
                    }, std::plus<double>() );
        firstMomentVorticityIntegral *= std::pow(_h/(double)res, 3);
        double secondMomentVorticityIntegral = tbb::parallel_reduce( 
                    tbb::blocked_range<int>(0,_nzp*_nyp*_nxp),
                    0.0,
                    [&](tbb::blocked_range<int> range, double running_total)
                    {
                        for (int tIdx=range.begin(); tIdx<range.end(); ++tIdx)
                        {
                            std::mt19937_64 rng;
                            rng.seed(tIdx);
                            int k = tIdx / (_nxp*_nyp);
                            int j = (tIdx % (_nxp*_nyp)) / _nxp;
                            int i = tIdx % _nxp;
                            double x = (double)i*_h;
                            double y = (double)j*_h;
                            double z = (double)k*_h;
                            for(int kk=0;kk<res;kk++)
                            {
                                for(int jj=0;jj<res;jj++)
                                {
                                    for(int ii=0;ii<res;ii++)
                                    {
                                        Vec<3, double> spos(x+((double)ii + unif(rng))/(double)res*_h,
                                                            y+((double)jj + unif(rng))/(double)res*_h,
                                                            z+((double)kk + unif(rng))/(double)res*_h);
                                        Vec<3, double> vort = getVorticityBSpline(spos, _vort_un, _vort_vn, _vort_wn);
                                        running_total += dot(vort, vort);
                                    }
                                }
                            }
                        }

                        return running_total;
                    }, std::plus<double>() );
        secondMomentVorticityIntegral *= std::pow(_h/(double)res, 3);
        std::cout << RED << "Vorticity Moment 1 = " << firstMomentVorticityIntegral << RESET << std::endl;
        std::cout << RED << "Vorticity Moment 2 = " << secondMomentVorticityIntegral << RESET << std::endl;
        foutU << firstMomentVorticityIntegral << " " << curr_time << std::endl;
        foutU2 << secondMomentVorticityIntegral << " " << curr_time << std::endl;
    } else {
        Eigen::VectorXd ones2form(_nE);
        Eigen::VectorXd vorts1form_double = prev_vorts1form.cast<double>();
        multiplyWithStarVort(ones2form, Eigen::VectorXd::Constant(_nE, 1.0));
        double firstMomentVorticityIntegral = vorts1form_double.transpose() * ones2form;
        double secondMomentVorticityIntegral = vorts1form_double.transpose() * vorts;
        firstMomentVorticityIntegral *= std::pow(_h, 3);
        secondMomentVorticityIntegral *= std::pow(_h, 3);
        std::cout << RED << "Vorticity Moment 1 = " << firstMomentVorticityIntegral << RESET << std::endl;
        std::cout << RED << "Vorticity Moment 2 = " << secondMomentVorticityIntegral << RESET << std::endl;
        foutU << firstMomentVorticityIntegral << " " << curr_time << std::endl;
        foutU2 << secondMomentVorticityIntegral << " " << curr_time << std::endl;
    }

    foutU.close();
    foutU2.close();
}

void COFLIPSolver::setupFromVDBFiles(const std::string& filepathVelField,
                                     const std::string& filepathDensityRhoField, 
                                     const std::string& filepathDensityTempField)
{
    readVDBField<MyReal>(filepathDensityRhoField, _rho, "density");
    readVDBField<MyReal>(filepathDensityTempField, _T, "density");
    readVDBField<MyReal>(filepathVelField, _un, "vel.x");
    readVDBField<MyReal>(filepathVelField, _vn, "vel.y");
    readVDBField<MyReal>(filepathVelField, _wn, "vel.z");

    if (!filepathVelField.empty() && sim_scheme == Scheme::CO_FLIP) {
        int ni = _nx;
        int nj = _ny;
        int nk = _nz;
        int x_nF = (ni+1)*nj*nk;
        int y_nF = ni*(nj+1)*nk;
        int bs_p_saved = bs_p;
        bs_p = 1;
        seedParticles();
        sampleParticlesFromGrid();
        bs_p = bs_p_saved;
        solveInterpDagger(0, 1);
        tbb::parallel_for(tbb::blocked_range<int>(0,_nF, TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
        {
            for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
                bool is_x_dir = tIdx < x_nF;
                bool is_y_dir = !is_x_dir && tIdx < (x_nF + y_nF);
                int comp_slice = is_x_dir ? (ni+1)*nj : (is_y_dir ? ni*(nj+1) : ni*nj);
                int comp_n = is_x_dir ? (ni+1) : ni;
                int comp_tIdx = is_x_dir ? tIdx : (is_y_dir ? (tIdx - x_nF) : (tIdx - (x_nF + y_nF)));
                int k = comp_tIdx/comp_slice;
                int j = (comp_tIdx%comp_slice)/comp_n;
                int i = comp_tIdx%comp_n;
                if (is_x_dir) {
                    _un(i,j,k) = fluxes[tIdx];
                } else if (is_y_dir) {
                    _vn(i,j,k) = fluxes[tIdx];
                } else {
                    _wn(i,j,k) = fluxes[tIdx];
                }
            }
        });
        fluxes_midpoint = fluxes;
    }
}

void COFLIPSolver::pressureProjectVelField()
{
    if (sim_scheme == Scheme::CO_FLIP) {
        if (use_pressure_solver) {
            projection();
        } else {
            projectionWithVort();
        }
    } else {
        projection();
    }
}