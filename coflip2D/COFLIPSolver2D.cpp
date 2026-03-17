#include "COFLIPSolver2D.h"
#include "../utils/bsplines.h"

inline Vec2d COFLIPSolver2D::traceFE(double dt, const Vec2d &pos, const Array2d& un, const Array2d& vn)
{
	Vec2d input = pos;
	Vec2d velocity = getVelocityBSpline(input, un, vn);
	input = input + dt*velocity;
	clampPos(input);
	return input;
}

inline Vec2d COFLIPSolver2D::traceRK3(double dt, const Vec2d &pos, const Array2d& un, const Array2d& vn)
{
	double c1 = 2.0 / 9.0*dt, c2 = 3.0 / 9.0 * dt, c3 = 4.0 / 9.0 * dt;
	Vec2d input = pos;
	Vec2d velocity1 = getVelocityBSpline(input, un, vn);
	Vec2d midp1 = input + ((0.5*dt))*velocity1;
	Vec2d velocity2 = getVelocityBSpline(midp1, un, vn);
	Vec2d midp2 = input + ((0.75*dt))*velocity2;
	Vec2d velocity3 = getVelocityBSpline(midp2, un, vn);
	input = input + c1*velocity1 + c2*velocity2 + c3*velocity3;
	clampPos(input);
	return input;
}

inline Vec2d COFLIPSolver2D::traceRK4(double dt, const Vec2d& pos, const Array2d& un, const Array2d& vn)
{
    double c1 = 1.0 / 6.0 * dt, c2 = 1.0 / 3.0 * dt, c3 = 1.0 / 3.0 * dt, c4 = 1.0 / 6.0 * dt;
    Vec2d input = pos;
    Vec2d velocity1 = getVelocityBSpline(input, un, vn);
    Vec2d midp1 = input + ((0.5 * dt)) * velocity1;
    clampPos(midp1);
    Vec2d velocity2 = getVelocityBSpline(midp1, un, vn);
    Vec2d midp2 = input + ((0.5 * dt)) * velocity2;
    clampPos(midp2);
    Vec2d velocity3 = getVelocityBSpline(midp2, un, vn);
    Vec2d midp3 = input + ((dt)) * velocity3;
    clampPos(midp3);
    Vec2d velocity4 = getVelocityBSpline(midp3, un, vn);
    input = input + c1 * velocity1 + c2 * velocity2 + c3 * velocity3 + c4 * velocity4;
    clampPos(input);
    return input;
}

inline Vec2d COFLIPSolver2D::solveODE(double dt, const Vec2d &pos, const Array2d& un, const Array2d& vn)
{
    double ddt = dt;
    Vec2d pos1 = traceRK4(ddt, pos, un, vn);
    ddt/=2.0;
    int substeps = 2;
    Vec2d pos2 = traceRK4(ddt, pos, un, vn);pos2 = traceRK4(ddt, pos2, un, vn);
    int iter = 0;

    while(dist(pos2,pos1)>1e-4*h && iter<10)
    {
        pos1 = pos2;
        ddt/=2.0;
        substeps *= 2;
        pos2 = pos;
        for(int j=0;j<substeps;j++)
        {
            pos2 = traceRK4(ddt, pos2, un, vn);
        }
        iter++;
    }
    return pos2;
}

inline Eigen::Matrix2d COFLIPSolver2D::pullbackRK4(double dt, Vec2d &inout_pos, const Eigen::Matrix2d& input_pullback, const Array2d& un, const Array2d& vn)
{
    double c1 = 1.0 / 6.0 * dt, c2 = 1.0 / 3.0 * dt, c3 = 1.0 / 3.0 * dt, c4 = 1.0 / 6.0 * dt;
    Vec2d input = inout_pos;
    Vec2d velocity1 = getVelocityBSpline(input, un, vn);
    Eigen::Matrix2d jacobianVelocity1 = getJacobianVelocityBSpline(input, un, vn);
    Eigen::Matrix2d midpullbackdot1 = -jacobianVelocity1.transpose()*input_pullback;
    Vec2d midp1 = input + (0.5 * dt) * velocity1;
    clampPos(midp1);
    Eigen::Matrix2d midpullback1 = input_pullback + (0.5 * dt) * midpullbackdot1;
    Vec2d velocity2 = getVelocityBSpline(midp1, un, vn);
    Eigen::Matrix2d jacobianVelocity2 = getJacobianVelocityBSpline(midp1, un, vn);
    Eigen::Matrix2d midpullbackdot2 = -jacobianVelocity2.transpose()*midpullback1;
    Vec2d midp2 = input + (0.5 * dt) * velocity2;
    clampPos(midp2);
    Eigen::Matrix2d midpullback2 = input_pullback + (0.5 * dt) * midpullbackdot2;
    Vec2d velocity3 = getVelocityBSpline(midp2, un, vn);
    Eigen::Matrix2d jacobianVelocity3 = getJacobianVelocityBSpline(midp2, un, vn);
    Eigen::Matrix2d midpullbackdot3 = -jacobianVelocity3.transpose()*midpullback2;
    Vec2d midp3 = input + dt * velocity3;
    clampPos(midp3);
    Eigen::Matrix2d midpullback3 = input_pullback + dt * midpullbackdot3;
    Vec2d velocity4 = getVelocityBSpline(midp3, un, vn);
    Eigen::Matrix2d jacobianVelocity4 = getJacobianVelocityBSpline(midp3, un, vn);
    Eigen::Matrix2d midpullbackdot4 = -jacobianVelocity4.transpose()*midpullback3;
    Eigen::Matrix2d pullback = input_pullback + c1 * midpullbackdot1 + c2 * midpullbackdot2 + c3 * midpullbackdot3 + c4 * midpullbackdot4;
    clampPullback(pullback);
    inout_pos = inout_pos + c1 * velocity1 + c2 * velocity2 + c3 * velocity3 + c4 * velocity4;
    clampPos(inout_pos);
    return pullback;
}

void COFLIPSolver2D::getCFL()
{
    _cfl = h / maxVel();
}

inline double COFLIPSolver2D::lerp(const double& v0, const double& v1, double c)
{
    return (1.0-c)*v0+c*v1;
}

inline double COFLIPSolver2D::bilerp(const double& v00, const double& v01, const double& v10, const double& v11, double cx, double cy)
{
    return lerp(lerp(v00,v01,cx), lerp(v10,v11,cx),cy);
}

void COFLIPSolver2D::semiLagAdvect(const Array2d & src, Array2d & dst, double dt, int ni, int nj, double off_x, double off_y)
{
    tbb::parallel_for(tbb::blocked_range2d<int,int>{0,ni,1,0,nj,1}, [&](const tbb::blocked_range2d<int,int> &r)
    {
        int ie = r.rows().end();
        int je = r.cols().end();
        for (int j = r.cols().begin(); j < je; ++j) {
            for (int i = r.rows().begin(); i < ie; ++i) {
                Vec2d pos = h*Vec2d(i, j) + h*Vec2d(off_x, off_y);
                Vec2d back_pos = solveODE(-dt, pos, u, v);
                dst(i, j) = sampleField(back_pos - h*Vec2d(off_x, off_y), src);
            }
        }
    });
}

void COFLIPSolver2D::advance(double dt, int frame, int delayed_reinit_frequency)
{
    switch(sim_scheme)
    {
        case R_POLYFLIP:
            advanceReflectionPOLYFLIP(dt, frame, delayed_reinit_frequency);
            break;
        case CF_POLYFLIP:
        case POLYFLIP:
            advanceCovectorPOLYFLIP(dt, frame, delayed_reinit_frequency);
            break;
        case CO_FLIP:
            advanceCOFLIP(dt, frame, delayed_reinit_frequency);
            break;
        case POLYPIC:
            advancePolyPIC(dt, frame);
            break;
        default:
            break;
    }
}

COFLIPSolver2D::COFLIPSolver2D(int nx, int ny, double L, int N, bool bc, Scheme s_scheme)
{
    sim_scheme = s_scheme;
    if (sim_scheme != Scheme::CO_FLIP || do_mass_lumping) {
        bs_p = 1;
    }
    use_neumann_boundary = bc;
    ni = nx;
    nj = ny;
    nV = (ni+1)*(nj+1);
    nF = (ni+1)*nj + ni*(nj+1);
    nC = ni*nj;
    nip = nx+1-bs_p;
    njp = ny+1-bs_p;
    h = L / (double)(nip);
    h_uniform = L / (double)(ni);
    u.resize(nx + 1, ny, 0);
    v.resize(nx, ny + 1, 0);
    u_save.resize(nx + 1, ny, 0);
    v_save.resize(nx, ny + 1, 0);
    u_mass.resize(nx + 1, ny, 1);
    v_mass.resize(nx, ny + 1, 1);
    curl.resize(nx+1, ny+1, 0);
    curl_mass.resize(nx+1, ny+1, 0);
    rho.resize(nx*(sim_scheme >= Scheme::R_POLYFLIP ? 2 : 1), ny*(sim_scheme >= Scheme::R_POLYFLIP ? 2 : 1), 0);
    temperature.resize(nx*(sim_scheme >= Scheme::R_POLYFLIP ? 2 : 1), ny*(sim_scheme >= Scheme::R_POLYFLIP ? 2 : 1), 0);
    pressure.resize(nx*ny);
    rhs.resize(nx*ny);
    boundaryMask.resize(nx,ny, 0);
    boundaryMask_nodes.resize(nx+1,ny+1, 0);
    emitterMask.resize(nx, ny, 0);

    m_N = N;

    if (sim_scheme == Scheme::CF_POLYFLIP) {
        backward_x.resize(nx, ny, 0);
        backward_y.resize(nx, ny, 0);
    }

    fluxes_original.resize(nF);
    fluxes.resize(nF);
    pre_solve_fluxes.resize(nF);
    solved_visc_fluxes.resize(nF);
    circulations.resize(nF);
    vorts.resize(nV);
    prev_vorts0form.resize(nV);
    streamfunction.resize(nV);
    fluxes.setZero();
    pre_solve_fluxes.setZero();
    solved_visc_fluxes.setZero();
    circulations.setZero();
    vorts.setZero();
    prev_vorts0form.setZero();
    streamfunction.setZero();
}

void COFLIPSolver2D::applyBuoyancyForce(Array2d &_v, double dt)
{
    if (alpha_buoyancy == 0. && beta_buoyancy == 0.) {
        return;
    }

    /// NOTE: rho and temperature represent two kinds of fluid
    /// with different density, so both rho and temperature act like drop force
    /// for smoke, you may want temperature acts like rising force, which change the - beta*temperature(i,j) to be + beta*temperature(i,j)
    tbb::parallel_for(tbb::blocked_range2d<int,int>{0,ni,1,0,nj+1,1}, [&](const tbb::blocked_range2d<int,int> &r)
    {
        int ie = r.rows().end();
        int je = r.cols().end();
        for (int j = r.cols().begin(); j < je; ++j) {
            for (int i = r.rows().begin(); i < ie; ++i) {
                Vec2d spos = (Vec2d(i,j) + Vec2d(0.5, 0.0)) * h_uniform;
                Vec2d zeroFormPos = spos - h_uniform/((double)(rho.ni/ni))*Vec2d(0.5);
                _v(i, j) += dt*(-alpha_buoyancy*sampleField(zeroFormPos, rho, true) + beta_buoyancy*sampleField(zeroFormPos, temperature, true));
            }
        }
    });
}

void COFLIPSolver2D::pressureProjectVelField()
{
    // NOTE: Make sure to call this function on vel fields that are fluxes.
    double tol = TOLERANCE;
    if (!use_pressure_solver && sim_scheme == Scheme::CO_FLIP) {
        projectionWithVort(tol);
    } else {
        projection(tol, use_neumann_boundary);
    }
}

void COFLIPSolver2D::buildMultiGridWithVort()
{
    std::cout << "building matrices..." << std::endl;
    //build the matrix
    //we are assuming a a whole fluid domain
    {
        tbb::enumerable_thread_specific<std::vector<Eigen::Triplet<double> > > parallelTripletList_dmatrix;
        tbb::enumerable_thread_specific<std::vector<Eigen::Triplet<double> > > parallelTripletList_starflux;
            
        tbb::parallel_for(tbb::blocked_range<int>(0,nF,TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
        {
            auto& localTripletList_dmatrix = parallelTripletList_dmatrix.local();
            auto& localTripletList_starflux = parallelTripletList_starflux.local();

            for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
                bool is_x_dir = tIdx < (ni+1)*nj;
                int j = is_x_dir ? tIdx / (ni+1) : (tIdx - (ni+1)*nj) / ni;
                int i = is_x_dir ? tIdx % (ni+1) : (tIdx - (ni+1)*nj) % ni;
                if (is_x_dir) {
                    if (boundaryMask_nodes(i,j) == 0) {
                        localTripletList_dmatrix.emplace_back(tIdx, j*(ni+1)+i, -1.0);
                    }
                    if (boundaryMask_nodes(i,j+1) == 0) {
                        localTripletList_dmatrix.emplace_back(tIdx, (j+1)*(ni+1)+i, 1.0);
                    }
                } else {
                    if (boundaryMask_nodes(i,j) == 0) {
                        localTripletList_dmatrix.emplace_back(tIdx, j*(ni+1)+i, 1.0);
                    }
                    if (boundaryMask_nodes(i+1,j) == 0) {
                        localTripletList_dmatrix.emplace_back(tIdx, j*(ni+1)+(i+1), -1.0);
                    }
                }
                
                int idx = is_x_dir ? i : j;
                int nidx = is_x_dir ? ni : nj;
                int other_idx = is_x_dir ? j : i;
                int other_nidx = is_x_dir ? nj : ni;
                if (sim_scheme == Scheme::CO_FLIP) {
                    if (use_DEC_diagonal_hodge_star) {
                        double factor = bs_p == 3 ? ((idx == 0 || idx == nidx) ? 1./3. : ((idx == 1 || idx == nidx-1) ? 0.5 : ((idx == 2 || idx == nidx-2) ? 5./6. : 1.0))) / ((other_idx == 0 || other_idx == other_nidx-1) ? 1./3. : ((other_idx == 1 || other_idx == other_nidx-2) ? 2./3. : 1.0)) : (bs_p == 2 ? ((idx == 0 || idx == nidx) ? 0.5 : ((idx == 1 || idx == nidx-1) ? 0.75 : 1.0)) / ((other_idx == 0 || other_idx == other_nidx-1) ? 0.5 : 1.0) : 1.0);
                        localTripletList_starflux.emplace_back(tIdx, tIdx, factor);
                    } else {
                        std::vector<std::array<double, 7> > Bs = bs_p == 3 ? B3s : (bs_p == 2 ? B2s : B1s);
                        std::vector<std::array<double, 7> > Ds = bs_p == 3 ? D2s : (bs_p == 2 ? D1s : D0s);
                        std::array<double, 7> Barray = Bs[std::min(std::min(2*bs_p-1, idx), nidx-idx)];
                        std::array<double, 7> Darray = Ds[std::min(std::min(2*(bs_p-1), other_idx), (other_nidx-1)-other_idx)];
                        if (nidx-idx < 2*bs_p-1) std::reverse(Barray.begin(), Barray.end());
                        if ((other_nidx-1)-other_idx < (2*(bs_p-1))) std::reverse(Darray.begin(), Darray.end());
                        for (int l = -bs_p; l <= bs_p; ++l) {
                            for (int k = -(bs_p-1); k <= (bs_p-1); ++k) {
                                int jdx = idx + l;
                                int other_jdx = other_idx + k;
                                if (jdx >= 0 && jdx <= nidx && other_jdx >=0 && other_jdx < other_nidx) {
                                    int tJdx = tIdx + l * (is_x_dir ? 1 : ni) + k * (is_x_dir ? (ni+1) : 1);
                                    double factor = Barray[l+3]*Darray[k+3];
                                    localTripletList_starflux.emplace_back(tIdx, tJdx, factor);
                                }
                            }
                        }
                    }
                } else {
                    localTripletList_starflux.emplace_back(tIdx, tIdx, 1.0);
                }
            }
        });
        std::vector<Eigen::Triplet<double> > tripletList_dmatrix;
        std::vector<Eigen::Triplet<double> > tripletList_starflux;
        HELPER::mergeLocalThreadVectors(tripletList_dmatrix, parallelTripletList_dmatrix);
        HELPER::mergeLocalThreadVectors(tripletList_starflux, parallelTripletList_starflux);

        Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> dmatrix(nF, nV);
        dmatrix.setFromTriplets(tripletList_dmatrix.begin(), tripletList_dmatrix.end());
        dtranspose_matrix_curl = dmatrix.transpose();

        starflux_matrix.resize(nF,nF);
        starflux_matrix.setFromTriplets(tripletList_starflux.begin(), tripletList_starflux.end());
        if (do_mass_lumping) {
            HELPER::rowSumMatrix(starflux_matrix, starflux_matrix);
        }
        starflux_cg.compute(starflux_matrix);
        std::cout << "did starflux_cg succeed? " << starflux_cg.info() << " ; 0<-success, 1<-not SPD" << std::endl;
    }

    {
        std::cout << "adding ones to the diagonal..." << std::endl;
        Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> streamfunction_laplacian_matrix = dtranspose_matrix_curl * starflux_matrix * dtranspose_matrix_curl.transpose();
        Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> almostIdentityMatrix(nV,nV);
        almostIdentityMatrix.reserve(Eigen::VectorXi::Constant(nV,1));
        tbb::parallel_for(tbb::blocked_range<int>(0,nV, TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
        {
            for (int i = r.begin(); i < r.end(); ++i) {
                bool have_diagonal = false;
                for(typename Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t>::InnerIterator it(streamfunction_laplacian_matrix,i); it; ++it) {
                    if (it.row() == it.col()) {
                        have_diagonal = true;
                        break;
                    }
                }
                if (!have_diagonal) {
                    almostIdentityMatrix.insert(i,i) = 1.;
                }
            }
        });
        almostIdentityMatrix.makeCompressed();
        streamfunction_laplacian_matrix = streamfunction_laplacian_matrix + almostIdentityMatrix;
        streamfunction_laplacian_matrix.makeCompressed();
        std::cout << "adding ones to the diagonal is done." << std::endl;

        mgLevelGenerator.generateLevelsGalerkinCoarsening2D(A_L_curl, R_L_curl, P_L_curl, S_L_curl, total_level_curl, streamfunction_laplacian_matrix, (ni+1), (nj+1));
        std::cout << "done building SV-laplacian matrix!" << std::endl;
    }

    {
        tbb::enumerable_thread_specific<std::vector<Eigen::Triplet<double> > > parallelTripletList_starvort;
        tbb::parallel_for(tbb::blocked_range2d<int,int>{0,ni+1,1,0,nj+1,1}, [&](const tbb::blocked_range2d<int,int> &r)
        {
            int ie = r.rows().end();
            int je = r.cols().end();
            auto& localTripletList_starvort = parallelTripletList_starvort.local();
            for (int j = r.cols().begin(); j < je; ++j) {
                for (int i = r.rows().begin(); i < ie; ++i) {
                    int tIdx = j*(ni+1) + i;
                    int idx = i;
                    int nidx = ni;
                    int other_idx = j;
                    int other_nidx = nj;
                    if (sim_scheme == Scheme::CO_FLIP) {
                        if (use_DEC_diagonal_hodge_star) {
                            // idk yet
                            localTripletList_starvort.emplace_back(tIdx, tIdx, 1.0);
                        } else {
                            std::vector<std::array<double, 7> > Bs = bs_p == 3 ? B3s : (bs_p == 2 ? B2s : B1s);
                            std::array<double, 7> Barray1 = Bs[std::min(std::min(2*bs_p-1, idx), nidx-idx)];
                            std::array<double, 7> Barray2 = Bs[std::min(std::min(2*bs_p-1, other_idx), other_nidx-other_idx)];
                            if (nidx-idx < 2*bs_p-1) std::reverse(Barray1.begin(), Barray1.end());
                            if (other_nidx-other_idx < 2*bs_p-1) std::reverse(Barray2.begin(), Barray2.end());
                            for (int l = -bs_p; l <= bs_p; ++l) {
                                for (int k = -bs_p; k <= bs_p; ++k) {
                                    int jdx = idx + l;
                                    int other_jdx = other_idx + k;
                                    if (jdx >= 0 && jdx <= nidx && other_jdx >=0 && other_jdx <= other_nidx) {
                                        int tJdx = tIdx + l + k * (ni+1);
                                        double factor = Barray1[l+3]*Barray2[k+3];
                                        localTripletList_starvort.emplace_back(tIdx, tJdx, factor);
                                    }
                                }
                            }
                        }
                    } else {
                        localTripletList_starvort.emplace_back(tIdx, tIdx, 1.0);
                    }
                }
            }
        });
        std::vector<Eigen::Triplet<double> > tripletList_starvort;
        HELPER::mergeLocalThreadVectors(tripletList_starvort, parallelTripletList_starvort);
        starvort_matrix.resize(nV,nV);
        starvort_matrix.setFromTriplets(tripletList_starvort.begin(), tripletList_starvort.end());
        if (do_mass_lumping) {
            HELPER::rowSumMatrix(starvort_matrix, starvort_matrix);
        }
        starvort_cg.compute(starvort_matrix);
        std::cout << "did starvort_cg succeed? " << starvort_cg.info() << " ; 0<-success, 1<-not SPD" << std::endl;
    }

    {
        tbb::enumerable_thread_specific<std::vector<Eigen::Triplet<double> > > parallelTripletList_starpressure;
        tbb::parallel_for(tbb::blocked_range2d<int,int>{0,ni,1,0,nj,1}, [&](const tbb::blocked_range2d<int,int> &r)
        {
            int ie = r.rows().end();
            int je = r.cols().end();
            auto& localTripletList_starpressure = parallelTripletList_starpressure.local();
            for (int j = r.cols().begin(); j < je; ++j) {
                for (int i = r.rows().begin(); i < ie; ++i) {
                    int tIdx = j*ni + i;
                    int idx = i;
                    int nidx = ni;
                    int other_idx = j;
                    int other_nidx = nj;
                    if (sim_scheme == Scheme::CO_FLIP) {
                        if (use_DEC_diagonal_hodge_star) {
                            // idk yet
                            localTripletList_starpressure.emplace_back(tIdx, tIdx, 1.0);
                        } else {
                            std::vector<std::array<double, 7> > Ds = bs_p == 3 ? D2s : (bs_p == 2 ? D1s : D0s);
                            std::array<double, 7> Darray1 = Ds[std::min(std::min(2*(bs_p-1), idx), (nidx-1)-idx)];
                            std::array<double, 7> Darray2 = Ds[std::min(std::min(2*(bs_p-1), other_idx), (other_nidx-1)-other_idx)];
                            if ((nidx-1)-idx < (2*(bs_p-1))) std::reverse(Darray1.begin(), Darray1.end());
                            if ((other_nidx-1)-other_idx < (2*(bs_p-1))) std::reverse(Darray2.begin(), Darray2.end());
                            for (int l = -(bs_p-1); l <= (bs_p-1); ++l) {
                                for (int k = -(bs_p-1); k <= (bs_p-1); ++k) {
                                    int jdx = idx + l;
                                    int other_jdx = other_idx + k;
                                    if (jdx >= 0 && jdx < nidx && other_jdx >=0 && other_jdx < other_nidx) {
                                        int tJdx = tIdx + l + k * ni;
                                        double factor = Darray1[l+3]*Darray2[k+3];
                                        localTripletList_starpressure.emplace_back(tIdx, tJdx, factor);
                                    }
                                }
                            }
                        }
                    } else {
                        localTripletList_starpressure.emplace_back(tIdx, tIdx, 1.0);
                    }
                }
            }
        });
        std::vector<Eigen::Triplet<double> > tripletList_starpressure;
        HELPER::mergeLocalThreadVectors(tripletList_starpressure, parallelTripletList_starpressure);
        starpressure_matrix.resize(nC,nC);
        starpressure_matrix.setFromTriplets(tripletList_starpressure.begin(), tripletList_starpressure.end());
        if (do_mass_lumping) {
            HELPER::rowSumMatrix(starpressure_matrix, starpressure_matrix);
        }
        starpressure_cg.compute(starpressure_matrix);
        std::cout << "did starpressure_cg succeed? " << starpressure_cg.info() << " ; 0<-success, 1<-not SPD" << std::endl;
    }
}

void COFLIPSolver2D::projectionWithVort(double tol)
{
    takeDualwrtStar(u, v, true, false);

    calculateCurl();

    rhs = vorts * h;
    double res_out; int iter_out;
    bool converged = AMGPCGSolvePrebuilt2D(A_L_curl[0],rhs,streamfunction,A_L_curl,R_L_curl,P_L_curl,S_L_curl,total_level_curl,tol,MAX_ITERATIONS,res_out,iter_out,ni+1,nj+1, false, true);
    if (!converged)
        std::cout << "WARNING: Streamfunction-vorticity solve failed!************************************************" << std::endl;
    std::cout << "#iteration:      " << iter_out << std::endl;
    std::cout << "estimated error: " << res_out << std::endl;

    // set dirichlet bdy conditions
    tbb::parallel_for(tbb::blocked_range2d<int,int>{0,ni+1,1,0,nj+1,1}, [&](const tbb::blocked_range2d<int,int> &r)
    {
        int ie = r.rows().end();
        int je = r.cols().end();
        for (int j = r.cols().begin(); j < je; ++j) {
            for (int i = r.rows().begin(); i < ie; ++i) {
                if (boundaryMask_nodes(i,j) != 0) {
                    int tIdx = j*(ni+1)+i;
                    streamfunction[tIdx] = 0;
                }
            }
        }
    });

    fluxes = dtranspose_matrix_curl.transpose() * streamfunction;
    tbb::parallel_for(tbb::blocked_range<int>(0,nF,TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
    {
        for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
            bool is_x_dir = tIdx < (ni+1)*nj;
            int j = is_x_dir ? tIdx / (ni+1) : (tIdx - (ni+1)*nj) / ni;
            int i = is_x_dir ? tIdx % (ni+1) : (tIdx - (ni+1)*nj) % ni;
            if (is_x_dir) {
                u(i,j) = fluxes[tIdx];
            } else {
                v(i,j) = fluxes[tIdx];
            }
        }
    });

    applyVelocityBoundary();

    takeDualwrtStar(u, v, false, false);
    std::cout  << GREEN << "fluxes.norm = " << fluxes.norm() << ", circulations.norm = " << circulations.norm()  << ", vorts.norm = " << vorts.norm() << RESET << std::endl;

    double avg_div = 0.0;
    double max_div = -1.0;
    for(int tIdx = 0; tIdx < nC; tIdx++)
    {
        int i = tIdx%ni;
        int j = tIdx / ni;
        double div = std::abs(u(i+1,j) - u(i,j) + v(i,j+1) - v(i,j)) / h;
        avg_div += div;
        max_div = std::max(div, max_div);
    }
    std::cout << RED << "max_div = " << max_div << ", avg_div = " << avg_div/(double)(nC) << RESET << std::endl;max_div_on_grid = max_div; 
}

void COFLIPSolver2D::projection(double tol, bool PURE_NEUMANN)
{
    for (int count = 0; count < projection_repeat_count; count++)
    {
        //build rhs;
        tbb::parallel_for(tbb::blocked_range<int>(0,nF,TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
        {
            for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
                bool is_x_dir = tIdx < (ni+1)*nj;
                int j = is_x_dir ? tIdx / (ni+1) : (tIdx - (ni+1)*nj) / ni;
                int i = is_x_dir ? tIdx % (ni+1) : (tIdx - (ni+1)*nj) % ni;
                fluxes[tIdx] = is_x_dir ? u(i,j) : v(i,j);
            }
        });
        rhs = dtranspose_matrix * fluxes;
        std::cout  << GREEN << "rhs.norm = " << rhs.norm() << ", rhs.mean = " << rhs.mean() << RESET << std::endl;

        double res_out; int iter_out;
        pressure.setZero();
        bool converged = false;
        if (use_pressure_solver && sim_scheme == Scheme::CO_FLIP) {
            Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> dtranspose_matrix_transpose = dtranspose_matrix.transpose();
            Eigen::VectorXd guess_fluxes = fluxes;
            auto multiply_func = [&](Eigen::VectorXd& output, const Eigen::VectorXd& input) {
                Eigen::VectorXd input_circulation = dtranspose_matrix_transpose * input;
                starflux_cg.setTolerance(TOLERANCE);
                starflux_cg.setMaxIterations(MAX_ITERATIONS);
                guess_fluxes = starflux_cg.solveWithGuess(input_circulation, guess_fluxes);
                output = dtranspose_matrix * guess_fluxes + almostIdentityMatrixCells * input;
            };
            converged = AMGPCGSolvePrebuilt2D<double>(multiply_func,rhs,pressure,A_L,R_L,P_L,S_L,total_level,tol,MAX_ITERATIONS,res_out,iter_out,ni,nj, PURE_NEUMANN);
        } else {
            converged = AMGPCGSolvePrebuilt2D(A_L[0],rhs,pressure,A_L,R_L,P_L,S_L,total_level,tol,MAX_ITERATIONS,res_out,iter_out,ni,nj, PURE_NEUMANN);
        }
        if (!converged)
            std::cout << "WARNING: Pressure projection solve failed!************************************************" << std::endl;
        std::cout << "#iteration:      " << iter_out << std::endl;
        std::cout << "estimated error: " << res_out << std::endl;

        Eigen::VectorXd delta_circulation = dtranspose_matrix.transpose() * pressure;
        starflux_cg.setTolerance(TOLERANCE);
        starflux_cg.setMaxIterations(MAX_ITERATIONS);
        Eigen::VectorXd delta_fluxes = starflux_cg.solve(delta_circulation);

        tbb::parallel_for(tbb::blocked_range<int>(0,nF,TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
        {
            for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
                bool is_x_dir = tIdx < (ni+1)*nj;
                int j = is_x_dir ? tIdx / (ni+1) : (tIdx - (ni+1)*nj) / ni;
                int i = is_x_dir ? tIdx % (ni+1) : (tIdx - (ni+1)*nj) % ni;
                if (is_x_dir) {
                    u(i,j) -= delta_fluxes(tIdx);
                } else {
                    v(i,j) -= delta_fluxes(tIdx);
                }
            }
        });

        applyVelocityBoundary();

        if (res_out < tol) break;
    }

    calculateCurl(true);

    std::cout  << GREEN << "fluxes.norm = " << fluxes.norm() << ", circulations.norm = " << circulations.norm()  << ", vorts.norm = " << vorts.norm() << RESET << std::endl;

    double avg_div = 0.0;
    double max_div = -1.0;
    for(int tIdx = 0; tIdx < nC; tIdx++)
    {
        int i = tIdx%ni;
        int j = tIdx / ni;
        double div = std::abs(u(i+1,j) - u(i,j) + v(i,j+1) - v(i,j)) / h;
        avg_div += div;
        max_div = std::max(div, max_div);
    }
    std::cout << RED << "max_div = " << max_div << ", avg_div = " << avg_div/(double)(nC) << RESET << std::endl;
    max_div_on_grid = max_div;
}

double COFLIPSolver2D::computeFTLE(double& det, const Eigen::Matrix2d& pullback)
{
    det = pullback.determinant();
    Eigen::JacobiSVD<Eigen::Matrix2d, Eigen::ComputeThinU | Eigen::ComputeThinV> svd(pullback);
    double max_sv = svd.singularValues()[0];
    double FTLE = std::log(max_sv);
    return FTLE;
}

void COFLIPSolver2D::solveInterpDaggerVorticity(bool do_pressureSolve) {
    bool doBSpline = true;
    if (bs_p == 1) {
        doBSpline = false;
    }

    // splat particle properties to grid
    int numParticles = cParticles.size();
    Eigen::VectorXd point_vort(numParticles);
    Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> interp_matrix(numParticles, nV);
    interp_matrix.reserve(Eigen::VectorXi::Constant(numParticles, doBSpline ? (bs_p+1)*(bs_p+1) : 2*2));
    if (doBSpline) {
        tbb::parallel_for(tbb::blocked_range<int>(0,cParticles.size(),TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
        {
            for (int i = r.begin(); i < r.end(); ++i) {
                CmapParticles p = cParticles[i];
                Vec2d pos = p.pos_temp;
                int ii = std::floor(pos[0]/h);
                int jj = std::floor(pos[1]/h);
                double alpha = pos[0]/h - (double)ii, beta = pos[1]/h - (double)jj;
                for(int jjj=jj;jjj<=jj+bs_p;jjj++)for(int iii=ii;iii<=ii+bs_p;iii++)
                {
                    double w = 0;
                    if (bs_p == 2) {
                        w = 
                            p.kernel2(ii == nip-1 ? 1.0 - alpha : alpha, 
                                        ii == nip-1 ? bs_p - (iii-ii) : iii-ii, 
                                        ii == 0 || ii == nip-1) *
                            p.kernel2(jj == njp-1 ? 1.0 - beta : beta, 
                                        jj == njp-1 ? bs_p - (jjj-jj) : jjj-jj, 
                                        jj == 0 || jj == njp-1);
                    } else {
                        w =
                            p.kernel3((ii == nip-1 || ii == nip-2) ? 1.0 - alpha : alpha, 
                                        (ii == nip-1 || ii == nip-2) ? bs_p - (iii-ii) : iii-ii, 
                                        (ii == 0 || ii == nip-1) ? 2 : ((ii == 1 || ii == nip-2) ? 1 : 0)) *
                            p.kernel3((jj == njp-1 || jj == njp-2) ? 1.0 - beta : beta,
                                        (jj == njp-1 || jj == njp-2) ? bs_p - (jjj-jj) : jjj-jj, 
                                        (jj == 0 || jj == njp-1) ? 2 : ((jj == 1 || jj == njp-2) ? 1 : 0));
                    }
                    int idx = jjj * (ni+1) + iii;
                    interp_matrix.insert(i, idx) = w * std::sqrt(p.volume);
                    point_vort(i) = p.vel_temp[0] * std::sqrt(p.volume);
                }
            }
        });
    } else {
        tbb::parallel_for(tbb::blocked_range<int>(0,numParticles,TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
        {
            for (int i = r.begin(); i < r.end(); ++i) {
                CmapParticles p = cParticles[i];
                Vec2d pos = p.pos_temp;
                pos[0] = std::min(std::max(0+TOLERANCE, pos[0]), ((double)ni)*h-TOLERANCE);
                pos[1] = std::min(std::max(0+TOLERANCE, pos[1]), ((double)nj)*h-TOLERANCE);
                int ii, jj;
                ii = std::floor(pos[0]/h);
                jj = std::floor(pos[1]/h);
                for(int jjj=jj;jjj<=jj+1;jjj++)for(int iii=ii;iii<=ii+1;iii++)
                {
                    Vec2d gpos = Vec2d(iii,jjj)*h + Vec2d(0,0.5)*h;
                    double w = p.kernel((pos[0] - gpos[0])/h)*p.kernel((pos[1] - gpos[1])/h);
                    int idx = jjj * (ni+1) + iii;
                    interp_matrix.insert(i, idx) = w * std::sqrt(p.volume);
                    point_vort(i) = p.vel_temp[0] * std::sqrt(p.volume);
                }
            }
        });
    }
    interp_matrix.makeCompressed();

    Eigen::VectorXd result_0form_dagger(nV);
    if (do_pressureSolve) {
        result_0form_dagger.setZero();
    } else {
        result_0form_dagger = prev_vorts0form;
    }
    std::cout << GREEN << "point_vort.norm = " << point_vort.norm() << RESET << std::endl;
    if (do_mass_lumping) {
        Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> fullInterpMat = interp_matrix.transpose() * interp_matrix;
        HELPER::rowSumMatrix(fullInterpMat, fullInterpMat);
        Eigen::ConjugateGradient<Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> > dummy_cg;
        dummy_cg.compute(fullInterpMat);
        result_0form_dagger = dummy_cg.solveWithGuess(interp_matrix.transpose() * point_vort, result_0form_dagger);
    } else {
        int iterations;
        double tolerance;
        bool success;
        Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> fullInterpMat(nF,nF);
        fullInterpMat.setZero();
        fullInterpMat.selfadjointView<Eigen::Lower>().rankUpdate(interp_matrix.transpose());
        Eigen::IncompleteCholesky<double, Eigen::Lower, Eigen::AMDOrdering<std::ptrdiff_t> > vort_precond(fullInterpMat);
        std::cout << "vort_precond.info(): " << vort_precond.info() << " 0: success, 1: numerical issue" << std::endl;
        if (vort_precond.info() == Eigen::Success) {
            success = PLSCGSolve(interp_matrix, point_vort, result_0form_dagger, vort_precond, (double)TOLERANCE, MAX_ITERATIONS, tolerance, iterations);
        } else {
            Eigen::LeastSquareDiagonalPreconditioner<double> diagPrecond;
            diagPrecond.compute(interp_matrix);
            success = PLSCGSolve(interp_matrix, point_vort, result_0form_dagger, diagPrecond, (double)TOLERANCE, MAX_ITERATIONS, tolerance, iterations);
        }
        std::cout << "#iteration:      " << iterations << std::endl;
        std::cout << "estimated error: " << tolerance << std::endl;
        if (!success) {
            printf("WARNING: Momentum map least squares solve failed!************************************************\n");
        }
    }
    std::cout << GREEN << "result_0form_dagger.norm = " << result_0form_dagger.norm() << RESET << std::endl;
    if (do_pressureSolve) {
        pressure_field_0form = result_0form_dagger;
    } else {
        prev_vorts0form = result_0form_dagger;
    }
}

std::tuple<double, double> COFLIPSolver2D::getCasimirsAtCustomRes(int res_amp) {
    int amped_ni = res_amp * ni;
    int amped_nj = res_amp * nj;
    int amped_nip = amped_ni+1-bs_p;
    int amped_njp = amped_nj+1-bs_p;
    double amped_h = (h * (double)nip)/(amped_njp);
    int amped_nF = (amped_ni+1)*amped_nj + amped_ni*(amped_nj+1);
    int amped_nV = (amped_ni+1)*(amped_nj+1);
    // splat particle properties to grid
    int numParticles = cParticles.size();
    Eigen::VectorXd point_circ(2*numParticles);
    Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> interp_matrix(2*numParticles, amped_nF);
    interp_matrix.reserve(Eigen::VectorXi::Constant(2*numParticles, (bs_p)*(bs_p+1)));
    tbb::parallel_for(tbb::blocked_range<int>(0,cParticles.size(),TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
    {
        for (int i = r.begin(); i < r.end(); ++i) {
            CmapParticles p = cParticles[i];
            Vec2d pos = p.pos_current;
            int ii = std::floor(pos[0]/amped_h);
            int jj = std::floor(pos[1]/amped_h);
            double alpha = pos[0]/amped_h - (double)ii, beta = pos[1]/amped_h - (double)jj;
            for(int jjj=jj;jjj<=jj+bs_p-1;jjj++)for(int iii=ii;iii<=ii+bs_p;iii++)
            {
                double w = 0;
                if (bs_p == 2) {
                    w = 
                        p.kernel2(ii == amped_nip-1 ? 1.0 - alpha : alpha, 
                                    ii == amped_nip-1 ? bs_p - (iii-ii) : iii-ii, 
                                    ii == 0 || ii == amped_nip-1) *
                        p.kernel1prime(jj == amped_njp-1 ? 1.0 - beta : beta, 
                                    jj == amped_njp-1 ? bs_p-1 - (jjj-jj) : jjj-jj, 
                                    jj == 0 || jj == amped_njp-1);
                } else {
                    w =
                        p.kernel3((ii == amped_nip-1 || ii == amped_nip-2) ? 1.0 - alpha : alpha, 
                                    (ii == amped_nip-1 || ii == amped_nip-2) ? bs_p - (iii-ii) : iii-ii, 
                                    (ii == 0 || ii == amped_nip-1) ? 2 : ((ii == 1 || ii == amped_nip-2) ? 1 : 0)) *
                        p.kernel2prime((jj == amped_njp-1 || jj == amped_njp-2) ? 1.0 - beta : beta,
                                    (jj == amped_njp-1 || jj == amped_njp-2) ? bs_p-1 - (jjj-jj) : jjj-jj, 
                                    (jj == 0 || jj == amped_njp-1) ? 2 : ((jj == 1 || jj == amped_njp-2) ? 1 : 0));
                }
                int idx = jjj * (amped_ni+1) + iii;
                interp_matrix.insert(i, idx) = w * std::sqrt(p.volume * res_amp*res_amp);
                point_circ(i) = p.vel[0] * std::sqrt(p.volume * res_amp*res_amp);
            }

            for(int jjj=jj;jjj<=jj+bs_p;jjj++)for(int iii=ii;iii<=ii+bs_p-1;iii++)
            {
                double w = 0;
                if (bs_p == 2) {
                    w = 
                        p.kernel1prime(ii == amped_nip-1 ? 1.0 - alpha : alpha, 
                                    ii == amped_nip-1 ? bs_p-1 - (iii-ii) : iii-ii,
                                    ii == 0 || ii == amped_nip-1) *
                        p.kernel2(jj == amped_njp-1 ? 1.0 - beta : beta,
                                    jj == amped_njp-1 ? bs_p - (jjj-jj) : jjj-jj,
                                    jj == 0 || jj == amped_njp-1);
                } else {
                    w =
                        p.kernel2prime((ii == amped_nip-1 || ii == amped_nip-2) ? 1.0 - alpha : alpha, 
                                    (ii == amped_nip-1 || ii == amped_nip-2) ? bs_p-1 - (iii-ii) : iii-ii, 
                                    (ii == 0 || ii == amped_nip-1) ? 2 : ((ii == 1 || ii == amped_nip-2) ? 1 : 0)) *
                        p.kernel3((jj == amped_njp-1 || jj == amped_njp-2) ? 1.0 - beta : beta,
                                    (jj == amped_njp-1 || jj == amped_njp-2) ? bs_p - (jjj-jj) : jjj-jj, 
                                    (jj == 0 || jj == amped_njp-1) ? 2 : ((jj == 1 || jj == amped_njp-2) ? 1 : 0));
                }
                int idx = (amped_ni+1)*amped_nj + jjj * amped_ni + iii;
                interp_matrix.insert(numParticles + i, idx) = w * std::sqrt(p.volume * res_amp*res_amp);
                point_circ(numParticles + i) = p.vel[1] * std::sqrt(p.volume * res_amp*res_amp);
            }
        }
    });
    interp_matrix.makeCompressed();

    std::cout << GREEN << "point_circ.norm = " << point_circ.norm() << RESET << std::endl;

    int iterations;
    double tolerance;
    bool success;
    Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> fullInterpMat(amped_nF,amped_nF);
    fullInterpMat.setZero();
    fullInterpMat.selfadjointView<Eigen::Lower>().rankUpdate(interp_matrix.transpose());
    Eigen::IncompleteCholesky<double, Eigen::Lower, Eigen::AMDOrdering<std::ptrdiff_t> > amped_precond(fullInterpMat);
    std::cout << "amped_precond.info(): " << amped_precond.info() << " 0: success, 1: numerical issue" << std::endl;
    Eigen::VectorXd amped_fluxes(amped_nF);
    amped_fluxes.setZero();
    if (amped_precond.info() == Eigen::Success) {
        success = PLSCGSolve(interp_matrix, point_circ, amped_fluxes, amped_precond, (double)TOLERANCE, MAX_ITERATIONS, tolerance, iterations);
    } else {
        Eigen::LeastSquareDiagonalPreconditioner<double> diagPrecond;
        diagPrecond.compute(interp_matrix);
        success = PLSCGSolve(interp_matrix, point_circ, amped_fluxes, diagPrecond, (double)TOLERANCE, MAX_ITERATIONS, tolerance, iterations);
    }
    std::cout << "#iteration:      " << iterations << std::endl;
    std::cout << "estimated error: " << tolerance << std::endl;
    if (!success) {
        printf("WARNING: Momentum map least squares solve failed!************************************************\n");
    }

    tbb::enumerable_thread_specific<std::vector<Eigen::Triplet<double> > > parallelTripletList_amped_dmatrix;
    Eigen::VectorXd amped_circulations(amped_nF);
    amped_circulations.setZero();
    tbb::parallel_for(tbb::blocked_range<int>(0,amped_nF,TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
    {
        auto& localTripletList_amped_dmatrix = parallelTripletList_amped_dmatrix.local();
        for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
            bool is_x_dir = tIdx < (amped_ni+1)*amped_nj;
            int j = is_x_dir ? tIdx / (amped_ni+1) : (tIdx - (amped_ni+1)*amped_nj) / amped_ni;
            int i = is_x_dir ? tIdx % (amped_ni+1) : (tIdx - (amped_ni+1)*amped_nj) % amped_ni;
            if (is_x_dir) {
                if (i != 0 && j != 0 && i != amped_ni) {
                    localTripletList_amped_dmatrix.emplace_back(tIdx, j*(amped_ni+1)+i, -1.0);
                }
                if (i != 0 && j != (amped_nj-1) && i != amped_ni) {
                    localTripletList_amped_dmatrix.emplace_back(tIdx, (j+1)*(amped_ni+1)+i, 1.0);
                }
            } else {
                if (i != 0 && j != 0 && j != amped_nj) {
                    localTripletList_amped_dmatrix.emplace_back(tIdx, j*(amped_ni+1)+i, 1.0);
                }
                if (i != (amped_ni-1) && j != 0 && j != amped_nj) {
                    localTripletList_amped_dmatrix.emplace_back(tIdx, j*(amped_ni+1)+(i+1), -1.0);
                }
            }
            int idx = is_x_dir ? i : j;
            int nidx = is_x_dir ? amped_ni : amped_nj;
            int other_idx = is_x_dir ? j : i;
            int other_nidx = is_x_dir ? amped_nj : amped_ni;
            if (sim_scheme == Scheme::CO_FLIP) {
                if (use_DEC_diagonal_hodge_star) {
                    double factor = bs_p == 3 ? ((idx == 0 || idx == nidx) ? 1./3. : ((idx == 1 || idx == nidx-1) ? 0.5 : ((idx == 2 || idx == nidx-2) ? 5./6. : 1.0))) / ((other_idx == 0 || other_idx == other_nidx-1) ? 1./3. : ((other_idx == 1 || other_idx == other_nidx-2) ? 2./3. : 1.0)) : (bs_p == 2 ? ((idx == 0 || idx == nidx) ? 0.5 : ((idx == 1 || idx == nidx-1) ? 0.75 : 1.0)) / ((other_idx == 0 || other_idx == other_nidx-1) ? 0.5 : 1.0) : 1.0);
                    amped_circulations(tIdx) += factor * amped_fluxes(tIdx);
                } else {
                    std::vector<std::array<double, 7> > Bs = bs_p == 3 ? B3s : (bs_p == 2 ? B2s : B1s);
                    std::vector<std::array<double, 7> > Ds = bs_p == 3 ? D2s : (bs_p == 2 ? D1s : D0s);
                    std::array<double, 7> Barray = Bs[std::min(std::min(2*bs_p-1, idx), nidx-idx)];
                    std::array<double, 7> Darray = Ds[std::min(std::min(2*(bs_p-1), other_idx), (other_nidx-1)-other_idx)];
                    if (nidx-idx < 2*bs_p-1) std::reverse(Barray.begin(), Barray.end());
                    if ((other_nidx-1)-other_idx < (2*(bs_p-1))) std::reverse(Darray.begin(), Darray.end());
                    for (int l = -bs_p; l <= bs_p; ++l) {
                        for (int k = -(bs_p-1); k <= (bs_p-1); ++k) {
                            int jdx = idx + l;
                            int other_jdx = other_idx + k;
                            if (jdx >= 0 && jdx <= nidx && other_jdx >=0 && other_jdx < other_nidx) {
                                int tJdx = tIdx + l * (is_x_dir ? 1 : amped_ni) + k * (is_x_dir ? (amped_ni+1) : 1);
                                double factor = Barray[l+3]*Darray[k+3];
                                amped_circulations(tIdx) += factor * amped_fluxes(tJdx);
                            }
                        }
                    }
                }
            } else {
                amped_circulations(tIdx) += amped_fluxes(tIdx);
            }
        }
    });
    std::vector<Eigen::Triplet<double> > tripletList_amped_dmatrix;
    HELPER::mergeLocalThreadVectors(tripletList_amped_dmatrix, parallelTripletList_amped_dmatrix);

    Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> amped_dmatrix(amped_nF, amped_nV);
    amped_dmatrix.setFromTriplets(tripletList_amped_dmatrix.begin(), tripletList_amped_dmatrix.end());
    Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> amped_dtranspose_matrix_curl = amped_dmatrix.transpose();

    Eigen::VectorXd amped_vorts = amped_dtranspose_matrix_curl * amped_circulations / amped_h;

    tbb::enumerable_thread_specific<std::vector<Eigen::Triplet<double> > > parallelTripletList_amped_starvort;
    tbb::parallel_for(tbb::blocked_range2d<int,int>{0,amped_ni+1,1,0,amped_nj+1,1}, [&](const tbb::blocked_range2d<int,int> &r)
    {
        int ie = r.rows().end();
        int je = r.cols().end();
        auto& localTripletList_amped_starvort = parallelTripletList_amped_starvort.local();
        for (int j = r.cols().begin(); j < je; ++j) {
            for (int i = r.rows().begin(); i < ie; ++i) {
                int tIdx = j*(amped_ni+1) + i;
                int idx = i;
                int nidx = amped_ni;
                int other_idx = j;
                int other_nidx = amped_nj;
                if (sim_scheme == Scheme::CO_FLIP) {
                    if (use_DEC_diagonal_hodge_star) {
                        // idk yet
                        localTripletList_amped_starvort.emplace_back(tIdx, tIdx, 1.0);
                    } else {
                        std::vector<std::array<double, 7> > Bs = bs_p == 3 ? B3s : (bs_p == 2 ? B2s : B1s);
                        std::array<double, 7> Barray1 = Bs[std::min(std::min(2*bs_p-1, idx), nidx-idx)];
                        std::array<double, 7> Barray2 = Bs[std::min(std::min(2*bs_p-1, other_idx), other_nidx-other_idx)];
                        if (nidx-idx < 2*bs_p-1) std::reverse(Barray1.begin(), Barray1.end());
                        if (other_nidx-other_idx < 2*bs_p-1) std::reverse(Barray2.begin(), Barray2.end());
                        for (int l = -bs_p; l <= bs_p; ++l) {
                            for (int k = -bs_p; k <= bs_p; ++k) {
                                int jdx = idx + l;
                                int other_jdx = other_idx + k;
                                if (jdx >= 0 && jdx <= nidx && other_jdx >=0 && other_jdx <= other_nidx) {
                                    int tJdx = tIdx + l + k * (amped_ni+1);
                                    double factor = Barray1[l+3]*Barray2[k+3];
                                    localTripletList_amped_starvort.emplace_back(tIdx, tJdx, factor);
                                }
                            }
                        }
                    }
                } else {
                    localTripletList_amped_starvort.emplace_back(tIdx, tIdx, 1.0);
                }
            }
        }
    });
    std::vector<Eigen::Triplet<double> > tripletList_amped_starvort;
    HELPER::mergeLocalThreadVectors(tripletList_amped_starvort, parallelTripletList_amped_starvort);
    Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> amped_starvort_matrix(amped_nV,amped_nV);
    amped_starvort_matrix.setFromTriplets(tripletList_amped_starvort.begin(), tripletList_amped_starvort.end());
    Eigen::ConjugateGradient<Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t>, Eigen::Lower|Eigen::Upper, Eigen::IncompleteCholesky<double, Eigen::Lower, Eigen::AMDOrdering<std::ptrdiff_t> > > amped_starvort_cg(amped_starvort_matrix);

    amped_starvort_cg.setTolerance(TOLERANCE);
    amped_starvort_cg.setMaxIterations(MAX_ITERATIONS);
    Eigen::VectorXd amped_vorts0form(amped_nV);
    amped_vorts0form.setZero();
    amped_vorts0form = amped_starvort_cg.solveWithGuess(amped_vorts, amped_vorts0form);
    if (amped_starvort_cg.info() == Eigen::Success) {
        std::cout << "#iteration:      " << amped_starvort_cg.iterations() << std::endl;
        std::cout << "estimated error: " << amped_starvort_cg.error() << std::endl;
    } else {
        std::cout << "amped_starvort_cg solver FAILED!!!" << std::endl;
    }

    Eigen::VectorXd onesNform = amped_starvort_matrix * Eigen::VectorXd::Constant(amped_nV, 1.0);
    double firstMomentVorticityIntegral = amped_vorts0form.transpose() * onesNform;
    firstMomentVorticityIntegral *= std::pow(amped_h, 2);
    double secondMomentVorticityIntegral = amped_vorts0form.transpose() * amped_vorts;
    secondMomentVorticityIntegral *= std::pow(amped_h, 2);
    return std::tie(firstMomentVorticityIntegral, secondMomentVorticityIntegral);
}

void COFLIPSolver2D::solveInterpDaggerVelocity(int stage_count, int currentframe, int delayed_reinit_frequency) {
    bool doBSpline = true;
    if (bs_p == 1) {
        doBSpline = false;
    }

    // splat particle properties to grid
    int numParticles = cParticles.size();
    Eigen::VectorXd point_circ(2*numParticles);
    Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> interp_matrix(2*numParticles, nF);
    interp_matrix.reserve(Eigen::VectorXi::Constant(2*numParticles, doBSpline ? (bs_p)*(bs_p+1) : 2*2));
    if (doBSpline) {
        tbb::parallel_for(tbb::blocked_range<int>(0,cParticles.size(),TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
        {
            for (int i = r.begin(); i < r.end(); ++i) {
                CmapParticles p = cParticles[i];
                Vec2d pos = p.pos_temp;
                int ii = std::floor(pos[0]/h);
                int jj = std::floor(pos[1]/h);
                double alpha = pos[0]/h - (double)ii, beta = pos[1]/h - (double)jj;
                for(int jjj=jj;jjj<=jj+bs_p-1;jjj++)for(int iii=ii;iii<=ii+bs_p;iii++)
                {
                    double w = 0;
                    if (bs_p == 2) {
                        w = 
                            p.kernel2(ii == nip-1 ? 1.0 - alpha : alpha, 
                                        ii == nip-1 ? bs_p - (iii-ii) : iii-ii, 
                                        ii == 0 || ii == nip-1) *
                            p.kernel1prime(jj == njp-1 ? 1.0 - beta : beta, 
                                        jj == njp-1 ? bs_p-1 - (jjj-jj) : jjj-jj, 
                                        jj == 0 || jj == njp-1);
                    } else {
                        w =
                            p.kernel3((ii == nip-1 || ii == nip-2) ? 1.0 - alpha : alpha, 
                                        (ii == nip-1 || ii == nip-2) ? bs_p - (iii-ii) : iii-ii, 
                                        (ii == 0 || ii == nip-1) ? 2 : ((ii == 1 || ii == nip-2) ? 1 : 0)) *
                            p.kernel2prime((jj == njp-1 || jj == njp-2) ? 1.0 - beta : beta,
                                        (jj == njp-1 || jj == njp-2) ? bs_p-1 - (jjj-jj) : jjj-jj, 
                                        (jj == 0 || jj == njp-1) ? 2 : ((jj == 1 || jj == njp-2) ? 1 : 0));
                    }
                    int idx = jjj * (ni+1) + iii;
                    interp_matrix.insert(i, idx) = w * std::sqrt(p.volume);
                    point_circ(i) = p.vel_temp[0] * std::sqrt(p.volume);
                }

                for(int jjj=jj;jjj<=jj+bs_p;jjj++)for(int iii=ii;iii<=ii+bs_p-1;iii++)
                {
                    double w = 0;
                    if (bs_p == 2) {
                        w = 
                            p.kernel1prime(ii == nip-1 ? 1.0 - alpha : alpha, 
                                        ii == nip-1 ? bs_p-1 - (iii-ii) : iii-ii,
                                        ii == 0 || ii == nip-1) *
                            p.kernel2(jj == njp-1 ? 1.0 - beta : beta,
                                        jj == njp-1 ? bs_p - (jjj-jj) : jjj-jj,
                                        jj == 0 || jj == njp-1);
                    } else {
                        w =
                            p.kernel2prime((ii == nip-1 || ii == nip-2) ? 1.0 - alpha : alpha, 
                                        (ii == nip-1 || ii == nip-2) ? bs_p-1 - (iii-ii) : iii-ii, 
                                        (ii == 0 || ii == nip-1) ? 2 : ((ii == 1 || ii == nip-2) ? 1 : 0)) *
                            p.kernel3((jj == njp-1 || jj == njp-2) ? 1.0 - beta : beta,
                                        (jj == njp-1 || jj == njp-2) ? bs_p - (jjj-jj) : jjj-jj, 
                                        (jj == 0 || jj == njp-1) ? 2 : ((jj == 1 || jj == njp-2) ? 1 : 0));
                    }
                    int idx = (ni+1)*nj + jjj * ni + iii;
                    interp_matrix.insert(numParticles + i, idx) = w * std::sqrt(p.volume);
                    point_circ(numParticles + i) = p.vel_temp[1] * std::sqrt(p.volume);
                }
            }
        });
    } else {
        tbb::parallel_for(tbb::blocked_range<int>(0,numParticles,TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
        {
            for (int i = r.begin(); i < r.end(); ++i) {
                CmapParticles p = cParticles[i];
                Vec2d pos = p.pos_temp;
                pos[0] = std::min(std::max(0+TOLERANCE, pos[0]), ((double)ni)*h-TOLERANCE);
                pos[1] = std::min(std::max(0.5*h+TOLERANCE, pos[1]), ((double)nj-0.5)*h-TOLERANCE);
                int ii, jj;
                ii = std::floor(pos[0]/h);
                jj = std::floor(pos[1]/h-0.5);
                for(int jjj=jj;jjj<=jj+1;jjj++)for(int iii=ii;iii<=ii+1;iii++)
                {
                    Vec2d gpos = Vec2d(iii,jjj)*h + Vec2d(0,0.5)*h;
                    double w = p.kernel((pos[0] - gpos[0])/h)*p.kernel((pos[1] - gpos[1])/h);
                    int idx = jjj * (ni+1) + iii;
                    interp_matrix.insert(i, idx) = w * std::sqrt(p.volume);
                    point_circ(i) = p.vel_temp[0] * std::sqrt(p.volume);
                }

                pos = p.pos_temp;
                pos[0] = std::min(std::max(0.5*h+TOLERANCE, pos[0]), ((double)ni-0.5)*h-TOLERANCE);
                pos[1] = std::min(std::max(0+TOLERANCE, pos[1]), ((double)nj)*h-TOLERANCE);
                ii = std::floor(pos[0]/h - 0.5);
                jj = std::floor(pos[1]/h);
                for(int jjj=jj;jjj<=jj+1;jjj++)for(int iii=ii;iii<=ii+1;iii++)
                {
                    Vec2d gpos = Vec2d(iii,jjj)*h + Vec2d(0.5,0)*h;
                    double w = p.kernel((pos[0] - gpos[0])/h)*p.kernel((pos[1] - gpos[1])/h);
                    int idx = (ni+1)*nj + jjj * ni + iii;
                    interp_matrix.insert(numParticles + i, idx) = w * std::sqrt(p.volume);
                    point_circ(numParticles + i) = p.vel_temp[1] * std::sqrt(p.volume);
                }
            }
        });
    }
    interp_matrix.makeCompressed();

    std::cout << GREEN << "point_circ.norm = " << point_circ.norm() << RESET << std::endl;
    if (!do_delta_circulation)
    {
        if (do_mass_lumping) {
            Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> fullInterpMat = interp_matrix.transpose() * interp_matrix;
            HELPER::rowSumMatrix(fullInterpMat, fullInterpMat);
            Eigen::ConjugateGradient<Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> > dummy_cg;
            dummy_cg.compute(fullInterpMat);
            pre_solve_fluxes = dummy_cg.solveWithGuess(interp_matrix.transpose() * point_circ, pre_solve_fluxes);
        } else {
            int iterations;
            double tolerance;
            bool success;
            if (precond_reset_frequency != 1 && stage_count == 0 && (currentframe%precond_reset_frequency == 0 || currentframe%delayed_reinit_frequency == 0)) {
                Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> fullInterpMat(nF,nF);
                fullInterpMat.setZero();
                fullInterpMat.selfadjointView<Eigen::Lower>().rankUpdate(interp_matrix.transpose());
                _precond.compute(fullInterpMat);
                std::cout << "_precond.info(): " << _precond.info() << " 0: success, 1: numerical issue" << std::endl;
            }
            if (precond_reset_frequency != 1 && _precond.info() == Eigen::Success) {
                success = PLSCGSolve(interp_matrix, point_circ, pre_solve_fluxes, _precond, (double)TOLERANCE, MAX_ITERATIONS, tolerance, iterations);
            } else {
                Eigen::LeastSquareDiagonalPreconditioner<double> diagPrecond;
                diagPrecond.compute(interp_matrix);
                success = PLSCGSolve(interp_matrix, point_circ, pre_solve_fluxes, diagPrecond, (double)TOLERANCE, MAX_ITERATIONS, tolerance, iterations);
            }
            if (precond_reset_frequency != 1 && iterations > (MAX_ITERATIONS / 5)) {
                Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> fullInterpMat(nF,nF);
                fullInterpMat.setZero();
                fullInterpMat.selfadjointView<Eigen::Lower>().rankUpdate(interp_matrix.transpose());
                _precond.compute(fullInterpMat);
            }
            std::cout << "#iteration:      " << iterations << std::endl;
            std::cout << "estimated error: " << tolerance << std::endl;
            if (!success) {
                printf("WARNING: Momentum map least squares solve failed!************************************************\n");
            }
        }
        fluxes = pre_solve_fluxes;
        std::cout << GREEN << "fluxes.norm = " << fluxes.norm() << RESET << std::endl;
    } else {
        circulations = interp_matrix.transpose() * point_circ;

        tbb::parallel_for(tbb::blocked_range<int>(0,nF,TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
        {
            for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
                bool is_x_dir = tIdx < (ni+1)*nj;
                int j = is_x_dir ? tIdx / (ni+1) : (tIdx - (ni+1)*nj) / ni;
                int i = is_x_dir ? tIdx % (ni+1) : (tIdx - (ni+1)*nj) % ni;
                if (is_x_dir) {
                    u(i,j) = circulations(tIdx);
                } else {
                    v(i,j) = circulations(tIdx);
                }
            }
        });

        takeDualwrtStar(u, v, false, true);
    }
}

void COFLIPSolver2D::advectCOFLIPHelper(int stage_count, int currentframe, int delayed_reinit_frequency, double dt, bool do_all) 
{
    if (alpha_buoyancy != 0. || beta_buoyancy != 0.) {
        tbb::parallel_for(tbb::blocked_range<int>(0,cParticles.size(),TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
        {
            for (int p = r.begin(); p < r.end(); ++p) {
                Vec2d spos = cParticles[p].pos_temp;
                Vec2d zeroFormPos = spos - h_uniform/((double)(rho.ni/ni))*Vec2d(0.5);
                cParticles[p].vel_temp[1] += dt*(-alpha_buoyancy*sampleField(zeroFormPos, rho, true) + beta_buoyancy*sampleField(zeroFormPos, temperature, true));
            }
        });
    }

    tbb::parallel_for(tbb::blocked_range<int>(0,cParticles.size(),TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
    {
        for (int p = r.begin(); p < r.end(); ++p) {
            Vec2d pos = cParticles[p].pos_temp;
            Eigen::Matrix2d pullback = pullbackRK4(dt, pos, Eigen::Matrix2d::Identity(), u, v);
            Eigen::Vector2d vel_temp_advected = pullback * 
                (Eigen::Vector2d() << cParticles[p].vel_temp[0], cParticles[p].vel_temp[1]).finished();
            cParticles[p].vel_temp = Vec2d(vel_temp_advected.data());
            cParticles[p].pos_temp = pos;
            cParticles[p].shorterm_pullback = pullback;

            if (do_all) {
                cParticles[p].pos_current = cParticles[p].pos_temp;
                cParticles[p].vel = cParticles[p].vel_temp;

                cParticles[p].longterm_pullback = pullback*cParticles[p].longterm_pullback;
                cParticles[p].delta_t += dt;
            }
        }
    });

    Eigen::VectorX<double> change_in_fluxes(nF);
    change_in_fluxes.setZero();
    if (viscosity != 0.f) {
        calculateCurl(true, true);
        change_in_fluxes = -(viscosity * dt / h) * Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t>(dtranspose_matrix_curl.transpose()) * prev_vorts0form;
        std::cout << GREEN << "change_in_fluxes.norm = " << change_in_fluxes.norm() << RESET << std::endl;
    }

    Eigen::VectorXd fluxes_midpoint(nF);
    tbb::parallel_for(tbb::blocked_range<int>(0,nF,TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
    {
        for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
            bool is_x_dir = tIdx < (ni+1)*nj;
            int j = is_x_dir ? tIdx / (ni+1) : (tIdx - (ni+1)*nj) / ni;
            int i = is_x_dir ? tIdx % (ni+1) : (tIdx - (ni+1)*nj) % ni;
            double value = is_x_dir ? u(i,j) : v(i,j);
            fluxes_midpoint[tIdx] = value;
        }
    });
    solveInterpDaggerVelocity(stage_count, currentframe, delayed_reinit_frequency);

    Eigen::VectorXd fluxes_diff = fluxes - fluxes_original;
    {
        std::cout << RED << "total_diif_projected.norm BEFORE PP: " << 2.*(fluxes_diff.transpose() * starflux_matrix * fluxes_midpoint)*std::pow(h,2) << RESET << std::endl;
    }

    // project out non-orthogonal part of the advection
    if (viscosity == 0. && alpha_buoyancy == 0. && beta_buoyancy == 0.) {
        Eigen::VectorXd circulations_original = starflux_matrix * fluxes_midpoint;
        double original_energy = fluxes_midpoint.transpose() * circulations_original;
        if (std::sqrt(original_energy) > TOLERANCE) {
            std::cout << RED << "total_diif_projected.norm BEFORE   : " << 2.*(fluxes_diff.transpose() * circulations_original)*std::pow(h,2) << RESET << std::endl;
            {
                double projected_fluxes_diff = fluxes_diff.transpose() * circulations_original;
                fluxes_diff -= projected_fluxes_diff * (fluxes_midpoint/original_energy);
            }
            std::cout << RED << "total_diif_projected.norm AFTER    : " << 2.*(fluxes_diff.transpose() * circulations_original)*std::pow(h,2) << RESET << std::endl;
        }
    }
    fluxes = fluxes_original + fluxes_diff;

    tbb::parallel_for(tbb::blocked_range<int>(0,nF,TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
    {
        for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
            bool is_x_dir = tIdx < (ni+1)*nj;
            int j = is_x_dir ? tIdx / (ni+1) : (tIdx - (ni+1)*nj) / ni;
            int i = is_x_dir ? tIdx % (ni+1) : (tIdx - (ni+1)*nj) % ni;
            if (is_x_dir) {
                u_save(i,j) = fluxes(tIdx);
                u(i,j) = fluxes(tIdx) + change_in_fluxes(tIdx);
            } else {
                v_save(i,j) = fluxes(tIdx);
                v(i,j) = fluxes(tIdx) + change_in_fluxes(tIdx);
            }
        }
    });

    double tol = TOLERANCE;
    if (use_pressure_solver) {
        projection(tol,use_neumann_boundary);
    } else {
        projectionWithVort(tol);
    }
}

void COFLIPSolver2D::advanceCOFLIP(double dt, int currentframe, int delayed_reinit_frequency)
{
    std::cout << BLUE <<  "COFLIP scheme frame " << currentframe << " starts !" << RESET << std::endl;
    getCFL();
    std::cout <<  "CFL: " << dt/_cfl << std::endl;

    tbb::parallel_for(tbb::blocked_range<int>(0,nF,TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
    {
        for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
            bool is_x_dir = tIdx < (ni+1)*nj;
            int j = is_x_dir ? tIdx / (ni+1) : (tIdx - (ni+1)*nj) / ni;
            int i = is_x_dir ? tIdx % (ni+1) : (tIdx - (ni+1)*nj) % ni;
            double value = is_x_dir ? u(i,j) : v(i,j);
            fluxes_original[tIdx] = value;
        }
    });
    
    Array2d u0, v0, u1, v1, u2, v2, utemp, vtemp;
    if (timeIntOrder == TimeIntegration::RK2) {
        u0 = u; v0 = v;
        tbb::parallel_for(tbb::blocked_range<int>(0,cParticles.size(),TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
        {
            for (int i = r.begin(); i < r.end(); ++i) {
                cParticles[i].pos_temp = cParticles[i].pos_current;
                cParticles[i].vel_temp = cParticles[i].vel;
            }
        });
        advectCOFLIPHelper(0, currentframe, delayed_reinit_frequency, dt);
        utemp = u; vtemp = v;
        std::cout << BLUE <<  "ODE stage 1 done..." << RESET << std::endl;
    } else if (timeIntOrder >= TimeIntegration::RK3) {
        u0 = u; v0 = v;
        tbb::parallel_for(tbb::blocked_range<int>(0,cParticles.size(),TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
        {
            for (int i = r.begin(); i < r.end(); ++i) {
                cParticles[i].pos_temp = cParticles[i].pos_current;
                cParticles[i].vel_temp = cParticles[i].vel;
            }
        });
        advectCOFLIPHelper(0, currentframe, delayed_reinit_frequency, 0.5*dt);
        std::cout << BLUE <<  "ODE stage 1 done..." << RESET << std::endl;

        if (timeIntOrder == TimeIntegration::RK3) {
            u1 = u; v1 = v;
            tbb::parallel_for(tbb::blocked_range<int>(0,cParticles.size(),TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
            {
                for (int i = r.begin(); i < r.end(); ++i) {
                    cParticles[i].pos_temp = cParticles[i].pos_current;
                    cParticles[i].vel_temp = cParticles[i].vel;
                }
            });
            advectCOFLIPHelper(1, currentframe, delayed_reinit_frequency, 0.75*dt);
            std::cout << BLUE <<  "ODE stage 2 done..." << RESET << std::endl;
        } else if (timeIntOrder == TimeIntegration::RK4) {
            u1 = u; v1 = v;
            tbb::parallel_for(tbb::blocked_range<int>(0,cParticles.size(),TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
            {
                for (int i = r.begin(); i < r.end(); ++i) {
                    cParticles[i].pos_temp = cParticles[i].pos_current;
                    cParticles[i].vel_temp = cParticles[i].vel;
                }
            });
            advectCOFLIPHelper(1, currentframe, delayed_reinit_frequency, 0.5*dt);
            u2 = u; v2 = v;
            std::cout << BLUE <<  "ODE stage 2 done..." << RESET << std::endl;
            tbb::parallel_for(tbb::blocked_range<int>(0,cParticles.size(),TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
            {
                for (int i = r.begin(); i < r.end(); ++i) {
                    cParticles[i].pos_temp = cParticles[i].pos_current;
                    cParticles[i].vel_temp = cParticles[i].vel;
                }
            });
            advectCOFLIPHelper(2, currentframe, delayed_reinit_frequency, dt);
            std::cout << BLUE <<  "ODE stage 3 done..." << RESET << std::endl;
        }
    }

    if (timeIntOrder == TimeIntegration::RK3) {
        tbb::parallel_for(tbb::blocked_range2d<int,int>{0,ni+1,1,0,nj,1}, [&](const tbb::blocked_range2d<int,int> &r)
        {
            int ie = r.rows().end();
            int je = r.cols().end();
            for (int j = r.cols().begin(); j < je; ++j) {
                for (int i = r.rows().begin(); i < ie; ++i) {
                    u(i,j) = 2./9.*u0(i,j) + 3./9.*u1(i,j) + 4./9.*u(i,j);
                }
            }
        });
        tbb::parallel_for(tbb::blocked_range2d<int,int>{0,ni,1,0,nj+1,1}, [&](const tbb::blocked_range2d<int,int> &r)
        {
            int ie = r.rows().end();
            int je = r.cols().end();
            for (int j = r.cols().begin(); j < je; ++j) {
                for (int i = r.rows().begin(); i < ie; ++i) {
                    v(i,j) = 2./9.*v0(i,j) + 3./9.*v1(i,j) + 4./9.*v(i,j);
                }
            }
        });
    } else if (timeIntOrder == TimeIntegration::RK4) {
        tbb::parallel_for(tbb::blocked_range2d<int,int>{0,ni+1,1,0,nj,1}, [&](const tbb::blocked_range2d<int,int> &r)
        {
            int ie = r.rows().end();
            int je = r.cols().end();
            for (int j = r.cols().begin(); j < je; ++j) {
                for (int i = r.rows().begin(); i < ie; ++i) {
                    u(i,j) = 1./6.*u0(i,j) + 2./6.*u1(i,j) + 2./6.*u2(i,j) + 1./6.*u(i,j);
                }
            }
        });
        tbb::parallel_for(tbb::blocked_range2d<int,int>{0,ni,1,0,nj+1,1}, [&](const tbb::blocked_range2d<int,int> &r)
        {
            int ie = r.rows().end();
            int je = r.cols().end();
            for (int j = r.cols().begin(); j < je; ++j) {
                for (int i = r.rows().begin(); i < ie; ++i) {
                    v(i,j) = 1./6.*v0(i,j) + 2./6.*v1(i,j) + 2./6.*v2(i,j) + 1./6.*v(i,j);
                }
            }
        });
    }

    if (timeIntOrder == TimeIntegration::RK2) {
        double error = 1.;
        double error_prev = 1.;
        double fluxes0_norm = 0.;
        {
            Eigen::VectorXd fluxes0(nF);
            tbb::parallel_for(tbb::blocked_range<int>(0,nF,TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
            {
                for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
                    bool is_x_dir = tIdx < (ni+1)*nj;
                    int j = is_x_dir ? tIdx / (ni+1) : (tIdx - (ni+1)*nj) / ni;
                    int i = is_x_dir ? tIdx % (ni+1) : (tIdx - (ni+1)*nj) % ni;
                    fluxes0[tIdx] = is_x_dir ? u0(i,j) : v0(i,j);
                }
            });
            fluxes0_norm = fluxes0.norm();
        }

        int iter = 0;
        do {
            tbb::parallel_for(tbb::blocked_range2d<int,int>{0,ni+1,1,0,nj,1}, [&](const tbb::blocked_range2d<int,int> &r)
            {
                int ie = r.rows().end();
                int je = r.cols().end();
                for (int j = r.cols().begin(); j < je; ++j) {
                    for (int i = r.rows().begin(); i < ie; ++i) {
                        u(i,j) = 0.5*u0(i,j) + 0.5*u(i,j);
                    }
                }
            });
            tbb::parallel_for(tbb::blocked_range2d<int,int>{0,ni,1,0,nj+1,1}, [&](const tbb::blocked_range2d<int,int> &r)
            {
                int ie = r.rows().end();
                int je = r.cols().end();
                for (int j = r.cols().begin(); j < je; ++j) {
                    for (int i = r.rows().begin(); i < ie; ++i) {
                        v(i,j) = 0.5*v0(i,j) + 0.5*v(i,j);
                    }
                }
            });
            tbb::parallel_for(tbb::blocked_range<int>(0,cParticles.size(),TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
            {
                for (int i = r.begin(); i < r.end(); ++i) {
                    cParticles[i].pos_temp = cParticles[i].pos_current;
                    cParticles[i].vel_temp = cParticles[i].vel;
                }
            });
            advectCOFLIPHelper(3, currentframe, delayed_reinit_frequency, dt);
            Eigen::VectorXd fluxes_temp(nF);
            tbb::parallel_for(tbb::blocked_range<int>(0,nF,TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
            {
                for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
                    bool is_x_dir = tIdx < (ni+1)*nj;
                    int j = is_x_dir ? tIdx / (ni+1) : (tIdx - (ni+1)*nj) / ni;
                    int i = is_x_dir ? tIdx % (ni+1) : (tIdx - (ni+1)*nj) % ni;
                    fluxes_temp[tIdx] = is_x_dir ? utemp(i,j) : vtemp(i,j);
                }
            });
            utemp = u; vtemp = v;
            tbb::parallel_for(tbb::blocked_range<int>(0,nF,TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
            {
                for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
                    bool is_x_dir = tIdx < (ni+1)*nj;
                    int j = is_x_dir ? tIdx / (ni+1) : (tIdx - (ni+1)*nj) / ni;
                    int i = is_x_dir ? tIdx % (ni+1) : (tIdx - (ni+1)*nj) % ni;
                    fluxes[tIdx] = is_x_dir ? u(i,j) : v(i,j);
                }
            });
            std::cout << GREEN << "fluxes.norm = " << fluxes.norm() << RESET << std::endl;
            error_prev = error;
            error = fluxes0_norm != 0. ? (fluxes-fluxes_temp).norm() / fluxes0_norm : (fluxes-fluxes_temp).norm();
            std::cout << YELLOW << "implicit fixed point error: " << error << RESET << std::endl;
            if (error < 5.*TOLERANCE) {
                break;
            }
            iter++;
        } while (do_implicit && (iter==1 || (iter < (MAX_ITERATIONS/10) && error_prev/error > 1. && !(error_prev/error < 1.1f && error < 500.*TOLERANCE))));
        tbb::parallel_for(tbb::blocked_range<int>(0,cParticles.size(),TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
        {
            for (int p = r.begin(); p < r.end(); ++p) {
                cParticles[p].pos_current = cParticles[p].pos_temp;
                cParticles[p].vel = cParticles[p].vel_temp;
                cParticles[p].longterm_pullback = cParticles[p].shorterm_pullback*cParticles[p].longterm_pullback;
                cParticles[p].delta_t += dt;
            }
        });
    } else {
        tbb::parallel_for(tbb::blocked_range<int>(0,cParticles.size(),TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
        {
            for (int i = r.begin(); i < r.end(); ++i) {
                cParticles[i].pos_temp = cParticles[i].pos_current;
                cParticles[i].vel_temp = cParticles[i].vel;
            }
        });
        advectCOFLIPHelper(3, currentframe, delayed_reinit_frequency, dt, true);
    }

    std::cout << BLUE <<  "FLOW and its lifted action compeleted!" << RESET << std::endl;
    std::cout << BLUE <<  "Circulation projected back to the grid!" << RESET << std::endl;

    Array2d rho_weight, T_weight;
    rho_weight.resize(rho.ni, rho.nj);
    T_weight.resize(temperature.ni, temperature.nj);
    rho_weight.assign(1e-4);
    T_weight.assign(1e-4);
    rho.assign(0);
    temperature.assign(0);
    // splat particle properties to grid
    int numParticles = cParticles.size();
    for(int i=0;i<numParticles;i++)
    {
        CmapParticles p = cParticles[i];
        Vec2d pos = p.pos_current;
        double used_h = h_uniform / (double)(rho.ni/ni);
        pos[0] = std::min(std::max(0.5*used_h+TOLERANCE, pos[0]), ((double)rho.ni-0.5)*used_h-TOLERANCE);
        pos[1] = std::min(std::max(0.5*used_h+TOLERANCE, pos[1]), ((double)rho.nj-0.5)*used_h-TOLERANCE);
        int ii, jj;
        ii = std::floor(pos[0]/used_h - 0.5);
        jj = std::floor(pos[1]/used_h - 0.5);
        for(int jjj=jj;jjj<=jj+1;jjj++)for(int iii=ii;iii<=ii+1;iii++)
        {
            Vec2d gpos = Vec2d(iii,jjj)*used_h + Vec2d(0.5,0.5)*used_h;
            double w = p.kernel((pos[0] - gpos[0])/used_h)*p.kernel((pos[1] - gpos[1])/used_h);

            double c0 = p.C_rho[0];
            double c1 = p.C_rho[1]*(gpos[0] - pos[0]);
            double c2 = p.C_rho[2]*(gpos[1] - pos[1]);
            double c3 = p.C_rho[3]*(gpos[0] - pos[0])*(gpos[1] - pos[1]);

            rho(iii,jjj) += w*(c0 + c1 + c2 + c3);
            rho_weight(iii,jjj) += w;
        }

        ii = std::floor(pos[0]/used_h - 0.5);
        jj = std::floor(pos[1]/used_h - 0.5);
        for(int jjj=jj;jjj<=jj+1;jjj++)for(int iii=ii;iii<=ii+1;iii++)
        {
            Vec2d gpos = Vec2d(iii,jjj)*used_h + Vec2d(0.5,0.5)*used_h;
            double w = p.kernel((pos[0] - gpos[0])/used_h)*p.kernel((pos[1] - gpos[1])/used_h);

            double c0 = p.C_temperature[0];
            double c1 = p.C_temperature[1]*(gpos[0] - pos[0]);
            double c2 = p.C_temperature[2]*(gpos[1] - pos[1]);
            double c3 = p.C_temperature[3]*(gpos[0] - pos[0])*(gpos[1] - pos[1]);

            temperature(iii,jjj) += w*(c0 + c1 + c2 + c3);
            T_weight(iii,jjj) += w;
        }
    }
    rho /= rho_weight;
    temperature /= T_weight;

    Array2d rho_save, temperature_save;
    rho_save = rho;
    temperature_save = temperature;

    emitSmoke();

    Array2d u_diff, v_diff, rho_diff, temperature_diff;
    u_diff = u;
    v_diff = v;
    rho_diff = rho;
    temperature_diff = temperature;
    u_diff -= u_save;
    v_diff -= v_save;
    rho_diff -= rho_save;
    temperature_diff -= temperature_save;

    double flip = do_delta_circulation ? 1.0 : (((do_particle_sample_after_first || currentframe != 0) && currentframe%((int)(delayed_reinit_frequency*substep))==0) ? 0.0 : 1.0);
    bool apply_cuttoff = do_delta_circulation ? false : true;

    double max_FTLE = tbb::parallel_reduce( 
        tbb::blocked_range<int>(0,cParticles.size()),
        TOLERANCE,
        [&](tbb::blocked_range<int> range, double running_max)
        {
            for (int tIdx=range.begin(); tIdx<range.end(); ++tIdx)
            {
                double det = 0.;
                running_max = std::max(running_max, computeFTLE(det, cParticles[tIdx].longterm_pullback));
            }

            return running_max;
        }, [](double a, double b) { return std::max(a,b); } );
    
    apply_cuttoff = apply_cuttoff && max_FTLE > adaptive_reset_cutoff;

    if (flip == 0.0) {
        if (do_particle_sample_after_first) {
            do_particle_sample_after_first = false;
            do_uniform_particle_seeding = false;
        }
        seedParticles(m_N);
        apply_cuttoff = false;
    }

    std::atomic<int> adaptive_reset_counter = 0;
    tbb::parallel_for(tbb::blocked_range<int>(0,cParticles.size(),TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
    {
        for (int i = r.begin(); i < r.end(); ++i) {
            Vec2d p_vel = cParticles[i].vel;
            Vec2d p_pos = cParticles[i].pos_current;
            Vec2d curr_dv = getVelocityBSpline(p_pos, u_diff, v_diff);
            p_vel = flip*(p_vel)
                    + (1-flip)*(getVelocityBSpline(p_pos, u_save, v_save))
                    + (do_delta_circulation ? Vec2d(0.0) : curr_dv)
                    ;

            double det = 0.;
            double FTLE = computeFTLE(det, cParticles[i].longterm_pullback);
            if (apply_cuttoff && FTLE > adaptive_reset_cutoff*0.6) {
                p_vel = getVelocityBSpline(p_pos, u, v);
                cParticles[i].longterm_pullback = Eigen::Matrix2d::Identity();
                cParticles[i].delta_t = 0.0;
                adaptive_reset_counter++;
            }
            cParticles[i].vel = p_vel;

            if (flip == 0.0)
            {
                cParticles[i].longterm_pullback = Eigen::Matrix2d::Identity();
                cParticles[i].delta_t = 0.0;
            }

            double p_rho = cParticles[i].rho;
            double p_temperature = cParticles[i].temperature;
            double used_h = h_uniform / (double)(rho.ni/ni);
            Vec2d zeroFormPos = p_pos - used_h*Vec2d(0.5,0.5);
            p_rho = flip*(p_rho + sampleField(zeroFormPos, rho_diff, true))
                    + (1-flip)*sampleField(zeroFormPos, rho, true);
            p_temperature = flip*(p_temperature + sampleField(zeroFormPos, temperature_diff, true))
                    + (1-flip)*sampleField(zeroFormPos, temperature, true);
            {
                Vec2d drhodx = sampleGradientField(zeroFormPos, rho, true);
                Vec2d dTdx = sampleGradientField(zeroFormPos, temperature, true);
                cParticles[i].C_rho = Vec4d(p_rho, drhodx[0], drhodx[1], sampleCrossHessianField(zeroFormPos, rho, true));
                cParticles[i].C_temperature = Vec4d(p_temperature, dTdx[0], dTdx[1], sampleCrossHessianField(zeroFormPos, temperature, true));
            }
            cParticles[i].rho = p_rho;
            cParticles[i].temperature = p_temperature;
        }
    });
    std::cout << BLUE <<  "adaptive_reset_counter: " << adaptive_reset_counter << ", (%): " << ((double)adaptive_reset_counter/(double)cParticles.size())*100. << RESET << std::endl;
    std::cout << BLUE <<  "Particle information sampled from the grid!" << RESET << std::endl;
    std::cout << BLUE <<  "end of frame!" << RESET << std::endl;
}

void COFLIPSolver2D::advectCovectorPOLYFLIPHelper(int stage_count, int currentframe, int delayed_reinit_frequency, double dt, bool do_all) 
{
    if (alpha_buoyancy != 0. || beta_buoyancy != 0.) {
        tbb::parallel_for(tbb::blocked_range<int>(0,cParticles.size(),TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
        {
            for (int p = r.begin(); p < r.end(); ++p) {
                Vec2d spos = cParticles[p].pos_temp;
                Vec2d zeroFormPos = spos - h_uniform/((double)(rho.ni/ni))*Vec2d(0.5);
                cParticles[p].vel_temp[1] += dt*(-alpha_buoyancy*sampleField(zeroFormPos, rho, true) + beta_buoyancy*sampleField(zeroFormPos, temperature, true));
            }
        });
    }

    tbb::parallel_for(tbb::blocked_range<int>(0,cParticles.size(),TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
    {
        for (int p = r.begin(); p < r.end(); ++p) {
            Vec2d pos = cParticles[p].pos_temp;
            pos = solveODE(dt, pos, u, v);
            cParticles[p].pos_temp = pos;

            if (do_all) {
                cParticles[p].pos_current = cParticles[p].pos_temp;
                cParticles[p].vel = cParticles[p].vel_temp;
                cParticles[p].delta_t += dt;
            }
        }
    });

    Eigen::VectorX<double> change_in_fluxes(nF);
    change_in_fluxes.setZero();
    if (viscosity != 0.f) {
        tbb::parallel_for(tbb::blocked_range<int>(0,nF,TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
        {
            for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
                bool is_x_dir = tIdx < (ni+1)*nj;
                int j = is_x_dir ? tIdx / (ni+1) : (tIdx - (ni+1)*nj) / ni;
                int i = is_x_dir ? tIdx % (ni+1) : (tIdx - (ni+1)*nj) % ni;
                fluxes[tIdx] = is_x_dir ? u(i,j) : v(i,j);
            }
        });
        calculateCurl(true, true);
        change_in_fluxes = -(viscosity * dt / h) * Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t>(dtranspose_matrix_curl.transpose()) * prev_vorts0form;
        std::cout << GREEN << "change_in_fluxes.norm = " << change_in_fluxes.norm() << RESET << std::endl;
    }

    if (sim_scheme != Scheme::POLYFLIP) {
        tbb::parallel_for(tbb::blocked_range2d<int,int>{0,ni,1,0,nj,1}, [&](const tbb::blocked_range2d<int,int> &r)
        {
            int ie = r.rows().end();
            int je = r.cols().end();
            for (int j = r.cols().begin(); j < je; ++j) {
                for (int i = r.rows().begin(); i < ie; ++i) {
                    backward_x(i, j) = h*((double)i + 0.5);
                    backward_y(i, j) = h*((double)j + 0.5);
                }
            }
        });
        updateBackward(dt, backward_x, backward_y);
    }

    u_mass.assign(1e-4);
    v_mass.assign(1e-4);
    u.assign(0);
    v.assign(0);
    for(int i=0;i<cParticles.size();i++)
    {
        CmapParticles p = cParticles[i];
        Vec2d pos = p.pos_temp;
        pos[0] = std::min(std::max(0+TOLERANCE, pos[0]), ((double)ni)*h-TOLERANCE);
        pos[1] = std::min(std::max(0.5*h+TOLERANCE, pos[1]), ((double)nj-0.5)*h-TOLERANCE);
        int ii, jj;
        ii = std::floor(pos[0]/h);
        jj = std::floor(pos[1]/h-0.5);
        for(int jjj=jj;jjj<=jj+1;jjj++)for(int iii=ii;iii<=ii+1;iii++)
        {
            Vec2d gpos = Vec2d(iii,jjj)*h + Vec2d(0,0.5)*h;
            double w = p.kernel((pos[0] - gpos[0])/h)*p.kernel((pos[1] - gpos[1])/h);
            double c0 = p.vel_temp[0];
            double c1 = p.C_x[1]*(gpos[0] - pos[0]);
            double c2 = p.C_x[2]*(gpos[1] - pos[1]);
            double c3 = p.C_x[3]*(gpos[0] - pos[0])*(gpos[1] - pos[1]);

            u(iii,jjj) += w * (c0 + c1 + c2 + c3);
            u_mass(iii,jjj) += w;
        }

        pos = p.pos_temp;
        pos[0] = std::min(std::max(0.5*h+TOLERANCE, pos[0]), ((double)ni-0.5)*h-TOLERANCE);
        pos[1] = std::min(std::max(0+TOLERANCE, pos[1]), ((double)nj)*h-TOLERANCE);
        ii = std::floor(pos[0]/h - 0.5);
        jj = std::floor(pos[1]/h);
        for(int jjj=jj;jjj<=jj+1;jjj++)for(int iii=ii;iii<=ii+1;iii++)
        {
            Vec2d gpos = Vec2d(iii,jjj)*h + Vec2d(0.5,0)*h;
            double w = p.kernel((pos[0] - gpos[0])/h)*p.kernel((pos[1] - gpos[1])/h);
            double c0 = p.vel_temp[1];
            double c1 = p.C_y[1]*(gpos[0] - pos[0]);
            double c2 = p.C_y[2]*(gpos[1] - pos[1]);
            double c3 = p.C_y[3]*(gpos[0] - pos[0])*(gpos[1] - pos[1]);

            v(iii,jjj) += w * (c0 + c1 + c2 + c3);
            v_mass(iii,jjj) += w;
        }
    }
    u /= u_mass;
    v /= v_mass;
    u_save = u;
    v_save = v;

    if (sim_scheme != Scheme::POLYFLIP) {
        Array2d u_sample = u;
        Array2d v_sample = v;
        // apply covector pullback to velocities
        tbb::parallel_for(tbb::blocked_range2d<int,int>{0,ni+1,1,0,nj,1}, [&](const tbb::blocked_range2d<int,int> &r)
        {
            int ie = r.rows().end();
            int je = r.cols().end();
            for (int j = r.cols().begin(); j < je; ++j) {
                for (int i = r.rows().begin(); i < ie; ++i) {
                    if (i != 0 && i != ni)
                    {
                        Vec2d pos = h * Vec2d(i, j) + h * Vec2d(0.0, 0.5);

                        Vec2d pos_front = pos + h * Vec2d(0.5,0.0);
                        double x_front_init = sampleField(pos_front - h * Vec2d(0.5), backward_x);
                        double y_front_init = sampleField(pos_front - h * Vec2d(0.5), backward_y);
                        Vec2d pos1_front = Vec2d(x_front_init, y_front_init);
                        Vec2d pos_back  = pos - h * Vec2d(0.5,0.0);
                        double x_back_init = sampleField(pos_back - h * Vec2d(0.5), backward_x);
                        double y_back_init = sampleField(pos_back - h * Vec2d(0.5), backward_y);
                        Vec2d pos1_back = Vec2d(x_back_init, y_back_init);
                        Vec2d diff = -pos1_back+pos1_front;
                        double distance = dist(pos_back,pos_front);
                        Vec2d vel_at_face(sampleField(pos - h * Vec2d(0., 0.5), u_sample), sampleField(pos - h * Vec2d(0.5, 0.), v_sample));
                        u(i,j) = dot(diff,vel_at_face) / distance;
                    }
                }
            }
        });
        tbb::parallel_for(tbb::blocked_range2d<int,int>{0,ni,1,0,nj+1,1}, [&](const tbb::blocked_range2d<int,int> &r)
        {
            int ie = r.rows().end();
            int je = r.cols().end();
            for (int j = r.cols().begin(); j < je; ++j) {
                for (int i = r.rows().begin(); i < ie; ++i) {
                    if (j != 0 && j != nj)
                    {
                        Vec2d pos = h * Vec2d(i, j) + h * Vec2d(0.5, 0.0);

                        Vec2d pos_front = pos + h * Vec2d(0.0,0.5);
                        double x_front_init = sampleField(pos_front - h * Vec2d(0.5), backward_x);
                        double y_front_init = sampleField(pos_front - h * Vec2d(0.5), backward_y);
                        Vec2d pos1_front = Vec2d(x_front_init, y_front_init);
                        Vec2d pos_back  = pos - h * Vec2d(0.0,0.5);
                        double x_back_init = sampleField(pos_back - h * Vec2d(0.5), backward_x);
                        double y_back_init = sampleField(pos_back - h * Vec2d(0.5), backward_y);
                        Vec2d pos1_back = Vec2d(x_back_init, y_back_init);
                        Vec2d diff = -pos1_back+pos1_front;
                        double distance = dist(pos_back,pos_front);
                        Vec2d vel_at_face(sampleField(pos - h * Vec2d(0., 0.5), u_sample), sampleField(pos - h * Vec2d(0.5, 0.), v_sample));
                        v(i,j) = dot(diff,vel_at_face) / distance;
                    }
                }
            }
        });
    }

    tbb::parallel_for(tbb::blocked_range<int>(0,nF,TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
    {
        for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
            bool is_x_dir = tIdx < (ni+1)*nj;
            int j = is_x_dir ? tIdx / (ni+1) : (tIdx - (ni+1)*nj) / ni;
            int i = is_x_dir ? tIdx % (ni+1) : (tIdx - (ni+1)*nj) % ni;
            if (is_x_dir) {
                u(i,j) += change_in_fluxes(tIdx);
            } else {
                v(i,j) += change_in_fluxes(tIdx);
            }
        }
    });

    double tol = TOLERANCE;
    projection(tol,use_neumann_boundary);
}

void COFLIPSolver2D::advanceCovectorPOLYFLIP(double dt, int currentframe, int delayed_reinit_frequency)
{
    std::cout << BLUE <<  "CovectorPOLYFLIP scheme frame " << currentframe << " starts !" << RESET << std::endl;
    getCFL();
    std::cout <<  "CFL: " << dt/_cfl << std::endl;
    Array2d u0, v0, u1, v1, u2, v2;
    if (timeIntOrder == TimeIntegration::RK2) {
        u0 = u; v0 = v;
        tbb::parallel_for(tbb::blocked_range<int>(0,cParticles.size(),TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
        {
            for (int i = r.begin(); i < r.end(); ++i) {
                cParticles[i].pos_temp = cParticles[i].pos_current;
                cParticles[i].vel_temp = cParticles[i].vel;
            }
        });
        advectCovectorPOLYFLIPHelper(0, currentframe, delayed_reinit_frequency, 0.5*dt);
        std::cout << BLUE <<  "ODE stage 1 done..." << RESET << std::endl;
    } else if (timeIntOrder >= TimeIntegration::RK3) {
        u0 = u; v0 = v;
        tbb::parallel_for(tbb::blocked_range<int>(0,cParticles.size(),TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
        {
            for (int i = r.begin(); i < r.end(); ++i) {
                cParticles[i].pos_temp = cParticles[i].pos_current;
                cParticles[i].vel_temp = cParticles[i].vel;
            }
        });
        advectCovectorPOLYFLIPHelper(0, currentframe, delayed_reinit_frequency, 0.5*dt);
        std::cout << BLUE <<  "ODE stage 1 done..." << RESET << std::endl;

        if (timeIntOrder == TimeIntegration::RK3) {
            u1 = u; v1 = v;
            tbb::parallel_for(tbb::blocked_range<int>(0,cParticles.size(),TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
            {
                for (int i = r.begin(); i < r.end(); ++i) {
                    cParticles[i].pos_temp = cParticles[i].pos_current;
                    cParticles[i].vel_temp = cParticles[i].vel;
                }
            });
            advectCovectorPOLYFLIPHelper(1, currentframe, delayed_reinit_frequency, 0.75*dt);
            std::cout << BLUE <<  "ODE stage 2 done..." << RESET << std::endl;
        } else if (timeIntOrder == TimeIntegration::RK4) {
            u1 = u; v1 = v;
            tbb::parallel_for(tbb::blocked_range<int>(0,cParticles.size(),TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
            {
                for (int i = r.begin(); i < r.end(); ++i) {
                    cParticles[i].pos_temp = cParticles[i].pos_current;
                    cParticles[i].vel_temp = cParticles[i].vel;
                }
            });
            advectCovectorPOLYFLIPHelper(1, currentframe, delayed_reinit_frequency, 0.5*dt);
            u2 = u; v2 = v;
            std::cout << BLUE <<  "ODE stage 2 done..." << RESET << std::endl;
            tbb::parallel_for(tbb::blocked_range<int>(0,cParticles.size(),TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
            {
                for (int i = r.begin(); i < r.end(); ++i) {
                    cParticles[i].pos_temp = cParticles[i].pos_current;
                    cParticles[i].vel_temp = cParticles[i].vel;
                }
            });
            advectCovectorPOLYFLIPHelper(2, currentframe, delayed_reinit_frequency, dt);
            std::cout << BLUE <<  "ODE stage 3 done..." << RESET << std::endl;
        }
    }

    if (timeIntOrder == TimeIntegration::RK3) {
        tbb::parallel_for(tbb::blocked_range2d<int,int>{0,ni+1,1,0,nj,1}, [&](const tbb::blocked_range2d<int,int> &r)
        {
            int ie = r.rows().end();
            int je = r.cols().end();
            for (int j = r.cols().begin(); j < je; ++j) {
                for (int i = r.rows().begin(); i < ie; ++i) {
                    u(i,j) = 2./9.*u0(i,j) + 3./9.*u1(i,j) + 4./9.*u(i,j);
                }
            }
        });
        tbb::parallel_for(tbb::blocked_range2d<int,int>{0,ni,1,0,nj+1,1}, [&](const tbb::blocked_range2d<int,int> &r)
        {
            int ie = r.rows().end();
            int je = r.cols().end();
            for (int j = r.cols().begin(); j < je; ++j) {
                for (int i = r.rows().begin(); i < ie; ++i) {
                    v(i,j) = 2./9.*v0(i,j) + 3./9.*v1(i,j) + 4./9.*v(i,j);
                }
            }
        });
    } else if (timeIntOrder == TimeIntegration::RK4) {
        tbb::parallel_for(tbb::blocked_range2d<int,int>{0,ni+1,1,0,nj,1}, [&](const tbb::blocked_range2d<int,int> &r)
        {
            int ie = r.rows().end();
            int je = r.cols().end();
            for (int j = r.cols().begin(); j < je; ++j) {
                for (int i = r.rows().begin(); i < ie; ++i) {
                    u(i,j) = 1./6.*u0(i,j) + 2./6.*u1(i,j) + 2./6.*u2(i,j) + 1./6.*u(i,j);
                }
            }
        });
        tbb::parallel_for(tbb::blocked_range2d<int,int>{0,ni,1,0,nj+1,1}, [&](const tbb::blocked_range2d<int,int> &r)
        {
            int ie = r.rows().end();
            int je = r.cols().end();
            for (int j = r.cols().begin(); j < je; ++j) {
                for (int i = r.rows().begin(); i < ie; ++i) {
                    v(i,j) = 1./6.*v0(i,j) + 2./6.*v1(i,j) + 2./6.*v2(i,j) + 1./6.*v(i,j);
                }
            }
        });
    }

    tbb::parallel_for(tbb::blocked_range<int>(0,cParticles.size(),TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
    {
        for (int i = r.begin(); i < r.end(); ++i) {
            cParticles[i].pos_temp = cParticles[i].pos_current;
            cParticles[i].vel_temp = cParticles[i].vel;
        }
    });
    advectCovectorPOLYFLIPHelper(3, currentframe, delayed_reinit_frequency, dt, true);

    std::cout << BLUE <<  "FLOW and its lifted action compeleted!" << RESET << std::endl;
    std::cout << BLUE <<  "Circulation projected back to the grid!" << RESET << std::endl;

    Array2d rho_weight, T_weight;
    rho_weight.resize(rho.ni, rho.nj);
    T_weight.resize(temperature.ni, temperature.nj);
    rho_weight.assign(1e-4);
    T_weight.assign(1e-4);
    rho.assign(0);
    temperature.assign(0);
    // splat particle properties to grid
    int numParticles = cParticles.size();
    for(int i=0;i<numParticles;i++)
    {
        CmapParticles p = cParticles[i];
        Vec2d pos = p.pos_current;
        double used_h = h / (double)(rho.ni/ni);
        pos[0] = std::min(std::max(0.5*used_h+TOLERANCE, pos[0]), ((double)rho.ni-0.5)*used_h-TOLERANCE);
        pos[1] = std::min(std::max(0.5*used_h+TOLERANCE, pos[1]), ((double)rho.nj-0.5)*used_h-TOLERANCE);
        int ii, jj;
        ii = std::floor(pos[0]/used_h - 0.5);
        jj = std::floor(pos[1]/used_h - 0.5);
        for(int jjj=jj;jjj<=jj+1;jjj++)for(int iii=ii;iii<=ii+1;iii++)
        {
            Vec2d gpos = Vec2d(iii,jjj)*used_h + Vec2d(0.5,0.5)*used_h;
            double w = p.kernel((pos[0] - gpos[0])/used_h)*p.kernel((pos[1] - gpos[1])/used_h);

            double c0 = p.C_rho[0];
            double c1 = p.C_rho[1]*(gpos[0] - pos[0]);
            double c2 = p.C_rho[2]*(gpos[1] - pos[1]);
            double c3 = p.C_rho[3]*(gpos[0] - pos[0])*(gpos[1] - pos[1]);

            rho(iii,jjj) += w*(c0 + c1 + c2 + c3);
            rho_weight(iii,jjj) += w;
        }

        ii = std::floor(pos[0]/used_h - 0.5);
        jj = std::floor(pos[1]/used_h - 0.5);
        for(int jjj=jj;jjj<=jj+1;jjj++)for(int iii=ii;iii<=ii+1;iii++)
        {
            Vec2d gpos = Vec2d(iii,jjj)*used_h + Vec2d(0.5,0.5)*used_h;
            double w = p.kernel((pos[0] - gpos[0])/used_h)*p.kernel((pos[1] - gpos[1])/used_h);

            double c0 = p.C_temperature[0];
            double c1 = p.C_temperature[1]*(gpos[0] - pos[0]);
            double c2 = p.C_temperature[2]*(gpos[1] - pos[1]);
            double c3 = p.C_temperature[3]*(gpos[0] - pos[0])*(gpos[1] - pos[1]);

            temperature(iii,jjj) += w*(c0 + c1 + c2 + c3);
            T_weight(iii,jjj) += w;
        }
    }
    rho /= rho_weight;
    temperature /= T_weight;

    Array2d rho_save, temperature_save;
    rho_save = rho;
    temperature_save = temperature;

    Array2d u_diff, v_diff, rho_diff, temperature_diff;
    u_diff = u;
    v_diff = v;
    rho_diff = rho;
    temperature_diff = temperature;
    u_diff -= u_save;
    v_diff -= v_save;
    rho_diff -= rho_save;
    temperature_diff -= temperature_save;

    double flip = do_delta_circulation ? 1.0 : (((do_particle_sample_after_first || currentframe != 0) && currentframe%((int)(delayed_reinit_frequency*substep))==0) ? 0.0 : 1.0);

    if (flip == 0.0) {
        if (do_particle_sample_after_first) {
            do_particle_sample_after_first = false;
            do_uniform_particle_seeding = false;
        }
        seedParticles(m_N);
    }

    tbb::parallel_for(tbb::blocked_range<int>(0,cParticles.size(),TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
    {
        for (int i = r.begin(); i < r.end(); ++i) {
            Vec2d p_vel = cParticles[i].vel;
            Vec2d p_pos = cParticles[i].pos_current;
            Vec2d curr_dv = getVelocity(p_pos, u_diff, v_diff);
            p_vel = flip*(p_vel)
                    + (1-flip)*(getVelocity(p_pos, u_save, v_save))
                    + (do_delta_circulation ? Vec2d(0.0) : curr_dv)
                    ;

            cParticles[i].vel = p_vel;
            cParticles[i].vel_temp = p_vel;

            {
                Eigen::Matrix2d dveldx = getJacobianVelocity(p_pos, u, v);
                Vec2d dudx(dveldx(0,0), dveldx(0,1));
                Vec2d dvdx(dveldx(1,0), dveldx(1,1));
                cParticles[i].C_x = Vec4d(
                    p_vel[0],
                    dudx[0],
                    dudx[1],
                    sampleCrossHessianField(p_pos - Vec2d(0.0, 0.5*h), u));
                cParticles[i].C_y = Vec4d(
                    p_vel[1],
                    dvdx[0],
                    dvdx[1],
                    sampleCrossHessianField(p_pos - Vec2d(0.5*h, 0.0), v));
            }

            if (flip == 0.0)
            {
                cParticles[i].delta_t = 0.0;
            }

            double p_rho = cParticles[i].rho;
            double p_temperature = cParticles[i].temperature;
            double used_h = h / (double)(rho.ni/ni);
            Vec2d zeroFormPos = p_pos - used_h*Vec2d(0.5,0.5);
            p_rho = flip*(p_rho + sampleField(zeroFormPos, rho_diff, true))
                    + (1-flip)*sampleField(zeroFormPos, rho, true);
            p_temperature = flip*(p_temperature + sampleField(zeroFormPos, temperature_diff, true))
                    + (1-flip)*sampleField(zeroFormPos, temperature, true);
            {
                Vec2d drhodx = sampleGradientField(zeroFormPos, rho, true);
                Vec2d dTdx = sampleGradientField(zeroFormPos, temperature, true);
                cParticles[i].C_rho = Vec4d(p_rho, drhodx[0], drhodx[1], sampleCrossHessianField(zeroFormPos, rho, true));
                cParticles[i].C_temperature = Vec4d(p_temperature, dTdx[0], dTdx[1], sampleCrossHessianField(zeroFormPos, temperature, true));
            }
            cParticles[i].rho = p_rho;
            cParticles[i].temperature = p_temperature;
        }
    });
    std::cout << BLUE <<  "Particle information sampled from the grid!" << RESET << std::endl;
    std::cout << BLUE <<  "end of frame!" << RESET << std::endl;
}

void COFLIPSolver2D::advanceReflectionPOLYFLIP(double dt, int currentframe, int delayed_reinit_frequency)
{
    std::cout << BLUE <<  "R+POLYFLIP scheme frame " << currentframe << " starts !" << RESET << std::endl;
    getCFL();
    std::cout <<  "CFL: " << dt/_cfl << std::endl;

    Array2d u0, v0;
    u0 = u;
    v0 = v;
    Eigen::VectorX<double> change_in_fluxes(nF);
    change_in_fluxes.setZero();
    if (viscosity != 0.f) {
        tbb::parallel_for(tbb::blocked_range<int>(0,nF,TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
        {
            for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
                bool is_x_dir = tIdx < (ni+1)*nj;
                int j = is_x_dir ? tIdx / (ni+1) : (tIdx - (ni+1)*nj) / ni;
                int i = is_x_dir ? tIdx % (ni+1) : (tIdx - (ni+1)*nj) % ni;
                fluxes[tIdx] = is_x_dir ? u(i,j) : v(i,j);
            }
        });
        calculateCurl(true, true);
        change_in_fluxes = -(viscosity * dt / h) * Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t>(dtranspose_matrix_curl.transpose()) * prev_vorts0form;
        std::cout << GREEN << "change_in_fluxes.norm = " << change_in_fluxes.norm() << RESET << std::endl;
    }
    tbb::parallel_for(tbb::blocked_range<int>(0,cParticles.size(),TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
    {
        for (int p = r.begin(); p < r.end(); ++p) {
            Vec2d pos = cParticles[p].pos_current;
            pos = solveODE(0.5*dt, pos, u, v);
            clampPos(pos);
            cParticles[p].pos_current = pos;
        }
    });

    u_mass.assign(1e-4);
    v_mass.assign(1e-4);
    u.assign(0);
    v.assign(0);
    for(int i=0;i<cParticles.size();i++)
    {
        CmapParticles p = cParticles[i];
        Vec2d pos = p.pos_current;
        pos[0] = std::min(std::max(0+TOLERANCE, pos[0]), ((double)ni)*h-TOLERANCE);
        pos[1] = std::min(std::max(0.5*h+TOLERANCE, pos[1]), ((double)nj-0.5)*h-TOLERANCE);
        int ii, jj;
        ii = std::floor(pos[0]/h);
        jj = std::floor(pos[1]/h-0.5);
        for(int jjj=jj;jjj<=jj+1;jjj++)for(int iii=ii;iii<=ii+1;iii++)
        {
            Vec2d gpos = Vec2d(iii,jjj)*h + Vec2d(0,0.5)*h;
            double w = p.kernel((pos[0] - gpos[0])/h)*p.kernel((pos[1] - gpos[1])/h);
            double c0 = p.C_x[0];
            double c1 = p.C_x[1]*(gpos[0] - pos[0]);
            double c2 = p.C_x[2]*(gpos[1] - pos[1]);
            double c3 = p.C_x[3]*(gpos[0] - pos[0])*(gpos[1] - pos[1]);

            u(iii,jjj) += w * (c0 + c1 + c2 + c3);
            u_mass(iii,jjj) += w;
        }

        pos = p.pos_current;
        pos[0] = std::min(std::max(0.5*h+TOLERANCE, pos[0]), ((double)ni-0.5)*h-TOLERANCE);
        pos[1] = std::min(std::max(0+TOLERANCE, pos[1]), ((double)nj)*h-TOLERANCE);
        ii = std::floor(pos[0]/h - 0.5);
        jj = std::floor(pos[1]/h);
        for(int jjj=jj;jjj<=jj+1;jjj++)for(int iii=ii;iii<=ii+1;iii++)
        {
            Vec2d gpos = Vec2d(iii,jjj)*h + Vec2d(0.5,0)*h;
            double w = p.kernel((pos[0] - gpos[0])/h)*p.kernel((pos[1] - gpos[1])/h);
            double c0 = p.C_y[0];
            double c1 = p.C_y[1]*(gpos[0] - pos[0]);
            double c2 = p.C_y[2]*(gpos[1] - pos[1]);
            double c3 = p.C_y[3]*(gpos[0] - pos[0])*(gpos[1] - pos[1]);

            v(iii,jjj) += w * (c0 + c1 + c2 + c3);
            v_mass(iii,jjj) += w;
        }
    }
    u /= u_mass;
    v /= v_mass;

    Array2d u_save2, v_save2;
    u_save2 = u;
    v_save2 = v;
    applyBuoyancyForce(v, 0.5*dt);

    Array2d u_diff3, v_diff3;
    u_diff3 = u;
    v_diff3 = v;
    tbb::parallel_for(tbb::blocked_range<int>(0,nF,TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
    {
        for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
            bool is_x_dir = tIdx < (ni+1)*nj;
            int j = is_x_dir ? tIdx / (ni+1) : (tIdx - (ni+1)*nj) / ni;
            int i = is_x_dir ? tIdx % (ni+1) : (tIdx - (ni+1)*nj) % ni;
            if (is_x_dir) {
                u_diff3(i,j) = change_in_fluxes(tIdx);
            } else {
                v_diff3(i,j) = change_in_fluxes(tIdx);
            }
        }
    });
    u += u_diff3;
    v += v_diff3;

    Array2d u_save, v_save;
    u_save = u;
    v_save = v;
    double tol = TOLERANCE;
    projection(tol,use_neumann_boundary);
    Array2d u_diff, v_diff;
    u_diff = u;
    v_diff = v;
    u_diff -= u_save;
    v_diff -= v_save;
    double flip = 1.0;

    tbb::parallel_for(tbb::blocked_range<int>(0,cParticles.size(),TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
    {
        for (int i = r.begin(); i < r.end(); ++i) {
            Vec2d p_vel = cParticles[i].vel;
            Vec2d p_pos = cParticles[i].pos_current;
            Vec2d curr_dv = getVelocity(p_pos, u_diff, v_diff);
            Vec2d curr_dv3 = getVelocity(p_pos, u_diff3, v_diff3);
            p_vel = flip*(p_vel)
                    + (1-flip)*(getVelocity(p_pos, u_save2, v_save2))
                    + 2.*curr_dv + 2.*curr_dv3;

            if (alpha_buoyancy != 0. || beta_buoyancy != 0.) {
                Vec2d zeroFormPos = p_pos - h_uniform/((double)(rho.ni/ni))*Vec2d(0.5);
                p_vel[1] += dt*(-alpha_buoyancy*sampleField(zeroFormPos, rho, true) + beta_buoyancy*sampleField(zeroFormPos, temperature, true));
            }

            cParticles[i].vel = p_vel;
            cParticles[i].vel_temp = p_vel;

            {
                Eigen::Matrix2d dveldx = getJacobianVelocity(p_pos, u, v);
                Vec2d dudx(dveldx(0,0), dveldx(0,1));
                Vec2d dvdx(dveldx(1,0), dveldx(1,1));
                cParticles[i].C_x = Vec4d(
                    p_vel[0],
                    dudx[0],
                    dudx[1],
                    sampleCrossHessianField(p_pos - Vec2d(0.0, 0.5*h), u));
                cParticles[i].C_y = Vec4d(
                    p_vel[1],
                    dvdx[0],
                    dvdx[1],
                    sampleCrossHessianField(p_pos - Vec2d(0.5*h, 0.0), v));
            }
        }
    });

    u *= 2;
    u -= u0;
    v *= 2;
    v -= v0;

    tbb::parallel_for(tbb::blocked_range<int>(0,cParticles.size(),TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
    {
        for (int p = r.begin(); p < r.end(); ++p) {
            Vec2d pos = cParticles[p].pos_current;
            pos = solveODE(0.5*dt, pos, u, v);
            clampPos(pos);
            cParticles[p].pos_current = pos;
        }
    });
    
    Array2d rho_weight, T_weight;
    rho_weight.resize(rho.ni, rho.nj);
    T_weight.resize(temperature.ni, temperature.nj);
    rho_weight.assign(1e-4);
    T_weight.assign(1e-4);
    rho.assign(0);
    temperature.assign(0);
    u_mass.assign(1e-4);
    v_mass.assign(1e-4);
    u.assign(0);
    v.assign(0);
    for(int i=0;i<cParticles.size();i++)
    {
        CmapParticles p = cParticles[i];
        Vec2d pos = p.pos_current;
        pos[0] = std::min(std::max(0+TOLERANCE, pos[0]), ((double)ni)*h-TOLERANCE);
        pos[1] = std::min(std::max(0.5*h+TOLERANCE, pos[1]), ((double)nj-0.5)*h-TOLERANCE);
        int ii, jj;
        ii = std::floor(pos[0]/h);
        jj = std::floor(pos[1]/h-0.5);
        for(int jjj=jj;jjj<=jj+1;jjj++)for(int iii=ii;iii<=ii+1;iii++)
        {
            Vec2d gpos = Vec2d(iii,jjj)*h + Vec2d(0,0.5)*h;
            double w = p.kernel((pos[0] - gpos[0])/h)*p.kernel((pos[1] - gpos[1])/h);
            double c0 = p.C_x[0];
            double c1 = p.C_x[1]*(gpos[0] - pos[0]);
            double c2 = p.C_x[2]*(gpos[1] - pos[1]);
            double c3 = p.C_x[3]*(gpos[0] - pos[0])*(gpos[1] - pos[1]);

            u(iii,jjj) += w * (c0 + c1 + c2 + c3);
            u_mass(iii,jjj) += w;
        }

        pos = p.pos_current;
        pos[0] = std::min(std::max(0.5*h+TOLERANCE, pos[0]), ((double)ni-0.5)*h-TOLERANCE);
        pos[1] = std::min(std::max(0+TOLERANCE, pos[1]), ((double)nj)*h-TOLERANCE);
        ii = std::floor(pos[0]/h - 0.5);
        jj = std::floor(pos[1]/h);
        for(int jjj=jj;jjj<=jj+1;jjj++)for(int iii=ii;iii<=ii+1;iii++)
        {
            Vec2d gpos = Vec2d(iii,jjj)*h + Vec2d(0.5,0)*h;
            double w = p.kernel((pos[0] - gpos[0])/h)*p.kernel((pos[1] - gpos[1])/h);
            double c0 = p.C_y[0];
            double c1 = p.C_y[1]*(gpos[0] - pos[0]);
            double c2 = p.C_y[2]*(gpos[1] - pos[1]);
            double c3 = p.C_y[3]*(gpos[0] - pos[0])*(gpos[1] - pos[1]);

            v(iii,jjj) += w * (c0 + c1 + c2 + c3);
            v_mass(iii,jjj) += w;
        }

        pos = p.pos_current;
        double used_h = h / (double)(rho.ni/ni);
        pos[0] = std::min(std::max(0.5*used_h+TOLERANCE, pos[0]), ((double)rho.ni-0.5)*used_h-TOLERANCE);
        pos[1] = std::min(std::max(0.5*used_h+TOLERANCE, pos[1]), ((double)rho.nj-0.5)*used_h-TOLERANCE);
        ii = std::floor(pos[0]/used_h - 0.5);
        jj = std::floor(pos[1]/used_h - 0.5);
        for(int jjj=jj;jjj<=jj+1;jjj++)for(int iii=ii;iii<=ii+1;iii++)
        {
            Vec2d gpos = Vec2d(iii,jjj)*used_h + Vec2d(0.5,0.5)*used_h;
            double w = p.kernel((pos[0] - gpos[0])/used_h)*p.kernel((pos[1] - gpos[1])/used_h);

            double c0 = p.C_rho[0];
            double c1 = p.C_rho[1]*(gpos[0] - pos[0]);
            double c2 = p.C_rho[2]*(gpos[1] - pos[1]);
            double c3 = p.C_rho[3]*(gpos[0] - pos[0])*(gpos[1] - pos[1]);

            rho(iii,jjj) += w*(c0 + c1 + c2 + c3);
            rho_weight(iii,jjj) += w;
        }

        ii = std::floor(pos[0]/used_h - 0.5);
        jj = std::floor(pos[1]/used_h - 0.5);
        for(int jjj=jj;jjj<=jj+1;jjj++)for(int iii=ii;iii<=ii+1;iii++)
        {
            Vec2d gpos = Vec2d(iii,jjj)*used_h + Vec2d(0.5,0.5)*used_h;
            double w = p.kernel((pos[0] - gpos[0])/used_h)*p.kernel((pos[1] - gpos[1])/used_h);

            double c0 = p.C_temperature[0];
            double c1 = p.C_temperature[1]*(gpos[0] - pos[0]);
            double c2 = p.C_temperature[2]*(gpos[1] - pos[1]);
            double c3 = p.C_temperature[3]*(gpos[0] - pos[0])*(gpos[1] - pos[1]);

            temperature(iii,jjj) += w*(c0 + c1 + c2 + c3);
            T_weight(iii,jjj) += w;
        }
    }
    u /= u_mass;
    v /= v_mass;
    rho /= rho_weight;
    temperature /= T_weight;

    u_save = u;
    v_save = v;
    Array2d rho_save, temperature_save;
    rho_save = rho;
    temperature_save = temperature;
    projection(tol,use_neumann_boundary);
    Array2d rho_diff, temperature_diff;
    u_diff = u;
    v_diff = v;
    rho_diff = rho;
    temperature_diff = temperature;
    u_diff -= u_save;
    v_diff -= v_save;
    rho_diff -= rho_save;
    temperature_diff -= temperature_save;
    flip = ((do_particle_sample_after_first || currentframe != 0) && currentframe%((int)(delayed_reinit_frequency*substep))==0) ? 0.0 : 1.0;

    if (flip == 0.0) {
        if (do_particle_sample_after_first) {
            do_particle_sample_after_first = false;
            do_uniform_particle_seeding = false;
        }
        seedParticles(m_N);
    }

    tbb::parallel_for(tbb::blocked_range<int>(0,cParticles.size(),TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
    {
        for (int i = r.begin(); i < r.end(); ++i) {
            Vec2d p_vel = cParticles[i].vel;
            Vec2d p_pos = cParticles[i].pos_current;
            Vec2d curr_dv = getVelocity(p_pos, u_diff, v_diff);
            p_vel = flip*(p_vel)
                    + (1-flip)*(getVelocity(p_pos, u_save, v_save))
                    + curr_dv;

            cParticles[i].vel = p_vel;
            cParticles[i].vel_temp = p_vel;

            {
                Eigen::Matrix2d dveldx = getJacobianVelocity(p_pos, u, v);
                Vec2d dudx(dveldx(0,0), dveldx(0,1));
                Vec2d dvdx(dveldx(1,0), dveldx(1,1));
                cParticles[i].C_x = Vec4d(
                    p_vel[0],
                    dudx[0],
                    dudx[1],
                    sampleCrossHessianField(p_pos - Vec2d(0.0, 0.5*h), u));
                cParticles[i].C_y = Vec4d(
                    p_vel[1],
                    dvdx[0],
                    dvdx[1],
                    sampleCrossHessianField(p_pos - Vec2d(0.5*h, 0.0), v));
            }

            double p_rho = cParticles[i].rho;
            double p_temperature = cParticles[i].temperature;
            double used_h = h / (double)(rho.ni/ni);
            Vec2d zeroFormPos = p_pos - used_h*Vec2d(0.5,0.5);
            p_rho = flip*(p_rho + sampleField(zeroFormPos, rho_diff, true))
                    + (1-flip)*sampleField(zeroFormPos, rho, true);
            p_temperature = flip*(p_temperature + sampleField(zeroFormPos, temperature_diff, true))
                    + (1-flip)*sampleField(zeroFormPos, temperature, true);
            {
                Vec2d drhodx = sampleGradientField(zeroFormPos, rho, true);
                Vec2d dTdx = sampleGradientField(zeroFormPos, temperature, true);
                cParticles[i].C_rho = Vec4d(p_rho, drhodx[0], drhodx[1], sampleCrossHessianField(zeroFormPos, rho, true));
                cParticles[i].C_temperature = Vec4d(p_temperature, dTdx[0], dTdx[1], sampleCrossHessianField(zeroFormPos, temperature, true));
            }
            cParticles[i].rho = p_rho;
            cParticles[i].temperature = p_temperature;
        }
    });
}

void COFLIPSolver2D::advancePolyPIC(double dt, int currentframe)
{
    std::cout << BLUE <<  "PolyPIC scheme frame " << currentframe << " starts !" << RESET << std::endl;

    tbb::parallel_for(tbb::blocked_range<int>(0,cParticles.size(),TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
    {
        for (int p = r.begin(); p < r.end(); ++p) {
            Vec2d pos = solveODE(dt,cParticles[p].pos_current, u, v);
            cParticles[p].pos_current = pos;
        }
    });
    Array2d u_weight, v_weight, rho_weight, T_weight;
    u_weight.resize(u.ni,u.nj);
    v_weight.resize(v.ni,v.nj);
    rho_weight.resize(rho.ni, rho.nj);
    T_weight.resize(temperature.ni, temperature.nj);
    u_weight.assign(1e-4);
    v_weight.assign(1e-4);
    rho_weight.assign(1e-4);
    T_weight.assign(1e-4);
    rho.assign(0);
    temperature.assign(0);

    // splat particle properties to grid
    int numParticles = cParticles.size();
    Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> interp_matrix(2*numParticles, nF);
    Eigen::VectorXd point_circ(2*numParticles);
    interp_matrix.reserve(Eigen::VectorXi::Constant(2*numParticles,2*2));
    for(int i=0;i<numParticles;i++)
    {
        CmapParticles p = cParticles[i];
        Vec2d pos = p.pos_current;
        pos[0] = std::min(std::max(0+TOLERANCE, pos[0]), ((double)ni)*h-TOLERANCE);
        pos[1] = std::min(std::max(0.5*h+TOLERANCE, pos[1]), ((double)nj-0.5)*h-TOLERANCE);
        int ii, jj;
        ii = std::floor(pos[0]/h);
        jj = std::floor(pos[1]/h-0.5);
        for(int jjj=jj;jjj<=jj+1;jjj++)for(int iii=ii;iii<=ii+1;iii++)
        {
            Vec2d gpos = Vec2d(iii,jjj)*h + Vec2d(0,0.5)*h;
            double w = p.kernel((pos[0] - gpos[0])/h)*p.kernel((pos[1] - gpos[1])/h);

            double c0 = p.C_x[0];
            double c1 = p.C_x[1]*(gpos[0] - pos[0]);
            double c2 = p.C_x[2]*(gpos[1] - pos[1]);
            double c3 = p.C_x[3]*(gpos[0] - pos[0])*(gpos[1] - pos[1]);

            u(iii,jjj) += w * (c0 + c1 + c2 + c3);
            u_weight(iii,jjj) += w;
            int idx = jjj * (ni+1) + iii;
            interp_matrix.insert(i, idx) = w;
            point_circ(i) = p.vel[0];
        }

        pos = p.pos_current;
        pos[0] = std::min(std::max(0.5*h+TOLERANCE, pos[0]), ((double)ni-0.5)*h-TOLERANCE);
        pos[1] = std::min(std::max(0+TOLERANCE, pos[1]), ((double)nj)*h-TOLERANCE);
        ii = std::floor(pos[0]/h - 0.5);
        jj = std::floor(pos[1]/h);
        for(int jjj=jj;jjj<=jj+1;jjj++)for(int iii=ii;iii<=ii+1;iii++)
        {
            Vec2d gpos = Vec2d(iii,jjj)*h + Vec2d(0.5,0)*h;
            double w = p.kernel((pos[0] - gpos[0])/h)*p.kernel((pos[1] - gpos[1])/h);

            double c0 = p.C_y[0];
            double c1 = p.C_y[1]*(gpos[0] - pos[0]);
            double c2 = p.C_y[2]*(gpos[1] - pos[1]);
            double c3 = p.C_y[3]*(gpos[0] - pos[0])*(gpos[1] - pos[1]);

            v(iii,jjj) += w * (c0 + c1 + c2 + c3);
            v_weight(iii,jjj) += w;
            int idx = (ni+1)*nj + jjj * ni + iii;
            interp_matrix.insert(numParticles + i, idx) = w;
            point_circ(numParticles + i) = p.vel[1];
        }

        pos = p.pos_current;
        double used_h = h / (double)(rho.ni/ni);
        pos[0] = std::min(std::max(0.5*used_h+TOLERANCE, pos[0]), ((double)rho.ni-0.5)*used_h-TOLERANCE);
        pos[1] = std::min(std::max(0.5*used_h+TOLERANCE, pos[1]), ((double)rho.nj-0.5)*used_h-TOLERANCE);
        ii = std::floor(pos[0]/used_h - 0.5);
        jj = std::floor(pos[1]/used_h - 0.5);
        for(int jjj=jj;jjj<=jj+1;jjj++)for(int iii=ii;iii<=ii+1;iii++)
        {
            Vec2d gpos = Vec2d(iii,jjj)*used_h + Vec2d(0.5,0.5)*used_h;
            double w = p.kernel((pos[0] - gpos[0])/used_h)*p.kernel((pos[1] - gpos[1])/used_h);

            double c0 = p.C_rho[0];
            double c1 = p.C_rho[1]*(gpos[0] - pos[0]);
            double c2 = p.C_rho[2]*(gpos[1] - pos[1]);
            double c3 = p.C_rho[3]*(gpos[0] - pos[0])*(gpos[1] - pos[1]);

            rho(iii,jjj) += w*(c0 + c1 + c2 + c3);
            rho_weight(iii,jjj) += w;
        }

        ii = std::floor(pos[0]/used_h - 0.5);
        jj = std::floor(pos[1]/used_h - 0.5);
        for(int jjj=jj;jjj<=jj+1;jjj++)for(int iii=ii;iii<=ii+1;iii++)
        {
            Vec2d gpos = Vec2d(iii,jjj)*used_h + Vec2d(0.5,0.5)*used_h;
            double w = p.kernel((pos[0] - gpos[0])/used_h)*p.kernel((pos[1] - gpos[1])/used_h);

            double c0 = p.C_temperature[0];
            double c1 = p.C_temperature[1]*(gpos[0] - pos[0]);
            double c2 = p.C_temperature[2]*(gpos[1] - pos[1]);
            double c3 = p.C_temperature[3]*(gpos[0] - pos[0])*(gpos[1] - pos[1]);

            temperature(iii,jjj) += w*(c0 + c1 + c2 + c3);
            T_weight(iii,jjj) += w;
        }
    }
    interp_matrix.makeCompressed();

    u /= u_weight;
    v /= v_weight;
    rho /= rho_weight;
    temperature /= T_weight;

    applyBuoyancyForce(v, dt);
    double tol = TOLERANCE;
    projection(tol, use_neumann_boundary);

    // gather from grid
    tbb::parallel_for(tbb::blocked_range<int>(0,cParticles.size(),TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
    {
        for (int i = r.begin(); i < r.end(); ++i) {
            Vec2d pos = cParticles[i].pos_current;
            cParticles[i].vel = getVelocity(pos, u, v);
            // update Cp
            {
                double used_h = h / (double)(rho.ni/ni);
                Vec2d zeroFormPos = pos - used_h*Vec2d(0.5);
                cParticles[i].rho = sampleField(zeroFormPos, rho, true);
                cParticles[i].temperature = sampleField(zeroFormPos, temperature, true);
                Vec2d drhodx = sampleGradientField(zeroFormPos, rho, true);
                Vec2d dTdx = sampleGradientField(zeroFormPos, temperature, true);
                cParticles[i].C_rho = Vec4d(cParticles[i].rho, drhodx[0], drhodx[1], sampleCrossHessianField(zeroFormPos, rho, true));
                cParticles[i].C_temperature = Vec4d(cParticles[i].temperature, dTdx[0], dTdx[1], sampleCrossHessianField(zeroFormPos, temperature, true));
            }
        }
    });
}

void COFLIPSolver2D::seedParticles(int N, bool set_intensity_from_density)
{
    if (!do_uniform_particle_seeding) {
        calculateCurl(true, true);
    }
    // initialize a uniform distribution between 0 and 1
    std::uniform_real_distribution<double> unif(0, 1);
    cParticles.resize(0);
    for (int j = 0; j < njp; j++) for (int i = 0; i < nip; i++)
    {
        int used_num_particles_per_cell = N;
        if (!do_uniform_particle_seeding) {
            Vec2d pos = (Vec2d(i,j) + Vec2d(0.5)) * h;
            double alpha_curl = std::clamp(std::abs(sampleFieldBSpline0form(pos, curl)) / m_max_curl, 0., 1.);
            // +8 ensures at least a radius 2 times bigger than its standard deviation
            alpha_curl = std::clamp(std::exp(std::log(alpha_curl)+8.), 0., 1.);
            used_num_particles_per_cell = std::floor(lerp(min_PPC_count, N, alpha_curl) + 0.5);
        }
        int used_N = std::sqrt(used_num_particles_per_cell)+TOLERANCE;
        std::mt19937_64 rng;
        int tId = j * nip + i;
        rng.seed(tId);
        double x = (double)i*h;
        double y = (double)j*h;
        
        for(int jj=0;jj<used_N;jj++)
        {
            for(int ii=0;ii<used_N;ii++)
            {
                CmapParticles p;
                p.pos_current = Vec2d(x+((double)ii + unif(rng))/(double)used_N*h,y+((double)jj + unif(rng))/(double)used_N*h);
                p.pos_temp = p.pos_current;
                p.volume = 1./(double)used_num_particles_per_cell;
                cParticles.push_back(p);
            }
        }
        int remainder_particles_count = used_num_particles_per_cell - (used_N*used_N);
        for (int count = 0; count < remainder_particles_count; count++) {
            CmapParticles p;
            p.pos_current = Vec2d(x+unif(rng)*h,y+unif(rng)*h);
            p.pos_temp = p.pos_current;
            p.volume = 1./(double)used_num_particles_per_cell;
            cParticles.push_back(p);
        }
    }
    int numParticles = cParticles.size();
    std::cout << YELLOW << "particle count: " << numParticles << RESET << std::endl;
}

void COFLIPSolver2D::setSparseParticles(int N, double magn)
{
    cParticles.resize(N);
    double xdist = (nip*h) / (double)N;
    double ydist = (njp*h) / (double)N;
    std::uniform_real_distribution<double> unif(0, 1);
    std::mt19937_64 rng;
    rng.seed(0);
    int used_N = std::sqrt(N)+TOLERANCE;
    for(int i = 0; i < used_N; ++i) {
        for(int j = 0; j < used_N; ++j) {
            int idx = j*used_N+i;
            cParticles[idx].pos_current = Vec2d(0.5*xdist + xdist * i, 0.5*ydist + ydist * j);
            cParticles[idx].pos_temp = cParticles[idx].pos_current;
            std::cout << "p: " << cParticles[idx].pos_current[0] << " " << cParticles[idx].pos_current[1] << std::endl;
            cParticles[idx].vel = Vec2d(unif(rng), unif(rng)) * magn;
            cParticles[idx].vel_temp = cParticles[idx].vel;
            std::cout << "vel: " << cParticles[idx].vel[0] << " " << cParticles[idx].vel[1] << std::endl;
            cParticles[idx].longterm_pullback = Eigen::Matrix2d::Identity();
            cParticles[idx].shorterm_pullback = Eigen::Matrix2d::Identity();
            cParticles[idx].delta_t = 0.0;
            cParticles[idx].volume = 1.0;
        }
    }
    int numParticles = cParticles.size();
    std::cout << YELLOW << "particle count: " << numParticles << RESET << std::endl;
}

void COFLIPSolver2D::sampleParticlesFromGrid()
{
    int numParticles = cParticles.size();
    tbb::parallel_for(tbb::blocked_range<int>(0,numParticles,TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
    {
        for (int i = r.begin(); i < r.end(); ++i) {
            Vec2d pos = cParticles[i].pos_current;
            cParticles[i].vel = getVelocityBSpline(pos, u, v);
            cParticles[i].vel_temp = cParticles[i].vel;
            cParticles[i].longterm_pullback = Eigen::Matrix2d::Identity();
            cParticles[i].shorterm_pullback = Eigen::Matrix2d::Identity();
            cParticles[i].delta_t = 0.0;
            // update Cp
            if (sim_scheme != Scheme::CO_FLIP) {
                Eigen::Matrix2d dveldx = getJacobianVelocity(pos, u, v);
                Vec2d dudx(dveldx(0,0), dveldx(0,1));
                Vec2d dvdx(dveldx(1,0), dveldx(1,1));
                cParticles[i].C_x = Vec4d(
                    cParticles[i].vel[0],
                    dudx[0],
                    dudx[1],
                    sampleCrossHessianField(pos - Vec2d(0.0, 0.5*h), u));
                cParticles[i].C_y = Vec4d(
                    cParticles[i].vel[1],
                    dvdx[0],
                    dvdx[1],
                    sampleCrossHessianField(pos - Vec2d(0.5*h, 0.0), v));
            }

            {
                double used_h = h_uniform / (double)(rho.ni/ni);
                Vec2d zeroFormPos = pos - used_h*Vec2d(0.5, 0.5);
                cParticles[i].rho = sampleField(zeroFormPos, rho, true);
                cParticles[i].temperature = sampleField(zeroFormPos, temperature, true);
                Vec2d drhodx = sampleGradientField(zeroFormPos, rho, true);
                Vec2d dTdx = sampleGradientField(zeroFormPos, temperature, true);
                cParticles[i].C_rho = Vec4d(cParticles[i].rho, drhodx[0], drhodx[1], sampleCrossHessianField(zeroFormPos, rho, true));
                cParticles[i].C_temperature = Vec4d(cParticles[i].temperature, dTdx[0], dTdx[1], sampleCrossHessianField(zeroFormPos, temperature, true));
            }
        }
    });
}

double COFLIPSolver2D::maxVel()
{
    int res = 4;
    std::uniform_real_distribution<double> unif(0, 1);
    double max_v = tbb::parallel_reduce( 
        tbb::blocked_range<int>(0,nip*njp),
        TOLERANCE,
        [&](tbb::blocked_range<int> range, double running_max)
        {
            for (int tIdx=range.begin(); tIdx<range.end(); ++tIdx)
            {
                std::mt19937_64 rng;
                rng.seed(tIdx);
                int j = tIdx / nip;
                int i = tIdx % nip;
                double x = (double)i*h;
                double y = (double)j*h;
                for(int jj=0;jj<res;jj++)
                {
                    for(int ii=0;ii<res;ii++)
                    {
                        Vec<2, double> spos(x+((double)ii + unif(rng))/(double)res*h,
                                            y+((double)jj + unif(rng))/(double)res*h);
                        Vec<2, double> vel = getVelocityBSpline(spos, u, v);
                        running_max = std::max(running_max, mag(vel));
                    }
                }
            }

            return running_max;
        }, [](double a, double b) { return std::max(a,b); } );
    return max_v;
}

void COFLIPSolver2D::setSmoke(double smoke_rise, double smoke_drop)
{
    alpha_buoyancy = smoke_drop;
    beta_buoyancy = smoke_rise;
}

void COFLIPSolver2D::updateBackward(double dt, Array2d &back_x, Array2d &back_y)
{
    // backward mapping
    double sub_dt = _cfl;
    double T = dt;
    double t = 0;
    while(t < T)
    {
        if (t + sub_dt > T) sub_dt = T - t;
        map_tempx.assign(ni, nj, 0.0);
        map_tempy.assign(ni, nj, 0.0);
        semiLagAdvect(back_x, map_tempx, sub_dt, ni, nj, 0.5, 0.5);
        semiLagAdvect(back_y, map_tempy, sub_dt, ni, nj, 0.5, 0.5);
        back_x = map_tempx;
        back_y = map_tempy;
        t += sub_dt;
    }
}

Vec2d sampleTGV(const Vec2d& pos) {
    return Vec2d(sin(pos[0])*cos(pos[1]), -cos(pos[0])*sin(pos[1]));
}

double sampleLeapfrog(const Vec2d& pos, double amp, double dist_a, double dist_b, double radius) {
    Vec2d vort_pos0 = Vec2d(-0.5*dist_a,-0.5*M_PI);
    Vec2d vort_pos1 = Vec2d(+0.5*dist_a,-0.5*M_PI);
    Vec2d vort_pos2 = Vec2d(-0.5*dist_b,-0.5*M_PI);
    Vec2d vort_pos3 = Vec2d(+0.5*dist_b,-0.5*M_PI);
    double r_sqr0 = dist2(pos - Vec2d(M_PI), vort_pos0);
    double r_sqr1 = dist2(pos - Vec2d(M_PI), vort_pos1);
    double r_sqr2 = dist2(pos - Vec2d(M_PI), vort_pos2);
    double r_sqr3 = dist2(pos - Vec2d(M_PI), vort_pos3);
    double c_a = amp*1000.0/(2.0*M_PI)*exp(-0.5*(r_sqr0)/(radius*radius));
    double c_b = -amp*1000.0/(2.0*M_PI)*exp(-0.5*(r_sqr1)/(radius*radius));
    double c_c = amp*1000.0/(2.0*M_PI)*exp(-0.5*(r_sqr2)/(radius*radius));
    double c_d = -amp*1000.0/(2.0*M_PI)*exp(-0.5*(r_sqr3)/(radius*radius));
    return c_a + c_b + c_c + c_d;
}

Vec2d sampleVortexSheet(const Vec2d& pos, double radius, double rotational_speed, double eps_smooth_gap) {
    Vec2d pos_from_center = pos - Vec2d(M_PI);
    double mag_pos = mag(pos_from_center);
    double t = std::clamp((mag_pos-(radius-eps_smooth_gap))/(2.*eps_smooth_gap),0.,1.);
    double smooth_factor = 1.-t*t*(3.-2.*t);
    return rotational_speed * smooth_factor * Vec2d(-pos_from_center[1], pos_from_center[0]);
}

void COFLIPSolver2D::setInitVelocityVortexSheet(double radius, double rotational_speed, double eps_smooth_gap)
{
    if (sim_scheme != Scheme::CO_FLIP) {
        tbb::parallel_for(tbb::blocked_range2d<int,int>{0,ni+1,1,0,nj,1}, [&](const tbb::blocked_range2d<int,int> &r)
        {
            int ie = r.rows().end();
            int je = r.cols().end();
            for (int j = r.cols().begin(); j < je; ++j) {
                for (int i = r.rows().begin(); i < ie; ++i) {
                    Vec2d pos = h * Vec2d(i, j) + h * Vec2d(0.0, 0.5);
                    u(i,j) = sampleVortexSheet(pos, radius, rotational_speed, eps_smooth_gap)[0];
                }
            }
        });
        tbb::parallel_for(tbb::blocked_range2d<int,int>{0,ni,1,0,nj+1,1}, [&](const tbb::blocked_range2d<int,int> &r)
        {
            int ie = r.rows().end();
            int je = r.cols().end();
            for (int j = r.cols().begin(); j < je; ++j) {
                for (int i = r.rows().begin(); i < ie; ++i) {
                    Vec2d pos = h * Vec2d(i, j) + h * Vec2d(0.5, 0.0);
                    v(i,j) = sampleVortexSheet(pos, radius, rotational_speed, eps_smooth_gap)[1];
                }
            }
        });
    } else {
        seedParticles(m_N);
        tbb::parallel_for(tbb::blocked_range<int>(0,cParticles.size(),TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
        {
            for (int i = r.begin(); i < r.end(); ++i) {
                Vec2d pos = cParticles[i].pos_current;
                Vec2d result = sampleVortexSheet(pos, radius, rotational_speed, eps_smooth_gap);
                cParticles[i].vel = result;
                cParticles[i].vel_temp = cParticles[i].vel;
            }
        });
        solveInterpDaggerVelocity(0, 0, 1);
        tbb::parallel_for(tbb::blocked_range<int>(0,nF,TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
        {
            for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
                bool is_x_dir = tIdx < (ni+1)*nj;
                int j = is_x_dir ? tIdx / (ni+1) : (tIdx - (ni+1)*nj) / ni;
                int i = is_x_dir ? tIdx % (ni+1) : (tIdx - (ni+1)*nj) % ni;
                if (is_x_dir) {
                    u(i,j) = fluxes(tIdx);
                } else {
                    v(i,j) = fluxes(tIdx);
                }
            }
        });
    }

    rho.assign(0.);
    temperature.assign(0.);
    tbb::parallel_for(tbb::blocked_range2d<int,int>{0,rho.ni,1,0,rho.nj,1}, [&](const tbb::blocked_range2d<int,int> &r)
    {
        int ie = r.rows().end();
        int je = r.cols().end();
        for (int j = r.cols().begin(); j < je; ++j) {
            for (int i = r.rows().begin(); i < ie; ++i) {
                int ratio = rho.ni/ni;
                Vec2d pos = (h_uniform/(double)ratio)*(Vec2d(i,j) + Vec2d(0.5, 0.5));
                double mag_pos = mag(pos - Vec2d(M_PI));
                if (mag_pos < radius) {
                    rho(i,j) = 1.;
                } else {
                    temperature(i,j) = 1.;
                }
            }
        }
    });

    return;
}

void COFLIPSolver2D::setInitLeapFrog(double amp, double dist_a, double dist_b, double rho_h, double rho_w)
{
    double a = 2.25*2.*M_PI/256.;
    vorts.setZero();
    //initialize curl;
    if (sim_scheme != Scheme::CO_FLIP) {
        tbb::parallel_for(tbb::blocked_range2d<int,int>{0,ni+1,1,0,nj+1,1}, [&](const tbb::blocked_range2d<int,int> &r)
        {
            int ie = r.rows().end();
            int je = r.cols().end();
            for (int j = r.cols().begin(); j < je; ++j) {
                for (int i = r.rows().begin(); i < ie; ++i) {
                    Vec2d pos = Vec2d(i, j)*h;
                    int tIdx = j*(ni+1) + i;
                    vorts[tIdx] = sampleLeapfrog(pos, amp, dist_a, dist_b, a);
                }
            }
        });
    } else {
        bool do_uniform_particle_seeding_save = do_uniform_particle_seeding;
        do_uniform_particle_seeding = true;
        seedParticles(m_N);
        do_uniform_particle_seeding = do_uniform_particle_seeding_save;
        tbb::parallel_for(tbb::blocked_range<int>(0,cParticles.size(),TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
        {
            for (int i = r.begin(); i < r.end(); ++i) {
                Vec2d pos = cParticles[i].pos_current;
                cParticles[i].vel = Vec2d(sampleLeapfrog(pos, amp, dist_a, dist_b, a));
                cParticles[i].vel_temp = cParticles[i].vel;
            }
        });
        solveInterpDaggerVorticity();
        vorts = starvort_matrix * prev_vorts0form;
    }

    rhs = vorts * h;
    double res_out; int iter_out;
    bool converged = AMGPCGSolvePrebuilt2D(A_L_curl[0],rhs,streamfunction,A_L_curl,R_L_curl,P_L_curl,S_L_curl,total_level_curl,TOLERANCE,MAX_ITERATIONS,res_out,iter_out,ni+1,nj+1, false, true);
    if (!converged)
        std::cout << "WARNING: Streamfunction-vorticity solve failed!************************************************" << std::endl;
    std::cout << "#iteration:      " << iter_out << std::endl;
    std::cout << "estimated error: " << res_out << std::endl;

    //compute u = curl psi
    tbb::parallel_for(tbb::blocked_range2d<int,int>{0,ni+1,1,0,nj+1,1}, [&](const tbb::blocked_range2d<int,int> &r)
    {
        int ie = r.rows().end();
        int je = r.cols().end();
        for (int j = r.cols().begin(); j < je; ++j) {
            for (int i = r.rows().begin(); i < ie; ++i) {
                if (boundaryMask_nodes(i,j) != 0) {
                    int tIdx = j*(ni+1) + i;
                    streamfunction[tIdx] = 0;
                }
            }
        }
    });
    tbb::parallel_for(tbb::blocked_range2d<int,int>{0,ni+1,1,0,nj,1}, [&](const tbb::blocked_range2d<int,int> &r)
    {
        int ie = r.rows().end();
        int je = r.cols().end();
        for (int j = r.cols().begin(); j < je; ++j) {
            for (int i = r.rows().begin(); i < ie; ++i) {
                double value = (streamfunction[(j+1)*(ni+1)+i] - streamfunction[j*(ni+1)+i]);
                u(i,j) = value;
            }
        }
    });
    tbb::parallel_for(tbb::blocked_range2d<int,int>{0,ni,1,0,nj+1,1}, [&](const tbb::blocked_range2d<int,int> &r)
    {
        int ie = r.rows().end();
        int je = r.cols().end();
        for (int j = r.cols().begin(); j < je; ++j) {
            for (int i = r.rows().begin(); i < ie; ++i) {
                double value = -(streamfunction[j*(ni+1)+(i+1)] - streamfunction[j*(ni+1)+i]);
                v(i,j) = value;
            }
        }
    });

    calculateCurl(true, true);
    std::cout  << GREEN << "fluxes.norm = " << fluxes.norm() << ", circulations.norm = " << circulations.norm()  << ", vorts.norm = " << vorts.norm() << RESET << std::endl;

    double avg_div = 0.0;
    double max_div = -1.0;
    for(int tIdx = 0; tIdx < nC; tIdx++)
    {
        int i = tIdx%ni;
        int j = tIdx / ni;
        double div = std::abs(u(i+1,j) - u(i,j) + v(i,j+1) - v(i,j)) / h;
        avg_div += div;
        max_div = std::max(div, max_div);
    }
    std::cout << RED << "max_div = " << max_div << ", avg_div = " << avg_div/(double)(nC) << RESET << std::endl;

    rho.assign(0);
    temperature.assign(0);
    tbb::parallel_for(tbb::blocked_range2d<int,int>{0,ni,1,0,nj,1}, [&](const tbb::blocked_range2d<int,int> &r)
    {
        int ie = r.rows().end();
        int je = r.cols().end();
        for (int j = r.cols().begin(); j < je; ++j) {
            for (int i = r.rows().begin(); i < ie; ++i) {
                Vec2d pos = h*(Vec2d(i,j) + Vec2d(0.5, 0.5));
                if (rho_h - rho_w < pos[1] && pos[1] < rho_h + rho_w && pos[0] > rho_w && pos[0] < 2*M_PI - rho_w)
                {
                    for (int ii = (rho.ni/ni)*i; ii < (rho.ni/ni)*i+(rho.ni/ni); ii++) {
                        for (int jj = (rho.nj/nj)*j; jj < (rho.nj/nj)*j+(rho.nj/nj); jj++) {
                            rho(ii,jj) = 1;
                        }
                    }
                }
                else {
                    for (int ii = (rho.ni/ni)*i; ii < (rho.ni/ni)*i+(rho.ni/ni); ii++) {
                        for (int jj = (rho.nj/nj)*j; jj < (rho.nj/nj)*j+(rho.nj/nj); jj++) {
                            temperature(ii,jj) = 1;
                        }
                    }
                }
            }
        }
    });
}

void COFLIPSolver2D::setInitReyleighTaylor(double layer_height)
{
    rho.assign(0.);
    temperature.assign(0.);
    tbb::parallel_for(tbb::blocked_range2d<int,int>{0,rho.ni,1,0,rho.nj,1}, [&](const tbb::blocked_range2d<int,int> &r)
    {
        int ie = r.rows().end();
        int je = r.cols().end();
        for (int j = r.cols().begin(); j < je; ++j) {
            for (int i = r.rows().begin(); i < ie; ++i) {
                int ratio = rho.ni/ni;
                Vec2d pos = (h_uniform/(double)ratio)*(Vec2d(i,j) + Vec2d(0.5, 0.5));
                double measured_L = (h*(double)nip);
                double height = measured_L*((double)nj/(double)ni)*0.5 + (std::cos(2.*M_PI*pos[0]/measured_L)-1.)*layer_height;
                if (pos[1] < height) {
                    temperature(i,j) = 1.;
                } else {
                    rho(i,j) = 1.;
                }
            }
        }
    });
}

void COFLIPSolver2D::setBoundaryMask(std::function<double(Vec2d pos)> sdf)
{
    tbb::parallel_for(tbb::blocked_range2d<int,int>{0,ni,1,0,nj,1}, [&](const tbb::blocked_range2d<int,int> &r)
    {
        int ie = r.rows().end();
        int je = r.cols().end();
        for (int j = r.cols().begin(); j < je; ++j) {
            for (int i = r.rows().begin(); i < ie; ++i) {
                //fluid=0, air=1, boundary=2, obstacle=3
                boundaryMask(i,j) = 0;

                Vec2d pos = h_uniform * Vec2d(i, j) + h_uniform * Vec2d(0.5); 
                if (sdf && sdf(pos) <= 0) boundaryMask(i,j) = 3;
            }
        }
    });
    tbb::parallel_for(tbb::blocked_range2d<int,int>{0,ni+1,1,0,nj+1,1}, [&](const tbb::blocked_range2d<int,int> &r)
    {
        int ie = r.rows().end();
        int je = r.cols().end();
        for (int j = r.cols().begin(); j < je; ++j) {
            for (int i = r.rows().begin(); i < ie; ++i) {
                //fluid=0, air=1, boundary=2, obstacle=3
                if (i == 0 || i == ni || j == 0 || j == nj) {
                    boundaryMask_nodes(i,j) = 2;
                } else if (boundaryMask(i,j) == 0 && boundaryMask(i-1,j) == 0 && boundaryMask(i,j-1) == 0 && boundaryMask(i-1,j-1) == 0) {
                    boundaryMask_nodes(i,j) = 0;
                } else if (boundaryMask(i,j) == 3 || boundaryMask(i-1,j) == 3 || boundaryMask(i,j-1) == 3 || boundaryMask(i-1,j-1) == 3) {
                    boundaryMask_nodes(i,j) = 3;
                } else {
                    boundaryMask_nodes(i,j) = 2;
                }
            }
        }
    });
}

void COFLIPSolver2D::buildMultiGrid(bool PURE_NEUMANN)
{
    //build the matrix
    //we are assuming a a whole fluid domain
    {
        tbb::enumerable_thread_specific<std::vector<Eigen::Triplet<double> > > parallelTripletList_dmatrix;
        tbb::enumerable_thread_specific<std::vector<Eigen::Triplet<double> > > parallelTripletList_invstarflux;
        tbb::parallel_for(tbb::blocked_range<int>(0,nF,TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
        {
            auto& localTripletList_dmatrix = parallelTripletList_dmatrix.local();
            auto& localTripletList_invstarflux = parallelTripletList_invstarflux.local();
            for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
                bool is_x_dir = tIdx < (ni+1)*nj;
                int j = is_x_dir ? tIdx / (ni+1) : (tIdx - (ni+1)*nj) / ni;
                int i = is_x_dir ? tIdx % (ni+1) : (tIdx - (ni+1)*nj) % ni;
                if (is_x_dir && i == 0) {
                    if (!PURE_NEUMANN) localTripletList_dmatrix.emplace_back(tIdx, j*ni+i, 1.0);
                } else if (is_x_dir && i == ni) {
                    if (!PURE_NEUMANN) localTripletList_dmatrix.emplace_back(tIdx, j*ni+(i-1), -1.0);
                } else if (!is_x_dir && j == 0) {
                    if (!PURE_NEUMANN) localTripletList_dmatrix.emplace_back(tIdx, j*ni+i, 1.0);
                } else if (!is_x_dir && j == nj) {
                    if (!PURE_NEUMANN) localTripletList_dmatrix.emplace_back(tIdx, (j-1)*ni+i, -1.0);
                } else if ((is_x_dir && boundaryMask(i,j) == 0 && boundaryMask(i-1,j) == 0) ||
                        (!is_x_dir && boundaryMask(i,j) == 0 && boundaryMask(i,j-1) == 0)) {
                    localTripletList_dmatrix.emplace_back(tIdx,is_x_dir ? j*ni+(i-1) : (j-1)*ni+i, -1.0);
                    localTripletList_dmatrix.emplace_back(tIdx,j*ni+i, 1.0);
                }
                int idx = is_x_dir ? i : j;
                int nidx = is_x_dir ? ni : nj;
                int other_idx = is_x_dir ? j : i;
                int other_nidx = is_x_dir ? nj : ni;
                if (sim_scheme == Scheme::CO_FLIP) {
                    if (use_DEC_diagonal_hodge_star) {
                        double factor = bs_p == 3 ? ((idx == 0 || idx == nidx) ? 1./3. : ((idx == 1 || idx == nidx-1) ? 0.5 : ((idx == 2 || idx == nidx-2) ? 5./6. : 1.0))) / ((other_idx == 0 || other_idx == other_nidx-1) ? 1./3. : ((other_idx == 1 || other_idx == other_nidx-2) ? 2./3. : 1.0)) : (bs_p == 2 ? ((idx == 0 || idx == nidx) ? 0.5 : ((idx == 1 || idx == nidx-1) ? 0.75 : 1.0)) / ((other_idx == 0 || other_idx == other_nidx-1) ? 0.5 : 1.0) : 1.0);
                        localTripletList_invstarflux.emplace_back(tIdx, tIdx, factor);
                    } else {
                        std::vector<std::array<double, 7> > Bs = bs_p == 3 ? B3s : (bs_p == 2 ? B2s : B1s);
                        std::vector<std::array<double, 7> > Ds = bs_p == 3 ? D2s : (bs_p == 2 ? D1s : D0s);
                        std::array<double, 7> Barray = Bs[std::min(std::min(2*bs_p-1, idx), nidx-idx)];
                        std::array<double, 7> Darray = Ds[std::min(std::min(2*(bs_p-1), other_idx), (other_nidx-1)-other_idx)];
                        if (nidx-idx < 2*bs_p-1) std::reverse(Barray.begin(), Barray.end());
                        if ((other_nidx-1)-other_idx < (2*(bs_p-1))) std::reverse(Darray.begin(), Darray.end());
                        for (int l = -bs_p; l <= bs_p; ++l) {
                            for (int k = -(bs_p-1); k <= (bs_p-1); ++k) {
                                int jdx = idx + l;
                                int other_jdx = other_idx + k;
                                if (jdx >= 0 && jdx <= nidx && other_jdx >=0 && other_jdx < other_nidx) {
                                    int tJdx = tIdx + l * (is_x_dir ? 1 : ni) + k * (is_x_dir ? (ni+1) : 1);
                                    double factor = Barray[l+3]*Darray[k+3];
                                    localTripletList_invstarflux.emplace_back(tIdx, tJdx, factor);
                                }
                            }
                        }
                    }
                } else {
                    localTripletList_invstarflux.emplace_back(tIdx, tIdx, 1.0);
                }
            }
        });
        std::vector<Eigen::Triplet<double> > tripletList_dmatrix;
        std::vector<Eigen::Triplet<double> > tripletList_invstarflux;
        HELPER::mergeLocalThreadVectors(tripletList_dmatrix, parallelTripletList_dmatrix);
        HELPER::mergeLocalThreadVectors(tripletList_invstarflux, parallelTripletList_invstarflux);
        Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> dmatrix(nF,nC);
        dmatrix.setFromTriplets(tripletList_dmatrix.begin(), tripletList_dmatrix.end());
        dtranspose_matrix = dmatrix.transpose();
        invstarflux_matrix.resize(nF,nF);
        invstarflux_matrix.setFromTriplets(tripletList_invstarflux.begin(), tripletList_invstarflux.end());
    }
    if (sim_scheme == Scheme::CO_FLIP && !use_DEC_diagonal_hodge_star) {
        HELPER::rowSumMatrix(invstarflux_matrix,invstarflux_matrix);
        invstarflux_matrix.diagonal().array() = 1. / invstarflux_matrix.diagonal().array();
    }
    Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> pressure_laplacian_matrix = dtranspose_matrix * invstarflux_matrix * dtranspose_matrix.transpose();
    almostIdentityMatrixCells.resize(nC,nC);
    almostIdentityMatrixCells.reserve(Eigen::VectorXi::Constant(nC,1));
    tbb::parallel_for(tbb::blocked_range<int>(0,nC, TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
    {
        for (int i = r.begin(); i < r.end(); ++i) {
            bool have_diagonal = false;
            for(typename Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t>::InnerIterator it(pressure_laplacian_matrix,i); it; ++it) {
                if (it.row() == it.col()) {
                    have_diagonal = true;
                }
            }
            if (!have_diagonal) {
                almostIdentityMatrixCells.insert(i,i) = 1.;
            }
        }
    });
    almostIdentityMatrixCells.makeCompressed();
    pressure_laplacian_matrix = pressure_laplacian_matrix + almostIdentityMatrixCells;
    mgLevelGenerator.generateLevelsGalerkinCoarsening2D(A_L, R_L, P_L, S_L, total_level, pressure_laplacian_matrix, ni, nj);
    std::cout << "done building PP-laplacian matrix!" << std::endl;
}

void COFLIPSolver2D::applyVelocityBoundary(bool do_set_obstacle_vel)
{
    tbb::parallel_for(tbb::blocked_range2d<int,int>{0,ni,1,0,nj,1}, [&](const tbb::blocked_range2d<int,int> &r)
    {
        int ie = r.rows().end();
        int je = r.cols().end();
        for (int j = r.cols().begin(); j < je; ++j) {
            for (int i = r.rows().begin(); i < ie; ++i) {
                if (use_neumann_boundary)
                {
                    if(i==0)
                    {
                        u(i,j) = 0;
                    }
                    if(j==0)
                    {
                        v(i, j) = 0;
                    }
                    if(i==ni-1)
                    {
                        u(i + 1, j) = 0;
                    }
                    if(j==nj-1)
                    {
                        v(i, j + 1) = 0;
                    }
                }

                if (do_set_obstacle_vel && boundaryMask(i,j) == 3)
                {
                    u(i,j) = 0;
                    u(i+1,j) = 0;
                    v(i, j) = 0;
                    v(i, j+1) = 0;
                }
            }
        }
    });
}

void COFLIPSolver2D::takeDualwrtStar(Array2d &_u, Array2d &_v, bool update_uv, bool flux2circulation_or_circulation2flux) {
    tbb::parallel_for(tbb::blocked_range<int>(0,nF,TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
    {
        for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
            bool is_x_dir = tIdx < (ni+1)*nj;
            int j = is_x_dir ? tIdx / (ni+1) : (tIdx - (ni+1)*nj) / ni;
            int i = is_x_dir ? tIdx % (ni+1) : (tIdx - (ni+1)*nj) % ni;
            double value = is_x_dir ? _u(i,j) : _v(i,j);
            if (flux2circulation_or_circulation2flux == false)
                fluxes[tIdx] = value;
            else
                circulations[tIdx] = value;
        }
    });
    if (flux2circulation_or_circulation2flux == false)
        circulations = starflux_matrix * fluxes;
    else {
        starflux_cg.setTolerance(TOLERANCE);
        starflux_cg.setMaxIterations(MAX_ITERATIONS);
        fluxes = starflux_cg.solveWithGuess(circulations, fluxes);
        if (starflux_cg.info() == Eigen::Success) {
            std::cout << "starflux_cg solver success!" << std::endl;
        } else {
            std::cout << "starflux_cg solver FAILED!!!" << std::endl;
        }
        std::cout << "#iteration:      " << starflux_cg.iterations() << std::endl;
        std::cout << "estimated error: " << starflux_cg.error() << std::endl;
    }
    if (update_uv) {
        tbb::parallel_for(tbb::blocked_range<int>(0,nF,TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
        {
            for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
                bool is_x_dir = tIdx < (ni+1)*nj;
                int j = is_x_dir ? tIdx / (ni+1) : (tIdx - (ni+1)*nj) / ni;
                int i = is_x_dir ? tIdx % (ni+1) : (tIdx - (ni+1)*nj) % ni;
                if (flux2circulation_or_circulation2flux == false) {
                    if (is_x_dir) {
                        _u(i,j) = circulations[tIdx];
                    } else {
                        _v(i,j) = circulations[tIdx];
                    }
                } else {
                    if (is_x_dir) {
                        _u(i,j) = fluxes[tIdx];
                    } else {
                        _v(i,j) = fluxes[tIdx];
                    }
                }
            }
        });
    }
}

void COFLIPSolver2D::calculateCurl(bool do_star_fluxes, bool calculate_starvort_inverse) {
    
    if (do_star_fluxes) {
        takeDualwrtStar(u, v, false, false);
    } else {
        tbb::parallel_for(tbb::blocked_range<int>(0,nF,TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
        {
            for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
                bool is_x_dir = tIdx < (ni+1)*nj;
                int j = is_x_dir ? tIdx / (ni+1) : (tIdx - (ni+1)*nj) / ni;
                int i = is_x_dir ? tIdx % (ni+1) : (tIdx - (ni+1)*nj) % ni;
                circulations[tIdx] = is_x_dir ? u(i,j) : v(i,j);
            }
        });
    }
    vorts = dtranspose_matrix_curl * circulations / h;

    if (calculate_starvort_inverse) {
        starvort_cg.setTolerance(TOLERANCE);
        starvort_cg.setMaxIterations(MAX_ITERATIONS);
        prev_vorts0form = starvort_cg.solveWithGuess(vorts, prev_vorts0form);
        if (starvort_cg.info() == Eigen::Success) {
            std::cout << "#iteration:      " << starvort_cg.iterations() << std::endl;
            std::cout << "estimated error: " << starvort_cg.error() << std::endl;
        } else {
            std::cout << "starvort_cg solver FAILED!!!" << std::endl;
        }
        tbb::parallel_for(tbb::blocked_range2d<int,int>{0,ni+1,1,0,nj+1,1}, [&](const tbb::blocked_range2d<int,int> &r)
        {
            int ie = r.rows().end();
            int je = r.cols().end();
            for (int j = r.cols().begin(); j < je; ++j) {
                for (int i = r.rows().begin(); i < ie; ++i) {
                    int tIdx = j*(ni+1)+i;
                    curl(i,j) = prev_vorts0form(tIdx);
                }
            }
        });
        int res = 4;
        std::uniform_real_distribution<double> unif(0, 1);
        m_max_curl = tbb::parallel_reduce( 
            tbb::blocked_range<int>(0,nip*njp),
            TOLERANCE,
            [&](tbb::blocked_range<int> range, double running_max)
            {
                for (int tIdx=range.begin(); tIdx<range.end(); ++tIdx)
                {
                    std::mt19937_64 rng;
                    rng.seed(tIdx);
                    int j = tIdx / nip;
                    int i = tIdx % nip;
                    double x = (double)i*h;
                    double y = (double)j*h;
                    for(int jj=0;jj<res;jj++)
                    {
                        for(int ii=0;ii<res;ii++)
                        {
                            Vec<2, double> spos(x+((double)ii + unif(rng))/(double)res*h,
                                                y+((double)jj + unif(rng))/(double)res*h);
                            double vort = sampleFieldBSpline0form(spos, curl);
                            running_max = std::max(running_max, std::abs(vort));
                        }
                    }
                }

                return running_max;
            }, [](double a, double b) { return std::max(a,b); } );
        cBar = color_bar(m_max_curl);
    }
}

void COFLIPSolver2D::emitSmoke()
{
    tbb::parallel_for(tbb::blocked_range2d<int,int>{0,ni,1,0,nj,1}, [&](const tbb::blocked_range2d<int,int> &r)
    {
        int ie = r.rows().end();
        int je = r.cols().end();
        for (int j = r.cols().begin(); j < je; ++j) {
            for (int i = r.rows().begin(); i < ie; ++i) {
                if (emitterMask(i, j) == 1)
                {
                    for (int ii = (rho.ni/ni)*i; ii < (rho.ni/ni)*i+(rho.ni/ni); ii++) {
                        for (int jj = (rho.nj/nj)*j; jj < (rho.nj/nj)*j+(rho.nj/nj); jj++) {
                            rho(ii, jj) += 0.01;
                            temperature(ii, jj) = std::clamp(temperature(ii, jj)-0.01, 0.,1.);
                        }
                    }
                }
            }
        }
    });
}

Vec2d COFLIPSolver2D::getVelocity(const Vec2d& pos, const Array2d& un, const Array2d& vn)
{
    double u_sample, v_sample;
    //offset of u, we are in a staggered grid
    Vec2d upos = pos - Vec2d(0.0, 0.5*h);
    u_sample = sampleField(upos, un);

    //offset of v, we are in a staggered grid
    Vec2d vpos = pos - Vec2d(0.5*h, 0.0);
    v_sample = sampleField(vpos, vn);

    return Vec2d(u_sample, v_sample);
}

Eigen::Matrix2d COFLIPSolver2D::getJacobianVelocity(const Vec2d& pos, const Array2d& un, const Array2d& vn)
{
    Vec2d grad_u_sample, grad_v_sample;
    //offset of u, we are in a staggered grid
    Vec2d upos = pos - Vec2d(0.0, 0.5*h);
    grad_u_sample = sampleGradientField(upos, un);

    //offset of v, we are in a staggered grid
    Vec2d vpos = pos - Vec2d(0.5*h, 0.0);
    grad_v_sample = sampleGradientField(vpos, vn);

    return (Eigen::Matrix2d() << grad_u_sample[0], grad_u_sample[1], 
                 grad_v_sample[0], grad_v_sample[1]).finished();
}

Vec2d COFLIPSolver2D::getPointwiseDivCurl(const Vec2d& pos, const Array2d& un, const Array2d& vn)
{
    Eigen::Matrix2d jac;
    if (bs_p == 1) {
        jac = getJacobianVelocity(pos, un, vn);
    } else {
        jac = getJacobianVelocityBSpline(pos, un, vn);
    }
    double div_part = jac.trace();
    double curl_part = jac(1,0) - jac(0,1);
    return Vec2d(div_part, curl_part);
}

double COFLIPSolver2D::sampleField(const Vec2d& pos, const Array2d& field, bool use_uniform)
{
    Vec2d spos = pos;
    double used_h = use_uniform ? h_uniform : h;
    used_h /= use_uniform ? (double)(field.ni/ni) : 1.;
    int i = std::floor(spos.v[0] / used_h), j = std::floor(spos.v[1] / used_h);
    return bilerp(field.boundedAt(i, j), field.boundedAt(i + 1, j),
                  field.boundedAt(i, j + 1), field.boundedAt(i + 1, j + 1), spos.v[0] / used_h - (double)i, spos.v[1] / used_h - (double)j);
}

double COFLIPSolver2D::sampleCrossHessianField(const Vec2d& pos, const Array2d& field, bool use_uniform)
{
    Vec2d spos = pos;
    double used_h = use_uniform ? h_uniform : h;
    used_h /= use_uniform ? (double)(field.ni/ni) : 1.;
    int i = std::floor(spos.v[0] / used_h), j = std::floor(spos.v[1] / used_h);
    return (field.boundedAt(i + 1, j + 1) - field.boundedAt(i, j + 1) -
            field.boundedAt(i + 1, j) + field.boundedAt(i, j)) / (used_h*used_h);
}

Vec2d COFLIPSolver2D::sampleGradientField(const Vec2d& pos, const Array2d& field, bool use_uniform)
{
    Vec2d spos = pos;
    double used_h = use_uniform ? h_uniform : h;
    used_h /= use_uniform ? (double)(field.ni/ni) : 1.;
    int i = std::floor(spos.v[0] / used_h), j = std::floor(spos.v[1] / used_h);
    double alpha = spos.v[0] / used_h - (double)i, beta = spos.v[1] / used_h - (double)j;
    return Vec2d(lerp(field.boundedAt(i + 1, j) - field.boundedAt(i, j),
                      field.boundedAt(i + 1, j + 1) - field.boundedAt(i, j + 1), beta),
                 lerp(field.boundedAt(i, j + 1) - field.boundedAt(i, j),
                      field.boundedAt(i + 1, j + 1) - field.boundedAt(i + 1, j), alpha)) / used_h;
}

Vec2d COFLIPSolver2D::getVelocityBSpline(const Vec2d& pos, const Array2d& un, const Array2d& vn, bool do_curlFree)
{
    // return Vec2d(sin(pos[0])*cos(pos[1]), -cos(pos[0])*sin(pos[1]));
    if (bs_p == 1)
        return getVelocity(pos, un, vn);

    double u_sample, v_sample;
    if (do_curlFree) {
        u_sample = sampleFieldBSplineCurlFree(pos, vn, 0);
        v_sample = sampleFieldBSplineCurlFree(pos, un, 1);
    } else {
        u_sample = sampleFieldBSpline(pos, un, 0);
        v_sample = sampleFieldBSpline(pos, vn, 1);
    }

    return Vec2d(u_sample, v_sample);
}

Eigen::Matrix2d COFLIPSolver2D::getJacobianVelocityBSpline(const Vec2d& pos, const Array2d& un, const Array2d& vn, bool do_curlFree)
{
    if (bs_p == 1)
        return getJacobianVelocity(pos, un, vn);

    Vec2d grad_u_sample, grad_v_sample;
    if (do_curlFree) {
        grad_u_sample = Vec2d(
            sampleGradientFieldBSplineCurlFree(pos, vn, 0, 0),
            sampleGradientFieldBSplineCurlFree(pos, vn, 0, 1));
        grad_v_sample = Vec2d(
            sampleGradientFieldBSplineCurlFree(pos, un, 1, 0),
            sampleGradientFieldBSplineCurlFree(pos, un, 1, 1));
    } else {
        grad_u_sample = Vec2d(
            sampleGradientFieldBSpline(pos, un, 0, 0),
            sampleGradientFieldBSpline(pos, un, 0, 1));
        grad_v_sample = Vec2d(
            sampleGradientFieldBSpline(pos, vn, 1, 0),
            sampleGradientFieldBSpline(pos, vn, 1, 1));
    }

    return (Eigen::Matrix2d() << grad_u_sample[0], grad_u_sample[1], 
                 grad_v_sample[0], grad_v_sample[1]).finished();
}

double COFLIPSolver2D::sampleFieldBSpline0form(const Vec2d& pos, const Array2d &field)
{
    if (bs_p == 1)
        return sampleField(pos, field);

    double xpos = pos.v[0];
    double ypos = pos.v[1];
    int i = std::floor(xpos / h), j = std::floor(ypos / h);
    double alpha = xpos / h - (double)i, beta = ypos / h - (double)j;

    std::array<int, 4> xoffsets, yoffsets, xshifts, yshifts;
    double primary_t, secondary_t;
    int idx, n_idx, other_idx, other_n_idx;
    xoffsets = {0, 1, 2, 3};
    yoffsets = {0, 0, 0, 0};
    xshifts = {0, 0, 0, 0};
    yshifts = {0, 1, 2, 3};
    primary_t = alpha;
    secondary_t = beta;
    idx = i;
    n_idx = nip;
    other_idx = j;
    other_n_idx = njp;

    int primary_bdy_version = 0;
    if (idx == 0 || idx == (n_idx-1)) {
        primary_bdy_version = 2;
        if (idx == (n_idx-1)) {
            primary_t = std::clamp(1.0 - primary_t, 0.0, 1.0);
            std::reverse(xoffsets.begin(), xoffsets.begin()+bs_p+1);
            std::reverse(yoffsets.begin(), yoffsets.begin()+bs_p+1);
        }
    } else if (bs_p != 2 && (idx == 1 || idx == (n_idx-2))) {
        primary_bdy_version = 1;
        if (idx == (n_idx-2)) {
            primary_t = std::clamp(1.0 - primary_t, 0.0, 1.0);
            std::reverse(xoffsets.begin(), xoffsets.begin()+bs_p+1);
            std::reverse(yoffsets.begin(), yoffsets.begin()+bs_p+1);
        }
    }

    int secondary_bdy_version = 0;
    if (other_idx == 0 || other_idx == (other_n_idx-1)) {
        secondary_bdy_version = 2;
        if (other_idx == (other_n_idx-1)) {
            secondary_t = std::clamp(1.0 - secondary_t, 0.0, 1.0);
            std::reverse(xshifts.begin(), xshifts.begin()+bs_p+1);
            std::reverse(yshifts.begin(), yshifts.begin()+bs_p+1);
        }
    } else if (bs_p != 2 && (other_idx == 1 || other_idx == (other_n_idx-2))) {
        secondary_bdy_version = 1;
        if (other_idx == (other_n_idx-2)) {
            secondary_t = std::clamp(1.0 - secondary_t, 0.0, 1.0);
            std::reverse(xshifts.begin(), xshifts.begin()+bs_p+1);
            std::reverse(yshifts.begin(), yshifts.begin()+bs_p+1);
        }
    }

    std::array<double, 4> basesp;
    std::array<double, 4> basess;
    if (bs_p == 2) {
        basesp = {B20(primary_t, primary_bdy_version), B21(primary_t, primary_bdy_version), B22(primary_t), 0.0};
        basess = {B20(secondary_t, secondary_bdy_version), B21(secondary_t, secondary_bdy_version), B22(secondary_t), 0.0};
    } else {
        basesp = {B30(primary_t, primary_bdy_version), B31(primary_t, primary_bdy_version), B32(primary_t, primary_bdy_version), B33(primary_t)};
        basess = {B30(secondary_t, secondary_bdy_version), B31(secondary_t, secondary_bdy_version), B32(secondary_t, secondary_bdy_version), B33(secondary_t)};
    }
    double result = 0.0;
    for (int k = 0; k < bs_p+1; k++) {
        for (int l = 0; l < bs_p+1; l++) {
            result += field(i + xoffsets[l] + xshifts[k], j + yoffsets[l] + yshifts[k]) * basesp[l] * basess[k];
        }
    }
    return result;
}

double COFLIPSolver2D::sampleFieldBSpline(const Vec2d& pos, const Array2d &field, int selected_row)
{
    double xpos = pos.v[0];
    double ypos = pos.v[1];
    int i = std::floor(xpos / h), j = std::floor(ypos / h);
    double alpha = xpos / h - (double)i, beta = ypos / h - (double)j;

    std::array<int, 4> xoffsets, yoffsets, xshifts, yshifts;
    double primary_t, secondary_t;
    int idx, n_idx, other_idx, other_n_idx;
    if (selected_row == 0) {
        xoffsets = {0, 1, 2, 3};
        yoffsets = {0, 0, 0, 0};
        xshifts = {0, 0, 0, 0};
        yshifts = {0, 1, 2, 3};
        primary_t = alpha;
        secondary_t = beta;
        idx = i;
        n_idx = nip;
        other_idx = j;
        other_n_idx = njp;
    } else {
        xoffsets = {0, 0, 0, 0};
        yoffsets = {0, 1, 2, 3};
        xshifts = {0, 1, 2, 3};
        yshifts = {0, 0, 0, 0};
        primary_t = beta;
        secondary_t = alpha;
        idx = j;
        n_idx = njp;
        other_idx = i;
        other_n_idx = nip;
    }

    int primary_bdy_version = 0;
    if (idx == 0 || idx == (n_idx-1)) {
        primary_bdy_version = 2;
        if (idx == (n_idx-1)) {
            primary_t = std::clamp(1.0 - primary_t, 0.0, 1.0);
            std::reverse(xoffsets.begin(), xoffsets.begin()+bs_p+1);
            std::reverse(yoffsets.begin(), yoffsets.begin()+bs_p+1);
        }
    } else if (bs_p != 2 && (idx == 1 || idx == (n_idx-2))) {
        primary_bdy_version = 1;
        if (idx == (n_idx-2)) {
            primary_t = std::clamp(1.0 - primary_t, 0.0, 1.0);
            std::reverse(xoffsets.begin(), xoffsets.begin()+bs_p+1);
            std::reverse(yoffsets.begin(), yoffsets.begin()+bs_p+1);
        }
    }

    int secondary_bdy_version = 0;
    if (other_idx == 0 || other_idx == (other_n_idx-1)) {
        secondary_bdy_version = 2;
        if (other_idx == (other_n_idx-1)) {
            secondary_t = std::clamp(1.0 - secondary_t, 0.0, 1.0);
            std::reverse(xshifts.begin(), xshifts.begin()+bs_p);
            std::reverse(yshifts.begin(), yshifts.begin()+bs_p);
        }
    } else if (bs_p != 2 && (other_idx == 1 || other_idx == (other_n_idx-2))) {
        secondary_bdy_version = 1;
        if (other_idx == (other_n_idx-2)) {
            secondary_t = std::clamp(1.0 - secondary_t, 0.0, 1.0);
            std::reverse(xshifts.begin(), xshifts.begin()+bs_p);
            std::reverse(yshifts.begin(), yshifts.begin()+bs_p);
        }
    }

    std::array<double, 4> basesp;
    std::array<double, 3> basess;
    if (bs_p == 2) {
        basesp = {B20(primary_t, primary_bdy_version), B21(primary_t, primary_bdy_version), B22(primary_t), 0.0};
        basess = {D10(secondary_t, secondary_bdy_version), D11(secondary_t), 0.0};
    } else {
        basesp = {B30(primary_t, primary_bdy_version), B31(primary_t, primary_bdy_version), B32(primary_t, primary_bdy_version), B33(primary_t)};
        basess = {D20(secondary_t, secondary_bdy_version), D21(secondary_t, secondary_bdy_version), D22(secondary_t)};
    }
    double result = 0.0;
    for (int k = 0; k < bs_p; k++) {
        for (int l = 0; l < bs_p+1; l++) {
            result += field(i + xoffsets[l] + xshifts[k], j + yoffsets[l] + yshifts[k]) * basesp[l] * basess[k];
        }
    }
    return result;
}

double COFLIPSolver2D::sampleGradientFieldBSpline(const Vec2d& pos, const Array2d &field, int selected_row, int selected_column)
{
    double xpos = pos.v[0];
    double ypos = pos.v[1];
    int i = std::floor(xpos / h), j = std::floor(ypos / h);
    double alpha = xpos / h - (double)i, beta = ypos / h - (double)j;

    std::array<int, 4> xoffsets, yoffsets, xshifts, yshifts;
    double primary_t, secondary_t;
    int idx, n_idx, other_idx, other_n_idx;
    if (selected_row == 0) {
        xoffsets = {0, 1, 2, 3};
        yoffsets = {0, 0, 0, 0};
        xshifts = {0, 0, 0, 0};
        yshifts = {0, 1, 2, 3};
        primary_t = alpha;
        secondary_t = beta;
        idx = i;
        n_idx = nip;
        other_idx = j;
        other_n_idx = njp;
    } else {
        xoffsets = {0, 0, 0, 0};
        yoffsets = {0, 1, 2, 3};
        xshifts = {0, 1, 2, 3};
        yshifts = {0, 0, 0, 0};
        primary_t = beta;
        secondary_t = alpha;
        idx = j;
        n_idx = njp;
        other_idx = i;
        other_n_idx = nip;
    }

    int case_index = (selected_column-selected_row+2)%2;

    int primary_bdy_version = 0;
    double mult = 1.0;
    if (idx == 0 || idx == (n_idx-1)) {
        primary_bdy_version = 2;
        if (idx == (n_idx-1)) {
            primary_t = std::clamp(1.0 - primary_t, 0.0, 1.0);
            std::reverse(xoffsets.begin(), xoffsets.begin()+bs_p+1);
            std::reverse(yoffsets.begin(), yoffsets.begin()+bs_p+1);
            mult = case_index == 0 ? -mult : mult;
        }
    } else if (bs_p != 2 && (idx == 1 || idx == (n_idx-2))) {
        primary_bdy_version = 1;
        if (idx == (n_idx-2)) {
            primary_t = std::clamp(1.0 - primary_t, 0.0, 1.0);
            std::reverse(xoffsets.begin(), xoffsets.begin()+bs_p+1);
            std::reverse(yoffsets.begin(), yoffsets.begin()+bs_p+1);
            mult = case_index == 0 ? -mult : mult;
        }
    }

    int secondary_bdy_version = 0;
    if (other_idx == 0 || other_idx == (other_n_idx-1)) {
        secondary_bdy_version = 2;
        if (other_idx == (other_n_idx-1)) {
            secondary_t = std::clamp(1.0 - secondary_t, 0.0, 1.0);
            std::reverse(xshifts.begin(), xshifts.begin()+bs_p);
            std::reverse(yshifts.begin(), yshifts.begin()+bs_p);
            mult = case_index == 1 ? -mult : mult;
        }
    } else if (bs_p != 2 && (other_idx == 1 || other_idx == (other_n_idx-2))) {
        secondary_bdy_version = 1;
        if (other_idx == (other_n_idx-2)) {
            secondary_t = std::clamp(1.0 - secondary_t, 0.0, 1.0);
            std::reverse(xshifts.begin(), xshifts.begin()+bs_p);
            std::reverse(yshifts.begin(), yshifts.begin()+bs_p);
            mult = case_index == 1 ? -mult : mult;
        }
    }

    std::array<double, 4> basesp;
    std::array<double, 3> basess;
    if (bs_p == 2) {
        std::array<double, 4> bases2p;
        std::array<double, 3> bases1s;
        if (case_index == 0) {
            bases2p = {B20x(primary_t, primary_bdy_version), B21x(primary_t, primary_bdy_version), B22x(primary_t), 0.0};
            bases1s = {D10(secondary_t, secondary_bdy_version), D11(secondary_t), 0.0};
        } else {
            bases2p = {B20(primary_t, primary_bdy_version), B21(primary_t, primary_bdy_version), B22(primary_t), 0.0};
            bases1s = {D10x(secondary_t, secondary_bdy_version), D11x(secondary_t), 0.0};
        } 
        basesp = bases2p;
        basess = bases1s;
    } else {
        std::array<double, 4> bases3p;
        std::array<double, 3> bases2s;
        if (case_index == 0) {
            bases3p = {B30x(primary_t, primary_bdy_version), B31x(primary_t, primary_bdy_version), B32x(primary_t, primary_bdy_version), B33x(primary_t)};
            bases2s = {D20(secondary_t, secondary_bdy_version), D21(secondary_t, secondary_bdy_version), D22(secondary_t)};
        } else {
            bases3p = {B30(primary_t, primary_bdy_version), B31(primary_t, primary_bdy_version), B32(primary_t, primary_bdy_version), B33(primary_t)};
            bases2s = {D20x(secondary_t, secondary_bdy_version), D21x(secondary_t, secondary_bdy_version), D22x(secondary_t)};
        } 
        basesp = bases3p;
        basess = bases2s;
    }
    double result = 0.0;
    for (int k = 0; k < bs_p; k++) {
        for (int l = 0; l < bs_p+1; l++) {
            result += field(i + xoffsets[l] + xshifts[k], j + yoffsets[l] + yshifts[k]) * basesp[l] * basess[k];
        }
    }
    return result * mult / h;
}

double COFLIPSolver2D::sampleFieldBSplineCurlFree(const Vec2d& pos, const Array2d &field, int selected_row)
{
    double xpos = pos.v[0];
    double ypos = pos.v[1];
    int i = std::floor(xpos / h), j = std::floor(ypos / h);
    double alpha = xpos / h - (double)i, beta = ypos / h - (double)j;

    std::array<int, 4> xoffsets, yoffsets, xshifts, yshifts;
    double primary_t, secondary_t;
    int idx, n_idx, other_idx, other_n_idx;
    if (selected_row == 0) {
        xoffsets = {0, 1, 2, 3};
        yoffsets = {0, 0, 0, 0};
        xshifts = {0, 0, 0, 0};
        yshifts = {0, 1, 2, 3};
        primary_t = alpha;
        secondary_t = beta;
        idx = i;
        n_idx = nip;
        other_idx = j;
        other_n_idx = njp;
    } else {
        xoffsets = {0, 0, 0, 0};
        yoffsets = {0, 1, 2, 3};
        xshifts = {0, 1, 2, 3};
        yshifts = {0, 0, 0, 0};
        primary_t = beta;
        secondary_t = alpha;
        idx = j;
        n_idx = njp;
        other_idx = i;
        other_n_idx = nip;
    }

    int primary_bdy_version = 0;
    if (idx == 0 || idx == (n_idx-1)) {
        primary_bdy_version = 2;
        if (idx == (n_idx-1)) {
            primary_t = std::clamp(1.0 - primary_t, 0.0, 1.0);
            std::reverse(xoffsets.begin(), xoffsets.begin()+bs_p);
            std::reverse(yoffsets.begin(), yoffsets.begin()+bs_p);
        }
    } else if (bs_p != 2 && (idx == 1 || idx == (n_idx-2))) {
        primary_bdy_version = 1;
        if (idx == (n_idx-2)) {
            primary_t = std::clamp(1.0 - primary_t, 0.0, 1.0);
            std::reverse(xoffsets.begin(), xoffsets.begin()+bs_p);
            std::reverse(yoffsets.begin(), yoffsets.begin()+bs_p);
        }
    }

    int secondary_bdy_version = 0;
    if (other_idx == 0 || other_idx == (other_n_idx-1)) {
        secondary_bdy_version = 2;
        if (other_idx == (other_n_idx-1)) {
            secondary_t = std::clamp(1.0 - secondary_t, 0.0, 1.0);
            std::reverse(xshifts.begin(), xshifts.begin()+bs_p+1);
            std::reverse(yshifts.begin(), yshifts.begin()+bs_p+1);
        }
    } else if (bs_p != 2 && (other_idx == 1 || other_idx == (other_n_idx-2))) {
        secondary_bdy_version = 1;
        if (other_idx == (other_n_idx-2)) {
            secondary_t = std::clamp(1.0 - secondary_t, 0.0, 1.0);
            std::reverse(xshifts.begin(), xshifts.begin()+bs_p+1);
            std::reverse(yshifts.begin(), yshifts.begin()+bs_p+1);
        }
    }

    std::array<double, 3> basesp;
    std::array<double, 4> basess;
    if (bs_p == 2) {
        basesp = {D10(primary_t, primary_bdy_version), D11(primary_t), 0.0};
        basess = {B20(secondary_t, secondary_bdy_version), B21(secondary_t, secondary_bdy_version), B22(secondary_t), 0.0};
    } else {
        basesp = {D20(primary_t, primary_bdy_version), D21(primary_t, primary_bdy_version), D22(primary_t)};
        basess = {B30(secondary_t, secondary_bdy_version), B31(secondary_t, secondary_bdy_version), B32(secondary_t, secondary_bdy_version), B33(secondary_t)};
    }
    double result = 0.0;
    for (int k = 0; k < bs_p+1; k++) {
        for (int l = 0; l < bs_p; l++) {
            result += field(i + xoffsets[l] + xshifts[k], j + yoffsets[l] + yshifts[k]) * basesp[l] * basess[k];
        }
    }
    return result;
}

double COFLIPSolver2D::sampleGradientFieldBSplineCurlFree(const Vec2d& pos, const Array2d &field, int selected_row, int selected_column)
{
    double xpos = pos.v[0];
    double ypos = pos.v[1];
    int i = std::floor(xpos / h), j = std::floor(ypos / h);
    double alpha = xpos / h - (double)i, beta = ypos / h - (double)j;

    std::array<int, 4> xoffsets, yoffsets, xshifts, yshifts;
    double primary_t, secondary_t;
    int idx, n_idx, other_idx, other_n_idx;
    if (selected_row == 0) {
        xoffsets = {0, 1, 2, 3};
        yoffsets = {0, 0, 0, 0};
        xshifts = {0, 0, 0, 0};
        yshifts = {0, 1, 2, 3};
        primary_t = alpha;
        secondary_t = beta;
        idx = i;
        n_idx = nip;
        other_idx = j;
        other_n_idx = njp;
    } else {
        xoffsets = {0, 0, 0, 0};
        yoffsets = {0, 1, 2, 3};
        xshifts = {0, 1, 2, 3};
        yshifts = {0, 0, 0, 0};
        primary_t = beta;
        secondary_t = alpha;
        idx = j;
        n_idx = njp;
        other_idx = i;
        other_n_idx = nip;
    }

    int case_index = (selected_column-selected_row+2)%2;

    int primary_bdy_version = 0;
    double mult = 1.0;
    if (idx == 0 || idx == (n_idx-1)) {
        primary_bdy_version = 2;
        if (idx == (n_idx-1)) {
            primary_t = std::clamp(1.0 - primary_t, 0.0, 1.0);
            std::reverse(xoffsets.begin(), xoffsets.begin()+bs_p);
            std::reverse(yoffsets.begin(), yoffsets.begin()+bs_p);
            mult = case_index == 1 ? -mult : mult;
        }
    } else if (bs_p != 2 && (idx == 1 || idx == (n_idx-2))) {
        primary_bdy_version = 1;
        if (idx == (n_idx-2)) {
            primary_t = std::clamp(1.0 - primary_t, 0.0, 1.0);
            std::reverse(xoffsets.begin(), xoffsets.begin()+bs_p);
            std::reverse(yoffsets.begin(), yoffsets.begin()+bs_p);
            mult = case_index == 1 ? -mult : mult;
        }
    }

    int secondary_bdy_version = 0;
    if (other_idx == 0 || other_idx == (other_n_idx-1)) {
        secondary_bdy_version = 2;
        if (other_idx == (other_n_idx-1)) {
            secondary_t = std::clamp(1.0 - secondary_t, 0.0, 1.0);
            std::reverse(xshifts.begin(), xshifts.begin()+bs_p+1);
            std::reverse(yshifts.begin(), yshifts.begin()+bs_p+1);
            mult = case_index == 0 ? -mult : mult;
        }
    } else if (bs_p != 2 && (other_idx == 1 || other_idx == (other_n_idx-2))) {
        secondary_bdy_version = 1;
        if (other_idx == (other_n_idx-2)) {
            secondary_t = std::clamp(1.0 - secondary_t, 0.0, 1.0);
            std::reverse(xshifts.begin(), xshifts.begin()+bs_p+1);
            std::reverse(yshifts.begin(), yshifts.begin()+bs_p+1);
            mult = case_index == 0 ? -mult : mult;
        }
    }

    std::array<double, 3> basesp;
    std::array<double, 4> basess;
    if (bs_p == 2) {
        std::array<double, 3> bases2p;
        std::array<double, 4> bases1s;
        if (case_index == 1) {
            bases2p = {D10(primary_t, primary_bdy_version), D11(primary_t), 0.0};
            bases1s = {B20x(secondary_t, secondary_bdy_version), B21x(secondary_t, secondary_bdy_version), B22x(secondary_t), 0.0};
        } else {
            bases2p = {D10x(primary_t, primary_bdy_version), D11x(primary_t), 0.0};
            bases1s = {B20(secondary_t, secondary_bdy_version), B21(secondary_t, secondary_bdy_version), B22(secondary_t), 0.0};
        } 
        basesp = bases2p;
        basess = bases1s;
    } else {
        std::array<double, 3> bases3p;
        std::array<double, 4> bases2s;
        if (case_index == 1) {
            bases3p = {D20(primary_t, primary_bdy_version), D21(primary_t, primary_bdy_version), D22(primary_t)};
            bases2s = {B30x(secondary_t, secondary_bdy_version), B31x(secondary_t, secondary_bdy_version), B32x(secondary_t, secondary_bdy_version), B33x(secondary_t)};
        } else {
            bases3p = {D20x(primary_t, primary_bdy_version), D21x(primary_t, primary_bdy_version), D22x(primary_t)};
            bases2s = {B30(secondary_t, secondary_bdy_version), B31(secondary_t, secondary_bdy_version), B32(secondary_t, secondary_bdy_version), B33(secondary_t)};
        } 
        basesp = bases3p;
        basess = bases2s;
    }
    double result = 0.0;
    for (int k = 0; k < bs_p+1; k++) {
        for (int l = 0; l < bs_p; l++) {
            result += field(i + xoffsets[l] + xshifts[k], j + yoffsets[l] + yshifts[k]) * basesp[l] * basess[k];
        }
    }
    return result * mult / h;
}

double COFLIPSolver2D::sampleFieldBSpline2form(const Vec2d& pos, const Array2d &field)
{
    if (bs_p == 1)
        return sampleField(pos, field);

    double xpos = pos.v[0];
    double ypos = pos.v[1];
    int i = std::floor(xpos / h), j = std::floor(ypos / h);
    double alpha = xpos / h - (double)i, beta = ypos / h - (double)j;

    std::array<int, 3> xoffsets, yoffsets, xshifts, yshifts;
    double primary_t, secondary_t;
    int idx, n_idx, other_idx, other_n_idx;
    xoffsets = {0, 1, 2};
    yoffsets = {0, 0, 0};
    xshifts = {0, 0, 0};
    yshifts = {0, 1, 2};
    primary_t = alpha;
    secondary_t = beta;
    idx = i;
    n_idx = nip;
    other_idx = j;
    other_n_idx = njp;

    int primary_bdy_version = 0;
    if (idx == 0 || idx == (n_idx-1)) {
        primary_bdy_version = 2;
        if (idx == (n_idx-1)) {
            primary_t = std::clamp(1.0 - primary_t, 0.0, 1.0);
            std::reverse(xoffsets.begin(), xoffsets.begin()+bs_p);
            std::reverse(yoffsets.begin(), yoffsets.begin()+bs_p);
        }
    } else if (bs_p != 2 && (idx == 1 || idx == (n_idx-2))) {
        primary_bdy_version = 1;
        if (idx == (n_idx-2)) {
            primary_t = std::clamp(1.0 - primary_t, 0.0, 1.0);
            std::reverse(xoffsets.begin(), xoffsets.begin()+bs_p);
            std::reverse(yoffsets.begin(), yoffsets.begin()+bs_p);
        }
    }

    int secondary_bdy_version = 0;
    if (other_idx == 0 || other_idx == (other_n_idx-1)) {
        secondary_bdy_version = 2;
        if (other_idx == (other_n_idx-1)) {
            secondary_t = std::clamp(1.0 - secondary_t, 0.0, 1.0);
            std::reverse(xshifts.begin(), xshifts.begin()+bs_p);
            std::reverse(yshifts.begin(), yshifts.begin()+bs_p);
        }
    } else if (bs_p != 2 && (other_idx == 1 || other_idx == (other_n_idx-2))) {
        secondary_bdy_version = 1;
        if (other_idx == (other_n_idx-2)) {
            secondary_t = std::clamp(1.0 - secondary_t, 0.0, 1.0);
            std::reverse(xshifts.begin(), xshifts.begin()+bs_p);
            std::reverse(yshifts.begin(), yshifts.begin()+bs_p);
        }
    }

    std::array<double, 3> basesp;
    std::array<double, 3> basess;
    if (bs_p == 2) {
        basesp = {D10(primary_t, primary_bdy_version), D11(primary_t), 0.0};
        basess = {D10(secondary_t, secondary_bdy_version), D11(secondary_t), 0.0};
    } else {
        basesp = {D20(primary_t, primary_bdy_version), D21(primary_t, primary_bdy_version), D22(primary_t)};
        basess = {D20(secondary_t, secondary_bdy_version), D21(secondary_t, secondary_bdy_version), D22(secondary_t)};
    }
    double result = 0.0;
    for (int k = 0; k < bs_p; k++) {
        for (int l = 0; l < bs_p; l++) {
            result += field(i + xoffsets[l] + xshifts[k], j + yoffsets[l] + yshifts[k]) * basesp[l] * basess[k];
        }
    }
    return result;
}

void COFLIPSolver2D::integratePressureForce(Array2d& un, Array2d& vn) {
    Eigen::VectorXd pressure_force_fluxes(nF);
    tbb::parallel_for(tbb::blocked_range<int>(0,nF,TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
    {
        for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
            bool is_x_dir = tIdx < (ni+1)*nj;
            int j = is_x_dir ? tIdx / (ni+1) : (tIdx - (ni+1)*nj) / ni;
            int i = is_x_dir ? tIdx % (ni+1) : (tIdx - (ni+1)*nj) % ni;
            if (is_x_dir) {
                pressure_force_fluxes(tIdx) = un(i,j);
            } else {
                pressure_force_fluxes(tIdx) = vn(i,j);
            }
        }
    });
    Eigen::VectorXd pressure_force_circulations = starflux_matrix * pressure_force_fluxes;

    Eigen::VectorXd pressure_field_2form(nC);
    pressure_field_2form.setZero();
    pressure_field_2form[0] = TOLERANCE;

    for (int i = 1; i < ni; i++) {
        pressure_field_2form[i] = pressure_field_2form[i-1] + pressure_force_circulations[i];
    }

    tbb::parallel_for(tbb::blocked_range<int>(0,ni,TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
    {
        for (int i = r.begin(); i < r.end(); ++i) {
            for (int j = 1; j < nj; j++) {
                pressure_field_2form[j*ni+i] = pressure_field_2form[(j-1)*ni+i] + pressure_force_circulations[((ni+1)*nj)+j*ni+i];
            }
        }
    });

    starpressure_cg.setTolerance(TOLERANCE);
    starpressure_cg.setMaxIterations(MAX_ITERATIONS);
    Eigen::VectorXd input = pressure_field_2form;
    pressure_field_2form = starpressure_cg.solve(input);
    if (starpressure_cg.info() == Eigen::Success) {
        std::cout << "#iteration:      " << starpressure_cg.iterations() << std::endl;
        std::cout << "estimated error: " << starpressure_cg.error() << std::endl;
    } else {
        std::cout << "starpressure_cg solver FAILED!!!" << std::endl;
    }
    Array2d pressure_field;
    pressure_field.resize(ni,nj);
    tbb::parallel_for(tbb::blocked_range2d<int,int>{0,ni,1,0,nj,1}, [&](const tbb::blocked_range2d<int,int> &r)
    {
        int ie = r.rows().end();
        int je = r.cols().end();
        for (int j = r.cols().begin(); j < je; ++j) {
            for (int i = r.rows().begin(); i < ie; ++i) {
                int tIdx = j*ni + i;
                pressure_field(i,j) = pressure_field_2form(tIdx);
            }
        }
    });

    tbb::parallel_for(tbb::blocked_range<int>(0,cParticles.size(),TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
    {
        for (int i = r.begin(); i < r.end(); ++i) {
            cParticles[i].vel_temp[0] = sampleFieldBSpline2form(cParticles[i].pos_current, pressure_field);
        }
    });
    pressure_field_0form.resize(nV);
    solveInterpDaggerVorticity(true);

    double min_pressure = 1e20;
    double max_pressure = -1e20;
    tbb::parallel_for(tbb::blocked_range<int>(0,nF,TBB_GRAINSIZE), [&](const tbb::blocked_range<int> &r)
    {
        for (int tIdx = r.begin(); tIdx < r.end(); ++tIdx) {
            bool is_x_dir = tIdx < (ni+1)*nj;
            int j = is_x_dir ? tIdx / (ni+1) : (tIdx - (ni+1)*nj) / ni;
            int i = is_x_dir ? tIdx % (ni+1) : (tIdx - (ni+1)*nj) % ni;
            if (is_x_dir) {
                un(i,j) = clamp(pressure_field_0form((j+1)*(ni+1)+i), min_pressure, max_pressure) - clamp(pressure_field_0form(j*(ni+1)+i), min_pressure, max_pressure);
            } else {
                vn(i,j) = clamp(pressure_field_0form(j*(ni+1)+(i+1)), min_pressure, max_pressure) - clamp(pressure_field_0form(j*(ni+1)+i), min_pressure, max_pressure);
            }
        }
    });
}

void COFLIPSolver2D::initSmokePlume()
{
    emitterMask.assign(0);
    double radius = 0.05;
    Vec2d center(0.25, 0.25);
    temperature.assign(1);
    rho.assign(0);
    tbb::parallel_for(tbb::blocked_range2d<int,int>{0,ni,1,0,nj,1}, [&](const tbb::blocked_range2d<int,int> &r)
    {
        int ie = r.rows().end();
        int je = r.cols().end();
        for (int j = r.cols().begin(); j < je; ++j) {
            for (int i = r.rows().begin(); i < ie; ++i) {
                Vec2d spos = (Vec2d(i,j) + Vec2d(0.5)) * h_uniform;
                if (dist(spos,center) < radius) {
                    emitterMask(i,j) = 1;
                    for (int ii = (rho.ni/ni)*i; ii < (rho.ni/ni)*i+(rho.ni/ni); ii++) {
                        for (int jj = (rho.nj/nj)*j; jj < (rho.nj/nj)*j+(rho.nj/nj); jj++) {
                            temperature(ii,jj) -= 0.01;
                            rho(ii,jj) += 0.01;
                        }
                    }
                }
            }
        }
    });

    return;
}

void COFLIPSolver2D::outputDensity(std::string folder, std::string file, int framenum, bool color_density, bool do_tonemapping, bool scaleDensity)
{
    boost::filesystem::create_directories(folder);
    std::string filestr;
    filestr = folder + file + std::string("_\%04d.bmp");
    char filename[1024];
    sprintf(filename, filestr.c_str(), framenum);
    if (color_density)
    {
        int size_amp = 4;
        std::vector<Vec3uc> color;
        color.resize(nC*size_amp*size_amp);
        cBar = color_bar(0,1);
        tbb::parallel_for(tbb::blocked_range2d<int,int>{0,ni*size_amp,1,0,nj*size_amp,1}, [&](const tbb::blocked_range2d<int,int> &r)
        {
            int ie = r.rows().end();
            int je = r.cols().end();
            for (int j = r.cols().begin(); j < je; ++j) {
                for (int i = r.rows().begin(); i < ie; ++i) {
                    if (boundaryMask(i/size_amp,j/size_amp) == 3)
                    {
                        color[j*ni*size_amp + i] = Vec3uc(0);
                    }
                    else
                    {
                        Vec2d spos = (Vec2d(i,j) + Vec2d(0.5)) * h_uniform/(double)size_amp;
                        Vec2d zeroFormPos = spos - h_uniform/((double)(rho.ni/ni))*Vec2d(0.5);
                        double value = std::clamp(sampleField(zeroFormPos, rho, true)/(scaleDensity ? 3.5 : 1.),0.,1.) - std::clamp(sampleField(zeroFormPos, temperature, true),0.,1.);
                        value = (value + 1.)*0.5;
                        if (do_tonemapping)
                        {
                            value = std::max(value, 0.);
                            value = std::sqrt(value)*(1.+std::sqrt(value)/std::pow(2.2,2.))/(1.+std::sqrt(value));
                        }
                        color[j*ni*size_amp + i] = cBar.toRGB(1.-value);
                    }
                }
            }
        });
        wrtieBMPuc3(filename, ni*size_amp, nj*size_amp, (unsigned char*)(&(color[0])));
    }
    else
        writeBMP(filename, ni, nj, rho.a.data);
}

void COFLIPSolver2D::outputVortVisualized(std::string folder, std::string file, int i)
{
    boost::filesystem::create_directories(folder);
    std::string filestr;
    filestr = folder + file + std::string("\%04d.bmp");
    char filename[1024];
    sprintf(filename, filestr.c_str(), i);
    int size_amp = 2;
    std::vector<Vec3uc> color;
    color.resize(size_amp*size_amp*nC);
    cBar = color_bar(0,0);
    calculateCurl(true, true);
    tbb::parallel_for(tbb::blocked_range2d<int,int>{0,size_amp*ni,1,0,size_amp*nj,1}, [&](const tbb::blocked_range2d<int,int> &r)
    {
        int ie = r.rows().end();
        int je = r.cols().end();
        for (int j = r.cols().begin(); j < je; ++j) {
            for (int i = r.rows().begin(); i < ie; ++i) {
                Vec2d pos = (Vec2d(i,j) + Vec2d(0.5)) * h_uniform/(double)size_amp;
                double vort = sampleFieldBSpline0form(pos, curl);
                color[j*size_amp*ni + i] = cBar.toRGB(std::abs(vort), 10);
            }
        }
    });
    wrtieBMPuc3(filename, size_amp*ni, size_amp*nj, (unsigned char*)(&(color[0])));
}

void COFLIPSolver2D::outputEnergy(std::string filename, double curr_time)
{
    boost::filesystem::create_directories(filename);
    std::ofstream foutU;
    std::string filenameU = filename + std::string("energy") + std::string(".txt");
    foutU.open(filenameU, curr_time == 0. ? std::ios_base::out : std::ios_base::app);
    if (sim_scheme != Scheme::CO_FLIP) {
        int res = 4;
        std::uniform_real_distribution<double> unif(0, 1);
        double energy = tbb::parallel_reduce( 
            tbb::blocked_range<int>(0,ni*nj),
            0.0,
            [&](tbb::blocked_range<int> range, double running_total)
            {
                for (int tIdx=range.begin(); tIdx<range.end(); ++tIdx)
                {
                    std::mt19937_64 rng;
                    rng.seed(tIdx);
                    int j = tIdx / ni;
                    int i = tIdx % ni;
                    double x = (double)i*h;
                    double y = (double)j*h;
                    for(int jj=0;jj<res;jj++)
                    {
                        for(int ii=0;ii<res;ii++)
                        {
                            Vec<2, double> spos(x+((double)ii + unif(rng))/(double)res*h,
                                                y+((double)jj + unif(rng))/(double)res*h);
                            Vec<2, double> vel = getVelocityBSpline(spos, u, v);
                            running_total += dot(vel, vel);
                        }
                    }
                }

                return running_total;
            }, std::plus<double>() );
        energy *= std::pow(h/(double)(res), 2);
        std::cout << RED << "Energy = " << energy << RESET <<  std::endl;
        foutU << energy << " " << curr_time << std::endl;
    } else {
        takeDualwrtStar(u, v, false, false);
        double energy = fluxes.transpose() * circulations;
        energy *= std::pow(h, 2);
        std::cout << GREEN << "energy.diff: " << energy-energy_prev << RESET << std::endl;
        energy_prev = energy;
        std::cout << RED << "Energy = " << energy << RESET <<  std::endl;
        foutU << energy << " " << curr_time << std::endl;
    }
    foutU.close();
}

void COFLIPSolver2D::outputVorticityIntegral(std::string filename, double curr_time, bool do_highres)
{
    boost::filesystem::create_directories(filename);
    std::ofstream foutU, foutU2, foutU3, foutU4, foutU5, foutU6, foutU7, foutU8, foutU9, foutU10;
    std::string filenameU = filename + std::string("vort1") + std::string(".txt");
    std::string filenameU2 = filename + std::string("vort2") + std::string(".txt");
    std::string filenameU3 = filename + std::string("vort3") + std::string(".txt");
    std::string filenameU4 = filename + std::string("vort4") + std::string(".txt");
    foutU.open(filenameU, curr_time == 0. ? std::ios_base::out : std::ios_base::app);
    foutU2.open(filenameU2, curr_time == 0. ? std::ios_base::out : std::ios_base::app);
    foutU3.open(filenameU3, curr_time == 0. ? std::ios_base::out : std::ios_base::app);
    foutU4.open(filenameU4, curr_time == 0. ? std::ios_base::out : std::ios_base::app);
    if (do_highres) {
        std::string filenameU5 = filename + std::string("vort1_res2") + std::string(".txt");
        std::string filenameU6 = filename + std::string("vort2_res2") + std::string(".txt");
        foutU5.open(filenameU5, curr_time == 0. ? std::ios_base::out : std::ios_base::app);
        foutU6.open(filenameU6, curr_time == 0. ? std::ios_base::out : std::ios_base::app);
        std::string filenameU7 = filename + std::string("vort1_res4") + std::string(".txt");
        std::string filenameU8 = filename + std::string("vort2_res4") + std::string(".txt");
        foutU7.open(filenameU7, curr_time == 0. ? std::ios_base::out : std::ios_base::app);
        foutU8.open(filenameU8, curr_time == 0. ? std::ios_base::out : std::ios_base::app);
        std::string filenameU9 = filename + std::string("vort1_res8") + std::string(".txt");
        std::string filenameU10 = filename + std::string("vort2_res8") + std::string(".txt");
        foutU9.open(filenameU9, curr_time == 0. ? std::ios_base::out : std::ios_base::app);
        foutU10.open(filenameU10, curr_time == 0. ? std::ios_base::out : std::ios_base::app);
    }

    calculateCurl(true, true);
    if (sim_scheme != Scheme::CO_FLIP) {
        int res = 6;
        std::uniform_real_distribution<double> unif(0, 1);
        double firstMomentVorticityIntegral = tbb::parallel_reduce( 
            tbb::blocked_range<int>(0,ni*nj),
            0.0,
            [&](tbb::blocked_range<int> range, double running_total)
            {
                for (int tIdx=range.begin(); tIdx<range.end(); ++tIdx)
                {
                    std::mt19937_64 rng;
                    rng.seed(tIdx);
                    int j = tIdx / ni;
                    int i = tIdx % ni;
                    double x = (double)i*h;
                    double y = (double)j*h;
                    for(int jj=0;jj<res;jj++)
                    {
                        for(int ii=0;ii<res;ii++)
                        {
                            Vec<2, double> spos(x+((double)ii + unif(rng))/(double)res*h,
                                                y+((double)jj + unif(rng))/(double)res*h);
                            double vort = sampleField(spos, curl);
                            running_total += vort;
                        }
                    }
                }

                return running_total;
            }, std::plus<double>() );
    
        double secondMomentVorticityIntegral = tbb::parallel_reduce( 
            tbb::blocked_range<int>(0,ni*nj),
            0.0,
            [&](tbb::blocked_range<int> range, double running_total)
            {
                for (int tIdx=range.begin(); tIdx<range.end(); ++tIdx)
                {
                    std::mt19937_64 rng;
                    rng.seed(tIdx);
                    int j = tIdx / ni;
                    int i = tIdx % ni;
                    double x = (double)i*h;
                    double y = (double)j*h;
                    for(int jj=0;jj<res;jj++)
                    {
                        for(int ii=0;ii<res;ii++)
                        {
                            Vec<2, double> spos(x+((double)ii + unif(rng))/(double)res*h,
                                                y+((double)jj + unif(rng))/(double)res*h);
                            double vort = sampleField(spos, curl);
                            running_total += vort*vort;
                        }
                    }
                }

                return running_total;
            }, std::plus<double>() );

        double thirdMomentVorticityIntegral = tbb::parallel_reduce( 
            tbb::blocked_range<int>(0,ni*nj),
            0.0,
            [&](tbb::blocked_range<int> range, double running_total)
            {
                for (int tIdx=range.begin(); tIdx<range.end(); ++tIdx)
                {
                    std::mt19937_64 rng;
                    rng.seed(tIdx);
                    int j = tIdx / ni;
                    int i = tIdx % ni;
                    double x = (double)i*h;
                    double y = (double)j*h;
                    for(int jj=0;jj<res;jj++)
                    {
                        for(int ii=0;ii<res;ii++)
                        {
                            Vec<2, double> spos(x+((double)ii + unif(rng))/(double)res*h,
                                                y+((double)jj + unif(rng))/(double)res*h);
                            double vort = sampleField(spos, curl);
                            running_total += vort*vort*vort;
                        }
                    }
                }

                return running_total;
            }, std::plus<double>() );

        double fourthMomentVorticityIntegral = tbb::parallel_reduce( 
            tbb::blocked_range<int>(0,ni*nj),
            0.0,
            [&](tbb::blocked_range<int> range, double running_total)
            {
                for (int tIdx=range.begin(); tIdx<range.end(); ++tIdx)
                {
                    std::mt19937_64 rng;
                    rng.seed(tIdx);
                    int j = tIdx / ni;
                    int i = tIdx % ni;
                    double x = (double)i*h;
                    double y = (double)j*h;
                    for(int jj=0;jj<res;jj++)
                    {
                        for(int ii=0;ii<res;ii++)
                        {
                            Vec<2, double> spos(x+((double)ii + unif(rng))/(double)res*h,
                                                y+((double)jj + unif(rng))/(double)res*h);
                            double vort = sampleField(spos, curl);
                            running_total += vort*vort*vort*vort;
                        }
                    }
                }

                return running_total;
            }, std::plus<double>() );

        firstMomentVorticityIntegral *= std::pow(h/(double)(res), 2);
        secondMomentVorticityIntegral *= std::pow(h/(double)(res), 2);
        thirdMomentVorticityIntegral *= std::pow(h/(double)(res), 2);
        fourthMomentVorticityIntegral *= std::pow(h/(double)(res), 2);
        std::cout << RED << "Vorticity Moment 1 = " << firstMomentVorticityIntegral << RESET << std::endl;
        std::cout << RED << "Vorticity Moment 2 = " << secondMomentVorticityIntegral << RESET << std::endl;
        std::cout << RED << "Vorticity Moment 3 = " << thirdMomentVorticityIntegral << RESET << std::endl;
        std::cout << RED << "Vorticity Moment 4 = " << fourthMomentVorticityIntegral << RESET << std::endl;
        foutU << firstMomentVorticityIntegral << " " << curr_time << std::endl;
        foutU2 << secondMomentVorticityIntegral << " " << curr_time << std::endl;
        foutU3 << thirdMomentVorticityIntegral << " " << curr_time << std::endl;
        foutU4 << fourthMomentVorticityIntegral << " " << curr_time << std::endl;
    } else {
        Eigen::VectorXd onesNform = starvort_matrix * Eigen::VectorXd::Constant(nV, 1.0);
        double firstMomentVorticityIntegral = prev_vorts0form.transpose() * onesNform;
        double secondMomentVorticityIntegral = prev_vorts0form.transpose() * vorts;
        Eigen::VectorXd vorts_mom3 = prev_vorts0form.array().cube();
        Eigen::VectorXd vorts_mom4 = prev_vorts0form.array().square() * prev_vorts0form.array().square();
        double thirdMomentVorticityIntegral = vorts_mom3.transpose() * onesNform;
        double fourthMomentVorticityIntegral = vorts_mom4.transpose() * onesNform;
        firstMomentVorticityIntegral *= std::pow(h, 2);
        secondMomentVorticityIntegral *= std::pow(h, 2);
        thirdMomentVorticityIntegral *= std::pow(h, 2);
        fourthMomentVorticityIntegral *= std::pow(h, 2);
        std::cout << RED << "Vorticity Moment 1 = " << firstMomentVorticityIntegral << RESET << std::endl;
        std::cout << RED << "Vorticity Moment 2 = " << secondMomentVorticityIntegral << RESET << std::endl;
        std::cout << RED << "Vorticity Moment 3 = " << thirdMomentVorticityIntegral << RESET << std::endl;
        std::cout << RED << "Vorticity Moment 4 = " << fourthMomentVorticityIntegral << RESET << std::endl;
        foutU << firstMomentVorticityIntegral << " " << curr_time << std::endl;
        foutU2 << secondMomentVorticityIntegral << " " << curr_time << std::endl;
        foutU3 << thirdMomentVorticityIntegral << " " << curr_time << std::endl;
        foutU4 << fourthMomentVorticityIntegral << " " << curr_time << std::endl;
        if (do_highres) {
            double firstMomentVorticityIntegral_res2;
            double secondMomentVorticityIntegral_res2;
            std::tie(firstMomentVorticityIntegral_res2, secondMomentVorticityIntegral_res2) = getCasimirsAtCustomRes(2);
            std::cout << RED << "Vorticity Moment 1 (at 2x res) = " << firstMomentVorticityIntegral_res2 << RESET << std::endl;
            std::cout << RED << "Vorticity Moment 2 (at 2x res) = " << secondMomentVorticityIntegral_res2 << RESET << std::endl;
            foutU5 << firstMomentVorticityIntegral_res2 << " " << curr_time << std::endl;
            foutU6 << secondMomentVorticityIntegral_res2 << " " << curr_time << std::endl;
            double firstMomentVorticityIntegral_res4;
            double secondMomentVorticityIntegral_res4;
            std::tie(firstMomentVorticityIntegral_res4, secondMomentVorticityIntegral_res4) = getCasimirsAtCustomRes(4);
            std::cout << RED << "Vorticity Moment 1 (at 4x res) = " << firstMomentVorticityIntegral_res4 << RESET << std::endl;
            std::cout << RED << "Vorticity Moment 2 (at 4x res) = " << secondMomentVorticityIntegral_res4 << RESET << std::endl;
            foutU7 << firstMomentVorticityIntegral_res4 << " " << curr_time << std::endl;
            foutU8 << secondMomentVorticityIntegral_res4 << " " << curr_time << std::endl;
            double firstMomentVorticityIntegral_res8;
            double secondMomentVorticityIntegral_res8;
            std::tie(firstMomentVorticityIntegral_res8, secondMomentVorticityIntegral_res8) = getCasimirsAtCustomRes(8);
            std::cout << RED << "Vorticity Moment 1 (at 8x res) = " << firstMomentVorticityIntegral_res8 << RESET << std::endl;
            std::cout << RED << "Vorticity Moment 2 (at 8x res) = " << secondMomentVorticityIntegral_res8 << RESET << std::endl;
            foutU9 << firstMomentVorticityIntegral_res8 << " " << curr_time << std::endl;
            foutU10 << secondMomentVorticityIntegral_res8 << " " << curr_time << std::endl;
        }
    }
    foutU.close();
    foutU2.close();
    foutU3.close();
    foutU4.close();
    if (do_highres) {
        foutU5.close();
        foutU6.close();
        foutU7.close();
        foutU8.close();
        foutU9.close();
        foutU10.close();
    }
}