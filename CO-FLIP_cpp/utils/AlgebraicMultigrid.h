#ifndef _AMG_H_
#define _AMG_H_

#include <iostream>
#include <vector>
#include "tbb/tbb.h"
#include "../include/vec.h"
#include <cmath>
#include "../include/sparse_matrix.h"
#include "blas_wrapper.h"
#include "GeometricLevelGen.h"

/*
given A_L, R_L, P_L, b,compute x using
Multigrid Cycles.

*/

template<class T>
void RBGS(const Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> &A,
          const Eigen::VectorX<T> &b,
          Eigen::VectorX<T> &x,
          int ni, int nj, int nk, int iternum, GMG gmg_flag, T wmax) {
    size_t num = 
        gmg_flag == GMG::EDGES ? (ni * (nj+1) * (nk+1) +
                                  (ni+1) * nj * (nk+1) +
                                  (ni+1) * (nj+1) * nk)
        : (gmg_flag == GMG::FACES ? ((ni+1) * nj * nk +
                                     ni * (nj+1) * nk +
                                     ni * nj * (nk+1))
        : (ni * nj * nk));
    size_t slice = ni * nj;

    Eigen::VectorX<T> r, d, d_temp;
    T wmin = (T)(1./30.)*wmax; // suggestion from: Parallel multigrid smoothing: polynomial versus Gauss–Seidel 2003
    T theta = (wmax+wmin)*(T)0.5;
    T delta = (wmax-wmin)*(T)0.5;
    T sigma = theta/delta;
    T rho = (T)1./sigma;
    if (gmg_flag == GMG::EDGES || gmg_flag == FACES) {
        r.resize(x.size());
        r.setZero();
        d.resize(x.size());
        d.setZero();
        d_temp.resize(x.size());
        d_temp.setZero();
        tbb::parallel_for(tbb::blocked_range<int>(0,num, num > 20000 ? num/100 : num), [&](const tbb::blocked_range<int> &range)
        {
            for (int thread_idx = range.begin(); thread_idx < range.end(); ++thread_idx) {
                int index = thread_idx;
                T sum = 0;
                T diag = 0;
                for (int ii = A.outerIndexPtr()[index]; ii < A.outerIndexPtr()[index + 1]; ii++) {
                    if (A.innerIndexPtr()[ii] != index)//none diagonal terms
                    {
                        sum += A.valuePtr()[ii] * x[A.innerIndexPtr()[ii]];
                    } else//record diagonal value A(i,i)
                    {
                        diag = A.valuePtr()[ii];
                    }
                }//A(i,:)*x for off-diag terms
                if (diag != 0) {
                    r[index] = (b[index] - sum) / diag - x[index];
                } else {
                    r[index] = 0;
                }
                d[index] = r[index] / theta;
            }
        });
    }
    for (int iter = 0; iter < iternum; iter++) {
        if (gmg_flag == GMG::EDGES || gmg_flag == GMG::FACES) {
            tbb::parallel_for(tbb::blocked_range<int>(0,num, num > 20000 ? num/100 : num), [&](const tbb::blocked_range<int> &range)
            {
                for (int thread_idx = range.begin(); thread_idx < range.end(); ++thread_idx) {
                    int index = thread_idx;

                    if (iter%2==0) {
                        x[index] += d[index];
                    } else {
                        x[index] += d_temp[index];
                    }
                    if (iter < iternum - 1) {
                        T sum = 0;
                        T diag = 0;
                        for (int ii = A.outerIndexPtr()[index]; ii < A.outerIndexPtr()[index + 1]; ii++) {
                            if (A.innerIndexPtr()[ii] != index)//none diagonal terms
                            {
                                if (iter%2==0) {
                                    sum += A.valuePtr()[ii] * d[A.innerIndexPtr()[ii]];
                                } else {
                                    sum += A.valuePtr()[ii] * d_temp[A.innerIndexPtr()[ii]];
                                }
                            } else//record diagonal value A(i,i)
                            {
                                diag = A.valuePtr()[ii];
                            }
                        }//A(i,:)*x for off-diag terms
                        if (diag != 0) {
                            if (iter%2==0) {
                                r[index] -= sum / diag + d[index];
                            } else {
                                r[index] -= sum / diag + d_temp[index];
                            }
                        }
                        T rho_next = T(1.) / ((T)2. * sigma - rho);
                        if (iter%2==0) {
                            d_temp[index] = rho_next*(rho*d[index] + (T)2. / delta * r[index]);
                        } else {
                            d[index] = rho_next*(rho*d_temp[index] + (T)2. / delta * r[index]);
                        }
                    }
                }
            });
            rho = T(1.) / ((T)2. * sigma - rho);
        } else {
            tbb::parallel_for((size_t) 0, num, (size_t) 1, [&](size_t thread_idx) {
                int k = thread_idx / slice;
                int j = (thread_idx % slice) / ni;
                int i = thread_idx % ni;
                if (k < nk && j < nj && i < ni) {
                    if ((i + j + k) % 2 == 1) {
                        int index = thread_idx;
                        T sum = 0;
                        T diag = 0;
                        for (int ii = A.outerIndexPtr()[index]; ii < A.outerIndexPtr()[index + 1]; ii++) {
                            if (A.innerIndexPtr()[ii] != index)//none diagonal terms
                            {
                                sum += A.valuePtr()[ii] * x[A.innerIndexPtr()[ii]];
                            } else//record diagonal value A(i,i)
                            {
                                diag = A.valuePtr()[ii];
                            }
                        }//A(i,:)*x for off-diag terms
                        if (diag != 0) {
                            x[index] = (b[index] - sum) / diag;
                        } else {
                            x[index] = 0;
                        }
                    }
                }
            });
            tbb::parallel_for((size_t) 0, num, (size_t) 1, [&](size_t thread_idx) {
                int k = thread_idx / slice;
                int j = (thread_idx % slice) / ni;
                int i = thread_idx % ni;
                if (k < nk && j < nj && i < ni) {
                    if ((i + j + k) % 2 == 0) {
                        int index = thread_idx;
                        T sum = 0;
                        T diag = 0;
                        for (int ii = A.outerIndexPtr()[index]; ii < A.outerIndexPtr()[index + 1]; ii++) {
                            if (A.innerIndexPtr()[ii] != index)//none diagonal terms
                            {
                                sum += A.valuePtr()[ii] * x[A.innerIndexPtr()[ii]];
                            } else//record diagonal value A(i,i)
                            {
                                diag = A.valuePtr()[ii];
                            }
                        }//A(i,:)*x for off-diag terms
                        if (diag != 0) {
                            x[index] = (b[index] - sum) / diag;
                        } else {
                            x[index] = 0;
                        }
                    }
                }
            });
        }
    }
}

template<class T>
void RBGS2D(const Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> &A,
            const Eigen::VectorX<T> &b,
            Eigen::VectorX<T> &x,
            int ni, int nj, int iternum,
            bool is_nondiagonal_hodgestar, GMG gmg_flag) {

    size_t num = gmg_flag != GMG::NODES ? ((ni+1) * nj + ni * (nj+1)) : (ni * nj);
    Eigen::VectorX<T> x_temp;
    if (is_nondiagonal_hodgestar || gmg_flag != GMG::NODES) {
        x_temp.resize(x.size());
        x_temp.setZero();
    }
    for (int iter = 0; iter < iternum; iter++) {
        if (is_nondiagonal_hodgestar || gmg_flag != GMG::NODES) {
            tbb::parallel_for(tbb::blocked_range<int>(0,num, num > 20000 ? num/100 : num), [&](const tbb::blocked_range<int> &range)
            {
                for (int thread_idx = range.begin(); thread_idx < range.end(); ++thread_idx) {
                    int index = thread_idx;
                    T sum = 0;
                    T diag = 0;
                    for (int ii = A.outerIndexPtr()[index]; ii < A.outerIndexPtr()[index + 1]; ii++) {
                        if (A.innerIndexPtr()[ii] != index)//none diagonal terms
                        {
                            if (iter%2==0) {
                                sum += A.valuePtr()[ii] * x[A.innerIndexPtr()[ii]];
                            } else {
                                sum += A.valuePtr()[ii] * x_temp[A.innerIndexPtr()[ii]];
                            }
                        } else//record diagonal value A(i,i)
                        {
                            diag = A.valuePtr()[ii];
                        }
                    }//A(i,:)*x for off-diag terms
                    if (diag != 0) {
                        T weight = 0.72;
                        if (iter%2==0) {
                            x_temp[index] = weight * (b[index] - sum) / diag + (static_cast<T>(1.) - weight) * x[index];
                        } else {
                            x[index] = weight * (b[index] - sum) / diag + (static_cast<T>(1.) - weight) * x_temp[index];
                        }
                    } else {
                        if (iter%2==0) {
                            x_temp[index] = 0;
                        } else {
                            x[index] = 0;
                        }
                    }
                }
            });
        } else {
            // size_t slice = ni * nj;
            tbb::parallel_for((size_t) 0, num, (size_t) 1, [&](size_t thread_idx) {
                int j = thread_idx / ni;
                int i = thread_idx % ni;
                if (j < nj && i < ni) {
                    if ((i + j) % 2 == 1) {
                        int index = thread_idx;
                        T sum = 0;
                        T diag = 0;
                        for (int ii = A.outerIndexPtr()[index]; ii < A.outerIndexPtr()[index + 1]; ii++) {
                            if (A.innerIndexPtr()[ii] != index)//none diagonal terms
                            {
                                sum += A.valuePtr()[ii] * x[A.innerIndexPtr()[ii]];
                            } else//record diagonal value A(i,i)
                            {
                                diag = A.valuePtr()[ii];
                            }
                        }//A(i,:)*x for off-diag terms
                        if (diag != 0) {
                            x[index] = (b[index] - sum) / diag;
                        } else {
                            x[index] = 0;
                        }
                    }
                }

            });

            tbb::parallel_for((size_t) 0, num, (size_t) 1, [&](size_t thread_idx) {
                int j = thread_idx / ni;
                int i = thread_idx % ni;
                if (j < nj && i < ni) {
                    if ((i + j) % 2 == 0) {
                        int index = thread_idx;
                        T sum = 0;
                        T diag = 0;
                        for (int ii = A.outerIndexPtr()[index]; ii < A.outerIndexPtr()[index + 1]; ii++) {
                            if (A.innerIndexPtr()[ii] != index)//none diagonal terms
                            {
                                sum += A.valuePtr()[ii] * x[A.innerIndexPtr()[ii]];
                            } else//record diagonal value A(i,i)
                            {
                                diag = A.valuePtr()[ii];
                            }
                        }//A(i,:)*x for off-diag terms
                        if (diag != 0) {
                            x[index] = (b[index] - sum) / diag;
                        } else {
                            x[index] = 0;
                        }
                    }
                }

            });
        }
    }
}

template<class T>
void restriction(const Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> &R,
                 const Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> &A,
                 const Eigen::VectorX<T> &x,
                 const Eigen::VectorX<T> &b_curr,
                 Eigen::VectorX<T> &b_next) {
    b_next.noalias() = R*(b_curr-A*x);
}

template<class T>
void prolongatoin(const Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> &P,
                  const Eigen::VectorX<T> &x_curr,
                  Eigen::VectorX<T> &x_next) {
    x_next += P*x_curr;
}

template<class T>
void amgVCycle(std::vector<Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> > &A_L,
               std::vector<Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> > &R_L,
               std::vector<Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> > &P_L,
               std::vector<T> & wmax,
               std::vector<Vec3i> &S_L,
               Eigen::VectorX<T> &x,
               const Eigen::VectorX<T> &b, GMG gmg_flag=GMG::NODES) {
    int total_level = A_L.size();
    std::vector<Eigen::VectorX<T> > x_L;
    std::vector<Eigen::VectorX<T> > b_L;
    x_L.resize(total_level);
    b_L.resize(total_level);
    b_L[0] = b;
    x_L[0] = x;
    for (int i = 1; i < total_level; i++) {
        int ni = S_L[i].v[0];
        int nj = S_L[i].v[1];
        int nk = S_L[i].v[2];
        int unknowns = 
            gmg_flag == GMG::EDGES ? (ni * (nj+1) * (nk+1) +
                                      (ni+1) * nj * (nk+1) +
                                      (ni+1) * (nj+1) * nk)
            : (gmg_flag == GMG::FACES ? ((ni+1) * nj * nk +
                                         ni * (nj+1) * nk +
                                         ni * nj * (nk+1))
            : (ni * nj * nk));
        x_L[i].resize(unknowns);
        x_L[i].setZero();
        b_L[i].resize(unknowns);
        b_L[i].setZero();
    }

    for (int i = 0; i < total_level - 1; i++) {
        RBGS(A_L[i], b_L[i], x_L[i], S_L[i].v[0], S_L[i].v[1], S_L[i].v[2], gmg_flag != GMG::NODES ? 1 : 4, gmg_flag, wmax[i]);
        restriction(R_L[i], A_L[i], x_L[i], b_L[i], b_L[i + 1]);
    }
    int i = total_level - 1;
    // RBGS(A_L[i], b_L[i], x_L[i], S_L[i].v[0], S_L[i].v[1], S_L[i].v[2], 500, gmg_flag, wmax[i]);
    Eigen::MatrixX<T> A(A_L[i]);
    x_L[i] = A.ldlt().solve(b_L[i]);
    for (int i = total_level - 2; i >= 0; i--) {
        prolongatoin(P_L[i], x_L[i + 1], x_L[i]);
        RBGS(A_L[i], b_L[i], x_L[i], S_L[i].v[0], S_L[i].v[1], S_L[i].v[2], gmg_flag != GMG::NODES ? 1 : 4, gmg_flag, wmax[i]);
    }
    x = x_L[0];
}

template<class T>
void amgVCycle2D(std::vector<Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> > &A_L,
                 std::vector<Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> > &R_L,
                 std::vector<Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> > &P_L,
                 std::vector<Vec2i> &S_L,
                 Eigen::VectorX<T> &x,
                 const Eigen::VectorX<T> &b,
                 bool is_nondiagonal_hodgestar, GMG gmg_flag=GMG::NODES) {
    int total_level = A_L.size();
    std::vector<Eigen::VectorX<T> > x_L;
    std::vector<Eigen::VectorX<T> > b_L;
    x_L.resize(total_level);
    b_L.resize(total_level);
    b_L[0] = b;
    x_L[0] = x;
    for (int i = 1; i < total_level; i++) {
        int ni = S_L[i].v[0];
        int nj = S_L[i].v[1];
        int unknowns = gmg_flag != GMG::NODES ? ((ni+1) * nj + ni * (nj+1)) : (ni * nj);
        x_L[i].resize(unknowns);
        x_L[i].setZero();
        b_L[i].resize(unknowns);
        b_L[i].setZero();
    }

    for (int i = 0; i < total_level - 1; i++) {
        RBGS2D(A_L[i], b_L[i], x_L[i], S_L[i].v[0], S_L[i].v[1], 4, is_nondiagonal_hodgestar, gmg_flag);
        restriction(R_L[i], A_L[i], x_L[i], b_L[i], b_L[i + 1]);
    }
    int i = total_level - 1;
    // RBGS2D(A_L[i], b_L[i], x_L[i], S_L[i].v[0], S_L[i].v[1], 500, is_nondiagonal_hodgestar, gmg_flag);
    Eigen::MatrixX<T> A(A_L[i]);
    x_L[i] = A.ldlt().solve(b_L[i]);
    for (int i = total_level - 2; i >= 0; i--) {
        prolongatoin(P_L[i], x_L[i + 1], x_L[i]);
        RBGS2D(A_L[i], b_L[i], x_L[i], S_L[i].v[0], S_L[i].v[1], 4, is_nondiagonal_hodgestar, gmg_flag);
    }
    x = x_L[0];
}

template<class T>
void amgPrecond(std::vector<Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> > &A_L,
                std::vector<Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> > &R_L,
                std::vector<Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> > &P_L,
                std::vector<T> & wmax,
                std::vector<Vec3i> &S_L,
                Eigen::VectorX<T> &x,
                const Eigen::VectorX<T> &b, GMG gmg_flag=GMG::NODES) {
    x.resize(b.size());
    x.setZero();
    amgVCycle(A_L, R_L, P_L, wmax, S_L, x, b, gmg_flag);
}

template<class T>
void amgPrecond2D(std::vector<Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> > &A_L,
                  std::vector<Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> > &R_L,
                  std::vector<Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> > &P_L,
                  std::vector<Vec2i> &S_L,
                  Eigen::VectorX<T> &x,
                  const Eigen::VectorX<T> &b,
                  bool is_nondiagonal_hodgestar, GMG gmg_flag=GMG::NODES) {
    x.resize(b.size());
    x.setZero();
    amgVCycle2D(A_L, R_L, P_L, S_L, x, b, is_nondiagonal_hodgestar, gmg_flag);
}

template<class T>
bool AMGPCGSolvePrebuilt2D(std::function<void(Eigen::VectorX<T>&, const Eigen::VectorX<T>&)> multiply_with_fixed_matrix,
                           const Eigen::VectorX<T> &rhs,
                           Eigen::VectorX<T> &result,
                           std::vector<Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> > &A_L,
                           std::vector<Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> > &R_L,
                           std::vector<Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> > &P_L,
                           std::vector<Vec2i> &S_L,
                           const int total_level,
                           T tolerance_factor,
                           int max_iterations,
                           T &residual_out,
                           int &iterations_out,
                           bool is_nondiagonal_hodgestar=false,
                           GMG gmg_flag=GMG::NODES
                           ) {
    T rhs_norm = rhs.template lpNorm<2>();
    if (rhs_norm < tolerance_factor) {
        result.setZero();
        iterations_out = 0;
        residual_out = 0;
        return true;
    }

    Eigen::VectorX<T> z, s, r;
    unsigned int n = result.size();
    Eigen::VectorX<T> tmp(n);
    multiply_with_fixed_matrix(tmp, result);
    r.noalias() = rhs - tmp;
    // if (PURE_NEUMANN) {
    //     r.array() = r.array() - r.mean();
    // }
    residual_out = r.template lpNorm<2>();
    T tol = std::max(tolerance_factor * rhs_norm, std::numeric_limits<T>::min());
    if (residual_out < tol) {
        iterations_out = 0;
        residual_out = residual_out / rhs_norm;
        return true;
    }

    s.resize(n);
    z.resize(n);
    amgPrecond2D(A_L, R_L, P_L, S_L, s, r, is_nondiagonal_hodgestar, gmg_flag);
    T rho = s.dot(r);
    if (rho == 0 || rho != rho) {
        iterations_out = 0;
        return false;
    }

    int iteration;
    for (iteration = 0; iteration < max_iterations; ++iteration) {
        multiply_with_fixed_matrix(tmp, s);
        T alpha = rho / s.dot(tmp);
        result += alpha*s;
        r -= alpha*tmp;
        // if (PURE_NEUMANN) {
        //     r.array() = r.array() - r.mean();
        // }
        residual_out = r.template lpNorm<2>();
        if (residual_out < tol) {
            iterations_out = iteration + 1;
            residual_out = residual_out / rhs_norm;
            return true;
        }
        amgPrecond2D(A_L, R_L, P_L, S_L, z, r, is_nondiagonal_hodgestar, gmg_flag);
        T rho_new = z.dot(r);
        T beta = rho_new / rho;
        s = beta*s+z;
        rho = rho_new;
    }
    iterations_out = iteration;
    residual_out = residual_out / rhs_norm;
    return false;
}

template<class T>
bool AMGPCGSolvePrebuilt2D(const Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> &fixed_matrix,
                           const Eigen::VectorX<T> &rhs,
                           Eigen::VectorX<T> &result,
                           std::vector<Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> > &A_L,
                           std::vector<Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> > &R_L,
                           std::vector<Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> > &P_L,
                           std::vector<Vec2i> &S_L,
                           const int total_level,
                           T tolerance_factor,
                           int max_iterations,
                           T &residual_out,
                           int &iterations_out,
                           int ni, int nj,
                           bool PURE_NEUMANN, bool is_nondiagonal_hodgestar=false, GMG gmg_flag=GMG::NODES) {
    T rhs_norm = rhs.template lpNorm<2>();
    if (rhs_norm < tolerance_factor) {
        result.setZero();
        iterations_out = 0;
        residual_out = 0;
        return true;
    }

    Eigen::VectorX<T> z, s, r;
    r.noalias() = rhs - fixed_matrix * result;
    if (PURE_NEUMANN) {
        r.array() = r.array() - r.mean();
    }
    residual_out = r.template lpNorm<2>();
    T tol = std::max(tolerance_factor * rhs_norm, std::numeric_limits<T>::min());
    if (residual_out < tol) {
        iterations_out = 0;
        residual_out = residual_out / rhs_norm;
        return true;
    }

    unsigned int n = result.size();
    s.resize(n);
    z.resize(n);
    amgPrecond2D(A_L, R_L, P_L, S_L, s, r, is_nondiagonal_hodgestar, gmg_flag);
    T rho = s.dot(r);
    if (rho == 0 || rho != rho) {
        iterations_out = 0;
        return false;
    }

    int iteration;
    Eigen::VectorX<T> tmp(n);
    for (iteration = 0; iteration < max_iterations; ++iteration) {
        tmp.noalias() = fixed_matrix * s;
        T alpha = rho / s.dot(tmp);
        result += alpha*s;
        r -= alpha*tmp;
        if (PURE_NEUMANN) {
            r.array() = r.array() - r.mean();
        }
        residual_out = r.template lpNorm<2>();
        if (residual_out < tol) {
            iterations_out = iteration + 1;
            residual_out = residual_out / rhs_norm;
            return true;
        }
        amgPrecond2D(A_L, R_L, P_L, S_L, z, r, is_nondiagonal_hodgestar, gmg_flag);
        T rho_new = z.dot(r);
        T beta = rho_new / rho;
        s = beta*s+z;
        rho = rho_new;
    }
    iterations_out = iteration;
    residual_out = residual_out / rhs_norm;
    return false;
}

template<class T>
bool AMGPCGSolvePrebuilt2D(std::function<void(Eigen::VectorX<T>&, const Eigen::VectorX<T>&)> multiply_with_fixed_matrix,
                           const Eigen::VectorX<T> &rhs,
                           Eigen::VectorX<T> &result,
                           std::vector<Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> > &A_L,
                           std::vector<Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> > &R_L,
                           std::vector<Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> > &P_L,
                           std::vector<Vec2i> &S_L,
                           const int total_level,
                           T tolerance_factor,
                           int max_iterations,
                           T &residual_out,
                           int &iterations_out,
                           int ni, int nj,
                           bool PURE_NEUMANN, bool is_nondiagonal_hodgestar=false, GMG gmg_flag=GMG::NODES) {
    T rhs_norm = rhs.template lpNorm<2>();
    if (rhs_norm < tolerance_factor) {
        result.setZero();
        iterations_out = 0;
        residual_out = 0;
        return true;
    }

    Eigen::VectorX<T> z, s, r;
    unsigned int n = result.size();
    Eigen::VectorX<T> tmp(n);
    multiply_with_fixed_matrix(tmp, result);
    r.noalias() = rhs - tmp;
    if (PURE_NEUMANN) {
        r.array() = r.array() - r.mean();
    }
    residual_out = r.template lpNorm<2>();
    T tol = std::max(tolerance_factor * rhs_norm, std::numeric_limits<T>::min());
    if (residual_out < tol) {
        iterations_out = 0;
        residual_out = residual_out / rhs_norm;
        return true;
    }

    s.resize(n);
    z.resize(n);
    amgPrecond2D(A_L, R_L, P_L, S_L, s, r, is_nondiagonal_hodgestar, gmg_flag);
    T rho = s.dot(r);
    if (rho == 0 || rho != rho) {
        iterations_out = 0;
        return false;
    }

    int iteration;
    for (iteration = 0; iteration < max_iterations; ++iteration) {
        multiply_with_fixed_matrix(tmp, s);
        T alpha = rho / s.dot(tmp);
        result += alpha*s;
        r -= alpha*tmp;
        if (PURE_NEUMANN) {
            r.array() = r.array() - r.mean();
        }
        residual_out = r.template lpNorm<2>();
        if (residual_out < tol) {
            iterations_out = iteration + 1;
            residual_out = residual_out / rhs_norm;
            return true;
        }
        amgPrecond2D(A_L, R_L, P_L, S_L, z, r, is_nondiagonal_hodgestar, gmg_flag);
        T rho_new = z.dot(r);
        T beta = rho_new / rho;
        s = beta*s+z;
        rho = rho_new;
    }
    iterations_out = iteration;
    residual_out = residual_out / rhs_norm;
    return false;
}

template<class T>
bool AMGPCGSolve(const Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> &matrix,
                 const Eigen::VectorX<T> &rhs,
                 Eigen::VectorX<T> &result,
                 T tolerance_factor,
                 int max_iterations,
                 T &residual_out,
                 int &iterations_out,
                 int ni, int nj, int nk, GMG gmg_flag=GMG::NODES) {
    T rhs_norm = rhs.template lpNorm<2>();
    if (rhs_norm < tolerance_factor) {
        result.setZero();
        iterations_out = 0;
        residual_out = 0;

        return true;
    }

    Eigen::VectorX<T> z, s, r;
    r.noalias() = rhs - matrix * result;
    residual_out = r.template lpNorm<2>();
    T tol = std::max(tolerance_factor * rhs_norm, std::numeric_limits<T>::min());
    if (residual_out < tol) {
        iterations_out = 0;
        residual_out = residual_out / rhs_norm;

        return true;
    }
    
    std::vector<Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> > A_L;
    std::vector<Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> > R_L;
    std::vector<Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> > P_L;
    std::vector<T> wmax;
    std::vector<Vec3i> S_L;
    int total_level;
    levelGen<T> amg_levelGen;
    amg_levelGen.generateLevelsGalerkinCoarsening(A_L, R_L, P_L, wmax, S_L, total_level, matrix, ni, nj, nk, gmg_flag);

    unsigned int n = matrix.n;
    s.resize(n);
    z.resize(n);
    amgPrecond(A_L, R_L, P_L, wmax, S_L, s, r, gmg_flag);
    T rho = s.dot(r);
    if (rho == 0 || rho != rho) {
        iterations_out = 0;
        return false;
    }

    int iteration;
    Eigen::VectorX<T> tmp(n);
    for (iteration = 0; iteration < max_iterations; ++iteration) {
        tmp.noalias() = matrix * s;
        T alpha = rho / s.dot(tmp);
        result += alpha*s;
        r -= alpha*tmp;
        residual_out = r.template lpNorm<2>();
        if (residual_out < tol) {
            iterations_out = iteration + 1;
            residual_out = residual_out / rhs_norm;

            return true;
        }
        amgPrecond(A_L, R_L, P_L, wmax, S_L, z, r, gmg_flag);
        T rho_new = z.dot(r);
        T beta = rho_new / rho;
        s = beta*s+z; 
        rho = rho_new;
    }
    iterations_out = iteration;
    residual_out = residual_out / rhs_norm;
    return false;
}


template<class T>
bool AMGPCGSolve(const Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> &fixed_matrix,
                 const Eigen::VectorX<T> &rhs,
                 Eigen::VectorX<T> &result,
                 std::vector<Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> > &A_L,
                 std::vector<Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> > &R_L,
                 std::vector<Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> > &P_L,
                 std::vector<T> &wmax,
                 std::vector<Vec3i> &S_L,
                 int total_level,
                 T tolerance_factor,
                 int max_iterations,
                 T &residual_out,
                 int &iterations_out,
                 int ni, int nj, int nk, GMG gmg_flag=GMG::NODES) {

    T rhs_norm = rhs.template lpNorm<2>();
    if (rhs_norm < tolerance_factor) {
        result.setZero();
        iterations_out = 0;
        residual_out = 0;
        return true;
    }

    Eigen::VectorX<T> z, s, r;
    r.noalias() = rhs - fixed_matrix * result;

    residual_out = r.template lpNorm<2>();
    T tol = std::max(tolerance_factor * rhs_norm, std::numeric_limits<T>::min());
    if (residual_out < tol) {
        iterations_out = 0;
        residual_out = residual_out / rhs_norm;

        return true;
    }  

    unsigned int n = result.size();
    s.resize(n);
    z.resize(n);
    amgPrecond(A_L, R_L, P_L, wmax, S_L, s, r, gmg_flag);
    T rho = s.dot(r);
    if (rho == 0 || rho != rho) {
        iterations_out = 0;
        return false;
    }

    int iteration;
    Eigen::VectorX<T> tmp(n);
    for (iteration = 0; iteration < max_iterations; ++iteration) {
        tmp.noalias() = fixed_matrix * s;
        T alpha = rho / s.dot(tmp);
        result += alpha*s;
        r -= alpha*tmp;
        residual_out = r.template lpNorm<2>();
        if (residual_out < tol) {
            iterations_out = iteration + 1;
            residual_out = residual_out / rhs_norm;

            return true;
        }  
        amgPrecond(A_L, R_L, P_L, wmax, S_L, z, r, gmg_flag);
        T rho_new = z.dot(r);
        T beta = rho_new / rho;
        s = beta*s+z;
        rho = rho_new;
    }
    iterations_out = iteration;
    residual_out = residual_out / rhs_norm;

    return false;
}


template<class T>
bool AMGPCGSolve(std::function<void(Eigen::VectorX<T>&, const Eigen::VectorX<T>&)> multiply_with_fixed_matrix,
                 const Eigen::VectorX<T> &rhs,
                 Eigen::VectorX<T> &result,
                 std::vector<Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> > &A_L,
                 std::vector<Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> > &R_L,
                 std::vector<Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> > &P_L,
                 std::vector<T> &wmax,
                 std::vector<Vec3i> &S_L,
                 int total_level,
                 T tolerance_factor,
                 int max_iterations,
                 T &residual_out,
                 int &iterations_out,
                 int ni, int nj, int nk, GMG gmg_flag=GMG::NODES) {

    T rhs_norm = rhs.template lpNorm<2>();
    if (rhs_norm < tolerance_factor) {
        result.setZero();
        iterations_out = 0;
        residual_out = 0;
        return true;
    }

    Eigen::VectorX<T> z, s, r;
    unsigned int n = result.size();
    Eigen::VectorX<T> tmp(n);
    multiply_with_fixed_matrix(tmp, result);
    r.noalias() = rhs - tmp;

    residual_out = r.template lpNorm<2>();
    T tol = std::max(tolerance_factor * rhs_norm, std::numeric_limits<T>::min());
    if (residual_out < tol) {
        iterations_out = 0;
        residual_out = residual_out / rhs_norm;

        return true;
    }  

    s.resize(n);
    z.resize(n);
    amgPrecond(A_L, R_L, P_L, wmax, S_L, s, r, gmg_flag);
    T rho = s.dot(r);
    if (rho == 0 || rho != rho) {
        iterations_out = 0;
        return false;
    }

    int iteration;
    for (iteration = 0; iteration < max_iterations; ++iteration) {
        multiply_with_fixed_matrix(tmp, s);
        T alpha = rho / s.dot(tmp);
        result += alpha*s;
        r -= alpha*tmp;
        residual_out = r.template lpNorm<2>();
        if (residual_out < tol) {
            iterations_out = iteration + 1;
            residual_out = residual_out / rhs_norm;

            return true;
        }  
        amgPrecond(A_L, R_L, P_L, wmax, S_L, z, r, gmg_flag);
        T rho_new = z.dot(r);
        T beta = rho_new / rho;
        s = beta*s+z;
        rho = rho_new;
    }
    iterations_out = iteration;
    residual_out = residual_out / rhs_norm;

    return false;
}

template<class T, class Preconditioner>
bool AMGPCGSolve(std::function<void(Eigen::VectorX<T>&, const Eigen::VectorX<T>&)> multiply_with_fixed_matrix,
                 const Eigen::VectorX<T> &rhs,
                 Eigen::VectorX<T> &result,
                 const Preconditioner& precond,
                 T tolerance_factor,
                 int max_iterations,
                 T &residual_out,
                 int &iterations_out,
                 T jacobi_iter_weight=1,
                 int jacobi_iter_count=1) {

    T rhs_norm = rhs.template lpNorm<2>();
    if (rhs_norm < tolerance_factor) {
        result.setZero();
        iterations_out = 0;
        residual_out = 0;
        return true;
    }

    Eigen::VectorX<T> z, s, r;
    unsigned int n = result.size();
    Eigen::VectorX<T> tmp(n);
    multiply_with_fixed_matrix(tmp, result);
    r.noalias() = rhs - tmp;

    residual_out = r.template lpNorm<2>();
    T tol = std::max(tolerance_factor * rhs_norm, std::numeric_limits<T>::min());
    if (residual_out < tol) {
        iterations_out = 0;
        residual_out = residual_out / rhs_norm;

        return true;
    }  

    s.resize(n);
    z.resize(n);
    s.setZero();
    for (int jacobi_iter=0; jacobi_iter < jacobi_iter_count; jacobi_iter++) {
        if (jacobi_iter == 0) {
            tmp = r;
        } else {
            multiply_with_fixed_matrix(tmp, s);
            tmp = r-tmp;
        }
        s += jacobi_iter_weight * precond.solve(tmp);
    }
    // s = precond.solve(r);
    T rho = s.dot(r);
    if (rho == 0 || rho != rho) {
        iterations_out = 0;
        return false;
    }

    int iteration;
    for (iteration = 0; iteration < max_iterations; ++iteration) {
        multiply_with_fixed_matrix(tmp, s);
        T alpha = rho / s.dot(tmp);
        result += alpha*s;
        r -= alpha*tmp;
        residual_out = r.template lpNorm<2>();
        if (residual_out < tol) {
            iterations_out = iteration + 1;
            residual_out = residual_out / rhs_norm;

            return true;
        }
        z.setZero();
        for (int jacobi_iter=0; jacobi_iter < jacobi_iter_count; jacobi_iter++) {
            if (jacobi_iter == 0) {
                tmp = r;
            } else {
                multiply_with_fixed_matrix(tmp, z);
                tmp = r-tmp;
            }
            z += jacobi_iter_weight * precond.solve(tmp);
        }
        // z = precond.solve(r);
        T rho_new = z.dot(r);
        T beta = rho_new / rho;
        s = beta*s+z;
        rho = rho_new;
    }
    iterations_out = iteration;
    residual_out = residual_out / rhs_norm;

    return false;
}


template<class T>
bool AMGPLSCGSolve(const Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> &fixed_matrix,
                 const Eigen::VectorX<T> &rhs,
                 Eigen::VectorX<T> &result,
                 std::vector<Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> > &A_L,
                 std::vector<Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> > &R_L,
                 std::vector<Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> > &P_L,
                 std::vector<T> &wmax,
                 std::vector<Vec3i> &S_L,
                 int total_level,
                 T tolerance_factor,
                 int max_iterations,
                 T &residual_out,
                 int &iterations_out,
                 int ni, int nj, int nk, GMG gmg_flag=GMG::NODES) {
    Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> fixed_matrix_transpose = fixed_matrix.transpose();
    T rhs_norm = (fixed_matrix_transpose * rhs).template lpNorm<2>();
    if (rhs_norm < tolerance_factor) {
        result.setZero();
        iterations_out = 0;
        residual_out = 0;
        return true;
    }

    Eigen::VectorX<T> z, s, r, r_normal;
    r.noalias() = rhs - fixed_matrix * result;
    r_normal.noalias() = fixed_matrix_transpose * r;
    residual_out = r_normal.template lpNorm<2>();
    T tol = std::max(tolerance_factor * rhs_norm, std::numeric_limits<T>::min());
    if (residual_out < tol) {
        iterations_out = 0;
        residual_out = residual_out / rhs_norm;

        return true;
    }  

    unsigned int m = fixed_matrix.rows();
    unsigned int n = result.size();
    s.resize(n);
    z.resize(n);
    amgPrecond(A_L, R_L, P_L, wmax, S_L, s, r_normal, gmg_flag);
    T rho = s.dot(r_normal);
    if (rho == 0 || rho != rho) {
        iterations_out = 0;
        return false;
    }

    int iteration;
    Eigen::VectorX<T> tmp(m);
    for (iteration = 0; iteration < max_iterations; ++iteration) {
        tmp.noalias() = fixed_matrix * s;
        T alpha = rho / tmp.squaredNorm();
        result += alpha*s;
        r -= alpha*tmp;
        r_normal.noalias() = fixed_matrix_transpose * r;
        residual_out = r_normal.template lpNorm<2>();
        if (residual_out < tol) {
            iterations_out = iteration + 1;
            residual_out = residual_out / rhs_norm;

            return true;
        }  
        amgPrecond(A_L, R_L, P_L, wmax, S_L, z, r_normal, gmg_flag);
        T rho_new = z.dot(r_normal);
        T beta = rho_new / rho;
        s = beta*s+z;
        rho = rho_new;
    }
    iterations_out = iteration;
    residual_out = residual_out / rhs_norm;

    return false;
}

template<class T, class Preconditioner>
bool PLSCGSolve(std::function<void(Eigen::VectorX<T>&, const Eigen::VectorX<T>&)> multiply_with_fixed_matrix,
                std::function<void(Eigen::VectorX<T>&, const Eigen::VectorX<T>&)> multiply_with_fixed_matrix_transpose,
                 const Eigen::VectorX<T> &rhs,
                 Eigen::VectorX<T> &result,
                 const Preconditioner& precond,
                 T tolerance_factor,
                 int max_iterations,
                 T &residual_out,
                 int &iterations_out,
                 T jacobi_iter_weight=1,
                 int jacobi_iter_count=1) {
    unsigned int n = result.size();
    Eigen::VectorX<T> tmp2(n);
    multiply_with_fixed_matrix_transpose(tmp2, rhs);
    T rhs_norm = tmp2.template lpNorm<2>();
    if (rhs_norm < tolerance_factor) {
        result.setZero();
        iterations_out = 0;
        residual_out = 0;
        return true;
    }

    Eigen::VectorX<T> z, s, r, r_normal(n);
    unsigned int m = rhs.size();
    Eigen::VectorX<T> tmp(m);
    multiply_with_fixed_matrix(tmp, result);
    r.noalias() = rhs - tmp;
    multiply_with_fixed_matrix_transpose(r_normal, r);
    residual_out = r_normal.template lpNorm<2>();
    T tol = std::max(tolerance_factor * rhs_norm, std::numeric_limits<T>::min());
    if (residual_out < tol) {
        iterations_out = 0;
        residual_out = residual_out / rhs_norm;

        return true;
    }  
    
    s.resize(n);
    z.resize(n);
    s.setZero();
    for (int jacobi_iter=0; jacobi_iter < jacobi_iter_count; jacobi_iter++) {
        if (jacobi_iter == 0) {
            tmp2 = r_normal;
        } else {
            multiply_with_fixed_matrix(tmp, s);
            multiply_with_fixed_matrix_transpose(tmp2, r-tmp);
        }
        s += jacobi_iter_weight * precond.solve(tmp2);
    }
    T rho = s.dot(r_normal);
    if (rho == 0 || rho != rho) {
        iterations_out = 0;
        return false;
    }

    int iteration;
    for (iteration = 0; iteration < max_iterations; ++iteration) {
        multiply_with_fixed_matrix(tmp, s);
        T alpha = rho / tmp.squaredNorm();
        result += alpha*s;
        r -= alpha*tmp;
        multiply_with_fixed_matrix_transpose(r_normal, r);
        residual_out = r_normal.template lpNorm<2>();
        if (residual_out < tol) {
            iterations_out = iteration + 1;
            residual_out = residual_out / rhs_norm;

            return true;
        }
        z.setZero();
        for (int jacobi_iter=0; jacobi_iter < jacobi_iter_count; jacobi_iter++) {
            if (jacobi_iter == 0) {
                tmp2 = r_normal;
            } else {
                multiply_with_fixed_matrix(tmp, z);
                multiply_with_fixed_matrix_transpose(tmp2, r-tmp);
            }
            z += jacobi_iter_weight * precond.solve(tmp2);
        }
        T rho_new = z.dot(r_normal);
        T beta = rho_new / rho;
        s = beta*s+z;
        rho = rho_new;
    }
    iterations_out = iteration;
    residual_out = residual_out / rhs_norm;

    return false;
}

template<class T, class Preconditioner>
bool PLSCGSolve(const Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> &fixed_matrix,
                 const Eigen::VectorX<T> &rhs,
                 Eigen::VectorX<T> &result,
                 const Preconditioner& precond,
                 T tolerance_factor,
                 int max_iterations,
                 T &residual_out,
                 int &iterations_out) {
    Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> fixed_matrix_transpose = fixed_matrix.transpose();
    T rhs_norm = (fixed_matrix_transpose * rhs).template lpNorm<2>();
    if (rhs_norm < tolerance_factor) {
        result.setZero();
        iterations_out = 0;
        residual_out = 0;
        return true;
    }

    Eigen::VectorX<T> z, s, r, r_normal;
    r.noalias() = rhs - fixed_matrix * result;
    r_normal.noalias() = fixed_matrix_transpose * r;
    residual_out = r_normal.template lpNorm<2>();
    T tol = std::max(tolerance_factor * rhs_norm, std::numeric_limits<T>::min());
    if (residual_out < tol) {
        iterations_out = 0;
        residual_out = residual_out / rhs_norm;

        return true;
    }  

    unsigned int m = fixed_matrix.rows();
    unsigned int n = result.size();
    s.resize(n);
    z.resize(n);
    s = precond.solve(r_normal);
    T rho = s.dot(r_normal);
    if (rho == 0 || rho != rho) {
        iterations_out = 0;
        return false;
    }

    int iteration;
    Eigen::VectorX<T> tmp(m);
    for (iteration = 0; iteration < max_iterations; ++iteration) {
        tmp.noalias() = fixed_matrix * s;
        T alpha = rho / tmp.squaredNorm();
        result += alpha*s;
        r -= alpha*tmp;
        r_normal.noalias() = fixed_matrix_transpose * r;
        residual_out = r_normal.template lpNorm<2>();
        if (residual_out < tol) {
            iterations_out = iteration + 1;
            residual_out = residual_out / rhs_norm;

            return true;
        }
        z = precond.solve(r_normal);
        T rho_new = z.dot(r_normal);
        T beta = rho_new / rho;
        s = beta*s+z;
        rho = rho_new;
    }
    iterations_out = iteration;
    residual_out = residual_out / rhs_norm;

    return false;
}

#endif
