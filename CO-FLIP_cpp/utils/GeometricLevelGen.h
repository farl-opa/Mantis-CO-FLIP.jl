#ifndef _geo_level_gen_h_
#define _geo_level_gen_h_

#include <iostream>
#include <vector>
#include <unordered_map>
#include "tbb/tbb.h"
#include "../include/vec.h"
#include <cmath>
#include "../include/sparse_matrix.h"
#include "blas_wrapper.h"
#include "../utils/Eigen/Sparse"
#include "../utils/Spectra/SymEigsSolver.h"
#include "../utils/Spectra/MatOp/SparseSymMatProd.h"


namespace HELPER {
    // grabbed from https://github.com/rgoldade/2DFluid/blob/master/Library/Utilities/Utilities._h#L43
    template <typename VectorType>
    void mergeLocalThreadVectors(VectorType& combinedVector, tbb::enumerable_thread_specific<VectorType>& parallelVector)
    {
        int vectorSize = 0;
        parallelVector.combine_each([&](const VectorType& localVector) {
            vectorSize += int(localVector.size());
        });

        combinedVector.reserve(combinedVector.size() + vectorSize);

        parallelVector.combine_each([&](const VectorType& localVector) {
            combinedVector.insert(combinedVector.end(), localVector.begin(), localVector.end());
        });
        parallelVector.clear();
    }

    template <typename T>
    void rowSumMatrix(const Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t>& inputMatrix, Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t>& outputMatrix)
    {
        std::vector<Eigen::Triplet<T> > tmp;
        for (int k=0; k<inputMatrix.outerSize(); ++k) {
            for (typename Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t>::InnerIterator it(inputMatrix,k); it; ++it)
            {
                // it.value();
                // it.row();   // row index
                // it.col();   // col index (here it is equal to k)
                // it.index(); // inner index, here it is equal to it.row()
                tmp.push_back(Eigen::Triplet<T>(it.row(), it.row() , it.value()));
            }
        }
        outputMatrix.setFromTriplets(tmp.begin(), tmp.end());
    }
}

enum GMG {NODES, EDGES, FACES, CELLS};
template<class T>
struct levelGen {

    void generateRP(const Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> &A,
                    Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> &R,
                    Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> &P,
                    int ni, int nj, int nk, int bs_p, GMG gmg_flag) {

        int nni = ceil((float) ni / 2.0);
        int nnj = ceil((float) nj / 2.0);
        int nnk = ceil((float) nk / 2.0);

        if (gmg_flag == GMG::EDGES) {
            size_t nE = (nk+1) * (nj+1) * ni +
                        (nk+1) * nj * (ni+1) +
                        nk * (nj+1) * (ni+1);
            size_t nnE = (nnk+1) * (nnj+1) * nni +
                         (nnk+1) * nnj * (nni+1) +
                         nnk * (nnj+1) * (nni+1);

            int x_nE = ni*(nj+1)*(nk+1);
            int y_nE = (ni+1)*nj*(nk+1);
            int x_nnE = nni*(nnj+1)*(nnk+1);
            int y_nnE = (nni+1)*nnj*(nnk+1);
            {
                tbb::enumerable_thread_specific<std::vector<Eigen::Triplet<T> > > parallelTripletList_r;
                tbb::enumerable_thread_specific<std::vector<Eigen::Triplet<T> > > parallelTripletList_p;
                tbb::parallel_for(tbb::blocked_range<int>(0,nnE,1000), [&](const tbb::blocked_range<int> &range)
                {
                    auto& localTripletList_r = parallelTripletList_r.local();
                    auto& localTripletList_p = parallelTripletList_p.local();
                    for (int tIdx = range.begin(); tIdx < range.end(); ++tIdx) {
                        
                        bool is_x_dir = tIdx < x_nnE;
                        bool is_y_dir = !is_x_dir && tIdx < (x_nnE + y_nnE);
                        // bool is_z_dir = !is_x_dir && !is_y_dir;
                        int comp_slice = is_x_dir ? nni*(nnj+1) : (is_y_dir ? (nni+1)*nnj : (nni+1)*(nnj+1));
                        int comp_n = is_x_dir ? nni : (nni+1);
                        int comp_tIdx = is_x_dir ? tIdx : (is_y_dir ? (tIdx - x_nnE) : (tIdx - (x_nnE + y_nnE)));
                        int k = comp_tIdx/comp_slice;
                        int j = (comp_tIdx%comp_slice)/comp_n;
                        int i = comp_tIdx%comp_n;
                        int idx = is_x_dir ? i : (is_y_dir ? j : k);
                        int other1_idx = is_x_dir ? j : (is_y_dir ? k : i);
                        int other2_idx = is_x_dir ? k : (is_y_dir ? i : j);
                        int nidx = is_x_dir ? ni : (is_y_dir ? nj : nk);
                        int other1_nidx = is_x_dir ? nj : (is_y_dir ? nk : ni);
                        int other2_nidx = is_x_dir ? nk : (is_y_dir ? ni : nj);
                        T sum_value = 0.;
                        std::vector<int> tJdx_idx_toadd;
                        std::vector<T> tJdx_value_toadd;
                        for (int l = -bs_p+1; l <= bs_p; l++)
                            for (int m = -bs_p; m <= bs_p; m++)
                                for (int q = -bs_p; q <= bs_p; q++) {
                                    T value = 0;
                                    if (bs_p == 1) {
                                        value = (1.) *
                                                (m == 0 ? 1. : 0.5) *
                                                (q == 0 ? 1. : 0.5);
                                    } else if (bs_p == 2) {
                                        int l_adjusted = 2*l-1;
                                        value = (std::abs(l_adjusted) == 1 ? 0.75 : 0.25) *
                                                (m == 0 ? 3./4. : (std::abs(m) == 1 ? 0.5 : 1./8.)) *
                                                (q == 0 ? 3./4. : (std::abs(q) == 1 ? 0.5 : 1./8.));
                                    } else if (bs_p == 3) {
                                        int l_adjusted = 2*l-1;
                                        value = (std::abs(l_adjusted) == 1 ? 11./16. : (std::abs(l_adjusted) == 3 ? 9./32. : 1./32.)) *
                                                (m == 0 ? 2./3. : (std::abs(m) == 1 ? 23./48. : (std::abs(m) == 2 ? 1./6. : 1./48.))) *
                                                (q == 0 ? 2./3. : (std::abs(q) == 1 ? 23./48. : (std::abs(q) == 2 ? 1./6. : 1./48.)));
                                    }
                                    value /= static_cast<T>(2.);
                                    sum_value += value;

                                    int jdx = idx * 2 + l;
                                    int other1_jdx = other1_idx * 2 + m;
                                    int other2_jdx = other2_idx * 2 + q;
                                    if (jdx >= 0 && jdx < nidx && 
                                        other1_jdx >=0 && other1_jdx <= other1_nidx && 
                                        other2_jdx >=0 && other2_jdx <= other2_nidx) {
                                        int tJdx = is_x_dir ? (jdx + ni * other1_jdx + ni*(nj+1) * other2_jdx) :
                                                (is_y_dir ? (x_nE + other2_jdx + (ni+1) * jdx + (ni+1)*nj * other1_jdx) :
                                                            ((x_nE+y_nE) + other1_jdx + (ni+1) * other2_jdx + (ni+1)*(nj+1) * jdx));
                                        tJdx_idx_toadd.push_back(tJdx);
                                        tJdx_value_toadd.push_back(value);
                                        localTripletList_p.emplace_back(tJdx, tIdx, value);
                                    }
                                }
                        for (int temp_idx = 0; temp_idx < tJdx_idx_toadd.size(); temp_idx++) {
                            localTripletList_r.emplace_back(tIdx, tJdx_idx_toadd[temp_idx], static_cast<T>(2.) * tJdx_value_toadd[temp_idx] / sum_value);
                        }
                    }
                });
                std::vector<Eigen::Triplet<T> > tripletList_r;
                std::vector<Eigen::Triplet<T> > tripletList_p;
                HELPER::mergeLocalThreadVectors(tripletList_r, parallelTripletList_r);
                HELPER::mergeLocalThreadVectors(tripletList_p, parallelTripletList_p);
                R.resize(nnE, nE);
                P.resize(nE, nnE);
                R.setFromTriplets(tripletList_r.begin(), tripletList_r.end());
                P.setFromTriplets(tripletList_p.begin(), tripletList_p.end());
            }
        } else if (gmg_flag == GMG::FACES) {
            size_t nF = (ni+1) * nj * nk +
                        ni * (nj+1) * nk +
                        ni * nj * (nk+1);
            size_t nnF = (nni+1) * nnj * nnk +
                         nni * (nnj+1) * nnk +
                         nni * nnj * (nnk+1);

            int x_nF = (ni+1) * nj * nk;
            int y_nF = ni * (nj+1) * nk;
            int x_nnF = (nni+1) * nnj * nnk;
            int y_nnF = nni * (nnj+1) * nnk;
            {
                tbb::enumerable_thread_specific<std::vector<Eigen::Triplet<T> > > parallelTripletList_r;
                tbb::enumerable_thread_specific<std::vector<Eigen::Triplet<T> > > parallelTripletList_p;
                tbb::parallel_for(tbb::blocked_range<int>(0,nnF,1000), [&](const tbb::blocked_range<int> &range)
                {
                    auto& localTripletList_r = parallelTripletList_r.local();
                    auto& localTripletList_p = parallelTripletList_p.local();
                    for (int tIdx = range.begin(); tIdx < range.end(); ++tIdx) {
                        
                        bool is_x_dir = tIdx < x_nnF;
                        bool is_y_dir = !is_x_dir && tIdx < (x_nnF + y_nnF);
                        // bool is_z_dir = !is_x_dir && !is_y_dir;
                        int comp_slice = is_x_dir ? (nni+1) * nnj : (is_y_dir ? nni * (nnj+1) : nni * nnj);
                        int comp_n = is_x_dir ? (nni+1) : nni;
                        int comp_tIdx = is_x_dir ? tIdx : (is_y_dir ? (tIdx - x_nnF) : (tIdx - (x_nnF + y_nnF)));
                        int k = comp_tIdx/comp_slice;
                        int j = (comp_tIdx%comp_slice)/comp_n;
                        int i = comp_tIdx%comp_n;
                        int idx = is_x_dir ? i : (is_y_dir ? j : k);
                        int other1_idx = is_x_dir ? j : (is_y_dir ? k : i);
                        int other2_idx = is_x_dir ? k : (is_y_dir ? i : j);
                        int nidx = is_x_dir ? ni : (is_y_dir ? nj : nk);
                        int other1_nidx = is_x_dir ? nj : (is_y_dir ? nk : ni);
                        int other2_nidx = is_x_dir ? nk : (is_y_dir ? ni : nj);
                        T sum_value = 0.;
                        std::vector<int> tJdx_idx_toadd;
                        std::vector<T> tJdx_value_toadd;
                        for (int l = -bs_p; l <= bs_p; l++)
                            for (int m = -bs_p+1; m <= bs_p; m++)
                                for (int q = -bs_p+1; q <= bs_p; q++) {
                                    T value = 0;
                                    if (bs_p == 1) {
                                        value = l == 0 ? 1.0 : 0.5;
                                    } else if (bs_p == 2) {
                                        int m_adjusted = 2*m-1;
                                        int q_adjusted = 2*q-1;
                                        value = (l == 0 ? 3./4. : (std::abs(l) == 1 ? 0.5 : 1./8.)) *
                                                (std::abs(m_adjusted) == 1 ? 0.75 : 0.25) *
                                                (std::abs(q_adjusted) == 1 ? 0.75 : 0.25);
                                    } else if (bs_p == 3) {
                                        int m_adjusted = 2*m-1;
                                        int q_adjusted = 2*q-1;
                                        value = (l == 0 ? 2./3. : (std::abs(l) == 1 ? 23./48. : (std::abs(l) == 2 ? 1./6. : 1./48.))) *
                                                (std::abs(m_adjusted) == 1 ? 11./16. : (std::abs(m_adjusted) == 3 ? 9./32. : 1./32.)) *
                                                (std::abs(q_adjusted) == 1 ? 11./16. : (std::abs(q_adjusted) == 3 ? 9./32. : 1./32.));
                                    }
                                    value /= static_cast<T>(4.);
                                    sum_value += value;

                                    int jdx = idx * 2 + l;
                                    int other1_jdx = other1_idx * 2 + m;
                                    int other2_jdx = other2_idx * 2 + q;
                                    if (jdx >= 0 && jdx <= nidx && 
                                        other1_jdx >=0 && other1_jdx < other1_nidx && 
                                        other2_jdx >=0 && other2_jdx < other2_nidx) {
                                        int tJdx = is_x_dir ? (jdx + (ni+1) * other1_jdx + (ni+1)*nj * other2_jdx) :
                                                (is_y_dir ? (x_nF + other2_jdx + ni * jdx + ni*(nj+1) * other1_jdx) :
                                                            ((x_nF+y_nF) + other1_jdx + ni * other2_jdx + ni*nj * jdx));
                                        tJdx_idx_toadd.push_back(tJdx);
                                        tJdx_value_toadd.push_back(value);
                                        localTripletList_p.emplace_back(tJdx, tIdx, value);
                                    }
                                }
                        for (int temp_idx = 0; temp_idx < tJdx_idx_toadd.size(); temp_idx++) {
                            localTripletList_r.emplace_back(tIdx, tJdx_idx_toadd[temp_idx], static_cast<T>(4.) * tJdx_value_toadd[temp_idx] / sum_value);
                        }
                    }
                });
                std::vector<Eigen::Triplet<T> > tripletList_r;
                std::vector<Eigen::Triplet<T> > tripletList_p;
                HELPER::mergeLocalThreadVectors(tripletList_r, parallelTripletList_r);
                HELPER::mergeLocalThreadVectors(tripletList_p, parallelTripletList_p);
                R.resize(nnF, nF);
                P.resize(nF, nnF);
                R.setFromTriplets(tripletList_r.begin(), tripletList_r.end());
                P.setFromTriplets(tripletList_p.begin(), tripletList_p.end());
            }
        } else {
            {
                tbb::enumerable_thread_specific<std::vector<Eigen::Triplet<T> > > parallelTripletList_r;
                tbb::enumerable_thread_specific<std::vector<Eigen::Triplet<T> > > parallelTripletList_p;
                tbb::parallel_for(tbb::blocked_range<int>(0,nni * nnj * nnk,1000), [&](const tbb::blocked_range<int> &range)
                {
                    auto& localTripletList_r = parallelTripletList_r.local();
                    auto& localTripletList_p = parallelTripletList_p.local();
                    for (int tIdx = range.begin(); tIdx < range.end(); ++tIdx) {
                        int k = tIdx / (nni * nnj);
                        int j = (tIdx % (nni * nnj)) / nni;
                        int i = tIdx % nni;
                        size_t index = (k * nnj + j) * nni + i;
                        for (int kk = 0; kk <= 1; kk++)
                            for (int jj = 0; jj <= 1; jj++)
                                for (int ii = 0; ii <= 1; ii++) {
                                    int iii = i * 2 + ii;
                                    int jjj = j * 2 + jj;
                                    int kkk = k * 2 + kk;
                                    if (iii < ni && jjj < nj && kkk < nk) {
                                        unsigned int index2 = (kkk * nj + jjj) * ni + iii;
                                        localTripletList_r.emplace_back(index,index2, static_cast<T>(1.0));//0.125));
                                        localTripletList_p.emplace_back(index2,index, static_cast<T>(0.125));//1.0));
                                    }
                                }
                    }
                });
                std::vector<Eigen::Triplet<T> > tripletList_r;
                std::vector<Eigen::Triplet<T> > tripletList_p;
                HELPER::mergeLocalThreadVectors(tripletList_r, parallelTripletList_r);
                HELPER::mergeLocalThreadVectors(tripletList_p, parallelTripletList_p);
                R.resize(nni * nnj * nnk, ni * nj * nk);
                P.resize(ni * nj * nk, nni * nnj * nnk);
                R.setFromTriplets(tripletList_r.begin(), tripletList_r.end());
                P.setFromTriplets(tripletList_p.begin(), tripletList_p.end());
            }
        }
    }

    void generateRP2D(const Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> &A,
                      Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> &R,
                      Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> &P,
                      int ni, int nj, int bs_p, GMG gmg_flag) {
        int nni = ceil((float) ni / 2.0);
        int nnj = ceil((float) nj / 2.0);
        
        if (gmg_flag == GMG::NODES) {
            std::vector<Eigen::Triplet<T> > tripletList_R;
            std::vector<Eigen::Triplet<T> > tripletList_P;

            for (int j = 0; j < nnj; j++)
                for (int i = 0; i < nni; i++) {
                    unsigned int index = j * nni + i;
                    for (int jj = 0; jj <= 1; jj++)
                        for (int ii = 0; ii <= 1; ii++) {
                            int iii = i * 2 + ii;
                            int jjj = j * 2 + jj;
                            if (iii < ni && jjj < nj) {
                                unsigned int index2 = jjj * ni + iii;
                                tripletList_R.emplace_back(index,index2, static_cast<T>(0.25));
                                tripletList_P.emplace_back(index2,index, static_cast<T>(1.0));
                            }
                        }
                }

            R.resize(nni*nnj,ni*nj);
            P.resize(ni*nj,nni*nnj);
            R.setFromTriplets(tripletList_R.begin(), tripletList_R.end());
            P.setFromTriplets(tripletList_P.begin(), tripletList_P.end());
        } else {
            size_t nF = (ni+1) * nj +
                        ni * (nj+1);
            size_t nnF = (nni+1) * nnj +
                         nni * (nnj+1);

            int x_nF = (ni+1) * nj;
            int x_nnF = (nni+1) * nnj;
            {
                tbb::enumerable_thread_specific<std::vector<Eigen::Triplet<T> > > parallelTripletList_r;
                tbb::enumerable_thread_specific<std::vector<Eigen::Triplet<T> > > parallelTripletList_p;
                tbb::parallel_for(tbb::blocked_range<int>(0,nnF,1000), [&](const tbb::blocked_range<int> &range)
                {
                    auto& localTripletList_r = parallelTripletList_r.local();
                    auto& localTripletList_p = parallelTripletList_p.local();
                    for (int tIdx = range.begin(); tIdx < range.end(); ++tIdx) {
                        bool is_x_dir = tIdx < x_nnF;
                        int comp_n = is_x_dir ? (nni+1) : nni;
                        int comp_tIdx = is_x_dir ? tIdx : tIdx - x_nnF;
                        int j = comp_tIdx/comp_n;
                        int i = comp_tIdx%comp_n;
                        int idx = is_x_dir ? i : j;
                        int other_idx = is_x_dir ? j : i;
                        int nidx = is_x_dir ? ni : nj;
                        int other_nidx = is_x_dir ? nj : ni;
                        T sum_value = 0.;
                        std::vector<int> tJdx_idx_toadd;
                        std::vector<T> tJdx_value_toadd;
                        for (int l = -bs_p; l <= bs_p; l++)
                            for (int m = -bs_p+1; m <= bs_p; m++) {
                                T value = 0;
                                if (bs_p == 1) {
                                    value = l == 0 ? 1.0 : 0.5;
                                } else if (bs_p == 2) {
                                    int m_adjusted = 2*m-1;
                                    value = (l == 0 ? 3./4. : (std::abs(l) == 1 ? 0.5 : 1./8.)) *
                                            (std::abs(m_adjusted) == 1 ? 0.75 : 0.25);
                                } else if (bs_p == 3) {
                                    int m_adjusted = 2*m-1;
                                    value = (l == 0 ? 2./3. : (std::abs(l) == 1 ? 23./48. : (std::abs(l) == 2 ? 1./6. : 1./48.))) *
                                            (std::abs(m_adjusted) == 1 ? 11./16. : (std::abs(m_adjusted) == 3 ? 9./32. : 1./32.));
                                }
                                value /= static_cast<T>(2.);
                                sum_value += value;

                                int jdx = idx * 2 + l;
                                int other_jdx = other_idx * 2 + m;
                                if (jdx >= 0 && jdx <= nidx && 
                                    other_jdx >=0 && other_jdx < other_nidx) {
                                    int tJdx = is_x_dir ? (jdx + (ni+1) * other_jdx) :
                                                          (x_nF + other_jdx + ni * jdx);
                                    tJdx_idx_toadd.push_back(tJdx);
                                    tJdx_value_toadd.push_back(value);
                                    localTripletList_p.emplace_back(tJdx, tIdx, value);
                                }
                            }
                        for (int temp_idx = 0; temp_idx < tJdx_idx_toadd.size(); temp_idx++) {
                            localTripletList_r.emplace_back(tIdx, tJdx_idx_toadd[temp_idx], static_cast<T>(2.) * tJdx_value_toadd[temp_idx] / sum_value);
                        }
                    }
                });
                std::vector<Eigen::Triplet<T> > tripletList_r;
                std::vector<Eigen::Triplet<T> > tripletList_p;
                HELPER::mergeLocalThreadVectors(tripletList_r, parallelTripletList_r);
                HELPER::mergeLocalThreadVectors(tripletList_p, parallelTripletList_p);
                R.resize(nnF, nF);
                P.resize(nF, nnF);
                R.setFromTriplets(tripletList_r.begin(), tripletList_r.end());
                P.setFromTriplets(tripletList_p.begin(), tripletList_p.end());
            }
        }
    }

    void generateLevelsGalerkinCoarseningRedoA_L(std::vector<Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> > &A_L,
            //given
                                          const std::vector<Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> > &R_L,
                                          const std::vector<Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> > &P_L,
                                          std::vector<T> & wmax, bool update_wmax,
                                          const int &total_level,
                                          Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> &A, float mult=0.5f) {
        A_L.resize(0);
        A_L.push_back(A);
        for (int i = 0; i < total_level - 1; i++) {
            Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> newA = R_L[i] * A_L[i] * P_L[i] * static_cast<T>(mult);
            A_L.push_back(newA);
        }
        if (update_wmax) {
            T max_wmax = 0;
            for (int i = 0; i < total_level; i++) {
                Eigen::DiagonalMatrix<T, Eigen::Dynamic> diag_mat;
                diag_mat.resize(A_L[i].rows());
                diag_mat.diagonal() = 1.f / A_L[i].diagonal().array();
                Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> eigs_mat = diag_mat * A_L[i];
                Spectra::SparseSymMatProd<T, Eigen::Lower, Eigen::RowMajor, std::ptrdiff_t> op(eigs_mat);
                Spectra::SymEigsSolver<Spectra::SparseSymMatProd<T, Eigen::Lower, Eigen::RowMajor, std::ptrdiff_t> > eigs(op, 1, 10);
                eigs.init();
                eigs.compute(Spectra::SortRule::LargestMagn, 100, 1e-2);
                Eigen::VectorX<T> evalues;
                if (eigs.info() == Spectra::CompInfo::Successful) {
                    evalues = eigs.eigenvalues();
                    T ev_max = (T)1.05 * evalues[0];
                    wmax.push_back(ev_max);
                    max_wmax = std::max(ev_max, max_wmax);
                    std::cout << "wmax: " << ev_max << " at level: " << i << std::endl;
                } else {
                    std::cout << "eigs solve failed..." << std::endl;
                }
            }
            // for (int i = 0; i < total_level; i++) {
            //     wmax[i] = max_wmax;
            // }
        }
    }

    void generateLevelsGalerkinCoarsening(std::vector<Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> > &A_L,
                                          std::vector<Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> > &R_L,
                                          std::vector<Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> > &P_L,
                                          std::vector<T> & wmax,
                                          std::vector<Vec3i> &S_L,
                                          int &total_level,
            //given
                                          Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> &A,
                                          int ni, int nj, int nk, int bs_p=1, GMG gmg_flag=GMG::NODES, float mult=0.5f) {
        std::cout << "building levels ...... " << std::endl;
        A_L.resize(0);
        R_L.resize(0);
        P_L.resize(0);
        wmax.resize(0);
        S_L.resize(0);
        total_level = 1;
        A.makeCompressed();
        A_L.push_back(A);
        S_L.push_back(Vec3i(ni, nj, nk));
        int nni = ni, nnj = nj, nnk = nk;
        unsigned int unknowns =
            gmg_flag == GMG::EDGES ? (ni * (nj+1) * (nk+1) +
                                      (ni+1) * nj * (nk+1) +
                                      (ni+1) * (nj+1) * nk)
            : (gmg_flag == GMG::FACES ? ((ni+1) * nj * nk +
                                         ni * (nj+1) * nk +
                                         ni * nj * (nk+1))
            : (ni * nj * nk));
        unsigned int lowest_res = 8;
        unsigned int unknowns_limit = gmg_flag == GMG::EDGES ? (3 * lowest_res * (lowest_res+1) * (lowest_res+1)) : (gmg_flag == GMG::FACES ? (3 * (lowest_res+1) * lowest_res * lowest_res) : (lowest_res * lowest_res * lowest_res));
        while (unknowns > unknowns_limit) {
            nni = ceil((float) nni / 2.0);
            nnj = ceil((float) nnj / 2.0);
            nnk = ceil((float) nnk / 2.0);

            S_L.push_back(Vec3i(nni, nnj, nnk));
            unknowns =
                gmg_flag == GMG::EDGES ? (nni * (nnj+1) * (nnk+1) +
                                          (nni+1) * nnj * (nnk+1) +
                                          (nni+1) * (nnj+1) * nnk)
                : (gmg_flag == GMG::FACES ? ((nni+1) * nnj * nnk +
                                             nni * (nnj+1) * nnk +
                                             nni * nnj * (nnk+1))
                : (nni * nnj * nnk));
            total_level++;
        }
        std::cout << "building levels, levels count: " << total_level << std::endl;

        for (int i = 0; i < total_level - 1; i++) {
            Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> R, P;
            generateRP(A_L[i], R, P, S_L[i].v[0], S_L[i].v[1], S_L[i].v[2], bs_p, gmg_flag);
            Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> newA = R * A_L[i] * P * static_cast<T>(mult);

            R_L.push_back(R);
            P_L.push_back(P);
            newA.makeCompressed();
            A_L.push_back(newA);
        }

        T max_wmax = 0;
        for (int i = 0; i < total_level; i++) {
            Eigen::DiagonalMatrix<T, Eigen::Dynamic> diag_mat;
            diag_mat.resize(A_L[i].rows());
            diag_mat.diagonal() = 1.f / A_L[i].diagonal().array();
            Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> eigs_mat = diag_mat * A_L[i];
            Spectra::SparseSymMatProd<T, Eigen::Lower, Eigen::RowMajor, std::ptrdiff_t> op(eigs_mat);
            Spectra::SymEigsSolver<Spectra::SparseSymMatProd<T, Eigen::Lower, Eigen::RowMajor, std::ptrdiff_t> > eigs(op, 1, 10);
            eigs.init();
            eigs.compute(Spectra::SortRule::LargestMagn, 100, 1e-2);
            Eigen::VectorX<T> evalues;
            if (eigs.info() == Spectra::CompInfo::Successful) {
                evalues = eigs.eigenvalues();
                T ev_max = (T)1.05 * evalues[0];
                wmax.push_back(ev_max);
                max_wmax = std::max(ev_max, max_wmax);
                std::cout << "wmax: " << ev_max << " at level: " << i << ", converged with num_iters: " << eigs.num_iterations() << std::endl;
            } else {
                std::cout << "eigs solve failed..." << std::endl;
            }
        }
        // for (int i = 0; i < total_level; i++) {
        //     wmax[i] = max_wmax;
        // }

        std::cout << "build levels done" << std::endl;
    }

    void generateLevelsGalerkinCoarsening2D(std::vector<Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> > &A_L,
                                            std::vector<Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> > &R_L,
                                            std::vector<Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> > &P_L,
                                            std::vector<Vec2i> &S_L,
                                            int &total_level,
            //given
                                            Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> &A,
                                            int ni, int nj, int bs_p=1, GMG gmg_flag=GMG::NODES, float mult=0.5f) {
        std::cout << "building levels ...... " << std::endl;
        A_L.resize(0);
        R_L.resize(0);
        P_L.resize(0);
        S_L.resize(0);
        total_level = 1;
        A.makeCompressed();
        A_L.push_back(A);
        S_L.push_back(Vec2i(ni, nj));
        int nni = ni, nnj = nj;
        unsigned int unknowns = gmg_flag != GMG::NODES ? ((ni+1) * nj + ni * (nj+1)) : (ni * nj);
        unsigned int lowest_res = 16;
        unsigned int unknowns_limit = gmg_flag != GMG::NODES ? (2 * (lowest_res+1) * lowest_res) : (lowest_res * lowest_res);
        while (unknowns > unknowns_limit) {
            nni = ceil((float) nni / 2.0);
            nnj = ceil((float) nnj / 2.0);

            S_L.push_back(Vec2i(nni, nnj));
            unknowns = gmg_flag != GMG::NODES ? ((nni+1) * nnj + nni * (nnj+1)) : (nni * nnj);
            total_level++;
        }
        std::cout << "building levels, levels count: " << total_level << std::endl;

        for (int i = 0; i < total_level - 1; i++) {
            Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> R, P;
            generateRP2D(A_L[i], R, P, S_L[i].v[0], S_L[i].v[1], bs_p, gmg_flag);
            Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> newA = R * A_L[i] * P * static_cast<T>(mult);

            R_L.push_back(R);
            P_L.push_back(P);
            newA.makeCompressed();
            A_L.push_back(newA);
        }

        std::cout << "build levels done" << std::endl;
    }

};


#endif
