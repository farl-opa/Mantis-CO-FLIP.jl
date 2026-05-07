#ifndef SPARSE_MATRIX_H
#define SPARSE_MATRIX_H

#include <iostream>
#include <vector>
#include "tbb/tbb.h"
#include "../utils/util.h"
#include "../utils/Eigen/Sparse"

//============================================================================
// Dynamic compressed sparse row matrix.

template<class T>
struct SparseMatrix {
    int n = 0; // dimension
    std::vector<std::vector<int> > index; // for each row, a list of all column indices (sorted)
    std::vector<std::vector<T> > value; // values corresponding to index
    ~SparseMatrix() {
        clear();
    }

    explicit SparseMatrix(int n_ = 0, int expected_nonzeros_per_row = 7)
            : n(n_), index(n_), value(n_) {
        for (int i = 0; i < n; ++i) {
            index[i].reserve(expected_nonzeros_per_row);
            value[i].reserve(expected_nonzeros_per_row);
        }
    }

    void clear(void) {
        n = 0;
        for (uint i = 0; i < index.size(); i++) {
            index[i].resize(0);
            index[i].shrink_to_fit();
        }
        for (uint i = 0; i < value.size(); i++) {
            value[i].resize(0);
            value[i].shrink_to_fit();
        }
        index.resize(0);
        index.shrink_to_fit();
        value.resize(0);
        value.shrink_to_fit();
    }

    void zero(void) {
        for (int i = 0; i < n; ++i) {
            index[i].resize(0);
            value[i].resize(0);
        }
    }

    void zero(int _n) {
        for (int i = 0; i < n; ++i) {
            index[i].reserve(_n);
            value[i].reserve(_n);
        }
    }

    void resize(int n_) {
        n = n_;
        index.resize(n);
        value.resize(n);
        // zero();
    }

    T operator()(int i, int j) const {
        for (int k = 0; k < index[i].size(); ++k) {
            if (index[i][k] == j) return value[i][k];
            else if (index[i][k] > j) return 0;
        }
        return 0;
    }

    void set_element(int i, int j, T new_value) {
        int k = 0;
        for (; k < index[i].size(); ++k) {
            if (index[i][k] == j) {
                value[i][k] = new_value;
                return;
            } else if (index[i][k] > j) {
                insert(index[i], k, j);
                insert(value[i], k, new_value);
                return;
            }
        }
        index[i].push_back(j);
        value[i].push_back(new_value);
    }

    void add_to_element(int i, int j, T increment_value) {
        int k = 0;
        for (; k < index[i].size(); ++k) {
            if (index[i][k] == j) {
                value[i][k] += increment_value;
                return;
            } else if (index[i][k] > j) {
                insert(index[i], k, j);
                insert(value[i], k, increment_value);
                return;
            }
        }
        index[i].push_back(j);
        value[i].push_back(increment_value);
    }

    // assumes indices is already sorted
    void add_sparse_row(int i, const std::vector<int> &indices, const std::vector<T> &values) {
        int j = 0, k = 0;
        while (j < indices.size() && k < index[i].size()) {
            if (index[i][k] < indices[j]) {
                ++k;
            } else if (index[i][k] > indices[j]) {
                insert(index[i], k, indices[j]);
                insert(value[i], k, values[j]);
                ++j;
            } else {
                value[i][k] += values[j];
                ++j;
                ++k;
            }
        }
        for (; j < indices.size(); ++j) {
            index[i].push_back(indices[j]);
            value[i].push_back(values[j]);
        }
    }

    void add_sparse_row(int i, const std::vector<int> &indices, T multiplier,
                        const std::vector<T> &values) {
        assert(i < n);
        int j = 0, k = 0;
        while (j < indices.size() && k < index[i].size()) {
            if (index[i][k] < indices[j]) {
                ++k;
            } else if (index[i][k] > indices[j]) {
                insert(index[i], k, indices[j]);
                insert(value[i], k, multiplier * values[j]);
                ++j;
            } else {
                value[i][k] += multiplier * values[j];
                ++j;
                ++k;
            }
        }
        for (; j < indices.size(); ++j) {
            index[i].push_back(indices[j]);
            value[i].push_back(multiplier * values[j]);
        }
    }

    // assumes matrix has symmetric structure - so the indices in row i tell us which columns to delete i from
    void symmetric_remove_row_and_column(int i) {
        for (int a = 0; a < index[i].size(); ++a) {
            int j = index[i][a];
            if (j != i) {
                for (int b = 0; b < index[j].size(); ++b) {
                    if (index[j][b] == i) {
                        erase(index[j], b);
                        erase(value[j], b);
                        break;
                    }
                }
            }
        }
        index[i].resize(0);
        value[i].resize(0);
    }

    void write_matlab(std::ostream &output, const char *variable_name) {
        output << variable_name << "=sparse([";
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < index[i].size(); ++j) {
                output << i + 1 << " ";
            }
        }
        output << "],...\n  [";
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < index[i].size(); ++j) {
                output << index[i][j] + 1 << " ";
            }
        }
        output << "],...\n  [";
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < value[i].size(); ++j) {
                output << value[i][j] << " ";
            }
        }
        output << "], " << n << ", " << n << ");" << std::endl;
    }
};


template<class T>
void multiply_sparse_matrices_with_diagonal_weighting(const SparseMatrix<T> &A, const std::vector<T> &diagD,
                                                      const SparseMatrix<T> &B, SparseMatrix<T> &C) {
    //assert(A.n==B.m);
    assert(diagD.size() == A.n);
    C.resize(A.n);
    C.zero();
    int num = A.n;
    tbb::parallel_for(tbb::blocked_range<int>(0,num,1000), [&](const tbb::blocked_range<int> &r)
    {
        for (int i = r.begin(); i < r.end(); ++i) {
        // for (int i = 0; i < A.n; ++i) {
            for (int p = 0; p < A.index[i].size(); ++p) {
                int k = A.index[i][p];
                C.add_sparse_row(i, B.index[k], A.value[i][p] * diagD[k], B.value[k]);
            }
        }
    });
}


template<class T>
void multiply_sparse_matrices(const SparseMatrix<T> &A, const SparseMatrix<T> &B, SparseMatrix<T> &C) {
    //assert(A.n==B.m);
    C.resize(A.n);
    C.zero();
    int num = A.n;
    tbb::parallel_for(tbb::blocked_range<int>(0,num,1000), [&](const tbb::blocked_range<int> &r)
    {
        for (int i = r.begin(); i < r.end(); ++i) {
        // for (int i = 0; i < A.n; ++i) {
            for (int p = 0; p < A.index[i].size(); ++p) {
                int k = A.index[i][p];
                C.add_sparse_row(i, B.index[k], A.value[i][p], B.value[k]);
            }
        }
    });
}


template<class T>
void transpose_matrix(const SparseMatrix<T> &A, SparseMatrix<T> &B) {
    for (int i = 0; i < A.n; ++i) {
        for (int p = 0; p < A.index[i].size(); ++p) {
            int k = A.index[i][p];
            B.set_element(k, i, A.value[i][p]);
        }
    }
}


template<class T>
void rowsum(SparseMatrix<T> &A) {
    for (int i = 0; i < A.n; ++i) {
        T rowsum = 0;
        for (int p = 0; p < A.index[i].size(); ++p) {
            rowsum += A.value[i][p];
        }
        A.index[i].resize(0);
        A.value[i].resize(0);
        A.index[i].push_back(i);
        A.value[i].push_back(rowsum);
    }
}


template<class T>
void rowsum_and_invert(SparseMatrix<T> &A) {
    for (int i = 0; i < A.n; ++i) {
        T rowsum = 0;
        for (int p = 0; p < A.index[i].size(); ++p) {
            rowsum += A.value[i][p];
        }
        A.index[i].resize(0);
        A.value[i].resize(0);
        A.index[i].push_back(i);
        A.value[i].push_back(T(1) / rowsum);
    }
}


typedef SparseMatrix<float> SparseMatrixf;
typedef SparseMatrix<double> SparseMatrixd;

// perform result=matrix*x
template<class T>
void multiply(const SparseMatrix<T> &matrix, const std::vector<T> &x, std::vector<T> &result, bool do_resize=true) {
    //needs parallel
    //assert(matrix.n==x.size());
    if (do_resize)
        result.resize(matrix.n);
    //for(int i=0; i<matrix.n; ++i){
    //   result[i]=0;
    //   for(int j=0; j<matrix.index[i].size(); ++j){
    //      result[i]+=matrix.value[i][j]*x[matrix.index[i][j]];
    //   }
    //}
    int num = matrix.n;
    tbb::parallel_for(tbb::blocked_range<int>(0,num,1000), [&](const tbb::blocked_range<int> &r)
    {
        for (int i = r.begin(); i < r.end(); ++i) {
            result[i] = 0;
            for (int j = 0; j < matrix.index[i].size(); ++j) {
                result[i] += matrix.value[i][j] * x[matrix.index[i][j]];
            }
        }
    });
}

// perform result=result-matrix*x
template<class T>
void multiply_and_subtract(const SparseMatrix<T> &matrix, const std::vector<T> &x, std::vector<T> &result) {
    //needs parallel
    assert(matrix.n == x.size());
    result.resize(matrix.n);
    //for(int i=0; i<matrix.n; ++i){
    // for(int j=0; j<matrix.index[i].size(); ++j){
    //  result[i]-=matrix.value[i][j]*x[matrix.index[i][j]];
    // }
    //}
    int num = matrix.n;
    tbb::parallel_for(tbb::blocked_range<int>(0,num,1000), [&](const tbb::blocked_range<int> &r)
    {
        for (int i = r.begin(); i < r.end(); ++i) {
            result[i] = 0;
            for (int j = 0; j < matrix.index[i].size(); ++j) {
                result[i] -= matrix.value[i][j] * x[matrix.index[i][j]];
            }
        }
    });
}

//============================================================================
// Fixed version of SparseMatrix. This is not a good structure for dynamically
// modifying the matrix, but can be significantly faster for matrix-vector
// multiplies due to better data locality.

template<class T>
struct FixedSparseMatrix {
    int n; // dimension
    std::vector<T> value; // nonzero values row by row
    std::vector<int> colindex; // corresponding column indices
    std::vector<int> rowstart; // where each row starts in value and colindex (and last entry is one past the end, the number of nonzeros)

    explicit FixedSparseMatrix(int n_ = 0)
            : n(n_), value(0), colindex(0), rowstart(n_ + 1) {}

    ~FixedSparseMatrix() {
        clear();
    }

    void clear(void) {
        n = 0;
        value.resize(0);
        value.shrink_to_fit();
        colindex.resize(0);
        colindex.shrink_to_fit();
        rowstart.resize(0);
        rowstart.shrink_to_fit();
    }

    void resize(int n_) {
        n = n_;
        rowstart.resize(n + 1);
    }

    void construct_from_matrix(const SparseMatrix<T> &matrix) {
        resize(matrix.n);
        rowstart[0] = 0;
        for (int i = 0; i < n; ++i) {
            rowstart[i + 1] = rowstart[i] + matrix.index[i].size();
        }
        value.resize(rowstart[n]);
        colindex.resize(rowstart[n]);
        int j = 0;
        for (int i = 0; i < n; ++i) {
            for (int k = 0; k < matrix.index[i].size(); ++k) {
                value[j] = matrix.value[i][k];
                colindex[j] = matrix.index[i][k];
                ++j;
            }
        }
    }

    void construct_from_eigen_matrix(Eigen::SparseMatrix<T, Eigen::RowMajor, std::ptrdiff_t> &matrix) {
        resize(matrix.rows());
        matrix.makeCompressed();
        rowstart.assign(matrix.outerIndexPtr(), matrix.outerIndexPtr() + (n+1));
        value.resize(rowstart[n]);
        colindex.resize(rowstart[n]);
        value.assign(matrix.valuePtr(), matrix.valuePtr() + rowstart[n]);
        colindex.assign(matrix.innerIndexPtr(), matrix.innerIndexPtr() + rowstart[n]);
    }

    void write_matlab(std::ostream &output, const char *variable_name) {
        output << variable_name << "=sparse([";
        for (int i = 0; i < n; ++i) {
            for (int j = rowstart[i]; j < rowstart[i + 1]; ++j) {
                output << i + 1 << " ";
            }
        }
        output << "],...\n  [";
        for (int i = 0; i < n; ++i) {
            for (int j = rowstart[i]; j < rowstart[i + 1]; ++j) {
                output << colindex[j] + 1 << " ";
            }
        }
        output << "],...\n  [";
        for (int i = 0; i < n; ++i) {
            for (int j = rowstart[i]; j < rowstart[i + 1]; ++j) {
                output << value[j] << " ";
            }
        }
        output << "], " << n << ", " << n << ");" << std::endl;
    }
};

typedef FixedSparseMatrix<float> FixedSparseMatrixf;
typedef FixedSparseMatrix<double> FixedSparseMatrixd;

// perform result=matrix*x
template<class T>
void multiply(const FixedSparseMatrix<T> &matrix, const std::vector<T> &x, std::vector<T> &result, bool do_resize=true) {
    //needs parallel
    //assert(matrix.n==x.size());
    if (do_resize)
        result.resize(matrix.n);
    //for(int i=0; i<matrix.n; ++i){
    //   result[i]=0;
    //   for(int j=matrix.rowstart[i]; j<matrix.rowstart[i+1]; ++j){
    //      result[i]+=matrix.value[j]*x[matrix.colindex[j]];
    //   }
    //}

    int num = matrix.n;
    tbb::parallel_for(tbb::blocked_range<int>(0,num,1000), [&](const tbb::blocked_range<int> &r)
    {
        for (int i = r.begin(); i < r.end(); ++i) {
            result[i] = 0;
            for (int j = matrix.rowstart[i]; j < matrix.rowstart[i + 1]; ++j) {
                result[i] += matrix.value[j] * x[matrix.colindex[j]];
            }
        }

    });


}


// perform C=scale*A*B
template<class T>
void multiplyMat(const FixedSparseMatrix<T> &A, const FixedSparseMatrix<T> &B, FixedSparseMatrix<T> &C, T scale) {
    //needs parallel
    C.clear();
    SparseMatrix<T> c;
    c.resize(A.n);
    c.zero();
    tbb::parallel_for(tbb::blocked_range<int>(0,c.n,1000), [&](const tbb::blocked_range<int> &r)
    {
        for (int i = r.begin(); i < r.end(); ++i) {
            for (size_t j = A.rowstart[i]; j < A.rowstart[i + 1]; ++j) {
                size_t k = A.colindex[j];
                T A_ik = A.value[j];
                for (size_t kkk = B.rowstart[k]; kkk < B.rowstart[k + 1]; ++kkk) {
                    c.add_to_element(i, B.colindex[kkk], scale * A_ik * B.value[kkk]);
                }

            }
        }
    });

    //for(int i=0; i<matrix.n; ++i){
    //   result[i]=0;
    //   for(int j=matrix.rowstart[i]; j<matrix.rowstart[i+1]; ++j){
    //      result[i]+=matrix.value[j]*x[matrix.colindex[j]];
    //   }
    //}

    C.construct_from_matrix(c);
    c.clear();


}


// perform A = coef*B'
template<class T>
void transposeMat(const FixedSparseMatrix<T> &B, FixedSparseMatrix<T> &A, T coef) {
    A.clear();

    //needs parallel
    SparseMatrix<T> a;
    int max_col = 0;
    //find how many columns does B have
    for (int i = 0; i < B.n; ++i) {
        for (int j = B.rowstart[i]; j < B.rowstart[i + 1]; ++j) {
            if (B.colindex[j] > max_col) {
                max_col = B.colindex[j];
            }

        }

    }
    a.resize(max_col + 1);
    a.zero();

    for (int i = 0; i < B.n; ++i) {
        for (int j = B.rowstart[i]; j < B.rowstart[i + 1]; ++j) {
            T val = B.value[j];
            int ii = B.colindex[j];
            int jj = i;
            a.set_element(ii, jj, coef * val);
        }
    }



    //for(int i=0; i<matrix.n; ++i){
    //   result[i]=0;
    //   for(int j=matrix.rowstart[i]; j<matrix.rowstart[i+1]; ++j){
    //      result[i]+=matrix.value[j]*x[matrix.colindex[j]];
    //   }
    //}

    A.construct_from_matrix(a);
    a.clear();


}

// perform result=result-matrix*x
template<class T>
void multiply_and_subtract(const FixedSparseMatrix<T> &matrix, const std::vector<T> &x, std::vector<T> &result) {
    //needs parallel
    assert(matrix.n == x.size());
    //result.resize(matrix.n);
    //for(int i=0; i<matrix.n; ++i){
    //   for(int j=matrix.rowstart[i]; j<matrix.rowstart[i+1]; ++j){
    //      result[i]-=matrix.value[j]*x[matrix.colindex[j]];
    //   }
    //}
    int num = matrix.n;
    tbb::parallel_for(tbb::blocked_range<int>(0,num,1000), [&](const tbb::blocked_range<int> &r)
    {
        for (int i = r.begin(); i < r.end(); ++i) {
            //result[i]=0;
            for (int j = matrix.rowstart[i]; j < matrix.rowstart[i + 1]; ++j) {
                result[i] -= matrix.value[j] * x[matrix.colindex[j]];
            }
        }
    });
}

#endif
