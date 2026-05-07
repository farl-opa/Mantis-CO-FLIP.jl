#ifndef BLAS_WRAPPER_H
#define BLAS_WRAPPER_H

// Simple placeholder code for BLAS calls - replace with calls to a real BLAS library

#include <vector>
#include "tbb/tbb.h"

namespace BLAS{

// dot products ==============================================================
template<class T>
inline double mean(const std::vector<T> &x)
{
    T sum = 0;
    for (size_t i = 0;i < x.size();i++)
    {
        sum += x[i];
    }
    return sum / static_cast<T>(x.size());
}
template<class T>
inline double abs_mean(const std::vector<T> &x)
{
    T sum = 0;
    for (size_t i = 0;i < x.size();i++)
    {
        sum += fabs(x[i]);
    }
    return sum / static_cast<T>(x.size());
}
template<class T>
inline double squared_norm(const std::vector<T> &x)
{
    T sum = 0;
    for (size_t i = 0;i < x.size();i++)
    {
        sum += x[i]*x[i];
    }
    return sum;
}
template<class T>
inline void subtractConst(std::vector<T> &x, const T a)
{
    size_t num = x.size();
    tbb::parallel_for((size_t)0, (size_t)num, (size_t)1, [&](size_t i) {
        x[i] -= a;
    });
}

template<class T>
inline double dot(const std::vector<T> &x, const std::vector<T> &y)
{ 
   //return cblas_ddot((int)x.size(), &x[0], 1, &y[0], 1); 

//   auto values = std::vector<double>(x.size());
//    tbb::parallel_for( tbb::blocked_range<int>(0,values.size()),
//                       [&](tbb::blocked_range<int> r)
//                       {
//                           for (int i=r.begin(); i<r.end(); ++i)
//                           {
//                               values[i] = x[i]*y[i];
//                           }
//                       });
//    auto total = tbb::parallel_reduce(
//            tbb::blocked_range<int>(0,values.size()),
//            0.0,
//            [&](tbb::blocked_range<int> r, double running_total)
//            {
//                for (int i=r.begin(); i<r.end(); ++i)
//                {
//                    running_total += values[i];
//                }
//
//                return running_total;
//            }, std::plus<double>() );
//    return (double)total;
   double sum = 0;
   for(unsigned int i = 0; i < x.size(); ++i)
      sum += x[i]*y[i];
   return sum;
}

// inf-norm (maximum absolute value: index of max returned) ==================

template<class T>
inline int index_abs_max(const std::vector<T> &x)
{ 
   //return cblas_idamax((int)x.size(), &x[0], 1); 
   int maxind = 0;
   T maxvalue = 0;
   for(unsigned int i = 0; i < x.size(); ++i) {
      if(std::abs(x[i]) > maxvalue) {
         maxvalue = std::abs(x[i]);
         maxind = i;
      }
   }
   return maxind;
}

// inf-norm (maximum absolute value) =========================================
// technically not part of BLAS, but useful

template<class T>
inline double abs_max(const std::vector<T> &x)
{
//    auto values = std::vector<double>(x.size());
//    tbb::parallel_for( tbb::blocked_range<int>(0,values.size()),
//                       [&](tbb::blocked_range<int> r)
//                       {
//                           for (int i=r.begin(); i<r.end(); ++i)
//                           {
//                               values[i] = std::fabs(x[i]);
//                           }
//                       });
//    auto total = tbb::parallel_reduce(
//            tbb::blocked_range<int>(0,values.size()),
//            0.0,
//            [&](tbb::blocked_range<int> r, double running_total)
//            {
//                for (int i=r.begin(); i<r.end(); ++i)
//                {
//                    running_total = std::max(running_total, values[i]);
//                }
//
//                return running_total;
//            }, [&](double x, double y)->double{
//                return std::max(x,y);
//            }
//            );
//    return (double)total;
    return std::abs(x[index_abs_max(x)]);
}

// saxpy (y=alpha*x+y) =======================================================

template<class T>
inline void add_scaled(T alpha, const std::vector<T> &x, std::vector<T> &y)
{ 
   //cblas_daxpy((int)x.size(), alpha, &x[0], 1, &y[0], 1); 
   //for(unsigned int i = 0; i < x.size(); ++i)
   //   y[i] += alpha*x[i];
	assert(y.size() == x.size());
	int num = x.size();
	tbb::parallel_for((int)0, (int)num, (int)1, [&] (int i){

		y[i] += alpha*x[i];
	});
}

}
#endif
