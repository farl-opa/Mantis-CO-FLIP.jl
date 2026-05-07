#ifndef BSPLINES_H
#define BSPLINES_H
#include <array>
#include <vector>

inline double B10(const double& x) {
    return 1.0 - x;
}
inline double B11(const double& x){
    return x;
}
inline double D10(const double& x, int bdy_version) {
    return (bdy_version == 0 ? 1.0 : 2.0) * (1.0 - x);
}
inline double D11(const double& x){
    return x;
}
inline double D10x(const double& x, int bdy_version) {
    return (bdy_version == 0 ? 1.0 : 2.0) * -1.0;
}
inline double D11x(const double& x){
    return 1.0;
}
//---------Quadratic bsplines------------
inline double B20(const double& x, int bdy_version) {
    double s = 1.0 - x;
    return (bdy_version == 0 ? 0.5 : 1.0) * (s*s);
}
inline double B21(const double& x, int bdy_version) {
    return bdy_version == 0 ? x*(1.0 - x) + 0.5 : x*(2.0 - 1.5*x);
}
inline double B22(const double& x){
    return 0.5 * (x*x);
}
inline double D20(const double& x, int bdy_version) {
    double s = 1.0 - x;
    return (bdy_version == 0 ? 0.5 : (bdy_version == 1 ? 3.0/4.0 : 3.0)) * (s*s);
}
inline double D21(const double& x, int bdy_version) {
    return (bdy_version == 0 || bdy_version == 1) ? x*(1.0 - x) + 0.5 : 3.0/2.0 * x*(2.0 - 1.5*x);
}
inline double D22(const double& x){
    return 0.5 * (x*x);
}
inline double B20x(const double& x, int bdy_version) {
    return bdy_version == 0 ? -(1.0 - x) : -2.0*(1.0 - x);
}
inline double B21x(const double& x, int bdy_version) {
    return bdy_version == 0 ? (1.0 - x) - x : 2.0*(1.0 - x) - x;
}
inline double B22x(const double& x){
    return x;
}
inline double D20x(const double& x, int bdy_version) {
    return (bdy_version == 0 ? -1.0 : (bdy_version == 1 ? -3.0/2.0 : -6.0)) * (1.0 - x);
}
inline double D21x(const double& x, int bdy_version) {
    return (bdy_version == 0 || bdy_version == 1) ? -2.0*x + 1.0 : 3.0/2.0 * (-3.0*x + 2.0);
}
inline double D22x(const double& x){
    return x;
}
inline double B20xx(const double& x) {
    return 1.0;
}
inline double B21xx(const double& x, int bdy_version) {
    return -2.0;
}
inline double B22xx(const double& x){
    return 1.0;
}
//----------Cubic bsplines--------------------
inline double B30(const double& x, int bdy_version) {
    double s2 = (1.-x)*(1.-x);
    double s3 = s2*(1.-x);
    return (bdy_version == 0 ? 1.0 / 6.0 : (bdy_version == 1 ? 1.0 / 4.0 : 1.0)) * s3;
}
inline double B31(const double& x, int bdy_version) {
    double x2 = x*x;
    double x3 = x2*x;
    return bdy_version == 0 ? 1.0 / 6.0 * (3.0*x3 - 6.0*x2 + 4.0) : (bdy_version == 1 ? 1.0 / 12.0 * (7*x3 - 15.0*x2 + 3.0*x + 7.0) : x * (7.0 / 4.0 * x2 - 9.0 / 2.0 * x + 3.0));
}
inline double B32(const double& x, int bdy_version){
    double x2 = x*x;
    double x3 = x2*x;
    return (bdy_version == 0 || bdy_version == 1) ? 1.0 / 6.0 * (-3.0*x3 + 3.0*x2 + 3.0*x + 1.0) : x2 * (3.0/2.0 - 11.0/12.0 * x);
}
inline double B33(const double& x){
    double x2 = x*x;
    double x3 = x2*x;
    return 1.0/6.0 * x3;
}
inline double B30x(const double& x, int bdy_version) {
    double s2 = (1.-x)*(1.-x);
    return (bdy_version == 0 ? -1.0 / 2.0 : (bdy_version == 1 ? -3.0 / 4.0 : -3.0)) * s2;
}
inline double B31x(const double& x, int bdy_version) {
    double x2 = x*x;
    return bdy_version == 0 ? 1.0 / 2.0 * (3.0*x2 - 4.0*x) : (bdy_version == 1 ? 1.0 / 4.0 * (7.0*x2 - 10.0*x + 1.0) : 21.0 / 4.0 * x2 - 9.0 * x + 3.0);
}
inline double B32x(const double& x, int bdy_version){
    double x2 = x*x;
    return (bdy_version == 0 || bdy_version == 1) ? 1.0 / 2.0 * (-3.0*x2 + 2.0*x + 1.0) : 3.0*x - 11.0/4.0*x2;
}
inline double B33x(const double& x){
    return 1.0/2.0 * (x*x);
}
inline double B30xx(const double& x, int bdy_version) {
    return (bdy_version == 0 ? 1.0 : (bdy_version == 1 ? 3.0 / 2.0 : 6.0)) * (1.0 - x);
}
inline double B31xx(const double& x, int bdy_version) {
    return bdy_version == 0 ? 3.0*x - 2.0 : (bdy_version == 1 ? 1.0 / 2.0 * (7.0*x - 5.0) : 21.0 / 2.0 * x - 9.0);
}
inline double B32xx(const double& x, int bdy_version){
    return (bdy_version == 0 || bdy_version == 1) ? -3.0*x + 1.0 : 3.0 - 11.0/2.0*x;
}
inline double B33xx(const double& x){
    return x;
}
//----------- cubic
std::array<double, 7> B3_5 = {1./5040., 1./42., 397./1680., 151./315., 397./1680., 1./42., 1./5040.}; //sum=1
std::array<double, 7> B3_4 = {1./3360., 239./10080., 397./1680., 151./315., 397./1680., 1./42., 1./5040.}; //sum=1
std::array<double, 7> B3_3 = {1./840., 29./840., 283./1260., 151./315., 397./1680., 1./42., 1./5040.}; //sum=1
std::array<double, 7> B3_2 = {0., 31./1680., 5./32., 183./560., 283./1260., 239./10080., 1./5040.}; //sum=3/4
std::array<double, 7> B3_1 = {0., 0., 7./80., 31./140., 5./32., 29./840., 1./3360.}; //sum=2/4
std::array<double, 7> B3_0 = {0., 0., 0., 1./7., 7./80., 31./1680., 1./840.}; //sum=1/4
std::vector<std::array<double, 7> > B3s = {B3_0, B3_1, B3_2, B3_3, B3_4, B3_5};
//------------ quad
std::array<double, 7> B2_3 = {0., 1./120., 13./60., 11./20., 13./60., 1./120., 0.}; //sum=1
std::array<double, 7> B2_2 = {0., 1./60., 5./24., 11./20., 13./60., 1./120., 0.}; //sum=1
std::array<double, 7> B2_1 = {0., 0., 7./60., 1./3., 5./24., 1./120., 0.}; //sum=2/3
std::array<double, 7> B2_0 = {0., 0., 0., 1./5., 7./60., 1./60., 0.}; //sum=1/3
std::vector<std::array<double, 7> > B2s = {B2_0, B2_1, B2_2, B2_3};
std::array<double, 7> D2_4 = B2_3;
std::array<double, 7> D2_3 = {0., 1./80., 13./60., 11./20., 13./60., 1./120., 0.}; //sum=1.004
std::array<double, 7> D2_2 = {0., 1./20., 5./16., 11./20., 13./60., 1./120., 0.}; //sum=1.1375
std::array<double, 7> D2_1 = {0., 0., 21./40., 3./4., 5./16., 1./80., 0.}; //sum=1.6
std::array<double, 7> D2_0 = {0., 0., 0., 9./5., 21./40., 1./20., 0.}; //sum=2.375
// std::array<double, 7> D2_3 = B2_3;
// std::array<double, 7> D2_2 = B2_3;
// std::array<double, 7> D2_1 = {0., 0., 21./80., 3./4., 15./32., 3./160., 0.}; //sum=3/2
// std::array<double, 7> D2_0 = {0., 0., 0., 9./5., 21./20., 3./20., 0.}; //sum=3
std::vector<std::array<double, 7> > D2s = {D2_0, D2_1, D2_2, D2_3, D2_4};
//------------ linear
std::array<double, 7> B1_1 = {0., 0., 1./6., 2./3., 1./6., 0., 0.}; //sum=1
std::array<double, 7> B1_0 = {0., 0., 0., 1./3., 1./6., 0., 0.}; //sum=1/2
std::vector<std::array<double, 7> > B1s = {B1_0, B1_1};
std::array<double, 7> D1_2 = B1_1;
std::array<double, 7> D1_1 = {0., 0., 1./3., 2./3., 1./6., 0., 0.}; //sum=1.167
std::array<double, 7> D1_0 = {0., 0., 0., 4./3., 1./3., 0., 0.}; //sum=1.667
std::vector<std::array<double, 7> > D1s = {D1_0, D1_1, D1_2};
std::vector<std::array<double, 7> > D0s = {std::array<double, 7>{0., 0., 0., 1.0, 0., 0., 0.}};
//////////////////////////////////
//----------- cubic
std::array<double, 6> BD3_5 = {1./720., 19./240., 151./360., 151./360., 19./240., 1./720.}; //sum=1
std::array<double, 6> BD3_4 = {1./480., 19./240., 151./360., 151./360., 19./240., 1./720.}; //sum=1 1/1440
std::array<double, 6> BD3_3 = {1./120., 7./60., 151./360., 151./360., 19./240., 1./720.}; //sum=1 2/45
std::array<double, 6> BD3_2 = {0., 5./48., 61./160., 61./160., 113./1440., 1./720.}; //sum=1363/1440
std::array<double, 6> BD3_1 = {0., 0., 31./80., 31./80., 53./480., 1./480.}; //sum=71/80
std::array<double, 6> BD3_0 = {0., 0., 0., 1./2., 9./80., 1./120.}; //sum=149/240
std::vector<std::array<double, 6> > BD3s = {BD3_0, BD3_1, BD3_2, BD3_3, BD3_4, BD3_5};
//------------ quad
std::array<double, 6> BD2_3 = {0., 1./24., 11./24., 11./24., 1./24., 0.}; //sum=1
std::array<double, 6> BD2_2 = {0., 1./12., 11./24., 11./24., 1./24., 0.}; //sum=1 1/24
std::array<double, 6> BD2_1 = {0., 0., 5./12, 5./12., 1./24., 0.}; //sum=21/24
std::array<double, 6> BD2_0 = {0., 0., 0., 1./2., 1./12., 0.}; //sum=14/24
std::vector<std::array<double, 6> > BD2s = {BD2_0, BD2_1, BD2_2, BD2_3};
//------------ linear
std::array<double, 6> BD1_1 = {0., 0., 1./2., 1./2., 0., 0.}; //sum=1
std::array<double, 6> BD1_0 = {0., 0., 0., 1./2., 0., 0.}; //sum=1/2
std::vector<std::array<double, 6> > BD1s = {BD1_0, BD1_1};
//----------- cubic
std::array<double, 6> DB3_4 = BD3_5; //sum=1
std::array<double, 6> DB3_3 = {1./480., 113./1440., 151./360., 151./360., 151./360., 19./240.}; //sum=
std::array<double, 6> DB3_2 = {1./120., 53./480., 61./160., 151./360., 19./240., 1./720.}; //sum=
std::array<double, 6> DB3_1 = {0., 9./80., 31./80., 61./160., 7./60., 1./480.}; //sum=
std::array<double, 6> DB3_0 = {0., 0., 1./2., 31./80, 5./48., 1./120.}; //sum=
std::vector<std::array<double, 6> > DB3s = {DB3_0, DB3_1, DB3_2, DB3_3, DB3_4};
//------------ quad
std::array<double, 6> DB2_2 = BD2_3; //sum=1
std::array<double, 6> DB2_1 = {0., 1./12., 5./12, 11./24., 1./24., 0.}; //sum=
std::array<double, 6> DB2_0 = {0., 0., 1./2., 5./12, 1./12., 0.}; //sum=
std::vector<std::array<double, 6> > DB2s = {DB2_0, DB2_1, DB2_2};
//------------ linear
std::array<double, 6> DB1_0 = BD1_1; //sum=1
std::vector<std::array<double, 6> > DB1s = {DB1_0};
//////////////////////////////////
//----------- cubic
std::array<double, 4> singleB3_3 = {1./24., 11./24., 11./24., 1./24.}; //sum=1
std::array<double, 4> singleB3_2 = {0., 13./48., 7./16., 1./24.}; //sum=3/4
std::array<double, 4> singleB3_1 = {0., 0., 7./16., 1./16.}; //sum=2/4
std::array<double, 4> singleB3_0 = {0., 0., 0., 1./4.}; //sum=1/4
std::vector<std::array<double, 4> > singleB3s = {singleB3_0, singleB3_1, singleB3_2, singleB3_3};
//------------ quad
std::array<double, 4> singleB2_2 = {0., 1./6., 2./3., 1./6.}; //sum=1
std::array<double, 4> singleB2_1 = {0., 0., 1./2., 1./6.}; //sum=2/3
std::array<double, 4> singleB2_0 = {0., 0., 0., 1./3.}; //sum=1/3
std::vector<std::array<double, 4> > singleB2s = {singleB2_0, singleB2_1, singleB2_2};
std::array<double, 4> singleD2_2 = singleB2_2; //sum=1
std::array<double, 4> singleD2_1 = {0., 0., 3./4., 1./4.}; //sum=1
std::array<double, 4> singleD2_0 = {0., 0., 0., 1.}; //sum=1
std::vector<std::array<double, 4> > singleD2s = {singleD2_0, singleD2_1, singleD2_2};
//------------ linear
std::array<double, 4> singleB1_1 = {0., 0., 1./2., 1./2.}; //sum=1
std::array<double, 4> singleB1_0 = {0., 0., 0., 1./2.}; //sum=1/2
std::vector<std::array<double, 4> > singleB1s = {singleB1_0, singleB1_1};
std::array<double, 4> singleD1_1 = singleB1_1; //sum=1
std::array<double, 4> singleD1_0 = {0., 0., 0., 1.}; //sum=1
std::vector<std::array<double, 4> > singleD1s = {singleD1_0, singleD1_1};
std::vector<std::array<double, 4> > singleD0s = {std::array<double, 4>{0., 0., 0., 1.0}};

#endif