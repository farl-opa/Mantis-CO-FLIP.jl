"""
Tests for the univariate spline spaces. These tests are based on the 
standard properties of Bezier curves. See 
https://en.wikipedia.org/wiki/B%C3%A9zier_curve#Properties.
"""

import Mantis

using Test

B1 = Mantis.FunctionSpaces.BSplineSpace([0.0, 0.5, 1.0], 3, [-1, 1, -1])
B2 = Mantis.FunctionSpaces.BSplineSpace([0.0, 0.5, 1.0], 4, [-1, 1, -1])
GB = Mantis.FunctionSpaces.GTBSplineSpace((B1, B2), [1, -1])