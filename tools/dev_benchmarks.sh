#!/bin/bash

# Make sure we are in the benchmarks folder.
cd benchmarks

# Takes care of dev-ing Mantis.
julia -e 'import Pkg; Pkg.activate("."); Pkg.develop(path=dirname(pwd())); Pkg.instantiate();'
