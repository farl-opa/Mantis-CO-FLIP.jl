#!/bin/bash

cd ../docs

julia -e 'import Pkg; Pkg.activate("."); Pkg.develop(path=dirname(pwd())); Pkg.instantiate();'

julia --color=yes --project make.jl

python -m http.server --bind localhost
