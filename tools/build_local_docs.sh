#!/bin/bash

# Make sure we are in the docs folder.
cd ../docs

# Build the html pages. Make sure to instantiate the correct environment.
# Also takes care of dev-ing Mantis.
julia -e 'import Pkg; Pkg.activate("."); Pkg.develop(path=dirname(pwd())); Pkg.instantiate();'
julia --color=yes --project make.jl

# Start a local server to view the html pages. We first try julia (requires LiveServer.jl) and then python.
{
    julia -e 'import Pkg; Pkg.activate("."); Pkg.instantiate(); using LiveServer; serve(dir="build");' 2>/dev/null
} || {
echo "LiveServer.jl is not installed. Trying python..."
if command -v python; then
    python -m http.server --bind localhost
elif command -v python3; then
    python3 -m http.server --bind localhost
else
    echo "Python is not installed or not in PATH. Please install python (3.x) or add this to PATH to run the local server."
fi
}
