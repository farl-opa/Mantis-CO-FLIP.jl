# How to create the MANTIS docs.
To create the docs, run `julia --color=yes --project make.jl` in the 
`docs/`-directory (you can leave the `--color=yes` option out if you 
don't want the printed output to be coloured). Then you need to create a 
(local) webserver to view the HTML docs, for which there are a few options. 
I use the python option; `python -m http.server --bind localhost` (after, 
in my case, activating my conda environment by executing `conda activate`). 
Run this command in the same `docs/` folder as well.