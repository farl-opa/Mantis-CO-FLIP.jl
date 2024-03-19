# How to build the MANTIS docs.
The main file structure of the the docs has now been created, so you 
only have to build the documentation. Fortunately, the heavy lifting 
will be done by `Julia` and `Python`.

To create the docs, follow these steps:
- Navigate to the `docs/` directory. If you open a terminal in the 
  `Mantis.jl` repo, this is a simple matter of executing 
  ```
  cd docs
  ```
  (or the equivalent on your system) in your terminal.
- Next, run 
  ```
  julia --color=yes --project make.jl
  ``` 
  in this directory (you can leave the `--color=yes` option out if you 
  don't want the printed output to be coloured). This will update (or 
  generate if it doesn't exist yet) a `build/`-directory in the 
  `docs/`-directory[^NoteOnGITtingBuildFolder]. All `.html` files are 
  generated here. *Make sure that you save all changes before running 
  this command or you won't see the changes!*
- Then you need to create a (local) webserver to view the HTML docs, for 
  which there are a few options, see the [Documenter Docs](https://documenter.juliadocs.org/stable/man/guide/). 
  Since I have `Python` installed, I use the python option. Run
  ```
  python -m http.server --bind localhost
  ``` 
  (after, in my case, activating my conda environment by executing 
  `conda activate`) in the `docs/`-directory. 
- In my case, this will result in VScode giving me the message: `Your 
  application running on port 8000 is available. See all forwarded ports`
  with the options `Open in Browser` and `Preview in Editor`. Click the 
  `Open in Browser`-option to see the html pages. You have to click the 
  `build/`-link when the browser opens. 
- Enjoy the Mantis docs!
- When done, you can kill the local webserver by using `Ctrl+C`.

[^NoteOnGITtingBuildFolder]:
    The `docs/build/`-folder should **never** be gitted. (See also the 
    warning in the [Documenter Docs](https://documenter.juliadocs.org/stable/man/guide/)). 
    The `.gitignore`-file is already setup to prevent this.
