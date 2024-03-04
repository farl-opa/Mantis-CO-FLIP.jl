using Documenter
using Mantis

Manual = [

]

Tutorials = [
    "Documents" => [
        "Tutorials/CreatingDocs.md",
    ],
    "Running MANTIS" => [
        "Tutorials/Test.md",
    ]
]

DevelDocs = [
    "DevelDocs/main.md",
    "Documents" => [
        "DevelDocs/Documentation.md",
    ],
]


Pages = [
    "index.md",
    "Manual" => Manual,
    "Tutorials" => Tutorials,
    "Developer Documentation" => DevelDocs,
]


# We set the LaTeX engine to be the (non-default) MathJax3 engine. This 
# allows for a more straightforward specification of the latex packages 
# and has a more consistent (with surrounding text) style for inline 
# math. The default options for MathJax will do for now.
math_engine = Documenter.MathJax3()


# The modules option will raise an error when some docstrings from the 
# listed modules are not included in the docs.
makedocs(
    modules  = [Mantis.Polynomials, Mantis.Quadrature], 
    format   = Documenter.HTML(mathengine=math_engine),
    sitename = "MANTIS.jl",
    authors  = "Diogo Costa Cabanas, Joey Dekker, Deepesh Toshniwal, Artur Palha",
    pages    = Pages,
)
