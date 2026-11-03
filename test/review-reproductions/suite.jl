using Pkg; Pkg.activate(ARGS[1]); 
using Test
cd(joinpath(ARGS[1],"test"))
include(joinpath(ARGS[1],"test","runtests.jl"))
