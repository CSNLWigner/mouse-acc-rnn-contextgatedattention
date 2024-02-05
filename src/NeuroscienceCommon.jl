__precompile__()
module NeuroscienceCommon
# neuroscience common tools in a julia package with submodules
    


include("nwbwrap.jl")
include("mathutils.jl")

include("electrophysiology.jl")
include("ai.jl")
include("discover.jl")

include("figs.jl")


end