__precompile__()
module NeuroscienceCommon
# neuroscience common tools in a julia package with submodules
    


include("common/nwbwrap.jl")
include("common/mathutils.jl")

include("common/electrophysiology.jl")
include("common/ai.jl")
include("common/discover.jl")

include("common/figs.jl")


end
