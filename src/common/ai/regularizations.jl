# part of ../ai.jl

function regularize(valuelist::Tuple, alpha::Number)
    # L2 norm for weights
    # means, and multipliers
    # regularization = alpha * mean(vcat(map(u->.^(vec(u),2), [ eirnnmodel[1].cell.Wi,
    #                                                   eirnnmodel[1].cell.Wh, 
    #                                                   eirnnmodel[2].weight, eirnnmodel[2].bias ] )...))
    regularization = alpha * mean(vcat(map(u->.^(vec(u),2), collect(valuelist) )...))
    return regularization
end

function sparsify(hs::AbstractArray, beta::Number)
    # L1 norm for activities
    # means, and multipliers
    # this only gives energy from the current state, need to accumulate
    sparsification = beta * mean(abs.(hs))       # L1
end

function energyminimization(hs::AbstractArray, beta::Number)
    # L2 norm for activities
    # means, and multipliers
    # this only gives energy from the current state, need to accumulate
    energy = beta * mean(hs.^2)       # L2
end

function nonnegative(hs::AbstractArray, gamma::Number)
    # ensures that all activity are positive (ensures nonzero variance), L1
    # H is (n_timecourse,n_hidden,n_trials)
    nonnegativity = gamma * mean(max.(-hs,0f0))
end

function symmetrize(dp, yp, delta::Number)
    # L2 norm for difference between the two outputs
    # take loss for each context
    halfpoint = size(yp,2)÷2
    visualcontextcrossentropy = Flux.logitcrossentropy(dp[:,1:halfpoint],yp[:,1:halfpoint])
    audiocontextcrossentropy = Flux.logitcrossentropy(dp[:,halfpoint+1:end],yp[:,halfpoint+1:end])
    symmetry = delta * abs2(visualcontextcrossentropy - audiocontextcrossentropy)
end
