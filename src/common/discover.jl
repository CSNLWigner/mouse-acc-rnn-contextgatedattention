__precompile__()
module Discover



export accuracy
export mld, multiplylastdimensions
export decodevariablefromactivity, projectontoaxis
export poissonprocesssequences
export estimatepoissonpointprocess




using DataFrames

using LinearAlgebra
using Distributions
using Statistics

using MLJ
using MLJLinearModels
using Flux




using ..MathUtils
using ..AI: GPSE



function decodevariablefromactivity(activities::Array{Float64,3},variable::Union{Vector{String},Vector{Integer},Vector{Float64},Vector{Bool}}; halfwidth=0, nfolds=5)
    # needs input in a multi-trial activity array:   [trials,timestamps,neurons]
    # variable with categorical values same number of elements as trials
    # half width is the number of timestamps to use on either side of the current timestamp: number of feature = (2*halfwidth+1) * nneurons
    ntrials,ntimestamps,nneurons = size(activities)
    accuracies = zeros(ntimestamps,3)
    coefficients = zeros(ntimestamps,nneurons+1) # will hold the intercept as the last value
    decoders = Machine[]
    @load LogisticClassifier pkg=MLJLinearModels verbosity=0

    y = coerce(variable, Multiclass)
    Threads.@threads for t in 1:ntimestamps
        ts = max(1,t-halfwidth):min(t+halfwidth,ntimestamps)
        X = table(reshape(activities[:,ts,:],ntrials,length(ts)*nneurons))
        # @info "schema" schema(X)
        # @info "levels" levels(y)
        decoder = machine(LogisticClassifier(penalty=:l2), X, y)
        # @info "machine" decoder
        e = evaluate!(decoder, resampling=StratifiedCV(nfolds=nfolds), verbosity=0, operation=predict_mode,
                      measure=[Accuracy()])
        # @info "params" e.fitted_params_per_fold
        # @info "evaluation" e.measure e.measurement
        # @info "measures" e.per_fold
        accuracies[t,:] = [ mean(e.per_fold[1]), std(e.per_fold[1]), std(e.per_fold[1])/sqrt(nfolds) ]
        coefficientsperfold = @asarray [ vcat(mean(reshape(getindex.(collect(fold)[2],2),length(ts),nneurons),dims=1)[1,:], fold.intercept) for fold in e.fitted_params_per_fold ]
        coefficients[t,:] = mean(coefficientsperfold; dims=1)
        push!(decoders, decoder)
        # if t>20 break end
    end
    return decoders, accuracies, coefficients
end



"""
project X activities onto axis determined by P
X is ntrials x ntimestamps × nneurons
P is           ntimestamps × nneurons + 1 (last is intercept)
"""
function projectontoaxis(X::AbstractArray{Float64,3},Pi::AbstractArray{Float64,2})
    if size(X,3)==size(Pi,2)
        intercept = zeros(size(Pi,1))
        P = Pi
    else
        intercept = Pi[:,end]
        P = @view Pi[:,1:end-1]
    end
    A = zeros(size(X,1),size(X,2))
    Threads.@threads for tr in axes(X,1)
        for t in axes(X,2)
            A[tr,t] = P[t,:]' * X[tr,t,:] .+ intercept[t]         # / sqrt(sum(P[t,:].^2))
        end
    end
    return A
end
function projectontoaxis(X::AbstractArray{Float64,3},Pi::AbstractArray{Float64,1})
    if size(X,3)==size(Pi,1)
        intercept = 0
        P = Pi
    else
        intercept = Pi[end]
        P = @view Pi[1:end-1]
    end
    A = zeros(size(X,1),size(X,2))
    Threads.@threads for tr in axes(X,1)
        for t in axes(X,2)
            A[tr,t] = P' * X[tr,t,:] .+ intercept
        end
    end
    return A
end















end
