__precompile__()

module AI


export GPSE
export feedbackrnn
export learnrnn
export fixedrnn



using Distributions
using LinearAlgebra: I

using DataFrames

using MLJ
using Flux
using Flux: onehot, onehotbatch, onecold
# using MLJFlux
# using MLJModelInterface
using Zygote: pullback
# ChainRulesCore.@non_differentiable foreach(f, ::Tuple{})
# Zygote.refresh()

using BSON: @save, load
import BSON.@load
include("ai/BSONextension.jl")
export @load, @save

using ..MathUtils: @horizontalvector, @asarray, gpsample




struct GPSE
    σ::Float64
    ρ::Float64
    l::Float64
end

GPSE() = GPSE(rand(Gamma(2)),rand(Gamma(1)),rand(Gamma(1)))
  
(m::GPSE)(t) = gpsample(t,m.σ,m.ρ,m.l)





# types, subroutines
include("ai/MLJcompatible.jl")
include("ai/modeltypes.jl")
include("ai/regularizations.jl")











function makesequence(X::Matrix{T}, sequencelength::Int) where {T}
    # flux style, sequences length vector of (features,samples)
    Xstr = Matrix{T}[]
    for sx in 1:sequencelength
        push!(Xstr, X[:,sx:end-sequencelength+sx])
        # push!(Xstr, [ X[:,sqx] for sqx = sx:sx+sequencelength ] )
    end
    return Xstr
end













function feedbackrnn(Xtr, ytr, Xte, yte, sequencelength::Int)
    

    
    Xstr = makesequence(Xtr, sequencelength)
    ystr = ytr[:,sequencelength:end]
    Xste = makesequence(Xte, sequencelength)
    yste = yte[:,sequencelength:end]

    yrstr = [ onecold(ytr[:,u:end+u-sequencelength],[0,1]) for u in 1:sequencelength]
    yrste = [ onecold(yte[:,u:end+u-sequencelength],[0,1]) for u in 1:sequencelength]

    # @info "Xstr ystr Xste yste" Xstr ystr Xste yste







    # @info "values" Xstr y
    @info "data sizes" size(Xstr) size(Xstr[1]) size(ystr) size(ystr[1]) size(Xste) size(Xste[1]) size(yste) size(yste[1])
    # @info "batch" size(Xstr[1]) size(Xstr[2])

    
    # inputs will be the stimuli (visual and audio), reward (calculated from decision)
    # input will also contain an externa true values for reward calculation, which will not be used by the network
    ninput = size(Xtr,1)
    nhidden = 24
    noutput = size(ytr,1)

    rnnmodel = Chain(   hiddenlayer=RNN(ninput, nhidden),  outputlayer=Dense(nhidden, noutput, σ),
                        decisionfunction=softmax )
    parameters = Flux.params(rnnmodel)

    
    @info "RNN model structure" rnnmodel



    # rnn should look for the past couple of trials
    # visual, audio, water/decision, context
    function cycle(xs,model)
        ntrials = size(xs[1],2)
        Flux.reset!(model)
        r = zeros(Float32,2,ntrials)                     # neutral reward for the first sequence point
        d = similar(r)
        for (i,x) in enumerate(xs)                        # go through the sequence-timepoints
            g = x[5:6,:]                                  # establish the goal decision based on context, onehot
            d = model( vcat( x[1:4,:],r  ) )            # step the model forward, get network output in this sequence-timepoint, onehot
            r = Float32.(onehotbatch(onecold(d,[0,1]).==onecold(g,[0,1]),[0,1]))    # calculate reward based on decision, that is used in the next sequence, onehot
            # r = Float32.(  onehotbatch(rand([0,1],ntrials), [0,1])  )           # crosscheck with random rewards as baseline
        end
        return d
    end


    

    # @info "start Win" rnnmodel[:hiddenlayer].cell.Wi[22:24,3:6]         # check for reward weights changes



    function sequencelastloss(xs,y)
        d = cycle(xs, rnnmodel)
        Flux.logitcrossentropy(d, y)         # use the final timepoint deicision variable, onehot+softmax
    end

    # d = cycle(Xstr,rnnmodel)


    # @info "parameters" typeof(parameters)
    # dp = gradient(()->sequencelastloss(Xstr,ystr), parameters)
    # @info "gradient dp" dp.params dp.grads
    # return 0,0,0

    # tx, ty = (Xstr, ystr)
    evalcb = () -> @show sequencelastloss(Xstr, ystr)
    for epoch = 1:1000
        Flux.train!(sequencelastloss, parameters, [(Xstr, ystr)], ADAM(1e-2))#, cb = Flux.throttle(evalcb, 5))
    end

    # @info "end Win" rnnmodel[:hiddenlayer].cell.Wi[22:24,3:6]

    Dtr = [ onecold(cycle(Xstr[1:t],rnnmodel),[0,1]) for t ∈ 1:sequencelength ]
    Ltr = [ sequencelastloss(Xstr[1:t],ystr) for t ∈ 1:sequencelength ]
    @info "D" size(Dtr) size(Dtr[1]) size(Ltr)
    
    Dte = [ onecold(cycle(Xste[1:t],rnnmodel),[0,1]) for t ∈ 1:sequencelength ]
    Lte = [ sequencelastloss(Xste[1:t],yste) for t ∈ 1:sequencelength ]
    @info "D̂" size(Dte) size(Dte[1]) size(Lte)

    @info "extra" size(Ltr)
    # return onecold(ystr,[0,1]), D, L, onecold(yste,[0,1]), D̂, L̂
    return yrstr, Dtr, Ltr, yrste, Dte, Lte


end

















function freezeweights!(model,regularization,excitatory,inhibitory)
    # input weights are one-to-one mapping
    if :fixedinput in regularization # explicit stimulus input
        model[1].cell.Wi[excitatory,excitatory] = I(length(excitatory))
        model[1].cell.Wi[inhibitory,excitatory] .= 1.        # overall start from 1s
        model[1].cell.Wi[inhibitory[1],excitatory[1:n_excitatory÷2]] = ones(n_inhibitory)
        model[1].cell.Wi[inhibitory[2],excitatory[1:n_excitatory÷2]] = zeros(n_inhibitory)
        model[1].cell.Wi[inhibitory[1],excitatory[1+n_excitatory÷2:n_excitatory]] = zeros(n_excitatory÷2)
        model[1].cell.Wi[inhibitory[2],excitatory[1+n_excitatory÷2:n_excitatory]] = ones(n_excitatory÷2)
    end
    # not specified stimulus input
    if :hiddeninput in regularization
        # input weights for dedicated pairs are cancelled
        model[1].cell.Wi[excitatory,:] .= 0.
        model[1].cell.Wi[inhibitory,:] .= 0.
    end
    # recurrent weights containing mutual inhibiting excitatory-inhibitory pairs
    if :ei in regularization
        for (ix,i) in enumerate(inhibitory)
            model[1].cell.Wh[i,i] = 0.      # inhibitory self stop
            model[1].cell.Wh[excitatory[2*ix-1:2*ix],i] .= 0.      # inhibitory self excitatory stop
            model[1].cell.Wh[excitatory,excitatory] .= I(length(excitatory))             # excitatory other stop (self keep)
            model[1].cell.Wh[i,5-ix*2:5-ix*2+1] .= 0.             # excitatory other inhibitory stop, manual, won't work for non 4-2 cell numbers
        end
    end
end

function boundei!(model,regularization,excitatory,inhibitory)
    # clamp hidden layer explicitely excitatory and inhibitory neurons to their allowed range
    if :ei in regularization
        model[1].cell.Wh[:,excitatory] = clamp.(model[1].cell.Wh[:,excitatory],0,1)
        model[1].cell.Wh[:,inhibitory] = clamp.(model[1].cell.Wh[:,inhibitory],-1,0)
    end
    # clamp inhibitory activity so that it wouldn't become exctitory by negative^2
    if :ei in regularization
        model[1].state[inhibitory] = clamp.(model[1].state[inhibitory],0,5)
    end
end










function learnrnn(Xs, ys, decisionpoints::Vector{Int}, rewardprofile::Vector{Float32}, machineparameters;
               train=false, persist=false, resume=false,
               filename="model")
    # regularization (:full,:fixedoutput,:stimulus/:hidden)
    # snapshots: model saves to continue from
    snapshots = machineparameters[:snapshots]
    # history can be generated and returned from saved model snapshots, like loss, activities, weights...

    ninput = size(Xs[1],1)
    ntimecourse = size(Xs)[1]
    ntrials = size(Xs[1],2)
    noutput = size(ys[1],1)
    nhidden = machineparameters[:nhidden]



    nepochs = snapshots[end]
    ndigitsnepochs = machineparameters[:ndigitsnepochs] # length("$nepochs") # for snapshots epoch numbering padding length

    firstepoch = 1                # if resume, this will be changed to snapshots[1]+1, see below
    lastepoch = snapshots[end]    # learning epoch range last epoch

    # tune the rate at which activities and weights are displayed during training
    npoints = 100
    epochskiprate = Int64(ceil(nepochs÷npoints,digits=-1))
    if nepochs<=100 epochskiprate = 1 end

    # learning rate
    eta = machineparameters[:learningrate]    # ADAM

    # regularization
    regularization = Symbol.(machineparameters[:regularization])
    alpha = machineparameters[:α]
    beta = machineparameters[:β]
    gamma = machineparameters[:γ]
    delta = machineparameters[:δ]



    addcontext = ninput==8
    # @info "direct context input: $addcontext"

    # activation function aliases
    activation = [tanh,relu][Int(:nonnegativerelu in regularization)+1]
    activation = [activation,σ][Int(:nonnegativesigmoid in regularization)+1]



    # rnn sequence cycler helper function
    function cycle(xs,model,rewardprofile;addcontext=false,record=false)
        # cycle through the model, and record activities
        decisionpoints = decisionpoints
        ntimecourse = length(xs)
        ntrials = size(xs[1],2)
    
        r = zeros(Float32,2,ntrials)            # neutral reward for the first trial sequence part
        d = similar(r)
        dp = similar(d)                               # store decision
        hs = 0                     # accumulates the activity for each timepoint and each hidden units without mutating arrays
    
        # activity record arrays
        if record
            H = zeros(Float32,ntimecourse,nhidden,ntrials)          # timecourse of hidden states for each trial
            O = zeros(Float32,ntimecourse,2,ntrials)                 # timecourse of output layer
            D = zeros(Int8,ntimecourse,2,ntrials)                 # timecourse of decision label
            R = zeros(Float32,ntimecourse,2,ntrials)                 # timecourse of reward
        end
        
        for (i,x) in enumerate(xs)                        # go through the sequence-timepoints
            g = x[5:6,:]                                  # establish the target decision based on context, onehot
    
            # apply reward profile, so that only after deicision reward is available, until next stimulus start
            reward = r.*rewardprofile[i] # reward at sequence point i, r coming from last decision
    
            # step the model forward, get network output in this sequence-timepoint, onehot
            if addcontext    # check if context was used as direct input
                d = model( vcat( x[1:4,:],reward,x[7:8,:]  ) )
            else
                d = model( vcat( x[1:4,:],reward  ) )
            end
            if i==decisionpoints[end] dp = copy(d) end
            if i in decisionpoints   # each decisionpoint along the history trials yields the subsequent reward
                # calculate subsequent reward based on decision, that is used in the next items in the sequence, onehot
                # # old: r = (  onehotbatch( onecold(ddecision,[0,1]).==onecold(g,[0,1]),[0,1] )  )
                d1 = d[1,:] .> d[2,:]       # convert probabilities to binary decision
                r1 = d1 .== g[1,:]          # compare to target, works for onehot and onecold encoding
                r2 = 1 .- r1                # convert to onehot
                r = hcat(r1,r2)'
            end
            # r = onehotbatch(rand(Float32, [0,1],ntrials), [0,1])           # crosscheck with random rewards as baseline
            hs = vcat(hs,model[1].state)
            # register timestep
            if record
                H[i,:,:] = model[1].state
                O[i,:,:] = model[2].weight * model[1].state .+ model[2].bias       # store decision without softmax -> y = Wh, d = softmax(y) 
                D[i,1,:] = convert.(Int8, d[1,:] .> d[2,:])            # decision label
                D[i,2,:] = 1 .- D[i,1,:]            # decision label one hot second class
                R[i,:,:] = reward
            end
        end
        if record
            return H,O,D,R,dp,hs
        else
            return dp,hs
        end
    end
    
    

    # loss function
    function sequencepointloss(xs,ys)
        dp,hs = cycle(xs, eirnnmodel, rewardprofile)
        yp = ys[decisionpoints[end]]
        compare = Flux.logitcrossentropy(dp, yp)
        loss = compare
        if :weightregularization in regularization
            loss += regularize(weightstoregularize, α)
        end
        if :sparse in regularization
            loss += sparsify(hs, β)
        end
        if :sparse in regularization
            loss += energyminimization(hs, β)
        end
        if :nonnegative in regularization
            loss += nonnegative(hs, γ)
        end
        if :contextsymmetric in regularization
            loss += symmetrize(dp, yp, δ)
        end
        return loss
    end








    # either load the model and just return it,
    #        resume training from a snapshot,
    # or     start from scratch to train

    if train





        if resume        # train from a snapshot (has to be the first in snapshots array)
            
            snap = lpad(snapshots[1],ndigitsnepochs,"0")
            @load(filename*"-e"*snap*".bson", @__MODULE__, eirnnmodel, l )
            # @load(filename*"-e"*snap*"-history.bson", @__MODULE__, Hs,Os,Ds,Rs,Wi,Wh,Wo,L,df )
            
            firstepoch = snapshots[1]+1

            # # preallocate history extension to new last snapshot:
            # plusepochs = (snapshots[end]-snapshots[1])÷epochskiprate

            # Hs = vcat(Hs,zeros(eltype(Hs),plusepochs,ntimecourse,nhidden,ntrials))
            # Os = vcat(Os,zeros(eltype(Os),plusepochs,ntimecourse,2,ntrials))
            # Ds = vcat(Ds,zeros(eltype(Ds),plusepochs,ntimecourse,2,ntrials))
            # Rs = vcat(Rs,zeros(eltype(Rs),plusepochs,ntimecourse,2,ntrials))

            # Wi = vcat(Wi,zeros(eltype(Wi),plusepochs,nhidden,ninput))
            # Wh = vcat(Wh,zeros(eltype(Wh),plusepochs,nhidden,nhidden))
            # Wb = vcat(Wb,zeros(eltype(Wb),plusepochs,noutput,nhidden))

            if length(l)>snapshots[1] l = l[1:snapshots[1]] end      # cut any further loss function values after the starting epoch, if any
            l = vcat(l,zeros(snapshots[end]-snapshots[1]))       # this holds the loss over learning epochs


        else         # train from first epoch


            eirnnmodel = Chain(   hiddenlayer=RNN(ninput, nhidden, activation),  outputlayer=Dense(nhidden, noutput, σ),
                                decisionfunction=softmax )

            # improve
            # 1) stoch help: random reset h
            # 2) teach context partially
            # 3) context competes with decision and also too quickly forget: - help maintain context: gates, exctitatory loops
            #                                                                - potentially longer training improves it

            # diagnostics        
            # dp = cycle(Xs, eirnnmodel)
            # H,D,R,dp = cycle(Xs, eirnnmodel; record=true)
            # loss = sequencepointloss(Xs,ys)
            # @info "loss" loss typeof(loss)
            # return
        

            # record learning
            l = zeros(nepochs)       # this holds the loss over learning epochs


            # Hs = zeros(Float32,nepochs÷epochskiprate,ntimecourse,nhidden,ntrials)          # timecourse of hidden states for each trial
            # Os = zeros(Float32,nepochs÷epochskiprate,ntimecourse,2,ntrials)                 # timecourse of decision
            # Ds = zeros(Int8,nepochs÷epochskiprate,ntimecourse,2,ntrials)                 # timecourse of decision
            # Rs = zeros(Float32,nepochs÷epochskiprate,ntimecourse,2,ntrials)                 # timecourse of reward

            # Wi = zeros(Float32,nepochs÷epochskiprate,nhidden,ninput)    # timecourses of weights to record
            # Wh = zeros(Float32,nepochs÷epochskiprate,nhidden,nhidden)
            # Wo = zeros(Float32,nepochs÷epochskiprate,noutput,nhidden)

            # df = zeros(ntrials)
            
        end


        # additional initialization
        parameters = Flux.params(eirnnmodel)

        weightstoregularize = (eirnnmodel[1].cell.Wi, eirnnmodel[1].cell.Wh, eirnnmodel[2].weight, eirnnmodel[2].bias)

        # hyperparameters when used with :hyperanneal
        hyperdecayrates = machineparameters[:hyperanneal]
        α = copy(alpha)    # alpha, regularization: weights L2
        β = copy(beta)     # beta: activity L1/L2  (sparse/energy constraint)
        γ = copy(gamma)     # gamma: nonnegative activity L1
        δ = copy(delta)     # delta: symmetry difference L1
        



        @info "model structure" eirnnmodel parameters[1][1:4,:] parameters[2][1:4,1:4] parameters[3]'

        

        # gradient learning loop
        for epoch = firstepoch:lastepoch

            # handle hyperparameter scheduler
            α,β,γ,δ = [α,β,γ,δ] .* hyperdecayrates

            # inline train
            Flux.reset!(eirnnmodel)        # reset the model at each start of a sequence (trial with history)
            l[epoch], back = pullback(() -> sequencepointloss(Xs, ys), parameters)     # get loss and pullback

            gradients = back(one(l[epoch]))                                          # calculate gradients from pullback
            Flux.update!(ADAM(eta), parameters, gradients)                             # update weights based on gradients


            # log display
            if mod(epoch,epochskiprate)==0
                print("$epoch> $(l[epoch])")
                # print("$epoch> $(L[epoch,1]-sum(L[epoch,3:end]))")
                # if :weightregularization in regularization print(", $(L[epoch,3])") end
                # if :sparse in regularization print(", $(L[epoch,4])") end
                # if :energy in regularization print(", $(L[epoch,4])") end
                # if :nonnegative in regularization print(", $(L[epoch,5])") end
                # if :contextsymmetric in regularization print(", $(L[epoch,6])") end
                println()
            end

            # # record activity
            # # if epoch<(nepochs-maxepochs) e = 1 else e = epoch-(nepochs-maxepochs) end
            # # e = max(1,epoch-(nepochs-maxepochs))
            # e = min(epoch÷epochskiprate+1,nepochs÷epochskiprate)
            # Flux.reset!(eirnnmodel)       # reset hidden states
            # Hs[e,:,:,:], Os[e,:,:,:], Ds[e,:,:,:], Rs[e,:,:,:], df = cycle(Xs, eirnnmodel, rewardprofile; record=true)
            # Wi[e,:,:] = eirnnmodel[1].cell.Wi
            # Wh[e,:,:] = eirnnmodel[1].cell.Wh
            # Wo[e,:,:] = eirnnmodel[2].weight
        
            # L[epoch,2] = L[epoch,1] - sum(L[epoch,3:end])    # save the decision loss and the regularization types separetely

            if persist && epoch in snapshots   # save each snapshot model and loss history
                snap = lpad(epoch,ndigitsnepochs,"0")
                @save(filename*"-e"*snap*".bson", eirnnmodel, l )
                # if epoch in history     # save history points (usually just at the last snapshot)
                #     @save(filename*"-e"*snap*"-history.bson", Hs,Os,Ds,Rs,Wi,Wh,Wo,L,df )
                # end
            end
        end

        

    else # test and record history for snapshots

        # load model at each snapshot and generate history of loss, activities, weights
        # calculate by loading models at snapshots, and return full history at snapshots

        nsnapshots = length(snapshots)
        Hs = zeros(Float32,nsnapshots,ntimecourse,nhidden,ntrials)          # timecourse of hidden states for each trial
        Os = zeros(Float32,nsnapshots,ntimecourse,2,ntrials)                 # timecourse of decision
        Ds = zeros(Int8,nsnapshots,ntimecourse,2,ntrials)                 # timecourse of decision
        Rs = zeros(Float32,nsnapshots,ntimecourse,2,ntrials)                 # timecourse of reward

        Wi = zeros(Float32,nsnapshots,nhidden,ninput)    # timecourses of weights to record
        Wh = zeros(Float32,nsnapshots,nhidden,nhidden)
        Wo = zeros(Float32,nsnapshots,noutput,nhidden)
    
        l = 0
        L = zeros(nsnapshots, 6)
        


        for (sx,s) in enumerate(snapshots)

            snap = lpad(s,ndigitsnepochs,"0")
            @load(filename*"-e"*snap*".bson", @__MODULE__, eirnnmodel, l )
            # @load(filename*"-e"*snap*"-history.bson", @__MODULE__, Hs,Os,Ds,Rs,Wi,Wh,Wo,L,df )

            weightstoregularize = (eirnnmodel[1].cell.Wi, eirnnmodel[1].cell.Wh, eirnnmodel[2].weight, eirnnmodel[2].bias)

            Flux.reset!(eirnnmodel)        # reset the unit activities at each snapshot

            Hs[sx,:,:,:], Os[sx,:,:,:], Ds[sx,:,:,:], Rs[sx,:,:,:], dp, hs = cycle(Xs, eirnnmodel, rewardprofile; record=true)
            Wi[sx,:,:] = eirnnmodel[1].cell.Wi
            Wh[sx,:,:] = eirnnmodel[1].cell.Wh
            Wo[sx,:,:] = eirnnmodel[2].weight

            L[sx,1] = l[s]    # overall loss at snapshot
            if :weightregularization in regularization L[sx,3] = regularize(weightstoregularize, alpha) end
            # if :energy in regularization || :sparse in regularization ||
            #         :contextsymmetric in regularization || :nonnegative in regularization
            #     dp,hs = cycle(Xs, eirnnmodel, rewardprofile)
            # end
            if :sparse in regularization L[sx,4] = sparsify(hs, beta) end
            if :energy in regularization L[sx,4] = energyminimization(hs, beta) end
            if :nonnegative in regularization L[sx,5] = nonnegative(hs, gamma) end
            if :contextsymmetric in regularization
                yp = ys[decisionpoints[end]]
                L[sx,6] = symmetrize(dp,yp, delta)
            end

            L[sx,2] = L[sx,1] - sum(L[sx,3:end])

        end

        return Hs,Os,Ds,Rs,Wi,Wh,Wo,L,l
    end        




    
end



















function fixedrnn(Xs, ys, decisionpoints, rewardprofile, Wi, Wh, Wo)
    nhidden = size(Wh,1)
    @info "started"
    ninput = size(Xs[1],1)
    ntimecourse = size(Xs)[1]
    ntrials = size(Xs[1],2)
    noutput = size(ys[1],1)

    # recurrentlayer = Flux.Recur(   Flux.RNNCell( NNlib.σ, Wi, Wh, zeros(Float32,nhidden), zeros(Float32,nhidden) )   )
    recurrentlayer = RNN( NNlib.σ, Wi, Wh, zeros(Float32,nhidden,1), randn(Float32,nhidden,1) )
    # recurrentlayer2 = RNN(ninput,nhidden)
    # @info "types" typeof(recurrentlayer) typeof(recurrentlayer2)
    # return

    outputlayer = Dense( Wo, zeros(Float32,noutput), σ)

    eirnnmodel = Chain( hiddenlayer=recurrentlayer,outputlayer=outputlayer,
                        decisionfunction=softmax )
    
    # fixed rnn, no need for trainable parameters
    # parameters = Flux.params(eirnnmodel)

    @info "model" eirnnmodel


    function cycle(xs,model;record=false)
        # cycle through the model, and record activities
        decisionpoints = decisionpoints
        ntimecourse = length(xs)
        ntrials = size(xs[1],2)

        r = zeros(Float32,2,ntrials)            # neutral reward for the first trial sequence part
        d = similar(r)
        dp = similar(d)                               # store decision


        # activity record arrays
        if record
            H = zeros(Float32,ntimecourse,nhidden,ntrials)          # timecourse of hidden states for each trial
            D = zeros(Float32,ntimecourse,2,ntrials)                 # timecourse of decision
            R = zeros(Float32,ntimecourse,2,ntrials)                 # timecourse of reward
        end
        
        for (i,x) in enumerate(xs)                        # go through the sequence-timepoints
            g = x[5:6,:]                                  # establish the goal decision based on context, onehot

            # apply reward profile, so that only after deicision reward is available, until next stimulus start
            reward = r.*rewardprofile[i] # reward at sequence point i, r coming from last decision

            # step the model forward, get network output in this sequence-timepoint, onehot
            d = model( vcat( x[1:4,:],reward  ) )

            if i==decisionpoints[end] dp = copy(d) end
            if i in decisionpoints   # each decisionpoint along the history trials yields the subsequent reward
                # calculate subsequent reward based on decision, that is used in the next items in the sequence, onehot
                r = (  onehotbatch( onecold(d,[0,1]).==onecold(g,[0,1]),[0,1] )  )    
            end

            # register timestep
            if record
                H[i,:,:] = model[1].state
                D[i,:,:] = model[2].weight * model[1].state .+ model[2].bias       # store decision without softmax -> y = Wh, d = softmax(y) 
                R[i,:,:] = reward
            end
        end
        if record
            return H,D,R,dp
        else
            return dp
        end
    end
        
    Flux.reset!(eirnnmodel)        # reset the model at each start of a sequence (trial with history)
    H,D,R,dp = cycle(Xs, eirnnmodel, record=true)


    return H,D,R,dp

end






end