


function createrewardprofile(Ts, sequencelength)
    repeat( vcat( ones(Float32, Ts[1][end]),
                            zeros(Float32,Ts[2][end]+2-Ts[2][begin]),
                            ones(Float32, Ts[4][end]-Ts[3][begin]) ), sequencelength )
end




function getortho(H,X,selecttrials, labeltimepoint)
    nhidden = size(H,2)
    ntimepoints = size(X,1)
    # calculate the angle between variables representations
    # ortho
    accuracies = zeros(eltype(H),ntimepoints,size(X,2),3)
    representations = zeros(ntimepoints,size(X,2),nhidden)
    ortho = zeros(ntimepoints,size(X,2),size(X,2))     # angle between representations of variables
    for xi in axes(X,2)
        d = Float64.(permutedims(H[:,:,selecttrials[xi]],[3,1,2]))      # change order to trials,timesteps,neurons for activity
        # d = standardize(ZScoreTransform, d, dims=(1,2))
        standardizearray!(d,dims=(1,2))
        # for t in axes(d,2)
        #     d[:,t,:] = standardize(ZScoreTransform, d[:,t,:], dims=1)
        # end
        l = Float64.(X[labeltimepoint,xi,selecttrials[xi]])   # and for target, to be
        # @info "l" size(l)
        # l = l[sortperm(rand(length(l)))]
        # d = [ 1*randn(256,ntimepoints,nhidden).-2; 1*randn(256,ntimepoints,nhidden).+2 ]
        # l = [ -zeros(256);ones(256) ]
        if length(unique(l))>1                      # e.g. decision may not be present before decision point
            _, a, c = decodevariablefromactivity(d,l)
        else
            a = 0.5 .* ones(ntimepoints,3)
            c = zeros(ntimepoints,nhidden+1,3)
        end
        accuracies[:,xi,:] = a
        representations[:,xi,:] = c[:,1:end-1,1]   # leave out the intercept: (end-1)
        # @info "intercepts" labelstask[xi] c[t,end] c[t,1:end-1]' a[t,1]
    end
    Threads.@threads for t in 1:ntimepoints      
        for xi in axes(X,2)
            for yi in axes(X,2)
                ortho[t,xi,yi] = angleofvectors(representations[t,xi,:],representations[t,yi,:])
            end
        end
    end
    return ortho
end



function preparedata(X, y, Ts)
    # X is a (timecourse, features, datapoints) array
    # y is a datapoints sequence for target labels

    # designate array sizes
    ntimepoints,_,ntrials = size(X)
    ntimecourse = Ts[end][end]
    

    # in this model, we want multiple trial timecourse to be the sequence:
    sequencelength = ntimepoints÷ntimecourse
    trialdecisionpoint = Ts[3][begin]
    decisionpoints = collect(trialdecisionpoint:ntimecourse:ntimecourse*(sequencelength-1)+trialdecisionpoint)
    decisionpoint = decisionpoints[end]

    
    # create reward profile, so that reward available post decision until next stimuli
    rewardprofile = createrewardprofile(Ts, sequencelength)

    # we need to convert the X and y to Flux RNN style sequence Xs ys
    Xs = [X[t,:,:] for t in axes(X,1)]
    ys = [y[t,:,:] for t in axes(y,1)]


    return Xs, ys, sequencelength, ntimepoints, ntimecourse, ntrials, decisionpoints, decisionpoint, rewardprofile
end







# general learning rnn
function modelrnn(X, y, Ts, machineparameters)
    # X is a (timecourse, features, datapoints) array
    # y is a datapoints sequence for target labels

    # seed the random number generator for reproducibility
    # seed will be the model id numerically
    Plots.Random.seed!(parse(Int,machineparameters[:modelid]))

    # maximum number of epochs to train
    nepochs = machineparameters[:snapshots][end]
    
    Xs, ys, sequencelength, ntimepoints, ntimecourse, ntrials,
         decisionpoints, decisionpoint, rewardprofile = preparedata(X, y, Ts)


    # establish coding scheme
    onehot = size(X,2)>=6
    if onehot
        codeindices = [1:2,3:4,5:6,7:8]
        context = [ repeat([1,0],1,ntrials÷2) repeat([0,1],1,ntrials÷2) ]
        code = [1,0]
    else
        codeindices = [1:1,2:2,3:3,4:4]
        context = [ repeat([1],1,ntrials÷2) repeat([-1],1,ntrials÷2) ]
        code = [1,-1]
    end


    


    # incongruency
    incongruentonly = length(unique([X[Ts[3][begin],:,k] for k in axes(X,3)]))==4


    # sequence diagnostics
    # @info "sequence"  length(Xs) size(Xs[1]) size(Xs[2]) size(Xs[end])
    # ax = plot(layout=(3,1),size=(1000,900))
    # for (a,t) in enumerate(127:129)
    #     axs = ax[a]
    #     plot!(axs,  [ Xs[n][j,a]*0.95+(2*j-1)/0.95   for n in eachindex(Xs), j in 1:size(Xs[1],1) ], ylims=(-0.05,7) )
    #     plot!(axs, ys[n])
    # end
    # display(ax)




    
    # train the model, and record neural activity

    
    nhidden = machineparameters[:nhidden]
    nfeatures = nhidden

    η = machineparameters[:learningrate]    # ADAM
    α = machineparameters[:α]
    β = machineparameters[:β]
    γ = machineparameters[:γ]
    δ = machineparameters[:δ]


    # hyperparameter decay rates
    hdr = [ 0.99, 0.99, 0.999, 1.0001]
    # hdr = [1-1e-3, 1-1e-3, 1-1e-3, 1-1e-3]
    # hdr = fill(1.005, 4)

    # sequence, hidden units, maxepochs, randomseed, and the snapshot will be automatically added
    folder = machineparameters[:modeltype]*"-s$sequencelength-h$nhidden"
    filename = machineparameters[:modeltype]*"-s$sequencelength-h$nhidden-r"*machineparameters[:modelid]
    path = joinpath(config[:rawmodelpathprefix],config[:modelspath],folder,filename)



    # train the model
    if config[:trainmodel]
        l = learnrnn( Xs, ys, decisionpoints, rewardprofile, machineparameters,
                      train=true,persist=config[:persistmodel],resume=config[:resumemodel],filename=path )
    end
    
    # test the model
    Hs,Os,Ds,Rs,Wi,Wh,Wo,L,l = learnrnn( Xs, ys, decisionpoints, rewardprofile, machineparameters,
                                       train=false, filename=path )
    
    
    # Hs,Ds,Rs are (nepochs,ntimecourse,{nhidden,2,2},ntrials)
    # view into the last learned snapshot
    H,O,D,R = Hs[end,:,:,:],Os[end,:,:,:],Ds[end,:,:,:],Rs[end,:,:,:]
    # H hidden layer, O output layer, D decision label, R reward
    # for the last learning epoch with dimensions (multitrialrnnsequence, neurons, trials)
    


    # analyse hidden layer


    # decode variables from recurrent layer activity
    #    returned neuron activities are in [timecourse,neurons,trials],
    #    while decoder requires [trials,timestamps,neurons] -> permutedims
    # also concentrate on the last trial, don't show the history trials
    # activity:
    Hp = Float64.(permutedims(H[end-ntimecourse+1:end,:,:],[3,1,2]))
    Op = Float64.(permutedims(O[end-ntimecourse+1:end,:,:],[3,1,2]))
    # labels: (decisionpoint is inside the active stimulus period: safe timecourse point)
    v = Float64.(X[decisionpoint,codeindices[1][begin],:])
    a = Float64.(X[decisionpoint,codeindices[2][begin],:])
    c = Float64.(context[1,:])
    targets = Float64.(y[decisionpoints,1,:])

    
    # decode:
    accuracies = zeros(ntimecourse, 2 + 4 + 1 + 2 +   1   + 4, 3)
    coefs = zeros(ntimecourse, size(accuracies,2), nhidden+1, 3)              # last stats
    accuraciesoutput = zeros(ntimecourse, 4, 3)
    coefsoutput = zeros(ntimecourse, size(accuraciesoutput,2), 2+1, 3)
    # general stimuli
    _,accuracies[:,1,:],coefs[:,1,:,:] = decodevariablefromactivity(Hp,v)
    _,accuracies[:,2,:],coefs[:,2,:,:] = decodevariablefromactivity(Hp,a)
    # relevant irrelevant context separation stimuli and decision
    contextmask = context[1,:].==code[1]
    _,accuracies[:,3,:],coefs[:,3,:,:] = decodevariablefromactivity(Hp[contextmask,:,:],Float64.(X[decisionpoint,codeindices[1][begin],contextmask]))
    _,accuracies[:,4,:],coefs[:,4,:,:] = decodevariablefromactivity(Hp[contextmask,:,:],Float64.(X[decisionpoint,codeindices[2][begin],contextmask]))
    contextmask = context[1,:].==code[2]
    _,accuracies[:,5,:],coefs[:,5,:,:] = decodevariablefromactivity(Hp[contextmask,:,:],Float64.(X[decisionpoint,codeindices[1][begin],contextmask]))
    _,accuracies[:,6,:],coefs[:,6,:,:] = decodevariablefromactivity(Hp[contextmask,:,:],Float64.(X[decisionpoint,codeindices[2][begin],contextmask]))
    # full context
    _,accuracies[:,7,:],coefs[:,7,:,:] = decodevariablefromactivity(Hp,c)
    
    # calculate congruency per trial, last is loss, previous are history trials (first index)
    # (trialsequenceposition, gonogo, congruency, context, trials)
    congruency = BitArray(undef,sequencelength,2,2,2,ntrials)
    for (dx,dp) in enumerate(decisionpoints)
        congruency[dx,:,:,:,:] = getcongruency( [X[dp,[codeindices[1][begin],codeindices[2][begin]],:]; context[1,:]'] )
    end

    # calculate congruent incongruent decision from output layer to accuracies list:
    if incongruentonly cnstart = 2 else cnstart = 1 end
    for cn in cnstart:2      # congruent
        for cx in 1:2      # visual and audio
            contextmask = context[1,:].==code[cx]
            mask = contextmask .& ((congruency[end,1,cn,cx,:].==1) .| (congruency[end,2,cn,cx,:].==1))
            ci = 7+(cx-1)*2+cn
            # decode from hidden and from output layer
            _,accuracies[:,ci,:],coefs[:,ci,:,:] = decodevariablefromactivity(Hp[mask,:,:],Float64.(y[decisionpoint,1,mask]))
            _,accuraciesoutput[:,ci-7,:],coefsoutput[:,ci-7,:,:] = decodevariablefromactivity(Op[mask,:,:],Float64.(y[decisionpoint,1,mask]))
        end
    end
    # decision label accuracy manually (without decoder), to establish model performance
    # calculate fraction correct answers in congruenccy categories in the two contexts, for the 
    function getfractioncorrect(decisionarray)
        fractioncorrect = zeros(ntimepoints, 8, 3)        # (timepoints, context+congruency+gonogo, stats)
        if incongruentonly cnstart = 2 else cnstart = 1 end
        for cn in cnstart:2      # congruent
            for cx in 1:2      # visual and audio context
                for sx in 1:2   # go nogo
                    contextmask = context[1,:].==code[cx]
                    ci = (cx-1)*4 + (cn-1)*2 + sx
                    for tgx in 1:sequencelength
                        # mask = contextmask .& ((congruency[1,cn,cx,:].==1) .| (congruency[2,cn,cx,:].==1))         # can be either go or nogo
                        mask = contextmask .& (congruency[tgx,sx,cn,cx,:].==1)                                     # go or nogo (sx)
                        timerange = (tgx-1)*ntimecourse+1:tgx*ntimecourse             # selects timepoints for a trial in a sequence
                        decision = decisionarray[timerange,1,mask] .== targets[tgx,mask]'           # y[:,1,mask]
                        fractioncorrect[timerange,ci,1] = dropdims(mean(decision, dims=2), dims=2)
                        fractioncorrect[timerange,ci,2] = dropdims(std(decision, dims=2), dims=2)
                        fractioncorrect[timerange,ci,3] = fractioncorrect[timerange,ci,2] ./ sqrt(sum(mask))
                    end
                end
            end
        end
        return fractioncorrect
    end
    fractioncorrect = getfractioncorrect(D)



    # display recorded activity in a couple of trials
    triallist = [collect(1:5); collect(ntrials-4:ntrials) ]  # get some trials from both contexts
    ax = plot(layout=(5,1), legend=false, xticks=nothing)
    # left_margin = 40*Plots.px, bottom_margin = 40*Plots.px, size=(7*200,5*200),

    for k in 1:5 vspan!(ax[k], vcat([  [Ts[2][begin],Ts[3][end]].+n*ntimecourse for n in 0:length(triallist)-1  ]...),color=:grey, alpha=0.3) end
    for k in 1:5 vline!(ax[k],[ntimecourse*length(triallist)/2],lw=2,color=:white) end
    for k in [1:1,1:2][Int(onehot)+1]
        axs = ax[1]
        plot!(axs, vcat([X[end-ntimecourse+1:end,k,t] for t in triallist]...), color=[:blue, :purple][k], ylims=(0.001,1.01), ylabel="visual")
        axs = ax[2]
        plot!(axs, vcat([X[end-ntimecourse+1:end,1+Int(onehot)+k,t] for t in triallist]...), color=[:green, :orange][k], ylims=(0.001,1.01), ylabel="audio")
        axs = ax[3]
        plot!(axs, vcat([y[end-ntimecourse+1:end,k,t] for t in triallist]...), color=[:lime, :red][k], ylims=(0.001,1.01), ylabel="target")
        axs = ax[4]
        plot!(axs, vcat([D[end-ntimecourse+1:end,k,t] for t in triallist ]...), color=[:navajowhite,:orange][k], ylabel="decision")
        axs = ax[5]
        plot!(axs, vcat([R[end-ntimecourse+1:end,k,t] for t in triallist ]...), color=[:gold,:red][k], ylims=(0.001,1.01),
                       ylabel="reward", xlabel="visiaul context trials"*repeat(" ",80)*"audio context trials")
    end
    # colors = [:blue,:green,:mediumvioletred]
    # for (cx,x) in enumerate([v,a,c])
    #     axs = ax[2+cx*2-1]
    #     ix = 1+2*(cx-1) # feature index: every odd
    #     vline!(axs,1:ntimecourse:((triallist[end]+1)*ntimecourse),lw=4,color=:grey,alpha=0.5)
    #     plot!(axs, mean(Hp[x.==1,ix,:],dims=3), ribbon=std(Hp[x.==1,ix,:],dims=3)./sqrt(ntrials), color=colors[cx])
    #     axs = ax[2+cx*2]
    #     vline!(axs,1:ntimecourse:((triallist[end]+1)*ntimecourse),lw=4,color=:grey,alpha=0.5)
    #     plot!(axs, mean(Hp[x.==0,ix,:],dims=2), ribbon=std(Hp[x.==0,ix,:],dims=2)./sqrt(ntrials), color=colors[cx], alpha=0.5)
    # end
    # display(ax)



    # figure
    # show activity representation
    nhiddenmax = min(nhidden,20) # max number of neurons in decoder coefficients plot
    axg = plot(layout=@layout([ grid(2,7); grid(3,nhiddenmax); grid(3,6); a{0.07h}; a{0.07h} ]),
               size=(5*400,8*300), xgrid=false, ygrid=false,
               left_margin = 20*Plots.px, bottom_margin = 20*Plots.px, 
               legend=:topleft, label=nothing)
    
    # first subplot section
    ax = axg[1,1]

    # display congruent incongruent in decision plots
    for cx in 1:2
        axs = ax[cx,4]
        vspan!(axs,[Ts[2][begin],Ts[3][end]], color=:grey, alpha=0.3, label=nothing)
        vline!(axs,[decisionpoints[1]],color=:black,lw=2, label=nothing)
        for cn in 1:2         # congruent incongruent
            for sx in 1:2     # go nogo
                # plx = accuracies[:,7+(cx-1)*2+cn,1]             # from hidden layer
                # plx = accuraciesoutput[:,(cx-1)*2+cn,:]               # from output layer
                plx = fractioncorrect[end-ntimecourse+1:end,(cx-1)*4+(cn-1)*2+sx,:]                     # from decision labels fraction correct
                plot!(axs, plx[:,1], ribbon=plx[:,3],
                    color=[:white,:red][sx], ls=[:solid,:dash][cn], label=["congruent","incongruent"][cn]*[" go","nogo"][sx],
                    ylims=(0.45,1.05))
            end
        end
    end

    # display decoders from the last learning epoch
    colors = [:blue :green :dodgerblue :darkgreen :navy :lime :mediumvioletred]
    labels = ["visual","audio","visual relevant","audio irrelevant",
              "visual irrelevant","audio relevant",
              "context" ]
    for (cx,label) in enumerate(labels)

        axs = ax[2-mod(cx,2),(cx-1)÷2+1+(cx==7)]
        vspan!(axs,[Ts[2][begin],Ts[3][end]], color=:grey, alpha=0.3, label=nothing)
        vline!(axs,[decisionpoints[1]],color=:black,lw=2, label=nothing)
        plot!(axs, accuracies[:,cx,1], ribbon=accuracies[:,cx,3], color=colors[cx], ylims=(0.45,1.05), label=label)
        if mod(cx,2)==0 xlabel!(axs,"timepoints") end
    end

    
    # next grid space
    # decoder coeficients, colors -> decoders, xticks -> hidden cells
    for nx in 1:nhiddenmax
        for (rx,region) in enumerate([Ts[2],Ts[3],Ts[4]])     # early stim, decision stim, post stim
            axs = axg[2,1][rx,nx]

            # bm = dropdims(mean(coefs[region,cx,1:end-1],dims=1),dims=1)
            # sm = dropdims(std(coefs[region,cx,1:end-1],dims=1),dims=1)
            # em = sm / sqrt(length(region))
            
            violin!(axs, coefs[region,1:7,nx,1], color=colors, legend=false)
            xticks!(axs, 1:7)
            hline!(axs,[0],color=:gold,alpha=0.8)
            yl = 3*mean(abs.(coefs[region,1:7,nx,1]))
            ylims!(axs,-yl,yl)

            if nx==1
                ylabel!(axs,["stim start - water","water - stim end","post stim"][rx])
                plot!(axs, left_margin=50*Plots.px)
            else
                plot!(axs, yticks=nothing)
            end
        end
    end

    

    # display loss over epochs
    legendplacement = :best
    plot!(ax[1,6], l, yaxis=:log, color=:red, label="loss", legend=legendplacement)
    # plot!(ax[1,7], L[:,2], yaxis=:log, color=:firebrick, label="crossentropy", legend=legendplacement)
    regularization = Symbol.(machineparameters[:regularization])
    if :weightregularization in regularization
        plot!(ax[1,7], L[:,3], yaxis=:log, color=:yellow, label="regularization", legend=legendplacement)
    end
    if :contextsymmetric in regularization
        plot!(ax[2,5], L[:,6], yaxis=:log, color=:fuchsia, label="symmetrization", legend=legendplacement)
    end
    if :sparse in regularization
        plot!(ax[2,6], L[:,4], yaxis=:log, color=:slategrey, label="sparseness", legend=legendplacement)
    elseif :energy in regularization
        plot!(ax[2,6], L[:,4], yaxis=:log, color=:slategrey, label="energy", legend=legendplacement)
    end
    if :nonnegative in regularization
        plot!(ax[2,7], L[:,5], yaxis=:log, color=:slategrey, label="nonnegative", legend=legendplacement)
    end



    # next grid space
    ax = axg[3,1]

    # display performance throughout epochs by congruency
    skiprate = nepochs÷size(Ds,1)
    cong = getcongruency( [ X[decisionpoint,[codeindices[1][begin],codeindices[2][begin]],:]; context[1,:]' ] )
    colors = [ :blue :purple; :green :orange ]
    labelstask = [ "visual go" "visual nogo" ; "audio go" "audio nogo" ]
    lss = [:solid :dash]
    labelscong = ["congruent " "incongruent "]
    for cx in [1,2]
        axs = ax[cx,6]
        if cx==1 title!(axs,"performance, congruency") end
        for st in 1:2 #[1:1,1:2][Int(onehot)+1]
            for cn in 1:2
                # output 1 is go, output 2 is nogo
                # subtraction: use correct minus incorrect value:
                # aux = Ds[:,decisionpoint,st,cong[st,cn,cx,:]]-Ds[:,decisionpoint,3-st,cong[st,cn,cx,:]]
                # or use go minus nogo value (if it is +, the response is go, if it is -, the response is nogo):
                aux = Ds[:,decisionpoint,1,cong[st,cn,cx,:]]-Ds[:,decisionpoint,2,cong[st,cn,cx,:]]
                if st==2
                    aux = 1 .- aux
                end
                m = mean(aux,dims=2)
                s = std(aux,dims=2) 
                e = s ./ sqrt(size(s,2))
                plot!(axs, skiprate:skiprate:nepochs, m, ribbon=e, lw=2, color=colors[cx,st], ls=lss[cn],
                           label=labelscong[cn]*labelstask[cx,st])
                hline!(axs,[0],color=:white,alpha=0.3,ls=:dash,label=nothing)
                # ylims!(axs, (ylims(axs)[1], ylims(axs)[2]*1.25))
            end
        end
        # plot!(axs,legend=:topright)
        plot!(axs,legend=nothing)
    end



    # display weights through epochs
    stimcolors = [:dodgerblue,:blue, :lime,:green, :fuchsia,:purple, :gold,:darkgoldenrod]
    addcontext = size(X,2)==8; if ! addcontext deleteat!(stimcolors,5:6) end
    labels = ["W input", "W recurrent", "W output"]
    # input and hidden will be colored by input, output will be colored by output
    axs = ax[1,3]
    for k in axes(Wi,3)
        plot!(axs, skiprate:skiprate:nepochs, Wi[:,:,k], lw=2, palette = palette([stimcolors[k], :black], size(Wi,3)+2), legend=false )
    end
    title!(axs,labels[1])
    
    axs = ax[1,4]
    for k in axes(Wh,3)
        plot!(axs, skiprate:skiprate:nepochs, Wh[:,:,k], lw=2, palette = palette([:mediumvioletred, :black], size(Wh,3)+2), legend=false )
    end
    title!(axs,labels[2])
    
    axs = ax[1,5]
    for k in axes(Wo,2)
        plot!(axs, skiprate:skiprate:nepochs, Wo[:,k,:], lw=2, ls=[:solid,:dash][k], palette = palette([:lime, :red], size(Wo,2)+2), legend=false )
    end
    title!(axs,labels[3])

    for wx in 1:3
        axs = ax[2,2+wx]
        W = [Wi,Wh,Wo][wx][end,:,:]
        heatmap!(axs,W; palette=palette(:berlin,5), clims=(-1.,1.), 
                 aspect_ratio=1,colorbar=:none, showaxis=false, xticks=nothing, yticks=nothing)
        if wx==1 plot!(axs,xticks=1:6, xticklabels=("visual +", "visual -", "audio +", "audio -", "reward +", "reward -")) end
    end



    # display last epoch hidden activities
    
    if :ei in machineparameters[:architecture]  # excitatory inhitory
        for cx in 1:2        # all same context
            for sx in 1:2    # contextual stimulus go vs nogo signals (sx)
                for (nx,neuronlist) in enumerate([1:n_excitatory, n_excitatory+1:n_excitatory+n_inhibitory])           # excitatory/inhitory neurons
                    label = ["excitatory","inhibitory"][nx]*" - "*["visual context","audio context"][cx]*", "*["go","nogo"][sx]
                    lss = [:solid :dash :solid :dash]     # goes over go nogo go nogo
                    # colors = [:purple :purple :seagreen :seagreen]
                    colors = [:blue :blue :green :green]   # goes over 2 visual neurons 2 audio
                    if nx==2 colors = colors[:,2:3]; ls = lss[:,[1,3]] end     # use inhibitory just 2, not onehot
                    axs = ax[2+sx,(cx-1)*2+nx]
                    h = dropdims(mean(Hp[(c.==(cx-1)) .& ((v,a)[cx].==(sx-1)),:,neuronlist], dims=1),dims=1)

                    vspan!(axs,[Ts[2][begin],Ts[3][end]], color=:grey, alpha=0.3, label=nothing)
                    vline!(axs,[decisionpoints[1]],color=:black,lw=2, label=nothing)
                    hline!(axs,[0],color=:white,alpha=0.3,ls=:dash,label=nothing)

                    plot!(axs, h, color=colors, ls=lss, lw=2,label=nothing)
                    plot!(axs,ylims=(-1,1),title=label)
                    # annotate!(axs,Ts[2][1]/2,mean(h[neuronlist]),"$(mean(Wh[5:6,neuronlist],dims=2))", textcolor=:white)
                end
            end
        end
    else      # or choose irrespectiv of excitatory inhibitory
        for cx in 1:2        # all same context
            for sx in 1:2    # contextual stimulus go vs nogo signals (sx)
                for (nx,neurons) in enumerate([1:nfeatures])
                    label = ["visual","audio"][cx]*"  context, "*["go","nogo"][sx]
                    axs = ax[cx,sx]
                    # trial averaged on the same type of trials
                    h = Hp[(c.==(code[cx])) .& ((v,a)[cx].==(code[sx])),:,neurons]
                    m = dropdims(mean(h, dims=1),dims=1)
                    s = dropdims(std(h, dims=1),dims=1)
                    e = s/sqrt(size(h,3))

                    vspan!(axs,[Ts[2][begin],Ts[3][end]], color=:grey, alpha=0.3, label=nothing)
                    vline!(axs,[decisionpoints[1]],color=:black,lw=2, label=nothing)
                    hline!(axs,[0],color=:white,alpha=0.3,ls=:dash,label=nothing)

                    plot!(axs, m, ribbon=e, lw=2, palette=palette(:autumn1,length(neurons)), label=nothing)       # color=colors

                    title!(axs, label)
                    if :nonnegativerelu in machineparameters[:architecture] || :nonnegativesigmoid in machineparameters[:architecture] yll = 0; ylm = 4 else yll = -2; ylm = 2 end
                    ylims!(axs,yll,ylm)
                    # annotate!(axs,Ts[2][1]/2,mean(h[neuronlist]),"$(mean(Wh[5:6,neuronlist],dims=2))", textcolor=:white)

                    if sx==1 ylabel!(axs, "hidden activity") end
                end
            end
        end
    end

    
    # context representation details

    # plot decoder weights throughout the timeline of the last trial
    axs = ax[3,5]
    heatmap!(axs, coefs[:,7,:,1]', colormap=:berlin, colorbar=nothing, clims=(-2.,2.) )

    # context dependent activities of cells
    axs = ax[3,1]
    heatmap!(axs, dropdims(mean(Hp[contextmask,:,:],dims=1),dims=1)', colormap=:thermal, clims=(0.,3.), colorbar=nothing, title="visual context"  )
    axs = ax[3,2]
    heatmap!(axs, dropdims(mean(Hp[.! contextmask,:,:],dims=1),dims=1)', colormap=:thermal, clims=(0.,3.), colorbar=nothing, title="audio context" )


    
    
    # next subplot block
    # plot trial history
    trialdecisionpoint = Ts[3][begin]
    Hq = Float64.(permutedims(H,[3,1,2]))
    Dq = Float64.(permutedims(D,[3,1,2]))
    c = Float64.(context[1,:])
    contextcolor = [:dodgerblue,:lime]
    _,contextaccuracy,_ = decodevariablefromactivity(Hq,c)

    performance = zeros(2,2,2)        # holds fraction correct for (context,go/nogo,congruency)
    dq = Float64.(X[decisionpoints,[3,5][Int(onehot)+1],:])
    for cx in 1:2
        contexttrials = (1:ntrials÷2) .+ (cx-1) * ntrials÷2
        contextmask = context[1,:].==code[cx]

        decisionaccuracies = zeros(ntimecourse, sequencelength, 3)
        decisionaccuraciesoutputlayer = zeros(3,ntimecourse, sequencelength, 3)
        for dx in 1:sequencelength
            _,decisionaccuracies[:,dx,:],_ = decodevariablefromactivity(
                        Hq[contexttrials,1+ntimecourse*(dx-1):ntimecourse*dx,:],
                        dq[dx,contexttrials])
            _,decisionaccuraciesoutputlayer[1,:,dx,:],_ = decodevariablefromactivity(
                        Dq[contexttrials,1+ntimecourse*(dx-1):ntimecourse*dx,:],
                        dq[dx,contexttrials])
            for cn in cnstart:2      # congruent
                mask = contextmask .& ((cong[1,cn,cx,:].==1) .| (cong[2,cn,cx,:].==1))
                _,decisionaccuraciesoutputlayer[1+cn,:,dx,:],_ = decodevariablefromactivity(
                            Dq[mask,1+ntimecourse*(dx-1):ntimecourse*dx,:],
                            dq[dx,mask])
            end
        end
        decisionaccuracies = reshape(decisionaccuracies,:,3)
        decisionaccuraciesoutputlayer = reshape(decisionaccuraciesoutputlayer,3,:,3)
        
        # trial = Int(ceil(rand()*ntrials))
        # Hs,Ds,Rs are (nepochs,ntimecourse,{nhidden,2,2},ntrials)
        # precision = mean((df[1,contexttrials].>df[2,contexttrials]).==y[decisionpoints[end],1,contexttrials])
        ! onehot ? comp = 0 : comp = D[trialdecisionpoint,2,contexttrials]
        precision = mean((D[trialdecisionpoint,1,contexttrials].>comp) .==
                             y[decisionpoints[end],1,contexttrials])
        
        # @info "context" cx size(df) contexttrials df[:,contexttrials] D[trialdecisionpoint,:,contexttrials] X[Ts[3][begin]+ntimepoints-ntimecourse,[1,3,5],contexttrials]
        # @info "sanity" D[trialdecisionpoint,:,contexttrials] y[decisionpoints[end],1,contexttrials]' ((D[trialdecisionpoint,1,contexttrials].>D[trialdecisionpoint,2,contexttrials]) .== y[decisionpoints[end],1,contexttrials])'

        
        axs = axg[3+cx,1]
        # plot!(axs,decisionaccuracies[:,1],ribbon=decisionaccuracies[:,3],color=contextcolor[cx],lw=2,label="decision hidden")
        # plot!(axs,decisionaccuraciesoutputlayer[1,:,1],ribbon=decisionaccuraciesoutputlayer[1,:,3],color=:orange,lw=2,label="decision output")
        # plot!(axs,decisionaccuraciesoutputlayer[2:3,:,1]',ribbon=decisionaccuraciesoutputlayer[2:3,:,3],color=[:white :red],lw=2,label="decision ".*["congruent" "incongruent"])

        plot!(axs,contextaccuracy[:,1],ribbon=contextaccuracy[:,3],color=:mediumvioletred,lw=3,label="context")

        for sx in 1:2          # go nogo
            for cn in cnstart:2      # congruent
                plx = fractioncorrect[:,(cx-1)*4+(cn-1)*2+sx,:]                     # from decision labels fraction correct
                plot!(axs, plx[:,1], ribbon=plx[:,3], color=[:white,:red][sx], ls=[:solid,:dash][cn], lw=3,
                      label="decision "*["congruent","incongruent"][cn]*[" go"," nogo"][sx])
                performance[cx,sx,cn] = plx[decisionpoint,1]
            end
        end

        ylims!(axs,0,1)

        vspan!(axs,reshape((repeat([Ts[2][begin] Ts[3][end]],sequencelength).+(0:ntimecourse:sequencelength*(ntimecourse-1)))',:,1), color=:grey, alpha=0.3, label=nothing)
        vline!(axs,decisionpoints,color=:black,lw=2, label=nothing)
        hline!(axs,[0],color=:white,alpha=0.3,ls=:dash,label=nothing)

        ylabel!(axs, ["visual","audio"][cx]*" context")
        ylabel!(axs, ["visual","audio"][cx]*" context")

    end

    plot!(axg,plot_title=filename*"-e$(machineparameters[:snapshots][end])")

    # display the entire plot grid
    if config[:showplot] display(axg) end
    if config[:saveplot]
        savefig(axg,joinpath(config[:modelresultspath],"models/",filename) )
    end

    


    # new figure
    # mutual information between task variables and neurons
    ax = plot(layout=(13,nhiddenmax),
               size=(nhiddenmax*240,13*200+80), xgrid=false, ygrid=false,
               top_margin = 60*Plots.px, left_margin = 60*Plots.px, bottom_margin = 20*Plots.px,
               legend=:topleft, label=nothing)

    # X is (ntimepoints,nfeatures,ntrials)
    # H is (ntimepoints,nhidden,ntrials)

    # add contextual information to X
    selecttrials = [ 1:ntrials, 1:ntrials, 1:ntrials,
                     1:ntrials÷2, 1:ntrials÷2, 1:ntrials÷2,
                     ntrials÷2+1:ntrials, ntrials÷2+1:ntrials, ntrials÷2+1:ntrials,
                     1:ntrials ]
    Xcx = cat(repeat(X[:,1:2:5,:],1,3,1), [ones(Float32,ntimepoints,1,ntrials÷2);;; zeros(Float32,ntimepoints,1,ntrials÷2)], dims=2)

    colors = [ :blue :green :orange   :dodgerblue :darkgreen :orange   :navy :lime :orange   :mediumvioletred]
    labelstask = [ "visual" "audio" "decision"   "visual rel" "audio irrel" "decision vis" "visual irrel" "audio rel" "decision aud"   "context"]

    
    # if export mit and ortho, go over all snapshots, and save distances into files
    # if no export, only calculate the last snapshot mit
    mit = 0
    if config[:exportalldistances]
        snapshotsfordistances = machineparameters[:snapshots]
    else
        snapshotsfordistances = [machineparameters[:snapshots][end]]
    end
    for (sx,s) in enumerate(snapshotsfordistances)
        H = Hs[sx,:,:,:]
        # snap = lpad(s,machineparameters[:ndigitsnepochs],"0")
        # @load(joinpath(config[:modelanalysispath],  "distances", filename*"-e"*snap*"-mit,ortho.bson"), @__MODULE__, Ts, performance, mit, ortho, labelstask, colors)
        if config[:exportalldistances]     # these does not need to be calculated for the last snap "mut" plot, only for export
            # ortho = [ getortho(H,Xcx,selecttrials, ntimepoints - Ts[end][end] + Ts[2][1]),  getortho(H,Xcx,selecttrials,decisionpoint) ]
            ortho = getortho(H,Xcx,selecttrials,decisionpoint)            # last is the time where the label to be included
            snap = lpad(s,machineparameters[:ndigitsnepochs],"0")
            performance = zeros(2,2,2)        # holds fraction correct for (context,go/nogo,congruency)
            fractioncorrect = getfractioncorrect(Ds[sx,:,:,:])
            for cx in 1:2      # context
                for sx in 1:2          # go nogo
                    for cn in cnstart:2      # congruent
                        performance[cx,sx,cn] = fractioncorrect[decisionpoint,(cx-1)*4+(cn-1)*2+sx,1]                     # from decision labels fraction correct
                    end
                end
            end
            @save(joinpath(config[:modelanalysispath],  "distances", filename*"-e"*snap*"-mit,ortho.bson"), Ts, performance, ortho, labelstask, colors)
        end
    end

    

    # use distances from the last epoch
    for xi in axes(Xcx,2), hi in 1:nhiddenmax
        axs = ax[xi,hi]

        # plot!(axs, mit[:,xi,hi], color=colors[xi], lw=3, legend=false)
        ylims!(axs, 0, 1)

        vspan!(axs,reshape((repeat([Ts[2][begin] Ts[3][end]],sequencelength).+(0:ntimecourse:sequencelength*(ntimecourse-1)))',:,1), color=:grey, alpha=0.3, label=nothing)
        vline!(axs,decisionpoints,color=:black,lw=2, label=nothing)
        # hline!(axs,[0],color=:white,alpha=0.3,ls=:dash,label=nothing)

        if xi==1 title!(axs, "hidden $hi") end
        if xi<size(Xcx,2) plot!(axs, xticks=nothing) end
        if hi==1 ylabel!(axs, labelstask[xi]) else plot!(axs,yticks=nothing) end

    end

    
    for hi in 1:nhiddenmax
        for (rx,region) in enumerate([Ts[2],Ts[3],Ts[4]])     # early stim, decision stim, post stim
            axs = ax[10+rx,hi]
            # plot distribution, but avoid nan results
            # if ~ any(isnan.(mit[ntimepoints-ntimecourse .+ region,:,hi]))
            #     violin!(axs, (1:size(Xcx,2))', mit[ntimepoints-ntimecourse .+ region,:,hi], color=colors, legend=false)
            # end
            xlims!(axs,0,size(Xcx,2)+1)
            ylims!(axs,0,1)

            if rx<3 plot!(axs, xticks=nothing) else plot!(axs, xticks=1:size(Xcx,2)) end
            if hi==1
                ylabel!(axs,["stim start - water","water - stim end","post stim"][rx])
                plot!(axs, left_margin=50*Plots.px)
            else
                plot!(axs,yticks=nothing)
            end

            # if hi==1 && rx==1 title!(axs, "periods distributions\nlast trial") end
        end
    end

    plot!(ax,plot_title=filename*"-e$(machineparameters[:snapshots][end])-mut")

    # this is too big plot, cannot show, only in file
    # if config[:showplot] display(ax) end
    if config[:saveplot]
        savefig(ax,joinpath(config[:modelresultspath],"models/",filename))
    end



end




function decodernn(X, y, Ts, machineparameters)
    # load models
    Plots.Random.seed!(parse(Int,machineparameters[:modelid]))

    nepochs = machineparameters[:snapshots][end]
    nsnapshots = length(machineparameters[:snapshots])
    nhidden = machineparameters[:nhidden]
    
    Xs, ys, sequencelength, ntimepoints, ntimecourse, ntrials,
         decisionpoints, decisionpoint, rewardprofile = preparedata(X, y, Ts)

    # sequence, hidden units, maxepochs, randomseed, and the snapshot will be automatically added
    folder = machineparameters[:modeltype]*"-s$sequencelength-h$nhidden"
    filename = machineparameters[:modeltype]*"-s$sequencelength-h$nhidden-r"*machineparameters[:modelid]
    path = joinpath(config[:rawmodelpathprefix],config[:modelspath],folder,filename)


    # coding onehot
    codeindices = [1:2,3:4,5:6,7:8]

    # labels: (decisionpoint is inside the active stimulus period: safe timecourse point)
    v = Float64.(X[decisionpoint,1,:])
    a = Float64.(X[decisionpoint,3,:])
    contexts = [ ones(ntrials÷2); zeros(ntrials÷2) ]
    targets = Float64.(y[decisionpoints,1,:])


    congruency = BitArray(undef,sequencelength,2,2,2,ntrials)
    for (dx,dp) in enumerate(decisionpoints)
        congruency[dx,:,:,:,:] = getcongruency( [X[dp,[1,3],:]; contexts'] )
    end
    s = [X[decisionpoints[end],[1,3],:]; contexts']
    # decision label accuracy manually (without decoder), to establish model performance
    # calculate fraction correct answers in congruenccy categories in the two contexts, for the 
    function getfractioncorrect(decisionarray)
        fractioncorrect = zeros(ntimepoints, 2, 2, 2, 3)        # (timepoints, context+congruency+gonogo, stats)
        for cn in 1:2      # congruent
            for cx in 1:2      # visual and audio context
                for sx in 1:2   # go nogo
                    contextmask = contexts.==1
                    for tgx in 1:sequencelength # go over trials
                        mask = contextmask .& (congruency[tgx,sx,cn,cx,:].==1)                                     # go or nogo (sx)
                        @info "" tgx sum(mask) sum(congruency[tgx,sx,cn,cx,:].==1)    mask .== (congruency[tgx,sx,cn,cx,:].==1)
                        timerange = (tgx-1)*ntimecourse+1:tgx*ntimecourse             # selects timepoints for a trial in a sequence
                        decision = decisionarray[timerange,1,mask] .== decisions[tgx,mask]'           # y[:,1,mask]
                        fractioncorrect[timerange,cx,cn,sx,1] = dropdims(mean(decision, dims=2), dims=2)
                        fractioncorrect[timerange,cx,cn,sx,2] = dropdims(std(decision, dims=2), dims=2)
                        fractioncorrect[timerange,cx,cn,sx,3] = fractioncorrect[timerange,cx,cn,sx,2] ./ sqrt(sum(mask))
                    end
                end
            end
        end
        return fractioncorrect
    end

    


    # load the model with tests
    Hs,Os,Ds,Rs,Wi,Wh,Wo,L,l = learnrnn( Xs, ys, decisionpoints, rewardprofile, machineparameters,
                                       train=false, filename=path )

    # calculate fraction correct at each decision point for each epoch for Ds
    congruency = getcongruencyhistory(X[decisionpoints,[1,3],:])
    congruencyexample = vec(prod(congruency .== [true, true, false, true, false], dims=1))
    fractioncorrects = zeros(nsnapshots, ntimepoints,4)
    contextaccuracy = zeros(nsnapshots, ntimepoints,2)
    targets = [X[:,1,1:ntrials÷2] X[:,3,ntrials÷2+1:end]]       # y contains onlyvalues only after decisionpoint, so need to recreate
    Hsp = Float64.(permutedims(Hs,[1,4,2,3])) # permute for decoders
    for (ex,e) in enumerate(machineparameters[:snapshots])
        # decode the context variable
        corrects = targets .== Ds[ex,:,1,:]
        fractioncorrects[ex,:,1] = mean(corrects,dims=2)
        fractioncorrects[ex,:,2] = mean(corrects[:,congruency[end,:]],dims=2)
        fractioncorrects[ex,:,3] = mean(corrects[:,.! congruency[end,:]],dims=2)
        fractioncorrects[ex,:,4] = mean(corrects[:,congruencyexample],dims=2)
        _,ca,_ = decodevariablefromactivity(Hsp[ex,:,:,:],contexts)
        contextaccuracy[ex,:,1] = ca[:,1]         # 1st is mean accuracy
        _,cae,_ = decodevariablefromactivity(Hsp[ex,congruencyexample,:,:],contexts[congruencyexample])
        contextaccuracy[ex,:,2] = cae[:,1]         # 2nd is example sequence accuracy
    end
    @info "r" size(fractioncorrects) size(contextaccuracy) # size(decisionaccuracy)
    decisions = Ds[:,:,1,:]
    @save(joinpath(config[:modelresultspath],"analysis/decoders/", "decoders-$(string(machineparameters[:modelid])).bson"),
                  contextaccuracy, fractioncorrects, Ts, decisionpoints, decisions)







end








function contextinferencernn(X, y, Ts, machineparameters)
    # find the first incongruent trial after congruents in the sequence
    # separate correct and error incongruent trials
    # look separately the activity after decision and reward input
    # repeat for the two contexts and go and nogo separately
    # -> identify the computation for the context inference
    fillrange!(machineparameters)

    Xs, ys, sequencelength, ntimepoints, ntimecourse, ntrials,
        decisionpoints, decisionpoint, rewardprofile = preparedata(X, y, Ts)
    maskcontext, maskvisualgo, maskaudiogo, maskgo = getmasks(X[decisionpoint,[1,3],:])
    congruency = getcongruencyhistory(X[decisionpoints,[1,3],:])

    nmodels = length(machineparameters[:modelids])
    nepochs = machineparameters[:snapshots][end]
    nsnapshots = length(machineparameters[:snapshots])
    nhidden = machineparameters[:nhidden]
    folder = machineparameters[:modeltype]*"-s$sequencelength-h$nhidden"


    # params
    sequencequerypositions = 1:4          # position of trial in the sequence for incongruent prceded by congruents, calculate



    if config[:contextinferencerecalculate]

        H = zeros(nmodels, length(sequencequerypositions), ntimepoints, nhidden, 2, 2, 2) #  context, gonogo, correct
        modalityorders = zeros(Int, nmodels, nhidden, 2)
        decisionorders = zeros(Int, nmodels, nhidden, 2)
        rewardinputweights = zeros(nmodels, nhidden, 2)                  # reward and punish
        rewardorders = zeros(Int, nmodels, nhidden, 2)
        contextorders = zeros(Int, nmodels, nhidden, 2)
        postpreorders = zeros(Int, nmodels, nhidden, 2, 2)
        Wis = zeros(nmodels, nhidden, size(X,2))
        Whs = zeros(nmodels, nhidden, nhidden)
        Wos = zeros(nmodels, 2, nhidden)

        for (mx,modelid) in enumerate(machineparameters[:modelids])

            print(modelid*" ")

            # load models
            Plots.Random.seed!(parse(Int,modelid))

            
            filename = machineparameters[:modeltype]*"-s$sequencelength-h$nhidden-r"*modelid
            path = joinpath(config[:rawmodelpathprefix],config[:modelspath],folder,filename)



            # labels: (decisionpoint is inside the active stimulus period: safe timecourse point)
            v = Float64.(X[decisionpoint,1,:])
            a = Float64.(X[decisionpoint,3,:])
            contexts = [ ones(ntrials÷2); zeros(ntrials÷2) ]





            # load the model with tests
            # Hs is (nsnapshots,ntimecourse,nhidden,ntrials)  
            Hs,Os,Ds,Rs,Wi,Wh,Wo,L,l = learnrnn( Xs, ys, decisionpoints, rewardprofile, machineparameters,
                                            train=false, filename=path )

            decisions = Ds[end,decisionpoints,1,:]
            targets = Float64.(y[decisionpoints,1,:])
            corrects = decisions .== targets

            # neuron roles  by weight order
            modalityorders[mx,:,:] = [sortperm(Wi[end,:,1]-Wi[end,:,3], rev=true) sortperm(Wi[end,:,2]-Wi[end,:,4], rev=true) ]
            decisionorders[mx,:,:] = [sortperm(Wo[end,1,:], rev=true) sortperm(Wo[end,2,:], rev=true) ]
            rewardinputweights[mx,:,:] = Wi[end,:,5:6]
            rewardorders[mx,:,:] = [sortperm(Wi[end,:,5], rev=true) sortperm(Wi[end,:,6], rev=true) ]
            Wis[mx,:,:] = Wi[end,:,:]
            Whs[mx,:,:] = Wh[end,:,:]
            Wos[mx,:,:] = Wo[end,:,:]
            


            # context inference at the last trial in the sequence to make sure it knows the context correctly
            Hp = Float64.(permutedims(Hs[end,end-Ts[end][end]+1:end,:,:],[3,1,2]))
            for (mskx,maskdir) in enumerate((maskgo, .! maskgo))
                _,accuracies,coefs = decodevariablefromactivity(Hp,Float64.(maskdir .& maskcontext))
                contextorders[mx,:,mskx] = sortperm( mean(coefs,dims=1)[1,1:nhidden,1], rev=true)
            end

            






            # collect each first incongruent trial after congruents in the sequence
            for sqp in sequencequerypositions
                # collect 2nd incongruent after incongruent trial
                maskcongruency = vec(prod(congruency[1:sqp,:] .== [trues(sqp-1); false], dims=1))

                # go over contexts and go nogo
                for cx in 1:2
                    for (gx,gonogo) in enumerate([1,0])
                        maskgonogo = y[decisionpoints[sqp],1,:].==gonogo
                        for dx in 1:2     # check correct or error
                            maskcorrect = corrects[sqp,:] .== [true,false][dx]
                            mask = [maskcontext, .! maskcontext][cx] .& maskgonogo .& maskcongruency .& maskcorrect
                            H[mx,sqp,:,:,cx,gx,dx] = dropdims(mean(Hs[end,:,:,mask],dims=3),dims=3)
                            # @info "" cx,gx,dx sum(mask) sum(maskcorrect)
                        end
                    end
                end


                # compare poststimulus activity to prestimulus activity, and if there is difference order neurons by it
                timestamps = 1:ntimecourse .+ (sqp-1)*ntimecourse
                for gx in 1:2
                    for (tx,timepartids)  in enumerate(([4,1],[3,2]))
                        postpreactivitydifference = dropdims(mean(H[mx,sqp,timestamps[Ts[timepartids[1]]],:,:,gx,:],dims=(1,3,4)) -
                                                            mean(H[mx,sqp,timestamps[Ts[timepartids[2]]],:,:,gx,:],dims=(1,3,4)),dims=(1,3,4))
                        postpreorders[mx,:,tx,gx] = sortperm(postpreactivitydifference, rev=true)
                    end
                end
            end



            
        end
        println()
        
        @save(joinpath(config[:modelresultspath],"analysis/contextinference/", "contextinference.bson"),
                    H, modalityorders, decisionorders, rewardinputweights, rewardorders, contextorders, postpreorders,
                        Wis, Whs, Wos, Ts, decisionpoints)

    else

        @load(joinpath(config[:modelresultspath],"analysis/contextinference/", "contextinference.bson"), @__MODULE__,
                    H, modalityorders, decisionorders, rewardinputweights, rewardorders, contextorders, postpreorders,
                        Wis, Whs, Wos)

    end




    # sequence and unit params
    sequencequerypositions = 1:4          # position of trial in the sequence for incongruent prceded by congruents, test
    bestlast = 3         #  number of best ordered units for a role


    # use only valid models, that has high performance in all incongruent both contexts after learning
    @load( joinpath(config[:modelanalysispath],  "suppression", "subspaces-rnn.bson"), @__MODULE__, validmodels)
    nmodels = sum(validmodels)
    H = H[validmodels,:,:,:,:,:,:]
    modalityorders = modalityorders[validmodels,:,:]
    decisionorders = decisionorders[validmodels,:,:]
    rewardinputweights = rewardinputweights[validmodels,:,:]
    rewardorders = rewardorders[validmodels,:,:]
    contextorders = contextorders[validmodels,:,:]
    postpreorders = postpreorders[validmodels,:,:,:]
    Wis = Wis[validmodels,:,:]
    Whs = Whs[validmodels,:,:]
    Wos = Wos[validmodels,:,:]





    # create aggregate from all models

    Ha = zeros(ntimecourse,4,4,2,2,2)         # mean, 4 order types (one + -, other + -), 4 orderings (modality, decision, reward, context)
    Hs = zeros(ntimecourse,4,4,2,2,2)         # std, 4 order types (one + -, other + -), 4 orderings (modality, decision, reward, context)
    Ho = zeros(ntimecourse,nhidden,4,2,2,2)   # n neurons, 4 orderings (modality, decision, reward, context)
    ns = zeros(2,2,2)                         # number of models in each category (context, decision, correctness)
    for mx in axes(H,1), sqp in sequencequerypositions
        timestamps = (sqp-1)*ntimecourse+1:sqp*ntimecourse       # we are only interestd in the trial at the queried sequence position 
        for cx in 1:2, gx in 1:2, dx in 1:2      # context, decision, correctness
            if all(.! isnan.(H[mx,sqp,timestamps,:,cx,gx,dx]))
                for (j,orders) in enumerate([modalityorders, decisionorders, rewardorders, postpreorders[:,:,2,:]])   # contextorders, 
                    for i in 1:4            # correct error go nogo series
                        neurons = [ orders[mx,1:bestlast,1], orders[mx,end-bestlast+1:end,1], 
                                    orders[mx,1:bestlast,2], orders[mx,end-bestlast+1:end,2] ][i]
                        # mean over selecte best neurons
                        Ha[:,i,j,cx,gx,dx] .+= dropdims(mean(H[mx,sqp,timestamps,neurons,cx,gx,dx],dims=(2)),dims=(2))
                        Hs[:,i,j,cx,gx,dx] .+= dropdims(std(H[mx,sqp,timestamps,neurons,cx,gx,dx],dims=(2)),dims=(2))
                    end
                    # register all neurons
                    Ho[:,:,j,cx,gx,dx] .+= H[mx,sqp,timestamps,orders[mx,:,gx],cx,gx,dx]
                end
                ns[cx,gx,dx] += 1
            end
        end
    end
    for t in axes(Ha,1), j in 1:3
        for i in 1:4
            Ha[t,i,j,:,:,:] ./= ns
            Hs[t,i,j,:,:,:] ./= ns
        end
        for n in 1:nhidden
            Ho[t,n,j,:,:,:] ./= ns
        end
    end
    


    # plot H in 4 + 4 subplots cx by gx, each hidden unit  on the same plot,  x is time, y is activity
    orderslabels = [["visual go", "audio go", "visual nogo", "audio nogo"],
                    ["go +", "go -", "nogo +", "nogo -"],
                    ["reward +", "reward -", "punish +", "punish -"]]
    timestamps = 1:ntimecourse
    ax = plot(layout=(6,4), size=(4*300,6*250), left_margin=30*Plots.px, legend=false, foreground_color_legend=nothing, background_color_legend=nothing)
    for cx in 1:2
        for gx in 1:2
            for dx  in 1:2
                for j in 1:3
                    axs = ax[cx+2*(j-1),gx+2*(dx-1)]
                    vspan!(axs, [Ts[2][begin], Ts[3][end]], color=:darkgrey, alpha=0.3, label=nothing)
                    vline!(axs, [Ts[3][begin]], color=:grey, lw=2, alpha=0.5, label=nothing)
                    hline!(axs,[0],color=:grey, ls=:dash, label=nothing)
                    
                    for i in 1:4
                        plot!(axs, Ha[:,i,j,cx,gx,dx], #ribbon=Hs[:,i,j,cx,gx,dx],
                                        lw=2, color=[:gold,:red][dx],
                                        ls=[:solid,:solid,:dash,:dash][i], alpha=[1,0.5,1,0.5][i], label=orderslabels[j][i])
                    end
                    plot!(ax, legend=:topleft)
                    ylims!(-1,1)
                    title!(axs,["go","nogo"][gx]*" "* ["correct","error"][dx] *" "*["visual","audio"][cx])
                    if gx+2*(dx-1)==1 ylabel!(axs,["modality","decision","reward"][j]) end
                end
            end
        end
    end
    # display(ax)


    # savefig(ax, joinpath(config[:modelresultspath],"analysis/", folder*"-contextinference.png") )




    # plot compound connections
    # compare the target neurons of the reward neurons to modality and decision indexes
    ax = plot(layout=(3,4), size=(4*300,3*250), left_margin=30*Plots.px, legend=false, foreground_color_legend=nothing, background_color_legend=nothing)

    modalitypositions = zeros(Int, nmodels, bestlast, bestlast, 2)
    decisionpositions = zeros(Int, nmodels, bestlast, bestlast, 2)
    rewardweightsdistribution = [Float64[], Float64[]]
    Fs = zeros(nhidden,nhidden,2)         #  reward and punish orders
    for mx in 1:nmodels, gx in 1:2
        for ri in 1:bestlast
            rewardoutputorders = sortperm(Whs[mx,:,rewardorders[mx,ri,1]], rev=true)
            
            modalitypositions[mx,:,ri,gx] = [ findfirst(rewardoutputorders[mi].==modalityorders[mx,:,gx]) for mi in 1:bestlast ]
            decisionpositions[mx,:,ri,gx] = [ findfirst(rewardoutputorders[mi].==decisionorders[mx,:,gx]) for mi in 1:bestlast ]
            
            push!(rewardweightsdistribution[1], Whs[mx,:,rewardorders[mx,ri,1]]...)
            push!(rewardweightsdistribution[2], Whs[mx,:,rewardorders[mx,ri,2]]...)

            rewardinputweights


        end
        
        # weights


        # recurrent connections
        Fs[:,:,1] .+= Whs[mx,rewardorders[mx,:,1],rewardorders[mx,:,1]]      # reward order
        Fs[:,:,2] .+= Whs[mx,rewardorders[mx,:,2],rewardorders[mx,:,2]]      # punish order
    end
    Fs ./= nmodels
    
    for gx in 1:2
        axs = ax[1,gx]
        histogram!(axs, vec(modalitypositions[:,:,:,gx]), bins=0:30, color=:purple, alpha=0.5, label="modality")
        histogram!(axs, vec(decisionpositions[:,:,:,gx]), bins=0:30, color=:darkorange, alpha=0.5, label="decision")

        axs = ax[2,gx]
        histogram!(axs,rewardweightsdistribution[gx],bins=-1:0.1:1)
    end
    
    colorspace = :RdBu   # :diverging_gkr_60_10_c40_n256
    for rp in 1:2
        M = Fs[:,:,rp]
        Mn = M^2

        axs = ax[rp,3]
        heatmap!(axs, M, color=colorspace, clim=limsabsmaxnegpos(M), yflip=true)
        axs = ax[rp,4]
        heatmap!(axs, Mn, color=colorspace, clim=limsabsmaxnegpos(Mn), yflip=true)
    end

    axs = ax[3,1]
    # scatter!(axs, vec(Wis[:,:,5]), vec(Wis[:,:,6]))
    # xlabel!(axs, "reward")
    # ylabel!(axs, "punish")



    # plot a histogram for each quadrant of the reward vs punish weights 2D space (flatten over models, like above in the scatter plot)
    histogram2d!(axs, vec(Wis[:,:,5]), vec(Wis[:,:,6]))


    â,r,pv = getcorrelation(vec(Wis[:,:,5]), vec(Wis[:,:,6]))
    @info "r=$(r) p=$(pv)"

    

    moffset = 6
    for mx in 2:4
        axs = ax[3,mx]
        scatter!(axs, vec(Wis[mx+moffset,:,5]), vec(Wis[mx+moffset,:,6]))
        xlabel!(axs, "reward")
    end


    # display(ax)



    # plot modality order  activities specific to each context in correct and error trials
    cm = cgrad([:red, :white, :green],[0,0.4,0.6,1])

    clims = [(-0.5,0.5), (-0.5,0.5), (-0.5,0.5), (-40,40)]
    axs = plot(layout=(4,8), size=(8*300,4*250), left_margin=30*Plots.px, top_margin=30*Plots.px, legend=false, foreground_color_legend=nothing, background_color_legend=nothing)
    for cx in 1:2, gx in 1:2, dx in 1:2, j=1:4
        ax = axs[(cx-1)*2+gx,dx+(j-1)*2]
        heatmap!(ax, Ho[timestamps,:,j,cx,gx,dx]', color=cm, clim=clims[j], yflip=true)         # 1 for modality in 3rd index
        vline!(ax, [Ts[2][begin],Ts[2][end],Ts[3][end]], color=:black, lw=2)
        title!(ax, ["modality","decision","reward","context"][j]*"\n"* ["correct","error"][dx] )
        ylabel!(ax, ["go","nogo"][gx]*"\n"*["visual","audio"][cx]*" context")

    end
    display(axs)
    # savefig(axs, joinpath(config[:modelresultspath],"analysis/", folder*"-contextinference-activitytraces.png") )

    


    # plot how the context representation flips on error trials after reward
    axs = plot(layout=(3,4),size=(4*300,3*250), left_margin=30*Plots.px, legend=false, foreground_color_legend=nothing, background_color_legend=nothing)
    j = 4     # context ordered neurons
    contextbestlast = 5
    ms = zeros(ntimecourse,2)
    me = zeros(ntimecourse,2)
    for cx in 1:2, gx in 1:2, dx in 1:2, flx in 1:2
        ax = axs[flx,(cx-1)*2+gx]
        if flx==1
            m = mean(Ho[:,1:contextbestlast,j,cx,gx,dx],dims=2)
            e = std(Ho[:,1:contextbestlast,j,cx,gx,dx],dims=2)./sqrt(contextbestlast)
            plot!(ax, m, ribbon=e, color=[:gold,:red][dx], lw=2, label=["correct","error"][dx])         # 1 for modality in 3rd index
        else
            m = mean(Ho[:,end-contextbestlast+1:end,j,cx,gx,dx],dims=2)
            e = std(Ho[:,end-contextbestlast+1:end,j,cx,gx,dx],dims=2)./sqrt(contextbestlast)
            plot!(ax, m, ribbon=e, ls=:dash, color=[:gold,:red][dx], lw=2, label=["correct","error"][dx])         # 1 for modality in 3rd index
        end
        ms[:,dx] += [1,-1][flx]*[1,-1][gx] * m / 8
        me[:,dx] += e/8

        ylims!(ax, clims[j])
        vline!(ax, [Ts[2][begin],Ts[2][end],Ts[3][end]], color=:grey, lw=2, label=nothing)
        hline!(ax, [0], color=:grey, label=nothing)
        xticks!(ax, [15,20,25,30])

        title!(ax, ["visual","audio"][cx]*" context "*["go","nogo"][gx])
        ylabel!(ax, ["v+ a- context cells", "v- a+ context cells"][flx])
        plot!(ax,  legend=:topleft, foreground_color_legend=nothing, background_color_legend=nothing)
    end

    me ./= sqrt(8)
    ax = axs[3,1]
    plot!(ax, ms, ribbon=me, color=[:gold :red], lw=2, label=["correct" "error"])         # 1 for modality in 3rd index
    plot!(ax, legend=:topleft, foreground_color_legend=nothing, background_color_legend=nothing)
    ylims!(ax, -10, 10)
    vline!(ax, [Ts[2][begin],Ts[2][end],Ts[3][end]], color=:grey, lw=2, label=nothing)
    hline!(ax, [0], color=:grey, label=nothing)
    xticks!(ax, [15,20,25,30])


    for i in 2:4 plot!(axs[3,i], axis=false) end


    display(axs)
    # savefig(axs, joinpath(config[:modelresultspath],"analysis/", folder*"-contextinference-errorcontextflip,best.png") )

end
















function analysernnortho(machineparameters)
    
    fillrange!(machineparameters) # fill if we want a range, between first and last

    # collect ids as strings
    modelids = machineparameters[:modelids]
    snapshots = lpad.(machineparameters[:snapshots],machineparameters[:ndigitsnepochs],"0")


    filenamebase = machineparameters[:modeltype]*"-s$(machineparameters[:nsequence])-h$(machineparameters[:nhidden])"
    pathbase = joinpath(config[:modelanalysispath],  "distances",filenamebase)

    
    # assess model sizes
    @load(joinpath(config[:modelanalysispath],  "distances", filenamebase*"-r"*modelids[1]*"-e"*snapshots[1]*"-mit,ortho.bson"), @__MODULE__,
          Ts, performance, mit, ortho, labelstask, colors)

    
    distance = ortho# eval(measure)
    nmodels = length(modelids)
    nsnaps = length(snapshots)
    ntimecourse = Ts[end][end]
    ntimepoints = machineparameters[:nsequence] * ntimecourse
    nfactors = size(distance,2)
    
    performances = zeros(nmodels,nsnaps,2,2,2)     # (models,nsnaps,context,go/nogo,congruency)
    distances = zeros(eltype(distance),nmodels,nsnaps,ntimecourse,nfactors,nfactors) # (nmodels,nsnaps,ntimecourse,nfactors,nneurons)

    for (ix,id) in enumerate(modelids)
        for (sx,sn) in enumerate(snapshots)
            @load( joinpath(config[:modelanalysispath],  "distances", filenamebase*"-r"*id*"-e"*sn*"-mit,ortho.bson"), @__MODULE__,
                           Ts, performance, mit, ortho, labelstask, colors )
            performances[ix,sx,:,:,:] = performance
            # @info "performance check" id,sn mean(performance)
            distance = ortho                  
            distances[ix,sx,:,:,:] = distance[end-ntimecourse+1:end,:,:]
        end
    end

    @info "size" nmodels nsnaps size(performances) size(distances) labelstask

    timecourse = 1:Ts[end][end]




    # change to absolute difference from 90
    # distances = - abs.(90 .- distances) .+ 90

    # find the worst performing congruency aspect over all contexts and go nogo stims
    perfs = minimum(reshape(performances,nmodels,nsnaps,:),dims=3)
    perflim = 0.0
    whatperf = "mini"
    # or calculate the average performance
    # perfs = mean(reshape(performances,nmodels,nsnaps,:),dims=3)
    # perflim = 0.75
    # whatperf = "mean"
    
    # find the orthogonality between specific variables for each model
    # varlist = [1,2,3,10]
    # varlist = [4,5,6,10]
    varlist = [1,2,3,4,5,6,7,8,9,10]
    
    @info "variables" labelstask[varlist]
    comparisons = [[1,2],[1,10],[2,10],[3,10],[1,3],[2,3]]
    comparisontimepoints = [Ts[2][1:1], Ts[2][1:1], Ts[2][1:1],  Ts[3], Ts[3], Ts[3] ]
    angles = zeros(nmodels,ntimecourse,length(comparisons))       # vis-aud, cx-vis, cx-aud, cx-dec, vis-dec, aud-dec


    perflim = 0.


    ax = plot(layout=(length(comparisons),2), size=(2*400,length(comparisons)*300), xgrid=false, ygrid=false,
                      left_margin = 30*Plots.px, bottom_margin = 30*Plots.px, legend=nothing)


    for vx in axes(comparisons,1)
        ix1 = varlist[comparisons[vx][1]]
        ix2 = varlist[comparisons[vx][2]]

        angles[:,:,vx] = distances[:,end,:,ix1,ix2]

        axs = ax[vx,1]
        for ix in axes(distances,1) # go over each model
            m = distances[ix,end,:,ix1,ix2]      # take the last snapshot for this display
            plot!(axs,timecourse,m,color=:crimson,alpha=0.2)
        end
        plot!(axs,timecourse,dropdims(mean(distances[:,end,:,ix1,ix2],dims=1),dims=1),color=:crimson,lw=2)

        hline!(axs,[90],color=:grey,alpha=0.5)
        ylims!(axs,60,120)
        xlabel!(axs,"timecourse")
        ylabel!(axs,"angle")
        title!(axs,"$(labelstask[ix1]) vs $(labelstask[ix2])\nall models")
        plot!(axs,left_margin=50*Plots.px)


        
        axs = ax[vx,2]
        for ix in axes(distances,1) # go over each model
            # m = abs.(distances[ix,:,ix1,ix2] .- 90)
            m = distances[ix,:,:,ix1,ix2] # draw each snapshot with a connected lineplot
            # violin!(axs, [perfs[ix,1]], m, color=:crimson, bar_width=0.01)
            mm = mean(m[:,comparisontimepoints[vx]],dims=2) # average over timepoints for each comparison
            # scatter!(axs, [perfs[ix,1]], [mm])
            plot!(axs, perfs[ix,:,1], mm, markershape=:circle, markerstrokewidth=0)
        end
        plot!(axs,left_margin=60*Plots.px)
        
        # define correlation test
        m = dropdims(mean(reshape(distances[:,:,comparisontimepoints[vx],ix1,ix2],:,length(comparisontimepoints[vx])),dims=2),dims=2)
        pf = vec(perfs[:,:,1])
        â,r,pv = getcorrelation(pf, m)
        r = round(r,digits=4)
        pv = round(pv,digits=8)
        plot!(axs, [perflim,1.], â[1].*[perflim,1.].+â[2], color=:white)
        annotate!(axs, perflim+(1-perflim)/2, 76, "r=$(r) p=$(pv)", fontsize=8)

        hline!(axs,[90],color=:grey,alpha=0.5)
        xlims!(axs,perflim,1)
        ylims!(axs,60,120)
        xlabel!(axs,"performance")
        ylabel!(axs,"angle")
        title!(axs,"$(labelstask[ix1]) vs $(labelstask[ix2])\nperformance")

    end
    
    plot!(ax, plot_title=filenamebase)

    display(ax)
    savefig(ax, joinpath(config[:modelanalysispath], filenamebase*"-$(whatperf)perf-ortho.png"))

    validmodels = mean(performances[:,end,:,:,:],dims=(2,3,4))[:,1,1,1] .> 0.9
    @save(joinpath(config[:modelanalysispath],  "distances", filenamebase*"-angles.bson"), angles, validmodels)

end











"""
load models from machineparameters, and for each model,
plot the activity of the hidden units over time for the last trial
for each hidden unit, plot the activity in visual and audio contexts
"""
function tracesuppression(X, y, Ts, machineparameters)

    # data
    Xs, ys, sequencelength, ntimepoints, ntimecourse, ntrials,
        decisionpoints, decisionpoint, rewardprofile = preparedata(X, y, Ts)
    nhidden = machineparameters[:nhidden]

    # input models
    fillrange!(machineparameters)
    nmodels = length(machineparameters[:modelids])
    filenamebase = machineparameters[:modeltype]*"-s$sequencelength-h$nhidden"

    # number of bests to display on plots
    nbest = 5
    bestdisplayindices = collect((nbest:-1:1)')./nbest



    if config[:backtracerecalculate]

        # masks
        maskcontext, maskvisualgo, maskaudiogo, maskgo = getmasks(X[decisionpoint,[1,3],:])
        maskcongruency = getcongruency1d(X[decisionpoint,[1,3],:])




        # storage array for display; only save the mean, as within trial types activity has very small variance
        abstractcells = zeros(nmodels, ntimecourse, nbest, 2, 2, 2)      # (nmodels, ntimecourse, nbest, gonogo cells, context, stimulus)
        performances = zeros(nmodels, 2, 2, 2)  # (nmodels, context, go/nogo, congruency)

        folder = machineparameters[:modeltype]*"-s$sequencelength-h$nhidden"
        for (mx,modelid) in enumerate(machineparameters[:modelids])
            # load the model
            filename = filenamebase*"-r"*modelid
            @info "$filename"

            # load precalculated model performance
            sn = lpad(machineparameters[:snapshots][end],machineparameters[:ndigitsnepochs],"0")
            @load( joinpath(config[:modelanalysispath],  "distances", filename*"-e"*sn*"-mit,ortho.bson"),
                @__MODULE__, performance )
            performances[mx,:,:,:] = performance

            # load weights and activities
            Hs,Os,Ds,Rs,Wis,Whs,Wos,L,l = learnrnn( Xs, ys, decisionpoints, rewardprofile, machineparameters, train=false,
                                                filename=joinpath(config[:rawmodelpathprefix],config[:modelspath],folder,filename) )
            # view into the last learned snapshot
            H = Hs[end,:,:,:]         # ntimepoints, nhidden, ntrials
            Wi = Wis[end,:,:]         # nhidden, nhidden
            Wh = Whs[end,:,:]         # nhidden, nhidden
            Wo = Wos[end,:,:]         # noutput, nhidden

            # target and output for the last trial in the sequence at decisionpoint
            target = y[decisionpoint,:,:]
            output = Os[end,decisionpoint,:,:]     # response from the last snapshot (epoch) at the decisionpoint




            # select high stimulus input drive neurons
            # while simultaneously inactive for the other modality
            visualgoorder = intersect(sortperm(Wi[:,1],rev=true), sortperm(Wi[:,3])[1:nhidden÷2])
            visualnogoorder = intersect(sortperm(Wi[:,2],rev=true), sortperm(Wi[:,4])[1:nhidden÷2])
            audiogoorder = intersect(sortperm(Wi[:,3],rev=true), sortperm(Wi[:,1])[1:nhidden÷2])
            audionogoorder = intersect(sortperm(Wi[:,4],rev=true), sortperm(Wi[:,2])[1:nhidden÷2])
            bestvisualgounits = visualgoorder[1:nbest]
            bestvisualnogounits = visualnogoorder[1:nbest]
            bestaudiogounits = audiogoorder[1:nbest]
            bestaudionogounits = audionogoorder[1:nbest]
            labelsbestvisualgounits = map(a->"$a",bestvisualgounits[:,:]') 
            labelsbestvisualnogounits = map(a->"$a",bestvisualnogounits[:,:]')
            labelsbestaudiogounits = map(a->"$a",bestaudiogounits[:,:]')
            labelsbestaudionogounits = map(a->"$a",bestaudionogounits[:,:]')




            # sort hidden units by how strongly they are active for go and nogo outputs
            gounits = sortperm(Wo[1,:],rev=true)
            bestgounits = gounits[1:nbest]
            nogounits = sortperm(Wo[2,:],rev=true)
            bestnogounits = nogounits[1:nbest]
            labelsgounits = map(a->"$a",bestgounits[:,:]')
            labelsnogounits = map(a->"$a",bestnogounits[:,:]')
            bestgonogounits = unique(vcat(bestgounits,bestnogounits))


            # now select the top best and negative best recurrent inputs to each top best go and nogo units
            gorecurrentorder = collect(hcat([sortperm(Wh[u,:],rev=true) for u in bestgounits]...)')
            bestgorecurrentunits = vec(gorecurrentorder[:,1:nbest])
            negbestgorecurrentunits = vec(gorecurrentorder[:,end-nbest+1:end])
            nogorecurrentorder = collect(hcat([sortperm(Wh[u,:],rev=true) for u in bestnogounits]...)')
            bestnogorecurrentunits = vec(nogorecurrentorder[:,1:nbest])
            negbestnogorecurrentunits = vec(nogorecurrentorder[:,end-nbest+1:end])
            labelsbestgorecurrentunits = map(a->"$a",bestgorecurrentunits[:,:]')
            labelsbestnogorecurrentunits = map(a->"$a",bestnogorecurrentunits[:,:]')
            labelsnegbestgorecurrentunits = map(a->"$a",negbestgorecurrentunits[:,:]')
            labelsnegbestnogorecurrentunits = map(a->"$a",negbestnogorecurrentunits[:,:]')


            # now select the top best and negative best reucrrent inputs for each strong visual and audio input units
            visualgorecurrentorder = collect(hcat([sortperm(Wh[u,:],rev=true) for u in bestvisualgounits]...)')
            bestvisualgorecurrentunits = vec(visualgorecurrentorder[:,1:nbest])
            negbestvisualgorecurrentunits = vec(visualgorecurrentorder[:,end-nbest+1:end])
            visualnogorecurrentorder = collect(hcat([sortperm(Wh[u,:],rev=true) for u in bestvisualnogounits]...)')
            bestvisualnogorecurrentunits = vec(visualnogorecurrentorder[:,1:nbest])
            negbestvisualnogorecurrentunits = vec(visualnogorecurrentorder[:,end-nbest+1:end])
            audiogorecurrentorder = collect(hcat([sortperm(Wh[u,:],rev=true) for u in bestaudiogounits]...)')
            bestaudiogorecurrentunits = vec(audiogorecurrentorder[:,1:nbest])
            negbestaudiogorecurrentunits = vec(audiogorecurrentorder[:,end-nbest+1:end])
            audionogorecurrentorder = collect(hcat([sortperm(Wh[u,:],rev=true) for u in bestaudionogounits]...)')
            bestaudionogorecurrentunits = vec(audionogorecurrentorder[:,1:nbest])
            negbestaudionogorecurrentunits = vec(audionogorecurrentorder[:,end-nbest+1:end])
            labelsbestvisualgorecurrentunits = map(a->"$a",bestvisualgorecurrentunits[:,:]')
            labelsbestvisualnogorecurrentunits = map(a->"$a",bestvisualnogorecurrentunits[:,:]')
            labelsnegbestvisualgorecurrentunits = map(a->"$a",negbestvisualgorecurrentunits[:,:]')
            labelsnegbestvisualnogorecurrentunits = map(a->"$a",negbestvisualnogorecurrentunits[:,:]')
            labelsbestaudiogorecurrentunits = map(a->"$a",bestaudiogorecurrentunits[:,:]')
            labelsbestaudionogorecurrentunits = map(a->"$a",bestaudionogorecurrentunits[:,:]')
            labelsnegbestaudiogorecurrentunits = map(a->"$a",negbestaudiogorecurrentunits[:,:]')
            labelsnegbestaudionogorecurrentunits = map(a->"$a",negbestaudionogorecurrentunits[:,:]')

            # find high input units that have high recurrent weights to the best abstract go and nogo units
            bestvisualgorecurrentunits = intersect(bestvisualgounits,bestgorecurrentunits)
            negbestvisualgorecurrentunits = intersect(bestvisualgounits,negbestgorecurrentunits)
            bestaudiogorecurrentunits = intersect(bestaudiogounits,bestgorecurrentunits)
            negbestaudiogorecurrentunits = intersect(bestaudiogounits,negbestgorecurrentunits)
            # same for nogo:
            bestvisualnogorecurrentunits = intersect(bestvisualnogounits,bestnogorecurrentunits)
            negbestvisualnogorecurrentunits = intersect(bestvisualnogounits,negbestnogorecurrentunits)
            bestaudionogorecurrentunits = intersect(bestaudionogounits,bestnogorecurrentunits)
            negbestaudionogorecurrentunits = intersect(bestaudionogounits,negbestnogorecurrentunits)

            labelsbestvisualgorecurrentunits = map(a->"$a",bestvisualgorecurrentunits[:,:]')
            labelsbestvisualnogorecurrentunits = map(a->"$a",bestvisualnogorecurrentunits[:,:]')
            labelsbestaudiogorecurrentunits = map(a->"$a",bestaudiogorecurrentunits[:,:]')
            labelsbestaudionogorecurrentunits = map(a->"$a",bestaudionogorecurrentunits[:,:]')


            # @info "high input to abstract - visual" bestvisualgorecurrentunits negbestvisualgorecurrentunits bestvisualnogorecurrentunits negbestvisualnogorecurrentunits
            # @info "high input to abstract - audio" bestaudiogorecurrentunits  negbestaudiogorecurrentunits bestaudionogorecurrentunits  negbestaudionogorecurrentunits
            # return
            


            
        
            # select amongst the best and negbest units those that have the smallest absolute value output weights
            outputabsordersmall = sortperm(dropdims(sum(abs.(Wo),dims=1),dims=1))     # smallest sum of absolute weights
            bestcontextrange = nhidden-2*nbest             # number of units to consider for context, excluding the go and nogo bests
            smalloutputunits = outputabsordersmall[1:min(length(outputabsordersmall),bestcontextrange)]
            bestcontextunits = intersect(smalloutputunits, bestgorecurrentunits )[1:nbest]
            negbestcontextunits = intersect(smalloutputunits, negbestgorecurrentunits)[1:nbest]
            labelsbestcontextunits = map(a->"$a",bestcontextunits[:,:]')
            labelsnegbestcontextunits = map(a->"$a",negbestcontextunits[:,:]')

            inputabsorderlarge = sortperm(dropdims(sum(abs.(Wi[:,1:4]),dims=2),dims=2),rev=true)     # largest sum of absolute weights
            largeinputunits = inputabsorderlarge[1:min(length(inputabsorderlarge),bestcontextrange)]
            bestcontextinputunits = intersect(smalloutputunits, largeinputunits)[1:nbest]
            # @info "units" bestcontextunits, bestcontextinputunits

            
            labelsgonogo = ["go","nogo"]
            labelsmodality = ["visual","audio"]


            


            
            # plot Wo two rows as two barcharts
            ax = plot(layout=(9,4), size=(4*300,9*200), xgrid=false, ygrid=false, foreground_color_legend=nothing,
                        left_margin = 30*Plots.px, bottom_margin = 30*Plots.px, top_margin = 30*Plots.px)

            annotate!(ax[1,3],-7, 1.5, "$filename")  # title



            axs = ax[1,1]
            bar!(axs,Wo[1,:], color=:darkgreen, alpha=0.7, label="go")
            bar!(axs,Wo[1,gounits], color=:lime, alpha=0.7, label="go goorder")
            ylabel!(axs,"outputweights")
            xlabel!(axs,"hidden units", legend=:top)
            annotate!(axs,-1, 1.7, "weights", :white)
            
            axs = ax[1,2]
            bar!(axs,Wo[2,:], color=:darkred, alpha=0.7, label="nogo")
            bar!(axs,Wo[2,gounits], color=:red, alpha=0.7, label="nogo goorder")
            # place legend in outer top center
            xlabel!(axs,"hidden units", legend=:top)

            

            # heatmap of Wo
            # axs = ax[1,3]
            # heatmap!(axs,Wo, color=:viridis, clims=(-1.5,1.5), aspect_ratio=1, showaxis=false, colorbar=false, legend=false)

            
            # input weights
            for (sgx,(order, omul, color)) in enumerate(zip((visualgoorder, visualnogoorder, audiogoorder, audionogoorder),(1,-1,1,-1),
                                                            (:blue,:purple,:green,:orange)))
                rx = (sgx-1)÷2+1
                alx = (omul+1)÷2+1
                axs = ax[1,2+rx]
                d = [Wi[order,sgx], reverse(Wi[order,sgx])][alx]
                bar!(axs, d, color=color, alpha=0.7, label=labelsgonogo[3-alx], legend=:top)
                if sgx==1 ylabel!(axs, "input weights") end
                title!(axs,labelsmodality[rx])
            end



            # for the first and last 3 units in goorder, plot the activity in visual and audio contexts
            # over the timecourse of the last trial in the sequence by H[end-timecourse:end,unit,trial mask] for each context
            # using the appropriate mask

            #  context/relevancy, gonogo cells+trials, stimulusmodality
            for (bx,bestunits) in enumerate(([bestgounits,bestnogounits],
                                            [bestvisualgorecurrentunits, bestvisualnogorecurrentunits],
                                            [bestaudiogorecurrentunits, bestaudionogorecurrentunits],
                                            [bestcontextunits, negbestcontextunits]))
                for cx in 1:2, gx in 1:2, sx in 1:2             # context, go/nogo cells, visual/audio
                    axs = ax[1+cx+(bx-1)*2,gx+(sx-1)*2]
                    vspan!(axs,[Ts[2][begin],Ts[3][end]], color=:grey, alpha=0.3, label=nothing)
                    vline!(axs,[decisionpoints[1]],color=:black,lw=2, label=nothing)
                    hline!(axs,[0],color=:grey, ls=:dash, label=nothing)

                    mask = [
                            [maskvisualgo,.!maskvisualgo][gx],
                            [maskaudiogo,.!maskaudiogo][gx]
                        ][sx]
                    mask = mask .& (maskcontext.==[true,false][cx]) .& (.! maskcongruency)

                    P = H[end-ntimecourse+1:end,bestunits[gx],mask]
                    m = dropdims(mean(P,dims=3), dims=3)
                    abstractcells[mx,:,1:size(m,2),gx,cx,sx] = m     # save the mean (nmodels, ntimecourse, nbest, gonogo cells, context, stimulus)
                    e = dropdims(std(P,dims=3), dims=3) ./ sqrt(sum(mask))

                    color = [:dodgerblue, :lime][sx]
                    label = [ [labelsgounits,labelsnogounits][gx],
                            [labelsbestvisualgorecurrentunits, labelsbestvisualnogorecurrentunits][gx],
                            [labelsbestaudiogorecurrentunits, labelsbestaudionogorecurrentunits][gx],
                            [labelsbestcontextunits, labelsnegbestcontextunits][gx]
                            ][bx]
                    plot!(axs, 1:ntimecourse, m, ribbon=e,
                        color=color, lw=2, alpha=bestdisplayindices,
                        label=label)

                    ylims!(axs,-1,1)
                    
                    if cx==1
                        title = labelsmodality[sx]*" "*labelsgonogo[gx]*" trials\n"
                        title = title * [ ["go cells","nogo cells"][gx],
                                        ["go visual input cells","nogo visual input cells"][gx],
                                        ["go audio input cells","nogo audio input cells"][gx],
                                        ["positive rec.","negative rec."][gx]*" context cells",
                                        ][bx]

                        title!(axs,title)
                    end
                    if gx+sx==2
                        ylabel!(axs,labelsmodality[cx]*" context")
                        if cx==1
                            annotate!(axs,-1, 1.7, (["abstract cells","visual input cells","audio input cells","context input+output cells"][bx],
                                    [:gold,:blue,:green,:mediumvioletred][bx], :left))
                        end
                    end
                end
            end





            # display(ax)



        end

        @save(joinpath(config[:modelanalysispath], "suppression", filenamebase*"-suppressiontraces-all.bson"), abstractcells, performances)
   
    else 


        @load(joinpath(config[:modelanalysispath], "suppression", filenamebase*"-suppressiontraces-all.bson"), @__MODULE__, abstractcells, performances)
        @assert(size(abstractcells,3)==nbest, "nbest mismatch")
        
    
    end





    # plot the abstract go and nogo cells averaged over models
    ax = plot(layout=(3,4), size=(4*300,3*200), xgrid=false, ygrid=false, legend=false, foreground_color_legend=nothing,
                left_margin = 30*Plots.px, bottom_margin = 50*Plots.px, top_margin = 30*Plots.px)

    labelstimulus = ["visual","audio"]
    colors = [:dodgerblue, :lime]

    # find the abstract go and nogo cells with the highest enhancement - suppression
    h = abstractcells[:,end-ntimecourse+1:end,:,:,:,:]
    es = h[:,Ts[3][begin],:,:,1,1] + h[:,Ts[3][begin],:,:,2,2] - h[:,Ts[3][begin],:,:,1,2] - h[:,Ts[3][begin],:,:,2,1]
    s = argmax(es,dims=2)

    for (cx,context) in enumerate(["visual","audio"])
        for (gx,gonogo) in enumerate(["go","nogo"])
            for (sx,stimulus) in enumerate(labelstimulus)
                
                axs = ax[cx,(sx-1)*2+gx]
                
                vspan!(axs,[Ts[2][begin],Ts[3][end]], color=:grey, alpha=0.3, label=nothing)
                vline!(axs,[decisionpoints[1]],color=:black,lw=2, label=nothing)
                hline!(axs,[0],color=:grey, ls=:dash, label=nothing)
                
                d = similar(h[:,:,1,gx,cx,sx])
                for k in 1:nmodels   d[k,:] = h[k,:,s[k,1,gx][2],gx,cx,sx]   end
                m = dropdims(mean(d,dims=(1)), dims=(1))
                e = dropdims(std(d,dims=(1)), dims=(1)) ./ sqrt(nmodels)
                # for n in 1:nmodels
                #     plot!(axs, 1:ntimecourse, abstractcells[n,end-ntimecourse+1:end,:,gx,cx,sx], color=:white, lw=1, alpha=0.5)
                # end
                plot!(axs, 1:ntimecourse, m, ribbon=e,color=colors[sx], lw=2, alpha=bestdisplayindices)

                if cx==1 title!(axs,gonogo*" cells"*"\n"*labelstimulus[sx]*" trials") end
                if gx+sx==2 ylabel!(axs,context*" context") end
                ylims!(axs,-0.5,0.5)
            end

        end
    end

    # plot    enhancement - suppression   vs    performance
    # performances dimensions: (nmodels, context, go/nogo, congruency)
    p = minimum(performances[:,:,:,2],dims=(2,3))[:,1,1] # incongruent only
    for (gx,gonogo) in enumerate(["go","nogo"])
        axs = ax[3,gx]
        scatter!(axs, p, es[s[:,1,gx]], color=[:white,:red][gx], markerstrokewidth=0 )
        
        
        â,r,pv = getcorrelation(p, es[s[:,1,gx]]) # this does not work yet in MathUtils
        plot!(axs, [0,1.], â[1].*[0,1.].+â[2], color=:gold, lw=1.5, alpha=0.8)
        r = round(r,digits=4)
        pv = round(pv,digits=8)
        annotate!(axs, 0.1, 3, "r=$(r) p=$(pv)", font(pointsize=8, color=:gold, halign=:left))
        
        xlabel!(axs,"performance")
        if gx==1 ylabel!(axs,"enhancement - suppression") end
        title!(axs,gonogo*" cells")

    end

    plot!(ax[3,3], axis=false)
    plot!(ax[3,4], axis=false)

    display(ax)

    savefig(joinpath(config[:modelanalysispath], filenamebase*"-suppressiontraces-all.png"))
end



function contextrepresentation(X, y, Ts, machineparameters)
    # data
    Xs, ys, sequencelength, ntimepoints, ntimecourse, ntrials,
        decisionpoints, decisionpoint, rewardprofile = preparedata(X, y, Ts)
    nhidden = machineparameters[:nhidden]
    noutput = size(y,2)
    ninput = size(X,2)
    timestamps = Ts[1][1]:Ts[end][end] .+ ntimepoints .- Ts[end][end]
    timestamps = 61:75
    labelstim = ["visual","audio"]
    labelinstr = ["go","nogo"]
    labelcontextproj = ["pre","start","decision"]

    filenamebase = machineparameters[:modeltype]*"-s$sequencelength-h$nhidden"

    # load context decoders for each model
    @load( joinpath(config[:modelanalysispath],  "contextrepresentation", "contextprojections-rnn.bson"), @__MODULE__, CRs, CRPs, SRs, SRPs, validmodels)
    # validmodels = validmodels .& rand([true, false], size(validmodels))
    CRs = CRs[validmodels,:,:,:]
    CRPs = CRPs[validmodels,:,:,:,:]    # nmodels, ncontexts, ngonogo, ntimepoints, nprojection
    SRs = SRs[validmodels,:,:,:]
    SRPs = SRPs[validmodels,:,:,:,:]    # nmodels, stimulus, ncontexts, ngonogo, ntimepoints
    nmodels = sum(validmodels)



    # get masks
    maskcontext, maskvisualgo, maskaudiogo, maskgo = getmasks(X[decisionpoint,[1,3],:])


    axs = plot(layout=(5,7), size=(7*350,5*300), xgrid=false, ygrid=false, legend=false,
                left_margin = 20*Plots.px, bottom_margin = 20*Plots.px, top_margin = 20*Plots.px)

    maskcontexts = [maskcontext, .! maskcontext ]
    maskgos = [maskgo, .! maskgo ]

    # context

    for ct in 1:3
        for gx in 1:2, cx in 1:2
            ax = axs[ct,(cx-1)*2+gx]
            m = CRPs[:,cx,gx,timestamps,ct]
            # plot!(ax, m', color=[:blue,:green][cx], ls=[:solid, :dash][gx], alpha=0.2)
            plot!(ax, mean(m,dims=1)', ribbon=std(m,dims=1)'/sqrt(nmodels), color=[:blue,:green][cx], ls=[:solid, :dash][gx], lw=2, fillalpha=0.3)
            @nolinebackground(ax,Ts[2][1],Ts[3][end],Ts[3][1],bg=:black)
            hline!(ax,[0],color=:grey, ls=:dash, label=nothing)
            vline!(ax,[[Ts[1][2],Ts[2][1],Ts[3][1]][ct]],color=:purple,label=nothing)
            ylims!(ax,-0.5,0.5)
            if cx+gx==2 ylabel!(ax,"$(labelcontextproj[ct])\ncontext proj") end
            if ct==1 title!(ax,"$(labelstim[cx]) context $(labelinstr[gx])" ) end
        end


        for (dx,m) in enumerate(( mean(CRPs[:,:,:,timestamps,ct],dims=(2,3))[:,1,1,:],
                                #   mean(CRPs[:,1,:,timestamps,ct] - CRPs[:,2,:,timestamps,ct], dims=2)[:,1,:] ))
                                  CRPs[:,1,1,timestamps,ct] - CRPs[:,2,1,timestamps,ct],
                                  CRPs[:,1,2,timestamps,ct] - CRPs[:,2,2,timestamps,ct]) )
            ax = axs[ct,4+dx]
        # m = mean(CRPs[:,:,:,timestamps,ct],dims=(2,3))[:,1,1,:]
        # m = mean(abs.(CRPs[:,1,:,timestamps,ct] - CRPs[:,2,:,timestamps,ct]),dims=(2))[:,1,:]
        # m = (CRPs[:,1,gx,timestamps,ct] - CRPs[:,2,gx,timestamps,ct])
        # plot!(ax, m', color=:mediumvioletred, alpha=0.2)
            plot!(ax, mean(m,dims=1)', ribbon=std(m,dims=1)'/sqrt(nmodels), color=:mediumvioletred, lw=2, fillalpha=0.3)
            @nolinebackground(ax,Ts[2][1],Ts[3][end],Ts[3][1],bg=:black)
            hline!(ax,[0],color=:grey, ls=:dash, label=nothing)
            vline!(ax,[[Ts[1][2],Ts[2][1],Ts[3][1]][ct]],color=:purple,label=nothing)
            # ylims!(ax,-0.15,0.15) 
            if dx==1 ylims!(ax,-0.15,0.15) else ylims!(ax,-0.5,0.5) end
            if ct==1 title!(ax,["mean context","diff context go", "diff context nogo"][dx] ) end
        end
    end

    # stimulus
    colors = [:deepskyblue :blue; :lime :green]
    for sx in 1:2, rx in 1:2
        cx = 2 - (sx==rx)
        for gx in 1:2
            ax = axs[3+sx,(cx-1)*2+gx]
            m = SRPs[:,sx,cx,gx,timestamps]
            # plot!(ax, m', color=colors[sx,rx], ls=[:solid, :dash][gx], alpha=0.2)
            plot!(ax, mean(m,dims=1)', ribbon=std(m,dims=1)'/sqrt(nmodels), color=colors[sx,rx], ls=[:solid, :dash][gx], lw=2, fillalpha=0.3)
            @nolinebackground(ax,Ts[2][1],Ts[3][end],Ts[3][1],bg=:black)
            hline!(ax,[0],color=:grey, ls=:dash, label=nothing)
            ylims!(ax,-1.5,1.5)
            if cx+gx==2 ylabel!(ax,"$(labelstim[sx])\nstimulus proj") end
            title!(ax,["","$(labelstim[cx]) context\n"][1 + (sx==1)]*["relevant","irrelevant"][rx]*" $(labelinstr[gx])" )
        end
    end
    # for sx in 1:2
    #     ax = axs[3+sx,5]
    #     m = mean(SRPs[:,sx,:,:,timestamps], dims=(2,3))[:,1,1,:]
    #     @info "$sx" m size(m)
    #     plot!(ax, mean(m,dims=1)', ribbon=std(m,dims=1)'/sqrt(nmodels), color=colors[sx,1], lw=2, fillalpha=0.3)
    #     ylims!(ax,-1.5,1.5)
    # end
    for dx in 5:7, sx in 1:2  plot!(axs[3+sx,dx], axis=false) end



    display(axs)
end















function subspacesrnn(X, y, Ts, machineparameters)

    # data
    Xs, ys, sequencelength, ntimepoints, ntimecourse, ntrials,
        decisionpoints, decisionpoint, rewardprofile = preparedata(X, y, Ts)
    nhidden = machineparameters[:nhidden]
    noutput = size(y,2)
    ninput = size(X,2)

    # input models
    fillrange!(machineparameters)
    nmodels = length(machineparameters[:modelids])
    filenamebase = machineparameters[:modeltype]*"-s$sequencelength-h$nhidden"

    # number of bests to display on plots
    nbest = 5
    bestdisplayindices = collect((nbest:-1:1)')./nbest



    if config[:subspacerecalculate]

        # masks
        maskcontext, maskvisualgo, maskaudiogo, maskgo = getmasks(X[decisionpoint,[1,3],:])
        maskcongruency = getcongruency1d(X[decisionpoint,[1,3],:])




        # storage array for display; only save the mean, as within trial types activity has very small variance
        abstractcells = zeros(nmodels, ntimecourse, nbest, 2, 2, 2)      # (nmodels, ntimecourse, nbest, gonogo cells, context, stimulus)
        performances = zeros(nmodels, 2, 2, 2)  # (nmodels, context, go/nogo, congruency)

        allFgo = zeros(nhidden,nhidden) # all ordered F
        allFnogo = zeros(nhidden,nhidden)
        allFmg = zeros(nhidden,nhidden) # all ordered F based on input, go only
        allFmn = zeros(nhidden,nhidden) # all ordered F based on input, nogo only
        allmgg = zeros(nhidden,2)
        allmng = zeros(nhidden,2)
        
        allCim = zeros(nhidden,4,2)             # four timepoints, go nogo
        allCign = zeros(nhidden,2)
        allCcm = zeros(nhidden,2)
        allCcgn = zeros(nhidden,2)
        
        # all model stores
        Fmns = zeros(nmodels,nhidden,nhidden)
        Fmgs = zeros(nmodels,nhidden,nhidden)
        DWis = zeros(nmodels,nhidden,ninput)
        DWhs = zeros(nmodels,nhidden,nhidden)
        DWos = zeros(nmodels,noutput,nhidden)
        Mis = zeros(nmodels,nhidden,2)
        Cis = zeros(nmodels,nhidden,4,2)         # four timepoints, go nogo
        Cims = zeros(nmodels,nhidden,4,2)         # four timepoints, go nogo
        Ccms = zeros(nmodels,nhidden,2)
        CRs = zeros(nmodels,2,ntimepoints,nhidden+1)
        CRPs = zeros(nmodels,2,2,ntimepoints,3)    # context, gonogo, ...., (pre, at start, and at decision projections)
        SRs = zeros(nmodels,2,2,nhidden)
        SRPs = zeros(nmodels,2,2,2,ntimepoints)
        Dis = zeros(nmodels,nhidden,2)
        Ris = zeros(nmodels,nhidden,2)
        Rims = zeros(nmodels,nhidden,2)
        Rigns = zeros(nmodels,nhidden,2,2)

        nmodels = 0
        validmodels = trues(length(machineparameters[:modelids]))
        folder = machineparameters[:modeltype]*"-s$sequencelength-h$nhidden"
        for (mx,modelid) in enumerate(machineparameters[:modelids])
            # load the model
            filename = filenamebase*"-r"*modelid
            # load precalculated model performance
            sn = lpad(machineparameters[:snapshots][end],machineparameters[:ndigitsnepochs],"0")
            @load( joinpath(config[:modelanalysispath],  "distances", filename*"-e"*sn*"-mit,ortho.bson"),
                @__MODULE__, performance )
            performances[mx,:,:,:] = performance

            # exclude bad models
            if mean(performance)<0.9
                @info("skip")
                validmodels[mx] = false
                continue
            end
            @info "$filename"          # dsplay good models
            nmodels += 1


            # load weights and activities
            Hs,Os,Ds,Rs,Wis,Whs,Wos,L,l = learnrnn( Xs, ys, decisionpoints, rewardprofile, machineparameters, train=false,
                                                filename=joinpath(config[:rawmodelpathprefix],config[:modelspath],folder,filename) )
            # view into the last learned snapshot
            H = Hs[end,:,:,:]         # ntimepoints, nhidden, ntrials
            Wi = Wis[end,:,:]         # nhidden, ninput
            Wh = Whs[end,:,:]         # nhidden, nhidden
            Wo = Wos[end,:,:]         # noutput, nhidden

            

            # get sort order of output weights
            gounits = sortperm(Wo[1,:],rev=true)
            nogounits = sortperm(Wo[2,:],rev=true)

            # get sort order for input weights
            # create an index that tracks if the unit is visual (+, begin) or auditory (-, end)
            # separate for go and nogo in 2nd dimension
            modalitiesindex = (Wi[:,1:2]) - (Wi[:,3:4])
            modalitygounits = sortperm(modalitiesindex[:,1],rev=true)
            modalitynogounits = sortperm(modalitiesindex[:,2],rev=true)
            modalityunits = [ modalitygounits modalitynogounits ]

            # get reward index: (separate for the go and the nogo reward)
            rewardindex = Wi[:,5:6]



            # get the index that shows the largest difference between contexts
            # in the last trial in incongruent trials, visual context is stronger: +, audio context is stronger: -
            contextindex = zeros(nhidden,4,2)   # nhidden, timepoint: (pre, start, dec, all), go/nogo
            # modeltimestampscontext = ntimepoints - Ts[end][end]  .+ vcat(Ts[begin],Ts[2][1:1])        # only off stimulus   # original: end-Ts[end][end]+1:end
            # modeltimestampscontext = ntimepoints - Ts[end][end]  .+ vcat(Ts[2][1:1])        # only off stimulus   # original: end-Ts[end][end]+1:end
            # modeltimestampscontext = ntimepoints - Ts[end][end]  .+ vcat(Ts[begin])        # only off stimulus   # original: end-Ts[end][end]+1:end
            # modeltimestampscontext = vcat([  vcat(Ts[begin],Ts[end]) .+ (i-1)*Ts[end][end] for i in 1:sequencelength]...)
            modeltimestampscontext = [ntimepoints - Ts[end][end]  .+ vcat(Ts[1]),
                                      ntimepoints - Ts[end][end]  .+ vcat(Ts[2][1:1]),
                                      ntimepoints - Ts[end][end]  .+ vcat(Ts[2][end:end]),
                                      ntimepoints - Ts[end][end]  .+ vcat(Ts...) ]
            for px in 1:4
                for (mskx,maskdir) in enumerate((maskgo, .! maskgo))
                    contextindex[:,px,mskx] = dropdims(mean(
                                    H[modeltimestampscontext[px], :, maskdir  .&     maskcontext  ] -  # .&  ( maskcongruency)] -
                                    H[modeltimestampscontext[px], :, maskdir  .& (.! maskcontext) ],      # .&  ( maskcongruency)],
                                                dims=(1,3)),dims=(1,3))
                end
            end
            # contextunits = sortperm(contextindex[:,2,1],rev=true)

            contextcoefs = zeros(nhidden,2)
            Hp = Float64.(permutedims(H[modeltimestampscontext[2],:,:],[3,1,2]))
            for (mskx,maskdir) in enumerate((maskgo, .! maskgo))
                _,accuracies,coefs = decodevariablefromactivity(Hp,Float64.(maskdir .& maskcontext))
                contextcoefs[:,mskx] = dropdims(mean(coefs,dims=1),dims=1)[1:nhidden,1]
            end

            # continuous context representation
            Hp = Float64.(permutedims(H,[3,1,2]))
            for (ct,contextprojectiontimestamps) in enumerate([Ts[1],Ts[2][1:1],Ts[3][1:1]])         # pre   start    decision
                for gx in 1:2                         # go   nogo
                    maskdir = [maskgo, .! maskgo][gx]
                    _,_,coefs = decodevariablefromactivity(Hp[maskdir,:,:], Float64.(maskcontext[maskdir]))
                    CRs[mx,gx,:,:] = coefs[:,:,1]
                    P = dropdims(mean(CRs[mx,gx,contextprojectiontimestamps,1:nhidden],dims=1),dims=1)
                    for cx in 1:2         # contexts
                        CRPs[mx,cx,gx,:,ct] = mean(projectontoaxis(Hp[maskdir .& [maskcontext, .! maskcontext][cx],:,:], P / norm(P)), dims=1)[1,:]
                    end
                end
            end

            # projections to visual and auditory cells
            for sx in 1:2
                for gx in 1:2
                    SRs[mx,sx,gx,:] = Wi[:,[0,2][sx] + gx]
                    P = SRs[mx,sx,gx,:]         # no timecourse for input weights
                    for cx in 1:2
                        SRPs[mx,sx,cx,gx,:] = mean(projectontoaxis(Hp[[maskcontext, .! maskcontext][cx] .& [maskgo, .! maskgo][gx],:,:], P / norm(P)), dims=1)[1,:]
                    end
                end
            end



            # decision index
            decisionindex = [ Wo[1,:] Wo[2,:] ]


            allmgg += Wo[:,modalitygounits]'
            allmng += Wo[:,modalitynogounits]'


            # order hidden units connections
            # sort by output go preference
            # D = [ Wo[:,gounits] Wo[:,nogounits]]
            Fgo = Wh[gounits,gounits]
            Fnogo = Wh[nogounits,nogounits]
            Cign = [ contextindex[gounits] contextindex[nogounits] ]
            Ccgn = [ contextcoefs[gounits] contextcoefs[nogounits] ] ./ maximum(contextcoefs)
            Rign = [ rewardindex[gounits,1] rewardindex[nogounits,1];;; rewardindex[gounits,2] rewardindex[nogounits,2] ]
            # sort by modality prefrerence index
            Fmg = Wh[modalitygounits,modalitygounits]
            Fmn = Wh[modalitynogounits,modalitynogounits]
            Cim = cat(contextindex[modalitygounits,:,1], contextindex[modalitynogounits,:,2],dims=3)     #  go index in modality goorder
            Ccm = [contextcoefs[modalitygounits,1] contextcoefs[modalitynogounits,2]] ./ maximum(contextcoefs,dims=1)
            Rim = [ rewardindex[modalitygounits,1] rewardindex[modalitynogounits,2]]
            
            Fmgs[mx,:,:] = Fmg
            Fmns[mx,:,:] = Fmn
            DWis[mx,:,:] = Wi
            DWhs[mx,:,:] = Wh
            DWos[mx,:,:] = Wo
            Cims[mx,:,:,:] = Cim
            Ccms[mx,:,:] = Ccm
            Mis[mx,:,:] = modalitiesindex
            Cis[mx,:,:,:] = contextindex
            Dis[mx,:,:] = decisionindex
            Ris[mx,:,:] = rewardindex
            Rims[mx,:,:] = Rim
            Rigns[mx,:,:,:] = Rign


            allFgo += Fgo
            allFnogo += Fnogo
            allFmg += Fmg
            allFmn += Fmn
            allCim += Cim
            allCign += Cign
            allCcm += Ccm
            allCcgn += Ccgn




        end
        allFgo, allFnogo, allFmg, allFmn, allCim, allCign, allCcm, allCcgn =
           allFgo/nmodels, allFnogo/nmodels, allFmg/nmodels, allFmn/nmodels, allCim/nmodels, allCign/nmodels, allCcm/nmodels, allCcgn/nmodels

        @save( joinpath(config[:modelanalysispath],  "suppression", "subspaces-rnn-reduced.bson"),
            abstractcells, performances, allFgo, allFnogo, allFmg, allFmn, allmgg, allmng, allCim, allCign, allCcm, allCcgn)

        @save( joinpath(config[:modelanalysispath],  "suppression", "subspaces-rnn.bson"),
              Fmgs, Fmns,  Mis, Cis, Dis, Ris, Cims, Ccms, Rims, DWis, DWhs, DWos, Rigns, validmodels)

        @save( joinpath(config[:modelanalysispath],  "contextrepresentation", "contextprojections-rnn.bson"), CRs, CRPs, SRs, SRPs, validmodels)
        

    else

        @load( joinpath(config[:modelanalysispath],  "suppression", "subspaces-rnn-reduced.bson"), @__MODULE__,
            abstractcells, performances, allFgo, allFnogo, allFmg, allFmn, allmgg, allmng, allCim, allCign, allCcm, allCcgn)

    end

    return
    pal = :RdBu
        
    
    # axs = plot(layout=(4,3), size=(3*400,4*400))
    # for hx=1:3
    #     k = [1,5,50][hx]
    #     ax = axs[1,hx]
    #     heatmap!(ax, eFgo^k, color=pal, aspect_ratio=:equal, yflip=true)
    #     ax = axs[2,hx]
    #     heatmap!(ax, eFnogo^k, color=pal, aspect_ratio=:equal, yflip=true)
    #     ax = axs[3,hx]
    #     heatmap!(ax, allFgo^k, color=pal, aspect_ratio=:equal, yflip=true)
    #     ax = axs[4,hx]
    #     heatmap!(ax, allFnogo^k, color=pal, aspect_ratio=:equal, yflip=true)
    # end
    # display(axs)

    @info "norm" norm(allFmg,2) norm(allFmn,2) 1/norm(allFmg,2) 1/norm(allFmn,2)
    axs = plot(layout=(3,3), size=(3*400,3*400))
    for hx=1:3
        k = [1,2,3][hx]
        ax = axs[1,hx]
        M = nonlinearmatrixpower(tanh,allFmg,k,5)
        heatmap!(ax, M, color=pal, clim=limsabsmaxnegpos(M), aspect_ratio=:equal, yflip=true)
        ax = axs[2,hx]
        M = nonlinearmatrixpower(tanh,allFmn,k,5)
        heatmap!(ax, M, color=pal, clim=limsabsmaxnegpos(M), aspect_ratio=:equal, yflip=true)
    end
    ax = axs[3,1]
    plot!(ax, allmgg, color=[:white :red],label=["go index modalitygo order" "nogo index modalitygo order"])
    plot!(ax, allmng, color=[:darkgrey :firebrick],label=["go index modalitynogo order" "nogo index modalitynogo order"])
    ax = axs[3,2]
    plot!(ax, allCim, color=:lightseagreen,ls=[:solid :dashdot], label=["go context index modalitygo order" "nogo context index modalitynogo order" ])
    ax = axs[3,3]
    plot!(ax, allCcm, color=:mediumvioletred,ls=[:solid :dashdot], label=[" go context decoder modalitygo order" "nogo context decoder modalitynogo order"])
    plot!(ax, abs.(allCcm), color=:gold,ls=[:solid :dashdot],label="|context decoder modality order|")
    
    
    
    # this is not needed as it is confirmed to be invariant:
    # plot!(ax, allCign, color=[:white :red],label=["context index go order" "context index nogo order"])
    # plot!(ax, allCcgn, color=[:darkgrey :mediumvioletred],label=["context index go order" "context index nogo order"])
    
    display(axs)
    

end










function outcomehistoryrnn(X, y, Ts, machineparameters; nlookback=3)

    # data
    Xs, ys, sequencelength, ntimepoints, ntimecourse, ntrials,
        decisionpoints, decisionpoint, rewardprofile = preparedata(X, y, Ts)
    nhidden = machineparameters[:nhidden]

    # input models
    fillrange!(machineparameters)
    nmodels = length(machineparameters[:modelids])
    filenamebase = machineparameters[:modeltype]*"-s$sequencelength-h$nhidden"


    # target variables to test
    targetlist = [X[decisionpoints,1,:], X[decisionpoints,3,:], y[decisionpoints,1,:]]
    targetlist = vcat(reshape.(targetlist,1,size(targetlist[1])...)...)
    targetlist = convert.(Float64,targetlist)
    labels = ["visual","audio","decision"]
    cols = [1,2,3] # cols in targetlist
    colors = [:blue,:green,:darkorange]

    contexts = ["visual","audio"]


    accuracieslists = []
    timestamps = 1:ntimecourse

    folder = machineparameters[:modeltype]*"-s$sequencelength-h$nhidden"
    for (mx,modelid) in enumerate(machineparameters[:modelids])
        # load the model
        filename = filenamebase*"-r"*modelid
        @info "$filename"

        if config[:recalculaternnoutcomehistory]

            # load weights and activities
            Hs,Os,Ds,Rs,Wis,Whs,Wos,L,l = learnrnn( Xs, ys, decisionpoints, rewardprofile, machineparameters, train=false,
                                                filename=joinpath(config[:rawmodelpathprefix],config[:modelspath],folder,filename) )
            # view into the last learned snapshot
            H = convert.(Float64,Hs[end,end-ntimecourse+1:end,:,:])         # ntimepoints, nhidden, ntrials
            H = permutedims(H,[3,1,2])
            output = Os[end,decisionpoint,:,:]     # response from the last snapshot (epoch) at the decisionpoint

            # target and output for the last trial in the sequence at decisionpoint

            accuracieslist = zeros(length(contexts),length(cols),nlookback,ntimecourse,3)
            coefficientslist = zeros(length(contexts),length(cols),nlookback,ntimecourse,nhidden+1) # will hold the intercept as the last value

            contextindiceslist = [1:ntrials÷2, ntrials÷2+1:ntrials]
            for (cx,contextindices) in enumerate(contextindiceslist)

                # perform decoding of all variables and all histories

                for v in eachindex(cols)
                    @info "decoding in $(contexts[cx]) context: $(labels[v]) $nlookback-length history"
                    for h in 1:nlookback
                        _, accuracieslist[cx,v,h,:,:], coefficientslist[cx,v,h,:,:] =
                          decodevariablefromactivity(H[contextindices,:,:], targetlist[v,end+h-nlookback,contextindices], nfolds=10)
                    end
                end
            end

            @save(joinpath(config[:modelanalysispath],"outcomehistory/","outcomehistory-$(string(modelid)).bson"), accuracieslist, coefficientslist)
        else
            @load(joinpath(config[:modelanalysispath],"outcomehistory/","outcomehistory-$(string(modelid)).bson"), @__MODULE__, accuracieslist, coefficientslist)
        end

        push!(accuracieslists, accuracieslist)
        
        
        # plot for individual models
        ax = plot(layout=(2,length(cols)),size=(1.2* length(cols)*300, 2*250), legend=false, left_margin=30*Plots.px)
        for cx in eachindex(contexts)
            for vx in eachindex(cols)
                axs = ax[cx,vx]
                @decoderbackground(axs, Ts[2][begin], Ts[3][end], Ts[3][begin])
                for h in nlookback:-1:1
                    m = accuracieslist[cx,vx,h,:,1]
                    e = accuracieslist[cx,vx,h,:,3]
                    plot!(axs, timestamps, m,
                            ribbon=e,
                            lw=[1,3,1][h], fillalpha=0.1, color=ifelse(h==nlookback,:white,colors[vx]), alpha=0.33+0.67*h,
                            label=ifelse(h==nlookback,"current","-$(nlookback-h)")*" trial")
                end
                ylims!(axs,0.45,1.05)
                if cx==1 plot!(axs,legend=:bottomleft, foreground_color_legend=nothing, background_color_legend=nothing) end
                if cx==1 title!(axs, labels[vx]) end
                if vx==1 ylabel!(axs, contexts[cx]*" context") end
            end
        end

        plot!(ax, plot_title=("outcome history, model "*modelid))

        if config[:showplot]
            display(ax)
        end
        if config[:saveplot]
            savefig(joinpath(config[:modelresultspath],"outcomehistory/","outcomehistory-$(string(modelid)).png"))
        end



    end


    accuracieslists = vcat(reshape.(accuracieslists,1,size(accuracieslists[1])...)...)  # concatenate mice in a new first dimension
    @info "size" size(accuracieslists)

    # plot aggregate
    ax = plot(layout=(2,length(cols)),size=(1.2* length(cols)*300, 2*250), legend=false, left_margin=30*Plots.px)
    for cx in eachindex(contexts)
        for vx in eachindex(cols)
            axs = ax[cx,vx]
            @decoderbackground(axs, Ts[2][begin], Ts[3][end], Ts[3][begin])
            for h in nlookback:-1:1
                m = dropdims(mean(accuracieslists[:,cx,vx,h,:,1],dims=1),dims=1)
                e = dropdims(std(accuracieslists[:,cx,vx,h,:,1],dims=1),dims=1)/sqrt(length(machineparameters[:modelids]))+
                    dropdims(mean(accuracieslists[:,cx,vx,h,:,3],dims=1),dims=1)
                    

                plot!(axs, timestamps, m, ribbon=e,
                        lw=[1,3,1][h], fillalpha=0.1, color=ifelse(h==nlookback,:white,colors[vx]), alpha=0.33+0.67*h,
                        label=ifelse(h==nlookback,"current","-$(nlookback-h)")*" trial")
            end
            ylims!(axs,0.45,1.05)
            if cx==1 plot!(axs,legend=:bottomleft, foreground_color_legend=nothing, background_color_legend=nothing) end
            if cx==1 title!(axs, labels[vx]) end
            if vx==1 ylabel!(axs, contexts[cx]*" context") end
        end
    end

    plot!(ax, plot_title=("outcome history, n=$(length(machineparameters[:modelids])) models"))
    if config[:showmultiplot]
        display(ax)
    end
    if config[:savemultiplot]
        savefig(joinpath(config[:modelresultspath],"outcomehistory/","outcomehistory-models.png"))
    end


end