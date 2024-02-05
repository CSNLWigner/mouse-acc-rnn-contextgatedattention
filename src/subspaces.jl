# functions for population activity in neural vectorspaces








function chancelevelshuffle(mouseid,nwbfile; nresamples=40)

    @info "chance level" mouseid
    triallist = nwbdf(nwbfile.trials)
    filter!(:difficulty=>u->u=="complex",triallist)


    neuronsspiketimeslist = gettrialrelativespiketimes(triallist, nwbfile.units)         # get spikes for trials
    timestamps, instantenousfiringrates = smoothinstantenousfiringrate(neuronsspiketimeslist, config[:stimuluslength], binsize=config[:dt])     # smooth 

    ntrials,ntimestamps,nneurons = size(instantenousfiringrates)
    accuracies = zeros(nresamples,ntimestamps,3)
    coefficients = zeros(nresamples,ntimestamps,nneurons+1,3)

    for resampleindex in 1:nresamples
        print(resampleindex," ")
        shuffledlabels = shuffle(triallist[!,:degree])
        _, accuracies[resampleindex,:,:], coefficients[resampleindex,:,:,:] =
            decodevariablefromactivity(instantenousfiringrates, shuffledlabels, halfwidth=2, nfolds=10)
    end
    println()

    accuracies = dropdims(mean(accuracies,dims=(1,2)),dims=(1,2))
    coefficients = dropdims(mean(coefficients,dims=(1,2)),dims=(1,2))
    
    @info "mean chance level" μ=accuracies[1] σ=accuracies[2] ε=accuracies[3]

    @save(config[:cachepath]*"subspace/chancelevel,shuffle-r$(nresamples)-$(string(mouseid)).bson", timestamps, accuracies, coefficients)

   

end




function decodevariables(mouseid::Symbol, nwbfile::PyObject)
    @info "mouseid" mouseid
    triallist = nwbdf(nwbfile.trials)
    filter!(:difficulty=>u->u=="complex",triallist)


    labels = ["visual","audio","context","decision","reward"]
    cols = [:degree,:freq,:context,:action,:water]
    contexts = ["visual","audio"]
    colors = [:blue,:green,:mediumvioletred,:darkorange,:gold]
    ntasks = length(cols)

    if config[:recalculatedecodevariables]
        neuronsspiketimeslist = gettrialrelativespiketimes(triallist, nwbfile.units)         # get spikes for trials
        timestamps, instantenousfiringrates = smoothinstantenousfiringrate(neuronsspiketimeslist, config[:stimuluslength], binsize=config[:dt])     # smooth 

        ntrials,ntimestamps,nneurons = size(instantenousfiringrates)
        accuracies = zeros(ntasks,ntimestamps,3)
        coefficients = zeros(ntasks,ntimestamps,nneurons+1,3)

        for taskindex in eachindex(cols)
            print("$(labels[taskindex]) ")
            _, accuracies[taskindex,:,:], coefficients[taskindex,:,:,:] =
                decodevariablefromactivity(instantenousfiringrates, triallist[!,cols[taskindex]], halfwidth=2, nfolds=10)

        end
        println()

        @save(config[:cachepath]*"subspace/decode,variables-$(string(mouseid)).bson", timestamps, accuracies, coefficients)

    else
        @load(config[:cachepath]*"subspace/decode,variables-$(string(mouseid)).bson", @__MODULE__, timestamps, accuracies, coefficients)
    end

    return
    ax = plot(layout=(1,ntasks),size=(1.2* ntasks*300, 1*300), legend=false, left_margin=30*Plots.px, bottom_margin=30*Plots.px)
    sm = 31 # smoothing window
    for taskindex in eachindex(cols)
        axs = ax[taskindex]
        @decoderbackground(axs, config[:stimulusstart], config[:stimulusend], config[:waterstart])
        m = MathUtils.convolve(accuracies[taskindex,:,1],ones(sm)./sm)
        e = MathUtils.convolve(accuracies[taskindex,:,3],ones(sm)./sm)
        plot!(axs, timestamps, m, ribbon=e, lw=2, fillalpha=0.3, color=colors[taskindex], label=nothing)
        xlims!(-1.2,4.2)
        ylims!(axs,0.45,1.05)
        title!(axs, labels[taskindex])
        if taskindex==1 ylabel!(axs, "accuracy") end
        xlabel!(axs, "time (s)")
    end

    plot!(ax, plot_title=("decode variables "*string(mouseid))*", neurons: $(size(coefficients,3)-1)")

    if config[:showplot]
        display(ax)
    end
    if config[:saveplot]
        savefig(joinpath(config[:resultspath],"subspace/","decode,variables-$(string(mouseid)).png"))
    end




end





"""
Compare choice DV to stimuli DV in each context
Find context-indeoendent choice subspace
"""
function choicegeometry(mouseid::Symbol, nwbfile::PyObject)

    @info "mouseid" mouseid
    triallist = nwbdf(nwbfile.trials)
    filter!(:difficulty=>u->u=="complex",triallist)


    labels = ["visual","audio","decision"]
    cols = [:degree,:freq,:action]
    contexts = ["visual","audio","both"]
    ncontexts = length(contexts)
    colors = [:blue,:green,:mediumvioletred,:darkorange,:gold]
    ntasks = length(cols)
    
    # load total choice subspace
    # @load(config[:cachepath]*"subspace/decode,variables-$(string(mouseid)).bson", @__MODULE__, timestamps, accuracies, coefficients)
    # accuracieschoice = accuracies[3,:,:]
    # coefficientschoice = coefficients[3,:,:]

    if config[:recalculatechoicegeometry]
        neuronsspiketimeslist = gettrialrelativespiketimes(triallist, nwbfile.units)
        timestamps, instantenousfiringrates = smoothinstantenousfiringrate(neuronsspiketimeslist, config[:stimuluslength], binsize=config[:dt])     # smooth 

        ntrials,ntimestamps,nneurons = size(instantenousfiringrates)
        accuracies = zeros(ntasks,ncontexts,ntimestamps,3)
        coefficients = zeros(ntasks,ncontexts,ntimestamps,nneurons+1,3)

        for taskindex in eachindex(cols)
            print("$(labels[taskindex])")
            for (cx,context) in enumerate(contexts)
                print(", $(context) context")
                if cx<3
                    mask = triallist[!,:context].==context
                else
                    mask = trues(nrow(triallist))
                end
                _, accuracies[taskindex,cx,:,:], coefficients[taskindex,cx,:,:,:] =
                    decodevariablefromactivity(instantenousfiringrates[mask,:,:], triallist[mask,cols[taskindex]], halfwidth=2, nfolds=10)
            end
            println()
        end
        
        @save(config[:cachepath]*"subspace/decode,choicegeometry-$(string(mouseid)).bson", timestamps, accuracies, coefficients)

    else
        @load(config[:cachepath]*"subspace/decode,choicegeometry-$(string(mouseid)).bson", @__MODULE__, timestamps, accuracies, coefficients)
    end






    # show degrees of choice to relevant and irrelevant stimulus

    axs = plot(layout=(ncontexts,3), size=(3*350, ncontexts*250), legend=false) 
    
    for (cx,context) in enumerate(contexts)
        for (sx,stimulus) in enumerate(labels[1:2])
            relevantf = sx==cx         # relevant stimulus flag
            if cx==3 relevantf = true end
            # @info "" size(coefficients[sx,cx,:,:]) size(timestamps)
            angles = angleofvectors(coefficients[sx,cx,:,1:end-1,1],coefficients[3,cx,:,1:end-1,1])            # end-1 -> we don't needt the bias coefficient for angle


            ax = axs[cx, sx]
            @decoderbackground(ax, config[:stimulusstart], config[:stimulusend], config[:waterstart], :darkgray)

            plot!(ax, timestamps, angles, lw=2, color=colors[sx], alpha=0.5+0.5*relevantf, label=nothing)
            ylims!(ax,0,120)
            hline!(ax,[90],ls=:dash,color=:darkgray)

            if sx==1 ylabel!(ax,contexts[cx]*" context\nangle bw. stimulus & decision") end
            if cx==1 title!(ax,labels[sx]*" stimulus") end
        end

        ax = axs[cx, 3]
        ylims!(ax, 0.45, 1.05)
        @decoderbackground(ax, config[:stimulusstart], config[:stimulusend], config[:waterstart], :darkgray)
        plot!(ax, timestamps, accuracies[3,cx,:,1], ribbon=accuracies[3,cx,:,3], lw=2, color=:darkorange, label=nothing)
        ylabel!(ax,"accuracy")
        if cx==1 title!(ax,"decision") end
    end


    display(axs)
end







"""
Learn stimulus DV in the early-drive period.
Compare late stimulus activity projected onto this early DV.
"""
function predictivechoicegeometry(mouseid::Symbol, nwbfile::PyObject)

    @info "mouseid" mouseid

    # get DVs from the early stimulus periods
    timestampindices = collect(26:75) .+ 150

    labelsrelevancy = ["relevant","irrelevant"]
    nrelevancies = length(labelsrelevancy)
    labelsmodalities = ["visual","audio"]
    nmodalities = length(labelsmodalities)

    if config[:recalculatepredictivechoicegeometry]

        triallist = nwbdf(nwbfile.trials)
        filter!(:difficulty=>u->u=="complex",triallist)
        addcongruencycolumn!(triallist)
        maperfs = movingaverageperformancetrialtypes(triallist)
        maskconsistent = highperformancetrialsmask(maperfs)
        maskexploratory = .! maskconsistent
        maskcongruent = triallist[!,:congruency].=="congruent"
        masksuccess = triallist[!,:success]

        neuronsspiketimeslist = gettrialrelativespiketimes(triallist, nwbfile.units)         # get spikes for trials
        timestamps, instantenousfiringrates = smoothinstantenousfiringrate(neuronsspiketimeslist, config[:stimuluslength], binsize=config[:dt])     # smooth


        ntrials,ntimestamps,nneurons = size(instantenousfiringrates)
        accuracies = zeros(nmodalities,nrelevancies,ntimestamps,3)
        coefficients = zeros(nmodalities,nrelevancies,ntimestamps,nneurons+1,3)
        projections = zeros(nmodalities,nrelevancies,2,ntimestamps,3)
        ntrials = zeros(nmodalities,nrelevancies,2)
        for (cx,context) in enumerate(labelsmodalities)
            for (sx, stimulus) in enumerate(labelsmodalities)
                mask = (triallist[!,:context].==context)  .& masksuccess  # .& maskconsistent     .& (.! maskcongruent)
                rx = 2 - Int(cx==sx)       # relevant, if the context is the same as the stimulus
                targetcolumn = [:degree,:freq][sx]
                _, accuracies[sx,rx,:,:], coefficients[sx,rx,:,:,:] =
                    decodevariablefromactivity(instantenousfiringrates[mask,:,:], triallist[mask,targetcolumn],
                                                halfwidth=2, nfolds=min(10,sum(mask)))
                # project activity to the stimuli axes from early period
                for gx in 1:2   # go nogo
                    maskaction = (triallist[!,:water].==[true,false][gx]) # separate go and nogo projections
                    projector = dropdims(mean(coefficients[sx,rx,timestampindices,:,1],dims=1), dims=1)
                    # projector = coefficients[sx,rx,timestampindices,:,1]
                    # projector = dropdims(mean(coefficients[sx,rx,:,:,1],dims=1), dims=1)
                    pr = projectontoaxis( instantenousfiringrates[mask .&  maskaction,:,:], projector./norm(projector))
                    projections[sx,rx,gx,:,1] = mean(pr, dims=1)      # statistics over trials
                    projections[sx,rx,gx,:,2] = std(pr, dims=1)
                    projections[sx,rx,gx,:,3] = projections[sx,rx,gx,:,2]/sqrt(size(pr,1))
                    ntrials[sx,rx,gx] = size(pr,1)
                end
            end
        end
        @save(joinpath(config[:cachepath],"subspace/","choicegeometry,predictive-$(string(mouseid)).bson"),
                    timestamps, timestampindices, accuracies, coefficients, projections, ntrials)
    else
        @load(joinpath(config[:cachepath],"subspace/","choicegeometry,predictive-$(string(mouseid)).bson"), @__MODULE__,
                    timestamps, timestampindices, accuracies, coefficients, projections, ntrials)
    end
    @info "" ntrials


    # plot
    axs = plot(layout=(2,3), size=(3*350,2*300), legend=nothing)
    colors = [:dodgerblue :darkblue; :lime :darkgreen]
    for (cx,context) in enumerate(labelsmodalities)
        for (sx, stimulus) in enumerate(labelsmodalities)
            rx = 2 - Int(cx==sx)       # relevant, if the context is the same as the stimulus
            ax = axs[sx,cx]
            for gx in 1:2   # go nogo
                plot!(ax, timestamps, projections[sx,rx,gx,:,1], ribbon=projections[sx,rx,gx,:,3], lw=2, color=colors[sx,rx], ls=[:solid,:dash][gx], label=nothing)
                if cx==1 ylabel!(ax, "early $(labelsmodalities[sx])\nprojection") end
                if sx==1 title!(ax, "$(labelsmodalities[cx]) context") end
                @nolinebackground(ax, config[:stimulusstart], config[:stimulusend], config[:waterstart], :darkgray)
                hline!(ax,[0],ls=:dash,color=:darkgray, label=nothing)
                # ylims!(ax,-800,800)
            end

            ax = axs[sx,3]
            m = abs.(projections[sx,rx,1,:,1]-projections[sx,rx,2,:,1])
            e = projections[sx,rx,1,:,3]+projections[sx,rx,2,:,3]
            m = @movingaverage(m,31)
            e = @movingaverage(e,31)
            plot!(ax, timestamps, m, ribbon=e, lw=2, color=colors[sx,rx], label=labelsrelevancy[rx])
            if cx==1 ylabel!(ax, "early $(labelsmodalities[sx])\nprojection difference go-nogo") end
            @nolinebackground(ax, config[:stimulusstart], config[:stimulusend], config[:waterstart], :darkgray)
            hline!(ax,[0],ls=:dash,color=:darkgray, label=nothing)
            # ylims!(ax,0,800)
            plot!(ax, legend=:topright, foreground_color_legend=nothing, background_color_legend=nothing)

        end
    end
    display(axs)
end





function outcomehistory(mouseids::Vector{Symbol}; nlookback=3)
    # number of trials to look back for trial information, including current trial
    brainarea = config[:sessions][mouseids[1]][:brainarea]
    @info "outcomehistory" nlookback brainarea n=length(mouseids) config[:recalculateoutcomehistory]


    labels = ["visual","audio","decision","reward"]
    cols = [:degree,:freq,:action,:water]
    contexts = ["visual","audio"]
    colors = [:blue,:green,:darkorange,:gold]

    accuracieslists = []
    timestamps = []
    for mouseid in mouseids
        @info "mouseid" mouseid
        nwbfile,_ = loadnwbdata(mouseid)


        triallist = nwbdf(nwbfile.trials)
        filter!(:difficulty=>u->u=="complex",triallist)
        # @info "trials" names(triallist) triallist


        # smooth spikes into rate, separately in the two contexts
        # collect variables
        timestamps = Float64[]
        activities = Array{Float64,3}[]   # (class,trials,timecourse+neurons)
        variablehistorieslist = []      # [contexts][variables,historycolumns]
        for currentcontext in contexts
            selectedtriallist = filter([:context]=>(c->c==currentcontext),triallist)
            neuronsspiketimeslist = gettrialrelativespiketimes(selectedtriallist[nlookback:end,:], nwbfile.units)         # get spikes for trials
            timestamps, instantenousfiringrates = smoothinstantenousfiringrate(neuronsspiketimeslist, config[:stimuluslength], binsize=config[:dt])     # smooth 
            # @info "$currentcontext context" size(instantenousfiringrates)
            push!(activities, instantenousfiringrates)
            
            variablehistories = []
            for col in cols
                variablehistory = []#Vector{typeof(col)}[]
                for h in 1:nlookback
                    push!(variablehistory,selectedtriallist[h:end+h-nlookback,col])
                end
                push!(variablehistories,@asarray(variablehistory))
            end
            push!(variablehistorieslist,variablehistories)
        end
        _,ntimestamps,nneurons = size(activities[1])
        # @info "histories" size(variablehistorieslist) size(variablehistorieslist[1]) size(variablehistorieslist[1][1])
        # @info "(trials,timecourse,neurons), contexts" size(activities)

        if config[:recalculateoutcomehistory]
            # perform decoding of all variables and all histories
            machinelist = []  # serial, no multiindex
            accuracieslist = zeros(length(contexts),length(cols),nlookback,ntimestamps,3)
            coefficientslist = zeros(length(contexts),length(cols),nlookback,ntimestamps,nneurons+1,3) # will hold the intercept as the last value

            for cx in eachindex(contexts)
                for v in eachindex(cols)
                    @info "decoding in $(contexts[cx]) context: $(cols[v]) $nlookback-length history"
                    for h in 1:nlookback
                        machine, accuracieslist[cx,v,h,:,:], coefficientslist[cx,v,h,:,:,:] =
                        decodevariablefromactivity(activities[cx], variablehistorieslist[cx][v][h,:], halfwidth=2, nfolds=10)
                        push!(machinelist, machine)
                    end
                end
            end

            @save(joinpath(config[:cachepath],"subspace/","outcomehistory-$(string(mouseid)).bson"), accuracieslist, coefficientslist)
        else
            @load(joinpath(config[:cachepath],"subspace/","outcomehistory-$(string(mouseid)).bson"), @__MODULE__, accuracieslist, coefficientslist)
        end

        push!(accuracieslists, accuracieslist)
        
        
        # plot for individual mice
        sm = 51 # accuracy smoothing for display
        # padding = (config[:posteventpadding] + config[:stimuluslength] - config[:preeventpadding]).*(0:nlookback-1)
        

        ax = plot(layout=(2,length(cols)),size=(1.2* length(cols)*300, 2*250), legend=false, left_margin=30*Plots.px)
        for cx in eachindex(contexts)
            for vx in eachindex(cols)
                axs = ax[cx,vx]
                @decoderbackground(axs, config[:stimulusstart], config[:stimulusend], config[:waterstart])
                for h in nlookback:-1:1
                    plot!(axs, timestamps, MathUtils.convolve(accuracieslist[cx,vx,h,:,1],ones(sm)./sm),
                            ribbon=MathUtils.convolve(accuracieslist[cx,vx,h,:,3],ones(sm)./sm),
                            lw=[1,3,1][h], fillalpha=0.1, color=ifelse(h==nlookback,:white,colors[vx]), alpha=0.33+0.67*h,
                            label=ifelse(h==nlookback,"current","-$(nlookback-h)")*" trial")
                end
                xlims!(-1.2,4.2)
                ylims!(axs,0.45,1.05)
                if cx==1 plot!(axs,legend=:topleft, foreground_color_legend=nothing, background_color_legend=nothing) end
                if cx==1 title!(axs, labels[vx]) end
                if vx==1 ylabel!(axs, contexts[cx]*" context") end
            end
        end

        plot!(ax, plot_title=("outcome history "*string(:mouseid)))

        if config[:showplot]
            display(ax)
        end
        if config[:saveplot]
            savefig(joinpath(config[:resultspath],"subspace/","outcomehistory-$(string(mouseid)).png"))
        end



    end

    
    accuracieslists = vcat(reshape.(accuracieslists,1,size(accuracieslists[1])...)...)  # concatenate mice in a new first dimension
    @info "size" size(accuracieslists)

    # plot aggregate
    sm = 51 # accuracy smoothing for display
    ax = plot(layout=(2,length(cols)),size=(1.2* length(cols)*300, 2*250), legend=false, left_margin=30*Plots.px)
    for cx in eachindex(contexts)
        for vx in eachindex(cols)
            axs = ax[cx,vx]
            @decoderbackground(axs, config[:stimulusstart], config[:stimulusend], config[:waterstart])
            for h in nlookback:-1:1
                m = MathUtils.convolve(dropdims(mean(accuracieslists[:,cx,vx,h,:,1],dims=1),dims=1),ones(sm)./sm)
                e = MathUtils.convolve( dropdims(std(accuracieslists[:,cx,vx,h,:,1],dims=1),dims=1)/sqrt(length(mouseids))+
                    dropdims(mean(accuracieslists[:,cx,vx,h,:,3],dims=1),dims=1),
                    ones(sm)./sm)

                plot!(axs, timestamps, m, ribbon=e,
                        lw=[1,3,1][h], fillalpha=0.1, color=ifelse(h==nlookback,:white,colors[vx]), alpha=0.33+0.67*h,
                        label=ifelse(h==nlookback,"current","-$(nlookback-h)")*" trial")
            end
            xlims!(-1.2,4.2)
            ylims!(axs,0.45,1.05)
            if cx==1 plot!(axs,legend=:topleft, foreground_color_legend=nothing, background_color_legend=nothing) end
            if cx==1 title!(axs, labels[vx]) end
            if vx==1 ylabel!(axs, contexts[cx]*" context") end
        end
    end
    
    plot!(ax, plot_title=("outcome history "*brainarea*" n=$(length(mouseids))"))
    if config[:showmultiplot]
        display(ax)
    end
    if config[:savemultiplot]
        savefig(joinpath(config[:resultspath],"subspace/","outcomehistory-$(brainarea).png"))
    end



end



 


"""
creates 2D coordinates for each trial
with projections to the decoder coefficients
of the variables, ordered and orthogonalized
by the Gram-Schmidt process
"""
function lowdimortho(mouseid,nwbfile,variablelist=[:degree,:freq])
    @info "lowdimortho" mouseid
    triallist = nwbdf(nwbfile.trials)
    filter!(:difficulty=>u->u=="complex",triallist)

    labels = ["visual","audio","context","decision","reward"]
    cols = [:degree,:freq,:context,:action,:water]
    variableindices = indexin(variablelist, cols)

    # load the basis vectors
    @load(config[:cachepath]*"subspace/decode,variables-$(string(mouseid)).bson", @__MODULE__, timestamps, accuracies, coefficients)

    # average over timepoints during stimulus
    stimstartindex = findfirst(timestamps.>=config[:stimulusstart])
    stimlengthindex = Int(round(config[:inputdrivenlength]/(config[:dt])))-1
    A = permutedims(   dropdims(mean(   coefficients[variableindices,stimstartindex:stimstartindex+stimlengthindex,1:end-1],   dims=2),dims=2),   (2,1)   )

    # factorize to orthonormal basis
    B = orthogonalizevectors(A)
    # prepare activity for projection
    neuronsspiketimeslist = gettrialrelativespiketimes(triallist, nwbfile.units)
    timestamps, instantenousfiringrates = smoothinstantenousfiringrate(neuronsspiketimeslist, config[:stimuluslength], binsize=config[:dt])     # smooth 
    ntrials,ntimestamps,nneurons = size(instantenousfiringrates)

    # projecti activity onto the new basis vectors
    instantenousfiringrates = instantenousfiringrates[:,stimstartindex:stimstartindex+stimlengthindex,:]
    projectedifrs = similar(instantenousfiringrates[:,:,1:length(variablelist)])
    for tx in axes(instantenousfiringrates,2), trx in axes(instantenousfiringrates,1)
        projectedifrs[trx,tx,:] = B'*instantenousfiringrates[trx,tx,:] ./ norm(instantenousfiringrates[trx,tx,:])
    end
    projectedifrs ./ norm(projectedifrs)



    S = dropdims(mean(projectedifrs,dims=2),dims=2)   # mean over timepoints
    S .-= mean(S,dims=1) # center over trials

    b = B'*A ./ norm(A)         # transform the original basis to display in the new coordinates
    @info "basis" b
    foreach(normalize!, eachcol(b))


    
    # export
    # @save(config[:cachepath]*"subspace/lowdimortho,viau-$(string(mouseid)).bson", triallist, S, b)




    # plot
    axs = plot(legend=false,size=(300,300),aspect_ratio=1, label=nothing)
    ax = axs

    b .*= 0.08               # display size
    o = -0.22                # offset
    colors = [:white :cyan; :purple :red ]
    w = 0.03  # annotation width

    # plot basis vectors
    plot!(ax,[0,b[1,1]].+o,[0,b[2,1]].+o,color=:navy,lw=6)
    plot!(ax,[0,b[1,2]].+o,[0,b[2,2]].+o,color=:darkgreen,lw=6)
    # plot mean early projections
    for (vix,vi) in enumerate([45,135]), (aux,au) in enumerate([5000,10000])
        mask = (triallist[!,:degree].==vi) .& (triallist[!,:freq].==au)
        scatter!(ax,S[mask,1],S[mask,2], color=colors[vix,aux], markerstrokewidth=0)         # label=string(vi)*"° "*string(au)*" Hz"

        # legend annotation
        scatter!(ax, -0.15 .+ [0] .+ (vix-1)*w,  0.15 .+ [0] .+ (aux-1)*w, color=colors[vix,aux], markerstrokewidth=0)
    end
    annotate!(ax, -0.15+w/2, 0.15+w+0.02, "visual\ngo nogo", font(pointsize=6, color=:White, halign=:center, valign=:bottom))
    annotate!(ax, -0.15-w-0.02, 0.15+w/2,  "audio\ngo nogo", font(pointsize=6, color=:White, halign=:right, valign=:center))


    xlims!(ax,-0.27,0.27)
    ylims!(ax,-0.27,0.27)
    xlabel!(ax,"visual DV")
    ylabel!(ax,"audio DV\n(orthogonal projection)")
    title!(ax, string(mouseid))
    
    display(axs)

end






function cognitivesurplus(mouseid,nwbfile)
    labels = ["consistent","ecploratory"]
    nstates = length(labels)         

    if config[:recalculatecognitivesurplus]
        nwbfile,_ = loadnwbdata(mouseid)
        triallist = nwbdf(nwbfile.trials)
        filter!(:difficulty=>u->u=="complex",triallist)
        addcongruencycolumn!(triallist)
        maperfs = movingaverageperformancetrialtypes(triallist)
        maskconsistent = highperformancetrialsmask(maperfs)
        maskexploratory = .! maskconsistent

        neuronsspiketimeslist = gettrialrelativespiketimes(triallist, nwbfile.units)         # get spikes for trials
        timestamps, instantenousfiringrates = smoothinstantenousfiringrate(neuronsspiketimeslist, config[:stimuluslength], binsize=config[:dt])     # smooth 

        ntrials,ntimestamps,nneurons = size(instantenousfiringrates)
        accuracies = zeros(nstates,ntimestamps,3)
        coefficients = zeros(nstates,ntimestamps,nneurons+1,3)

        for (sx,mask) in  enumerate([maskconsistent,maskexploratory])
            machine, accuracies[sx,:,:], coefficients[sx,:,:,:] =
            decodevariablefromactivity(instantenousfiringrates[mask,:,:], triallist[mask,:context], halfwidth=2, nfolds=10)
        end

        @save(joinpath(config[:cachepath],"subspace/","cognitivesurplus-$(string(mouseid)).bson"), accuracies, coefficients)
    else
        @load(joinpath(config[:cachepath],"subspace/","cognitivesurplus-$(string(mouseid)).bson"), @__MODULE__, accuracies, coefficients)
    end


end




function mutualswitchsubspaces(mouseids::Vector{Symbol})
    nmice = length(mouseids)
    sort!(mouseids)

    @info "mutual switch subspaces" n=nmice


    labels = ["visual","audio","context","decision","reward"]
    ntasks = length(labels)


    timestamps = collect(config[:stimulusstart]+config[:preeventpadding]:config[:dt]:config[:stimulusend]+config[:posteventpadding]-config[:dt]) .+ config[:dt]/2

    axs = plot(layout=(3,nmice),size=(300*nmice, 3*250), legend=false, left_margin=30*Plots.px, bottom_margin=30*Plots.px)

    for (n,mouseid) in enumerate(mouseids)
        @info "mouseid" mouseid
        nwbfile,_ = loadnwbdata(mouseid)


        triallist = nwbdf(nwbfile.trials)
        filter!(:difficulty=>u->u=="complex",triallist)
        addcongruencycolumn!(triallist)

        neuronsspiketimeslist = gettrialrelativespiketimes(triallist, nwbfile.units)         # get spikes for trials
        timestamps, instantenousfiringrates = smoothinstantenousfiringrate(neuronsspiketimeslist, config[:stimuluslength], binsize=config[:dt])     # smooth

        ntrials,ntimestamps,nneurons = size(instantenousfiringrates)
        H = instantenousfiringrates   # shorthand
    

        # get single modality trials
        triallistsingle = nwbdf(nwbfile.trials)
        filter!(:difficulty=>u->u=="simple",triallistsingle)
        addcongruencycolumn!(triallistsingle)

        neuronsspiketimeslist = gettrialrelativespiketimes(triallistsingle, nwbfile.units)         # get spikes for trials
        _, instantenousfiringrates = smoothinstantenousfiringrate(neuronsspiketimeslist, config[:stimuluslength], binsize=config[:dt])     # smooth
        Hsingle = instantenousfiringrates   # shorthand



        timerange = (timestamps .>= config[:stimulusstart]) .& (timestamps .<= config[:stimulusend])

        maskcontext = triallist[!,:context].=="visual"
        maskgo = triallist[!,:water].==true
        maskvisualgo = triallist[!,:degree].==45
        maskvisualnogo = triallist[!,:degree].==135
        maskaudiogo = triallist[!,:freq].==5000
        maskaudionogo = triallist[!,:freq].==10000
        maskcongruency = triallist[!,:congruency].=="congruent"


        # modality index
        modalityindex =  zeros(nneurons,2)
        for (mskx,(maskvisual,maskaudio)) in enumerate(zip((maskvisualgo,maskvisualnogo),(maskaudiogo,maskaudionogo)))
            modalityindex[:,mskx] = dropdims(mean(abs.(H[maskvisual,timerange,:]),dims=(1,2)),dims=(1,2)) - 
                                        dropdims(mean(abs.(H[maskaudio,timerange,:]),dims=(1,2)),dims=(1,2))
        end
        modalitygounits = sortperm(modalityindex[:,1],rev=true)
        modalitynogounits = sortperm(modalityindex[:,2],rev=true)
        modalityunits = [ modalitygounits modalitynogounits ]


        # modality index single
        maskgosingle = triallistsingle[!,:water].==true
        maskvisualgosingle = triallistsingle[!,:degree].==45
        maskvisualnogosingle = triallistsingle[!,:degree].==135
        maskaudiogosingle = triallistsingle[!,:freq].==5000
        maskaudionogosingle = triallistsingle[!,:freq].==10000
        maskcongruencysingle = triallistsingle[!,:congruency].=="congruent"

        
        modalityindexsingle =  zeros(nneurons,2)
        for (mskx,(maskvisual,maskaudio)) in enumerate(zip((maskvisualgosingle,maskvisualnogosingle),(maskaudiogosingle,maskaudionogosingle)))
            modalityindexsingle[:,mskx] = dropdims(mean(abs.(Hsingle[maskvisual,timerange,:]),dims=(1,2)),dims=(1,2)) - 
                                        dropdims(mean(abs.(Hsingle[maskaudio,timerange,:]),dims=(1,2)),dims=(1,2))
        end
        modalitygounitssingle = sortperm(modalityindexsingle[:,1],rev=true)
        modalitynogounitssingle = sortperm(modalityindexsingle[:,2],rev=true)
        modalityunitssingle = [ modalitygounitssingle modalitynogounitssingle ]




        # context index
        timerangescontext = [(timestamps .>= (config[:stimulusstart]) - 1.00) .& (timestamps .<= (config[:stimulusstart] - 0.25 )),
                             (timestamps .>= (config[:stimulusstart]) + 0.0) .& (timestamps .<= (config[:stimulusstart] + 0.75 )),
                             (timestamps .>= (config[:waterstart]) - 0.25) .& (timestamps .<= (config[:waterstart] + 0.5 )),
                              trues(size(timestamps)) ]

        contextindex = zeros(nneurons,length(timerangescontext),2)
        for px in 1:4, (mskx,maskdir) in enumerate((maskgo, .! maskgo))
            contextindex[:,px,mskx] = 
                dropdims(mean(H[maskdir .&     maskcontext   .&  (maskcongruency),timerangescontext[px],:],dims=(1,2)),dims=(1,2)) -
                dropdims(mean(H[maskdir .& (.! maskcontext)  .&  (maskcongruency),timerangescontext[px],:],dims=(1,2)),dims=(1,2))
        end


        @save(joinpath(config[:cachepath],"subspace/","mutualswitchsubspaces-$(string(mouseid)).bson"),
            modalityunits, modalityindex, modalityunitssingle, modalityindexsingle, contextindex)
        



        # plotting


        ax = axs[1,n]
        plot!(ax,contextindex[modalityunitssingle[:,1],3,1],color=:white,title=string(mouseid))
        plot!(ax,contextindex[modalityunitssingle[:,2],3,2],color=:red)
        hline!(ax,[0],ls=:dash,color=:white,alpha=0.5)
        ylabel!(ax,"context index")
        xlabel!(ax,"neurons (modality index ordered)")



        ax = axs[2,n]
        scatter!(ax,modalityunitssingle,contextindex[:,3,:],color=[:white :red], label=["go","nogo"], title=string(mouseid))
        hline!(ax,[0],ls=:dash,color=:white,alpha=0.5, label=nothing)
        vline!(ax,[0],ls=:dash,color=:white,alpha=0.5, label=nothing)
        ylabel!(ax,"context index")
        xlabel!(ax,"modality index")



        ax = axs[3,n]
        scatter!(ax,modalityindexsingle[:,1],modalityindexsingle[:,2],color=:lightseagreen, title=string(mouseid))
        hline!(ax,[0],ls=:dash,color=:white,alpha=0.5, label=nothing)
        vline!(ax,[0],ls=:dash,color=:white,alpha=0.5, label=nothing)
        xlabel!(ax,"modality index go")
        ylabel!(ax,"modality index nogo")





    end
    display(axs)

end












function decoderelevancy(mouseids::Vector{Symbol})
    brainarea = config[:sessions][mouseids[1]][:brainarea]
    @info "relevancy by context" brainarea n=length(mouseids) config[:recalculatedecoderelevancy]


    labelsstimulus = ["visual","audio"]
    colsstimulus = [:degree, :freq]
    labelscontexts = ["visual","audio"]
    labelsrelevancies = ["relevant" "irrelevant"; "irrelevant" "relevant"]
    colors = [:dodgerblue :green; :blue :lime]     # relevancy x stimulus

    accuracieslists = []
    timestamps = []
    timestampindices = collect(1:300)   .+ 300                   #   collect(26:75) .+ 150          # get DVs from the early stimulus periods
    congruencprefixes = ["all", "congruent","incongruent"]
    for mouseid in mouseids
        @info "mouseid" mouseid


        if config[:recalculatedecoderelevancy]
            nwbfile,_ = loadnwbdata(mouseid)


            triallist = nwbdf(nwbfile.trials)
            filter!(:difficulty=>u->u=="complex",triallist)
            addcongruencycolumn!(triallist)

            neuronsspiketimeslist = gettrialrelativespiketimes(triallist, nwbfile.units)         # get spikes for trials
            timestamps, instantenousfiringrates = smoothinstantenousfiringrate(neuronsspiketimeslist, config[:stimuluslength], binsize=config[:dt])     # smooth

            ntrials,ntimestamps,nneurons = size(instantenousfiringrates)

            accuracies = zeros(3,2,2,ntimestamps,3)    # (congruency, context, stimulus, time, stats)
            coefficients = zeros(3,2,2,ntimestamps,nneurons+1,3) # (congruency, context, stimulus, time, coefficients, stats)
            projections = zeros(3,2,2,2,ntimestamps,3)   # (congruency, context, stimulus, gonogo, time, stats)

            for (congruencyindex,congruencyprefix) in enumerate(congruencprefixes)
                for (contextindex,context) in enumerate(labelscontexts)
                    mask = (triallist[!,:context].==context )
                    if congruencyprefix!="all"
                        mask = mask .& (triallist[!,:congruency].==congruencyprefix)
                    end
                    for (stimulusindex,stimulus) in enumerate(labelsstimulus)
                        print("$(congruencyprefix) $(context) context/$(stimulus) stimulus, ")
                        _, accuracies[congruencyindex,contextindex,stimulusindex,:,:], coefficients[congruencyindex,contextindex,stimulusindex,:,:,:] =
                            decodevariablefromactivity(instantenousfiringrates[mask,:,:], triallist[mask,colsstimulus[stimulusindex]], halfwidth=2, nfolds=10)
                        # project activity to the stimuli axes from early period
                        for gx in 1:2   # go nogo
                            maskaction = (triallist[!,:water].==[true,false][gx]) # separate go and nogo projections
                            projector = dropdims(mean(coefficients[congruencyindex,contextindex,stimulusindex,timestampindices,1:end-1,1],dims=1), dims=1)
                            # projector = coefficients[sx,rx,timestampindices,:,1]
                            # projector = dropdims(mean(coefficients[sx,rx,:,:,1],dims=1), dims=1)
                            pr = projectontoaxis( instantenousfiringrates[mask .&  maskaction,:,:], projector./norm(projector))
                            projections[congruencyindex,contextindex,stimulusindex,gx,:,1] = mean(pr, dims=1)      # statistics over trials
                            projections[congruencyindex,contextindex,stimulusindex,gx,:,2] = std(pr, dims=1)
                            projections[congruencyindex,contextindex,stimulusindex,gx,:,3] =
                                     projections[congruencyindex,contextindex,stimulusindex,gx,:,2]/sqrt(size(pr,1))
                        end
                    end
                end
            end
            println()

            @save(config[:cachepath]*"subspace/decode,relevancy-$(string(mouseid)).bson", timestamps, accuracies, coefficients, projections)
        end
    end


    return
    
    
    # display
    congruencyprefix = "all"
    
    for mouseid in mouseids
        @info "mouseid" mouseid
        @load(config[:cachepath]*"subspace/decode,relevancy-$(string(mouseid)).bson", @__MODULE__, timestamps, accuracies, coefficients)
        push!(accuracieslists, accuracies)

        ax = plot(layout=(1,2),size=(1.2* 2*300, 1*300), legend=false, left_margin=15*Plots.px, bottom_margin=15*Plots.px)
        sm = 31 # smoothing window
        for (contextindex,context) in enumerate(labelscontexts), (stimulusindex,stimulus) in enumerate(labelsstimulus)
            axs = ax[stimulusindex]
            @decoderbackground(axs, config[:stimulusstart], config[:stimulusend], config[:waterstart])
            m = MathUtils.convolve(accuracies[contextindex,stimulusindex,:,1],ones(sm)./sm)
            e = MathUtils.convolve(accuracies[contextindex,stimulusindex,:,3],ones(sm)./sm)
            plot!(axs, timestamps, m, ribbon=e, lw=2, fillalpha=0.3, color=colors[contextindex,stimulusindex], alpha=0.8,
                  label=labelsrelevancies[contextindex,stimulusindex]*" "*context*" context")
            if contextindex==2
                xlims!(-1.2,4.2)
                ylims!(axs,0.45,1.05)
                plot!(axs,legend=:topleft, foreground_color_legend=nothing, background_color_legend=nothing)
                title!(axs, stimulus*" stimulus")
            end
            if stimulusindex==1 ylabel!(axs, context*" context\naccuracy") end
            xlabel!(axs, "time (s)")
        end
    
        plot!(ax, plot_title=("decode stimulus relevancy "*string(mouseid)))
    
        if config[:showplot]
            display(ax)
        end
        if config[:saveplot]
            savefig(joinpath(config[:resultspath],"subspace/","decode,relevancy-$(string(mouseid)).png"))
        end

    end


    accuracieslists = vcat(reshape.(accuracieslists,1,size(accuracieslists[1])...)...)

    # plot for grouped animals
    ax = plot(layout=(1,2),size=(1.2* 2*300, 1*300), legend=false, left_margin=15*Plots.px, bottom_margin=15*Plots.px)
    sm = 31 # smoothing window
    for (contextindex,context) in enumerate(labelscontexts), (stimulusindex,stimulus) in enumerate(labelsstimulus)
        axs = ax[stimulusindex]
        @decoderbackground(axs, config[:stimulusstart], config[:stimulusend], config[:waterstart])
        m = MathUtils.convolve(dropdims(mean(accuracieslists[:,1,contextindex,stimulusindex,:,1],dims=1),dims=1),ones(sm)./sm)
        e = MathUtils.convolve(dropdims(std(accuracieslists[:,1,contextindex,stimulusindex,:,1],dims=1),dims=1)/sqrt(length(mouseids))+
            dropdims(mean(accuracieslists[:,1,contextindex,stimulusindex,:,3],dims=1),dims=1),
            ones(sm)./sm)
        plot!(axs, timestamps, m, ribbon=e, lw=2, fillalpha=0.3, color=colors[contextindex,stimulusindex], alpha=0.8,
                label=labelsrelevancies[contextindex,stimulusindex]*" "*context*" context")
        if contextindex==2
            xlims!(-1.2,4.2)
            ylims!(axs,0.45,1.05)
            plot!(axs,legend=:topleft, foreground_color_legend=nothing, background_color_legend=nothing)
            title!(axs, stimulus*" stimulus")
        end
        if stimulusindex==1 ylabel!(axs, context*" context\naccuracy") end
        xlabel!(axs, "time (s)")
    end


    plot!(ax, plot_title=("decode stimulus relevancy $(brainarea)"))

    if config[:showmultiplot]
        display(ax)
    end
    if config[:savemultiplot]
        savefig(joinpath(config[:resultspath],"subspace/","decode,relevancy-$(brainarea).png"))
    end



end



function suppressionbehaviour(mouseid,lick=nothing)
    @info "suppression and behaviour, $mouseid"
    labelsstates = ["consistent","ecploratory"]
    nstates = length(labelsstates)
    labelsrelevancy = ["relevant","irrelevant"]
    nrelevancies = length(labelsrelevancy)
    labelsmodalities = ["visual","audio"]
    nmodalities = length(labelsmodalities)
    lickcontrollabel = ifelse(isnothing(lick), "", ",$(lick)control")


    if config[:recalculatesuppressionbehaviour]
        nwbfile,_ = loadnwbdata(mouseid)
        triallist = nwbdf(nwbfile.trials)
        filter!(:difficulty=>u->u=="complex",triallist)
        addcongruencycolumn!(triallist)
        maperfs = movingaverageperformancetrialtypes(triallist)
        maskconsistent = highperformancetrialsmask(maperfs)
        maskexploratory = .! maskconsistent

        neuronsspiketimeslist = gettrialrelativespiketimes(triallist, nwbfile.units)         # get spikes for trials
        timestamps, instantenousfiringrates = smoothinstantenousfiringrate(neuronsspiketimeslist, config[:stimuluslength], binsize=config[:dt])     # smooth 
        ntrials,ntimestamps,nneurons = size(instantenousfiringrates)


        # collect projection operators from full stimuli:
        projectors = zeros(nmodalities,nrelevancies,ntimestamps,nneurons+1,3)
        for (sx, stimulus) in enumerate(labelsmodalities)
            for (cx, context) in enumerate(labelsmodalities)
                rx = 2 - Int(cx==sx)       # relevant, if the context is the same as the stimulus
                maskcontext = triallist[!,:context].==context
                _, _, projectors[sx,rx,:,:,:] =
                decodevariablefromactivity(instantenousfiringrates[maskcontext,:,:],
                                            triallist[maskcontext,[:degree,:freq][sx]], halfwidth=2, nfolds=10)
            end
        end

        # project activity onto this stable subspace
        accuracies = zeros(nstates,nmodalities,nrelevancies,ntimestamps,3)
        coefficients = zeros(nstates,nmodalities,nrelevancies,ntimestamps,nneurons+1,3)
        projections = zeros(nstates,nmodalities,nrelevancies,2,ntimestamps,3)
        ntrials = zeros(nstates,nmodalities,nrelevancies,4)
        for (mx,maskstate) in  enumerate([maskconsistent,maskexploratory])
            for (cx,context) in enumerate(labelsmodalities)
                for (sx, stimulus) in enumerate(labelsmodalities)
                    maskcontext = (triallist[!,:context].==context)
                    mask = maskstate .& maskcontext
                    if ! isnothing(lick)    # compose mask for lick trials
                        lickmask = ifelse(lick=="lick", .! triallist[!,:action], triallist[!,:action])
                        mask = mask .& lickmask
                    end
                    rx = 2 - Int(cx==sx)       # relevant, if the context is the same as the stimulus
                    targetcolumn = [:degree,:freq][sx]
                    ntrials[mx,sx,rx,1] = sum(mask)     # number of trials
                    ntrials[mx,sx,rx,2] = length(unique(triallist[mask,targetcolumn])) # number of unique stimulus identities
                    ntrials[mx,sx,rx,3] = sum(triallist[mask,targetcolumn] .∈ Ref([45,5000]) ) # number of go trials (45°, 5kHz)
                    ntrials[mx,sx,rx,4] = sum(triallist[mask,targetcolumn] .∈ Ref([135, 10000])) # number of nogo trials (135°, 10kHz)
                    if ntrials[mx,sx,rx,2]<2 continue end            # cannot decode from only one stimulus
                    _, accuracies[mx,sx,rx,:,:], coefficients[mx,sx,rx,:,:,:] =
                        decodevariablefromactivity(instantenousfiringrates[mask,:,:], triallist[mask,targetcolumn],
                                                   halfwidth=2, nfolds=min(10,sum(mask)))
                    
                    # project activity to the stimulus axis
                    for gx in 1:2
                        maskaction = (triallist[!,:water].==[true,false][gx]) # separate go and nogo projections
                        pr = projectontoaxis(instantenousfiringrates[mask .&  maskaction,:,:], projectors[sx,rx,:,:,1])
                        projections[mx,sx,rx,gx,:,1] = mean(pr, dims=1)      # statistics over trials
                        projections[mx,sx,rx,gx,:,2] = std(pr, dims=1)
                        projections[mx,sx,rx,gx,:,3] = projections[mx,sx,rx,gx,:,2]/sqrt(size(pr,1))
                    end
                end
            end
        end

        @save(joinpath(config[:cachepath],"subspace/","suppressionbehaviour$(lickcontrollabel)-$(string(mouseid)).bson"),
                      timestamps, instantenousfiringrates, maskconsistent, accuracies, coefficients, projections, ntrials)
    else
        @load(joinpath(config[:cachepath],"subspace/","suppressionbehaviour$(lickcontrollabel)-$(string(mouseid)).bson"), @__MODULE__,
                      timestamps, instantenousfiringrates, maskconsistent, accuracies, coefficients, projections, ntrials)

    end
end








function projectactivitycontext(session::Dict{Symbol,Any},nwbfile::PyObject)
    triallist = nwbdf(nwbfile.trials)
    filter!(:difficulty=>u->u=="complex",triallist)

    timestamps = Float64[]
    activities = Array{Float64,3}[]   # (class,trials,timecourse+neurons)
    contexts = String[]
    for (cx,context) in enumerate(["visual","audio"])
        selectedtriallist = filter([:context]=>(c->c==context), triallist)       # select trials
        neuronsspiketimeslist = gettrialrelativespiketimes(selectedtriallist, nwbfile.units)         # get spikes for trials
        timestamps, instantaneousfiringratetrials = smoothinstantenousfiringrate(neuronsspiketimeslist, config[:stimuluslength], binsize=config[:dt])     # smooth
        push!(activities, instantaneousfiringratetrials)             # create X
        append!(contexts, repeat([context],size(instantaneousfiringratetrials,1) ) )         # create y
    end
    activities = vcat(activities...)
    @info "(trials,timecourse,neurons)" size(activities)
    
    machine, accuracies, coefficients = decodevariablefromactivity(activities,contexts)

    serialize(config[:modelspath]*"subspace/project,context,congruency-$(session[:mouseid]).dat",
              (machine, accuracies, coefficients))

    @info "size" size(accuracies) size(coefficients)
    ax = plot(size=(1.2* 3*200,3*200), legend=false)
    axs = ax
    plot!(axs, timestamps, accuracies[:,1], lw=2, ribbon=accuracies[:,3],fillalpha=0.5, color=:mediumvioletred)
    @plotdecoder axs config :binary
    # display(ax)
    savefig(config[:resultspath]*"subspace/project,context,congruency-$(session[:mouseid]).png")


    # projectactivity()


end











function conditionaveragepca(session::Dict{Symbol,Any},nwbfile::PyObject)
    triallist = nwbdf(nwbfile.trials)
    filter!(:difficulty=>u->u=="complex",triallist)
    # @info "condition average PCA" triallist

    cols = [:context,:degree,:freq]
    vals = [["visual",45,5000], ["visual",45,10000], ["visual",135,5000], ["visual",135,10000],
            ["audio",45,5000], ["audio",45,10000], ["audio",135,5000], ["audio",135,10000]]
    n_conditions = length(vals)
    colors = [:navy, :dodgerblue, :darkred, :red, :darkgreen, :gold, :lightgreen, :darkorange]

    # sort trial ids into meaningful 
    timestamps = Float64[]
    activities = Array{Float64,3}[]   # (class,trials,timecourse+neurons)
    for val in vals
        selectedtriallist = filter(cols => (c,v,a)->(c==val[1]) && (v==val[2]) && (a==val[3]), triallist)
        neuronsspiketimeslist = gettrialrelativespiketimes(selectedtriallist, nwbfile.units)         # get spikes for trials
        timestamps, instantaneousfiringratetrials = smoothinstantenousfiringrate(neuronsspiketimeslist, config[:stimuluslength], binsize=config[:dt])     # smooth 
        push!(activities, instantaneousfiringratetrials)             # create X
    end


    n_timecourse = length(timestamps)
    n_neurons = size(activities[1],3)

    # create condition averages
    R = vcat([ mean(activity,dims=1)   for activity in activities  ]...)
    X = R .- mean(R, dims=1)


    # calculate PCA
    n_components = 2
    P = zeros(n_timecourse, n_components, n_neurons)
    for t in 1:n_timecourse
        pcat = fit(PCA, X[:,t,:]', maxoutdim=n_components)
        P[t,:,:] = projection(pcat)'
    end

    @info "size" size(P)


    Ps = dropdims(mean(P[150:200,:,:],dims=1),dims=1)
    Xs = dropdims(mean(X[:,150:200,:],dims=2),dims=2)
    @info "Ps" size(Ps) size(Xs)

    Y = zeros(n_conditions, n_components)
    for sx in 1:n_conditions
        Y[sx,:] = Ps*Xs[sx,:]
    end
    @info "sY" size(Y)



    # plot

    ax = scatter(  Y[:,1], Y[:,2], color=colors, legend=false, markerstrokewidth=0, markersize=8, title=session[:mouseid]  )
    display(ax)

end










function predictneurons(mouseid::Symbol, nwbfile::PyObject; consistentonly=false)
    
    @info "mouseid" mouseid
    triallist = nwbdf(nwbfile.trials)
    filter!(:difficulty=>u->u=="complex",triallist)

    addcongruencycolumn!(triallist)
    consistentlabel = ["",",consistent"][1+consistentonly]
    if consistentonly
        maperfs = movingaverageperformancetrialtypes(triallist)
        maskconsistent = highperformancetrialsmask(maperfs)
        triallist = triallist[maskconsistent,:]
    end

    neuronsspiketimeslist = gettrialrelativespiketimes(triallist, nwbfile.units)         # get spikes for trials
    timestamps, instantenousfiringrates = smoothinstantenousfiringrate(neuronsspiketimeslist, config[:stimuluslength], binsize=config[:dt])     # smooth 

    ntrials,ntimestamps,nneurons = size(instantenousfiringrates)


    cols = [:degree,:freq,:context,:action,:water]
    vals1 = [45 5000 "visual" true true]
    contexts = ["visual","audio"]
    nvariables = length(cols)

    labels = ["visual" "audio" "context" "decision" "reward"]
    colors = [:blue :green :mediumvioletred :darkorange :gold]

    predictorcombinations = [ [1], [2], [4], [1,4], [2,4], [3], [1,2,3], [1,2,3,4]  ]
    labelscombinations = ["visual","audio","decision","visual+decision","audio+decision","context","visual-audio-context","visual-audio-context+reward"]
    colorscombinations = [:blue :green :darkorange :purple :olive :mediumvioletred :grey :gold]
    ncombinations = length(predictorcombinations)
    # @info "" cols(predictorcombinations[1])

    labelscontexts = ["all", "visual", "auditory"]
    trialscontexts = [trues(ntrials), triallist[!,:context] .== "visual", triallist[!,:context] .== "audio" ]
    ncontexts = length(trialscontexts)


    if config[:recalculatepredictneurons]
        R2s = zeros(ncombinations,ncontexts,ntimestamps,nneurons,3)    # individual nneurons, total R2
        # coefficients = zeros(ncombinations,ntimestamps,nneurons,3)        # coefficients for neurons
    


        # predictors = Matrix(Float64.( 2 .- (    triallist[:,cols] .== repeat(vals1,nrow(triallist))  ) ) )    # smaller number <=> class 1
        
        for (cx,trialscontext) in enumerate(trialscontexts)
            for i in eachindex(predictorcombinations)
                if cx>1 && i in [6,7,8] continue end
                # if i != 4 continue end
                X = triallist[trialscontext,cols[predictorcombinations[i]]]
                Y = instantenousfiringrates[trialscontext,:,:]
                _, R2s[i,cx,:,:,:], _ = predictvariablefromactivity(X, Y, multivariate=false,
                                                  solver=["analaytical","cg"][1+consistentonly], nfolds=10)    # "ch" "a"
            end
        end

        @save(config[:cachepath]*"subspace/predict,neurons$(consistentlabel)-$(string(mouseid)).bson", timestamps, R2s)
    else
        @load(config[:cachepath]*"subspace/predict,neurons$(consistentlabel)-$(string(mouseid)).bson", @__MODULE__, timestamps, R2s)
    end

    
    


    # plot
    R2s[R2s.<0] .= NaN
    includeincompare = [1,2,3,4,5]
    nincludeincompare = length(includeincompare)
    # axs = plot(layout=(ncombinations,1), size=(1*350, ncombinations*300), legend=false)
    axs = plot(layout=(7,3), size=(3*350, 7*300), legend=false, bottom_margin=30*Plots.px, left_margin=40*Plots.px)

    for (ix,i) in enumerate(includeincompare), cx in 1:ncontexts                   # 1:ncombinations
        ax = axs[ix,cx]
        @nolinebackground(ax, config[:stimulusstart], config[:stimulusend], config[:waterstart], bg=:black)
        # plot!(ax, timestamps, R2s[i,:,:,1], lw=2, ribbon=R2s[i,:,:,3], fillalpha=0.5, color=colorscombinations[i], alpha=0.5)
        plot!(ax, timestamps, R2s[i,cx,:,:,1], lw=1, color=colorscombinations[i], alpha=0.5)

        m = [ mean(R2s[i,cx,t,R2s[i,cx,t,:,1].>0,1]) for t in axes(R2s,3)]
        plot!(ax, timestamps, @movingaverage(m,11), lw=2, color=:white)
        if i in [1,3,4]
            plot!(axs[nincludeincompare+1,cx], timestamps, @movingaverage(m,11), lw=2, color=colorscombinations[i], label=labelscombinations[i])
        end
        if i in [2,3,5]
            plot!(axs[nincludeincompare+2,cx], timestamps, @movingaverage(m,11), lw=2, color=colorscombinations[i], label=labelscombinations[i])
        end

        ylims!(ax, 0, 1.0)
        if i==1 title!(ax,["",string(mouseid)*consistentlabel,"."][cx]*
                           "\n"*"$(labelscombinations[i])\n$(labelscontexts[cx]) context")
        else title!(ax,"$(labelscombinations[i])\n$(labelscontexts[cx]) context") end
        ylabel!(ax,"R²")
        xlabel!(ax,"time (s)")
        annotate!(ax, 4.4, 0.97, text("fr>0: $(round(mean(R2s[i,cx,151:450,:,1].>0),digits=2))", 8, :white, :right))
    end

    for k in 1:2, cx in 1:ncontexts
        ax = axs[nincludeincompare+k,cx]
        @nolinebackground(ax, config[:stimulusstart], config[:stimulusend], config[:waterstart], bg=:black)
        plot!(ax, legend=:topright, foreground_color_legend=nothing, background_color_legend=nothing)
        ylims!(ax, 0, 0.6)
        title!(ax,["visual partial","auditory partial"][k]*"\n$(labelscontexts[cx]) context")
        ylabel!(ax,"R²")
        xlabel!(ax,"time (s)")
    end



    # ax = axs[ncombinations+1]
    # for i in 1:ncombinations
    #     bar!(ax, [i], [ mean(R2s[i,151:450,:,1].>0) ], lw=2, color=colorscombinations[i], alpha=0.5, label=labelscombinations[i])
    # end
    # ylabel!(ax,"fraction of R² > 0")

    display(axs)
    savefig(joinpath(config[:resultspath],"subspace/","predict,neurons$(consistentlabel)-$(string(mouseid)).png"))

end