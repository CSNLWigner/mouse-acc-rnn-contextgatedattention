# make figures
const lowprob = config[:behaviouralmodellowprob]
const cm = cgrad([:red, :white, :green],[0,0.4,0.6,1])

function figure1()
    figurepostfix = "paradigms,behavior,model"
    @info "making figure 1" figurepostfix

    mouseids = collectsubjectids("ACC")
    nmice = length(mouseids)
    
    axs = plot(layout=@layout( [ a b{0.3w}; c d e{0.33w} ] ), size=(3*350, 2*300),legend=false)









    ax = axs[1]
    paradigmimage = load(joinpath(config[:publicationfigurespath],"parts/","experimental,paradigm,acc.png"))
    plot!(ax, paradigmimage, top_margin=40*Plots.px, left_margin=40*Plots.px, bottom_margin=70*Plots.px)
    xlims!(ax, 0,1320)          # 1320  x 495
    ylims!(ax, 0,495)
    plot!(ax, xticks=false, yticks=false, axis=false)
    @panellabel ax "A" -0.20 -0.1

    




    ax = axs[2]

    t = range(-1.5, stop=4.5, length=1000)
    yscale = 0.8
    visual = ( 3 .>=  t .>= 0) * yscale
    audio = (3 .>= t .>= 0) * yscale
    reward = ((t .>= 2.01) .| (t .< 0) ) * yscale
    ys = [3, 2, 1]
    xs = [0.2,0.2,2.2]
    for (v,x,y,c,l) in zip([visual, audio, reward], xs, ys, [:navy, :darkgreen, :red],["visual", "audio", "reward" ] )
        plot!(ax, t, y .+ v, color=c, lw=2)
        annotate!(ax, x, y+0.5, text("$(l)",c,10,:left))
    end
    xlims!(ax, -1.5, 4.5)
    ylims!(ax, 1-0.05, 1+3.05-0.2)
    plot!(ax, yticks=nothing, xticks=[0,2,3])
    xlabel!(ax, "time from stimulus onset [s]")
    @decoderbackground ax 0 3 2 :darkgrey
    @panellabel ax "B" -0.05 1.1





    ax = axs[3]
    plot!(ax, axis=false, left_margin=30*Plots.px)
    
    mouseid = :AC006
    labels = ["go congruent" "go incongruent"; "nogo congruent" "nogo incongruent"]
    nwbfile,_ = loadnwbdata(mouseid)
    triallist = nwbdf(nwbfile.trials)
    filter!(:difficulty=>u->u=="complex",triallist)
    addcongruencycolumn!(triallist)
    maperfs = movingaverageperformancetrialtypes(triallist)
    highperfs = highperformancetrialsmask(maperfs)
    contextboundary = findfirst(triallist[!,:context].==triallist[end,:context])
    boundaries = [1 contextboundary-1; contextboundary nrow(triallist)]
    choices = choicesselected(triallist)
    for sx in 1:2, congi in 1:2
        ix = (sx-1)*2+congi
        plot!(inset_subplots=(3, bbox(0, 1-ix*0.22, 1, 0.18, :bottom, :left)))
        ax = axs[5+ix]
        for cx in 1:2         # contexts
            if sx+congi==2 annotate!(ax, sum(boundaries[cx,1]),1.1, text(triallist[contextboundary-2+cx,:context]*"\ncontext",[:navy,:darkgreen][cx],:left, :bottom, 8)) end
            # plot moving averages
            trl = range(boundaries[cx,:]...)
            plot!(ax, trl, maperfs[trl,sx,congi], color=[:lightseagreen, :indianred][sx], ls=[:solid, :dash][congi], lw=2, alpha=0.8,label=nothing)
            # also plot actual successes and failures for each as markers
            for s in [false,true]   # failure, success
                ch = choices[(sx-1)*2+congi]
                mask = ch[2].==s
                scatter!(ax, ch[1][mask], zeros(sum(mask)).-0.2, marker=[:x,:o][Int(s)+1],
                            markersize=2.5, markerstrokewidth=1, color=:black, alpha=0.5, label=nothing)
            end
            # plot!(ax, legend=:right, foreground_color_legend=nothing, background_color_legend=nothing)
            annotate!(ax, 1, 0.3, text(labels[sx,congi],[:lightseagreen, :indianred][sx], :left, 6))
            plot!(ax,xaxis=false, ytickfontsize=6, ytickhalign=:left)
        end
        # plot!(ax, bottom_margin=-4*Plots.px, top_margin=-4*Plots.px)
        if sx+congi<4 plot!(ax,xticks=nothing) end
        plot!(ax, yticks=[0, 0.5, 1.0], ylims=[-0.4, 1.0], xlims=[-2,boundaries[2,2]+1])
        # plot!(ax, ytickfonthalign=:left, xtickfontvalign=:bottom)
        hline!(ax,[0.5],color=:grey, ls=:dash, label=nothing)
        # vline!(ax, [contextboundary-0.5], color=:grey, label=nothing)
    end
    plot!(inset_subplots=(3, bbox(0, 0, 1, 0.06, :bottom, :left)))
    ax = axs[10]

    bar!(ax, .! highperfs, color=:darkorange, linecolor=:darkorange, label=nothing)
    bar!(ax,    highperfs, color=:purple,     linecolor=:purple,     label=nothing)
    # vline!(ax, [contextboundary-0.5], color=:grey, label=nothing)
    annotate!(ax, 0, 1.6, text("exploratory trials", :darkorange, :bottom, :left, 6, "Helvetica Bold"))
    annotate!(ax, 0, 1.06, text("consistent trials", :purple, :bottom, :left, 6, "Helvetica Bold"))
    ylims!(ax, 0.05,1.15)
    xlims!(ax,-2,boundaries[2,2]+1, xticks=[1,boundaries[2,:]...]) # xtickfontsize=8
    xlabel!(ax, "trials")
    plot!(ax, yaxis=false)
    plot!(ax, bottom_margin=20*Plots.px)
    
    @panellabel axs[3] "C" -0.15 1.1


    


    ax = axs[4]
    colors = [:navy :darkgreen]
    plot!(ax, axis=false)
    nhighperfs = zeros(nmice,2)
    fractioncorrects = zeros(nmice,2)
    lls = zeros(nmice,2,5) # loglikelihoods (nmice, context, models)
    for (n,mouseid) in enumerate(mouseids)
        nwbfile,_ = loadnwbdata(mouseid)
        triallist = nwbdf(nwbfile.trials)
        filter!(:difficulty=>u->u=="complex",triallist)
        addcongruencycolumn!(triallist)
        maperfs = movingaverageperformancetrialtypes(triallist)
        highperfs = highperformancetrialsmask(maperfs)
        contextboundary = findfirst(triallist[!,:context].==triallist[end,:context])
        boundaries = [1 contextboundary-1; contextboundary nrow(triallist)]
        for (cx,context) in enumerate(["visual","audio"])
            maskhp = (triallist[!,:context].==context)  .&  highperfs 
            maskhpic =  maskhp .&  (triallist[!,:congruency].=="incongruent")

            # number of consistent trials
            nhighperfs[n,cx] = sum(maskhp)
            
            # fraction correct
            fractioncorrects[n,cx] = sum(triallist[maskhp,:success])/nhighperfs[n,cx]

            # models; need choices data, and model p estimates
            choices = triallist[maskhpic, :action]
            # zero parameter context opposite
            targets = triallist[maskhpic, [:degree,:freq][3-cx]] .==  [45,5000][3-cx]   # opposite
            p = clamp.(.! targets, lowprob, 1-lowprob)
            lls[n,cx,1] = mean(bernoulliloglikelihood(choices,p))

            # zero parameter context aware
            targets = triallist[maskhpic, [:degree,:freq][cx]] .==  [45,5000][cx]   # same
            p = clamp.(.! targets, lowprob, 1-lowprob)
            lls[n,cx,2] = mean(bernoulliloglikelihood(choices,p))
             
            # one parameter mean bias
            lls[n,cx,3] = mean(bernoulliloglikelihood(choices,mean(choices)))
            
            # one parameter bias, based on the targets from context aware
            targets = triallist[maskhpic, :water]    # go or nogo contextual
            p = [modelbias(targets,β) for β in -1+2*lowprob:lowprob*2:1-lowprob]
            clamp!.(p, lowprob, 1-lowprob)
            llsbias = dropdims(mean(hcat([bernoulliloglikelihood.(choices,pix) for pix in p]...),dims=1),dims=1)
            lls[n,cx,4] = maximum(llsbias)
            # βm = (-1+2*lowprob:lowprob*2:1-lowprob)[argmax(llsbias)]
            
            # one parameter lapse
            targets = triallist[maskhpic, :water]    # go or nogo contextual
            p = [modellapse(targets,λ) for λ in lowprob:lowprob:1-lowprob]
            clamp!.(p, lowprob, 1-lowprob)
            llslapse = dropdims(mean(hcat([bernoulliloglikelihood.(choices,pix) for pix in p]...),dims=1),dims=1)
            lls[n,cx,5] = maximum(llslapse)
            # λm = (lowprob:lowprob:1-lowprob)[argmax(llslapse)]

            # @info "" mouseid context lls[n,cx,1:2] lls[n,cx,3:5] βm λm
            # display(plot([-1+2*lowprob:lowprob*2:1-lowprob lowprob:lowprob:1-lowprob],[llsbias llslapse], label=["bias" "lapse"], lw=2, title="$mouseid $context"))

        end
    end

    if nmice>4 
        @info "lls NaN" sum(isnan.(lls))
        lls = lls[1:4,:,:]
        # replace!(lls, NaN=>missing)
        # @info "corrected" sum(ismissing.(lls))
    end

    ix = 1
    plot!(inset_subplots=(4, bbox(0.1, 1-ix*0.25, 1, 0.23, :bottom, :left)))
    ax = axs[10+ix]
    bar!(ax, (1:nmice).+[-0.165 +0.165 ], nhighperfs, bar_width=0.33, color=colors, linecolor=colors, label=nothing)
    xlims!(ax, 0, nmice+1)
    ylims!(ax, 0, 35)
    plot!(ax, xticks=false, yticks=[0,10,20,30], ytickfontsize=6, ylabel="number\nof trials", ylabelfontsize=7, yguidehalign=:left)

    ix = 2
    plot!(inset_subplots=(4, bbox(0.1, 1-ix*0.25, 1, 0.23, :bottom, :left)))
    ax = axs[10+ix]
    bar!(ax, (1:nmice).+[-0.165 +0.165 ], fractioncorrects, bar_width=0.33,  color=colors, linecolor=colors, label=nothing)
    xlims!(ax, 0, nmice+1)
    ylims!(ax, 0, 1.05)
    plot!(ax, xticks=false, yticks=[0,1], ytickfontsize=6, ylabel="fraction\ncorrect", ylabelfontsize=7, yguidehalign=:left)

    ix = 3
    plot!(inset_subplots=(4, bbox(0.1, 1-ix*0.25, 1, 0.23, :bottom, :left)))
    ax = axs[10+ix]
    bar!(ax, (1:nmice).+range(-0.3,0.3,length=4)', reshape(permutedims(lls[:,:,1:2],(1,3,2)),:,4), bar_width=0.15, 
           color=[:white colors[1] :white colors[2]], linecolor=[colors[1] colors[1] colors[2] colors[2]], label=nothing)
    xlims!(ax, 0, nmice+1)
    # ylims!(ax, -5.05, 0.05)
    plot!(ax, xticks=(1:nmice,[]), xlabel="mice", ytickfontsize=6, ylabel="log\nlikelihood", ylabelfontsize=7, yguidehalign=:left)

    ix = 4
    plot!(inset_subplots=(4, bbox(0.1, 1-ix*0.25, 1, 0.23, :bottom, :left)))
    ax = axs[10+ix]
    colorlist = [colors[1] colors[1] colors[1] colors[2] colors[2] colors[2]]
    bar!(ax, (1:nmice).+range(-0.30,0.30,length=6)', reshape(permutedims(lls[:,:,3:5],(1,3,2)),:,6), bar_width=0.10, 
           color=colorlist, linecolor=colorlist, alpha=[0.3 0.6 1 0.3 0.6 1], label=nothing)
    xlims!(ax, 0, nmice+1)
    ylims!(ax, -0.85, 0.05)
    plot!(ax, xticks=(1:nmice,[]), xlabel="mice", ytickfontsize=6, ylabel="log\nlikelihood", ylabelfontsize=7, yguidehalign=:left)

    @panellabel axs[4] "D" -0.15 1.1


    






    ax = axs[5]

    plot!(ax,axis=false,xlims=(0,20),ylims=(0,20))
    
    plot!(ax, [10,11,11,10,10],[5,5,15,15,5], lw=3, color=:black)
    
    plot!(ax,[(6,8),(9,8)],arrow=arrow(:closed), lw=3, color=:navy)
    annotate!(ax, 4, 8, text("visual", 10, :right, :navy, "Helvetica Bold"))
    
    plot!(ax,[(6,10),(9,10)],arrow=arrow(:closed), lw=3, color=:darkgreen)
    annotate!(ax, 4, 10, text("audio", 10, :right, :darkgreen, "Helvetica Bold"))
    
    plot!(ax,[(6,17),(9,13)],arrow=arrow(:closed), lw=3, color=:red)
    annotate!(ax, 4, 19, text("previous reward", 10, :left, :red, "Helvetica Bold"))

    plot!(ax,[(12,11),(15,11)],arrow=arrow(:closed), lw=3, color=:darkorange)
    annotate!(ax, 16, 11, text("decision", 10, :left, :darkorange, "Helvetica Bold"))

    plot!(ax, map(u->(u[1]+10.5,u[2]+3.9), Plots.partialcircle(0/360*2*pi,30/360*2*pi,30,2)), color=:black, lw=3)
    plot!(ax, map(u->(u[1]+10.5,u[2]+3.9), Plots.partialcircle(150/360*2*pi,360/360*2*pi,30,2)), color=:black, lw=3)
    plot!(ax, map(u->(u[1]+10.65,u[2]+4.35), Plots.partialcircle(150/360*2*pi,155/360*2*pi,2,1.75)), color=:black, lw=3, arrow=(:tail,:closed))
    annotate!(ax, 10.5, 0, text("previous state", 10, :center, :bottom, :black, "Helvetica Bold"))
    

    @panellabel ax "E" -0.02 1.1




    plot!(axs, tick_direction=:out, xgrid=false, ygrid=false)#, ytickfonthalign=:left, xtickfontvalign=:bottom, xguidevalign=:top)
    
    display(axs)
    if config[:publishfigures]
        savefig(joinpath(config[:publicationfigurespath],"Figure1-$(figurepostfix).png"))
        savefig(joinpath(config[:publicationfigurespath],"Figure1-$(figurepostfix).pdf"))
    end

end
















function figure2()
    figurepostfix = "irrelevant,suppression"
    @info "making figure 2" figurepostfix



    axs = plot(layout=@layout([[a b c d; e f g h] i{0.2w} ]),size=(5*350, 2*300), legend=false,
               left_margin=20*Plots.px, bottom_margin=15*Plots.px, top_margin=20*Plots.px, right_margin=20*Plots.px)




    



    # panels first 2×2 blocks
    # mice suppression

    
    mouseidsv1 = collectsubjectids("V1")
    nmicev1 = length(mouseidsv1)
    mousev1ind = 1:nmicev1

    mouseidsacc = collectsubjectids("ACC")
    nmiceacc = length(mouseidsacc)
    mouseaccind = nmicev1+1:nmicev1+nmiceacc


    labelsstimulus = ["visual","audio"]
    labelscontexts = ["visual","audio"]
    labelsrelevancies = ["relevant" "irrelevant"; "irrelevant" "relevant"]
    colors = [:deepskyblue :green; :blue :lime]     # relevancy x stimulus

    timestamps = []
    accuracieslists = []
    for mouseids in [mouseidsv1,mouseidsacc]
        for mouseid in mouseids
            @load(config[:cachepath]*"subspace/decode,relevancy-$(string(mouseid)).bson", @__MODULE__, timestamps, accuracies)
            push!(accuracieslists, accuracies[1,:,:,:,:])     # collect only the decoder trained and tested on all trials
        end
    end
    accuracieslists = vcat(reshape.(accuracieslists,1,size(accuracieslists[1])...)...)


    sm = 31 # smoothing window
    for (bx,mouseind) in enumerate((mousev1ind,mouseaccind))
        for (stimulusindex,stimulus) in enumerate(labelsstimulus)
            ax = axs[(stimulusindex-1)*4+bx] # axs[stimulusindex,bx]
            @decoderbackground(ax, config[:stimulusstart], config[:stimulusend], config[:waterstart], bg=:white)
            for contextindex in ([2,1],[1,2])[stimulusindex]
                context = labelscontexts[contextindex]
                m = @movingaverage(dropdims(mean(accuracieslists[mouseind,contextindex,stimulusindex,:,1],dims=1),dims=1),sm)
                e = @movingaverage(dropdims(std(accuracieslists[mouseind,contextindex,stimulusindex,:,1],dims=1),dims=1)/sqrt(length(mouseind))+
                    dropdims(mean(accuracieslists[mouseind,contextindex,stimulusindex,:,3],dims=1),dims=1), sm)
                plot!(ax, timestamps, m, ribbon=e, lw=2, fillalpha=0.3, color=colors[contextindex,stimulusindex], alpha=0.8,
                        label=labelsrelevancies[contextindex,stimulusindex]*" ("*context*" context)")
                if bx==1 ylabel!(ax, "accuracy"); plot!(ax, left_margin=30*Plots.px) end
                if stimulusindex==2 xlabel!(ax, "time from stimulus onset [s]") end
            end
            xlims!(ax,-1.2,4.2)
            ylims!(ax,0.45,1.05)
            plot!(ax,legend=:topleft, foreground_color_legend=nothing, background_color_legend=nothing)
            title!(ax, [["V1\n","ACC\n"][bx],""][stimulusindex]*stimulus*" stimulus")
            if stimulusindex==1 @panellabel ax ["A","B"][bx] -0.25 1.15 end
        end
    end















    # right 2×2 block: mouse performance and suppression

    # accuracies  is (nstates,nmodalities,nrelevancies,ntimestamps,3)               (last is stats)
    # coefficients  is (nstates,nmodalities,nrelevancies,ntimestamps,nneurons+1)      # last is intercept
    # projections is (nstates,nmodalities,nrelevancies,2,ntimestamps,3)             (2 is gonogo, last is stats)
    accuraciesallareas = []
    # colors = [:fuchsia :gold; :purple  :darkorange]       #  vis-aud × consistency
    colors = [:purple :darkorange]                         # consistency
    alphas = [1.0 0.3]
    relevancylabels = ["relevant","irrelevant"]
    consistencylabels = ["consistent","exploratory"]
    for (mx,brainarea) in enumerate(["V1","ACC"])
    
        mouseids = collectsubjectids(brainarea)
        nmice = length(mouseids)
        timestamps = nothing
        accuraciesall = []
        for mouseid in mouseids
            @load(joinpath(config[:cachepath],"subspace/","suppressionbehaviour-$(string(mouseid)).bson"), @__MODULE__,
                           timestamps, accuracies)
            push!(accuraciesall, accuracies)
        end
        ntimestamps = length(timestamps)
        accuraciesall = @catvecleftsingleton accuraciesall
        push!(accuraciesallareas, accuraciesall)

        for sx in 1:2        # stimulus modalities
            ax = axs[(sx-1)*4+2+mx]
            vspan!(ax,[config[:stimulusstart],config[:stimulusend]], color=:grey, alpha=0.3, label=nothing)
            vline!(ax,[config[:waterstart]],color=:white, alpha=0.5, lw=2, label=nothing)
            hline!(ax,[0.5],color=:grey, ls=:dash, label=nothing)
            for bx in 1:2        # consistency
                for rx in 1:2    # relevancies
                    m = @movingaverage(dropdims(mean(accuraciesall[:,bx,sx,rx,:,1],dims=1),dims=1),sm)
                    # e = @movingaverage(dropdims(std(accuraciesall[:,bx,sx,rx,:,1],dims=1),dims=1)/sqrt(nmice)+
                    #     dropdims(mean(accuraciesall[:,bx,sx,rx,:,3],dims=1),dims=1), sm)
                    # normal error bars have no meaning here, beacuse we want to compare within individual mice
                    # so we show here the CV errors improved by the mouse-average
                    e = @movingaverage(dropdims(mean(accuraciesall[:,bx,sx,rx,:,3],dims=1)/sqrt(nmice),dims=1), sm)
                    plot!(ax,timestamps,m,ribbon=e,lw=3,color=colors[bx], alpha=alphas[rx], fillalpha=0.1,
                        label=relevancylabels[rx]*", "*consistencylabels[bx])
                end
            end
            plot!(ax,legend=:topright, foreground_color_legend=nothing, background_color_legend=nothing)
            # xlims!(ax,-1.2,4.2)
            ylims!(ax,0.3,1.25)
            yticks!(ax,0.5:0.25:1)
            if mx==1 ylabel!(ax, "accuracy"); plot!(ax,left_margin=50*Plots.px) end
            title!(ax, ["$(brainarea)\n",""][sx]*labelsstimulus[sx]*" stimulus")
            if sx==2 xlabel!(ax, "time from stimulus onset [s]") end
            plot!(ax, bottom_margin=40*Plots.px)
            if sx==1 @panellabel ax ["C","D"][mx] -0.25 1.15 end
        end
    end





    ax = axs[9]
    colors = [ :purple :darkorange ]     # brainarea × consistency
    # create the difference between relevant and irrelevant projections
    # accuraciesallareas is a (narea)(nmice,nstates,nmodalities,nrelevancies,ntimestamps,3)
    timerange = (timestamps .>= config[:stimulusstart]+0.5) .& (timestamps .< config[:waterstart])
    # timerange = (timestamps .>= config[:stimulusstart]+0.5) .& (timestamps .< config[:stimulusend] + config[:posteventpadding])
    @info "" sum(timerange)
    tickpoints = Float64[]
    ps = Float64[]
    for (mx,brainarea) in enumerate(["V1","ACC"])
        # individual mice
        d = accuraciesallareas[mx][:,:,:,1,timerange,1] .- accuraciesallareas[mx][:,:,:,2,timerange,1]
        d = reshape(d,size(d,1)*size(d,2),:)            # reshape to (mice*var, timestamps)
        mouseind = (mousev1ind,mouseaccind .- mouseaccind[begin] .+ mousev1ind[end]*2 .+ 3)[mx]
        barind = [mouseind; mouseind.-mouseind[begin].+mouseind[end].+1] + repeat([0.05,-0.05],length(mouseind))
        # all mice
        ma = similar(accuraciesallareas[mx][1,:,:,:,:,1])
        for bx in 1:2, sx in 1:2, rx in 1:2
                ma[bx,sx,rx,:] = @movingaverage(dropdims(mean(accuraciesallareas[mx][:,bx,sx,rx,:,1],dims=1),dims=1),5)
        end
        
        for sx in 1:2
            d = ma[:,sx,1,timerange] .- ma[:,sx,2,timerange]
            # d = reshape(d,size(d,1),:)            # reshape to (variables, timestamps)
            mouseind = [[1],[4]][sx]
            barind = [mouseind; mouseind.-mouseind[begin].+mouseind[end].+1] + repeat([0.05,-0.05],length(mouseind))   .+ (mx-1)*8
            push!(tickpoints, mean(barind))
            boxplot!(   ax, barind', d', color=colors, alpha=mean(alphas),
                        outliers=false, whisker_range=1, notch=true,
                        label=[  nothing, [["consistent" "exploratory"] repeat([nothing nothing],1,length(mouseind)-1)]  ][Int(sx+mx==4)+1]   )
            push!(ps, pvalue(OneSampleTTest(d[1,:],d[2,:])))
        end
    end
    @info "p" ps

    pss = ifelse.(ps .< 0.0001, "***", "ns")
    psr = pss .*" \n" .* ["p=0.09", "p=0.81", "p<10⁻²⁰", "p<10⁻²⁰"]
    hline!(ax,[0],color=:grey, ls=:dash, label=nothing)
    plot!(ax, legends=:topleft, foreground_color_legend=nothing, background_color_legend=nothing)
    plot!(ax, left_margin=50*Plots.px)
    # annotate!(ax, tickpoints, -0.08, text.(ps .< 0.05, ["*" "ns"]), fontsize=12, halign=:center, valign=:bittin, color=:black)
    annotate!(ax, tickpoints, [0.3,0.3,0.6,0.6], text.( psr, 8, :black, :top, :center))#, fontsize=6, halign=:center, valign=:bittin, color=:black)
    # do the above annotate with just 2 digits and scientific notation of the p-values
    

    xticks!(ax,tickpoints,["V1\nvisual", "V1\naudio", "ACC\nvisual", "ACC\naudio"])
    xlims!(ax,0,14)
    yticks!(ax,[0,0.2,0.4])
    ylims!(ax,-0.1,0.65)

    # title!(ax,"mouse")
    ylabel!(ax, "accuracy difference\nrelevant - irrelevant")

    @panellabel ax "E" -0.08 1.055











    # @panellabels axs [5,6,7,8,9]
    
    plot!(axs, tick_direction=:out, xgrid=false, ygrid=false)#, ytickfonthalign=:left, xtickfontvalign=:bottom, xguidevalign=:top)
    
    display(axs)
    if config[:publishfigures]
        savefig(joinpath(config[:publicationfigurespath],"Figure2-$(figurepostfix).png"))
        savefig(joinpath(config[:publicationfigurespath],"Figure2-$(figurepostfix).pdf"))
    end




end




function figure3()
    
    figurepostfix = "mouse,contextinference,outcomehistory,geometry"
    @info "making figure 3" figurepostfix


    mouseids = collectsubjectids("ACC")
    nmice = length(mouseids)


    
    axs = plot(layout=(2,3), size=(3*350, 2*300), bottom_margin=20*Plots.px, left_margin=20*Plots.px, grid=false, legend=false)




    ax = axs[1,1]

    plot!(ax, left_margin=30*Plots.px, top_margin=30*Plots.px)
    accuraciesmice = zeros(nmice,5,600,3)
    timestamps = nothing
    for (n,mouseid) in enumerate(mouseids)
        @load(config[:cachepath]*"subspace/decode,variables-$(string(mouseid)).bson", @__MODULE__, timestamps, accuracies, coefficients)
        accuraciesmice[n,:,:,:] = accuracies
    end
    ntimestamps = length(timestamps)

    sm = 31 # smoothing window
    cx = 3  # context index
    @decoderbackground(ax, config[:stimulusstart], config[:stimulusend], config[:waterstart], bg=:white)
    for n in 1:nmice
        m = MathUtils.convolve( accuraciesmice[n,cx,:,1],ones(sm)./sm)
        plot!(ax, timestamps, m, lw=1, color=:grey, alpha=0.3, label=nothing)
    end
    m = MathUtils.convolve( dropdims(mean(accuraciesmice[:,cx,:,1],dims=1),dims=1), ones(sm)./sm)
    semcv = dropdims(mean(accuraciesmice[:,cx,:,3],dims=1),dims=1)
    e = MathUtils.convolve( dropdims(std(accuraciesmice[:,cx,:,1],dims=1),dims=1)/sqrt(nmice)+semcv, ones(sm)./sm)
    plot!(ax, timestamps, m, ribbon=e, lw=2, fillalpha=0.3, color=:mediumvioletred, label=nothing)
    xlims!(ax,-1.2,4.2)
    ylims!(ax,0.45,1.05)
    ylabel!(ax, "context accuracy")
    xlabel!(ax, "time [s]")





  


    ax = axs[1,2]
    # context in consistent and exploratory blocks
    accuraciesmice = zeros(nmice,2,600,3)
    for (n,mouseid) in enumerate(mouseids)
        @load(joinpath(config[:cachepath],"subspace/","cognitivesurplus-$(string(mouseid)).bson"), @__MODULE__, accuracies, coefficients)
        accuraciesmice[n,:,:,:] = accuracies
    end

    cognitivesurplus = accuraciesmice[:,1,:,1]-accuraciesmice[:,2,:,1]
    # cognitivesurplussem = accuraciesmice[:,1,:,3]+accuraciesmice[:,2,:,3]
    # cognitivesurplustimeaverage = dropdims(mean(cognitivesurplus,dims=2),dims=2)
    
    # @info "cognitivesurplus" mouseids nmice size(accuraciesmice) cognitivesurplustimeaverage
    
    # plot!(ax, timestamps, [@movingaverage(cognitivesurplus[n,:],sm) for n in 1:nmice], color=:grey, lw=2, alpha=0.5, label=nothing)
    # hline!(ax, [0], color=:grey, ls=:dash, lw=2)

    # xlims!(ax,-1.2,4.2)
    # ylims!(ax,-0.4,0.4)
    # ylabel!(ax, "accuracy difference\nconsistent - exploratory")
    # xlabel!(ax, "time [s]")
    
    
    
    boxplot!(ax, collect(1:nmice)', cognitivesurplus', color=:rebeccapurple, notch=true, label=nothing)

    boxplot!(ax, 6*ones(1,1),reshape(cognitivesurplus,nmice*ntimestamps,:), color=:rebeccapurple, alpha=0.7, notch=true, label=nothing)
    hline!(ax, [0], color=:grey, ls=:dash, lw=2)
    xticks!(ax, [1:nmice ; nmice+2], ["","","mice","","all"])
    ylims!(ax, -0.46, 0.46)
    ylabel!(ax, "context accuracy difference\nconsistent - exploratory")
    
    
    



    
    ax = axs[1,3]
    # outcome history mice
    nlookback = 3
    vx = 3
    contexts = ["visual","audio"]
    colors = [:lightseagreen,:olive]
    accuracieslists = []
    timestamps = -1.495:0.01:4.495
    ntimestamps = length(timestamps)
    for mouseid in mouseids
        @load(joinpath(config[:cachepath],"subspace/","outcomehistory-$(string(mouseid)).bson"), @__MODULE__, accuracieslist, coefficientslist)
        push!(accuracieslists, accuracieslist)
    end
    n_example = 4
    sm = 51
    accuracy = reshape(permutedims(accuracieslists[n_example][:,vx,end:-1:begin,:,:],(1,3,2,4)),2,nlookback*ntimestamps,3)
    timestampslookback = [timestamps; timestamps.+6; timestamps.+2*6]
    for cx in eachindex(contexts)
        plot!(ax, timestampslookback, @movingaverage(accuracy[cx,:,1],sm),
              ribbon=@movingaverage(accuracy[cx,:,3],sm), lw=2, fillalpha=0.3, color=colors[cx], label=contexts[cx]*" context")
    end
    for h in 1:nlookback
        @decoderbackground(ax, config[:stimulusstart]+(nlookback-h)*6, config[:stimulusend]+(nlookback-h)*6, config[:waterstart]+(nlookback-h)*6, bg=nothing)
    end
    xlims!(ax,-1.2,4.2+12)
    ylims!(ax,0.45,1.05)
    plot!(ax,legend=:topright, foreground_color_legend=nothing, background_color_legend=nothing, bottom_margin=50*Plots.px)
    xlabel!(ax, "time from reference trial [s]")
    ylabel!(ax, "decision accuracy")




   
    ax = axs[2,1]
    # geometry

    mouseid = :AC006
    @load(config[:cachepath]*"subspace/lowdimortho,viau-$(string(mouseid)).bson", @__MODULE__, triallist, S, b)

    b .*= 0.08               # display size
    o = -0.24                # offset
    colors = [:purple :darkcyan; :darkorange :red ]
    w = 0.03  # annotation width

    # plot basis vectors
    plot!(ax,[0,b[1,1]].+o,[0,b[2,1]].+o,color=:navy,lw=5)
    plot!(ax,[0,b[1,2]].+o,[0,b[2,2]].+o,color=:darkgreen,lw=5)
    # plot mean early projections
    for (vix,vi) in enumerate([45,135]), (aux,au) in enumerate([5000,10000])
        mask = (triallist[!,:degree].==vi) .& (triallist[!,:freq].==au)
        scatter!(ax,S[mask,1],S[mask,2], color=colors[vix,aux], markerstrokewidth=0, markersize=3)         # label=string(vi)*"° "*string(au)*" Hz"

        # legend annotation
        scatter!(ax, -0.13 .+ [0] .+ (vix-1)*w,  0.17 .+ [0] .+ (aux-1)*w, color=colors[vix,aux], markerstrokewidth=0, markersize=3)
    end
    annotate!(ax, -0.13+w/2, 0.17+w+0.02, "visual\ngo nogo", font(pointsize=6, color=:black, halign=:center, valign=:bottom))
    annotate!(ax, -0.13-w+0.005, 0.17+w/2,  "audio nogo\ngo", font(pointsize=6, color=:black, halign=:right, valign=:center))


    xlims!(ax,-0.27,0.27)
    ylims!(ax,-0.27,0.27)
    xlabel!(ax,"visual DV")
    ylabel!(ax,"audio DV\n(orthogonal projection)")







    # angles
    # load each mouse decoder coefficients for all variables
    mouseids = collectsubjectids("ACC")
    nmice = length(mouseids)
    nvariables = 5
    timeindices = 151:200   # stim start to end
    angles = zeros(nmice, nvariables, nvariables)
    for (n,mouseid) in enumerate(mouseids)
        @load(joinpath(config[:cachepath],"subspace/","decode,variables-$(string(mouseid)).bson"), @__MODULE__, coefficients)
        for k in 1:nvariables, j in k+1:nvariables
            if k==3 & j==4
                meancoefficients = dropdims(mean(coefficients[:,401:450,1:end-1],dims=2),dims=2)
            else
                meancoefficients = dropdims(mean(coefficients[:,151:200,1:end-1],dims=2),dims=2)
            end
            angles[n,k,j] = angleofvectors(meancoefficients[k,:],meancoefficients[j,:])
        end
    end
    # angles .*= π/180


    # stats
    ps = []
    for angle in (angles[:,1,2],angles[:,1,3], angles[:,2,3], angles[:,3,4])
        @info "" pvalue(OneSampleTTest(angle, 90))
        push!(ps,pvalue(OneSampleTTest(angle, 90)))
    end

    labels = ["visual","audio","context","decision","reward"]
    micercoords = 5*ones(nmice)
    

    ax = axs[2,2]
    scatter!(ax, angles[:,1,2], micercoords, color=:darkcyan, markerstrokewidth=0, label="visual ⋅ audio, p=$(round(ps[1],digits=2))")
    vline!(ax, [90], color=:grey, ls=:dash,label=nothing)
    xlims!(ax, 0, 180)
    ylims!(ax,0,15)
    plot!(ax, legend=:topleft, foreground_color_legend=nothing, background_color_legend=nothing, yaxis=false)
    xlabel!(ax, "angle between DVs [°]")

    ax = axs[2,3]
    scatter!(ax, angles[:,1,3], micercoords.+1, color=:purple, markerstrokewidth=0, label="context ⋅ visual, p=$(round(ps[2],digits=2))")
    scatter!(ax, angles[:,2,3], micercoords, color=:olive,  markerstrokewidth=0, label="context ⋅ audio, p=$(round(ps[3],digits=2))")
    scatter!(ax, angles[:,3,4], micercoords.-1, color=:darkgoldenrod,  markerstrokewidth=0, label="context ⋅ decision, p=$(round(ps[4],digits=2))")
    vline!(ax, [90], color=:grey, ls=:dash,label=nothing)
    xlims!(ax, 0, 180)
    ylims!(ax,0,15)
    plot!(ax, legend=:topleft, foreground_color_legend=nothing, background_color_legend=nothing, markerstrokewidth=0, yaxis=false)
    xlabel!(ax, "angle between DVs [°]")










    @panellabels axs
    
    plot!(axs, tick_direction=:out, xgrid=false, ygrid=false)#, ytickfonthalign=:left, xtickfontvalign=:bottom, xguidevalign=:top)
    
    display(axs)
    if config[:publishfigures]
        savefig(joinpath(config[:publicationfigurespath],"Figure3-$(figurepostfix).png"))
        savefig(joinpath(config[:publicationfigurespath],"Figure3-$(figurepostfix).pdf"))
    end




end










function figure4()

    figurepostfix = "rnn,contextinference,outcomehistory,suppression"
    @info "making figure4" figurepostfix


    axs = plot(layout=(2,3), size=(3*350, 2*300), bottom_margin=20*Plots.px, left_margin=20*Plots.px, grid=false, legend=false)
    
    # model panels
    machineparameters = YAML.load_file("params-rnn.yaml"; dicttype=Dict{Symbol,Any})




    ax = axs[1,1]
    plot!(ax)
    modeids = 101:120
    ntimepoints = 75
    snapshots = machineparameters[:snapshots]
    contextaccuracymodels = zeros(length(modeids),length(snapshots),ntimepoints,2)
    fractioncorrectsmodels = zeros(length(modeids),length(snapshots),ntimepoints,4)
    Ts = nothing
    decisionpoints =  nothing
    for (mx,mid) in enumerate(modeids)
        modelidexample = lpad(mid,4,"0")
        @load(joinpath(config[:modelresultspath],"analysis/decoders/", "decoders-$(modelidexample).bson"), @__MODULE__,
              fractioncorrects, contextaccuracy, Ts, decisionpoints)
        fractioncorrectsmodels[mx,:,:,:] = fractioncorrects
        contextaccuracymodels[mx,:,:,:] = contextaccuracy
    end
    cm = dropdims(mean(contextaccuracymodels[:,:,decisionpoints[end],1],dims=1),dims=1)
    ce = dropdims(std(contextaccuracymodels[:,:,decisionpoints[end],1],dims=1),dims=1)/sqrt(length(modeids))
    fm = dropdims(mean(fractioncorrectsmodels[:,:,decisionpoints[end],:],dims=1),dims=1)
    fe = dropdims(std(fractioncorrectsmodels[:,:,decisionpoints[end],:],dims=1),dims=1)/sqrt(length(modeids))
    plot!(ax,snapshots, cm, ribbon=ce, color=:mediumvioletred, lw=2, alpha=0.5, fillalpha=0.1, label="context decoder")
    plot!(ax,snapshots, fm[:,2], ribbon=fe[:,2], color=:darkorange, lw=2, alpha=0.5, fillalpha=0.1, label="decision, congruent")
    plot!(ax,snapshots, fm[:,3], ribbon=fe[:,3], color=:darkorange, lw=2, alpha=0.5, fillalpha=0.1, ls=:dash, label="decision, incongruent")

    plot!(ax, legend=:bottomright, foreground_color_legend=nothing, background_color_legend=nothing)
    plot!(ax, top_margin=30*Plots.px, bottom_margin=30*Plots.px)
    ylims!(ax,0.48,1.05)
    hline!(ax, [0.5], color=:grey, ls=:dash, label=nothing)
    xlabel!(ax, "training epoch")
    ylabel!(ax, "acc. / frac. corr.")



    ax = axs[1,2]
    # plot an example of congruent and incongruen trials timecourse with context and decision
    # this particular trial is saved as congruent = [false, true, true, false, true]
    modelidexample = "0114"
    incongruentsequence = [false, false, true, false, true]
    @load(joinpath(config[:modelresultspath],"analysis/decoders/", "decoders-$(modelidexample).bson"), @__MODULE__,
          fractioncorrects, contextaccuracy, Ts, decisionpoints)
    times = (1:Ts[end][end]*5) .- Ts[end][end]*(5-1)
    for (gx,g) in enumerate(incongruentsequence)
        ind = (gx-1)*Ts[end][end]+1:min(gx*Ts[end][end]+1,length(times))
        plot!(ax,times[ind],fractioncorrects[end,ind,3], lw=2, color=:darkorange, ls=[:solid,:dash][Int(g)+1])
    end
    plot!(ax,times,contextaccuracy[end,:,2], lw=2, color=:mediumvioletred)
    for s in -60:15:0 
        @decoderbackground(ax, s+Ts[2][begin], s+Ts[3][end], s+Ts[3][begin], bg=nothing)
    end
    ylims!(ax,0.45,1.05)
    ylabel!(ax, "acc. / frac. corr.")
    xlabel!(ax, "time from reference trial [AU]")










    ax = axs[1,3]
    nlookback = 3
    contexts = ["visual","audio"]
    colors = [:lightseagreen,:olive]
    vx = 3
    modelid = "0116"
    sm = 51
    Ts = Ts = [1:3,4:9,10:12,13:15]
    timestamps = 1:Ts[end][end] # model timepoints
    ntimestamps = Ts[end][end]
    @load(joinpath(config[:modelanalysispath],"outcomehistory/","outcomehistory-$(string(modelid)).bson"), @__MODULE__, accuracieslist, coefficientslist)
    accuracy = reshape(permutedims(accuracieslist[:,vx,end:-1:begin,:,:],(1,3,2,4)),2,nlookback*ntimestamps,3)
    timestampslookback = [timestamps; timestamps.+Ts[end][end]; timestamps.+2*+Ts[end][end]]
    for cx in eachindex(contexts)
        plot!(ax, timestampslookback, accuracy[cx,:,1], ribbon=accuracy[cx,:,3], 
                  lw=2, fillalpha=0.3, color=colors[cx], label=contexts[cx]*" context")
        for h in 1:nlookback
            @decoderbackground(ax, Ts[2][begin]+(nlookback-h)*Ts[end][end], Ts[3][end]+(nlookback-h)*Ts[end][end], Ts[2][end]+(nlookback-h)*Ts[end][end], bg=nothing)
        end
    end
    # xlims!(ax,-1.2,4.2+12)
    ylims!(ax,0.45,1.05)
    plot!(ax,legend=:bottom, foreground_color_legend=nothing, background_color_legend=nothing)
    xlabel!(ax, "time from reference trial [AU]")
    ylabel!(ax, "decision accuracy")

    

    





    # right 3×1 block
    # model suppression

    machineparameters = YAML.load_file("params-rnn.yaml"; dicttype=Dict{Symbol,Any})
    sequencelength = 5
    nhidden = 30
    Ts = [1:3,4:9,10:12,13:15]
    ntimecourse = Ts[end][end]
    filenamebase = machineparameters[:modeltype]*"-s$sequencelength-h$nhidden"

    nbest = 5

    @load(joinpath(config[:modelanalysispath], "suppression", filenamebase*"-suppressiontraces-all.bson"), @__MODULE__,
                   abstractcells, performances)
    @assert(size(abstractcells,3)==nbest, "nbest $(size(abstractcells,3)) ≠ $nbest mismatch")
    
    nmodels = size(abstractcells,1)                             

    # plot the abstract go and nogo cells averaged over models

    labelstimulus = ["visual","audio"]
    # colors = [:dodgerblue, :lime]

    # find the abstract go and nogo cells with the highest enhancement - suppression
    h = abstractcells[:,end-ntimecourse+1:end,:,:,:,:]      # last trial only in each sequence
    # calculate the difference between the relevant and irrelevant responses for each cell, separately for go and nogo cells:
    es = h[:,Ts[3][begin],:,:,1,1] + h[:,Ts[3][begin],:,:,2,2] - h[:,Ts[3][begin],:,:,1,2] - h[:,Ts[3][begin],:,:,2,1]
    # find the indices of the best go and nogo cells
    s = argmax(es,dims=2)

    
    labelscontexts = ["visual","audio"]
    labelsrelevancies = ["relevant" "irrelevant"; "irrelevant" "relevant"]
    colors = [:deepskyblue :green; :blue :lime]     # relevancy x stimulus
    maxcells = 1
    for (sx,stimulus) in enumerate(labelstimulus)
        ax = axs[2,sx]
        vspan!(ax,[Ts[2][begin],Ts[3][end]].-Ts[2][begin], color=:grey, alpha=0.3, label=nothing)
        vline!(ax,[Ts[3][begin]].-Ts[2][begin],color=:white, alpha=0.5, lw=2, label=nothing)
        hline!(ax,[0],color=:grey, ls=:dash, label=nothing)
    
        for cx in ([2,1],[1,2])[sx]
            context = labelscontexts[cx]
        
            d = similar(h[:,:,1:2*maxcells,1,cx,sx])
            for k in 1:nmodels   d[k,:,:] = cat( [ h[k,:,s[k,maxcells,gx][2],gx,cx,sx] for gx in 1:2 ]..., dims=2)  end
            d = reshape(permutedims(d, (1,3,2)), (nmodels*2*maxcells,ntimecourse))

            m = dropdims(mean(d,dims=(1)), dims=(1))
            e = dropdims(std(d,dims=(1)), dims=(1)) ./ sqrt(nmodels*2*maxcells)
            plot!(ax, (1:ntimecourse) .- Ts[2][begin], m, ribbon=e, color=colors[cx,sx], lw=2,
                      label=labelsrelevancies[cx,sx]*" ("*context*" context)")

        end
        plot!(ax, legend=:topleft, foreground_color_legend=nothing, background_color_legend=nothing)
        title!(ax,labelstimulus[sx]*" stimulus")
        if sx==1 ylabel!(ax,"activity") end
        ylims!(ax,-0.5,0.5)
        xlims!(ax,Ts[1][begin]-Ts[2][begin],Ts[end][end]-Ts[2][begin])
        xlabel!(ax,"time from stimulus onset [AU]")

    end

    
    
    # plot    enhancement - suppression   vs    performance
    # performances dimensions: (nmodels, context, go/nogo, congruency)
    p = minimum(performances[:,:,:,2],dims=(2,3))[:,1,1] # incongruent only
    ax = axs[2,3]

    ess = vcat( [ es[s[:,1,gx]] for gx in 1:2 ]... )
    scatter!(ax, p, ess, color=:black, markerstrokewidth=0 )
    
    
    â,r,pv = getcorrelation(repeat(p,2), ess) # this does not work yet in MathUtils
    plot!(ax, [0,1.], â[1].*[0,1.].+â[2], color=:red, lw=1.5, alpha=0.8)
    r = round(r,digits=2)
    # if round(pv,digits=8) == 0 pvs = "≤1${}$" else pvs = "=$(round(pv,digits=8))" end
    pve = ceil(log10(pv)) # this equals 1e-12, but typing below manually is easier
    annotate!(ax, 0.1, 3, "r=$(r) p<10⁻¹¹", font(pointsize=8, color=:red, halign=:left))
    xlims!(ax,0,1)
    xlabel!(ax,"performance")
    ylabel!(ax,"activity difference\nrelevant - irrelevant")


    @panellabels axs [5]
    
    plot!(axs, tick_direction=:out, xgrid=false, ygrid=false)#, ytickfonthalign=:left, xtickfontvalign=:bottom, xguidevalign=:top)
    
    display(axs)
    if config[:publishfigures]
        savefig(joinpath(config[:publicationfigurespath],"Figure4-$(figurepostfix).png"))
        savefig(joinpath(config[:publicationfigurespath],"Figure4-$(figurepostfix).pdf"))
    end

end














function figure5()
    figurepostfix = "contextgatedmutualfeedback"
    @info "making figure 5" figurepostfix



    axs = plot(layout=(2,3),size=(3*350, 2*300), legend=false,
               left_margin=15*Plots.px, bottom_margin=15*Plots.px, top_margin=-15*Plots.px, right_margin=15*Plots.px)





    # # subspace suppression schematics

    # ax = axs[1,1]
    # plot!(ax, aspect_ratio=:equal, left_margin=30*Plots.px, top_margin=30*Plots.px)
    # plot!(ax,[(2,7),(10,7)], lw=2, alpha=0.5, color=:navy, label="visual subspace")
    # plot!(ax,[(2,7),(2,15)], lw=2, alpha=0.5, color=:darkgreen, label="audio subspace")
    # plot!(ax,[(2,7),(8,7)],arrow=(5,5,:closed), lw=5, color=:navy, label="visual activity")
    # plot!(ax,[(2,7),(2,13)],arrow=(5,5,:closed), lw=5, color=:darkgreen, label="audio activity")
    # plot!(ax,legend=:topleft, foreground_color_legend=nothing, background_color_legend=nothing)
    # xlims!(ax,-15,10)
    # ylims!(ax,0,20)
    # title!(ax,"total stimulus space")
    # plot!(ax, axis=false)

    # @panellabel ax "A" -0.2 1.1

    # ax = axs[1,2]
    # plot!(ax, aspect_ratio=:equal)
    # plot!(ax,[(2,2),(10,2)], lw=2, alpha=0.5, color=:navy)
    # plot!(ax,[(2,2),(2,10)], lw=2, alpha=0.5, color=:darkgreen)
    # plot!(ax,[(2,2),(8,2)],arrow=(5,5,:closed), lw=5, color=:navy)
    # plot!(ax,[(2,2),(2,4.5)],arrow=(5,5,:closed), lw=5, color=:darkgreen)
    # plot!(ax,[(2,12),(10,12)], lw=2, alpha=0.5, color=:navy)
    # plot!(ax,[(2,12),(2,20)], lw=2, alpha=0.5, color=:darkgreen)
    # plot!(ax,[(2,12),(4.5,12)],arrow=(5,5,:closed), lw=5, color=:navy)
    # plot!(ax,[(2,12),(2,18)],arrow=(5,5,:closed), lw=5, color=:darkgreen)
    # xlims!(ax,-10,10)
    # ylims!(ax,0,20)
    # annotate!(ax,0,15,text("audio context",8,:right,:darkgreen))
    # annotate!(ax,0,5,text("visual context",8,:right,:navy))
    # title!(ax,"supressed irrelevant activity")
    # plot!(ax, axis=false)

    # ax = axs[1,3]
    # plot!(ax, aspect_ratio=:equal)
    # plot!(ax,[(2,7),(10,15)], lw=2, alpha=0.5, color=:darkorange, label="decision subspace")
    # plot!(ax,[(2,7),(8,13)],arrow=(5,5,:closed), lw=5, color=:darkorange,label="decision activity")
    # plot!(ax,legend=:topleft, foreground_color_legend=nothing, background_color_legend=nothing)
    # xlims!(ax,0,20)
    # ylims!(ax,0,20)
    # title!(ax,"abstract decision space")
    # plot!(ax, axis=false)




    # connection and activity structure of
    # context gated mutual feedback

    # mutual  inhibition feedback
    ax = axs[1,1]
    mutualfeedbackimage = load(joinpath(config[:publicationfigurespath],"parts/","mutualfeedback-schematics.png"))
    plot!(ax, mutualfeedbackimage, top_margin=50*Plots.px, left_margin=30*Plots.px)
    # xlims!(ax, 40,1380)
    # ylims!(ax, 110,600)
    plot!(ax, xticks=false, yticks=false, axis=false)
    @panellabel ax "B" -0.2 -0.6



    # load model data
    @load( joinpath(config[:modelanalysispath],  "suppression", "subspaces-rnn-reduced.bson"), @__MODULE__, allFmg, allFmn)

    
    # recurrent connections
    function fillmatrixtimeline(M,gap=size(1)÷10,border=1,maxpow=6,powers=1:1)
        s = size(M,1)
        Mborder = zeros(s+2*border,s+2*border)
        Mtimeline = zeros(s+(gap+2*border)*maxpow,s+(gap+2*border)*maxpow)
        for n in powers
            Mp = M^powers[n]
            Mp ./= absmax(Mp)
            gb = (gap+2*border)*(n-1)
            Mtimeline[gb+1:gb+s+2*border,gb+1:gb+s+2*border] = Mborder
            Mtimeline[gb+1+border:gb+s+border,gb+1+border:gb+s+border] = Mp
        end
        return Mtimeline
    end

        # heatmap!(inset=(6, bbox(0.1*n, 1-0.1*n, 1, 1,:bottom, :left)), subplot=11+n, Mp,
        #       color=colorspace, clim=limsabsmaxnegpos(Mp), aspect_ratio=:equal, yflip=true, axis=false)
    # end


    ax = axs[1,2]
    # colorspace = :RdBu   # :diverging_gkr_60_10_c40_n256
    colorspace = cm
    M = (allFmg+allFmn)/2      # average over go and nogo representations
    nhidden = size(M,1)
    # Mr = fillmatrixtimeline(M,5,1,6,1:1)
    heatmap!(ax, M, color=colorspace, clim=limsabsmaxnegpos(M), aspect_ratio=:equal, yflip=true, axis=false)
    # narrow colorbar:
    plot!(inset_subplots=(2, bbox(1.02, 0, 0.05, 1, :bottom, :left)))
    plot!(axs[7], axis=false)
    heatmap!(twinx(axs[(*)(length(axs))]), repeat(-1:1/200:1,1,2), color=colorspace, yticks=((0,200,400),("-1 AU","0 AU","1 AU")), colorbar=false, xticks=false)
    @panellabel ax "C" -0.2 -0.2
 
    ax = axs[1,3]
    plot!(ax,  axis=false, aspect_ratio=:equal, colormap=false)
    
    # M = M^6       # nonlinearmatrixpower(tanh,M,6,1)
    Mr = fillmatrixtimeline(M,5,1,6,1:6)
    heatmap!(ax, Mr, color=colorspace, clim=limsabsmaxnegpos(Mr), aspect_ratio=:equal, yflip=true, axis=false)
    # narrow colorbar:
    # plot!(ax, inset_subplots=(6, bbox(1.0, 0, 0.05, 1, :bottom, :left)), axis=false)
    # heatmap!(twinx(axs[(*)(length(axs))]), repeat(-1:1/200:1,1,2), color=colorspace, yticks=((0,200,400),("-1 AU","0 AU","1 AU")), colorbar=false, xticks=false)
    @panellabel ax "D" -0.2 -0.2







    # activity structure with modality index context index
    # load for V1, ACC, and models
    panels = ["F","G","E"]
    labelagent = ["V1","ACC","RNN"]

    # load mouse data
    mouseidsV1 = collectsubjectids("V1")
    modalityindexV1 = []
    contextindexV1 = []
    nmicemax = [0,0]
    for mouseid in mouseidsV1
        @load(joinpath(config[:cachepath],"subspace/","mutualswitchsubspaces-$(string(mouseid)).bson"),  @__MODULE__,
            modalityunitssingle, modalityindexsingle, contextindex)
        modalityindex = dropdims(mean([modalityindexsingle[modalityunitssingle[:,1],1] modalityindexsingle[modalityunitssingle[:,2],2] ],  dims=2), dims=2)
        contextindex = dropdims(mean([contextindex[modalityunitssingle[:,1],1] contextindex[modalityunitssingle[:,2],2] ],  dims=2), dims=2)
        push!(modalityindexV1,modalityindex)
        push!(contextindexV1,contextindex)
        nmicemax[1] = max(nmicemax[1],length(contextindex))
    end
    mouseidsACC = collectsubjectids("ACC")
    modalityindexACC = []
    contextindexACC = []
    for mouseid in mouseidsACC
        @load(joinpath(config[:cachepath],"subspace/","mutualswitchsubspaces-$(string(mouseid)).bson"),  @__MODULE__,
             modalityunitssingle, modalityindexsingle, contextindex)
        modalityindex = dropdims(mean([modalityindexsingle[modalityunitssingle[:,1],1] modalityindexsingle[modalityunitssingle[:,2],2] ],  dims=2), dims=2)
        contextindex = dropdims(mean([contextindex[modalityunitssingle[:,1],1] contextindex[modalityunitssingle[:,2],2] ],  dims=2), dims=2)
        push!(modalityindexACC,modalityindex)
        push!(contextindexACC,contextindex)
        nmicemax[2] = max(nmicemax[2],length(contextindex))
    end
    # load model data
    @load( joinpath(config[:modelanalysispath],  "suppression", "subspaces-rnn-reduced.bson"), @__MODULE__, allCim)
    @load( joinpath(config[:modelanalysispath],  "suppression", "subspaces-rnn.bson"), @__MODULE__,
                 Fmgs, Fmns,  Mis, Cis, Dis, Ris, Cims, Rims, Rigns, DWis, DWhs, DWos, validmodels) 


    # plot context indices with neuron ordering by modality index
    numoutliers = 6 # leave out this many neurons at the beginning and end of the neuron ordering
    agpanel = [2,3,1]
    for ag in [1,2,3]
        ax = axs[2,agpanel[ag]]
        if ag<3
            contextindex = ag==1 ? contextindexV1 : contextindexACC
            mouseids = ag==1 ? mouseidsV1 : mouseidsACC
            miceneuronpositions = 1:nmicemax[ag]
            contextindices = NaN .* ones(nmicemax[ag],length(mouseids))
            neuronpositionsconcat = Int64[]
            contextindicesconcat = Float64[]
            for mid in eachindex(mouseids)
                nneurons = size(contextindex[mid],1)
                shift = (nmicemax[ag] - nneurons)÷2
                neuronpositions = miceneuronpositions[(1:nneurons) .+ shift]
                # for mean display
                contextindices[neuronpositions,mid] = contextindex[mid]
                # concatenate for correlation
                neuronpositionsconcat = [neuronpositionsconcat; neuronpositions]
                contextindicesconcat = [contextindicesconcat; contextindex[mid]]
            end

            # individual mice
            plot!(ax, contextindices, lw=1, color=:grey, alpha=0.5)

            # mean of mice
            miceneuronpositionsnooutlier = miceneuronpositions[numoutliers:end-numoutliers]       # take into account only multiple mice data
            ci = [ mean(contextindices[n, .! isnan.(contextindices[n,:])]) for n in miceneuronpositionsnooutlier]
            plot!(ax,miceneuronpositionsnooutlier,ci,lw=2,color=:lightseagreen)

            

            # correlation statistics
            â,r,p = getcorrelation(neuronpositionsconcat, contextindicesconcat)
            plot!(ax, miceneuronpositions[[1,end]], â[1].*miceneuronpositions[[1,end]].+â[2], color=:red, ls=[:dash :solid][ag], lw=1.5, alpha=0.8)
            @info "corr" brainarea=["V1","ACC"][ag] â[2] r p n=length(neuronpositionsconcat)
            r = round(r,digits=2)
            p = round(p,digits=[2,4][ag])
            annotate!(ax, miceneuronpositions[end], 9, "r=$(r) p=$(p)", font(pointsize=8, color=:red, halign=:right))

            hline!(ax,[0], ls=:dash, color=:grey, alpha=0.5)
            ylims!(ax, -10, 10)

            xticks!(ax,[1,nmicemax[ag]÷2,nmicemax[ag]])

        elseif ag==3
            Cims = Cims[validmodels,:,:]
            nneurons = size(Cims,2)

            m = dropdims(mean(Cims,dims=(1,3)),dims=(1,3)) # mean over models and go-nogo
            e = dropdims(std(Cims,dims=(1,3)),dims=(1,3))./sqrt(2*size(Cims,1)) 
            plot!(ax, 1:nneurons, m, ribbon=e, lw=2, color=:lightseagreen, facealpha=0.3)

            # correlation statistics
            â,r,p = getcorrelation(1:nneurons, m)
            plot!(ax, [1,nneurons], â[1].*[1,nneurons].+â[2], color=:red, lw=1.5, alpha=0.8)
            @info "corr context modality" "models" sum(validmodels)  â[2] r p
            r = round(r,digits=2)
            p = round(p,digits=3)
            annotate!(ax, nneurons, 0.045, "r=$(r) p=$(p)", font(pointsize=8, color=:red, halign=:right))

            hline!(ax,[0], ls=:dash, color=:grey, alpha=0.5)
            ylims!(ax, -0.055, 0.055)
            yticks!(ax, -0.04:0.02:0.04)
            xticks!(ax,[1,size(allCim,1)÷2,size(allCim,1)])
        end



        title!(ax,labelagent[ag])

        xlabel!(ax, "neurons")
        ylabel!(ax, "context index")
    
        @panellabel ax panels[ag] -0.2 1.3

    end


    # statistics for modality index values vs context index (not the sorting)
    for ag in 1:2
        modalityindex = ag==1 ? modalityindexV1 : modalityindexACC
        contextindex = ag==1 ? contextindexV1 : contextindexACC
        mouseids = ag==1 ? mouseidsV1 : mouseidsACC
        # ax = axs[4,ag]
        modalityindicesconcat = Float64[]
        contextindicesconcat = Float64[]
        for mid in eachindex(mouseids)
            # concatenate for correlation
            modalityindicesconcat = [modalityindicesconcat; modalityindex[mid]]
            contextindicesconcat = [contextindicesconcat; contextindex[mid]]
        end

        # individual mice
        # scatter!(ax, modalityindicesconcat, contextindicesconcat, lw=1, color=:grey, alpha=0.5)

       

        # correlation statistics
        â,r,p = getcorrelation(modalityindicesconcat, contextindicesconcat) # this does not work yet in MathUtils
        # plot!(ax, modalityindicesconcat[[1,end]], â[1].*modalityindicesconcat[[1,end]].+â[2], color=:red, lw=1.5, alpha=0.8)
        @info "corr modalityindex" brainarea=["V1","ACC"][ag] â[2] r p n=length(modalityindicesconcat)
        r = round(r,digits=2)
        # annotate!(ax, modalityindicesconcat[end], 9, "r=$(r) p="*["4.9⋅10⁻⁵","5.0⋅10⁻⁵"][ag], font(pointsize=8, color=:red, halign=:right))

        # hline!(ax,[0], ls=:dash, color=:grey, alpha=0.5)
        # ylims!(ax, -11, 11)


    end

    








    plot!(axs, tick_direction=:out, xgrid=false, ygrid=false)#, ytickfonthalign=:left, xtickfontvalign=:bottom, xguidevalign=:top)
    
    display(axs)
    if config[:publishfigures]
        savefig(joinpath(config[:publicationfigurespath],"Figure5-$(figurepostfix).png"))
        savefig(joinpath(config[:publicationfigurespath],"Figure5-$(figurepostfix).pdf"))
    end



end