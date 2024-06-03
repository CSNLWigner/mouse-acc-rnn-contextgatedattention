# make figures
const lowprob = config[:behaviouralmodellowprob]
const cm = cgrad([:red, :white, :green],[0,0.4,0.6,1])
const dpi = config[:figuredpi]

function figure1()
    figurepostfix = "paradigms,behavior"
    @info "making figure 1" figurepostfix

    mouseids = collectsubjectids("ACC")
    nmice = length(mouseids)
    
    axs = plot(layout=@layout( [ a b{0.3w}; c{0.66h} d{0.3w} ] ), size=(3*300, 5*300÷2), dpi=dpi, legend=false)









    ax = axs[1]
    paradigmimage = load(joinpath(config[:publicationfigurespath],"parts/","experimental,paradigm,acc.png"))
    plot!(ax, paradigmimage, top_margin=40*Plots.px, left_margin=40*Plots.px, bottom_margin=70*Plots.px)
    xlims!(ax, 0,1320)          # 1320  x 495
    ylims!(ax, 0,495)
    plot!(ax, xticks=false, yticks=false, axis=false)
    @panellabel ax "a" -0.12 -0.18

    




    ax = axs[2]

    t = range(-1.5, stop=4.5, length=1000)
    yscale = 0.8
    visual = (0 .<= t .<= 3) * yscale
    audio = (0 .<= t .<= 3) * yscale
    reward = (2 .<= t .<= 3) * yscale
    ys = [3, 2, 1]
    xs = [0.2,0.2,0.2]
    for (v,x,y,c,l) in zip([visual, audio, reward], xs, ys, [:navy, :darkgreen, :red],["visual", "auditory", "reward" ] )
        plot!(ax, t, y .+ v, color=c, lw=2)
        annotate!(ax, x, y+0.5, text("$(l)",c,9,:left))
    end
    xlims!(ax, -1.5, 4.5)
    ylims!(ax, 1-0.05, 1+3.05-0.2)
    plot!(ax, yticks=nothing, xticks=[0,2,3])
    xlabel!(ax, "time from stimulus onset [s]")
    @decoderbackground ax 0 3 2 :darkgrey
    @panellabel ax "b" -0.1 1.15





    ax = axs[3]
    plot!(ax, axis=false, left_margin=30*Plots.px)
    
    mouseid = :AC006
    labels = ["─── go congruent" "- - go incongruent"; "── nogo congruent" "- - nogo incongruent"]
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
        plot!(axs[3], inset_subplots=(3, bbox(0, 1-ix*0.22, 1, 0.18, :bottom, :left)))
        ax = axs[4+ix]
        for cx in 1:2         # contexts
            if sx+congi==2 annotate!(ax, sum(boundaries[cx,1]),1.1, text(["visual","auditory"][cx]*"\ncontext",[:navy,:darkgreen][cx],:left, :bottom, 8)) end
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
        ylabel!(ax, "performance", ylabelfontsize=6)
    end
    
    plot!(axs[3], inset_subplots=(3, bbox(0, 0, 1, 0.06, :bottom, :left)))
    ax = axs[9]
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
    
    @panellabel axs[3] "c" -0.12 1.02


    


    # ax = axs[4] # probabilities of sequence lengths
    # # axsp = plot(layout=(2,1))


    # # load a dict of repeated probability simulations by mouseid containing a dict for each context
    # # load mouse consecutive consistent blocks numbers
    # mouseids = collectsubjectids("ACC")
    # # mouseid = :AC006
    # @load(joinpath(config[:cachepath], "behaviour", "probabilityconsistentperiods.bson"), @__MODULE__, probabilityconsistentperiods)
    # @load(joinpath(config[:cachepath], "behaviour", "movingaveragestatistics,mice.bson"), @__MODULE__, consecutivesegmentsmice)

    # markershapes = [:circle,:square,:diamond,:utriangle]

    # for (n,mouseid) in enumerate(mouseids)

    #     numconsistent = sum.(consecutivesegmentsmice[mouseid])
    #     conseqconsistent = maximum.(consecutivesegmentsmice[mouseid])

    #     # find the less likely context, and restrict display to that context
    #     lesslikelycontextid = argmin(numconsistent)
    #     context = ["visual","audio"][lesslikelycontextid]
    #     @info "$mouseid" numconsistent lesslikelycontextid context p=probabilityconsistentperiods[mouseid][context][:pchoicebias]
    #     ntrials = probabilityconsistentperiods[mouseid][context][:ntrials]
    #     prob_ntrials_successes = probabilityconsistentperiods[mouseid][context][:prob_ntrials_successes] 
    #     prob_atleastone_consecutive_length = probabilityconsistentperiods[mouseid][context][:prob_atleastone_consecutive_length]


    #     # sizes: prob_ntrials_successes: (ntrials), prob_atleastone_consecutive_length: (ntrials, conseucive length)
    #     prob_ntrials_successes[prob_ntrials_successes.==0] .= 0.5/config[:generativeconsecutivesamplingrepeats]
    #     prob_atleastone_consecutive_length[prob_atleastone_consecutive_length.==0] .= 0.5/config[:generativeconsecutivesamplingrepeats]

    #     plot!(ax, prob_ntrials_successes, color=:dodgerblue, yscale=:log10, alpha=0.5)
    #     plot!(ax, prob_atleastone_consecutive_length, color=:purple, alpha=0.5)
    #     xlims!(ax, 0, 28)
    #     ylims!(ax, 1e-6, 5e-2)


    #     scatter!(ax, [numconsistent[lesslikelycontextid]], [prob_ntrials_successes[numconsistent[lesslikelycontextid]]],
    #                 markercolor=:dodgerblue, markerstrokewidth=0, markershape=markershapes[n], alpha=0.8)
    #     scatter!(ax, [conseqconsistent[lesslikelycontextid]], [prob_atleastone_consecutive_length[conseqconsistent[lesslikelycontextid]]],
    #                 markercolor=:purple, markerstrokewidth=0, markershape=markershapes[n], alpha=0.8)

    # end

    # @panellabel ax "d" -0.20 3

    


    ax = axs[4]
    plot!(ax, axis=false, left_margin=40*Plots.px, right_margin=105*Plots.px, top_margin=40*Plots.px)
    axsslot = 4
    axendstack = 9
    colors = [:navy :darkgreen]

    @load(joinpath(config[:cachepath], "behaviour","behaviouralstatistics.bson"), @__MODULE__, ntrials, nhighperfs, fractioncorrects, lls)

    @info "% hp tr" perchp = nhighperfs ./ ntrials

    ix = 1 # number of high performance trials
    plot!(axs[axsslot], inset_subplots=(axsslot, bbox(0.1, 1-ix*0.20, 1, 0.18, :bottom, :left)))
    ax = axs[axendstack+ix]
    bar!(ax, (1:nmice).+[-0.165 +0.165 ], nhighperfs, bar_width=0.33, color=colors, linecolor=colors, label=nothing)
    xlims!(ax, 0, nmice+1)
    ylims!(ax, 0, 35)
    plot!(ax, xticks=false, yticks=[0,10,20,30], ytickfontsize=6, ylabel="number\nof trials", ylabelfontsize=7, yguidehalign=:left)
    annotate!(ax, 1.5, 42, text("visual context", :navy, :left, 6))
    annotate!(ax, 1.5, 36, text("auditory context", :darkgreen, :left, 6))
    @panellabel ax "d" -0.40 1.15




    ix = 2    # probability of consistent trials by chance
    @load(joinpath(config[:cachepath], "behaviour", "probabilityconsistentperiods,incongruentbias.bson"), @__MODULE__, probabilityconsistentperiods)
    @load(joinpath(config[:cachepath], "behaviour", "movingaveragestatistics,mice.bson"), @__MODULE__, consecutivesegmentsmice)

    plot!(axs[axsslot], inset_subplots=(axsslot, bbox(0.1, 1-ix*0.20, 1, 0.18, :bottom, :left)))
    ax = axs[axendstack+ix]

    prob_numconsistents = zeros(nmice,2)
    prob_conseqconsistents = zeros(nmice,2)

    for (n,mouseid) in enumerate(mouseids)

        numconsistent = sum.(consecutivesegmentsmice[mouseid])
        conseqconsistent = maximum.(consecutivesegmentsmice[mouseid])

        
        # find the less likely context, and restrict display to that context
        for (cx,context) in enumerate(["visual","audio"])

            # @info "$mouseid $context" numconsistent cx context p=probabilityconsistentperiods[mouseid][context][:pchoicebias]
            ntrials = probabilityconsistentperiods[mouseid][context][:ntrials]
            prob_ntrials_successes = probabilityconsistentperiods[mouseid][context][:prob_ntrials_successes] 
            prob_atleastone_consecutive_length = probabilityconsistentperiods[mouseid][context][:prob_atleastone_consecutive_length]

            prob_ntrials_successes[prob_ntrials_successes.==0] .= 0.5/config[:generativeconsecutivesamplingrepeats]
            prob_atleastone_consecutive_length[prob_atleastone_consecutive_length.==0] .= 0.5/config[:generativeconsecutivesamplingrepeats]

            # ["num. of consistent", "at least conseq. consist."]
            

            prob_numconsistents[n,cx] = prob_ntrials_successes[numconsistent[cx]]
            prob_conseqconsistents[n,cx] = prob_atleastone_consecutive_length[conseqconsistent[cx]]
        end
    end
    @info "" prob_numconsistents log10.(prob_numconsistents)
    bar!(ax, (1:nmice).+[-0.165 +0.165 ], log10.(prob_numconsistents), bar_width=0.33, color=colors, linecolor=colors, label=nothing)
    hline!(ax, [log10(0.05)], lw=1, color=:grey, ls=:dash, label=nothing)
    xlims!(ax, 0, nmice+1)
    # ylims!(ax, 0, 35)
    plot!(ax, xticks=(1:nmice,[]), xlabel="mice", ytickfontsize=6, ylabel="log probab. of\nconsist. by chance", ylabelfontsize=6, yguidehalign=:left)
    @panellabel ax "e" -0.40 0.95




    ix = 3 # fraction correct trials
    plot!(axs[axsslot],inset_subplots=(axsslot, bbox(0.1, 1-ix*0.20, 1, 0.18, :bottom, :left)))
    ax = axs[axendstack+ix]
    bar!(ax, (1:nmice).+[-0.165 +0.165 ], fractioncorrects, bar_width=0.33,  color=colors, linecolor=colors, label=nothing)
    xlims!(ax, 0, nmice+1)
    ylims!(ax, 0, 1.05)
    plot!(ax, xticks=false, yticks=[0,1], ytickfontsize=6, ylabel="fraction\ncorrect", ylabelfontsize=7, yguidehalign=:left)
    @panellabel ax "f" -0.40 0.95


    sourcedata = DataFrame( mouseid=reshape(repeat(reshape(mouseids,1,:),2,1),:,1)[:,1],
                            context=reshape(repeat(["visual", "audio"],1,4),1,:)[1,:],
                            nhighperfs=reshape(nhighperfs,:,1)[:,1],
                            proportionhighperfs=reshape(nhighperfs ./ ntrials,:,1)[:,1], 
                            probnumconsistents=reshape(prob_numconsistents,:,1)[:,1],
                            fractioncorrects=reshape(fractioncorrects,:,1)[:,1] )
    CSV.write(joinpath(config[:publicationfigurespath],"sourcedata","fig1","fig1-def.csv"), sourcedata)



    ix = 4 # lls 1/2   # models 1
    plot!(axs[axsslot],inset_subplots=(axsslot, bbox(0.1, 1-ix*0.20, 1, 0.18, :bottom, :left)))
    ax = axs[axendstack+ix]
    bar!(ax, (1:nmice).+range(-0.3,0.3,length=4)', reshape(permutedims(lls[:,:,1:2],(1,3,2)),:,4), bar_width=0.15, 
           color=[:white colors[1] :white colors[2]], linecolor=[colors[1] colors[1] colors[2] colors[2]], label=nothing)
    xlims!(ax, 0, nmice+1)
    # ylims!(ax, -5.05, 0.05)
    plot!(ax, xticks=(1:nmice,[]), xlabel="mice", ytickfontsize=6, ylabel="model log\nlikelihood", ylabelfontsize=6, yguidehalign=:left)
    # legend
    wh = 0.23; hh = 0.15/1.2*8.5
    for (mx,m) in enumerate([1,2]), cx in 1:2
        plot!(ax, rectangle(nmice+0.5+wh*(cx-1),-4.50-hh*(mx-1),wh,hh), linecolor=[:navy,:darkgreen][cx],
                 fillcolor=[:white,[:navy,:darkgreen][cx]][mx], alpha=[1,1][mx], label=nothing)
        if cx==1
            annotate!(ax,nmice+0.5+2*wh,-4.50-hh*(mx-1)+hh/2,text([" context opposite"," context aware"][mx],:vcenter,:left,6))
        end
    end
    @panellabel ax "g" -0.40 0.95


    ix = 5 # lls 2/2       # models 2
    plot!(axs[axsslot], inset_subplots=(axsslot, bbox(0.1, 1-ix*0.20, 1, 0.18, :bottom, :left)))
    ax = axs[axendstack+ix]
    colorlist = [colors[1] colors[1] colors[1] colors[2] colors[2] colors[2]]
    bar!(ax, (1:nmice).+range(-0.30,0.30,length=6)', reshape(permutedims(lls[:,:,3:5],(1,3,2)),:,6), bar_width=0.10, 
           color=colorlist, linecolor=colorlist, alpha=[0.3 0.6 1 0.3 0.6 1], label=nothing)
    xlims!(ax, 0, nmice+1)
    ylims!(ax, -0.85, 0.05)
    plot!(ax, xticks=false, ytickfontsize=6, ylabel="model log\nlikelihood", ylabelfontsize=6, yguidehalign=:left)
    # legend
    wh = 0.2; hh = 0.19/1.2*8.5/10
    for (mx,m) in enumerate([3,4,5]), cx in 1:2
        plot!(ax, rectangle(nmice+0.5+wh*(cx-1),-0.50-hh*(mx-1),wh,hh), color=[:navy,:darkgreen][cx],alpha=[0.2,0.6,1.0][mx], label=nothing)
        if cx==1
            annotate!(ax,nmice+0.5+2*wh,-0.50-hh*(mx-1)+hh/2,text([" context unaware"," context aware bias"," context aware lapse"][mx],
                                                          :vcenter,:left,6))
        end
    end
    @panellabel ax "h" -0.40 0.95

    sourcedata = DataFrame(mouseid=reshape(repeat(reshape(mouseids,1,:),2,1),:,1)[:,1],
                           context=reshape(repeat(["visual", "audio"],1,4),1,:)[1,:])
    for (k,name) in enumerate(["contextopposite","contextaware"," contextunaware"," contextawarebias","contextawarelapse"])
        sourcedata[!,Symbol(name)] = reshape(lls[:,:,k],nmice*2)
    end

    CSV.write(joinpath(config[:publicationfigurespath],"sourcedata","fig1","fig1-gh.csv"), sourcedata)






    



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


    l = @layout(  [  [ x    y;
                      u  v  w  z];
                     [a b c d e{0.25w};
                      f g h i j]   ]      )
    
    axs = plot(layout=l, size=(5*350, 4*350), legend=false, dpi=dpi,
               left_margin=20*Plots.px, bottom_margin=15*Plots.px, top_margin=20*Plots.px, right_margin=20*Plots.px)





    # spikes and firing rate blocks



    mouseids = collectsubjectids("ACC")
    nmice = length(mouseids)
    mouseid = mouseids[4]

    @info "mouseid" mouseid
    nwbfile,_ = loadnwbdata(mouseid)
    triallist = nwbdf(nwbfile.trials)
    filter!(:difficulty=>u->u=="complex",triallist)
    addcongruencycolumn!(triallist)




    neuronsspiketimeslist = gettrialrelativespiketimes(triallist, nwbfile.units)         # get spikes for trials
    ntrials = length(neuronsspiketimeslist)
    nneurons = length(neuronsspiketimeslist[1])
    trialwidth = config[:stimuluslength] + abs(config[:preeventpadding]) + abs(config[:posteventpadding])


    # selecting the neurons to display
    rastertriallist = [30:35,82:87]      # visual and auditory example trials

    @load(joinpath(config[:cachepath],"subspace/","mutualswitchsubspaces-$(string(mouseid)).bson"),  @__MODULE__,
            modalityunitssingle, modalityindexsingle, contextindex)

            
    # modalityindex = modalityindexsingle[:,1] - modalityindexsingle[:,2]
    modalityorder = [sortperm(modalityindexsingle[:,g]) for g in 1:2 ]
    ncellneighbours = 10
    bestvisualcells = [modalityorder[1][1:1+ncellneighbours-1], modalityorder[2][1:1+ncellneighbours-1]]         # go and nogo best
    bestaudiocells = [modalityorder[1][end-ncellneighbours+1:end], modalityorder[2][end-ncellneighbours+1:end]]
    selectedcells = [bestvisualcells[2][1] bestvisualcells[1][2]; bestaudiocells[1][6] bestaudiocells[2][3]]          # stim × gonogo

    


    
    # raster
    for (cx,context) in enumerate(("visual","audio"))
        ax = axs[cx]
        for (tr,trial) in enumerate(rastertriallist[cx])
            for n in 1:nneurons
                # order neurons by goorder
                scatter!(ax, neuronsspiketimeslist[trial][n] .+ (tr-1)*trialwidth,
                            repeat([n], length(neuronsspiketimeslist[trial][n])), 
                            marker = (:vline, 3, :red), markersize = 0.5, markerstrokewidth=0.1)
            end
            @nolinebackground(ax, config[:stimulusstart]+(tr-1)*trialwidth, config[:stimulusend]+(tr-1)*trialwidth,
                                   config[:waterstart]+(tr-1)*trialwidth, bg=:darkgrey)
            visstim = (Int(triallist[trial,:degree]))
            audstim = (Int(triallist[trial,:freq]/1000))
            annotate!(ax,(tr-1)*trialwidth+trialwidth/2,nneurons+2,text("$(visstim)° $(audstim)kHz ",9,:black,:hcenter,:bottom))
        end
        xlabel!(ax, "time [s]")
        ylabel!(ax,"cell id")
        annotate!(ax,-2.5,nneurons+2,text("$context context",10,[:blue,:green][cx],:hcenter,:bottom))
        # annotate specific cells in next subplots
        yticks!(ax,(vcat([10,20],selectedcells[:]), vcat(string.([10,20]),["c","d","e","f"])))
        plot!(ax,top_margin=60*Plots.px, left_margin=40*Plots.px)
        @panellabel ax ["a","b"][cx] -0.10 1.18
    end


    colorcondition = [:dodgerblue :navy; :lime :darkgreen]
    alphacondition = [1.0 1.0]   # go nogo
    lscondition = [:solid :dash]
    sm = 31
    labelrelevance = ["relevant","irrelevant"]
    labelstimulus = ["visual","auditory"]
    labelgonogo = [" 45°" "135°"; " 5kHz" "10kHz"]
    for ix in 1:2     # cells within sensitivity groups
        for clx in 1:2      # vis sensitive cell, audio sensitive cell
            cell = selectedcells[clx,ix]
            gx = ix   # match cell sensitivity to stimulus class

            ax = axs[2+(clx-1)*2+ix]

            @nolinebackground(ax, config[:stimulusstart], config[:stimulusend], config[:waterstart], bg=:darkgrey)

            sourcedata = DataFrame()
            for rx in 1:2
                cx = 2 - (clx==rx) # context index
                context = ("visual","audio")[cx]

                mask = (triallist[!,:context].==context) .& (triallist[!,[:degree,:freq][clx]].==[[45,135],[5000,10000]][clx][gx])

                ts, ifr = smoothinstantenousfiringrate(neuronsspiketimeslist[mask], config[:stimuluslength], binsize=config[:dt])
                
                m = @movingaverage(mean(ifr[:,:,cell],dims=1)[1,:], sm)
                e = @movingaverage(std(ifr[:,:,cell],dims=1)[1,:]./sqrt(sum(mask)), sm)

                plot!(ax, ts, m, ribbon=e, color=colorcondition[clx,rx], lw=3, ls=lscondition[gx], alpha=alphacondition[gx], fillalpha=0.2*alphacondition[gx],
                    label="$(labelstimulus[clx]) stim. $(labelgonogo[clx,gx]) $(labelrelevance[rx]) ($context context)")
                
                sourcedata[!,Symbol("time"*labelrelevance[rx])] = ts
                sourcedata[!,Symbol("firingrate"*labelrelevance[rx]*"mean")] = m
                sourcedata[!,Symbol("firingrate"*labelrelevance[rx]*"sem")] = e
            end
            CSV.write(joinpath(config[:publicationfigurespath],"sourcedata","fig2","fig2-$(["c","d","e","f"][(clx-1)*2+ix]).csv"), sourcedata)

            ylims!(ax,([[0,49],[0,69],[0,79],[0,13]][(clx-1)*2+ix])...)
            # yticks!(ax,0:20:max(ylim()))
            if ix+clx==2 ylabel!(ax, "firing rate [Hz]") end
            xlabel!(ax, "time from stimulus onset [s]")
            @panellabel ax ["c","d","e","f"][(clx-1)*2+ix] (-0.16,-0.21)[1+(ix+clx==2)] 1.06
            plot!(ax, legend=:topright, foreground_color_legend=nothing, background_color_legend=nothing, legendfontsize=8)
            plot!(ax,top_margin=40*Plots.px, bottom_margin=40*Plots.px)
        end
    end

    frblax = 6


    












    # decoding blocks



    # panels first 2×2 decoding blocks
    # mice suppression

    
    mouseidsv1 = collectsubjectids("V1")
    nmicev1 = length(mouseidsv1)
    mousev1ind = 1:nmicev1

    mouseidsacc = collectsubjectids("ACC")
    nmiceacc = length(mouseidsacc)
    mouseaccind = nmicev1+1:nmicev1+nmiceacc


    labelsstimulus = ["visual","auditory"]
    labelscontexts = ["visual","auditory"]
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
    sourcedata = DataFrame()
    sourcedata[!,Symbol("timestamps")] = timestamps
    for (bx,mouseind) in enumerate((mousev1ind,mouseaccind))
        for (stimulusindex,stimulus) in enumerate(labelsstimulus)
            ax = axs[frblax+(stimulusindex-1)*5+bx] # axs[stimulusindex,bx]
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
                sourcedata[!,Symbol("accuracy"*["V1","ACC"][bx]*labelsrelevancies[contextindex,stimulusindex]*context*"mean")] = m
                sourcedata[!,Symbol("accuracy"*["V1","ACC"][bx]*labelsrelevancies[contextindex,stimulusindex]*context*"sem")] = e
            end
            xlims!(ax,-1.2,4.2)
            ylims!(ax,0.45,1.05)
            plot!(ax,legend=:topleft, foreground_color_legend=nothing, background_color_legend=nothing)
            title!(ax, [["V1\n","ACC\n"][bx],""][stimulusindex]*stimulus*" stimulus",  titlefont=font(12))
            if stimulusindex==1
                @panellabel ax ["g","h"][bx] -0.30 1.20
            end
        end
    end
    CSV.write(joinpath(config[:publicationfigurespath],"sourcedata","fig2","fig2-gh.csv"), sourcedata)














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
    sourcedata = DataFrame()
    sourcedata[!,Symbol("timestamps")] = timestamps
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
            ax = axs[frblax+(sx-1)*5+2+mx]
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
                    
                    sourcedata[!,Symbol("accuracy"*brainarea*relevancylabels[rx]*consistencylabels[bx]*"mean")] = m
                    sourcedata[!,Symbol("accuracy"*brainarea*relevancylabels[rx]*consistencylabels[bx]*"sem")] = e
                end
            end
            plot!(ax,legend=:topright, foreground_color_legend=nothing, background_color_legend=nothing)
            # xlims!(ax,-1.2,4.2)
            ylims!(ax,0.3,1.25)
            yticks!(ax,0.5:0.25:1)
            if mx==1 ylabel!(ax, "accuracy"); plot!(ax,left_margin=50*Plots.px) end
            title!(ax, ["$(brainarea)\n",""][sx]*labelsstimulus[sx]*" stimulus",  titlefont=font(12))
            if sx==2 xlabel!(ax, "time from stimulus onset [s]") end
            plot!(ax, bottom_margin=40*Plots.px)
            if sx==1 @panellabel ax ["i","j"][mx] -0.30 1.20 end
        end
    end
    CSV.write(joinpath(config[:publicationfigurespath],"sourcedata","fig2","fig2-ij.csv"), sourcedata)
    
    

    # block averaging the accuracies timeseries to remove the autocorrelation
    # display the s.t.d. of the accuracies with various time windows for ACC
    blocklengths = 2:300
    ss = [[],[],[],[]]
    ac = zeros(4,300,8)
    for bl in blocklengths
        for mx in 1:4
            # standard deviation
            s = []
            nblocks = 300 ÷ bl
            for b in 1:nblocks
                for bx in 1:2, sx in 1:2, rx in 1:2
                    blockrange = 150 .+ ((b-1)*bl+1:b*bl)
                    m = dropdims(mean(accuraciesallareas[2][mx:mx,bx,sx,rx,blockrange,1],dims=1),dims=1)
                    # @info "block length $bl block number $b" blockrange size(m) size(se)
                    push!(s, std(m))
                end
            end
            push!(ss[mx], s)
            
            # autocorrelation:
            for bx in 1:2, sx in 1:2, rx in 1:2
                a = autocor(accuraciesallareas[2][mx,bx,sx,rx,151:450,1],0:299)
                ac[mx,:,bx+(sx-1)*2+(rx-1)*4] = a
            end
        end
    end
    sourcedata = DataFrame()
    sourcedata[!,Symbol("lag")] = collect(0:299)./1000
    ax = axs[frblax+5]
    hline!(ax, [0], color=:grey, ls=:dash, label=nothing)
    for mx in 1:4
        # for (bx,bl) in enumerate(blocklengths)
            # @info "" bx bl ss[bx]
            # scatter!(ax, fill(bl,length(ss[mx][bx])), ss[mx][bx], color=:blue)
            # scatter!(ax[1], [bl], [mean(ss[mx][bx])], yerror=std(ss[mx][bx])/sqrt(length(ss[mx][bx])))
            # boxplot!(axautocorrelation, fill(bl,length(ss[bx])), ss[bx], color=:blue, alpha=0.5, outliers=false, whisker_range=1, notch=true)
        # end
        m = mean(ac[mx,:,:],dims=2)[:,1]
        e = std(ac[mx,:,:],dims=2)[:,1]/sqrt(8)
        plot!(ax, 0:299, m, ribbon=e, color=:grey, alpha=0.5, label=nothing)
        sourcedata[!,Symbol("autocorrelationmouse$(mx)mean")] = m
        sourcedata[!,Symbol("autocorrelationmouse$(mx)sem")] = e
    end
    plot!(ax, 0:299, mean(ac[:,:,:],dims=(1,3))[1,:,1], ribbon=std(ac[:,:,:],dims=(1,3))[1,:,1]/sqrt(8*4), lw=3, color=:black, label=nothing)
    xticks!(ax,0:50:300,string.(0:0.5:3))
    xlabel!(ax, "lag [s]")
    ylabel!(ax, "autocorrelation")
    @panellabel ax "k" -0.30 1.20
    CSV.write(joinpath(config[:publicationfigurespath],"sourcedata","fig2","fig2-k.csv"), sourcedata)




    # block averaging the accuracies timeseries to remove the autocorrelation
    # display the s.t.d. of the accuracies with various time windows for ACC
    blocklengths = 2:300
    ss = [[],[],[],[]]
    ac = zeros(4,300,8)
    for bl in blocklengths
        for mx in 1:4
            # standard deviation
            s = []
            nblocks = 300 ÷ bl
            for b in 1:nblocks
                for bx in 1:2, sx in 1:2, rx in 1:2
                    blockrange = 150 .+ ((b-1)*bl+1:b*bl)
                    m = dropdims(mean(accuraciesallareas[2][mx:mx,bx,sx,rx,blockrange,1],dims=1),dims=1)
                    # @info "block length $bl block number $b" blockrange size(m) size(se)
                    push!(s, std(m))
                end
            end
            push!(ss[mx], s)
            
            # autocorrelation:
            for bx in 1:2, sx in 1:2, rx in 1:2
                a = autocor(accuraciesallareas[2][mx,bx,sx,rx,151:450,1],0:299)
                ac[mx,:,bx+(sx-1)*2+(rx-1)*4] = a
            end
        end
    end
    ax = axs[frblax+5]
    hline!(ax, [0], color=:grey, ls=:dash, label=nothing)
    for mx in 1:4
        for (bx,bl) in enumerate(blocklengths)
            # @info "" bx bl ss[bx]
            # scatter!(ax, fill(bl,length(ss[mx][bx])), ss[mx][bx], color=:blue)
            # scatter!(ax[1], [bl], [mean(ss[mx][bx])], yerror=std(ss[mx][bx])/sqrt(length(ss[mx][bx])))
            # boxplot!(axautocorrelation, fill(bl,length(ss[bx])), ss[bx], color=:blue, alpha=0.5, outliers=false, whisker_range=1, notch=true)
        end
        plot!(ax, 0:299, mean(ac[mx,:,:],dims=2), ribbon=std(ac[mx,:,:],dims=2)/sqrt(8), color=:grey, alpha=0.5, label=nothing)
    end
    plot!(ax, 0:299, mean(ac[:,:,:],dims=(1,3))[1,:,1], ribbon=std(ac[:,:,:],dims=(1,3))[1,:,1]/sqrt(8*4), lw=3, color=:black, label=nothing)
    xticks!(ax,0:50:300,string.(0:0.5:3))
    xlabel!(ax, "lag [s]")
    ylabel!(ax, "autocorrelation")
    @panellabel ax "K" -0.30 1.20
    


    ax = axs[frblax+10]
    colors = [ :purple :darkorange ]     # brainarea × consistency
    # create the difference between relevant and irrelevant projections
    # accuraciesallareas is a (narea)(nmice,nstates,nmodalities,nrelevancies,ntimestamps,3)
    timerange = (timestamps .>= config[:stimulusstart]+0.60) .& (timestamps .< config[:stimulusend])
    @info "" sum(timerange)
    blocksize = 60
    nblocks = sum(timerange) ÷ blocksize        # number of blocks within the valid time range for suppression
    tickpoints = Float64[]
    ps = Float64[]
    ts = Float64[]
    sourcedata = DataFrame()
    for (mx,brainarea) in enumerate(["V1","ACC"])
        # individual mice
        d = accuraciesallareas[mx][:,:,:,1,timerange,1] .- accuraciesallareas[mx][:,:,:,2,timerange,1]
        d = reshape(d,size(d,1)*size(d,2),:)            # reshape to (mice*var, timestamps)
        mouseind = (mousev1ind,mouseaccind .- mouseaccind[begin] .+ mousev1ind[end]*2 .+ 3)[mx]
        barind = [mouseind; mouseind.-mouseind[begin].+mouseind[end].+1] + repeat([0.05,-0.05],length(mouseind))
        # all mice
        ma = similar(accuraciesallareas[mx][1,:,:,:,1:nblocks,1]) # create mean over autocorrelating blocks
        for bx in 1:2, sx in 1:2, rx in 1:2
            a = reshape(accuraciesallareas[mx][:,bx,sx,rx,timerange,1],size(accuraciesallareas[mx],1),blocksize,nblocks)
            a = mean(a, dims=2)[:,1,:]
            # ma[bx,sx,rx,:] = @movingaverage(dropdims(mean(a,dims=1),dims=1),5)
            ma[bx,sx,rx,:] = dropdims(mean(a,dims=1),dims=1)
        end
        
        for sx in 1:2
            d = ma[:,sx,1,:] .- ma[:,sx,2,:]
            # d = reshape(d,size(d,1),:)            # reshape to (variables, timestamps)
            mouseind = [[1],[4]][sx]
            barind = [mouseind; mouseind.-mouseind[begin].+mouseind[end].+1] + repeat([0.05,-0.05],length(mouseind))   .+ (mx-1)*8
            push!(tickpoints, mean(barind))
            boxplot!(   ax, barind', d', color=colors, alpha=mean(alphas),
                        outliers=false, whisker_range=5, notch=false,
                        label=[  nothing, [["consistent" "exploratory"] repeat([nothing nothing],1,length(mouseind)-1)]  ][Int(sx+mx==4)+1]   )
            ttest = OneSampleTTest(d[1,:],d[2,:])
            push!(ps, pvalue(ttest))
            push!(ts, ttest.t)

            sourcedata[!,Symbol(brainarea*"consistent"*labelsstimulus[sx])] = d[1,:]
            sourcedata[!,Symbol(brainarea*"exploratory"*labelsstimulus[sx])] = d[2,:]
        end
    end
    @info "p" ps
    @info "t" ts

    CSV.write(joinpath(config[:publicationfigurespath],"sourcedata","fig2","fig2-l.csv"), sourcedata)

    pss = ifelse.(ps .< 0.05, "*", "ns")
    psr = pss .*" \np=" .* string.(round.(ps,digits=3))
    hline!(ax,[0],color=:grey, ls=:dash, label=nothing)
    plot!(ax, legends=:topleft, foreground_color_legend=nothing, background_color_legend=nothing)
    plot!(ax, left_margin=50*Plots.px)
    annotate!(ax, tickpoints, [0.3,0.3,0.6,0.6], text.( psr, 8, :black, :top, :center))#, fontsize=6, halign=:center, valign=:bittin, color=:black)
    # do the above annotate with just 2 digits and scientific notation of the p-values
    

    xticks!(ax,tickpoints,["V1\nvisual", "V1\nauditory", "ACC\nvisual", "ACC\nauditory"])
    xlims!(ax,0,14)
    yticks!(ax,[0,0.2,0.4])
    ylims!(ax,-0.1,0.65)

    # title!(ax,"mouse")
    ylabel!(ax, "accuracy difference\nrelevant - irrelevant")

    @panellabel ax "l" -0.30 1.20











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


    
    axs = plot(layout=(2,3), size=(3*350, 2*300), dpi=dpi, bottom_margin=20*Plots.px, left_margin=20*Plots.px, grid=false, legend=false)




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
    sourcedata = DataFrame()
    sourcedata[!,Symbol("timestamps")] = timestamps
    for n in 1:nmice
        m = MathUtils.convolve( accuraciesmice[n,cx,:,1],ones(sm)./sm)
        plot!(ax, timestamps, m, lw=1, color=:grey, alpha=0.3, label=nothing)
        sourcedata[!,Symbol("accuracymouse$(n)")] = m
    end
    m = MathUtils.convolve( dropdims(mean(accuraciesmice[:,cx,:,1],dims=1),dims=1), ones(sm)./sm)
    semcv = dropdims(mean(accuraciesmice[:,cx,:,3],dims=1),dims=1)
    e = MathUtils.convolve( dropdims(std(accuraciesmice[:,cx,:,1],dims=1),dims=1)/sqrt(nmice)+semcv, ones(sm)./sm)
    sourcedata[!,Symbol("accuracyallmean")] = m
    sourcedata[!,Symbol("accuracyallsem")] = e

    plot!(ax, timestamps, m, ribbon=e, lw=2, fillalpha=0.3, color=:mediumvioletred, label=nothing)
    xlims!(ax,-1.2,4.2)
    ylims!(ax,0.45,1.05)
    ylabel!(ax, "context accuracy")
    xlabel!(ax, "time [s]")

    @panellabel ax "a"
    CSV.write(joinpath(config[:publicationfigurespath],"sourcedata","fig3","fig3-a.csv"), sourcedata)



  


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
    
    @panellabel ax "b"
    sourcedata = DataFrame(Pair.(mouseids,[cognitivesurplus[mx,:] for mx in 1:nmice])...)
    insertcols!(sourcedata, 1, :timestamps => timestamps)
    CSV.write(joinpath(config[:publicationfigurespath],"sourcedata","fig3","fig3-b.csv"), sourcedata)


    
    ax = axs[1,3]
    # outcome history mice
    nlookback = 3
    vx = 3
    contexts = ["visual","auditory"]
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
    sourcedata = DataFrame(timestampslookback=timestampslookback)
    for cx in eachindex(contexts)
        m = @movingaverage(accuracy[cx,:,1],sm)
        e = @movingaverage(accuracy[cx,:,3],sm)
        plot!(ax, timestampslookback, m, ribbon=e, lw=2, fillalpha=0.3, color=colors[cx], label=contexts[cx]*" context")
        sourcedata[!,Symbol("accuracy"*contexts[cx]*"mean")] = m
        sourcedata[!,Symbol("accuracy"*contexts[cx]*"sem")] = e
    end
    CSV.write(joinpath(config[:publicationfigurespath],"sourcedata","fig3","fig3-c.csv"), sourcedata)
    for h in 1:nlookback
        @decoderbackground(ax, config[:stimulusstart]+(nlookback-h)*6, config[:stimulusend]+(nlookback-h)*6, config[:waterstart]+(nlookback-h)*6, bg=nothing)
    end
    xlims!(ax,-1.2,4.2+12)
    ylims!(ax,0.45,1.05)
    plot!(ax,legend=:topright, foreground_color_legend=nothing, background_color_legend=nothing, bottom_margin=50*Plots.px)
    xlabel!(ax, "time from reference trial [s]")
    ylabel!(ax, "decision accuracy")

    @panellabel ax "c"





   
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
    sourcedata = DataFrame(visual=Int[], auditory=Int[], v=Float64[], a=Float64[])
    for (vix,vi) in enumerate([45,135]), (aux,au) in enumerate([5000,10000])
        mask = (triallist[!,:degree].==vi) .& (triallist[!,:freq].==au)
        scatter!(ax,S[mask,1],S[mask,2], color=colors[vix,aux], markerstrokewidth=0, markersize=3)         # label=string(vi)*"° "*string(au)*" Hz"

        # legend annotation
        scatter!(ax, -0.10 .+ [0] .+ (vix-1)*w,  0.17 .+ [0] .+ (aux-1)*w, color=colors[vix,aux], markerstrokewidth=0, markersize=3)

        sourcedata = vcat(sourcedata, DataFrame(visual=vi, auditory=au, v=S[mask,1], a=S[mask,2]))
    end
    CSV.write(joinpath(config[:publicationfigurespath],"sourcedata","fig3","fig3-d.csv"), sourcedata)
    annotate!(ax, -0.10+w/2, 0.17+w+0.02, "visual\ngo nogo", font(pointsize=6, color=:black, halign=:center, valign=:bottom))
    annotate!(ax, -0.10-w+0.005, 0.17+w/2,  "auditory nogo\ngo", font(pointsize=6, color=:black, halign=:right, valign=:center))


    xlims!(ax,-0.27,0.27)
    ylims!(ax,-0.27,0.27)
    xlabel!(ax,"visual DV")
    ylabel!(ax,"auditory DV\n(orthogonal projection)")

    @panellabel ax "d"






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
        ttest = OneSampleTTest(angle, 90)
        @info "" pvalue(ttest) ttest.t
        push!(ps,pvalue(ttest))
    end

    labels = ["visual","auditory","context","decision","reward"]
    micercoords = 5*ones(nmice)
    

    ax = axs[2,2]
    scatter!(ax, angles[:,1,2], micercoords, color=:darkcyan, markerstrokewidth=0, label="visual - auditory, p=$(round(ps[1],digits=2))")
    vline!(ax, [90], color=:grey, ls=:dash,label=nothing)
    xlims!(ax, 0, 180)
    ylims!(ax,0,15)
    plot!(ax, legend=:topleft, foreground_color_legend=nothing, background_color_legend=nothing, yaxis=false)
    xlabel!(ax, "angle between DVs [°]")
    @panellabel ax "e"



    ax = axs[2,3]
    scatter!(ax, angles[:,1,3], micercoords.+1, color=:purple, markerstrokewidth=0, label="context - visual, p=$(round(ps[2],digits=2))")
    scatter!(ax, angles[:,2,3], micercoords, color=:olive,  markerstrokewidth=0, label="context - auditory, p=$(round(ps[3],digits=2))")
    scatter!(ax, angles[:,3,4], micercoords.-1, color=:darkgoldenrod,  markerstrokewidth=0, label="context - decision, p=$(round(ps[4],digits=2))")
    vline!(ax, [90], color=:grey, ls=:dash,label=nothing)
    xlims!(ax, 0, 180)
    ylims!(ax,0,15)
    plot!(ax, legend=:topleft, foreground_color_legend=nothing, background_color_legend=nothing, markerstrokewidth=0, yaxis=false)
    xlabel!(ax, "angle between DVs [°]")
    @panellabel ax "f"

    sourcedata = DataFrame(mouseids=mouseids,visualauditory=angles[:,1,2], contextvisual=angles[:,1,3], contextauditory=angles[:,2,3], contextdecision=angles[:,3,4])
    CSV.write(joinpath(config[:publicationfigurespath],"sourcedata","fig3","fig3-ef.csv"), sourcedata)








    
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


    axs = plot(layout=@layout(  [ a b c{0.5w}; d{0.22w} e{0.1w} f{0.1w} g h; i j k{0.5w}] ), size=(4*350, 3*350), dpi=dpi,
               bottom_margin=20*Plots.px, left_margin=40*Plots.px, grid=false, legend=false)
    
    # [a b c{0.5w}]          heights=[1/3,1/3,1/3])
    # axs = plot(layout=(4,3), size=(3*350, 4*300), dpi=dpi,
    #            bottom_margin=20*Plots.px, left_margin=40*Plots.px, grid=false, legend=false)
    
    # axs = plot(layout=grid(4,3, widths=[1/4 1/4 1/2; 1/3 1/3 1/3; 1/3 1/3 1/3]), size=(3*350, 3*350), dpi=dpi,
    #            bottom_margin=20*Plots.px, left_margin=40*Plots.px, grid=false, legend=false)
    

    # model panels
    machineparameters = YAML.load_file("params-rnn.yaml"; dicttype=Dict{Symbol,Any})



    # RNN schematics
    ax = axs[1]

    # plot!(ax,axis=false,xlims=(0,20),ylims=(0,20))
    
    # plot!(ax, [10,11,11,10,10],[5,5,15,15,5], lw=3, color=:black)
    
    # plot!(ax,[(6,8),(9,8)],arrow=arrow(:closed), lw=3, color=:navy)
    # annotate!(ax, 4, 8, text("visual", 10, :right, :navy, "Helvetica Bold"))
    
    # plot!(ax,[(6,10),(9,10)],arrow=arrow(:closed), lw=3, color=:darkgreen)
    # annotate!(ax, 4, 10, text("auditory", 10, :right, :darkgreen, "Helvetica Bold"))
    
    # plot!(ax,[(6,17),(9,13)],arrow=arrow(:closed), lw=3, color=:red)
    # annotate!(ax, 4, 19, text("previous reward", 10, :left, :red, "Helvetica Bold"))

    # plot!(ax,[(12,11),(15,11)],arrow=arrow(:closed), lw=3, color=:darkorange)
    # annotate!(ax, 16, 11, text("decision", 10, :left, :darkorange, "Helvetica Bold"))

    # plot!(ax, map(u->(u[1]+10.5,u[2]+3.9), Plots.partialcircle(0/360*2*pi,30/360*2*pi,30,2)), color=:black, lw=3)
    # plot!(ax, map(u->(u[1]+10.5,u[2]+3.9), Plots.partialcircle(150/360*2*pi,360/360*2*pi,30,2)), color=:black, lw=3)
    # plot!(ax, map(u->(u[1]+10.65,u[2]+4.35), Plots.partialcircle(150/360*2*pi,155/360*2*pi,2,1.75)), color=:black, lw=3, arrow=(:tail,:closed))
    # annotate!(ax, 10.5, 0, text("previous state", 10, :center, :bottom, :black, "Helvetica Bold"))

    # @panellabel ax "a" -0.2 1.05

    
    rnnschematicsimage = load(joinpath(config[:publicationfigurespath],"parts/","RNN-architecture-schematics-circle.png"))
    plot!(ax, rnnschematicsimage)#, top_margin=50*Plots.px, left_margin=30*Plots.px)
    # xlims!(ax, 40,1380)
    # ylims!(ax, 110,600)
    plot!(ax, xticks=false, yticks=false, axis=false)

    @panellabel ax "a" -0.25 -0.82

    




    ax = axs[2]
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

    @panellabel ax "b" -0.3 1.05
    sourcedata = DataFrame()
    sourcedata[!,Symbol("epoch")] = snapshots
    sourcedata[!,Symbol("contextaccuracymean")] = cm
    sourcedata[!,Symbol("contextaccuracysem")] = ce
    sourcedata[!,Symbol("fractioncorrectcongruentmean")] = fm[:,2]
    sourcedata[!,Symbol("fractioncorrectcongruentsem")] = fe[:,2]
    sourcedata[!,Symbol("fractioncorrectincongruentmean")] = fm[:,3]
    sourcedata[!,Symbol("fractioncorrectincongruentsem")] = fe[:,3]
    CSV.write(joinpath(config[:publicationfigurespath],"sourcedata","fig4","fig4-b.csv"), sourcedata)




    ax = axs[3]
    # plot an example of congruent and incongruen trials timecourse with context and decision
    # this particular trial is saved as congruent = [false, true, true, false, true]
    modelidexample = "0114"
    incongruentsequence = [false, false, true, false, true]
    @load(joinpath(config[:modelresultspath],"analysis/decoders/", "decoders-$(modelidexample).bson"), @__MODULE__,
          fractioncorrects, contextaccuracy, Ts, decisionpoints)
    times = (1:Ts[end][end]*5) .- Ts[end][end]*(5-1)
    for (gx,g) in enumerate(incongruentsequence)
        ind = (gx-1)*Ts[end][end]+1:min(gx*Ts[end][end]+1,length(times))
        plot!(ax,times[ind],fractioncorrects[end,ind,4], lw=2, color=:darkorange, ls=[:solid,:dash][Int(g)+1])
        plot!(ax,times[ind],fractioncorrects[end,ind,1], lw=1, color=:gray)
    end
    plot!(ax,times,contextaccuracy[end,:,2], lw=2, color=:mediumvioletred)
    for s in -60:15:0 
        @decoderbackground(ax, s+Ts[2][begin], s+Ts[3][end], s+Ts[3][begin], bg=nothing)
    end
    ylims!(ax,0.45,1.05)
    ylabel!(ax, "acc. / frac. corr.")
    xlabel!(ax, "timesteps from reference trial")

    @panellabel ax "c" -0.13 1.05

    sourcedata = DataFrame()
    sourcedata[!,Symbol("times")] = times
    sourcedata[!,Symbol("fractioncorrectdecision")] = fractioncorrects[end,:,4]
    sourcedata[!,Symbol("fractioncorrectdecisionmean")] = contextaccuracy[end,:,1]
    sourcedata[!,Symbol("fractioncorrectcontext")] = contextaccuracy[end,:,2]
    CSV.write(joinpath(config[:publicationfigurespath],"sourcedata","fig4","fig4-c.csv"), sourcedata)







    
    ax = axs[4]
    # outcome history
    nlookback = 3
    contexts = ["visual","auditory"]
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
    sourcedata = DataFrame(timestampslookback=timestampslookback)
    for cx in eachindex(contexts)
        plot!(ax, timestampslookback, accuracy[cx,:,1], ribbon=accuracy[cx,:,3], 
                  lw=2, fillalpha=0.3, color=colors[cx], label=contexts[cx]*" context")
        sourcedata[!,Symbol("accuracy"*contexts[cx]*"mean")] = accuracy[cx,:,1]
        sourcedata[!,Symbol("accuracy"*contexts[cx]*"sem")] = accuracy[cx,:,3]
    end
    CSV.write(joinpath(config[:publicationfigurespath],"sourcedata","fig4","fig4-d.csv"), sourcedata)
    for h in 1:nlookback
        @decoderbackground(ax, Ts[2][begin]+(nlookback-h)*Ts[end][end], Ts[3][end]+(nlookback-h)*Ts[end][end], Ts[2][end]+(nlookback-h)*Ts[end][end], bg=nothing)
    end
    # xlims!(ax,-1.2,4.2+12)
    ylims!(ax,0.45,1.05)
    plot!(ax,legend=(0.2,0.2), foreground_color_legend=nothing, background_color_legend=nothing)
    xlabel!(ax, "timesteps from reference trial")
    ylabel!(ax, "decision accuracy")

    @panellabel ax "d" -0.3 1.1


    





    # model suppression

    # angles
    nsequence = 5
    nhidden = 30
    filenamebase = machineparameters[:modeltype]*"-s$(machineparameters[:nsequence])-h$(machineparameters[:nhidden])"
    @load(joinpath(config[:modelanalysispath],  "distances", filenamebase*"-angles.bson"), @__MODULE__, angles, validmodels)
    angles = angles[validmodels,:,:]          # vis-aud, cx-vis, cx-aud, cx-dec, vis-dec, aud-dec
    comparetimepoints = [Ts[2][1],Ts[2][1],Ts[2][1],Ts[3][1],Ts[3][1],Ts[3][1]]
    nmodels = size(angles,1)


    # stats
    ps = []
    for a in axes(angles,3)
        if a>4 break end
        angle = angles[:,comparetimepoints[a],a]
        ttest = OneSampleTTest(angle, 90)
        @info "" pvalue(ttest) ttest.t
        push!(ps,pvalue(ttest))
    end

    labels = ["visual","auditory","context","decision","reward"]
    

    ax = axs[5]

    m = mean(angles[:,comparetimepoints[1],1])
    s = 2*std(angles[:,comparetimepoints[1],1])
    scatter!(ax, [m], [1.0], color=:darkcyan, markerstrokewidth=0, label="visual ⋅ auditory")
    plot!(ax, [m-s,m+s],[1.0, 1.0], color=:darkcyan, lw=2, label=nothing)
    vline!(ax, [90], color=:grey, ls=:dash,label=nothing)
    xlims!(ax, 0, 180)
    ylims!(ax,0,2)
    plot!(ax, legend=:topleft, foreground_color_legend=nothing, background_color_legend=nothing, yaxis=false)
    annotate!(ax, 120, 1.0, text("p=$(round(ps[1],digits=2))",:darkcyan, :left, :vcenter, 8))
    xlabel!(ax, "angle between DVs [°]", xguidefontsize=8)
    xticks!(ax, [45, 90, 135])
    plot!(ax, right_margin=0*Plots.px)    # keep together the same letter-labeled panels

    @panellabel ax "e" -0.3 1.1

    ax = axs[6]

    m = mean(angles[:,comparetimepoints[2],2])
    s = 2*std(angles[:,comparetimepoints[2],2])
    scatter!(ax, [m], [1.1], color=:purple, markerstrokewidth=0, label=" context ⋅ visual")
    plot!(ax, [m-s,m+s],[1.1, 1.1], color=:purple, lw=2, label=nothing)

    m = mean(angles[:,comparetimepoints[3],3])
    s = 2*std(angles[:,comparetimepoints[3],3])
    scatter!(ax, [m], [1.0], color=:olive,  markerstrokewidth=0, label=" context ⋅ auditory")
    plot!(ax, [m-s,m+s],[1.0, 1.0], color=:olive, lw=2, label=nothing)

    m = mean(angles[:,comparetimepoints[4],4])
    s = 2*std(angles[:,comparetimepoints[4],4])
    scatter!(ax, [m], [0.9], color=:darkgoldenrod,  markerstrokewidth=0, label=" context ⋅ decision")
    plot!(ax, [m-s,m+s],[0.9, 0.9], color=:darkgoldenrod, lw=2, label=nothing)

    vline!(ax, [90], color=:grey, ls=:dash,label=nothing)
    xlims!(ax, 0, 180)
    ylims!(ax,0,2)
    plot!(ax, legend=:topleft, foreground_color_legend=nothing, background_color_legend=nothing, markerstrokewidth=0, yaxis=false)
    annotate!(ax, 120, 1.1, text("p=$(round(ps[2],digits=2))",:purple, :left, :vcenter, 8))
    annotate!(ax, 120, 1.0, text("p=$(round(ps[3],digits=2))",:olive, :left, :vcenter, 8))
    annotate!(ax, 120, 0.9, text("p=$(round(ps[4],digits=2))",:darkgoldenrod, :left, :vcenter, 8))
    xlabel!(ax, "angle between DVs [°]", xguidefontsize=8)
    xticks!(ax, [45, 90, 135])
    plot!(ax, left_margin=0*Plots.px, right_margin=30*Plots.px)    # keep together the same letter-labeled panels

    @panellabel ax "f" -0.3 1.1

    sourcedata = DataFrame()
    sourcedata[!,Symbol("anglesvisualauditory")] = angles[:,comparetimepoints[1],1] 
    sourcedata[!,Symbol("anglescontextvisual")] = angles[:,comparetimepoints[2],2]
    sourcedata[!,Symbol("anglescontextauditory")] = angles[:,comparetimepoints[3],3]
    sourcedata[!,Symbol("anglescontextdecision")] = angles[:,comparetimepoints[4],4]
    CSV.write(joinpath(config[:publicationfigurespath],"sourcedata","fig4","fig4-ef.csv"), sourcedata)





    # stimulus subspaces

    @load( joinpath(config[:modelanalysispath],  "contextrepresentation", "contextprojections-rnn.bson"), @__MODULE__, SRs, SRPs, validmodels)
    SRs = SRs[validmodels,:,:,:]
    SRPs = SRPs[validmodels,:,:,:,:]    # nmodels, stimulus, ncontexts, ngonogo, ntimepoints
    nmodels = sum(validmodels)
    ntimecourse = Ts[end][end]
    timestamps = 61:75
    labelstimulus = ["visual","auditory"]
    labelinstr = ["go","nogo"]
    colors = [:deepskyblue :blue; :lime :green]
    sourcedata = DataFrame()
    for sx in 1:2, rx in 1:2
        cx = 2 - (sx==rx)
        ax = axs[6+sx]
        m = mean(SRPs[:,sx,cx,:,timestamps],dims=2)[:,1,:]
        # plot!(ax, m', color=colors[sx,rx], ls=[:solid, :dash][gx], alpha=0.2)
        plot!(ax, (1:ntimecourse) .- Ts[2][begin], mean(m,dims=1)', ribbon=std(m,dims=1)'/sqrt(nmodels), color=colors[sx,rx], lw=2, fillalpha=0.3,
                    label=["relevant","irrelevant"][rx]*"($(labelstimulus[cx]) context)")
        sourcedata[!,Symbol("stimulus"*labelstimulus[sx]*labelinstr[rx]*"mean")] = mean(m,dims=1)[1,:]
        sourcedata[!,Symbol("stimulus"*labelstimulus[sx]*labelinstr[rx]*"sem")] = std(m,dims=1)[1,:]/sqrt(nmodels)
        if rx==1
            # @nolinebackground(ax,Ts[2][1],Ts[3][end],Ts[3][1],bg=:darkgrey)
            vspan!(ax,[Ts[2][begin],Ts[3][end]].-Ts[2][begin], color=:grey, alpha=0.3, label=nothing)
            vline!(ax,[Ts[3][begin]].-Ts[2][begin],color=:white, alpha=0.5, lw=2, label=nothing)
            hline!(ax,[0],color=:grey, ls=:dash, label=nothing)
            plot!(ax, legend=:bottomleft, foreground_color_legend=nothing, background_color_legend=nothing)
            ylims!(ax,-0.5,1.5)
        end
        if rx==1 && sx==1
            ylabel!(ax,"stimulus projection")
            @panellabel ax "g" -0.3 1.1
        end
        xlabel!(ax,"timesteps from stimulus onset")
        title!(ax,labelstimulus[sx]*" stimulus")
    end









    # context representation flip
    # ax = axs[9]
    # @load( joinpath(config[:modelanalysispath],  "contextrepresentation", "contextprojections-rnn.bson"), @__MODULE__, CRs, CRPs, validmodels)
    # CRs = CRs[validmodels,:,:,:]
    # CRPs = CRPs[validmodels,:,:,:,:]    # nmodels, ncontexts, ngonogo, ntimepoints, nprojection
    # nmodels = sum(validmodels)

    # colors = [:fuchsia, :purple]
    # colors = [:deeppink, :rebeccapurple]
    # lss = [:solid, :dash]

    # # show the context DV projection difference between contexts at pre and dec projection operators
    # # @nolinebackground(ax,Ts[2][1],Ts[3][end],Ts[3][1],bg=:white)
    # vspan!(ax,[Ts[2][begin],Ts[3][end]].-Ts[2][begin], color=:grey, alpha=0.3, label=nothing)
    # vline!(ax,[Ts[3][begin]].-Ts[2][begin],color=:white, alpha=0.5, lw=2, label=nothing)
    # hline!(ax,[0],color=:grey, ls=:dash, label=nothing)
    # for (px,ct) in enumerate([1,3])
    #     for (dx,m) in enumerate(( CRPs[:,1,1,timestamps,ct] - CRPs[:,2,1,timestamps,ct],
    #                               CRPs[:,1,2,timestamps,ct] - CRPs[:,2,2,timestamps,ct]) )
    #         plot!(ax, (1:ntimecourse) .- Ts[2][begin], mean(m,dims=1)', ribbon=std(m,dims=1)'/sqrt(nmodels), color=colors[px], lw=2, ls=lss[dx], fillalpha=0.3,
    #                 label=["pre","dec"][px]*" "*["go","nogo"][dx])
    #     end
    #     # vline!(ax,[[Ts[1][2],Ts[2][1],Ts[3][1]][ct]],color=colors[px],lw=2,label=nothing)
    #     plot!(ax,fill([Ts[1][3],Ts[2][1],Ts[3][1]][ct],2) .-Ts[2][begin],[-1,-0.7],color=colors[px],lw=2,label=nothing)
    # end
    # ylims!(ax,-0.8,0.8)
    # yticks!(ax,[-0.5,0,0.5])
    # plot!(ax, legend=:topleft, foreground_color_legend=nothing, background_color_legend=nothing)
    # ylabel!(ax,"context projection")
    # xlabel!(ax,"timesteps from stimulus onset")
    # @panellabel ax "h" -0.3 1.1






    # output/decision subspace

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

    labelstimulus = ["visual","auditory"]
    # colors = [:dodgerblue, :lime]

    # find the abstract go and nogo cells with the highest enhancement - suppression
    h = abstractcells[:,end-ntimecourse+1:end,:,:,:,:]      # last trial only in each sequence
    # calculate the difference between the relevant and irrelevant responses for each cell, separately for go and nogo cells:
    es = h[:,Ts[3][begin],:,:,1,1] + h[:,Ts[3][begin],:,:,2,2] - h[:,Ts[3][begin],:,:,1,2] - h[:,Ts[3][begin],:,:,2,1]
    # find the indices of the best go and nogo cells
    s = argmax(es,dims=2)

    
    labelscontexts = ["visual","auditory"]
    labelsrelevancies = ["relevant" "irrelevant"; "irrelevant" "relevant"]
    colors = [:deepskyblue :green; :blue :lime]     # relevancy x stimulus
    maxcells = 1
    sourcedata = DataFrame()
    for (sx,stimulus) in enumerate(labelstimulus)
        ax = axs[8+sx]
        vspan!(ax,[Ts[2][begin],Ts[3][end]].-Ts[2][begin], color=:grey, alpha=0.3, label=nothing)
        vline!(ax,[Ts[3][begin]].-Ts[2][begin],color=:white, alpha=0.5, lw=2, label=nothing)
        hline!(ax,[0],color=:grey, ls=:dash, label=nothing)
    
        for cx in ([1,2],[2,1])[sx]
            context = labelscontexts[cx]
        
            d = similar(h[:,:,1:2*maxcells,1,cx,sx])
            for k in 1:nmodels   d[k,:,:] = cat( [ h[k,:,s[k,maxcells,gx][2],gx,cx,sx] for gx in 1:2 ]..., dims=2)  end
            d = reshape(permutedims(d, (1,3,2)), (nmodels*2*maxcells,ntimecourse))

            m = dropdims(mean(d,dims=(1)), dims=(1))
            e = dropdims(std(d,dims=(1)), dims=(1)) ./ sqrt(nmodels*2*maxcells)
            plot!(ax, (1:ntimecourse) .- Ts[2][begin], m, ribbon=e, color=colors[cx,sx], lw=2,
                      label=labelsrelevancies[cx,sx]*" ("*context*" context)")
            
            sourcedata[!,Symbol(stimulus*context*"mean")] = m
            sourcedata[!,Symbol(stimulus*context*"sem")] = e
        end
        plot!(ax, legend=:topleft, foreground_color_legend=nothing, background_color_legend=nothing)
        title!(ax,labelstimulus[sx]*" stimulus")
        ylims!(ax,-0.5,0.5)
        xlims!(ax,Ts[1][begin]-Ts[2][begin],Ts[end][end]-Ts[2][begin])
        xlabel!(ax,"timesteps from stimulus onset")
        if sx==1
            ylabel!(ax,"activity")
            @panellabel ax "h" -0.3 1.1
        else
            plot!(ax, left_margin=20*Plots.px)    # keep together the same letter-labeled panels
        end
    
    end
    CSV.write(joinpath(config[:publicationfigurespath],"sourcedata","fig4","fig4-h.csv"), sourcedata)
    





    
    # plot    enhancement - suppression   vs    performance
    # performances dimensions: (nmodels, context, go/nogo, congruency)
    p = minimum(performances[:,:,:,2],dims=(2,3))[:,1,1] # incongruent only
    ax = axs[11]

    ess = vcat( [ es[s[:,1,gx]] for gx in 1:2 ]... )
    scatter!(ax, p, ess, color=:black, markerstrokewidth=0 )
    
    
    â,r,pv,t = getcorrelation(repeat(p,2), ess) # this does not work yet in MathUtils
    plot!(ax, [0,1.], â[1].*[0,1.].+â[2], color=:red, lw=1.5, alpha=0.8)
    r = round(r,digits=2)
    # if round(pv,digits=8) == 0 pvs = "≤1${}$" else pvs = "=$(round(pv,digits=8))" end
    pve = ceil(log10(pv)) # this equals 1e-12, but typing below manually is easier
    annotate!(ax, 0.1, 3, "r=$(r) p<10⁻¹¹", font(pointsize=8, color=:red, halign=:left))
    xlims!(ax,0,1)
    xlabel!(ax,"performance")
    ylabel!(ax,"activity difference\nrelevant - irrelevant")
    plot!(ax, right_margin=20*Plots.px)
    @info "performance vs activity difference:" r pv t

    @panellabel ax "i" -0.1 1.1

    sourcedata = DataFrame(p=pv,ess=ess)
    CSV.write(joinpath(config[:publicationfigurespath],"sourcedata","fig4","fig4-i.csv"), sourcedata)

    

    
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



    axs = plot(layout=@layout([ a{0.333w} b c; d e f g]),size=(4*300+70, 2*300+30), legend=false, dpi=dpi,
               left_margin=15*Plots.px, bottom_margin=15*Plots.px, top_margin=-15*Plots.px, right_margin=15*Plots.px)





    # # subspace suppression schematics

    # ax = axs[1,1]
    # plot!(ax, aspect_ratio=:equal, left_margin=30*Plots.px, top_margin=30*Plots.px)
    # plot!(ax,[(2,7),(10,7)], lw=2, alpha=0.5, color=:navy, label="visual subspace")
    # plot!(ax,[(2,7),(2,15)], lw=2, alpha=0.5, color=:darkgreen, label="auditory subspace")
    # plot!(ax,[(2,7),(8,7)],arrow=(5,5,:closed), lw=5, color=:navy, label="visual activity")
    # plot!(ax,[(2,7),(2,13)],arrow=(5,5,:closed), lw=5, color=:darkgreen, label="auditory activity")
    # plot!(ax,legend=:topleft, foreground_color_legend=nothing, background_color_legend=nothing)
    # xlims!(ax,-15,10)
    # ylims!(ax,0,20)
    # title!(ax,"total stimulus space")
    # plot!(ax, axis=false)

    # @panellabel ax "a" -0.2 1.1

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
    # annotate!(ax,0,15,text("auditory context",8,:right,:darkgreen))
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
    ax = axs[1]
    crossmutualimage = load(joinpath(config[:publicationfigurespath],"parts/","cross,mutual.png"))
    plot!(ax, crossmutualimage, top_margin=50*Plots.px, left_margin=30*Plots.px, bottom_margin=45*Plots.px)
    # xlims!(ax, 40,1380)
    # ylims!(ax, 110,600)
    plot!(ax, xticks=false, yticks=false, axis=false)
    @panellabel ax "b" -0.18 -0.57
    @panellabel ax "c"  0.45  -0.57



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


    ax = axs[2]
    # colorspace = :RdBu   # :diverging_gkr_60_10_c40_n256
    colorspace = cm
    M = (allFmg+allFmn)/2      # average over go and nogo representations
    nhidden = size(M,1)
    # Mr = fillmatrixtimeline(M,5,1,6,1:1)
    heatmap!(ax, M, color=colorspace, clim=limsabsmaxnegpos(M), aspect_ratio=:equal, yflip=true, axis=false)
    # narrow colorbar:
    plot!(inset_subplots=(2, bbox(1.02, 0, 0.05, 1, :bottom, :left)))
    plot!(axs[end], axis=false)
    heatmap!(twinx(axs[(*)(length(axs))]), repeat(-1:1/200:1,1,2), color=colorspace,
              yticks=((0,200,400),("-1 AU","0 AU","1 AU")), colorbar=false, xticks=false)
    
    @panellabel ax "d" -0.2 -0.2
 
    ax = axs[3]
    plot!(ax,  axis=false, aspect_ratio=:equal, colormap=false)
    
    # M = M^6       # nonlinearmatrixpower(tanh,M,6,1)
    Mr = fillmatrixtimeline(M,5,1,6,1:6)
    heatmap!(ax, Mr, color=colorspace, clim=limsabsmaxnegpos(Mr), aspect_ratio=:equal, yflip=true, axis=false)
    # narrow colorbar:
    # plot!(ax, inset_subplots=(6, bbox(1.0, 0, 0.05, 1, :bottom, :left)), axis=false)
    # heatmap!(twinx(axs[(*)(length(axs))]), repeat(-1:1/200:1,1,2), color=colorspace, yticks=((0,200,400),("-1 AU","0 AU","1 AU")), colorbar=false, xticks=false)
    @panellabel ax "e" -0.2 -0.2

    sourcedata = Tables.table(M)
    CSV.write(joinpath(config[:publicationfigurespath],"sourcedata","fig5","fig5-de.csv"), sourcedata)




    # activity structure with modality index context index
    # load for V1, ACC, and models
    panels = ["g","h","f"]
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
        contextindex = dropdims(mean([contextindex[modalityunitssingle[:,1],:,1];;; contextindex[modalityunitssingle[:,2],:,2] ],  dims=3), dims=3)
        push!(modalityindexV1,modalityindex)
        push!(contextindexV1,contextindex)
        nmicemax[1] = max(nmicemax[1],size(contextindex,1))
    end
    mouseidsACC = collectsubjectids("ACC")
    modalityindexACC = []
    contextindexACC = []
    for mouseid in mouseidsACC
        @load(joinpath(config[:cachepath],"subspace/","mutualswitchsubspaces-$(string(mouseid)).bson"),  @__MODULE__,
             modalityunitssingle, modalityindexsingle, contextindex)
        modalityindex = dropdims(mean([modalityindexsingle[modalityunitssingle[:,1],1] modalityindexsingle[modalityunitssingle[:,2],2] ],  dims=2), dims=2)
        contextindex = dropdims(mean([contextindex[modalityunitssingle[:,1],:,1];;; contextindex[modalityunitssingle[:,2],:,2] ],  dims=3), dims=3)
        push!(modalityindexACC,modalityindex)
        push!(contextindexACC,contextindex)
        nmicemax[2] = max(nmicemax[2],size(contextindex,1))
    end
    # load model data
    @load( joinpath(config[:modelanalysispath],  "suppression", "subspaces-rnn-reduced.bson"), @__MODULE__, allCim)
    @load( joinpath(config[:modelanalysispath],  "suppression", "subspaces-rnn.bson"), @__MODULE__,
                 Fmgs, Fmns,  Mis, Cis, Dis, Ris, Cims, Ccms, Rims, Rigns, DWis, DWhs, DWos, validmodels) 


    # plot context indices with neuron ordering by modality index
    numoutliers = 6 # leave out this many neurons at the beginning and end of the neuron ordering
    agpanel = [2,3,1]
    neuronpositionsconcat = Int64[]
    contextindicesconcat = zeros(Float64, 0, 3)
    for ag in [1,2,3]
        sourcedata = DataFrame(mouseid=String[], neuronposition=Int[], contextindex=Float64[])
        ax = axs[3+agpanel[ag]]
        if ag<3
            contextindex = ag==1 ? contextindexV1 : contextindexACC
            mouseids = ag==1 ? mouseidsV1 : mouseidsACC
            miceneuronpositions = 1:nmicemax[ag]
            contextindices = NaN .* ones(nmicemax[ag],length(mouseids),4)
            neuronpositionsconcat = Int64[]
            contextindicesconcat = zeros(Float64, 0, 4)
            for mid in eachindex(mouseids)
                nneurons = size(contextindex[mid][:,4],1)
                shift = (nmicemax[ag] - nneurons)÷2
                neuronpositions = miceneuronpositions[(1:nneurons) .+ shift]
                # for mean display
                contextindices[neuronpositions,mid,:] = contextindex[mid]
                # concatenate for correlation
                neuronpositionsconcat = [neuronpositionsconcat; neuronpositions]
                contextindicesconcat = [contextindicesconcat; contextindex[mid]]
                sourcedata = vcat(sourcedata, DataFrame(mouseid=mouseids[mid], neuronposition=neuronpositions, contextindex=contextindex[mid][:,4]))
            end
            CSV.write(joinpath(config[:publicationfigurespath],"sourcedata","fig5","fig5-"*["g","h"][ag]*".csv"), sourcedata)

            # individual mice
            plot!(ax, contextindices[:,:,4], lw=1, color=:grey, alpha=0.5)

            # mean of mice
            miceneuronpositionsnooutlier = miceneuronpositions[numoutliers:end-numoutliers]       # take into account only multiple mice data
            ci = [ mean(contextindices[n, .! isnan.(contextindices[n,:,4]),4]) for n in miceneuronpositionsnooutlier]
            plot!(ax,miceneuronpositionsnooutlier,ci,lw=2,color=:lightseagreen)

            

            # correlation statistics
            â,r,p,t = getcorrelation(neuronpositionsconcat, contextindicesconcat[:,4])
            plot!(ax, miceneuronpositions[[1,end]], â[1].*miceneuronpositions[[1,end]].+â[2], color=:red, ls=[:dash :solid][ag], lw=1.5, alpha=0.8)
            @info "corr" brainarea=["V1","ACC"][ag] â[2] r p t n=length(neuronpositionsconcat)
            r = round(r,digits=2)
            # p = round(p,digits=[2,4][ag])
            p = ["=0.41","<10⁻⁶"][ag]
            annotate!(ax, 3, -6, "r=$(r) p$(p)", font(pointsize=8, color=:red, halign=:left))

            hline!(ax,[0], ls=:dash, color=:grey, alpha=0.5)
            ylims!(ax, -8, 8)

            xticks!(ax,[1,nmicemax[ag]÷2,nmicemax[ag]])

        elseif ag==3
            Cims = Cims[validmodels,:,:,:]
            nneurons = size(Cims,2)
            
            colorseries = [:purple, :maroon, :orange, :lightseagreen]
            for px in 4:4        # context index calculation timepoint: pre, at start, dec, all

                m = dropdims(mean(Cims[:,:,px,:],dims=(1,3)),dims=(1,3)) # mean over models and go-nogo
                e = dropdims(std(Cims[:,:,px,:],dims=(1,3)),dims=(1,3))./sqrt(2*size(Cims,1)) 
                plot!(ax, 1:nneurons, m, ribbon=e, lw=2, color=colorseries[px], facealpha=0.3)

                # correlation statistics
                â,r,p,t = getcorrelation(1:nneurons, m)
                plot!(ax, [1,nneurons], â[1].*[1,nneurons].+â[2], color=:red, lw=1.5, alpha=0.8)
                @info "RNN models, corr context modality" sum(validmodels)  â[2] r p t
                r = round(r,digits=2)
                # p = round(p,digits=4)        # <10⁻³ <10⁻⁶
                p = ["<10⁻³","<10⁻¹⁶","<10⁻¹⁶","<10⁻¹⁶"]
                annotate!(ax, [1,nneurons÷2,1,2][px], [-0.06,0.14,-0.2,-0.15][px], "r=$(r) p$(p[px])",
                              font(pointsize=8, color=:red, halign=[:left,:center,:left,:left][px]))
            

                sourcedata = DataFrame(neuronposition=1:nneurons, contextindex=m, sem=e)
            end

            hline!(ax,[0], ls=:dash, color=:grey, alpha=0.5)
            ylims!(ax, -0.2, 0.2)
            yticks!(ax, -0.2:0.1:0.2)
            xticks!(ax,[1,size(allCim,1)÷2,size(allCim,1)])
            CSV.write(joinpath(config[:publicationfigurespath],"sourcedata","fig5","fig5-f.csv"), sourcedata)
        end

        return

        # title!(ax,labelagent[ag])
        annotate!(ax,0,[10,10,0.26][ag],text(labelagent[ag],10,:left))

        xlabel!(ax, "neurons")
        ylabel!(ax, "context index")
    
        @panellabel ax panels[ag] -0.3 1.3

    end

    ax = axs[7]
    # plot!(ax,  axis=true)
    labels = ["pre","start","dec"]
    colors = [:purple, :maroon, :orange]
    rs = []
    ps = []
    ts = []
    for px in 1:3
        â,r,p,t = getcorrelation(neuronpositionsconcat, contextindicesconcat[:,px])
        @info "$(labels[px])" r p
        push!(rs,r)
        push!(ps,p)
        push!(ts,t)
        # pss = ["=2⋅10⁻³","=0.02","=2⋅10⁻⁴"]
        rd = round(r,digits=2)
        p = round(p,digits=3)
        annotate!(ax, px, 0.01, "r=$(rd)\np=$(p)", font(pointsize=8, color=:red, halign=:center, :bottom))
    end
    @info "pre start dec stats" rs ps ts
    bar!(ax, [1, 2, 3], rs, widths=0.1, color=colors, linecolor=colors, label=nothing)
    ylims!(ax,-0.33,0.0)
    yticks!(ax, -0.3:0.1:0)
    xticks!(ax,[1,2,3],labels)#,rotation=45)
    xlabel!(ax, "context index time")
    ylabel!(ax, "correlation")
    @panellabel ax "i" -0.4 1.3

    sourcedata = DataFrame(indextime=labels, r=rs, p=ps)
    CSV.write(joinpath(config[:publicationfigurespath],"sourcedata","fig5","fig5-i.csv"), sourcedata)

    # plot!(axs[7], inset_subplots=(7, bbox(0.70, 0.80, 0.30, 0.20, :bottom, :left)))
    # labels = ["pre","start","dec"]
    # ax = axs[end]
    # @nolinebackground ax 0 3 2 bg=:white
    # for (ix,interval) in enumerate(([-1,-0.25],[0,0.75],[1.75,2.5]))
    #     vspan!(ax, interval, color=colors[ix], label=nothing)
    # end
    # ylims!(ax,0.4,1)
    # xlims!(ax,-1.5,4.5)
    # plot!(ax, yaxis=nothing)
    # xticks!(ax, [0,2,3.],["", "", ""])
    # for x in [0,2,3] annotate!(ax, x, 0.32, text("$x", 6, :top)) end
    # annotate!(ax, 1.5, 0.17, text("time from stimulus onset [s]", 6, :top))




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
            contextindicesconcat = [contextindicesconcat; contextindex[mid][:,3]]
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



"""
plot each mouse behaviour and chance consistent occurrence
"""
function supplementaryfigure1()

    figurepostfix = "behavior,detailed"
    @info "making supplementary figure 1" figurepostfix


    mouseids = collectsubjectids("ACC")


    l = @layout([  a{0.5w} b{0.25w} c;
                   d{0.5w} e{0.25w} f;
                   g{0.5w} h{0.25w} i;
                   j{0.5w} k{0.25w} l ])

    axs = plot(layout=l, size=(4*350,4*300), dpi=dpi, legend=nothing)
    axinsetzero = 12
    plot!(axs[1], left_margin=40*Plots.px, top_margin=30*Plots.px)
    

    for (n,mouseid) in enumerate(mouseids)
        
        spn = (n-1)*3+1
        ax = axs[spn]

        plot!(ax, axis=false)
        
        labels = ["── go congruent" "--- go incongruent"; "── nogo congruent" "--- nogo incongruent"]
        nwbfile,_ = loadnwbdata(mouseid)
        triallist = nwbdf(nwbfile.trials)
        filter!(:difficulty=>u->u=="complex",triallist)
        addcongruencycolumn!(triallist)
        maperfs = movingaverageperformancetrialtypes(triallist)
        highperfs = highperformancetrialsmask(maperfs)
        contextboundary = findfirst(triallist[!,:context].==triallist[end,:context])
        boundaries = [1 contextboundary-1; contextboundary nrow(triallist)]
        choices = choicesselected(triallist)
        sourcedata = DataFrame()
        for sx in 1:2, congi in 1:2
            ix = (sx-1)*2+congi
            plot!(axs[spn], inset_subplots=(spn, bbox(0, 1-ix*0.22, 1, 0.18, :bottom, :left)))
            ax = axs[axinsetzero+(n-1)*5+ix]
            for cx in 1:2         # contexts
                if sx+congi==2 annotate!(ax, sum(boundaries[cx,1]),1.1, text(["visual","auditory"][cx]*"\ncontext",[:navy,:darkgreen][cx],:left, :bottom, 8)) end
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
            sourcedata[!,Symbol(["go","nogo"][sx]*"-"*["congruent","incongruent"][congi])] = maperfs[:,sx,congi]
        end
        CSV.write(joinpath(config[:publicationfigurespath],"sourcedata","sfig1","sfig1-"*["a","b","c","d"][n]*".csv"), sourcedata)
        
        plot!(axs[spn], inset_subplots=(spn, bbox(0, 0, 1, 0.06, :bottom, :left)))
        ax = axs[axinsetzero+(n-1)*5+5]
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
        
        @panellabel axs[spn] ["a","b","c","d"][n] -0.1 1.12
        @panellabel axs[spn+1] ["e","f","g","h"][n] -20 0.3


    end


    # probabilities of sequence lengths
    # axsp = plot(layout=(2,1))


    # load a dict of repeated probability simulations by mouseid containing a dict for each context
    # load mouse consecutive consistent blocks numbers


    @load(joinpath(config[:cachepath], "behaviour", "probabilityconsistentperiods,incongruentbias.bson"), @__MODULE__, probabilityconsistentperiods)
    @load(joinpath(config[:cachepath], "behaviour", "movingaveragestatistics,mice.bson"), @__MODULE__, consecutivesegmentsmice)

    # markershapes = [:circle,:square,:diamond,:utriangle]
    markershapes = repeat([:circle],length(mouseids))
    labels = ["number of consistent trials", "length of longest consecutive\nconsistent trials"]

    for (n,mouseid) in enumerate(mouseids)
        spn = (n-1)*3+1

        numconsistent = sum.(consecutivesegmentsmice[mouseid])
        conseqconsistent = maximum.(consecutivesegmentsmice[mouseid])

        # find the less likely context, and restrict display to that context
        for (cx,context) in enumerate(["visual","audio"])
            ax = axs[spn+cx]
            @info "$mouseid $context" numconsistent cx context p=probabilityconsistentperiods[mouseid][context][:pchoicebias]
            ntrials = probabilityconsistentperiods[mouseid][context][:ntrials]
            prob_ntrials_successes = probabilityconsistentperiods[mouseid][context][:prob_ntrials_successes] 
            prob_atleastone_consecutive_length = probabilityconsistentperiods[mouseid][context][:prob_atleastone_consecutive_length]


            # sizes: prob_ntrials_successes: (ntrials), prob_atleastone_consecutive_length: (ntrials, conseucive length)
            prob_ntrials_successes[prob_ntrials_successes.==0] .= 0.5/config[:generativeconsecutivesamplingrepeats]
            prob_atleastone_consecutive_length[prob_atleastone_consecutive_length.==0] .= 0.5/config[:generativeconsecutivesamplingrepeats]

            plot!(ax, prob_ntrials_successes, color=:dodgerblue, lw=1.5, yscale=:log10, alpha=0.5, label="simulation")
            # plot!(ax, prob_atleastone_consecutive_length, lw=1.5, color=:purple, alpha=0.5, label=nothing)

            xlims!(ax, 0, 60)
            ylims!(ax, 1e-7, 1e-1)

            
            
            if cx==1 plot!(ax, ylabel="probability of\nbehaviour by chance", left_margin=30*Plots.px) end
            if n==4 plot!(ax, xlabel="number of consistent trials") end

            scatter!(ax, [numconsistent[cx]], [prob_ntrials_successes[numconsistent[cx]]],
                        markercolor=:dodgerblue, markerstrokewidth=0, markershape=markershapes[n], alpha=0.8, label="simulation (number of trials from mouse)")#labels[1])
            # scatter!(ax, [conseqconsistent[cx]], [prob_atleastone_consecutive_length[conseqconsistent[cx]]],
            #             markercolor=:purple, markerstrokewidth=0, markershape=markershapes[n], alpha=0.8, label=labels[2])
            
            if n==1 annotate!(ax, 30, 0.11, text(["visual","auditory"][cx]*" context", :hcenter, [:blue,:green][cx], 10)) end
            if n==1 && cx==1 plot!(ax, legend=:bottomleft, foreground_color_legend=nothing, background_color_legend=nothing, legendfontsize=8) end

            sourcedata = DataFrame(probability=prob_ntrials_successes)
            # sourcedata[!,
            CSV.write(joinpath(config[:publicationfigurespath],"sourcedata","sfig1","sfig1-"*["e","f","g","h"][n]*["-visual","-auditory"][cx]*".csv"), sourcedata)

        end


    end

    # @panellabel ax "d" -0.20 1.2


    plot!(axs, tick_direction=:out, xgrid=false, ygrid=false)
    display(axs)

    savefig(joinpath(config[:publicationfigurespath],"SupplementaryFigure1-$(figurepostfix).png"))
    savefig(joinpath(config[:publicationfigurespath],"SupplementaryFigure1-$(figurepostfix).pdf"))


end






function supplementaryfigure2()
    figurepostfix = "irrelevant,suppression,individual"
    @info "making figure 2" figurepostfix


    axs = plot(layout=(8,4),size=(4*350, 8*300), legend=false, dpi=dpi,
               left_margin=20*Plots.px, bottom_margin=15*Plots.px, top_margin=20*Plots.px, right_margin=20*Plots.px)





    # panels first 2×2 blocks
    # mice suppression

    
    mouseidsv1 = collectsubjectids("V1")
    nmicev1 = length(mouseidsv1)
    mousev1ind = 1:nmicev1

    mouseidsacc = collectsubjectids("ACC")
    nmiceacc = length(mouseidsacc)
    mouseaccind = nmicev1+1:nmicev1+nmiceacc


    labelsstimulus = ["visual","auditory"]
    labelscontexts = ["visual","auditory"]
    labelsrelevancies = ["relevant" "irrelevant"; "irrelevant" "relevant"]
    colors = [:deepskyblue :green; :blue :lime]     # relevancy x stimulus

    timestamps = []
    accuracieslists = []
    projectionslists = []
    for mouseids in [mouseidsv1,mouseidsacc]
        for mouseid in mouseids
            @load(config[:cachepath]*"subspace/decode,relevancy-$(string(mouseid)).bson", @__MODULE__, timestamps, accuracies, projections)
            push!(accuracieslists, accuracies[1,:,:,:,:])     # collect only the decoder trained and tested on all trials
            push!(projectionslists, projections[1,:,:,:,:,:])
        end
    end
    accuracieslists = vcat(reshape.(accuracieslists,1,size(accuracieslists[1])...)...)
    projectionslists = vcat(reshape.(projectionslists,1,size(projectionslists[1])...)...)


    sm = 51 # smoothing window

    for (brx,mouseind) in enumerate((mousev1ind,mouseaccind))
        if brx==1 continue end
        sourcedata = DataFrame()
        for (stimulusindex,stimulus) in enumerate(labelsstimulus)
            for contextindex in ([1,2],[2,1])[stimulusindex]
                context = labelscontexts[contextindex]

                # draw individual mice
                for (mx,mouseid) in enumerate(mouseind)
                    ax = axs[stimulusindex+(mx-1)*2,1] # axs[stimulusindex,bx]
                    @decoderbackground(ax, config[:stimulusstart], config[:stimulusend], config[:waterstart], bg=:white)
                    m = @movingaverage(accuracieslists[mouseid,contextindex,stimulusindex,:,1],sm)
                    e = @movingaverage(accuracieslists[mouseid,contextindex,stimulusindex,:,3],sm)
                    plot!(ax, timestamps, m, ribbon=e, lw=1.5, color=colors[contextindex,stimulusindex], alpha=0.85, fillalpha=0.25,
                          label=labelsrelevancies[contextindex,stimulusindex]*" ("*context*" context)")
                    
                    # # draw mean
                    # m = @movingaverage(dropdims(mean(accuracieslists[mouseind,contextindex,stimulusindex,:,1],dims=1),dims=1),sm)
                    # e = @movingaverage(dropdims(std(accuracieslists[mouseind,contextindex,stimulusindex,:,1],dims=1),dims=1)/sqrt(length(mouseind))+
                    #     dropdims(mean(accuracieslists[mouseind,contextindex,stimulusindex,:,3],dims=1),dims=1), sm)
                    # plot!(ax, timestamps, m, ribbon=e, lw=2, fillalpha=0.3, color=colors[contextindex,stimulusindex], alpha=0.8,
                    #         label=labelsrelevancies[contextindex,stimulusindex]*" ("*context*" context)")
                    # if bx==1 ylabel!(ax, "accuracy"); plot!(ax, left_margin=30*Plots.px) end
                    plot!(ax,legend=:topleft, foreground_color_legend=nothing, background_color_legend=nothing)
                    xlims!(ax,-1.2,4.2)
                    ylims!(ax,0.45,1.05)
                    title!(ax, ["decoding\nmouse $mx",""][stimulusindex])
                    if stimulusindex==2 xlabel!(ax, "time from stimulus onset [s]") end
                    ylabel!(ax, stimulus*" stimulus\naccuracy");
                    if stimulusindex==2 plot!(ax, bottom_margin=60*Plots.px) end
                    plot!(ax, left_margin=60*Plots.px, right_margin=40*Plots.px)
                    if mx==1 && stimulusindex==1 @panellabel ax "a" -0.25 1.15 end

                    sourcedata[!,Symbol("mouse$mouseid"*labelsrelevancies[contextindex,stimulusindex]*"("*context*"context)mean")] = m
                    sourcedata[!,Symbol("mouse$mouseid"*labelsrelevancies[contextindex,stimulusindex]*"("*context*"context)sem")] = e
                end
            end
        end
        CSV.write(joinpath(config[:publicationfigurespath],"sourcedata","sfig2","sfig2-a"*".csv"), sourcedata)
    end





    # projections to DVs
    colors = [:deepskyblue :darkblue; :lime :darkgreen]
    for (brx,mouseind) in enumerate((mousev1ind,mouseaccind))
        if brx==1 continue end
        sourcedata = DataFrame()
        for (sx, stimulus) in enumerate(labelsstimulus)
            for (rx,labelrelevance) in enumerate(["relevant","irrelevant"])
                cx = [sx,3-sx][rx]
                for (mx,mouseid) in enumerate(mouseind)
                    ax = axs[sx+(mx-1)*2,2]
                    @nolinebackground(ax, config[:stimulusstart], config[:stimulusend], config[:waterstart], :darkgray)
                    hline!(ax,[0],ls=:dash,color=:darkgray, label=nothing)
                    m = @movingaverage(abs.(projectionslists[mouseid,sx,cx,1,:,1] - projectionslists[mouseid,sx,cx,2,:,1]), sm)
                    e = @movingaverage(projectionslists[mouseid,sx,cx,1,:,3] + projectionslists[mouseid,sx,cx,2,:,3], sm)
                    plot!(ax, timestamps, m, ribbon=e, lw=2, color=colors[sx,rx],
                            alpha=0.85, fillalpha=0.25, label="$(labelrelevance) ($(labelscontexts[cx]) context)")
                    plot!(ax, legend=:topleft, foreground_color_legend=nothing, background_color_legend=nothing)
                    ylabel!(ax, "$(stimulus) projection\ngo-nogo abs. diff.")
                    if sx==2 xlabel!(ax, "time from stimulus onset [s]") end
                    title!(ax, ["subspace projection\nmouse $mx",""][sx])
                    ylims!(ax, 0, ylims(ax)[2])
                    if mx==1 && sx==1 @panellabel ax "b" -0.25 1.15 end

                    sourcedata[!,Symbol("mouse$mouseid"*labelrelevance*"("*labelscontexts[cx]*"context)mean")] = m
                    sourcedata[!,Symbol("mouse$mouseid"*labelrelevance*"("*labelscontexts[cx]*"context)sem")] = e
    
                end

                # ax = axs[sx,3]
                # m = abs.(projections[sx,rx,1,:,1]-projections[sx,rx,2,:,1])
                # e = projections[sx,rx,1,:,3]+projections[sx,rx,2,:,3]
                # m = @movingaverage(m,31)
                # e = @movingaverage(e,31)
                # plot!(ax, timestamps, m, ribbon=e, lw=2, color=colors[sx,rx], label=labelsrelevancy[rx])
                # if cx==1 ylabel!(ax, "early $(labelsmodalities[sx])\nprojection difference go-nogo") end
                # @nolinebackground(ax, config[:stimulusstart], config[:stimulusend], config[:waterstart], :darkgray)
                # hline!(ax,[0],ls=:dash,color=:darkgray, label=nothing)
                # # ylims!(ax,0,800)


            end
        end
        CSV.write(joinpath(config[:publicationfigurespath],"sourcedata","sfig2","sfig2-b"*".csv"), sourcedata)
    end






    # right 2×2 block: mouse performance and suppression

    # accuracies  is (nstates,nmodalities,nrelevancies,ntimestamps,3)               (last is stats)
    # coefficients  is (nstates,nmodalities,nrelevancies,ntimestamps,nneurons+1)      # last is intercept
    # projections is (nstates,nmodalities,nrelevancies,2,ntimestamps,3)             (2 is gonogo, last is stats)
    accuraciesallareas = []
    # colors = [:fuchsia :gold; :purple  :darkorange]       #  vis-aud × consistency
    colors = [:purple :darkorange]                         # consistency
    alphas = [1.0 0.8]
    relevancylabels = ["relevant","irrelevant"]
    consistencylabels = ["consistent","exploratory"]
    for (brx,brainarea) in enumerate(["V1","ACC"])
        if brx==1 continue end
    
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

        sourcedatac = DataFrame()
        sourcedatad = DataFrame()
        for sx in 1:2        # stimulus modalities
            for rx in 1:2    # relevancies
                for bx in 1:2        # consistency
                    
                    # individual
                    for (mx,mouseid) in enumerate(mouseids)
                        ax = axs[sx+(mx-1)*2,2+rx]
                        vspan!(ax,[config[:stimulusstart],config[:stimulusend]], color=:grey, alpha=0.3, label=nothing)
                        vline!(ax,[config[:waterstart]],color=:white, alpha=0.5, lw=2, label=nothing)
                        hline!(ax,[0.5],color=:grey, ls=:dash, label=nothing)
                        m = @movingaverage(accuraciesall[mx,bx,sx,rx,:,1],sm)
                        e = @movingaverage(accuraciesall[mx,bx,sx,rx,:,3],sm)
                        plot!(ax,timestamps,m,ribbon=e,lw=1.5,color=colors[bx], alpha=0.85, fillalpha=0.25,
                              label=relevancylabels[rx]*", "*consistencylabels[bx])

                        # mean
                        # m = @movingaverage(dropdims(mean(accuraciesall[:,bx,sx,rx,:,1],dims=1),dims=1),sm)
                        # # e = @movingaverage(dropdims(std(accuraciesall[:,bx,sx,rx,:,1],dims=1),dims=1)/sqrt(nmice)+
                        # #     dropdims(mean(accuraciesall[:,bx,sx,rx,:,3],dims=1),dims=1), sm)
                        # # normal error bars have no meaning here, beacuse we want to compare within individual mice
                        # # so we show here the CV errors improved by the mouse-average
                        # e = @movingaverage(dropdims(mean(accuraciesall[:,bx,sx,rx,:,3],dims=1)/sqrt(nmice),dims=1), sm)
                        # plot!(ax,timestamps,m,ribbon=e,lw=3,color=colors[bx], alpha=alphas[rx], fillalpha=0.1,
                        #     label=relevancylabels[rx]*", "*consistencylabels[bx])
                        plot!(ax,legend=:topright, foreground_color_legend=nothing, background_color_legend=nothing)
                        # xlims!(ax,-1.2,4.2)
                        ylims!(ax,0.3,1.25)
                        yticks!(ax,0.5:0.25:1)
                        if bx==1
                            # ylabel!(ax, "accuracy");
                            plot!(ax,left_margin=50*Plots.px)
                        end
                        ylabel!(ax, ["visual","auditory"][sx]*" stimulus\naccuracy");
                        title!(ax, ["behaviour $(relevancylabels[rx])\nmouse $mx",""][sx])
                        if sx==2 xlabel!(ax, "time from stimulus onset [s]") end
                        if mx==1 && sx==1 @panellabel ax ["c", "d"][rx] -0.25 1.15 end

                        if sx==1
                            sourcedatac[!,Symbol("mouse$mouseid"*consistencylabels[bx]*relevancylabels[rx]*"mean")] = m
                            sourcedatac[!,Symbol("mouse$mouseid"*consistencylabels[bx]*relevancylabels[rx]*"sem")] = e
                        elseif sx==2
                            sourcedatad[!,Symbol("mouse$mouseid"*consistencylabels[bx]*relevancylabels[rx]*"mean")] = m
                            sourcedatad[!,Symbol("mouse$mouseid"*consistencylabels[bx]*relevancylabels[rx]*"sem")] = e
                        end
                    end
                end
            end
        end
        CSV.write(joinpath(config[:publicationfigurespath],"sourcedata","sfig2","sfig2-c.csv"), sourcedatac)
        CSV.write(joinpath(config[:publicationfigurespath],"sourcedata","sfig2","sfig2-d.csv"), sourcedatad)
    end




    
    
    
    plot!(axs, tick_direction=:out, xgrid=false, ygrid=false)
    display(axs)

    if config[:publishfigures]
        savefig(joinpath(config[:publicationfigurespath],"SupplementaryFigure2-$(figurepostfix).png"))
        savefig(joinpath(config[:publicationfigurespath],"SupplementaryFigure2-$(figurepostfix).pdf"))
    end



end




function supplementaryfigure3()
    
    figurepostfix = "context,driftcontrol"
    @info "making supplementary figure 3" figurepostfix

    blockwidth = 60
    nblocks = 600 ÷ blockwidth
    sm = 51 # smoothing window
    colors = [:deeppink,:rebeccapurple,:black]
    
    sourcedata = DataFrame()
    
    axs = plot(layout=(1,4),size=(1.2* 4*300, 1*300), legend=false, left_margin=30*Plots.px, top_margin=15*Plots.px, bottom_margin=30*Plots.px, dpi=dpi)

    mouseids = collectsubjectids("ACC")
    for (mx,mouseid) in enumerate(mouseids)
    
        @load(config[:cachepath]*"subspace/decode,context,trialcourse-$(string(mouseid)).bson", @__MODULE__, timestamps, accuracies, coefficients)
        accuraciesholdmidedgeout = copy(accuracies)
        # load uncontrolled context
        @load(config[:cachepath]*"subspace/decode,variables-$(string(mouseid)).bson", @__MODULE__, accuracies, coefficients)

        
        ax = axs[mx]
        @decoderbackground(ax, config[:stimulusstart], config[:stimulusend], config[:waterstart], :white)
        
        accuraciesblockaveraged = zeros(2,nblocks)
        for (i,accur) in enumerate((accuraciesholdmidedgeout[1,:,:],accuraciesholdmidedgeout[2,:,:],accuracies[3,:,:]))
            m = MathUtils.convolve(accur[:,1],ones(sm)./sm)
            e = MathUtils.convolve(accur[:,3],ones(sm)./sm)
            plot!(ax, timestamps, m, ribbon=e, lw=2, color=colors[i], alpha=0.7, fillalpha=0.15, label=["test edge", "test middle","all"][i])
            xlims!(ax,-1.5,4.5)
            yticks!(ax,0.5:0.1:1.0)
            ylims!(ax,0.45,1.25)
            title!(ax, "mouse $mx")
            if mx==1 ylabel!(ax, "context accuracy") end
            xlabel!(ax, "time [s]")

            if i<3 accuraciesblockaveraged[i,:] = mean(reshape(accur[:,1],blockwidth,nblocks), dims=1)[1,:] end

            sourcedata[!,Symbol("mouse$mouseid"*["testedge","testmiddle","all"][i]*"mean")] = m
            sourcedata[!,Symbol("mouse$mouseid"*["testedge","testmiddle","all"][i]*"sem")] = e
        end
        plot!(ax, legend=:topleft, foreground_color_legend=nothing, background_color_legend=nothing)


        ttest = OneSampleTTest(accuraciesblockaveraged[1,:], accuraciesblockaveraged[2,:])
        p = pvalue(ttest)
        t = ttest.t

        annotate!(ax, 3.5, 1.124, text("n=$nblocks\np=$(round(p,digits=2))\nt=$(round(t,digits=2))", :left, 8))

        @panellabel ax ["a","b","c","d"][mx] -0.25 1.1

        CSV.write(joinpath(config[:publicationfigurespath],"sourcedata","sfig3","sfig3-"*["a","b","c","d"][mx]*".csv"), sourcedata)
    end


    plot!(axs, tick_direction=:out, xgrid=false, ygrid=false)




    display(axs)


    if config[:publishfigures]
        savefig(joinpath(config[:publicationfigurespath],"SupplementaryFigure3-$(figurepostfix).png"))
        savefig(joinpath(config[:publicationfigurespath],"SupplementaryFigure3-$(figurepostfix).pdf"))
    end

end














function reliableneurons(R2,aggr=mean,th=0.05)
    m = Float64[]
    conds = zeros(Int,size(R2,1))
    for t in axes(R2,1)
        cond = R2[t,:].>th
        push!(m,aggr(R2[t,cond]))
        conds[t] = findmax(R2[t,:])[2]
    end
    return m,conds
end


function interp(m,timestamps)
    masknotnan = .! isnan.(m)
    f = findfirst(masknotnan)
    l = findlast(masknotnan)
    if f===nothing || l===nothing
        return m, timestamps
    end
    masknotnanfullrange = f:l
    timestampsnotnan = timestamps[masknotnanfullrange]
    timeindex = collect(1:length(timestamps))
    timeindexnotnan = timeindex[masknotnan]
    mnotnan = m[masknotnan]
    itp = interpolate((timeindexnotnan,), mnotnan, Gridded(Linear()))
    # mi = itp.(timeindexnotnan)
    # return mi, timestampsnotnan
    mi = itp.(timeindex[masknotnanfullrange])
    return mi, timestampsnotnan
end


function supplementaryfigure4()

    
    figurepostfix = "choice,lickcontrol"
    @info "making supplementary figure 4" figurepostfix


    # right 2×2 block: mouse performance and suppression

    # accuracies  is (nstates,nmodalities,nrelevancies,ntimestamps,3)               (last is stats)
    # coefficients  is (nstates,nmodalities,nrelevancies,ntimestamps,nneurons+1)      # last is intercept
    # projections is (nstates,nmodalities,nrelevancies,2,ntimestamps,3)             (2 is gonogo, last is stats)
    accuraciesallareas = []
    # colors = [:fuchsia :gold; :purple  :darkorange]       #  vis-aud × consistency
    colorsangles = [:deepskyblue :blue; :lime :green]     # relevancy x stimulus
    colorslick = [:black, :red]                            # lick, no-lick
    alphas = [1.0 0.3]
    labelsrelevancy = ["relevant","irrelevant"]
    labelsconsistency = ["consistent","exploratory"]
    labelsstimulus = ["visual","auditory"]
    # colors = [:deepskyblue :green; :blue :lime]     # relevancy x stimulus
    sm = 31 # smoothing window


    brainarea = "ACC"
    
    mouseids = collectsubjectids(brainarea)
    nmice = length(mouseids)
    timestamps = nothing
    timestampindices = nothing
    coefficientsall = []
    projectionsall = []
    ntrialsprojectionall = []
    accuracieslickall = []
    ntrialslickall = []
    accuraciesnolickall = []
    ntrialsnolickall = []
    accuraciesfullall = []

    for mouseid in mouseids
        # choice geometry
        @load(config[:cachepath]*"subspace/decode,choicegeometry-$(string(mouseid)).bson", @__MODULE__, timestamps, accuracies, coefficients)
        push!(coefficientsall, coefficients) 
        

        # predictive choice geomertry (projecting late stimulus related activity onto choice-less early stimulus subspaces)
        @load(joinpath(config[:cachepath],"subspace/","choicegeometry,predictive-$(string(mouseid)).bson"), @__MODULE__, timestamps, timestampindices, projections, ntrials)
        push!(projectionsall, projections)
        push!(ntrialsprojectionall, ntrials)

        # lick control
        @load(joinpath(config[:cachepath],"subspace/","suppressionbehaviour,lickcontrol-$(string(mouseid)).bson"), @__MODULE__,
                        accuracies, ntrials)
        push!(accuracieslickall, accuracies)
        push!(ntrialslickall, ntrials)

        @load(joinpath(config[:cachepath],"subspace/","suppressionbehaviour,nolickcontrol-$(string(mouseid)).bson"), @__MODULE__,
            accuracies, ntrials)
        push!(accuraciesnolickall, accuracies)
        push!(ntrialsnolickall, ntrials)

        @load(joinpath(config[:cachepath],"subspace/","suppressionbehaviour-$(string(mouseid)).bson"), @__MODULE__,
                        accuracies)
        push!(accuraciesfullall, accuracies)
    end

    ntimestamps = length(timestamps)
    accuracieslickall = @catvecleftsingleton accuracieslickall
    projectionsall = @catvecleftsingleton projectionsall
    ntrialsprojectionall = @catvecleftsingleton ntrialsprojectionall
    ntrialsprojectionall = Int16.(ntrialsprojectionall)
    ntrialslickall = @catvecleftsingleton ntrialslickall
    ntrialslickall = Int16.(ntrialslickall)
    accuraciesnolickall = @catvecleftsingleton accuraciesnolickall
    ntrialsnolickall = @catvecleftsingleton ntrialsnolickall
    ntrialsnolickall = Int16.(ntrialsnolickall)
    accuraciesfullall = @catvecleftsingleton accuraciesfullall


    msps = []     # subplot collection per mouse
    for (mx, mouseid) in enumerate(mouseids)

        axs = plot(layout=(6,1),size=(1*350, 6*300), grid=false, legend=false, dpi=dpi)
                # , left_margin=20*Plots.px, bottom_margin=15*Plots.px, top_margin=60*Plots.px, right_margin=20*Plots.px)
        
        



        # partial neural activity prediction
        consistentlabel = ["",",consistent"][2]
        @load(config[:cachepath]*"subspace/predict,neurons$(consistentlabel)-$(string(mouseid)).bson", @__MODULE__, timestamps, R2s)
        nneurons = size(R2s,4)
        R2s[R2s.<0] .= NaN
        predictorcombinations = [ [1], [2], [4], [1,4], [2,4], [3], [1,2,3], [1,2,3,4]  ]
        labelscombinations = ["visual","audio","decision","visual+decision","audio+decision","context","visual-audio-context","visual-audio-context+reward"]
        labelscombinationsirrelevant = ["visual irrelevant","audio irrelevant","decision","visual irrelevant+decision","audio irrelevant+decision","context irrelevant","visual-audio-context irrelevant","visual-audio-context+reward irrelevant"]
        includeincompare = [1,2,3,4,5]
        nincludeincompare = length(includeincompare)
        colorscombinations = [:dodgerblue :lime :darkorange :purple :olive :mediumvioletred :grey :gold]
        colorscombinationsirrelevant = [:darkblue :darkgreen]
        ma = 21
        # aggr = x->reduce(+,x,init=0)/length(x)            # mean
        aggr = x->reduce(max, x, init=0)                   # maximum
        labelscontexts = ["all", "visual", "auditory"]
        # trialscontexts = [trues(ntrials), triallist[!,:context] .== "visual", triallist[!,:context] .== "audio" ]
        # R2sism = zeros(nincludeincompare,3,ntimestamps)
        conds = zeros(Int,nincludeincompare,3,ntimestamps)
        for (ix,i) in enumerate(includeincompare)                   # 1:ncombinations
            if i in [1,3,4]  # visual stimulus
                cx = 2     # visual context (relevant)
                m,conds[i,cx,:] = reliableneurons(R2s[i,cx,:,:,1],aggr)
                mi,timestampsnotnan = interp(m, timestamps)
                plot!(axs[1+4], timestamps, @movingaverage(m,3), lw=2, color=colorscombinations[i], alpha=0.05, label=nothing)
                plot!(axs[1+4], timestampsnotnan, @movingaverage(mi,ma), lw=2, color=colorscombinations[i], label=labelscombinations[i])
                # same stimulus in the irrelevant (opposite) context
                if i==1
                    cx = 3     # audio context (irrelevant)
                    m,conds[i,cx,:] = reliableneurons(R2s[i,cx,:,:,1],aggr)
                    mi,timestampsnotnan = interp(m, timestamps)
                    plot!(axs[2+4], timestamps, @movingaverage(m,3), lw=2, color=colorscombinationsirrelevant[i], alpha=0.05, label=nothing)
                    plot!(axs[2+4], timestampsnotnan, @movingaverage(mi,ma), lw=2, color=colorscombinationsirrelevant[i], alpha=0.5, label=labelscombinations[i]*" irrelevant")
                end
            end
            if i in [2,3,5]  # auditory stimulus
                cx = 3     # audio context (relevant)
                m,conds[i,cx,:] = reliableneurons(R2s[i,cx,:,:,1],aggr)
                plot!(axs[2+4], timestamps, @movingaverage(m,ma), lw=2, color=colorscombinations[i], label=labelscombinations[i])
                if i==2
                    cx = 2     # visual context (irrelevant)
                    m,conds[i,cx,:] = reliableneurons(R2s[i,cx,:,:,1],aggr)
                    mi,timestampsnotnan = interp(m, timestamps)
                    plot!(axs[1+4], timestamps, @movingaverage(m,3), lw=2, color=colorscombinationsirrelevant[i], alpha=0.05, label=nothing)
                    plot!(axs[1+4], timestampsnotnan, @movingaverage(mi,ma), lw=2, color=colorscombinationsirrelevant[i], alpha=0.5, label=labelscombinations[i]*" irrelevant")
                end
            end
            if i in [4]
                # irrelevant other stimulus + choice in this context:
                cx = 3
                m,conds[i,cx,:] = reliableneurons(R2s[i,cx,:,:,1],aggr)
                mi,timestampsnotnan = interp(m, timestamps)
                # R2sism[i,cx,:] = mi
                plot!(axs[2+4], timestamps, @movingaverage(m,3), lw=2, color=colorscombinations[i], alpha=0.05, label=nothing)
                plot!(axs[2+4], timestampsnotnan, @movingaverage(mi,ma), lw=2, color=colorscombinations[i], alpha=0.7, label=labelscombinationsirrelevant[i])
            end
            if i in [5]
                # irrelevant other stimulus + choice in this context:
                cx = 2
                m,conds[i,cx,:] = reliableneurons(R2s[i,cx,:,:,1],aggr)
                mi,timestampsnotnan = interp(m, timestamps)
                # R2sism[i,cx,:] = mi
                plot!(axs[1+4], timestamps, @movingaverage(m,3), lw=2, color=colorscombinations[i], alpha=0.05, label=nothing)
                plot!(axs[1+4], timestampsnotnan, @movingaverage(mi,ma), lw=2, color=colorscombinations[i], alpha=0.7, label=labelscombinationsirrelevant[i])
            end
            
        end

        for sx in 1:2
            cx = sx+1
            ax = axs[sx+4]
            @nolinebackground(ax, config[:stimulusstart], config[:stimulusend], config[:waterstart], bg=:white)
            plot!(ax, legend=:topleft, foreground_color_legend=nothing, background_color_legend=nothing)
            ylims!(ax, 0, 1.19)
            # title!(ax,["visual partial","auditory partial"][sx]*"\n$(labelscontexts[cx]) context")
            if mx==1 ylabel!(ax,["visual context","auditory context"][sx]*" R²") end
            if sx==2 xlabel!(ax,"time [s]") end
            if sx==1
                plot!(ax, top_margin=40*Plots.px)
                @panellabel ax ["i","j","k","l"][mx] -0.25 1.2
            end
        end



        # stats
        for sx in 1:2
            cx = sx+1            # each stimulus in its relevant context
            irsx = 3-sx         # other stimulus in the irrelevant context
            rel = []
            irrel = []
            ts = Int[]
            blockwidth = 60
            n = 3
            for t in 150-30+60+1:150+300   #axes(R2s,3)
            # if [cx,mx] in [[3,2]] tbl = 150+210+1:150+240; n = 6; else tbl = 150-30+60+1:150+300; n = 3 end
            # for t in tbl
                if n*blockwidth < t < (n+1)*blockwidth
                    n += 1
                    push!(rel, Float64[])
                    push!(irrel, Float64[])
                end
                # condc = R2s[sx+3,cx,t,:,1].>0    # conditions for valid predictable neurons
                # condd = R2s[3,cx,t,:,1].>0
                condr = R2s[sx+3,cx,t,:,1].>0    # conditions for valid predictable neurons
                condi = R2s[irsx+3,cx,t,:,1].>0
                # condr = R2sism[sx+3,cx,t].>0    # conditions for valid predictable neurons
                # condi = R2sism[irsx+3,cx,t].>0
                if sum(condr)>0 && sum(condi)>0       # if both lines are valid, fill in
                    push!(rel[end], aggr(R2s[sx+3,cx,t,condr,1]) )
                    push!(irrel[end], aggr(R2s[irsx+3,cx,t,condi,1]) )
                    # push!(rel[end], aggr(R2sism[sx+3,cx,t]) )
                    # push!(irrel[end], aggr(R2sism[irsx+3,cx,t]) )
                    # push!(ts, timestamps[t])   # save timepoints for diagnostics
                    # @info "$mouseid cx$sx block $n time $t" mean(rel[end]-irrel[end]) std(rel[end]-irrel[end])/sqrt(length(rel[end]))
                end
            end
            @info "$mouseid cx $sx choice control statistics" mean(mean.(rel .- irrel)) std(mean.(rel .- irrel))/sqrt(length(rel))
            # c = @movingaverage(vcat(c...),21)
            # d = @movingaverage(vcat(d...),21)

            ttest = OneSampleTTest(mean.(rel .- irrel))        # within block averaging to avoid autocorrelation
            p = pvalue(ttest) / 2      # mean - mean will be >, so we are checking only for >0,  one-tailed
            @info "$mouseid R² -> differs -> $(labelsstimulus[sx]) no.t.=$(length(rel))" p ttest.t m=mean.(rel .- irrel)
            # display(plot(ts,[c d]))
            p < 1e-12 ? ps = "p<10⁻¹²" : ps = "p=$(round(p,digits=3))"
            p < 1e-4 && p > 1e-12 ? ps = "p<10⁻⁴" : nothing
            t = round(ttest.t, digits=2)

            if ([cx,mx] in [[2,2],[2,3],[3,1],[3,2],[3,3]])
                # aesthetically remove panels without data
                plot!(axs[sx+4],legend=false)
            else # provide stats on panels with data
                annotate!(axs[sx+4], -1.3, 0.04, text("n=$(length(rel))\nt=$t\n$(ps)", :left, :bottom, 8))
            end
        end
        




        # predictive choice geometry
        sourcedata = DataFrame()
        colors = [:dodgerblue :darkblue; :lime :darkgreen]
        for (sx, stimulus) in enumerate(labelsstimulus)
            ax = axs[2+sx]
            for (rx,relevance) in enumerate(labelsrelevancy)
                m = abs.(projectionsall[mx,sx,rx,1,:,1]-projectionsall[mx,sx,rx,2,:,1])
                e = projectionsall[mx,sx,rx,1,:,3]+projectionsall[mx,sx,rx,2,:,3]
                m = @movingaverage(m,sm)
                e = @movingaverage(e,sm)
                plot!(ax, timestamps, m, ribbon=e, lw=2, color=colors[sx,rx], label=labelsrelevancy[rx])
                if mx==1 ylabel!(ax, "early $(labelsstimulus[sx]) proj.\ngo-nogo abs. diff.") end
                if sx==2 xlabel!(ax, "time from stimulus onset [s]") end
                plot!(ax, legend=:topright, foreground_color_legend=nothing, background_color_legend=nothing)
                @info "$(labelsstimulus[sx]) $(labelsrelevancy[rx])" gn=ntrialsprojectionall[mx,sx,rx,:]
                sourcedata[!,Symbol("mouse$mouseid"*labelsstimulus[sx]*labelsrelevancy[rx]*"mean")] = m
                sourcedata[!,Symbol("mouse$mouseid"*labelsstimulus[sx]*labelsrelevancy[rx]*"sem")] = e
            end
            @nolinebackground(ax, config[:stimulusstart], config[:stimulusend], config[:waterstart], bg=:white)
            @nolinebackground(ax, timestamps[timestampindices[1]], timestamps[timestampindices[end]], nothing, bg=nothing)
            hline!(ax,[0],ls=:dash,color=:darkgray, label=nothing)
            ylims!(ax, -5, 45)
            if sx==2 plot!(ax, bottom_margin=60*Plots.px) end
            if sx==1 @panellabel ax ["e","f","g","h"][mx] -0.25 1.1 end
        end
        CSV.write(joinpath(config[:publicationfigurespath],"sourcedata","sfig4","sfig4-"*["e","f","g","h"][mx]*".csv"), sourcedata)



        # lick controlled decoding
        colors = [:purple :darkorange]                         # consistency
        blockwidth = 60
        suppresstimes = 150+60+1:150+300
        nblocks = length(suppresstimes) ÷ blockwidth
        explickstats = zeros(nmice,2,2,nblocks)                    # (mice,stimulus,relevance,block-averaged time)
        lickandallstats = zeros(nmice,2,2,2,nblocks)                     # (mice,stimulus,relevance,licking,block-averaged time)
        sourcedata = DataFrame()
        for (lx, (accuraciesall,ntrialsall)) in enumerate(zip((accuracieslickall,accuraciesnolickall),(ntrialslickall,ntrialsnolickall)))
            if lx==2 continue end     # no concern with no lick
            lickcontrollabel = ["lick","no lick"][lx]
            for sx in 1:2        # stimulus modalities
                ax = axs[sx]
                vspan!(ax,[config[:stimulusstart],config[:stimulusend]], color=:grey, alpha=0.3, label=nothing)
                vline!(ax,[config[:waterstart]],color=:white, alpha=0.5, lw=2, label=nothing)
                hline!(ax,[0.5],color=:grey, ls=:dash, label=nothing)
                for bx in 2:2 # 1:2        # consistency
                    for rx in 1:2 # 1:2    # relevancies
                        @info "$mouseid" sx e="exploratory lick only tr=$(ntrialslickall[mx,bx,sx,rx,1]+ntrialsnolickall[mx,bx,sx,rx,1]) (go:nogo=$(ntrialsall[mx,bx,sx,rx,3]):$(ntrialsall[mx,bx,sx,rx,4]))" c="consistent lick only tr=$(ntrialslickall[mx,1,sx,rx,1]+ntrialsnolickall[mx,1,sx,rx,1]) (go:nogo=$(ntrialsall[mx,1,sx,rx,3]):$(ntrialsall[mx,1,sx,rx,4]))" crel="consistent lick only relevant tr=$(ntrialslickall[mx,1,sx,1,1]+ntrialsnolickall[mx,1,sx,1,1]) (go:nogo=$(ntrialsall[mx,1,sx,1,3]):$(ntrialsall[mx,1,sx,1,4]))"

                        # irrelevant, exploratory
                        if ntrialsall[mx,bx,sx,rx,1]>=10
                            m = @movingaverage(accuraciesall[mx,bx,sx,rx,:,1],sm)
                            e = @movingaverage(accuraciesall[mx,bx,sx,rx,:,3],sm)
                            mfa = @movingaverage(accuraciesfullall[mx,bx,sx,rx,:,1],sm)
                            efa = @movingaverage(accuraciesfullall[mx,bx,sx,rx,:,3],sm)
                            # normal error bars have no meaning here, beacuse we want to compare within individual mice
                            # so we show here the CV errors improved by the mouse-average
                            # e = @movingaverage(dropdims(mean(accuraciesall[mx:mx,bx,sx,rx,:,3],dims=1)/sqrt(nmice),dims=1), sm)
                            plot!(ax,timestamps,m,ribbon=e,lw=1,color=colorslick[lx], alpha=alphas[rx], fillalpha=0.1,
                                label="exploratory, $(labelsrelevancy[rx]), $(lickcontrollabel) only trials")
                            
                            # if lx==2
                            plot!(ax,timestamps,mfa,ribbon=efa,lw=3, color=colors[bx], alpha=alphas[rx], fillalpha=0.1,
                                label="exploratory, $(labelsrelevancy[rx]), all trials")


                            explickstats[mx,sx,rx,:] = mean(reshape(accuraciesall[mx,bx,sx,rx,suppresstimes,1],blockwidth,nblocks),dims=1)[1,:]
                            lickandallstats[mx,sx,rx,1,:] = mean(reshape(accuraciesall[mx,bx,sx,rx,suppresstimes,1],blockwidth,nblocks),dims=1)[1,:]
                            lickandallstats[mx,sx,rx,2,:] = mean(reshape(accuraciesfullall[mx,bx,sx,rx,suppresstimes,1],blockwidth,nblocks),dims=1)[1,:]

                            sourcedata[!,Symbol("mouse$mouseid"*labelsstimulus[sx]*labelsrelevancy[rx]*"explick"*lickcontrollabel*"mean")] = m
                            sourcedata[!,Symbol("mouse$mouseid"*labelsstimulus[sx]*labelsrelevancy[rx]*"explick"*lickcontrollabel*"sem")] = e
                            sourcedata[!,Symbol("mouse$mouseid"*labelsstimulus[sx]*labelsrelevancy[rx]*"expallmean")] = mfa
                            sourcedata[!,Symbol("mouse$mouseid"*labelsstimulus[sx]*labelsrelevancy[rx]*"expallsem")] = efa
                        end
                        # consistent control                            
                        # mfaconsr = @movingaverage(accuraciesall[mx,1,sx,1,:,1],sm) # consistent control relevant
                        # efaconsr = @movingaverage(accuraciesall[mx,1,sx,1,:,3],sm)
                        mfacons = @movingaverage(accuraciesall[mx,1,sx,2,:,1],sm) # consistent control irrelevant
                        efacons = @movingaverage(accuraciesall[mx,1,sx,2,:,3],sm)
    
                        # if ntrialsall[mx,1,sx,1,1]>=10 
                        #     plot!(ax,timestamps,mfaconsr,ribbon=efaconsr,lw=3, color=colors[1], alpha=alphas[1], fillalpha=0.1,
                        #         label="consistent, $(labelsrelevancy[1]), $(lickcontrollabel) only trials")
                        # end
                        if rx==2          # add a consistent control for irrelevant lick only
                            if ntrialsall[mx,1,sx,2,1]>=10 
                                plot!(ax,timestamps,mfacons,ribbon=efacons,lw=3, color=:fuchsia, alpha=alphas[rx], fillalpha=0.1,
                                    label="consistent, $(labelsrelevancy[rx]) $(lickcontrollabel) only trials")
                            end
                            sourcedata[!,Symbol("mouse$mouseid"*labelsstimulus[sx]*labelsrelevancy[rx]*"consallmean")] = mfacons
                            sourcedata[!,Symbol("mouse$mouseid"*labelsstimulus[sx]*labelsrelevancy[rx]*"consallsem")] = efacons
                        end
                    end
                end
                plot!(ax,legend=:topright, foreground_color_legend=nothing, background_color_legend=nothing, legendfontsize=8)
                xlims!(ax,-1.5,4.5)
                ylims!(ax,0.3,1.35)
                yticks!(ax,0.5:0.25:1)
                plot!(ax, left_margin=30*Plots.px)
                if mx==1 ylabel!(ax, "stimulus accuracy"); plot!(ax,left_margin=50*Plots.px) end
                # if sx==1 plot!(ax, top_margin=30*Plots.px) end
                if sx==2 xlabel!(ax, "time from stimulus onset [s]"); plot!(ax, bottom_margin=40*Plots.px) end
                # title!(ax, ["$(string(mouseid))\n$(labelsrelevancy[2]) $(labelsconsistency[2])\n",""][sx]*labelsstimulus[sx]*" stimulus")
                title!(ax, labelsstimulus[sx]*" stimulus")


                # relevant vs irrelevant
                if mx in [2,3]
                    ttest = OneSampleTTest(explickstats[mx,sx,1,:], explickstats[mx,sx,2,:])
                    p = pvalue(ttest)
                    t = ttest.t
                    annotate!(ax, 3.5, 0.37, text("p=$(round(p,digits=3))\nt=$(round(t,digits=1))", :left, 8))
                end

                # relevant and irrelevant    lick vs. all
                if [mx,sx] in [        [2,1], [3,1],          [1,2], [2,2], [3,2], [4,2] ]
                    ttest = OneSampleTTest(lickandallstats[mx,sx,1,1,:], lickandallstats[mx,sx,1,2,:])
                    p = pvalue(ttest)
                    t = ttest.t
                    annotate!(ax, -1.2, 0.37, text("p=$(round(p,digits=3))\nt=$(round(t,digits=1))", :left, 8))
                    if [mx,sx] == [4,2]
                        @info "" lickandallstats[mx,sx,1,1,:] lickandallstats[mx,sx,1,2,:] mean(lickandallstats[mx,sx,1,1,:]) mean(lickandallstats[mx,sx,1,2,:])
                    end
                end
                if [mx,sx] in [ [1,1], [2,1], [3,1], [4,1],          [2,2], [3,2] ]
                    ttest = OneSampleTTest(lickandallstats[mx,sx,2,1,:], lickandallstats[mx,sx,2,2,:])
                    p = pvalue(ttest)
                    t = ttest.t
                    annotate!(ax, 0.6, 0.37, text("p=$(round(p,digits=3))\nt=$(round(t,digits=1))", :left, 8))
                end

                
                if sx==1 @panellabel ax ["a","b","c","d"][mx] -0.25 1.15 end
                if sx==2 plot!(ax, bottom_margin=60*Plots.px) end
            end
        end
        
        plot!(axs, tick_direction=:out, xgrid=false, ygrid=false)#, ytickfonthalign=:left, xtickfontvalign=:bottom, xguidevalign=:top)
        
        push!(msps,axs)


        CSV.write(joinpath(config[:publicationfigurespath],"sourcedata","sfig4","sfig4-"*["a","b","c","d"][mx]*".csv"), sourcedata)
        
    end     

    maxs = plot(msps..., layout=(1,nmice), size=(nmice*350, 6*300), dpi=dpi)
    display(maxs)


    if config[:publishfigures]
        savefig(joinpath(config[:publicationfigurespath],"SupplementaryFigure4-$(figurepostfix).png"))
        savefig(joinpath(config[:publicationfigurespath],"SupplementaryFigure4-$(figurepostfix).pdf"))
    end

end




function statshelper()

    



    mouseids = collectsubjectids("ACC")
    # mouseids = ["AC006"]
    nmice = length(mouseids)
    
    triallists = []
    ifrs = []
    for mouseid in mouseids

        @info "mouseid" mouseid
        nwbfile,_ = loadnwbdata(mouseid)
        triallist = nwbdf(nwbfile.trials)
        filter!(:difficulty=>u->u=="complex",triallist)
        addcongruencycolumn!(triallist)




        neuronsspiketimeslist = gettrialrelativespiketimes(triallist, nwbfile.units)         # get spikes for trials
        ntrials = length(neuronsspiketimeslist)
        nneurons = length(neuronsspiketimeslist[1])
        trialwidth = config[:stimuluslength] + abs(config[:preeventpadding]) + abs(config[:posteventpadding])


        # selecting the neurons to display
        rastertriallist = [30:35,82:87]      # visual and auditory example trials

        @load(joinpath(config[:cachepath],"subspace/","mutualswitchsubspaces-$(string(mouseid)).bson"),  @__MODULE__,
                modalityunitssingle, modalityindexsingle, contextindex)

                
        # modalityindex = modalityindexsingle[:,1] - modalityindexsingle[:,2]
        modalityorder = [sortperm(modalityindexsingle[:,g]) for g in 1:2 ]
        ncellneighbours = 10
        bestvisualcells = [modalityorder[1][1:1+ncellneighbours-1], modalityorder[2][1:1+ncellneighbours-1]]         # go and nogo best
        bestaudiocells = [modalityorder[1][end-ncellneighbours+1:end], modalityorder[2][end-ncellneighbours+1:end]]
        selectedcells = [bestvisualcells[2][1] bestvisualcells[1][2]; bestaudiocells[1][6] bestaudiocells[2][3]]          # stim × gonogo

        

        # firing rate statistics
        ts, ifr = smoothinstantenousfiringrate(neuronsspiketimeslist, config[:stimuluslength], binsize=config[:dt])

        push!(triallists, triallist)
        push!(ifrs, ifr)
    end

    nneurons = map(mouseid->size(ifrs[mouseid],3), 1:nmice)
    @info "all neurons" nneurons sum(nneurons)

    

    # mean and variance of firing rate responsiveness to stimuli in the two contexts
    precision = 6
    prestim = 1:150
    earlystim = 151:200
    midstim = 251:300

    ns = zeros(Int,nmice,2,2,2,2,2)         # timeperiods, <>, cx, stim, go/nogo
    frcomp = zeros(2,2,2,2,2,4)         # timeperiods, <>, cx, stim, go/nogo, stats+pvalue
    for (px,stimchangepos) in enumerate([1,2])          # compare at between different time period pairs: 1: early-pre, 2: mid-early
        for (ox,op) in enumerate((<,>))
            for (cx,context) in enumerate(["visual","audio"]),
                 (sx,(col,vals)) in enumerate(zip( (:degree,:freq),((45,135),(5000,10000)) )),
                   (gx,val) in enumerate(vals)
                frp = []; fre = []; frm = []     # firing rate at periods
                for mx in 1:nmice
                    mask = (triallists[mx][!,:context].==context) .& (triallists[mx][!,col].==val)

                    # average over spec. timepoints -> (trials × neurons)
                    p = mean(ifrs[mx][mask,prestim,:],dims=2)[:,1,:]
                    e = mean(ifrs[mx][mask,earlystim,:],dims=2)[:,1,:]
                    m = mean(ifrs[mx][mask,midstim,:],dims=2)[:,1,:]


                    # calculate trial-averages and 2nd order stats
                    mp = mean(p,dims=1)[1,:]; sp = std(p,dims=1)[1,:]; ep = sp / sqrt(size(p,1))
                    me = mean(e,dims=1)[1,:]; se = std(e,dims=1)[1,:]; ee = se / sqrt(size(e,1))
                    mm = mean(m,dims=1)[1,:]; sm = std(m,dims=1)[1,:]; em = sm / sqrt(size(m,1))


                    if stimchangepos==1          # check the mean firing rate over trials for each neuron
                        maskop = op.(me,mp)      # and compare them between varioous time periods
                    else
                        maskop = op.(mm,me)
                    end

                    # add op-conditioned neurons time-resolved response to the list as bulk concatenation
                    frp = [frp; reshape(p[:,maskop], :)]
                    fre = [fre; reshape(e[:,maskop], :)]
                    frm = [frm; reshape(m[:,maskop], :)]
                    # frp = [frp; p[:,maskop]]
                    # fre = [fre; e[:,maskop]]
                    # frm = [frm; m[:,maskop]]

                    ns[mx,px,ox,cx,sx,gx] = sum(maskop)
                end
                # mp = mean(frp,dims=1)[1,:]; sp = std(frp,dims=1)[1,:]; ep = sp / sqrt(size(frp,1))
                # me = mean(fre,dims=1)[1,:]; se = std(fre,dims=1)[1,:]; ee = se / sqrt(size(fre,1))
                # mm = mean(frm,dims=1)[1,:]; sm = std(frm,dims=1)[1,:]; em = sm / sqrt(size(frm,1))
                # mp = mean(frp); sp = std(frp); ep = sp / sqrt(size(frp,1))
                # me = mean(fre); se = std(fre); ee = se / sqrt(size(fre,1))
                # mm = mean(frm); sm = std(frm); em = sm / sqrt(size(frm,1))

                # frcomp[px,ox,cx,sx,gx,:] = [ f(    [   (me - mp), (mm - me)   ][px] )
                #         for f in ( mean, std, u->std(u)/sqrt(sum(ns[:,px,ox,cx,sx,gx])) ) ]  ./ [mp, me][px]
                frcomp[px,ox,cx,sx,gx,1:3] = [ f(    [   (fre - frp), (frm - fre)   ][px] )
                        for f in ( mean, std, u->std(u)/sqrt(sum(ns[:,px,ox,cx,sx,gx])) ) ]  ./ mean([frp, fre][px])
                frcomp[px,ox,cx,sx,gx,4] = pvalue(OneSampleTTest([   (fre - frp)./mean(frp), (frm - fre)./mean(fre)   ][px]))
            end
        end
    end



    # go nogo              ×        mean +/- sem
    for px in 1:2, ox in 1:2
        mfrr = [ mean( frcomp[px,ox,1,1,:,[1,3,4]], dims=1)[1,:] ;; mean( frcomp[px,ox,2,2,:,[1,3,4]], dims=1)[1,:] ]
        mfri = [ mean( frcomp[px,ox,1,2,:,[1,3,4]], dims=1)[1,:] ;; mean( frcomp[px,ox,2,1,:,[1,3,4]], dims=1)[1,:] ]

        # relevant and irrelevant:
        nr = [ sum(ns[:,px,ox,1,1,:],dims=1)[1,:];; sum(ns[:,px,ox,2,2,:],dims=1)[1,:] ]        # relevant, (modality by gonogo)
        ni = [ sum(ns[:,px,ox,1,2,:],dims=1)[1,:];; sum(ns[:,px,ox,2,1,:],dims=1)[1,:] ]        # irrelevant
        @info "fr compare $px-$ox" nr mfrr' ni mfri'
    end


    # find cells that have simultaneously larger response in relevant than in irrelevant context
    collectedcells = zeros(2,2)      # stimulus, go/nogo
    contextlabels = ["visual","audio"]
    for mx in 1:nmice


        # collect cells that are larger in relevant
        # first collect > cell mask for relevant, then for irrelevant context
        frc = zeros(3,2,2,2,nneurons[mx],4)         # rx, stim, go/nogo, stats+pvalue
        for (sx,(col,vals)) in enumerate(zip( (:degree,:freq),((45,135),(5000,10000)) )),
                (gx,val) in enumerate(vals)

                
                for cx in (sx,3-sx) # relevant, irrelevant
                    rx = 2 - (cx==sx)
                    mask = (triallists[mx][!,:context].==contextlabels[cx]) .& (triallists[mx][!,col].==val)
                
                    # average over spec. timepoints -> (trials × neurons)
                    p = mean(ifrs[mx][mask,prestim,:],dims=2)[:,1,:]
                    e = mean(ifrs[mx][mask,earlystim,:],dims=2)[:,1,:]
                    m = mean(ifrs[mx][mask,midstim,:],dims=2)[:,1,:]

                    frc[:,rx,sx,gx,:,1] = @asarray [ mean(a,dims=1)[1,:] for a in (p,e,m) ]
                    frc[:,rx,sx,gx,:,2] = @asarray [ std(a,dims=1)[1,:] for a in (p,e,m) ]
                    frc[:,rx,sx,gx,:,3] = @asarray [ std(a,dims=1)[1,:]/sqrt(size(a,1)) for a in (p,e,m) ]
        
                end
        end
        
        # find cells that are consistent in the direction for relevance-irrelevance at the late stimulus period with 2 sem

        tp = 3 # time period    pre early mid
        for sx in 1:2
            for gx in 1:2
                # mask = frc[tp,1,sx,gx,:,1] - frc[tp,1,sx,gx,:,3] *2 .> frc[tp,2,sx,gx,:,1] + frc[tp,2,sx,gx,:,3] *2
                mask = frc[tp,1,sx,gx,:,1] + frc[tp,1,sx,gx,:,3] *2 .< frc[tp,2,sx,gx,:,1] - frc[tp,2,sx,gx,:,3] *2
                collectedcells[sx,gx] += sum(mask)
            end
        end

    end
    @info "rel>irrel" collectedcells
    
end


