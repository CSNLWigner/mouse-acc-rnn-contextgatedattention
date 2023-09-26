# physiology functions



function perieventtimehistogram(nwbfile)

    triallist = nwbdf(nwbfile.trials)
    filter!(:difficulty=>u->u=="complex",triallist)

    conditionlist = [ [:context,:degree,:freq]=>(c,v,a)->(c,v,a)==(context,visual,audio)
                      for context in ["visual","audio"] for visual in [45,135] for audio in [5000,10000] ]
    labels = String[  "$(context) " * ([visual,audio][(context=="audio")+1]) * " " * ("congruent","incongruent")[(visual!=audio)+1]
                      for context in ["visual","audio"] for visual in ["go","nogo"] for audio in ["go","nogo"]    ]

    colors = [:navy,:purple,:dodgerblue,:cyan,:darkgreen,:fuchsia,:green,:seagreen]

    conditionlist[6:7] = conditionlist[7:-1:6]
    labels[6:7] = labels[7:-1:6]
    

    selectedtriallist = []
    selectedtrials_neuronsspiketimeslist = Vector{Vector{Vector{Float64}}}[]    #(condition,trials,neurons,spiketimes)
    for condition in conditionlist
        push!(selectedtriallist, filter(condition, triallist))
        push!(selectedtrials_neuronsspiketimeslist, gettrialrelativespiketimes(selectedtriallist[end], nwbfile.units))
    end


    ax = plot(layout=(8,4), size=(1.2* 4*200,8*200))

    for ix in 1:length(conditionlist)
        frms, fres = PSTH(selectedtrials_neuronsspiketimeslist[ix], stimuluslength)
        for nx in 1:4
            axs = ax[ix,nx]
            frm = frms.data[:,nx]
            fre = fres.data[:,nx]
            plot!(axs, frms.timestamps, frm, ribbon=(frm-fre, frm+fre), fillalpha=0.2, color=colors[ix])
            ylabel!(axs, labels[ix])
        end

    end

    display(ax)
end























function simulatemlppp(spikes, rates, ts)

    Λ = estimatepoissonpointprocess(spikes,ts)


    axs = plot(layout=(3,1), size=(1*1000, 3*200), legend=false)
    
    ax = axs[1]
    plot!(ax, ts, Λ')
    
    ax = axs[2]
    plot!(ax, ts, rates')
    
    ax = axs[3]
    for (n,s) in enumerate(spikes)
        for x in s              #fill(n,length(s))
            plot!(ax, [x,x], [n.-0.2,n.+0.2], markersize=1, markerstrokewidth=0, color=:white)
        end
    end
    xlims!(ax,ts[1],ts[end])
    display(axs)

end
