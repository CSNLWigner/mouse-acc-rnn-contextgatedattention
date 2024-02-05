__precompile__()
module ElectroPhysiology

export gettrialrelativespiketimes
export countspikes,smoothinstantenousfiringrate
export PSTH

export generatepoissonpointprocess


using Statistics: mean, std
using Distributions
using LinearAlgebra

using ..NWBWrap
using ..MathUtils












function gettrialrelativespiketimes(trialstable, spiketimestable; preeventpadding=-1.5, posteventpadding=1.5)
    # Returns a list of trials which are between to the trialstable start_time and stop_time,
    # relative to the trial start time, and padding with pre and post trial start spike times
    n_units = length(spiketimestable["id"])
    neuronsspiketimeslist = Vector{Vector{Float64}}[]
    for trial in eachrow(trialstable)
        trialunitspiketimes = spiketimestable.get_unit_spike_times(index=Tuple(collect(1:n_units).-1),
                in_interval=(trial.start_time+preeventpadding, trial.stop_time+posteventpadding))
        trialunitspiketimes = map( a -> a .- trial.start_time, trialunitspiketimes)
        # @info "trial" trial trialunitspiketimes
        push!(neuronsspiketimeslist, trialunitspiketimes)
    end
    return neuronsspiketimeslist
end



"""
count number of spikes in bins in a
vector (neurons) of vector of spike times
"""
function countspikes(neuronsspiketimes::Vector{Vector{Float64}};
                       bin=0.01, starttime=0., stoptime=Inf)

    starttime = starttime==0 ? minimum(minimum.(neuronsspiketimes)) : starttime
    stoptime = stoptime==Inf ? maximum(maximum.(neuronsspiketimes)) : stoptime
    starttime = floor(starttime / bin) * bin
    stoptime = ceil(stoptime / bin) * bin

    timestamps = starttime:bin:stoptime
    
    activity = zeros(length(timestamps),length(neuronsspiketimes))
    for (tx,t) in enumerate(timestamps)
        activity[tx,:] = hcat([ map(nst->sum( (nst.>=t) .& (nst.<t+bin) ), neuronsspiketimes)]...)' ./ bin    # yields timebins × neurons matrix
    end
    return collect(timestamps),activity
end




"""
returns a trials × timepoints × neurons tensor of smoothed firing rates
"""
function smoothinstantenousfiringrate(neuronsspiketimeslist::Vector{Vector{Vector{Float64}}}, eventlength::Float64;
                       kernelwidth=0.1, binsize=0.01, preeventpadding=-1.5, posteventpadding=1.5)
    # smoothing gaussian kernel applied after spike counting in timebins with pre and post event padding
    # returns the timestamps vector and a 3D 
    ntrials = length(neuronsspiketimeslist)
    nneurons = length(neuronsspiketimeslist[1])
    timestamps = preeventpadding:binsize:eventlength+posteventpadding-binsize         # iterator
    activity = zeros(ntrials,length(timestamps),nneurons)
    
    Threads.@threads for tx in eachindex(timestamps)
        t = timestamps[tx]
        activity[:,tx,:] = hcat([  map( neuronspiketimes->sum((neuronspiketimes.>t) .& (neuronspiketimes.<t+binsize) ),  trial)
                 for trial in neuronsspiketimeslist ]...)' ./ binsize           # yields trials x neurons matrix
    end
    Threads.@threads for trx in axes(activity,1)
        for nx in axes(activity,3)
            # smooth!(activity[trx,:,nx],binsize,kernelwidth=kernelwidth)
            activity[trx,:,nx] = smooth(activity[trx,:,nx],binsize,kernelwidth=kernelwidth)
        end
    end
    return collect(timestamps),activity
end
















function PSTH(neuronsspiketimeslist::Vector{Vector{Vector{Float64}}}, eventlength; binsize=0.01, preeventpadding=-1.5, posteventpadding=1.5)
    # get precise peristimulus time histogram from trial lists
    # each trial consists of T=start:binsize:end timecourse N neurons 
    n_neurons = length(neuronsspiketimeslist[1])
    timestamps = preeventpadding:binsize:eventlength+posteventpadding-binsize         # iterator
    m = zeros(length(timestamps),n_neurons)   # mean in Hz, with spiketimes in seconds
    e = zeros(length(timestamps),n_neurons)   # s.e.m. in Hz, with spiketimes in seconds
    for (tx,t) in enumerate(timestamps)
        aux = hcat([  map( neuronspiketimes->sum((neuronspiketimes.>t) .& (neuronspiketimes.<t+binsize) ),  trial)
                       for trial in neuronsspiketimeslist ]...)'
        m[tx,:] = mean(aux,dims=1)'
        e[tx,:] = std(aux,dims=1)'/sqrt(length(neuronsspiketimeslist))
    end
    M = TimeSeries(timestamps.+binsize/2,m./binsize)
    E = TimeSeries(timestamps.+binsize/2,e./binsize)
    return M,E
end
function PSTH(neuronsspiketimeslist::Vector{Vector{Vector{Float64}}}, eventlength, binsize, preeventpadding, posteventpadding)
    PSTH(neuronsspiketimeslist::Vector{Vector{Vector{Float64}}}, eventlength, binsize=binsize,
              preeventpadding=preeventpadding, posteventpadding=posteventpadding)
end















end
