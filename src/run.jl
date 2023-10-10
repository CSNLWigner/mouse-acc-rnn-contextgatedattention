__precompile__()

module Run




using Plots
using StatsPlots: violin

import Statistics.mean
using Random: randperm, shuffle
using Distributions
using StatsBase
using MultivariateStats
using HypothesisTests: OneSampleTTest, pvalue
using Flux: onehot, onehotbatch, onecold



using ArgParse
using YAML
using CSV
using DataFrames
import Dates.DateTime
using Images, FileIO



using LinearAlgebra


include("NeuroscienceCommon.jl")
using .NeuroscienceCommon.NWBWrap
NWBWrap.__init__()
using .NeuroscienceCommon.MathUtils
using .NeuroscienceCommon.Figs
using .NeuroscienceCommon.ElectroPhysiology
using .NeuroscienceCommon.Discover
using .NeuroscienceCommon.AI



include("options.jl")

include("preprocess.jl")
include("physiology.jl")
include("subspaces.jl")
include("nnmodels.jl")
include("figures.jl")














function main()



    if commandlineoptions[:_COMMAND_]==:list
        mouseids = parsesubjectids()
        for mouseid in mouseids
            nwbfile,_ = loadnwbdata(mouseid)
            if commandlineoptions[:list][:_COMMAND_]==:trials
                # filter(r-> ((r.stop_time>=config[:start_stop][1]) && (r.start_time<=config[:start_stop][2])), nwbdf(nwbfile.trials))
                @info "trials" mouseid nwbdf(nwbfile.trials)
            elseif commandlineoptions[:list][:_COMMAND_]==:units
                @info "trials" mouseid nunits=length(nwbfile.units["id"])
            elseif commandlineoptions[:list][:_COMMAND_]==:psth
                perieventtimehistogram(nwbfile)
            end
        end



    elseif commandlineoptions[:_COMMAND_]==:chancelevelshuffle
        mouseids = parsesubjectids()
        for mouseid in mouseids
            nwbfile,_ = loadnwbdata(mouseid)
            chancelevelshuffle(mouseid,nwbfile)
        end

    elseif commandlineoptions[:_COMMAND_]==:decodevariables
        mouseids = parsesubjectids()
        for mouseid in mouseids
            nwbfile,_ = loadnwbdata(mouseid)
            decodevariables(mouseid,nwbfile)
        end

    elseif commandlineoptions[:_COMMAND_]==:lowdimortho
        mouseids = parsesubjectids()
        for mouseid in mouseids
            if config[:sessions][mouseid][:brainarea]=="ACC"
                nwbfile,_ = loadnwbdata(mouseid)
                lowdimortho(mouseid,nwbfile)
            end
        end

    elseif commandlineoptions[:_COMMAND_]==:cognitivesurplus
        mouseids = parsesubjectids()
        for mouseid in mouseids
            nwbfile,_ = loadnwbdata(mouseid)
            cognitivesurplus(mouseid,nwbfile)
        end

    elseif commandlineoptions[:_COMMAND_]==:mutualswitchsubspaces
        mouseids = parsesubjectids()
        mutualswitchsubspaces(mouseids)
        

    elseif commandlineoptions[:_COMMAND_]==:decoderelevancy
        mouseids = parsesubjectids()
        decoderelevancy(mouseids)

    elseif commandlineoptions[:_COMMAND_]==:suppressionbehaviour
        mouseids = parsesubjectids()
        for mouseid in mouseids
            suppressionbehaviour(mouseid)
        end

    elseif commandlineoptions[:_COMMAND_]==:projectcontext
        mouseids = parsesubjectids()
        for mouseid in mouseids
            nwbfile,io = loadnwbdata(mouseid)
            projectactivitycontext(config[:sessions][mouseid], nwbfile)
        end

    elseif commandlineoptions[:_COMMAND_]==:outcomehistory
        mouseids = parsesubjectids()
        outcomehistory(mouseids)
    

    elseif commandlineoptions[:_COMMAND_]==:conditionaveragepca
        mouseids = parsesubjectids()
        for mouseid in mouseids
            nwbfile,io = loadnwbdata(mouseid)
            conditionaveragepca(config[:sessions][mouseid],nwbfile)
        end




    elseif commandlineoptions[:_COMMAND_]==:rnn
        machineparameters = YAML.load_file("params-rnn.yaml"; dicttype=Dict{Symbol,Any})
        # Ts = [1:1,2:3,4:4,5:5]
        Ts = [1:3,4:9,10:12,13:15]
        # Ts = [1:10,11:30,31:40,41:50]
        data = generatetimecoursebehaviour(S=machineparameters[:nsequence], Ts=Ts; incongruentonly=false, addcontext=false, onehot=true)
        # :hyperanneal :contextsymmetric :regularization :sparse/:energy :nonnegative/:nonnegativerelu
        @info "modelids" machineparameters[:modelids] machineparameters[:snapshots] machineparameters[:regularization]
        if (   ("nonnegativerelu" in machineparameters[:regularization]) & (!occursin("relu",machineparameters[:modeltype]))) | 
           ( (!("nonnegativerelu" in machineparameters[:regularization])) &  occursin("relu",machineparameters[:modeltype]))
           error("reguralization and filename inconsistent")
        end
        fillrange!(machineparameters) # test if we want a range of models, between first and last each entry is filled
        for id in machineparameters[:modelids]
            machineparameters[:modelid] = id
            @info "id" id
            modelrnn(data..., Ts, machineparameters)
        end

    
    elseif commandlineoptions[:_COMMAND_]==:rnndecode
        machineparameters = YAML.load_file("params-rnn.yaml"; dicttype=Dict{Symbol,Any})
        Ts = [1:3,4:9,10:12,13:15]
        data = generatetimecoursebehaviour(S=machineparameters[:nsequence], Ts=Ts; incongruentonly=false, addcontext=false, onehot=true)
        @info "decoding from models"
        fillrange!(machineparameters)
        for id in machineparameters[:modelids]
            machineparameters[:modelid] = id
            @info "id" id
            decodernn(data..., Ts, machineparameters)
        end

    elseif commandlineoptions[:_COMMAND_]==:rnncontextinference
        machineparameters = YAML.load_file("params-rnn.yaml"; dicttype=Dict{Symbol,Any})
        Ts = [1:3,4:9,10:12,13:15]
        data = generatetimecoursebehaviour(S=machineparameters[:nsequence], Ts=Ts; incongruentonly=false, addcontext=false, onehot=true)
        @info "investigating context inference by reward"
        contextinferencernn(data..., Ts, machineparameters)

    
    elseif commandlineoptions[:_COMMAND_]==:rnntracesuppression
        machineparameters = YAML.load_file("params-rnn.yaml"; dicttype=Dict{Symbol,Any})
        Ts = [1:3,4:9,10:12,13:15]
        data = generatetimecoursebehaviour(S=machineparameters[:nsequence], Ts=Ts; incongruentonly=false, addcontext=false, onehot=true)
        tracesuppression(data..., Ts, machineparameters)

    elseif commandlineoptions[:_COMMAND_]==:rnnsubspaces
        machineparameters = YAML.load_file("params-rnn.yaml"; dicttype=Dict{Symbol,Any})
        Ts = [1:3,4:9,10:12,13:15]
        data = generatetimecoursebehaviour(S=machineparameters[:nsequence], Ts=Ts; incongruentonly=false, addcontext=false, onehot=true)
        subspacesrnn(data..., Ts, machineparameters)

    elseif commandlineoptions[:_COMMAND_]==:rnnoutcomehistory
        machineparameters = YAML.load_file("params-rnn.yaml"; dicttype=Dict{Symbol,Any})
        Ts = [1:3,4:9,10:12,13:15]
        data = generatetimecoursebehaviour(S=machineparameters[:nsequence], Ts=Ts; incongruentonly=false, addcontext=false, onehot=true)
        outcomehistoryrnn(data..., Ts, machineparameters)



    
    
    
    elseif commandlineoptions[:_COMMAND_]==:figure
        theme(:default)
        @info "options" commandlineoptions[:figure]
        if commandlineoptions[:figure][:_COMMAND_]==Symbol("1")
            figure1()
        elseif commandlineoptions[:figure][:_COMMAND_]==Symbol("2")
            figure2()
        elseif commandlineoptions[:figure][:_COMMAND_]==Symbol("3")
            figure3()
        elseif commandlineoptions[:figure][:_COMMAND_]==Symbol("4")
            figure4()
        elseif commandlineoptions[:figure][:_COMMAND_]==Symbol("5")
            figure5()
        end
        theme(:darkscience)
    end



    

    


end




main()





end