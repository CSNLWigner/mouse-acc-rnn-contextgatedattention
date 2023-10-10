


function parsesubjectids()
    if isempty(commandlineoptions[:subjectids])
        mouseids = Symbol.(keys(config[:sessions]))
    else
        mouseids = @. Symbol(getindex(commandlineoptions[:subjectids],1))
    end
end


function collectsubjectids(brainarea)
    mouseids = Symbol[]
    for (mouseid,session) in config[:sessions]
        if session[:brainarea]==brainarea
            push!(mouseids,mouseid)
        end
    end
    sort!(mouseids)
    return mouseids
end









function loadnwbdata(mouseid::String)
    @info "loadnwbdata" nwb  fn=joinpath(config[:rawdatapathprefix], config[:nwbdatapath], mouseid * ".nwb")
    io = nwb.NWBHDF5IO(joinpath(config[:rawdatapathprefix], config[:nwbdatapath], mouseid * ".nwb"), mode="r")
    nwbfile = io.read()
    return nwbfile,io
end
loadnwbdata(mouseid::Symbol) = loadnwbdata(string(mouseid))









function generatebehaviourfromsession(session::Dict{Symbol,Any},nwbfile::PyObject)
    # extend mouse behaviour data by generating random repeated samples from the contexts
    # preserves context order, and context switch will be at half point of training sets
    

    data = nwbdf(nwbfile.trials)
    filter!(:difficulty=>u->u=="complex",data)

    data = data[!,[:degree, :freq, :context, :water]]
    ndata = nrow(data)

    contextsplit = findfirst(data[!,:context].!=data[1,:context] )-1
    splitproportion = 0.6

    trainend = ( Int(round(contextsplit * splitproportion)), Int(round((ndata-contextsplit) * splitproportion)) )

    trainidices = (collect(1:trainend[1]), collect(contextsplit+1:contextsplit+trainend[2]))
    testindices = (collect(trainend[1]+1:contextsplit), collect(contextsplit+1+trainend[2]:ndata))

    # @info "boundaries" contextsplit trainend
    # @info "indices" trainidices testindices

    
    Xtr = vcat( [ onehotbatch(data[vcat(trainidices...),column],unique(data[vcat(trainidices...),column])) for column in ["degree","freq","water","context"] ]... )
    ytr = Matrix{Int}(onehotbatch(data[vcat(trainidices...),:water],unique(data[vcat(trainidices...),:water])))
    Xte = vcat( [ onehotbatch(data[vcat(testindices...),column],unique(data[vcat(testindices...),column])) for column in ["degree","freq","water","context"] ]... )
    yte = Matrix{Int}(onehotbatch(data[vcat(testindices...),:water],unique(data[vcat(testindices...),:water])))
    # ytr = @horizontalvector Int.(data[!,:water])                       # probabilistic, possible 0,1 goal values





    X1 = Xtr[:,1:trainend[1]]
    y1 = ytr[:,1:trainend[1]]
    X2 = Xtr[:,trainend[1]+1:end]
    y2 = ytr[:,trainend[1]+1:end]
    Xte1 = Xte[:,1:length(testindices[1])]
    yte1 = yte[:,1:length(testindices[1])]
    Xte2 = Xte[:,length(testindices[1])+1:end]
    yte2 = yte[:,length(testindices[1])+1:end]

    # @info "sizes 1-2" map(size, [X1,y1,X2,y2,Xte1,yte1,Xte2,yte2])


    # the final number of trials will be N*L*2 (for the two contexts)
    # N number of sampling batches
    # L number of trials in a sampling batch
    N = 6
    L = min(20,size(Xte1,2),size(Xte2,2))

    Xtr = fill(Float32[],size(Xtr,1),0)
    ytr = fill(Float32[],size(ytr,1),0)
    Xte = fill(Float32[],size(Xte,1),0)
    yte = fill(Float32[],size(yte,1),0)
    
    # first context
    for i in 1:N
        ind = rand(1:size(X1,2),L)
        Xtr = hcat(Xtr, X1[:,ind])
        ytr = hcat(ytr, y1[:,ind])
        ind = rand(1:size(Xte1,2),L)
        Xte = hcat(Xte, Xte1[:,ind])
        yte = hcat(yte, yte1[:,ind])
    end
    # second context
    for i in 1:N
        ind = rand(1:size(X2,2),L)
        Xtr = hcat(Xtr, X2[:,ind])
        ytr = hcat(ytr, y2[:,ind])
        ind = rand(1:size(Xte2,2),L)
        Xte = hcat(Xte, Xte2[:,ind])
        yte = hcat(yte, yte2[:,ind])
    end

    Xtr = Float32.(Xtr)
    ytr = Float32.(ytr)
    Xte = Float32.(Xte)
    yte = Float32.(yte)

    # @info "Xtr ytr Xte yte" Xtr ytr Xte yte

    return Xtr, ytr, Xte, yte
end






function getparts(N::Int64;addcontext::Bool=false)
    if addcontext d = 8 else d = 6 end      # context switch point at half point
    X = zeros(Float32, d,N)
    X[[1,3],:] = rand([0,1],2,N)          # degree and freq
    X[[2,4],:] = 1 .- X[[1,3],:]          # onehot to [1:2,:] -> degree,  [3:4,:] -> freq
    X[5:6,1:N÷2] = X[1:2,1:N÷2]           # degree, visual context
    X[5:6,N÷2+1:end] = X[3:4,N÷2+1:end]   # freq, audio context
    if addcontext
        X[7,1:N÷2] = ones(N÷2)                   # record context
        X[8,N÷2+1:end] = ones(N÷2)
    end

    y = copy(X[5:6,:])

    # @info "X" X y
    return X,y
end


function generatecombinatorialsequences(S::Int;incongruentonly::Bool=false,addcontext::Bool=false,onehot::Bool=false)
    # this generates trials so that each possible combination has the same number of occurrences
    # N is the number of presequences, while context switch point at half point of the list of sequences
    # there are 8 possible combinations of trials, 4 for one context and 4 for the other context

    if onehot
        stimuli = [[1,0],[0,1]]
        stimulusindices = [1:2,3:4]
    else
        stimuli = [1,-1]
        stimulusindices = [1:1,2:2]
    end

    if incongruentonly
        # variationsonehot = [[1,0,0,1],[0,1,1,0]]  # one hot, incongruent only visual and audio in the first 4 slots
        variationsonehot = [ [stimuli[1];stimuli[2]], [stimuli[2];stimuli[1]] ]  # incongruent only visual and audio in the first 2/4 slots
    else
        # variationsonehot = [[1,0,1,0],[1,0,0,1],[0,1,1,0],[0,1,0,1]]  # one hot visual and audio in the first 4 slots
        variationsonehot = [ [stimuli[1];stimuli[1]], [stimuli[1];stimuli[2]],
                             [stimuli[2];stimuli[1]], [stimuli[2];stimuli[2]] ]  # visual and audio in the first 2/4 slots
    end
    nvalues = length(variationsonehot)       # simplified to the four combinatinos of 2 by 2 stimuli matrix
    tlist = collect(1:nvalues)'
    
    sequences = zeros(Int,nvalues^S,S)
    # create a sequence, where each timepoint can have all combinations
    for s in 1:S
        # @info "v=$nvalues, S=$S, s=$s" nvalues^(S-s) nvalues^(s-1)
        A = repeat(tlist,nvalues^(S-s),nvalues^(s-1))
        A = reshape(A,nvalues^S,1)
        sequences[:,s] = copy(A)
    end
    
    Xs = ([ reduce(hcat,variationsonehot[sequences[:,s]]) for s in 1:S  ])   # indexed by the above simplified tlist



    # add choice and if required, context
    for s in 1:S
        Xs[s] = [ Xs[s] Xs[s];
                  Xs[s][stimulusindices[1],:] Xs[s][stimulusindices[2],:] ]
        if addcontext
            if onehot 
                contextdata = [ [ones(1,nvalues^S); zeros(1,nvalues^S)] [zeros(1,nvalues^S); ones(1,nvalues^S)] ]
            else
                contextdata = [ ones(1,nvalues^S) -ones(1,nvalues^S) ]
            end
            Xs[s] = [ Xs[s]; contextdata ]
        end
    end

    return Xs
end


function generatetimecoursebehaviour(;S::Int=4, Ts::Vector{UnitRange{Int}}=[1:1,2:2,3:3,4:4],
                                      incongruentonly::Bool=false, addcontext::Bool=false, onehot::Bool=true)
    # S is number of sequencees
    # generate trial points
    Xp = generatecombinatorialsequences(S,incongruentonly=incongruentonly,addcontext=addcontext,onehot=onehot)
    N = size(Xp[1],2) # number of datapoints
    if onehot
        stimulusindices = 1:4
        decisionindices = 5:6
        contextindices = 7:8
    else
        stimulusindices = 1:2
        decisionindices = 3:3
        contextindices = 4:4
    end
    # create timecourse within trial
    # N number of trials, Ts timecourse timepoints in unitranges 1) pre 2) during stim 3) during response+stim and 4) after stim
    # Ts = [1:5,6:10,11:15,16:20]
    # Ts = [1:15,16:35,36:45,46:60]
    # Ts = [1:150,160:350,360:450,460:600]
    T = Ts[end][end]      # find the last element of the unitranges, as number of 
    # get onehot 2D encoded trials, where features concatenated: visual audio decision (context)

    X = zeros(Float32,T*S,size(Xp[1])...)
    # ys = zeros(Float32,T,size(y)...)
    # @info "sizes" size(Xs)
    for s in 1:S
        for n in 1:N
            # stimuli are 0-1-0 for the corresponding one-hot channel:
            # off stimulus is zero, it is already assigned at array creation
            # on stimulus one hot pulse:
            X[[Ts[2]...,Ts[3]...].+(s-1)*T,stimulusindices,n] =
                ones(Float32,length(Ts[2])+length(Ts[3]),length(stimulusindices)) .* Xp[s][stimulusindices,n]'

            # choice is not used as input, it is a replacement for reward time delayed feedback
            # calculated here, and copied for ys
            # it is only available from 2 sec, simulated as the 3rd and 4th component of the timecourse
            X[[Ts[3]...,Ts[4]...].+(s-1)*T,decisionindices,n] =
                ones(Float32,length(Ts[3])+length(Ts[4]),length(decisionindices)) .* Xp[s][decisionindices,n]'

            # context is created here if necessary if direct context input is used
            if addcontext
                X[(s-1)*T+1:s*T,contextindices,n] = ones(Float32,T,length(contextindices)) .* Xp[s][contextindices,n]'
            end
        end
    end

    y = copy(X[:,decisionindices,:])       # copied from calculated correct decision above as learning target

    return X,y
end











function generatebehaviourrandom(;N=300,addcontext::Bool=false)
    # extend mouse behaviour data by generating random repeated samples from the contexts
    # preserves context order, and context switch will be at half point of training sets





    Xtr, ytr = getparts(N;addcontext)
    Xte, yte = getparts(N;addcontext)


    Xtr = Float32.(Xtr)
    ytr = Float32.(ytr)
    Xte = Float32.(Xte)
    yte = Float32.(yte)

    # @info "=?" Xtr==Xte

    # @info "Xtr ytr Xte yte" Xtr ytr Xte yte

    return Xtr, ytr, Xte, yte
end    








function generatetimecoursebehaviourrandom(;N::Int=300, Ts::Vector{UnitRange{Int}}=[1:1,2:2,3:3,4:4], addcontext::Bool=true)
    # N number of trials, Ts timecourse timepoints in unitranges 1) pre 2) during stim 3) during response+stim and 4) after stim
    # N = 20
    # Ts = [1:5,6:10,11:15,16:20]
    # Ts = [1:15,16:35,36:45,46:60]
    # Ts = [1:150,160:350,360:450,460:600]
    T = Ts[end][end]      # find the last element of the unitranges, as number of 
    # get onehot 2D encoded trials, where features concatenated: visual audio decision (context)
    X,y = getparts(N,addcontext=addcontext)

    # @info "X y" X y

    # now create timecourses for each trial in a new dimension in
    # flux RNN style: sequences as vector of (features,observations/samples)
    Xs = zeros(Float32,T,size(X)...)
    # ys = zeros(Float32,T,size(y)...)
    # @info "sizes" size(Xs)
    for i in 1:N
        # stimuli are 0-1-0 for the corresponding one-hot channel:
        # off stimulus is zero, it is already assigned at array creation
        # on stimulus one hot pulse:
        Xs[[Ts[2]...,Ts[3]...],1:4,i] = ones(Float32,length(Ts[2])+length(Ts[3]),4) .* X[1:4,i]'

        # choice is not used as input, it is a replacement for reward time delayed feedback
        # calculated here, and copied for ys
        # it is only available from 2 sec, simulated as the 3rd and 4th component of the timecourse
        Xs[[Ts[3]...,Ts[4]...],5:6,i] = ones(Float32,length(Ts[3])+length(Ts[4]),2) .* X[5:6,i]'

        # context is created here if necessary if direct context input is used
        if addcontext
            Xs[:,7:8,i] = ones(Float32,T,2) .* X[7:8,i]'
        end
    end
    ys = copy(Xs[:,5:6,:])      # copied from calculated correct decision above as learning target





    return Xs,ys
    
end


"""
return stimulus masks for context, visualgo, audiogo, go
x must be (stimulus,trial) dimensions
"""
function getmasks(x)
    ntrials = size(x,2)
    maskvisualgo = x[1,:].==1
    maskaudiogo = x[2,:].==1
    maskcontext = [trues(ntrials÷2); falses(ntrials÷2)]
    maskgo = [ maskvisualgo[maskcontext];  maskaudiogo[.!maskcontext] ]
    return maskcontext, maskvisualgo, maskaudiogo, maskgo
end


"""
gives a BitArray mask with size of the number of trials
X must be (stimulus,trial) dimensions
stimulus must have index -> value: 1 2 3 -> v+ a+ cv = 1, and v- a- cv = 0
yields an overall AND mask over (stimulus type, congruency, context)
"""
function getcongruency(x)
    measure = unique(x)
    congruencylist = BitArray(undef, 2,2,2,size(x,2))         # (gonogo, congruency, context, trials)
    for (sx,s) in enumerate(measure)   # go, nogo
        for (cx,c) in enumerate(measure)    # visual context, audio context
                congruencylist[sx,1,cx,:] = (x[1,:].==s) .& (x[2,:].==s)   .& (x[3,:].==c)   # congruent
                congruencylist[sx,2,cx,:] = (x[1,:].==s) .& (x[2,:].==measure[3-sx]) .& (x[3,:].==c)   # incongruent
        end
    end
    return congruencylist
end


"""
x must be (sequence,stimuli,trial) dimensions
congruent when the two stimuli are the same
returns (sequence,trial) array, where true if 
"""
function getcongruencyhistory(x)
    congruencyhistory = BitArray(undef, size(x,1), size(x,3))         # (sequence,trials)
    # compare first and second index of the 2nd dimension (stimuli) at each sequence
    congruencyhistory = (x[:,1,:].==x[:,2,:])   # congruent = true
end


"""
X must be (stimulus,trial) dimensions
stimulus must have index -> value: 1 2 -> v a
returns a boolean bitarray of congruency by trials
"""
function getcongruency1d(x)
    congruencylist = BitArray(undef, size(x,2))         # (congruency, trials)
    congruencylist = (x[1,:].==x[2,:])   # congruent = true
    return congruencylist
end



"""
add congruency column to the trials=events
dataframe after context column
"""
function addcongruencycolumn!(df::DataFrame; cols=[:degree,:freq], insertion=:context, levels=[45 135; 5000 10000])
    congruent = Int.(((df[!,cols[1]].==levels[1,1]) .== (df[!,cols[2]].==levels[2,1])).+1)
    insertcols!(df, insertion, :congruency => ["incongruent","congruent"][congruent] )
    return df
end






"""
separately collect trials for each of
    - context
    - relevant modality go nogo
    - congruency
    returns a 4D boolean array of size (trials,context,gonogo,congruency)
"""
function masktrialtypes(trials::DataFrame; stimcols=[:degree,:freq], contextvalues=[:visual,:audio])
    trialtypesmasks = zeros(Bool, size(trials,1),2,2,2)        # (trials,context,gonogo,congruency)
    for (cx,c) in enumerate(contextvalues)
        for (sx,s) in enumerate(unique(trials[!,stimcols[cx]]))
            for (gx,g) in enumerate(unique(trials[!,:congruency]))
                trialtypesmasks[:,cx,sx,gx] = (trials[!,:context].==c) .& (trials[!,stimcols[cx]].==s) .& (trials[!,:congruency].==g)
            end
        end
    end
    return trialtypesmasks
end



"""
calculate moving average performance for each trial type
    returns a 4D array of size (trials,context,gonogo,congruency)
"""
function movingaverageperformancetrialtypes(trials::DataFrame; ma=20)
    borderpoints = [ f(trials[!,:context].==context) for context in unique(trials[!,:context]), f in (findfirst,findlast) ]
    trialparts = zeros(nrow(trials),2,2)        # (trials,gonogo,congruency)
    for cx in axes(borderpoints,1)
        for (gx,g) in enumerate(["congruent","incongruent"]), (sx,s) in enumerate([true,false])
            for t in range(borderpoints[cx,1],borderpoints[cx,2]) # oving average contexts separately
                indices = range( max(borderpoints[cx,1],t-ma÷2), min(t+ma÷2,borderpoints[cx,2]) )
                maskgonogo = (trials[indices,:water] .== s) .& (trials[indices,:congruency] .== g)
                trialparts[t,sx,gx] = mean(trials[indices,:success][maskgonogo])
            end
        end
    end
    return trialparts
end

"""
high performance trials are defined as all congruency and action
trials moving averages should have above chance pereformance
"""
function highperformancetrialsmask(maperfs; threshold=0.5)
    dropdims(all(reshape(maperfs, :, 4) .> threshold, dims=2),dims=2)
end


"""
the input _trials_ should contain congruency column
returns a list of 4 arrays
    - congruent go
    - congruent nogo
    - incongruent go
    - incongruent nogo
each array contains a vector of
    - indices of trials
    - success of trials
"""
function choicesselected(trials::DataFrame)
    congruentmask = trials[!, :congruency].=="congruent"
    choices = []
    for stimulusmask in [true,false]   # go nogo (water or not)
        for congruencymask in [congruentmask, .! congruentmask]  # congruent incongruent
            mask = congruencymask .& (trials[!,:water].==stimulusmask)
            push!(choices, [findall(mask), trials[mask,:success]])
        end
    end
    return choices
end




"""
model choices of the anmimal and a β bias
parameter, for first and second target stimulus
"""
function modelbias(target,β=0.,r=[false,true])
    p = zeros(size(target))
    p[target.==r[1]] .= 1 + β
    p[target.==r[2]] .= β
    return p
end


"""
model choices of the anmimal and a λ lapse
parameter, for first and second target stimulus
"""
function modellapse(target,λ=0.,r=[false,true])
    p = zeros(size(target))
    p[target.==r[1]] .= 1 - λ
    p[target.==r[2]] .= λ
    return p
end
