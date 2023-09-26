# command line

isinteractive() ? args = Main.args : args = ARGS
argparsesettings = ArgParseSettings()

@add_arg_table! argparsesettings begin
    # "--config", "-c"
    #     help = "configuration filename"
    #     arg_type = String
    #     default = "params.yml"
    "--subjectids", "-s"
        help = "list experimental subject id overriding config file list
                -s mouseid1 -s mouseid2 ..."
        nargs = 1
        action = :append_arg
        arg_type = String
    # "--startstop", "-s"
    #     help = "start and and in seconds of the slice of the recording session: -s 100. 130. "
    #     nargs = 2
    #     arg_type = Float64
    #     default = [0., Inf]
    "list"
        help = "query nwbfile, not staged"
        action = :command
    "nwb"
        help = "create and export nwb format from csv folder data"
        action = :command
    "chancelevelshuffle"
        help = "chance level by shuffling trials"
        action = :command
    "decodevariables"
        help = "decode stimulus, context and decision"
        action = :command
    "lowdimortho"
        help = "show low dimensional orthogonal representation of stimuli"
        action = :command
    "cognitivesurplus"
        help = "decode context in consistent and exploratory trials"
        action = :command
    "mutualswitchsubspaces"
        help = "load decoders and with stimulus index compare context subspaces"
        action = :command
    "decoderelevancy"
        help = "decode stimulus, context and decision"
        action = :command
    "suppressionbehaviour"
        help = "decode relevant or irrelevant stimulus from consistent and exploratory trials"
        action = :command
    "conditionaveragepca"
        help = "condition averaged PCA (Murray et al. 2017)"
        action = :command
    "projectcontext"
        help = "project to decoder DV"
        action = :command
    "outcomehistory"
        help = "decode stimulus, decision and reward of previous trials from current trial"
        action = :command
    "mlppp"
        help = "Demonstrating maximum likelihood poisson point process"
        action = :command
    "rnnsession"
        help = "model mouse behaviour data with RNNs"
        action = :command
    "rnn"
        help = "model generated behaviour data with RNNs"
        action = :command
    "rnndecode"
        help = "decode variables from RNN models at snapshots"
        action = :command
    "rnncontextinference"
        help = "context inference computations in RNN models"
        action = :command
    "rnnmir"
        help = "analyse multiple RNNs with mutual information ratio"
        action = :command
    "rnnortho"
        help = "analyse multiple RNNs with orthogonal representation"
        action = :command
    "rnntracesuppression"
        help = "analyse multiple RNNs via backtracing suppression for abstraction"
        action = :command
    "rnnsubspaces"
        help = "analyse multiple RNNs via identifying subspaces and activity flow"
        action = :command
    "rnnirrelsupp"
        help = "model generated behaviour data with simple model suppressing irrelevance"
        action = :command
    "rnnoutcomehistory"
        help = "decode stimulus, decision and reward of previous trials from current trial for models"
        action = :command
    "test"
        help = "fast test of an idea"
        action = :command
    "figure"
        help = "generate figures"
        action = :command
end




@add_arg_table! argparsesettings["list"] begin
    "trials"
        help = "list trial tart stop times and values"
        action = :command
    "units"
        help = "list single units"
        action = :command
    "psth"
        help = "plot peri event time histogram"
        action = :command
end


@add_arg_table! argparsesettings["figure"] begin
    "1"
        help = "figure 1"
        action = :command
    "2"
        help = "figure 2"
        action = :command
    "3"
        help = "figure 3"
        action = :command
    "4"
        help = "figure 4"
        action = :command
    "5"
        help = "figure 5"
        action = :command
end










commandlineoptions = parse_args(args, argparsesettings; as_symbols=true)
@info "command line options" commandlineoptions














# configuration file

config = YAML.load_file("params.yaml"; dicttype=Dict{Symbol,Any})
@info "config YAML" config







# config[:brainareas] = Dict("V1"=>["ME108","ME110","ME112","ME113",
#                                   "DT008","DT009","DT014","DT017","DT018","DT019","DT021","DT030","DT031","DT032",
#                                   "SZ010"],
#                            "ACC"=>["AC001","AC003","AC004","AC006","AC007"])
# config[:areas] = [ ba for dn in config[:datanames] for ba in keys(config[:brainareas]) if dn in config[:brainareas][ba]]

# @info "config YAML + brain areas" config
