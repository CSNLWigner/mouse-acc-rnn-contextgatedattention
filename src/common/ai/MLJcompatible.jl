

# MLJModelInterface.@mlj_model mutable struct YourModel <: MLJModelInterface.Deterministic
#     a::Float64 = 0.5::(_ > 0)
#     b::String  = "svd"::(_ in ("svd","qr"))
# end





# # if `b` is a builder, then `b(model, rng, shape...)` is called to make a
# # new chain, where `shape` is the return value of this method:
# function MLJFlux.shape(model::NeuralNetworkClassifier, X, y)
#     levels = MLJModelInterface.classes(y[1])
#     n_output = length(levels)
#     n_input = Tables.schema(X).names |> length
#     return (n_input, n_output)
# end

# # builds the end-to-end Flux chain needed, given the `model` and `shape`:
# MLJFlux.build(model::NeuralNetworkClassifier, rng, shape) =
#     Flux.Chain(build(model.builder, rng, shape...),
#                model.finaliser)

# # returns the model `fitresult` (see "Adding Models for General Use"
# # section of the MLJ manual) which must always have the form `(chain,
# # metadata)`, where `metadata` is anything extra neede by `predict` may
# # require:
# MLJFlux.fitresult(model::NeuralNetworkClassifier, chain, y) =
#     (chain, MLJModelInterface.classes(y[1]))

# function MLJModelInterface.predict(model::NeuralNetworkClassifier,
#                                    fitresult,
#                                    Xnew)
#     chain, levels = fitresult
#     X = reformat(Xnew)
#     probs = vcat([chain(tomat(X[:,i]))' for i in 1:size(X, 2)]...)
#     return MLJModelInterface.UnivariateFinite(levels, probs)
# end

# MLJModelInterface.metadata_model(NeuralNetworkClassifier,
#                                  input=Table(Continuous),
#                                  target=AbstractVector{<:Finite},
#                                  path="MLJFlux.NeuralNetworkClassifier",
#                                  descr="A neural network model for making "*
#                                  "probabilistic predictions of a "*
#                                  "`Multiclass` or `OrderedFactor` target, "*
#                                  "given a table of `Continuous` features. ")


























# function makesequence(X::DataFrame, sequencelength::Int)
#     Xstr = DataFrame[]
#     for sx in 1:nrow(X)-sequencelength
#         push!(Xstr, X[sx:sx+sequencelength,:])
#         # push!(Xstr, [ X[sqx,:] for sqx = sx:sx+sequencelength ] )
#     end
#     return Xstr
# end



# mutable struct HierarchicalRNNBuilder <: MLJFlux.Builder
# 	n :: Int
# end


# function MLJFlux.build(rnn::HierarchicalRNNBuilder, rng, n_in, n_out)
#     return Chain(   RNN(n_in, rnn.n),  Dense(rnn.n, n_out)   )
# end



# function hierarchicalrnnmljflux(data::DataFrame,ylabel::Symbol,sequencelength::Int)

    
#     data = coerce( data, (Symbol.(names(data)) .=> Multiclass)... )
#     X, y = unpack( data, !=(ylabel), ==(ylabel) )
#     X = (          coerce( X, (Symbol.(names(X)) .=> Continuous)... )      .- 1.5 ) .* 2 

#     # create rnn sequences
#     # switchpoint = findfirst(data[!,:context] .!= data[1,:context])
#     Xstr = makesequence(X, sequencelength)


#     @info "nnc type" type(NeuralNetworkClassifier) NeuralNetworkClassifier


#     function rnnloss(xs, ys)
#         Flux.reset!(rnnmodel)
#         return sum(Flux.logitcrossentropy.([rnnmodel(x) for x in xs], ys))
#     end



#     NeuralNetworkClassifier = @load NeuralNetworkClassifier verbosity=0
#     rnnmodel = NeuralNetworkClassifier(builder=HierarchicalRNNBuilder(24),
#                        epochs=10, loss=rnnloss)

#     # pipe = @pipeline( ContinuousEncoder, rnnmodel, name=RNNPipe )


#     rnn = machine(rnnmodel, Xstr, y)

#     results = evaluate!( rnn, verbosity=1, #acceleration=CPUThreads(),
#                     # resampling=Holdout(rng=123, fraction_train=0.7),
#                     operation=predict_mode,
#                     measure=Accuracy()  )


#     @info "rnn" results

# end
