# struct FbRNNCell{F,A,V,S,B}
#     σ::F
#     Wi::A
#     Wh::A
#     b::V
#     state0::S
#     f::B
#   end
  
#   FbRNNCell(in::Integer, out::Integer, σ=tanh; init=Flux.glorot_uniform, initb=zeros32, init_state=zeros32, init_function=x->x) = 
#     FbRNNCell(σ, init(out, in), init(out, out), initb(out), init_state(out,1), init_function )
  
#   function (m::FbRNNCell{F,A,V,<:AbstractMatrix{T}})(h, x::Union{AbstractVecOrMat{T},OneHotArray}) where {F,A,V,T}
#     σ, Wi, Wh, b = m.σ, m.Wi, m.Wh, m.b
#     h = σ.(Wi*x .+ Wh*h .+ b)
#     sz = size(x)
#     return h, reshape(h, :, sz[2:end]...)
#   end
  
#   @functor RNNCell
  
#   function Base.show(io::IO, l::FbRNNCell)
#     print(io, "FbRNNCell(", size(l.Wi, 2), ", ", size(l.Wi, 1))
#     l.σ == identity || print(io, ", ", l.σ)
#     print(io, ")")
#   end
  
#   """
#       FbRNN(in::Integer, out::Integer, σ = tanh, f = identity)
  
#   Recurrent layer with feedback after output from external function;
#   like `RNN` layer, but with the output fed back into the input
#   through an external dependency function each time step.
#   """
#   FbRNN(a...; ka...) = Flux.Recur(FbRNNCell(a...; ka...))
#   Recur(m::FbRNNCell) = Flux.Recur(m, m.state0)

