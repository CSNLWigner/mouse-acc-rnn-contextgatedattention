__precompile__()
module MathUtils


export @asarray, @horizontalvector, @mapseq, @catvecleftsingleton
export vectriu, mld
export kde
export @movingaverage
export convolve, smooth!, smooth
export absmax, limsabsmaxnegpos
export getcorrelation,marcenkopasturlimits
export gpcovse, gpsample
export erode1d!, dilate1d!
export mutualinformation, mutualinformationtable
export bernoulliloglikelihood
export nonlinearmatrixpower, angleofvectors, orthogonalizevectors
export standardizearray!

using Distributions
using StatsBase
using HypothesisTests
using LinearAlgebra

using Zygote: Buffer


# combination function simplicity macros

macro asarray(vectorofvectors)
     :( reduce(vcat, map(transpose, $(esc(vectorofvectors)) ) ) )
end

macro horizontalvector(vector)
    :( reshape(   $(esc(vector)), 1, :   ) )
end


macro mapseq(f,xs)
    :(   [ $(esc(f))(x) for x in $(esc(xs)) ]    )
end


"""
concatenate a vector of arrays along a new first dimension
"""
macro catvecleftsingleton(A)
    :(   vcat(reshape.($(esc(A)),1,size($(esc(A))[1])...)...)   )
end




# linear algebra ultilities

function vectriu(A::AbstractArray,k=1)
    # return the upper triangular matrix values in vector and its cartesian indices
    # k for diagonal
    # simple
    # h = [ A[i, j] for i = 1:size(A, 1), j = 1:size(A, 2) if i >= j + k ]
    # same with CartesianIndices
    c = [ cart for cart in CartesianIndices(A) if cart[2] >= cart[1] + k]
    return A[c], c
end


function multiplylastdimensions(A::AbstractArray,B::AbstractArray)
    # will multiply the last dimensions of B with A
    # noncommutative!
    sA = size(A)
    sB = size(B)
    C = zeros(sB[1],sA[1],sB[3])
    for i in 1:sB[1]
        C[i,:,:] = A * B[i,:,:]
    end
    return C
end
mld = multiplylastdimensions








# statistics utilities





function optimizesignalreversefit(p::AbstractVector, ic)
    # return a single gauissan with mode fit using reverse kldivergence
    function reverseklloss(a)
        x = collect(range(0.,1.,length(p)))
        q = pdf.( Normal(a[1],a[2]), x )
        e = kldivergence(q,p)
    end 
    r = optimize(reverseklloss, ic)
    μ,σ = Optim.minimizer(r)
end


function kde(x; n=100, σ=2.25)
    dists = Normal.(x, sqrt(σ))
    x_range = range(minimum(x), maximum(x), length = n)
    densities = sum(pdf.(dist, x_range) for dist in dists)
    return x_range, densities
end

macro movingaverage(x, sm)
    :( MathUtils.convolve($(esc(x)), ones($(esc(sm)))./$(esc(sm))) )
end


function convolve(x::Vector{Float64},v::Vector{Float64}; mode=:same)
    nx = length(x)
    nvh = length(v)÷2
    xp = zeros(nx+2*nvh) #  initializing proper size
    xp[1:nvh] .= copy(x[1]) # constant padding pre and post domain of x
    xp[nvh+nx:end] .= copy(x[end])
    xp[nvh:nvh+nx-1] = copy(x)    # avoiding modifying function argument
    cp = [ xp[t-nvh:t+nvh]'*v for t in nvh+1:nvh+nx ]
    if mode==:same
        return cp
    else
        error("Convolution mode: $(mode) not implemented")
    end
end


function smooth!(x,samplingperiod;kernelwidth=3,σ=1)
    # provide gaussian smoothing
    kernelxrange = collect(-kernelwidth*σ:samplingperiod:kernelwidth*σ)
    k = pdf.(Normal.(0,σ), kernelxrange)
    k = k./sum(k)
    x = convolve(x,k,mode=:same)
end
function smooth(x,samplingperiod;kernelwidth=3,σ=1)
    s = copy(x)
    s = smooth!(s,samplingperiod;kernelwidth,σ)
end


absmax(x) = maximum(abs.(x))
limsabsmaxnegpos(x) = (-absmax(x),absmax(x))








# information theoretic functions
function makepmf(x,e)    # supplied edges
    h = normalize( fit(Histogram, x, e), mode=:probability )
    pmf = h.weights
end
function makepmf(x)      # automatic edges
    h = normalize( fit(Histogram, x), mode=:probability )
    pmf = h.weights
end


function mutualinformation(X::AbstractArray{<:Real},Y::AbstractArray{<:Real}, edges=nothing)
    x = vec(X)
    y = vec(Y)
    if edges===nothing
        pxy = makepmf((x,y))
        px = makepmf(x)
        py = makepmf(y)
    else
        pxy = makepmf((x,y),edges)
        px = makepmf(x,edges[1])
        py = makepmf(y,edges[2])
    end

    # @info "pdfs" pxy px .* py' px' py' sum(pxy) sum(px) sum(py)

    
    # the followings two options are both possible, kldivergence from StatsBase is faster though
    # miq = (pxy ./ (px .* py') )
    # mi = pxy .* log.(miq)
    # mim = sum(vec(mi)[.!isnan.(vec(mi))]')
    # @info "mutualinformation" miq mi sum(mi) vec(mi)' vec(mi)[.!isnan.(vec(mi))]' mim

    kl = kldivergence(pxy, px .* py')
    # @info "kl divergence" kl
end


"""
    Returns a dim factors by dim data   matrix of
    mutual informations between X data and Z factors
"""
function mutualinformationtable(X::AbstractMatrix,Z::AbstractMatrix, args...)
    ndata, nobservations = size(X)
    nfactors,npredictions = size(Z)
    @assert nobservations==npredictions "The number of data points must be the same: second dimensions of X and Z."
    mutualinformationtable = zeros(eltype(X),nfactors,ndata)
    for r in 1:nfactors, d in 1:ndata
        mutualinformationtable[r,d] = mutualinformation(X[d,:], Z[r,:], args)
    end
    mutualinformationtable
end
mutualinformationtable(X::AbstractMatrix,Z::AbstractMatrix) = mutualinformationtable(X::AbstractMatrix,Z::AbstractMatrix, nothing)


function bernoulliloglikelihood(k,p)
    @. k * log(  p  )   +   (1 - k) * log( (1 - p) )
end







"""
kth power of M matrix, but at each multiplication
apply the scalar m, and the function f
"""
function nonlinearmatrixpower(f,M,k,m=1)
    R = copy(M)
    for i in 1:k
        R = f.(R * R .* m)
    end
    return R
end



function angleofvectors(v1::Vector{<:Number},v2::Vector{<:Number})        #; returnscale=:degree)
    s = v1' * v2 / (norm(v1) * norm(v2))
    s = max(min(1.,s),-1.)
    acos(s) * 180 / π
end
function angleofvectors(M1::Matrix{<:Number},M2::Matrix{<:Number};dims=2)        #; returnscale=:degree)
    @assert size(M1)==size(M2) "Matrices must have the same size"
    adim = dims==1 ? 2 : 1
    angles = zeros(eltype(M1),size(M1,adim))
    if dims==1
        for t in axes(M1,adim)
            angles[t] = angleofvectors(M1[:,t],M2[:,t])
        end
    else
        for t in axes(M1,adim)
            angles[t] = angleofvectors(M1[t,:],M2[t,:])
        end
    end
    return angles
end


"""
Given a set of vectors, transform to
an orthogonal set like Gram-Smidth process
but with QR decomposition
"""
function orthogonalizevectors(B)
    if typeof(B)<:AbstractVector || size(B,2)==1
        return B
    end
    F = factorize(B)
    return Matrix(F.Q)
end



function standardizearray!(d;dims=2)
    m = mean(d,dims=dims)
    d = d .- m
    s = std(d, dims=dims)
    s[s.==0] .= 1.
    d = d ./ s
end




function getcorrelation(x,y)
    â = [ x ones(eltype(x),size(x,1))] \ y
    c = HypothesisTests.CorrelationTest(x,y)
    p = pvalue(c)
    return â, c.r, p
end






# gaussian processes utilities
gpkernelse(x1,x2,stdev,timescale) = stdev^2 *  exp( - (x1-x2) * (x1-x2)' /2 /timescale^2 )



function gpcovse(t::Vector{Float64},σ::Float64,ρ::Float64,l::Float64)
    N = length(t)
    D = Buffer(diagm(ones(N)))                # avoid mutating arrays
    # D = diagm(ones(N))
    for j = 1:N, i = 1:j
        D[i,j] = D[j,i] = gpkernelse(t[i],t[j],ρ,l)
    end
    Σ = Distributions.PDMat( copy(D)  + diagm(fill(σ,N))  )
end

function gpsample(t::Vector{Float64},σ::Float64,ρ::Float64,l::Float64; minrate=1e-5, nsample=1000)
    N = length(t)
    Σ = gpcovse(t,σ,ρ,l)
    samples = Buffer(rand(MvNormal(zeros(N), Σ ),nsample))            # sample the gp
    λ = Buffer(mean(copy(samples),dims=2))                                  # get a mean of multiple samples; avoid mutating arrays
    λ[λ.<minrate] .= minrate                            # avoid small, zero or negative rates at any timepoint
    return copy(λ)                                                          # return the mean of the gaussian process
end








# image morphology utilities


function erode1d!(h)
    x = copy(h)
    h[1] = minimum([x[1],x[1+1]])
    for i in 2:length(x)-1
        h[i] = minimum([x[i-1],x[i],x[i+1]])
    end
    h[end] = minimum([x[end-1],x[end]])
end
function dilate1d!(h)
    x = copy(h)
    h[1] = minimum([x[1],x[1+1]])
    for i in 2:length(x)-1
        h[i] = maximum([x[i-1],x[i],x[i+1]])
    end
    h[end] = minimum([x[end-1],x[end]])
end








# combinatorics utilities


function combination(N,K)
    @assert N>=K "n must be greater than or equal to k"
    factorial(big(N)) / factorial(big(K)) / factorial(big(N-K))
end

"""
Returns the probability of k consecutive successes in n trials
anywhere in the sequence of trials.  Works with either n or k as vector.
"""
function probabilityconsecutive(N::Integer,K::Integer)
    @assert N>=K "n must be greater than or equal to k"
    @. (N-K+1)/combination(N,K)
end




"""
Returns the probabilistic of I blocks of each K consecutive successes in N trials.
"""
function probabilityfixednumberofconsecutive(N::Integer, K::Integer, I::Integer)
    # @assert N>=K && K>=I "N ≥ K, K ≥ I"
    if N < K || I < 1
        # Base case: if N is less than K or I is less than 1, return 0
        return 0
    end
    if N == K && I == 1
        # Base case: if N is equal to K and I is equal to 1, return 1
        return 1
    end
    
    # Recursive case: otherwise, return the sum of two cases:
    # Case 1: the first block of K points is followed by a gap of at least one point
    # Case 2: the first block of K points is followed by another block of K points
    # For each case, reduce N by K and I by 1 and multiply by the probability of choosing K points from N
    (N - K + 1) * factorial(big(K)) / prod(N:-1:N-K+1) *
            ( probabilityfixednumberofconsecutive(N-K-1, K, I-1) +
            probabilityfixednumberofconsecutive(N-K,   K, I-1) )
end



"""
Returns the number of ways K points can be partitioned into I blocks of different sizes.
"""
function numberofwayspartition(N::Integer, I::Integer)
    # # Base case: if N is less than 1 or I is less than 1, return 0
    # if N < 1 || I < 1
    #     return 0
    # end
    # Base case: if N is equal to 1 and I is equal to 1, return 1
    if N == 1 && I == 1
        return 1
    end
    # Create a table of size (N+1) x (I+1) and initialize it with zeros
    table = zeros(Int, N+1, I+1)

    # Fill after the first row and column the rest of the table using the recurrence relation:
    # table[n, i] = table[n-1, i-1] + i * table[n-1, i]
    # This means that we can partition n points into i blocks of different sizes by either:
    # - Taking one point and making it a new block, and then partitioning the remaining n-1 points into i-1 blocks
    # - Taking one point and adding it to any of the existing i blocks, and then partitioning the remaining n-1 points into i blocks
    for n in 2:N+1, i in 2:I+1
        table[n, i] = table[n-1, i-1] + i * table[n-1, i]
    end

    # Return the value at the bottom right corner of the table
    return table[N+1, I+1]

end


"""
Returns the probability of I blocks of each length of consecutive successes in N trials.
"""
function probabilityanynumberofconsecutive(N::Integer)
    prob = 0
    for I in 1:N
        for K in 1:floor(N/I)
            # Call the table function to get the number of ways to partition
            # N points into I blocks of different sizes
            ways = numberofwayspartition(N, I)
            # Multiply the number of ways by the probability of choosing K points from N points
            # prob += ways * (N - K + 1) * factorial(K) / prod(N:-1:N-K+1)
            prob += ways * (N - K + 1) * combine(N,K)
        end
    end
    # Divide the probability by C(N,K), where K is the largest possible size of a block (which is N if I = 1)
    prob /= combination(N,N)
    return prob

  end
  
    
   

end