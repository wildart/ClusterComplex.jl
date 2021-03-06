module ClusterComplex

using ComputationalHomology: complex, group, persistenthomology, dim, witness, StandardReduction, TwistReduction
using Clustering: ClusteringResult, nclusters, assignments, counts
using Distances: Distances
using MultivariateStats: fit, PCA, mean, projection
using Statistics: mean, cov, covm
using LinearAlgebra: inv, pinv, norm, eigen, isposdef, diagm, diag, Diagonal
using Distributions: MvNormal, MixtureModel, ContinuousMultivariateDistribution,
                     logpdf, fit_mle, FullNormal, DiagNormal, sqmahal!

import Clustering: nclusters, counts, assignments

export clustercomplex, ModelClusteringResult, models, model, CustomClusteringResult

include("types.jl")
include("datasets.jl")
include("dominance.jl")
include("plotrecipe.jl")

function assignments(data::AbstractMatrix{T}, models::Vector{DS}) where {T <: Real, DS <: CMDist}
    lpp = hcat([logpdf(m, data) for m in models]...)
    return [findmax(r)[2] for r in eachrow(lpp)]
end

function invcov(S)
    return try
        inv(S)
    catch
        pinv(S) # use pseudoinverse
    end
end

function rescalecov(C)
    F = eigen(C)
    V = F.values

    # replace negative eigenvalues by zero
    V = max.(V, 0)

    # reconstruct correlation matrix
    CC = F.vectors * diagm(0=>V) * F.vectors'

    # rescale correlation matrix
    T = 1 ./ sqrt.(diag(CC))
    C = CC .* (T * T')
    return C
end

"""Calculate the distances from points, `X`, to a linear manifold with `basis` vectors, translated to `origin` point.

`ocss` parameter turns on the distance calculatation to orthogonal complement subspace of the given manifold.
"""
function distance_to_manifold(X::AbstractMatrix{T}, origin::AbstractVector{T}, basis::AbstractMatrix{T};
                              ocss::Bool = false) where T<:Real
    N, n = size(X)
    M = size(basis,2)
    # vector to hold distances of points from basis
    distances = zeros(T, n)
    tran = similar(X)
    @fastmath @inbounds for i in 1:n
        @simd for j in 1:N
            tran[j,i] = X[j,i] - origin[j]
            if !ocss
                distances[i] += tran[j,i]*tran[j,i]
            end
        end
    end
    proj = transpose(basis) * tran
    @fastmath @inbounds for i in 1:n
        b = 0.0
        @simd for j in 1:M
            b += proj[j,i]*proj[j,i]
        end
        distances[i] = sqrt(abs(distances[i]-b))
    end
    return distances
end

"""
    model(X, partition)

Constract Gaussian model from data using MLE.

# Arguments
- `X::AbstractMatrix`: the dataset matrix
- `partition::Vector{Int}`: the partition indexes
"""
function model(data::AbstractMatrix{<:Real}, partition::Vector{Int})
    pdata = view(data, :, partition)
    return try
        fit_mle(FullNormal, pdata)
    catch
        fit_mle(DiagNormal, pdata)
    end
end

"""
    ssmahalonobis(X, partition[; kwargs...])

Calculate Mahalanobis distance in a point to linear manifold distance
space which is constructed from from `partition` of the dataset `X`.

# Arguments
- `X::AbstractMatrix`: the dataset matrix.
- `partition::Vector{Int}`: the partition of the dataset.
- `subspacemaxdim::Integer = size(data,1)-1`: the maximal dimension of the partition subspace
"""
function ssmahalonobis(data::AbstractMatrix{T}, partition::Vector{Int};
                       subspacemaxdim::Integer = size(data,1)-1) where {T<:Real}
    # calculate projection to PC subspace
    m = fit(PCA, data[:,partition], maxoutdim=subspacemaxdim)

    # a linear manifold and its orthoganal compliment
    D1 = distance_to_manifold(data, mean(m), projection(m))
    D2 = distance_to_manifold(data, mean(m), projection(m); ocss=true)

    # use distances as space dimensions
    δ = hcat(D1, D2)

    # determine covariance matrix from the distances associated with cluster
    δs = δ[partition, :]
    μ = mean(δs, dims=1)
    C = covm(δs, μ, 1)

    # calculate Mahalanobis to all the points with cluster-derived covariance matrix
    dist = Distances.Mahalanobis(invcov(C))
    D = Distances.colwise(dist, data, zeros(T, size(μ,2)))
    return mean(m), C, D
end

"""
    clustercomplex(data, partition::ClusteringResult, χ [; kwargs...])

Construct simplicial complex from `data` and its `partition` with maximal distance `χ`.
Returns a simplicial complex with weights evaluated from distances.

# Arguments
- `data::AbstractMatrix`: the dataset matrix.
- `partition::ClusteringResult`: the partition of the dataset.
- `χ::Real`: the maximal radius of the elliptical sphere.
- `maxoutdim::Integer = 1`: the maximal dimension of the simplicial complex
- `expansion::Symbol = :incremental`: the default simplicial complex construction method
- `ν::Integer=0`: the witness comples construction mode (see `witness`)
- `subspacemaxdim::Integer = size(data,1)-1`: the maximal dimension of the partition subspace
"""
function clustercomplex(data::AbstractMatrix{T}, partition::P, χ::T;
                        maxoutdim::Integer = 1,
                        expansion = :incremental,
                        ν::Integer=0,
                        kwargs...) where {T <: Real, P <: ClusteringResult}

    M, N = size(data)         # number of points
    K = nclusters(partition)  # number of partitions
    mc = Array{MvNormal}(undef, K)

    # construct models
    assign = assignments(partition)
    for i in unique(assign)
        idxs = findall(assign .== i)
        mc[i] = model(data, idxs)
    end

    clustercomplex(data, mc, χ, assignments=assign, maxoutdim=maxoutdim, expansion=expansion)
end

function clustercomplex(data::AbstractMatrix{T}, models::Vector{DS}, χ::T;
                        assignments::Vector{Int} = Int[],
                        maxoutdim::Integer = 1,
                        expansion = :incremental,
                        ν::Integer=0) where {T <: Real, DS <: CMDist}

    M, N = size(data)         # number of points
    K = length(models)        # number of partitions
    D = Array{T}(undef, N, K) # distance matrix from partitions

    # calculate distance matrix with appropriate method
    for (i,m) in enumerate(models)
        Dᵢ = @view D[:, i]
        sqmahal!(Dᵢ, m, data)
        Dᵢ .= sqrt.(Dᵢ)
    end

    cplx, w = witness(D', χ, ν=ν, maxoutdim=min(maxoutdim, K), expansion=expansion)
    return cplx, w, ModelClusteringResult(models, assignments)
end

end
