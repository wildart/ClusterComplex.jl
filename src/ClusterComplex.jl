module ClusterComplex

import ComputationalHomology: complex, group, simplices, persistenthomology, dim, witness, StandardReduction
import Clustering: ClusteringResult, nclusters, assignments
import Distances
import MultivariateStats: fit, PCA, mean, projection
import Statistics: cov, mean
import LinearAlgebra: inv, pinv, norm

export clustercomplex

include("datasets.jl")
include("dominance.jl")

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
    pointToManifoldMahalanobis(X, μ, B, assignments)

Calculate Mahalanobis distance in a point to linear manifold distance space which is constructed from the data `X`.

# Arguments
- `X::AbstractMatrix`: the dataset matrix
- `μ::AbstractVector`: the partition mean
- `B::AbstractMatrix`: the partition basis
- `assignments`: the partition point assignments
"""
function pointToManifoldMahalanobis(X::AbstractMatrix{T}, μ::AbstractVector{T}, B::AbstractMatrix{T},
                                    assignments::AbstractVector{<:Integer}) where T <: Real
    # calculate distances for all data points to
    # a linear manifold and its orthoganal compliment
    D1 = distance_to_manifold(X, μ, B)
    D2 = distance_to_manifold(X, μ, B; ocss=true)

    # use distances as space dimensions
    δ = hcat(D1, D2)'

    # determine covariance matrix from the distances associated with cluster
    δs = δ[:,assignments]
    S = cov(δs, dims=2)
    S⁻¹ = try
        inv(S)
    catch
        pinv(S) # use pseudoinverse
    end

    # calculate Mahalanobis to all the points with cluster-derived covariance matrix
    return Distances.colwise(Distances.Mahalanobis(S⁻¹), δ, zeros(T, size(δ)...)), δ, S
end

function invcov(data)
    S = cov(data, dims=2)
    S⁻¹ = try
        inv(S)
    catch
        pinv(S) # use pseudoinverse
    end
    return S⁻¹
end

"""
    mahalonobis!(D, data, partition::ClusteringResult, χ[; kwargs...])

Fill the distance matrix `D` using a Mahalonobis distance from `partition` to `data` points.

# Arguments
- `data::AbstractMatrix`: the dataset matrix.
- `partition::ClusteringResult`: the partition of the dataset.
- `χ::Real`: the maximal radius of the sphere.
- `subspacemaxdim::Integer = size(data,1)-1`: the maximal dimension of the partition subspace
- `maxoutdim::Integer`: the maximal dimension of the simplicial complex
- `ν::Integer=0`: the witness comples construction mode (see `witness`)
"""
function mahalonobis!(D::AbstractMatrix{T},
                      data::AbstractMatrix{T},
                      partition::P) where {T <: Real, P <: ClusteringResult}

    N = size(data,2)         # number of points
    K = nclusters(partition) # number of partitions

    size(D) != (N,K) && throw(DimensionMismatch("Incorrect distance matrix size: $(size(D)) != ($N,$K)"))

    # calculate distance matrix from partitions
    assign = assignments(partition)
    for i in 1:K
        pdataidxs = assign .== i
        pdata = view(data, :, pdataidxs)
        μ = vec(mean(pdata, dims=2))
        dist = Distances.Mahalanobis(invcov(pdata))
        D[:, i] .= Distances.colwise(dist, data, μ)
    end
end

"""
    subspacemahalonobis(data, partition::ClusteringResult, χ[; kwargs...])

Construct a simplicial complex from `data` using its `partition` and outputs its filtration up to a maximal radius `χ`.

# Arguments
- `data::AbstractMatrix`: the dataset matrix.
- `partition::ClusteringResult`: the partition of the dataset.
- `χ::Real`: the maximal radius of the sphere.
- `subspacemaxdim::Integer = size(data,1)-1`: the maximal dimension of the partition subspace
- `maxoutdim::Integer`: the maximal dimension of the simplicial complex
- `ν::Integer=0`: the witness comples construction mode (see `witness`)
"""
function subspacemahalonobis!(D::AbstractMatrix{T},
                              data::AbstractMatrix{T},
                              partition::P;
                              subspacemaxdim::Integer = size(data,1)-1) where {T <: Real, P <: ClusteringResult}

    N = size(data,2)         # number of points
    K = nclusters(partition) # number of partitions

    size(D) != (N,K) && throw(DimensionMismatch("Incorrect distance matrix size: $(size(D)) != ($N,$K)"))

    # calculate distance matrix from partitions
    assign = assignments(partition)
    for i in 1:K
        pdataidxs = assign .== i
        pdata = view(data, :, pdataidxs)
        m = fit(PCA, pdata, maxoutdim=subspacemaxdim)
        D[:, i] .= first(pointToManifoldMahalanobis(data, mean(m), projection(m), pdataidxs))
    end
end

"""
    clustercomplex(data, partition::ClusteringResult, χ [; kwargs...])

Construct simplicial complex from `data` and its `partition` with maximal distance `χ`.
Returns a simplicial complex with weights evaluated from distances.

# Arguments
- `data::AbstractMatrix`: the dataset matrix.
- `partition::ClusteringResult`: the partition of the dataset.
- `χ::Real`: the maximal radius of the elliptical sphere.
- `method::Symbol = :mahalonobis`: the default construction method
- `maxoutdim::Integer = 1`: the maximal dimension of the simplicial complex
- `expansion::Symbol = :incremental`: the default simplicial complex construction method
- `ν::Integer=0`: the witness comples construction mode (see `witness`)
- `subspacemaxdim::Integer = size(data,1)-1`: the maximal dimension of the partition subspace
"""
function clustercomplex(data::AbstractMatrix{T}, partition::P, χ::T;
                        method=:mahalonobis,
                        maxoutdim::Integer = 1,
                        expansion = :incremental,
                        ν::Integer=0,
                        kwargs...) where {T <: Real, P <: ClusteringResult}

    N = size(data,2)          # number of points
    K = nclusters(partition)  # number of partitions
    D = Array{T}(undef, N, K) # distance matrix from partitions

    # calculate distance matrix with appropriate method
    if method == :mahalonobis
        mahalonobis!(D, data, partition)
    elseif method == :subspacemahalonobis
        params = filter(p->first(p) == :subspacemaxdim, kwargs)
        ssmaxdim = get(params, :subspacemaxdim, size(data,1)-1)
        subspacemahalonobis!(D, data, partition, subspacemaxdim=ssmaxdim)
    else
        throw(ArgumentError("Invalid method name $(method)"))
    end

    return witness(D', χ, ν=ν, maxoutdim=min(maxoutdim, K), expansion=expansion)
end

end
