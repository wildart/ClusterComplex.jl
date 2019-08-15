import Clustering: ClusteringResult, nclusters, assignments, counts

struct MahalonobisCluster{T<:Real}
    mu::AbstractVector{T}
    sigma::AbstractMatrix{T}
    idx::Vector{Int}
end

struct MahalonobisClusteringResult <: ClusteringResult
    clusters::Vector{MahalonobisCluster}
    MahalonobisClusteringResult(mcs::Vector{<:MahalonobisCluster}) = new(mcs)
end
Base.show(io::IO, result::MahalonobisClusteringResult) =
    print(io, "MahalonobisClustering: $(length(result.clusters)) clusters")

clusters(mcr::MahalonobisClusteringResult) = mcr.clusters

nclusters(mcr::MahalonobisClusteringResult) = length(mcr.clusters)
counts(mcr::MahalonobisClusteringResult) = map(c->length(c.idx), mcr.clusters)
assignments(mcr::MahalonobisClusteringResult) =
    map(first, sort!([(i,j) for (i,c) in enumerate(mcr.clusters) for j in c.idx], by=e->e[2]))
