const CMDist = ContinuousMultivariateDistribution

"""Model-based clustering results
"""
struct ModelClusteringResult <: ClusteringResult
    models::Vector{CMDist}
    assignments::Vector{Int}
end
Base.show(io::IO, mcr::ModelClusteringResult) = print(io, "Model Clustering: $(nclusters(mcr)) clusters")
assignments(clust::ModelClusteringResult) = clust.assignments
nclusters(clust::ModelClusteringResult) = length(clust.models)
counts(clust::ModelClusteringResult) = map(i->count(clust.assignments .== i), 1:nclusters(clust))
