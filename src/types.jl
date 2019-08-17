const CMDist = ContinuousMultivariateDistribution

"""Model-based clustering"""
struct ModelClusteringResult <: ClusteringResult
    models::Vector{CMDist}
    assignments::Vector{Int}
end
Base.show(io::IO, mcr::ModelClusteringResult) = print(io, "Model Clustering: $(nclusters(mcr)) clusters")
assignments(clust::ModelClusteringResult) = clust.assignments
nclusters(clust::ModelClusteringResult) = length(clust.models)
counts(clust::ModelClusteringResult) = map(i->count(clust.assignments .== i), 1:nclusters(clust))
models(clust::ModelClusteringResult) = clust.models

"""User-defined clustering"""
struct CustomClusteringResult <: ClusteringResult
    counts::Vector{Int}
    assignments::Vector{Int}
end
Base.show(io::IO, R::CustomClusteringResult) = print(io, "Custom Clustering: $(nclusters(R)) clusters")
CustomClusteringResult(a::Vector{Int}) = CustomClusteringResult([count(==(i), a) for i in unique(a)],a)
