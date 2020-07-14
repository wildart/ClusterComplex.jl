const CMDist = ContinuousMultivariateDistribution

"""Model-based clustering"""
struct ModelClusteringResult <: ClusteringResult
    models::Vector{CMDist}
    assignments::Vector{Int}
end
Base.show(io::IO, R::ModelClusteringResult) = print(io, "Model Clustering: $(nclusters(R)) clusters")
nclusters(R::ModelClusteringResult) = length(R.models)
counts(R::ModelClusteringResult) = map(i->count(R.assignments .== i), 1:nclusters(R))
models(R::ModelClusteringResult) = R.models

"""User-defined clustering"""
struct CustomClusteringResult <: ClusteringResult
    assignments::Vector{Int}
    centers::Matrix
end
CustomClusteringResult(a::Vector{Int}) = CustomClusteringResult(a, zeros(0,0))
Base.show(io::IO, R::CustomClusteringResult) = print(io, "Custom Clustering: $(nclusters(R)) clusters")
counts(R::CustomClusteringResult) = [count(==(i), R.assignments) for i in unique(R.assignments)]
nclusters(R::CustomClusteringResult) = length(counts(R))
