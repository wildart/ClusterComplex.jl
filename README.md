# Cluster Complex

This package includes simplicial complex construction methods from linear manifold and subspace clusterings.

## Usage

```julia
julia> using Clustering, ClusterComplex

julia> X, β_prof = ClusterComplex.dataset("TwoMoons")
([1.10724 0.965332 … 2.16474 1.89748; 0.0516 0.0640259 … 0.565906 0.470026], (2, 0))

julia> cls = kmeans(X, 10);

julia> nclusters(cls)
10

julia> cplx, w = clustercomplex(X, cls, 15.0, maxoutdim=10);

julia> cplx
SimplicialComplex((10, 45, 120, 210, 252, 210, 120, 45, 10, 1))
```

## Resources

- "Topological Structure of Linear Manifold Clustering", TBA

