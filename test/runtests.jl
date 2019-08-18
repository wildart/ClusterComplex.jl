using ClusterComplex, Clustering, Test
import Random

@testset "Complex" begin

    Random.seed!(49184624);

    K = 10
    d = 15.0

    X, β_prof, L = ClusterComplex.dataset("TwoMoons")
    cls = kmeans(X, K)
    cplx, w, mcr = clustercomplex(X, cls, d, maxoutdim=2)

    @test length(size(cplx)) == 3
    @test maximum(map(maximum, values(w))) < d

    @test nclusters(mcr) == K
    @test assignments(mcr) == assignments(cls)

    XC = CustomClusteringResult(L)
    @test nclusters(XC) == 2
    @test counts(XC) == [500, 500]
    @test assignments(XC) == L

end
