using ClusterComplex, Test
import Random

@testset "Complex" begin

    Random.seed!(49184624)

    K = 10
    d = 15.0

    X, β_prof = ClusterComplex.dataset("TwoMoons")
    cls = kmeans(X, K)
    cplx, w = clustercomplex(X, cls, d, maxoutdim=K)

    @test length(size(cplx)) == K
    @test maximum(map(maximum, values(w))) < d

end

