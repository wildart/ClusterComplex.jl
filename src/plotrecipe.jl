using RecipesBase
import LinearAlgebra: Symmetric, eigen, norm
import Clustering: assignments

rotationmat(θ) = [cos(θ) -sin(θ); sin(θ) cos(θ)]

function elliptical_bound(S::AbstractMatrix{<:Real}, χ::Real)
    F = eigen(Symmetric(S))
    λ = F.values
    FI = sortperm(λ, rev=true)
    EV = F.vectors[:,FI[1]]
    ϕ = atan(EV[2], EV[1])
    if ϕ < 0
        ϕ += 2π
    end

    θ = range(0.0, 2π+0.01, step=0.1)
    a, b = χ*sqrt(λ[FI[1]]), χ*sqrt(λ[FI[2]])
    x, y  = a.*cos.(θ), b.*sin.(θ)
    ellipse = rotationmat(ϕ) * hcat(x, y)'
    expvar = λ./sum(λ)
    return ellipse, F.vectors[:,FI] .* sqrt.(λ[FI]')
end

# Plot a 2D elliptical boundary around cluster `mc`
@recipe function f(mc::T, args...; pc=false) where {T<:MahalonobisCluster}
    χ = length(args) > 0 ? args[1] : 1.0
    B, V = elliptical_bound(mc.sigma, χ)

    if pc
        pts = [mc.mu.+V[:,1] mc.mu mc.mu.-V[:,2]]
        @series begin
            seriestype := :path
            linewidth --> 2
            label := ""
            pts[1,:], pts[2,:]
        end
    end

    @series begin
        seriestype := :path
        linewidth --> 1
        linecolor --> :black
        label --> "χ=$χ"
        B[1,:].+mc.mu[1], B[2,:].+mc.mu[2]
    end
end

@recipe function f(mcr::T, args...; colors=false) where {T<:MahalonobisClusteringResult}
    χ = length(args) > 0 ? args[1] : 1.0
    for (i,mc) in enumerate(mcr.clusters)
        @series begin
            label --> "CB$i"
            linecolor --> (colors ? i : :black)
            mc, χ
        end
    end
end

# Plot a clustering
@recipe function f(cl::T, data::AbstractMatrix, args...) where {T<:ClusteringResult}
    L = assignments(cl)
    for l in sort!(unique(L))
        pts = view(data, :, findall(isequal(l), L))
        @series begin
            seriestype := :scatter
            markercolor --> l
            label --> "C$l"
            pts[1,:], pts[2,:]
        end
    end
end
