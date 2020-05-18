using Base.Threads
import ComputationalHomology: betti, homology

"""Caluclate relative dominance

See section 3 in Vin de Silva and Gunnar Carlsson, Topological estimation using witness complexes, 2004
"""
function dominance(flt, profile, final, ϵ = 0.0001; minval = 0.0)
    i = minval
    R0, R1, K = -eps(), 0.0, 0.0
    fcplx = nothing
    for (v, cpl) in flt
        if i <= v
            i += ϵ
        else
            continue
        end
        @debug "Complex" cpl
        bn = betti(homology(cpl))
        @debug "Filtration" value=v betti=bn
        if R0 < 0.0 && length(bn) >= length(profile) && bn[1:length(profile)] == profile
            R0 = v
            fcplx = copy(cpl)
        end
        if length(bn) >= length(final) && bn[1:length(final)] == final
            R1 = v - ϵ
            K = v
            break
        end
        tmp = v
    end
    return R0, R1, K, fcplx
end

function findprofile(flt, profile, ϵ = 0.0001; minval = 0.0, debug = false)
    i = minval
    fcplx = nothing
    for (v, cpl) in flt
        if i <= v
            i += ϵ
        else
            continue
        end
        debug && print("$cpl: ")
        bn = betti(homology(cpl))
        debug && println("$v => ", bn)
        if length(bn) >= length(profile) && bn[1:length(profile)] == profile
            fcplx = cpl
            break
        end
    end
    return copy(fcplx)
end

function dominance(flt, profile; reduction = StandardReduction)
    tmpflt = similar(flt)

    pl = length(profile)

    K = R0 = R1 = Inf
    res = ntuple(v->0, length(profile))
    final = ntuple(v->v == 1 ? 1 : 0, length(profile))
    for (fv, splxs) in simplices(flt)

        # construct incrementally filtration
        if length(splxs) > 0
            for s in splxs
                push!(tmpflt, s, fv)
            end

            ph = persistenthomology(reduction, tmpflt)

            # calculate betti profile
            maxdim = min(pl-1, dim(complex(tmpflt)))
            res = zeros(Int, length(profile))
            for d in 0:maxdim
                res[d+1] = group(ph,d)
            end
            res = tuple(res...)
        end
        # println(res[1:pl-1], " ", profile[1:pl-1], " ", final[1:pl-1], " => ",  res[pl:end], " ", profile[pl:end])

        if res[1:pl-1] == profile[1:pl-1] && res[pl:end] >= profile[pl:end] && isinf(R0)
            R0 = fv
            @debug "Betti profile @ $fv" β=res β⁰=profile R0
        end

        if res[1:pl-1] != profile[1:pl-1] && !isinf(R0) && isinf(R1)
            R1 = fv
            @debug "Betti profile @ $fv" β=res β⁰=profile R0 R1
        end

        if res[1:pl-1] == final[1:pl-1] && res[pl:end] >= final[pl:end] && !isinf(R0) && !isinf(R1) #res == (1,0,0) #&& isinf(K)
            K = fv
            @debug "Betti profile @ $fv" β=res β⁰=profile R0 R1 K
        end
        @debug "Betti profile @ $fv" β=res

        # missed the start of the interval
        isinf(R0) && res[1] < profile[1] && break

        # no interval end
        !(isinf(K) || isinf(R1)) && break
    end
    if isinf(K)
        K = maximum(flt)
    end
    if isinf(R1)
        R1 = maximum(flt)
    end

    # relative domanance calculation (cases)
    if K != 0
        if R1 == R0 == K
            # found all profiles at the same time (a,b,c) & (1, 0, 0)
            RelD = 1
        else
            # if K < R0 then 1 otherwise proper relative dominance
            RelD = min(1, (R1-R0)/K)
        end
    else
        # K == 0 then first complex in filtration is already 1 connected component
        # so if R1-R0 is ∞ (not found in filtration) then no dominance detected
        RelD = isinf(R1-R0) ? 0 : 1
    end
    # RelD = min(1, K == 0.0 ? 1.0 : (R1-R0)/K)
    RelD = any(isinf, [R0,R1,K]) ? NaN : (R1-R0)/K
    if all(iszero, [R0,R1,K])
        RelD = 1.0
    end

    @debug "Relative dominance" RelD R0 R1 K

    return RelD, R0, R1, K
end
