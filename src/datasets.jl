import DelimitedFiles
import GZip
import Random

function loaddata(fio, trans = true, skip=false)
    skiplines = 0
    if skip
        l = readline(fio)
        while (l[1] == '@')
            skiplines += 1
            l = readline(fio)
        end
        seekstart(fio)
    end
    data = DelimitedFiles.readdlm(fio, ',', Float64, '\n', skipstart=skiplines)
    lbls = convert(Vector{Int}, data[:, end]) # class in last column
    data = data[:,1:end-1] # drop last column
    return (trans ? data' : data), lbls
end

function opendatafile(func, fname)
    fio = if endswith(fname, ".gz")
        GZip.open(fname)
    else
        open(fname)
    end
    try
        func(fio)
    finally
        close(fio)
    end
end

# :Name => (File, PROF)
const DATASETS = Dict(
    "TwoMoons" => ("twomoons.csv", (2, 0)),
    "TreeMoons" => ("treemoons.csv", (3, 0)),
    "Circles"  => ("circles.csv", (2, 2)),
    "OIP15"    => ("optical-k15.csv.gz", (1, 5)),
    "OIP300"   => ("optical-k300.csv.gz",(1, 1)),
    "Sphere"   => ("sphere.csv", (1, 0, 1)),
    "Wine"     => ("wine.dat", (1, 0)),
    "Glass"     => ("glass.dat", (1, 0)),
)

function dataset(ds::String; trans = true)
    @assert haskey(DATASETS, ds) "Dataset $ds not found"
    dname = dirname(@__DIR__)
    ds = DATASETS[ds]
    skip = endswith(ds[1], ".dat")
    X, L = opendatafile(joinpath(dname, "data", ds[1])) do fio
        loaddata(fio, trans, skip)
    end
    @debug "Data loaded" size=size(X)
    return X, ds[2], L
end

function sphere(n::Int)
    X = randn(3, n)
    X ./= mapslices(norm, X,  dims=1)
    return X
end
