import DelimitedFiles
import GZip
import Random

function loaddata(fio, droplast = true, trans = true)
    data = DelimitedFiles.readdlm(fio, ',', Float64, '\n')
    if droplast
        data = data[:,1:end-1] # drop last column
    end
    return trans ? data' : data
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
    "Circles"  => ("circles.csv", (2, 2)),
    "OIP15"     => ("optical-k15.csv.gz", (1, 5)),
    "OIP300"    => ("optical-k300.csv.gz",(1, 1)),
    "Sphere"   => ("sphere.csv", (1, 0, 1)),
)

function dataset(ds::String; droplast = true, trans = true)
    @assert haskey(DATASETS, ds) "Dataset $ds not found"
    dname = dirname(@__DIR__)
    ds = DATASETS[ds]
    X = opendatafile(joinpath(dname, "data", ds[1])) do fio
        loaddata(fio, droplast, trans)
    end
    @debug "Data loaded" size=size(X)
    return X, ds[2]
end

function sphere(n::Int)
    X = randn(3, n)
    X ./= mapslices(norm, X,  dims=1)
    return X
end
