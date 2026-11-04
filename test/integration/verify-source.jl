# Root-environment tests must exercise the intended API revision, not a stale
# installed package or an accidental LOAD_PATH override. Compare executable
# source with this checkout; documentation-only commits do not invalidate a pin.
function verify_growthmodels_source()
    expected = normpath(joinpath(@__DIR__, "../.."))
    loaded = dirname(dirname(pathof(GrowthModels)))
    for subtree in ("src", "ext")
        for (dir, _, files) in walkdir(joinpath(expected, subtree))
            for file in files
                endswith(file, ".jl") || continue
                reference = joinpath(dir, file)
                installed = joinpath(loaded, relpath(reference, expected))
                isfile(installed) && read(installed) == read(reference) ||
                    error("Loaded GrowthModels differs from this checkout: $installed")
            end
        end
    end
    println("Verified GrowthModels source: ", loaded)
    return loaded
end
verify_growthmodels_source()
