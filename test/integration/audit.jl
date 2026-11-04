using Test, GrowthModels, TOML
include("verify-source.jl")
const analysis_root = normpath(joinpath(@__DIR__, "../../../aggregate-transfers"))
const inventory = joinpath(analysis_root,"docs/growthmodels-api-callers.txt")
const callers = filter(line -> !isempty(line) && !startswith(line,"#"),readlines(inventory))
function syntax_errors(expr)
    expr isa Expr || return false
    expr.head in (:error,:incomplete) && return true
    return any(syntax_errors,expr.args)
end
const retired = r"\b(?:SolvedModel|StateSpaceHyperParams|HyperParams|solve_HJB|StateEvolution|statespace_k_dot)\b|\.variables\.(?:k|z|η|c|y)\b|\.value\.(?:v|A|dVf|dVb)\b|\.(?:policy_function|kdot_function)\b"
@testset "Active downstream source and environment audit" begin
    @test !isempty(callers)
    for relative in callers
      @testset "$relative" begin
        path = joinpath(analysis_root,relative)
        @test isfile(path)
        text = read(path,String)
        @test !syntax_errors(Meta.parseall(text; filename=path))
        code_lines = filter(line -> !startswith(lstrip(line),"#"),split(text,'\n'))
        @test !any(line -> occursin(retired,line),code_lines)
        for match in eachmatch(r"(?m)^\s*include\(\"([^\"\n]+)\"\)",text)
            @test isfile(normpath(joinpath(dirname(path),match.captures[1])))
        end
      end
    end
    for environment in ("", "filter", "code/joint-generative-model")
        project = TOML.parsefile(joinpath(analysis_root,environment,"Project.toml"))
        @test project["sources"]["GrowthModels"]["rev"] == "v1.0.0"
        @test project["sources"]["GrowthModels"]["url"] == "https://github.com/EdJeeOnGitHub/GrowthModels.jl"
        manifest = TOML.parsefile(joinpath(analysis_root,environment,"Manifest.toml"))
        package = only(manifest["deps"]["GrowthModels"])
        @test package["repo-rev"] == "v1.0.0"
        @test package["repo-url"] == "https://github.com/EdJeeOnGitHub/GrowthModels.jl"
        @test package["version"] == "1.0.0"
    end
end
