# Execute Julia snippets in README order, sharing their documented session.
using GrowthModels
for block in eachmatch(r"```julia\n(.*?)```"s,read(joinpath(@__DIR__,"..","README.md"),String))
    include_string(Main,block.captures[1],"README.md")
end
println("README Julia examples passed")
