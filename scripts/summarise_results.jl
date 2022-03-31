using DrWatson
quickactivate(@__DIR__, "AIR")

using Weave

# Julia markdown to PDF
weave(projectdir("notebooks/results.jmd"); doctype = "md2pdf", out_path = projectdir("notebooks"))

results = collect_results(datadir("sims"))
res = results.result
out = summarise_trial.(res)
