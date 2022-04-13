using DrWatson
quickactivate(@__DIR__, "AIR")

using Weave
using Highlights

# Julia markdown to PDF
weave(
    projectdir("notebooks/results.jmd");
    doctype = "md2pdf",
    out_path = projectdir("notebooks"),
    template = projectdir("notebooks", "custom.tpl"),
    highlight_theme = Highlights.Themes.GitHubTheme
)
