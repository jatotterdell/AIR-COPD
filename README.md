# AIR

AIR: **A**daptive clinical trial **I**nvestigating GLA-targeted therapies in the **R**espiratory disease COPD

## Background

The aim for AIR is to evaluate multiple novel interventions against a shared control group. The primary outcome is neutraphil elastase (NE) at D28 (i.e. 28 days after randomisation).

----

## Scripts

This code base is using the Julia Language and [DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/) to make a reproducible scientific project named
> AIR

To (locally) reproduce this project, do the following:

1. Download this code base. Notice that raw data are typically not included in the
   git-history and may need to be downloaded independently.
2. Open a Julia console and do:

   ```julia
   julia> using Pkg
   julia> Pkg.add("DrWatson") # install globally, for using `quickactivate`
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box, including correctly finding local paths.

### Run Simulation

To run a simulation file using multi-threading, call for example

```shell
julia -t auto scripts/run_simulations.jl
```

or with `auto` replaced with the number of threads.

Alternatively, if sysimage has been created can instead use

```shell
julia -t auto --sysimage=AIRproject.so scripts/run_simulations.jl
```

### Tests

Some basic tests implemented in `test/` which can be run via

```julia
pkgs> activate .
(AIR) pkg> test
```
