using Gen
using UnicodePlots
using Distributions
using Plots
using Revise

ENV["GKSwstype"]="160" # fixes some plotting warnings

include(joinpath(@__DIR__, "helpers.jl"))
include(joinpath(@__DIR__, "models.jl"))

"""
    inference_procedure

Performs Metropolis-Hastings MCMC.
"""
function inference_procedure(gm_args::Tuple,
                             obs::Gen.ChoiceMap,
                             steps::Int = 100)

    tr, ls = Gen.generate(model, gm_args, obs)

    println("Initial logscore: $(ls)")

    # count the number of accepted moves and track accepted proposals
    acceptance_count = 0
    traces = Vector{Gen.DynamicDSLTrace}(undef, steps) 

    for i = 1:steps
        if i % 100 == 0
            println("$(i) steps completed")
        end

        tr, accepted = mh(tr, proposal, ())
        traces[i] = tr
        acceptance_count += Int(accepted)
    end

    acceptance_ratio = acceptance_count / steps

    println("Final logscore: $(get_score(tr))")
    println("Acceptance ratio: $(acceptance_ratio)")

    return (traces, acceptance_ratio)
end


function main()

    t = 60 # 2 seconds of observations
    (gargs, obs, truth) = data_generating_procedure(t)

    (traces, aratio) = inference_procedure(gargs, obs, 2500)    

    
    display(plot_traces(truth, traces[1000:end]))

    println("press enter to exit the program")
    readline()
    return nothing
end


#main();
