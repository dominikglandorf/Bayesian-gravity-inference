using Gen
using UnicodePlots
using Distributions
using Plots

include(joinpath(@__DIR__, "helpers.jl"))

"""
    inference_procedure

Performs Metropolis-Hastings MCMC.
"""
function inference_procedure(gm_args::Tuple,
                             obs::Gen.ChoiceMap,
                             steps::Int = 100)

    # start with an initial guess of physical latents
    # `ls` is the log score or how well this
    # initial guess explains the observations
    tr, ls = Gen.generate(model, gm_args, obs)

    println("Initial logscore: $(ls)")

    # count the number of accepted moves and track accepted proposals
    acceptance_count = 0
    traces = Vector{Gen.DynamicDSLTrace}(undef, steps) 

    for i = 1:steps
        if i % 10 == 0
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

    t = 120 # 2 seconds of observations
    (gargs, obs, truth) = data_generating_procedure(t)

    (traces, aratio) = inference_procedure(gargs, obs, 500)    

    display(plot_traces(truth, traces))

    println("press enter to exit the program")
    readline()
    return nothing
end


main();
