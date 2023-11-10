using Gen
using UnicodePlots
using Distributions
using Plots
using Printf

include(joinpath(@__DIR__, "helpers.jl"))

"""
    inference_procedure

Performs particle filter inference with rejuvenation.
"""
function inference_procedure(gm_args::Tuple,
                             obs::Vector{Gen.ChoiceMap},
                             particles::Int = 100)
    get_args(t) = (t, gm_args[2:3]...)

    # initialize particle filter
    state = Gen.initialize_particle_filter(model, get_args(0), EmptyChoiceMap(), particles)

    # Then increment through each observation step
    for (t, o) = enumerate(obs)
        # apply a rejuvenation move to each particle
        step_time = @elapsed begin
            for i=1:particles
                state.traces[i], _  = mh(state.traces[i], proposal, ())
            end
        
            Gen.maybe_resample!(state, ess_threshold=particles/2) 
            Gen.particle_filter_step!(state, get_args(t), (UnknownChange(), NoChange(), NoChange()), o)
        end

        if t % 10 == 0
            @printf "%s time steps completed (last step was %0.2f seconds)\n" t step_time
        end
    end

    # return the "unweighted" set of traces after t steps
    return Gen.sample_unweighted_traces(state, particles)
end

function main()

    t = 120 # 2 seconds of observations
    (gargs, obs, truth) = data_generating_procedure(t)

    choices = get_choices(truth)
    obs = Vector{Gen.ChoiceMap}(undef, t)
    for i = 1:t
        prefix = :kernel => i => :observe
        cm = choicemap()
        set_submap!(cm, prefix, get_submap(choices, prefix))
        obs[i] = cm
    end

    traces = inference_procedure(gargs, obs, 60)

    display(plot_traces(truth, traces))
    
    println("press enter to exit the program")
    readline()
    return nothing
end


main();
