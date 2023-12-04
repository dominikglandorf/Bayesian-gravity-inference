using Gen
using Distributions
using Plots
using Printf
using Revise
ENV["GKSwstype"]="160" # fixes some plotting warnings

include(joinpath(@__DIR__, "helpers.jl"))
include(joinpath(@__DIR__, "models.jl"))

"""
    inference_procedure

Performs particle filter inference with rejuvenation.
"""
function inference_procedure(gm_args::Tuple,
                             obs::Vector{Gen.ChoiceMap},
                             particles::Int = 100)
    
    get_args(t) = (t, gm_args[2:3]...)
    state = Gen.initialize_particle_filter(model_switch, get_args(0), EmptyChoiceMap(), particles)

    for (t, o) = enumerate(obs)
        
        step_time = @elapsed begin
            for i=1:particles
                state.traces[i], _  = mh(state.traces[i], proposal, ())
            end
        
            Gen.maybe_resample!(state, ess_threshold=particles/2) 
            Gen.particle_filter_step!(state, get_args(t), (UnknownChange(), NoChange(), NoChange()), o)
        end

        if t % 10 == 0
            @printf "%s time steps completed (last step was %0.2f seconds)\n" t step_time
            gravities = [t[:gravity] for t in state.traces]
        end
    end

    return state.traces
end

function main()

    t = 60
    (gargs, obs, truth) = data_generating_procedure(t)

    choices = get_choices(truth)
    obs = Vector{Gen.ChoiceMap}(undef, t)
    for i = 1:t
        prefix = :kernel => i => :observations
        cm = choicemap()
        set_submap!(cm, prefix, get_submap(choices, prefix))
        obs[i] = cm
    end

    traces = inference_procedure(gargs, obs, 1000)    

    display(plot_traces(truth, traces))

    println("press enter to exit the program")
    readline()
    return nothing
end


#main();
