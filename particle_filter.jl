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
                             particles::Int = 50,
                             gen_model=model_baseline)
    
    get_args(t) = (t, gm_args[2:3]...)
    state = Gen.initialize_particle_filter(gen_model, get_args(0), EmptyChoiceMap(), particles)

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
        end
    end

    return Gen.sample_unweighted_traces(state, 5*particles)
end

function main()

    t = 10
    (gargs, obs, truth) = data_generating_procedure(t, 1.5, 1., -0.981)

    # plot observation in red and true trajectory in blue
    truth_states = get_retval(truth)
    xs, ys = get_trajectory(truth_states)
    plt = plot(xs, ys, xlabel="x", ylabel="y", ylim=(-0.1, 1.8), xlim=(-0.1,2.5), color="red", label="Observation")

    # plot true trajectory in blue
    pred_t = 100 - t
    next_states = Gen.Unfold(kernel)(pred_t, truth_states[t], gargs[2])
    true_xs, true_ys = get_trajectory(next_states)
    
    choices = get_choices(truth)
    obs = Vector{Gen.ChoiceMap}(undef, t)
    for i = 1:t
        prefix = :kernel => i => :observations
        cm = choicemap()
        set_submap!(cm, prefix, get_submap(choices, prefix))
        obs[i] = cm
    end

    traces = inference_procedure(gargs, obs, 1000, model_switch)
    plt_post = plot_traces(truth, traces)
    Plots.savefig(plt_post, "plots/posterior.png")

    num_samples_vis = 100
    trace_samples = rand(traces, num_samples_vis)
    for trace in trace_samples
        pb.setGravity(0, 0, trace[:gravity]; physicsClientId = gargs[2].client)
        local states = get_retval(trace)
        setproperties(truth_states[t].kinematics[1]; linear_vel=states[t].kinematics[1].linear_vel)
        local next_states = Gen.Unfold(kernel)(pred_t, truth_states[t], gargs[2])
        local pred_xs, pred_ys = get_trajectory(next_states)
        plot!(plt, pred_xs, pred_ys, color="gray", line=(1, 0.5), label="")
    end

    plot!(plt, true_xs, true_ys, color="blue", label="True trajectory")
    Plots.savefig(plt, "plots/prediction.png")
end


main();
