using Plots
ENV["GKSwstype"]="160" # fixes some plotting warnings

include(joinpath(@__DIR__, "helpers.jl"))
include(joinpath(@__DIR__, "models.jl"))
include(joinpath(@__DIR__, "particle_filter.jl"))

# generate data
t = 20
pred_t = 200
(gargs, obs, truth) = data_generating_procedure(t, .75, .5, 0)

function get_trajectory(states)
    # get the x and y positions
    xs = map(st -> st.kinematics[1].position[1], states)
    ys = map(st -> st.kinematics[1].position[3], states)
    return xs, ys
end

# plot observation
truth_states = get_retval(truth)
xs, ys = get_trajectory(truth_states)
plt = plot(xs, ys, xlabel="x", ylabel="y", ylim=(-1.8, 1.8), xlim=(-2.5,2.5), color="red", label="Observation")

# plot true trajectory in blue
next_states = Gen.Unfold(kernel)(pred_t, truth_states[t], gargs[2])
true_xs, true_ys = get_trajectory(next_states)

# do inference
burn_in = 1
choices = get_choices(truth)
obs = Vector{Gen.ChoiceMap}(undef, t)
for i = 1:t
    prefix = :kernel => i => :observations
    cm = choicemap()
    set_submap!(cm, prefix, get_submap(choices, prefix))
    obs[i] = cm
end
traces = inference_procedure(gargs, obs, 200)
posterior_gravity = [t[:gravity] for t in traces[burn_in:end]]
plt_post = plot_traces(truth, traces)
Plots.savefig(plt_post, "plots/posterior.png")

# do prediction with latent posteriors

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