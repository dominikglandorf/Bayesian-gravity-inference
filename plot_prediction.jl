using Plots

include(joinpath(@__DIR__, "helpers.jl"))
include(joinpath(@__DIR__, "models.jl"))

t = 60
(gargs, obs, truth) = data_generating_procedure(t)

# do inference
# for now, let's assume we found the true latents
states = get_retval(truth)

# do prediction with latent posteriors

pb.setGravity(0, 0, -5.; physicsClientId = gargs[2].client)
setproperties(states[t]; kinematics = states[t].kinematics)
next_states = Gen.Unfold(kernel)(60, states[t], gargs[2])


function get_trajectory(states)
    # get the x and y positions
    xs = map(st -> st.kinematics[1].position[1], states)
    ys = map(st -> st.kinematics[1].position[3], states)
    return xs, ys
end

xs, ys = get_trajectory(states)
plt = plot(xs, ys, xlabel="x", ylabel="y", ylim=(-2.0,2.0), xlim=(-2.5,2.5), color="red", label="Observation")

pred_xs, pred_ys = get_trajectory(next_states)
plot!(plt, pred_xs, pred_ys, color="gray", label="Prediction")

display(plt)