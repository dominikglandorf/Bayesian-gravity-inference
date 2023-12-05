using Plots

include(joinpath(@__DIR__, "helpers.jl"))
include(joinpath(@__DIR__, "models.jl"))

ENV["GKSwstype"]="160" # fixes some plotting warnings
# set up scene and model parameters
client, obj_id = scene()
sim = BulletSim(;client=client)
obj = RigidBody(obj_id)
init_state = BulletState(sim, [obj])
gargs = (2, # number of steps (2s)
            sim,
            init_state)


# generate traces
function get_histogram(model, title)
    traces = [generate(model, gargs) for _ in 1:10000]

    # visualize the gravity  position of the objects across time
    gravity_logs = [t[:gravity] for (t, _) in traces]
    Plots.histogram(gravity_logs, title="Prior: $(title) model", xlabel="Gravity", ylabel="Probability (approx)", legend=false)
end
    
histograms = [get_histogram(m, title) for (m, title) in zip([model_baseline, model_switch], ["flat prior", "bimodal prior"])]
plt = plot(histograms..., layout=(2, 1), link=:x)

Plots.savefig(plt, "plots/priors.png")
