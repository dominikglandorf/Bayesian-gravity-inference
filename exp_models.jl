using Plots
ENV["GKSwstype"]="160" # fixes some plotting warnings

include(joinpath(@__DIR__, "helpers.jl"))
include(joinpath(@__DIR__, "models.jl"))
include(joinpath(@__DIR__, "particle_filter.jl"))

function get_posterior_statistics(gargs, obs, model, num_particles = 100)
    traces = inference_procedure(gargs, obs, num_particles, model)
    posterior_gravity = [t[:gravity] for t in traces]
    return abs(mean(posterior_gravity)), std(posterior_gravity)
end

function analyse_inference(t, true_gravity, gen_model, pred_t = 200, num_runs = 10)
    # generate data
    (gargs, obs, truth) = data_generating_procedure(t, .75, .5, true_gravity)

    # do inference
    choices = get_choices(truth)
    obs = Vector{Gen.ChoiceMap}(undef, t)
    for i = 1:t
        prefix = :kernel => i => :observations
        cm = choicemap()
        set_submap!(cm, prefix, get_submap(choices, prefix))
        obs[i] = cm
    end

    posterior_stats = [get_posterior_statistics(gargs, obs, gen_model) for i in 1:num_runs]
    return map(x -> x[1], posterior_stats), map(x -> x[2], posterior_stats)
end

gravity_conditions = [-.1, -.05, 0.]
ts = [10, 20, 30]
models = zip(["baseline", "switch"], [model_baseline, model_switch])

for gravity in gravity_conditions
    @show gravity
    for (name, model) in models
        @show name
        for t in ts
            @show t
            biases, stds = analyse_inference(t, gravity, model)
            plt = Plots.histogram(biases, title="Biases, Gravity=$(gravity), Model=$(name), t=$(t)", xlabel="Bias", ylabel="Frequency", legend=false)
            Plots.savefig(plt, "plots/biases_gravity_$(gravity)_model_$(name)_t_$(t).png")
            @show round(mean(biases); digits=3)
            @show round(mean(stds); digits=3)
        end
    end
end
