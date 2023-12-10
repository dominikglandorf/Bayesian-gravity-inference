using Plots
ENV["GKSwstype"]="160" # fixes some plotting warnings

include(joinpath(@__DIR__, "helpers.jl"))
include(joinpath(@__DIR__, "models.jl"))
include(joinpath(@__DIR__, "particle_filter.jl"))

function get_posterior_statistics(gargs, obs, model, truth, title, num_particles = 1000)
    traces = inference_procedure(gargs, obs, num_particles, model)
    plt_post = plot_traces(truth, traces)
    Plots.savefig(plt_post, "plots/posterior_$(title).png")
    posterior_gravity = [t[:gravity] for t in traces]
    @show median(posterior_gravity)
    @show abs(median(posterior_gravity))
    @show truth[:gravity]
    return abs(median(posterior_gravity) - truth[:gravity]), std(posterior_gravity)
end

function analyse_inference(t, true_gravity, gen_model, title, pred_t = 200, num_runs = 5)
    # generate data
    (gargs, obs, truth) = data_generating_procedure(t, 1.5, 1., true_gravity)

    # do inference
    choices = get_choices(truth)
    obs = Vector{Gen.ChoiceMap}(undef, t)
    for i = 1:t
        prefix = :kernel => i => :observations
        cm = choicemap()
        set_submap!(cm, prefix, get_submap(choices, prefix))
        obs[i] = cm
    end

    posterior_stats = [get_posterior_statistics(gargs, obs, gen_model, truth, "$(title)_$(i)", 250) for i in 1:num_runs]
    return map(x -> x[1], posterior_stats), map(x -> x[2], posterior_stats)
end

gravity_conditions = [-.981]
ts = [10, 30, 50]
models = zip(["switch"], [model_switch])

for gravity in gravity_conditions
    @show gravity
    for (name, model) in models
        @show name
        for t in ts
            @show t
            biases, stds = analyse_inference(t, gravity, model, "$(gravity)_model_$(name)_t_$(t)")
            plt = Plots.histogram(biases, title="Biases, Gravity=$(gravity), Model=$(name), t=$(t)", xlabel="Bias", ylabel="Frequency", legend=false)
            Plots.savefig(plt, "plots/biases_gravity_$(gravity)_model_$(name)_t_$(t).png")
            @show round(mean(biases); digits=3)
            @show round(mean(stds); digits=3)
        end
    end
end
