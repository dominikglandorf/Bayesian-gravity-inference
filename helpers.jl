using Revise
using Gen
using PyCall
using PhySMC
using PhyBullet
using Accessors

################################################################################
# Scene
################################################################################

function scene(
    gravity::Float64=-10.,
    obj_position::NTuple{3, Float64} = (0.0, 0.0, 0.),
    obj_velocity::NTuple{3, Float64} = (0.15, 0., 0.1)
    )

    # for debugging
    client = @pycall pb.connect(pb.GUI)::Int64
    #client = @pycall pb.connect(pb.DIRECT)::Int64

    pb.resetDebugVisualizerCamera(6, 0, 0, [0.0, 0.0, 0.0]; physicsClientId=client)
    pb.setGravity(0, 0, gravity; physicsClientId = client)
    
    #  add walls
    wall_dims = [ # Width, length, height
        [6.0, 0.1, 4.0],
        [6.0, 0.1, 4.0],
        [0.01, 0.2, 4.0],
        [0.01, 0.2, 4.0],
        [6.0, 0.2, 0.01],
        [6.0, 0.2, 0.01]
    ] 
    wall_positions = [
        [0, 0.1, 0], # Back Wall
        [0, -0.1, 0], # Front Wall
        [-3.0, 0.0, 0.0],  # Left Wall
        [3.0, 0.0, 0.0],  # Right Wall
        [0, 0.0, 2.0],  # Top Wall
        [0, 0.0, -2.0],  # Bottom Wall

    ]
    wall_colors = [
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1]
    ]
    for (dims, pos, col) in zip(wall_dims, wall_positions, wall_colors)
        wall_col_id = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=dims./2, physicsClientId=client)
        wall_obj_id = pb.createMultiBody(baseCollisionShapeIndex=wall_col_id, basePosition=pos, physicsClientId=client)
        pb.changeDynamics(wall_obj_id, -1; mass=0., restitution=1., lateralFriction=0., physicsClientId=client)
        pb.changeVisualShape(wall_obj_id, -1, rgbaColor=col, physicsClientId=client)
    end

    # add an object
    obj_dims = [1, 0.01, 0.7]
    texture_id = pb.loadTexture("materials/DVD_logo_lila.png")

    obj_col_id = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=obj_dims/2, physicsClientId=client)
    obj_id = pb.createMultiBody(baseCollisionShapeIndex=obj_col_id, basePosition=obj_position, physicsClientId=client)
    pb.changeVisualShape(obj_id, -1; textureUniqueId=texture_id)
    pb.changeDynamics(obj_id, -1; mass=1.0, restitution=1., lateralFriction=0., physicsClientId=client)
    pb.resetBaseVelocity(obj_id, linearVelocity=obj_velocity)

    pb.disconnect(client)

    (client, obj_id)
end


"""
    data_generating_procedure(t::Int64)

Create a trial (ground truth and observations) with `t` timepoints
"""
function data_generating_procedure(t::Int64)

    client, obj_ramp_id, obj_table_id = ramp()

    # configure simulator with the provided
    # client id
    sim = BulletSim(;client=client)
    # These are the objects of interest in the scene
    # (the rest is static)
    obj_ramp = RigidBody(obj_ramp_id)
    obj_table = RigidBody(obj_table_id)
    # Retrieve the default latents for the objects
    # as well as their initial positions
    # Note: alternative latents will be suggested by the `prior`
    init_state = BulletState(sim, [obj_ramp, obj_table])
    # arguments for `model`
    gargs = (t, # number of steps
             sim,
             init_state)

    # execute `model`
    trace, _ = Gen.generate(model, gargs)
    choices = get_choices(trace)
    # extract noisy positions
    obs = Gen.choicemap()
    for i = 1:t
        addr = :kernel => i => :observe
        _choices = Gen.get_submap(choices, addr)
        Gen.set_submap!(obs, addr, _choices)
    end
    
    return (gargs, obs, trace)

end

################################################################################
# Distributions
################################################################################

struct TruncNorm <: Gen.Distribution{Float64} end
const trunc_norm = TruncNorm()
function Gen.random(::TruncNorm, mu::U, noise::T, low::T, high::T) where {U<:Real,T<:Real}
    d = Distributions.Truncated(Distributions.Normal(mu, noise),
                                low, high)
    return Distributions.rand(d)
end;
function Gen.logpdf(::TruncNorm, x::Float64, mu::U, noise::T, low::T, high::T) where {U<:Real,T<:Real}
    d = Distributions.Truncated(Distributions.Normal(mu, noise),
                                low, high)
    return Distributions.logpdf(d, x)
end;

################################################################################
# Generative Model
################################################################################

@gen function prior(ls::RigidBodyLatents)
    mass= @trace(gamma(1.2, 10.), :mass)
    println(@which setproperties(ls.data;
    mass = mass))
    new_ls = setproperties(ls.data;
                           mass = mass)
    new_latents = RigidBodyLatents(new_ls)
    return new_latents
end

@gen function observe(k::RigidBodyState)
    pos = k.position # XYZ position
    # add noise to position
    obs = @trace(broadcasted_normal(pos, 0.01), :position)
    return obs
end

@gen function kernel(t::Int, prev_state::BulletState, sim::BulletSim)
    # use of PhySMC.step
    next_state::BulletState = PhySMC.step(sim, prev_state)
    # elem state could be a different type
    # here we have two `RigidBody` elements
    # so  `next_state.kinematics = [RigidBodyState, RigidBodyState]`
    obs = @trace(Gen.Map(observe)(next_state.kinematics), :observe)
    return next_state
end

@gen function model(t::Int, sim::BulletSim, template::BulletState)
    # sample new mass and restitution for objects
    latents = @trace(Gen.Map(prior)(template.latents), :prior)
    init_state = setproperties(template; latents = latents)
    # simulate `t` timesteps
    states = @trace(Gen.Unfold(kernel)(t, init_state, sim), :kernel)
    return states
end

################################################################################
# Inference
################################################################################

# this proposal function implements a truncated random walk for the both mass priors
@gen function proposal(tr::Gen.Trace)
    # get previous values from `tr`
    choices = get_choices(tr)
    prev_mass_1 = choices[:prior => 1 => :mass]
    prev_mass_2 = choices[:prior => 2 => :mass]
    
    # sample new values conditioned on the old ones
    mass_1 = {:prior => 1 => :mass} ~ trunc_norm(prev_mass_1, .25, 0., Inf)
    mass_2 = {:prior => 2 => :mass} ~ trunc_norm(prev_mass_2, .25, 0., Inf)
    
    # the return of this function is not
    # neccessary but could be useful
    # for debugging.
    return (mass_1, mass_2)
end

################################################################################
# Visuals
################################################################################

function plot_trace(tr::Gen.Trace, title="Trajectory")
    (t, _, _) = get_args(tr)
    # get the prior choice for the two masses
    choices = get_choices(tr)
    masses = [round(choices[:prior => i => :mass], digits=2) for i in 1:2]

    # get the x positions
    states = get_retval(tr)
    xs = [map(st -> st.kinematics[i].position[1], states) for i = 1:2]

    # return plot
    plot(1:t, xs, title=title, labels=["ramp: $(masses[1])" "table: $(masses[2])"], xlabel="t", ylabel="x")
end

"""
plot_traces(truth::Gen.DynamicDSLTrace, traces::Vector{Gen.DynamicDSLTrace})

Display the observed and final simulated trajectory as well as distributions for latents and the score
"""
function plot_traces(truth::Gen.DynamicDSLTrace, traces::Vector{Gen.DynamicDSLTrace})
    t = length(truth[:kernel])
    
    observed_plt = plot_trace(truth, "True trajectory")
    simulated_plt = plot_trace(last(traces), "Last trace")

    num_traces = length(traces)
    mass_logs = [[t[:prior => i => :mass] for t in traces] for i in 1:2]
    scores = [get_score(t) for t in traces]

    scores_plt = plot(1:num_traces, scores, title="Scores", xlabel="trace number", ylabel="log score")
    mass_plts = [Plots.histogram(1:num_traces, mass_logs[i], title="Mass $(i == 1 ? "Ramp object" : "Table object")", legend=false) for i in 1:2]
    ratio_plt = Plots.histogram(1:num_traces, mass_logs[1]./mass_logs[2], title="mass ramp object / mass table object", legend=false)
    plot(observed_plt, simulated_plt, mass_plts..., scores_plt, ratio_plt,  size=(1200, 800))
end
