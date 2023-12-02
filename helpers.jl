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
    obj_velocity::NTuple{3, Float64} = (1.5, 0., 1.5)
    )

    # for debugging
    #client = @pycall pb.connect(pb.GUI)::Int64
    client = @pycall pb.connect(pb.DIRECT)::Int64

    pb.resetDebugVisualizerCamera(3, 0, 0, [0.0, 0.0, 0.0]; physicsClientId=client)
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
    obj_dims = [1, 0.01, .5]
    texture_id = pb.loadTexture("materials/dvd_texture.png")

    obj_col_id = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=obj_dims/2, physicsClientId=client)
    obj_id = pb.createMultiBody(baseCollisionShapeIndex=obj_col_id, basePosition=obj_position, physicsClientId=client)
    pb.changeVisualShape(obj_id, -1; textureUniqueId=texture_id)
    pb.changeDynamics(obj_id, -1; mass=1.0, restitution=1., lateralFriction=0., physicsClientId=client)
    
    #print(obj_id)
    pb.resetBaseVelocity(obj_id, linearVelocity=obj_velocity, physicsClientId=client)

    #pb.disconnect(client)

    (client, obj_id)
end


"""
    data_generating_procedure(t::Int64)

Create a trial (ground truth and observations) with `t` timepoints
"""
function data_generating_procedure(t::Int64)

    client, obj_id = scene()

    # configure simulator with the provided
    # client id
    sim = BulletSim(;client=client)
    # These are the objects of interest in the scene
    # (the rest is static)
    obj = RigidBody(obj_id)
    # Retrieve the default latents for the objects
    # as well as their initial positions
    # Note: alternative latents will be suggested by the `prior`
    init_state = BulletState(sim, [obj])
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
        addr = :kernel => i => :observations
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
# Inference
################################################################################

# this proposal function implements a truncated random walk for the both mass priors
@gen function proposal(tr::Gen.Trace)
    # get previous values from `tr`
    choices = get_choices(tr)
    prev_gravity = choices[:gravity]
    gravity ~ trunc_norm(prev_gravity, .01, -5., 5.)
    
    prev_vel_x = choices[:obj_prior => 1 => :start_x_vel]
    @trace(trunc_norm(prev_vel_x, .01, -3., 3.), :obj_prior => 1 => :start_x_vel)
    prev_vel_z = choices[:obj_prior => 1 => :start_z_vel]
    @trace(trunc_norm(prev_vel_z, .01, -3., 3.), :obj_prior => 1 => :start_z_vel)
    
end

################################################################################
# Visuals
################################################################################

function plot_trace(tr::Gen.Trace, title="Trajectory")
    (t, _, _) = get_args(tr)
    # get the prior choice for the two masses
    choices = get_choices(tr)
    gravity = round(choices[:gravity], digits=2)

    # get the x and y positions
    states = get_retval(tr)
    xs = map(st -> st.kinematics[1].position[1], states)
    ys = map(st -> st.kinematics[1].position[3], states)

    # return plot
    plot(xs, ys, title=title, labels="gravity: $(gravity)", xlabel="x", ylabel="y", ylim=(-2.0,2.0), xlim=(-2.5,2.5))
end

"""
plot_traces(truth::Gen.DynamicDSLTrace, traces::Vector{Gen.DynamicDSLTrace})

Display the observed and final simulated trajectory as well as distributions for latents and the score
"""
function plot_traces(truth::Gen.DynamicDSLTrace, traces::Vector{Gen.DynamicDSLTrace})
    t = length(truth[:kernel])
    
    observed_plt = plot_trace(truth, "True trajectory (Gravity: $(round(truth[:gravity], digits=2)))")
    scores = [get_score(t) for t in traces]
    _, max_index = findmax(scores)
    simulated_plt = plot_trace(traces[max_index], "Trajectory in best trace")

    num_traces = length(traces)
    gravity_logs = [t[:gravity] for t in traces]

    scores_plt = scatter(1:num_traces, scores, title="Scores", xlabel="trace number", ylabel="log score")
    gravity_plt = Plots.histogram(1:num_traces, gravity_logs, title="Posterior estimate of gravity", legend=false)
    plot(observed_plt, simulated_plt, gravity_plt, scores_plt,  size=(1200, 800))
end
