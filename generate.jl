using Gen
using PyCall
using PhySMC
using PhyBullet
using Accessors
using Plots
using Revise

include(joinpath(@__DIR__, "helpers.jl"))
include(joinpath(@__DIR__, "models.jl"))

function main()

    # start with a ball above a table
    client, obj_id = scene(0.)

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
    gargs = (120, # number of steps (2s)
             sim,
             init_state)

    # execute `model`
    traces = [generate(model, gargs, choicemap(:gravity => 0)) for _ in 1:4]
    for (t, _) in traces
        display(t[:obj_prior=>1=>:start_x_vel])
        display(t[:obj_prior=>1=>:start_z_vel])
    end

    # visualize the x  position of the objects across time
    
    plts = plot([plot_trace(trace) for (trace, _) in traces]...)
    display(plts);

    println("press enter to exit the program")
    readline()
    pb.disconnect(client)
    return nothing
end


main();
