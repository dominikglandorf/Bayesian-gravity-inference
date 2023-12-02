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
    # set up scene and model parameters
    client, obj_id = scene(0.)
    sim = BulletSim(;client=client)
    obj = RigidBody(obj_id)
    init_state = BulletState(sim, [obj])
    gargs = (120, # number of steps (2s)
             sim,
             init_state)

    # generate traces
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
