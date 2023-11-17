using PhyBullet

################################################################################
# Generative Models
################################################################################

@gen function prior(element::RigidBody, client::Int)
    start_x_vel ~ normal(0., 1.)
    start_z_vel ~ normal(0., 1.)
    #pb.resetBaseVelocity(element.object_id, linearVelocity=[start_x_vel, 0., start_z_vel]; physicsClientId = client)
    return (start_x_vel, start_z_vel)
end

@gen function observe(k::RigidBodyState)
    pos = k.position # XYZ position
    # add noise to position
    obs = @trace(broadcasted_normal(pos, 0.01), :position)
    return obs
end

@gen function kernel(t::Int, prev_state::BulletState, sim::BulletSim)
    next_state::BulletState = PhySMC.step(sim, prev_state)
    observations ~ Gen.Map(observe)(next_state.kinematics)
    return next_state
end

@gen function model(t::Int, sim::BulletSim, template::BulletState)
    # sample new mass and restitution for objects
    obj_prior ~ Gen.Map(prior)(template.elements, fill(sim.client, length(template.elements)))
    
    gravity ~ normal(0., .5)
    pb.setGravity(0, 0, gravity; physicsClientId = sim.client)
    pb.resetBaseVelocity(6, linearVelocity=[.1, 0., .2]; physicsClientId = sim.client)

    # simulate `t` timesteps
    states = @trace(Gen.Unfold(kernel)(t, template, sim), :kernel)
    return states
end
