using PhyBullet

################################################################################
# Generative Models
################################################################################

@gen function prior(element_state::RigidBodyState)
    start_x_vel ~ normal(0., 1.)
    start_z_vel ~ normal(0., 1.)
    new_state = setproperties(element_state; linear_vel=[start_x_vel, 0, start_z_vel])
    return new_state
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

@gen function model_baseline(t::Int, sim::BulletSim, template::BulletState)
    obj_prior ~ Gen.Map(prior)(template.kinematics)
    init_state = setproperties(template; kinematics = obj_prior)
    
    gravity ~ normal(-0.1, .25)
    pb.setGravity(0, 0, gravity; physicsClientId = sim.client)

    # simulate `t` timesteps
    states = @trace(Gen.Unfold(kernel)(t, init_state, sim), :kernel)
    return states
end

@gen function model_switch(t::Int, sim::BulletSim, template::BulletState)
    obj_prior ~ Gen.Map(prior)(template.kinematics)
    init_state = setproperties(template; kinematics = obj_prior)

    gravity_present ~ bernoulli(0.5)
    if gravity_present
        gravity ~ normal(-0.1, .025)
    else
        gravity ~ normal(0., .01)
    end
    
    pb.setGravity(0, 0, gravity; physicsClientId = sim.client)

    # simulate `t` timesteps
    states = @trace(Gen.Unfold(kernel)(t, init_state, sim), :kernel)
    return states
end
