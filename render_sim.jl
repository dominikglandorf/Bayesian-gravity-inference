using PhyBullet
using Images
using FileIO
using VideoIO

include(joinpath(@__DIR__, "helpers.jl"))

# SETUP SCENE
client, obj_id = scene()
sim = BulletSim(;client=client)
obj = RigidBody(obj_id)
state = BulletState(sim, [obj])
t = 60

pb.setGravity(0, 0, -.5; physicsClientId = sim.client)

# SIMULATE AND SAVE IMAGES
for i in 1:t
    global state = PhySMC.step(sim, state)
    viewMatrix = pb.computeViewMatrix(
        cameraEyePosition=[0, -3, 0],
        cameraTargetPosition=[0, 0, 0],
        cameraUpVector=[0, 0, 1]
    )
    projMatrix = pb.computeProjectionMatrixFOV(
        fov=60.,
        aspect=float(640) / 480,
        nearVal=0.1,
        farVal=100
    )
    width, height, rgbImg, depthImg, segImg = pb.getCameraImage(
        640, 480, viewMatrix, projMatrix
    )
   
    img_array_float = float32.(rgbImg) / 255.0
    img = colorview(RGBA, permutedims(img_array_float, (3, 1, 2)))
    save("render/frame$(i).png", img)
end