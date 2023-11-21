#!/bin/bash

# Create the frames
julia --project=. render_sim.jl

if [ $? -eq 0 ]; then
    cd render

    ffmpeg -framerate 30 -i frame%d.png -c:v libx264 -pix_fmt yuv420p output_video.mp4

    if [ $? -eq 0 ]; then
        echo "Video created successfully."
    else
        echo "Error occurred in creating the video."
    fi
else
    echo "Error occurred in executing the Julia script."
fi
