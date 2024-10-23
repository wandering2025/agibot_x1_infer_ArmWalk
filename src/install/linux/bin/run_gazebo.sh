#!/bin/bash

if [ -f ./install/share/ros2_plugin_proto/local_setup.bash ]; then
    source ./install/share/ros2_plugin_proto/local_setup.bash
elif [ -f ../share/ros2_plugin_proto/local_setup.bash ]; then
    source ../share/ros2_plugin_proto/local_setup.bash
fi

if [ -f ./install/share/robot_description/local_setup.bash ]; then
    source ./install/share/robot_description/local_setup.bash
elif [ -f ../share/robot_description/local_setup.bash ]; then
    source ../share/robot_description/local_setup.bash
fi

ros2 launch robot_description gazebo_sim.launch.py
