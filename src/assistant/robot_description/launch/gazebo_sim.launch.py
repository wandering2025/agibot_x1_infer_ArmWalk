import os
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, RegisterEventHandler
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command, ThisLaunchFileDir
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    declare_spawn_x_val = DeclareLaunchArgument('spawn_x_val', default_value='0.0', description='Spawn X Value')
    declare_spawn_y_val = DeclareLaunchArgument('spawn_y_val', default_value='0.0', description='Spawn Y Value')
    declare_spawn_z_val = DeclareLaunchArgument('spawn_z_val', default_value='0.63', description='Spawn Z Value')
    declare_spawn_yaw_val = DeclareLaunchArgument('spawn_yaw_val', default_value='0.00', description='Spawn Yaw Value')
    declare_use_sim_time = DeclareLaunchArgument('use_sim_time', default_value='true', description='Use sim time if true')
    declare_joy_teleop = DeclareLaunchArgument('joy_teleop', default_value='true', description='Set to "false" not to run joy teleop')

    # rviz
    rviz2_node = Node(
        package='rviz2',
        executable='rviz2',
        output='screen',
        arguments=['-d', [os.path.join(get_package_share_directory('robot_description'), 'rviz', 'urdf.rviz')]]
    )

    # robot_state_publisher
    model_path = os.path.join(get_package_share_directory("robot_description"), "urdf/robot.xacro")
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': Command(['xacro ', model_path]), 'use_sim_time': True}]
    )

    # gazebo
    gazebo_node = IncludeLaunchDescription(PythonLaunchDescriptionSource([os.path.join(get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')]))

    spawn_entity_node = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description',
                   '-entity', 'zy',
                   '-x', LaunchConfiguration('spawn_x_val'),
                   '-y', LaunchConfiguration('spawn_y_val'),
                   '-z', LaunchConfiguration('spawn_z_val'),
                   '-Y', LaunchConfiguration('spawn_yaw_val')],
                   output='screen')

    # joint_state_broadcaster
    joint_state_broadcaster_node = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster"],
        output='screen',
    )

    # controller 
    controller_node = Node(
        package="controller_manager",
        executable="spawner",
        arguments=['effort_controller'],
    )

    return LaunchDescription([
        declare_spawn_x_val,
        declare_spawn_y_val,
        declare_spawn_z_val,
        declare_spawn_yaw_val,
        declare_use_sim_time,
        declare_joy_teleop,
        # rviz2_node,
        robot_state_publisher_node,
        gazebo_node,
        spawn_entity_node,
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=spawn_entity_node,
                on_exit=[joint_state_broadcaster_node])),
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=joint_state_broadcaster_node,
                on_exit=[controller_node])),
    ])
    
    
