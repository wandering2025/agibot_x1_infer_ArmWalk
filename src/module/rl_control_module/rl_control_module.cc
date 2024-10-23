// Copyright (c) 2023, AgiBot Inc.
// All rights reserved.
#include "rl_control_module/rl_control_module.h"
#include <yaml-cpp/yaml.h>
#include "aimrt_module_ros2_interface/channel/ros2_channel.h"
#include "my_ros2_proto/msg/joint_command.hpp"
#include "rl_control_module/control_mode.h"
#include "rl_control_module/global.h"
#include "rl_control_module/utilities.h"

#include <geometry_msgs/msg/twist.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

namespace xyber_x1_infer::rl_control_module {

bool RlControlModule::Initialize(aimrt::CoreRef core) {
  // Save aimrt framework handle
  core_ = core;

  SetLogger(core_.GetLogger());

  auto file_path = core_.GetConfigurator().GetConfigFilePath();
  try {
    if (!file_path.empty()) {
      YAML::Node cfg_node = YAML::LoadFile(file_path.data());

      freq_ = cfg_node["freq"].as<int32_t>();
      use_sim_handles_ = cfg_node["use_sim_handles"].as<bool>();

      // ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ init controller ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
      ControlCfg control_conf;
      control_conf.joint_conf["init_state"] =
          cfg_node["control_conf"]["joint_conf"]["init_state"].as<std::map<std::string, double>>();
      control_conf.joint_conf["stiffness"] =
          cfg_node["control_conf"]["joint_conf"]["stiffness"].as<std::map<std::string, double>>();
      control_conf.joint_conf["damping"] =
          cfg_node["control_conf"]["joint_conf"]["damping"].as<std::map<std::string, double>>();
      control_conf.ordered_joint_name.clear();
      for (auto iter = cfg_node["control_conf"]["joint_conf"]["init_state"].begin();
           iter != cfg_node["control_conf"]["joint_conf"]["init_state"].end(); iter++) {
        control_conf.ordered_joint_name.push_back(iter->first.as<std::string>());
      }

      control_conf.walk_step_conf.action_scale =
          cfg_node["control_conf"]["walk_step_conf"]["action_scale"].as<double>();
      control_conf.walk_step_conf.decimation =
          cfg_node["control_conf"]["walk_step_conf"]["decimation"].as<int32_t>();
      control_conf.walk_step_conf.cycle_time =
          cfg_node["control_conf"]["walk_step_conf"]["cycle_time"].as<double>();
      control_conf.walk_step_conf.sw_mode =
          cfg_node["control_conf"]["walk_step_conf"]["sw_mode"].as<bool>();
      control_conf.walk_step_conf.cmd_threshold =
          cfg_node["control_conf"]["walk_step_conf"]["cmd_threshold"].as<double>();
      control_conf.obs_scales.lin_vel =
          cfg_node["control_conf"]["obs_scales"]["lin_vel"].as<double>();
      control_conf.obs_scales.ang_vel =
          cfg_node["control_conf"]["obs_scales"]["ang_vel"].as<double>();
      control_conf.obs_scales.dof_pos =
          cfg_node["control_conf"]["obs_scales"]["dof_pos"].as<double>();
      control_conf.obs_scales.dof_vel =
          cfg_node["control_conf"]["obs_scales"]["dof_vel"].as<double>();
      control_conf.obs_scales.quat = cfg_node["control_conf"]["obs_scales"]["quat"].as<double>();
      control_conf.onnx_conf.policy_file =
          cfg_node["control_conf"]["onnx_conf"]["policy_file"].as<std::string>();
      control_conf.onnx_conf.actions_size =
          cfg_node["control_conf"]["onnx_conf"]["actions_size"].as<int32_t>();
      control_conf.onnx_conf.obs_size =
          cfg_node["control_conf"]["onnx_conf"]["observations_size"].as<int32_t>();
      control_conf.onnx_conf.num_hist =
          cfg_node["control_conf"]["onnx_conf"]["num_hist"].as<int32_t>();
      control_conf.onnx_conf.obs_clip =
          cfg_node["control_conf"]["onnx_conf"]["observations_clip"].as<double>();
      control_conf.onnx_conf.actions_clip =
          cfg_node["control_conf"]["onnx_conf"]["actions_clip"].as<double>();

      rl_controller_ = std::make_unique<RLController>(control_conf, use_sim_handles_);
      // ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

      // ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ init  subscriber ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
      joy_start_sub_ =
          core_.GetChannelHandle().GetSubscriber(cfg_node["sub_joy_start_name"].as<std::string>());
      joy_zero_sub_ =
          core_.GetChannelHandle().GetSubscriber(cfg_node["sub_joy_zero_name"].as<std::string>());
      joy_stand_sub_ =
          core_.GetChannelHandle().GetSubscriber(cfg_node["sub_joy_stand_name"].as<std::string>());
      joy_walk_sub_ =
          core_.GetChannelHandle().GetSubscriber(cfg_node["sub_joy_work_name"].as<std::string>());
      joy_vel_sub_ =
          core_.GetChannelHandle().GetSubscriber(cfg_node["sub_joy_vel_name"].as<std::string>());
      imu_sub_ =
          core_.GetChannelHandle().GetSubscriber(cfg_node["sub_imu_data_name"].as<std::string>());
      joint_state_sub_ = core_.GetChannelHandle().GetSubscriber(
          cfg_node["sub_joint_state_name"].as<std::string>());
      joint_cmd_pub_ =
          core_.GetChannelHandle().GetPublisher(cfg_node["pub_joint_cmd_name"].as<std::string>());

      last_set_start_time_ = high_resolution_clock::now();
      last_set_zero_time_ = high_resolution_clock::now();
      last_set_stand_time_ = high_resolution_clock::now();
      last_set_walk_time_ = high_resolution_clock::now();

      bool ret = aimrt::channel::Subscribe<std_msgs::msg::Float32>(
          joy_start_sub_, [this](const std::shared_ptr<const std_msgs::msg::Float32>& msg) {
            if (!Throttler(high_resolution_clock::now(), last_set_start_time_, 1000ms)) {
              return;
            }
            if (ControlMode::IDLE != rl_controller_->GetMode()) {
              rl_controller_->SetMode(ControlMode::IDLE);
              AIMRT_INFO("[ZERO] -> [IDLE]");
            } else {
              rl_controller_->SetMode(ControlMode::ZERO);
              AIMRT_INFO("[IDLE] -> [ZERO]");
            }
          });

      ret = aimrt::channel::Subscribe<std_msgs::msg::Float32>(
          joy_zero_sub_, [this](const std::shared_ptr<const std_msgs::msg::Float32>& msg) {
            if (!Throttler(high_resolution_clock::now(), last_set_zero_time_, 1000ms)) {
              return;
            }
            if (ControlMode::STAND == rl_controller_->GetMode()) {
              rl_controller_->SetMode(ControlMode::ZERO);
              AIMRT_INFO("[STAND] -> [ZERO]");
            }
          });

      ret = aimrt::channel::Subscribe<std_msgs::msg::Float32>(
          joy_stand_sub_, [this](const std::shared_ptr<const std_msgs::msg::Float32>& msg) {
            if (!Throttler(high_resolution_clock::now(), last_set_stand_time_, 1000ms)) {
              return;
            }
            if (ControlMode::ZERO == rl_controller_->GetMode()) {
              rl_controller_->SetMode(ControlMode::STAND);
              AIMRT_INFO("[ZERO] -> [STAND]");
            } else if (ControlMode::WALK == rl_controller_->GetMode()) {
              rl_controller_->SetMode(ControlMode::STAND);
              AIMRT_INFO("[WALK] -> [STAND]");
            }
          });

      ret = aimrt::channel::Subscribe<std_msgs::msg::Float32>(
          joy_walk_sub_, [this](const std::shared_ptr<const std_msgs::msg::Float32>& msg) {
            if (!Throttler(high_resolution_clock::now(), last_set_walk_time_, 1000ms)) {
              return;
            }
            if (ControlMode::STAND == rl_controller_->GetMode()) {
              rl_controller_->SetMode(ControlMode::WALK);
              AIMRT_INFO("[STAND] -> [WALK]");
            }
          });

      ret = aimrt::channel::Subscribe<geometry_msgs::msg::Twist>(
          joy_vel_sub_, [this](const std::shared_ptr<const geometry_msgs::msg::Twist>& msg) {
            rl_controller_->SetCmdData(*msg);
          });

      ret = aimrt::channel::Subscribe<sensor_msgs::msg::Imu>(
          imu_sub_, [this](const std::shared_ptr<const sensor_msgs::msg::Imu>& msg) {
            rl_controller_->SetImuData(*msg);
          });

      ret &= aimrt::channel::Subscribe<sensor_msgs::msg::JointState>(
          joint_state_sub_, [this](const std::shared_ptr<const sensor_msgs::msg::JointState>& msg) {
            rl_controller_->SetJointStateData(*msg);
          });
      AIMRT_CHECK_ERROR_THROW(ret, "Subscribe failed.");
      // ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

      // ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ init  publisher ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
      executor_ = core_.GetExecutorManager().GetExecutor("rl_control_pub_thread");
      AIMRT_CHECK_ERROR_THROW(executor_, "Can not get executor 'rl_control_pub_thread'.");
      if (use_sim_handles_) {
        aimrt::channel::RegisterPublishType<std_msgs::msg::Float64MultiArray>(joint_cmd_pub_);
      } else {
        aimrt::channel::RegisterPublishType<my_ros2_proto::msg::JointCommand>(joint_cmd_pub_);
      }
      // ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
    }
  } catch (const std::exception& e) {
    AIMRT_ERROR("Init failed, {}", e.what());
    return false;
  }
  AIMRT_INFO("Init succeeded.");
  return true;
}

bool RlControlModule::Start() {
  AIMRT_INFO("thread safe [{}]", executor_.ThreadSafe());
  try {
    executor_.Execute([this]() { MainLoop(); });
    AIMRT_INFO("Started succeeded.");
  } catch (const std::exception& e) {
    AIMRT_ERROR("Start failed, {}", e.what());
    return false;
  }
  return true;
}

void RlControlModule::Shutdown() { run_flag_.store(false); }

bool RlControlModule::MainLoop() {
  try {
    while (run_flag_) {
      if (rl_controller_->IsReady()) {
        break;
      }
      AIMRT_INFO("Waiting for the sim/real to start!!!.");
      std::this_thread::sleep_for(1000ms);
    }

    AIMRT_INFO("Start MainLoop.");
    auto const period = nanoseconds(1'000'000'000 / freq_);
    time_point<high_resolution_clock, nanoseconds> next_iteration_time =
        high_resolution_clock::now();
    while (run_flag_) {
      next_iteration_time += period;
      std::this_thread::sleep_until(next_iteration_time);

      if (use_sim_handles_) {
        std_msgs::msg::Float64MultiArray msg;
        rl_controller_->GetJointCmdData(msg);
        aimrt::channel::Publish<std_msgs::msg::Float64MultiArray>(joint_cmd_pub_, msg);
      } else {
        my_ros2_proto::msg::JointCommand msg;
        rl_controller_->GetJointCmdData(msg);
        aimrt::channel::Publish<my_ros2_proto::msg::JointCommand>(joint_cmd_pub_, msg);
      }
    }
    AIMRT_INFO("Exit MainLoop.");
  } catch (const std::exception& e) {
    AIMRT_ERROR("Exit MainLoop with exception, {}", e.what());
  }

  return true;
}

}  // namespace xyber_x1_infer::rl_control_module
