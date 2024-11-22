// Copyright (c) 2023, AgiBot Inc.
// All rights reserved.
#include "rl_control_module/rl_controller.h"
#include <chrono>
#include <iostream>

using namespace std::chrono;
namespace xyber_x1_infer::rl_control_module {

RLController::RLController(const ControlCfg& control_conf, bool use_sim_handles)
    : control_conf_(control_conf),
      use_sim_handles_(use_sim_handles),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
  LoadModel();

  actions_.resize(control_conf_.onnx_conf.actions_size);
  observations_.resize(control_conf_.onnx_conf.obs_size * control_conf_.onnx_conf.num_hist);
  last_actions_.resize(control_conf_.onnx_conf.actions_size);
  last_actions_.setZero();

  propri_history_buffer_.resize(control_conf_.onnx_conf.obs_size *
                                control_conf_.onnx_conf.num_hist);

  init_joint_angles_.resize(control_conf_.onnx_conf.actions_size);
  for (size_t i = 0; i < control_conf_.onnx_conf.actions_size; ++i) {
    init_joint_angles_(i) =
        control_conf_.joint_conf["init_state"][control_conf_.ordered_joint_name[i]];
  }

  current_joint_angles_.resize(control_conf_.onnx_conf.actions_size);
  if (use_sim_handles_) {
    control_mode_ = ControlMode::ZERO;
  } else {
    control_mode_ = ControlMode::IDLE;
  }
  joint_name_index_.clear();
  loop_count_ = 0;
  low_pass_filters_.clear();
  for (size_t i = 0; i < control_conf_.onnx_conf.actions_size; ++i) {
    low_pass_filters_.emplace_back(100, 0.001);
  }

  sim_joint_cmd_.data.resize(control_conf_.onnx_conf.actions_size);

  real_joint_cmd_.name = control_conf_.ordered_joint_name;
  real_joint_cmd_.position.resize(control_conf_.onnx_conf.actions_size);
  real_joint_cmd_.velocity.resize(control_conf_.onnx_conf.actions_size);
  real_joint_cmd_.effort.resize(control_conf_.onnx_conf.actions_size);
  real_joint_cmd_.damping.resize(control_conf_.onnx_conf.actions_size);
  real_joint_cmd_.damping.resize(control_conf_.onnx_conf.actions_size);
  real_joint_cmd_.stiffness.resize(control_conf_.onnx_conf.actions_size);
}

void RLController::SetMode(const ControlMode control_mode) {
  control_mode_.store(control_mode);
  trans_mode_percent_ = 0.0;

  std::shared_lock<std::shared_mutex> lock(joint_state_mutex_);
  for (size_t j = 0; j < control_conf_.ordered_joint_name.size(); ++j) {
    current_joint_angles_(j) =
        joint_state_data_.position[joint_name_index_[control_conf_.ordered_joint_name[j]]];
  }
}

void RLController::SetCmdData(const geometry_msgs::msg::Twist joy_data) {
  std::unique_lock<std::shared_mutex> lock(joy_mutex_);
  joy_data_ = joy_data;
}

void RLController::SetImuData(const sensor_msgs::msg::Imu imu_data) {
  std::unique_lock<std::shared_mutex> lock(imu_mutex_);
  imu_data_ = imu_data;
}

void RLController::SetJointStateData(const sensor_msgs::msg::JointState joint_state_data) {
  std::unique_lock<std::shared_mutex> lock(joint_state_mutex_);
  joint_state_data_ = joint_state_data;

  if (joint_name_index_.empty()) {
    for (size_t i = 0; i < joint_state_data_.name.size(); ++i) {
      joint_name_index_[joint_state_data_.name[i]] = i;
    }
  }
}

ControlMode RLController::GetMode() { return control_mode_.load(std::memory_order_acquire); }

bool RLController::IsReady() {
  std::shared_lock<std::shared_mutex> lock(joint_state_mutex_);

  if (joint_name_index_.empty()) {
    return false;
  }
  return true;
}

void RLController::GetJointCmdData(std_msgs::msg::Float64MultiArray& joint_cmd) {
  Update();
  joint_cmd = sim_joint_cmd_;
}

void RLController::GetJointCmdData(my_ros2_proto::msg::JointCommand& joint_cmd) {
  Update();
  joint_cmd = real_joint_cmd_;
}

void RLController::LoadModel() {
  // create env
  std::shared_ptr<Ort::Env> onnxEnvPrt(
      new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "LeggedOnnxController"));
  Ort::SessionOptions sessionOptions;
  sessionOptions.SetInterOpNumThreads(1);
  session_ptr_ = std::make_unique<Ort::Session>(
      *onnxEnvPrt, control_conf_.onnx_conf.policy_file.c_str(), sessionOptions);

  // get input and output info
  input_names_.clear();
  output_names_.clear();
  input_shapes_.clear();
  output_shapes_.clear();
  Ort::AllocatorWithDefaultOptions allocator;
  for (size_t i = 0; i < session_ptr_->GetInputCount(); ++i) {
    char* tempstring =
        new char[strlen(session_ptr_->GetInputNameAllocated(i, allocator).get()) + 1];
    strcpy(tempstring, session_ptr_->GetInputNameAllocated(i, allocator).get());
    input_names_.push_back(tempstring);
    input_shapes_.push_back(
        session_ptr_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
  }

  for (size_t i = 0; i < session_ptr_->GetOutputCount(); ++i) {
    char* tempstring =
        new char[strlen(session_ptr_->GetOutputNameAllocated(i, allocator).get()) + 1];
    strcpy(tempstring, session_ptr_->GetOutputNameAllocated(i, allocator).get());
    output_names_.push_back(tempstring);
    output_shapes_.push_back(
        session_ptr_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
  }
}

void RLController::Update() {
  UpdateStateEstimation();

  switch (control_mode_.load()) {
    case ControlMode::IDLE:
      HandleIdleMode();
      break;
    case ControlMode::ZERO:
      HandleZeroMode();
      break;
    case ControlMode::STAND:
      HandleStandMode();
      break;
    case ControlMode::WALK:
      HandleWalkMode();
      break;
    default:
      break;
  }
}

void RLController::HandleIdleMode() {
  if (use_sim_handles_) {
    sim_joint_cmd_.data =
        std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  } else {
    real_joint_cmd_.position = std::vector<double>(control_conf_.onnx_conf.actions_size, 0.0);
    real_joint_cmd_.velocity = std::vector<double>(control_conf_.onnx_conf.actions_size, 0.0);
    real_joint_cmd_.effort = std::vector<double>(control_conf_.onnx_conf.actions_size, 0.0);
    real_joint_cmd_.damping = std::vector<double>(control_conf_.onnx_conf.actions_size, 0.0);
    real_joint_cmd_.damping = std::vector<double>(control_conf_.onnx_conf.actions_size, 0.0);
    real_joint_cmd_.stiffness = std::vector<double>(control_conf_.onnx_conf.actions_size, 0.0);
  }
}

void RLController::HandleZeroMode() {
  if (trans_mode_percent_ <= 1) {
    for (size_t j = 0; j < control_conf_.ordered_joint_name.size(); ++j) {
      int index = joint_name_index_[control_conf_.ordered_joint_name[j]];
      double pos_des = current_joint_angles_(j) * (1 - trans_mode_percent_);
      if (j == 4 || j == 5 || j == 10 || j == 11) {
        // Magic number: the ankle joint
        // sim
        sim_joint_cmd_.data[j] = 250.0 * (pos_des - joint_state_data_.position[index]) +
                                 0.45 * (0.0 - joint_state_data_.velocity[index]);
        // real
        real_joint_cmd_.position[j] = pos_des;
        real_joint_cmd_.velocity[j] = 0.0;
        real_joint_cmd_.effort[j] = 0.0;
        real_joint_cmd_.stiffness[j] = 80.0;
        real_joint_cmd_.damping[j] = 1.5;
      } else {
        // sim
        sim_joint_cmd_.data[j] = 300.0 * (pos_des - joint_state_data_.position[index]) +
                                 1.0 * (0.0 - joint_state_data_.velocity[index]);
        // real
        real_joint_cmd_.position[j] = pos_des;
        real_joint_cmd_.velocity[j] = 0.0;
        real_joint_cmd_.effort[j] = 0.0;
        real_joint_cmd_.stiffness[j] = 80.0;
        real_joint_cmd_.damping[j] = 1.5;
      }
    }
    trans_mode_percent_ += 1 / trans_mode_duration_ms_;
    trans_mode_percent_ = std::min(trans_mode_percent_, scalar_t(1));
  }
}

void RLController::HandleStandMode() {
  if (trans_mode_percent_ <= 1) {
    for (size_t j = 0; j < control_conf_.ordered_joint_name.size(); ++j) {
      int index = joint_name_index_[control_conf_.ordered_joint_name[j]];
      double pos_des =
          current_joint_angles_(j) * (1 - trans_mode_percent_) +
          trans_mode_percent_ *
              control_conf_.joint_conf["init_state"][control_conf_.ordered_joint_name[j]];
      if (j == 4 || j == 5 || j == 10 || j == 11) {
        // Magic number: the ankle joint
        // sim
        sim_joint_cmd_.data[j] = 250.0 * (pos_des - joint_state_data_.position[index]) +
                                 0.45 * (0.0 - joint_state_data_.velocity[index]);
        // real
        real_joint_cmd_.position[j] = pos_des;
        real_joint_cmd_.velocity[j] = 0.0;
        real_joint_cmd_.effort[j] = 0.0;
        real_joint_cmd_.stiffness[j] = 80.0;
        real_joint_cmd_.damping[j] = 1.5;
      } else {
        // sim
        sim_joint_cmd_.data[j] = 300.0 * (pos_des - joint_state_data_.position[index]) +
                                 1.0 * (0.0 - joint_state_data_.velocity[index]);
        // real
        real_joint_cmd_.position[j] = pos_des;
        real_joint_cmd_.velocity[j] = 0.0;
        real_joint_cmd_.effort[j] = 0.0;
        real_joint_cmd_.stiffness[j] = 80.0;
        real_joint_cmd_.damping[j] = 1.5;
      }
    }
    trans_mode_percent_ += 1 / trans_mode_duration_ms_;
    trans_mode_percent_ = std::min(trans_mode_percent_, scalar_t(1));
  }
  loop_count_ = 0;
}

void RLController::HandleWalkMode() {
  // compute observation & actions
  if (loop_count_ % control_conf_.walk_step_conf.decimation == 0) {
    ComputeObservation();
    ComputeActions();
  }
  loop_count_++;

  // set action
  for (size_t i = 0; i < control_conf_.ordered_joint_name.size(); i++) {
    std::string joint_name = control_conf_.ordered_joint_name[i];
    scalar_t pos_des = actions_[i] * control_conf_.walk_step_conf.action_scale +
                       control_conf_.joint_conf["init_state"][joint_name];
    double stiffness = control_conf_.joint_conf["stiffness"][joint_name];
    double damping = control_conf_.joint_conf["damping"][joint_name];

    // sim
    sim_joint_cmd_.data[i] =
        stiffness * (pos_des - joint_state_data_.position[joint_name_index_[joint_name]]) +
        damping * (0.0 - joint_state_data_.velocity[joint_name_index_[joint_name]]);
    // real
    int index = joint_name_index_[control_conf_.ordered_joint_name[i]];
    if (i == 4 || i == 5 || i == 10 || i == 11) {
      double tau_des =
          stiffness * (pos_des - propri_.joint_pos[i]) + damping * (0 - propri_.joint_vel[i]);
      low_pass_filters_[i].input(tau_des);
      double tau_des_lp = low_pass_filters_[i].output();
      real_joint_cmd_.position[i] = 0.0;
      real_joint_cmd_.velocity[i] = 0.0;
      real_joint_cmd_.effort[i] = tau_des_lp;
      real_joint_cmd_.stiffness[i] = 0.0;
      real_joint_cmd_.damping[i] = 2.0;
    } else {
      low_pass_filters_[i].input(pos_des);
      double pos_des_lp = low_pass_filters_[i].output();
      real_joint_cmd_.position[i] = pos_des_lp;
      real_joint_cmd_.velocity[i] = 0.0;
      real_joint_cmd_.effort[i] = 0.0;
      real_joint_cmd_.stiffness[i] = stiffness;
      real_joint_cmd_.damping[i] = damping;
    }
    last_actions_(i, 0) = actions_[i];
  }
}

void RLController::UpdateStateEstimation() {
  {
    std::shared_lock<std::shared_mutex> lock(joint_state_mutex_);
    propri_.joint_pos.resize(control_conf_.onnx_conf.actions_size);
    propri_.joint_vel.resize(control_conf_.onnx_conf.actions_size);
    for (size_t i = 0; i < control_conf_.ordered_joint_name.size(); ++i) {
      std::string joint_name = control_conf_.ordered_joint_name[i];
      propri_.joint_pos(i) = joint_state_data_.position[joint_name_index_[joint_name]];
      propri_.joint_vel(i) = joint_state_data_.velocity[joint_name_index_[joint_name]];
    }
  }

  {
    std::shared_lock<std::shared_mutex> lock(imu_mutex_);
    vector3_t angular_vel;
    angular_vel(0) = imu_data_.angular_velocity.x;
    angular_vel(1) = imu_data_.angular_velocity.y;
    angular_vel(2) = imu_data_.angular_velocity.z;
    propri_.base_ang_vel = angular_vel;

    vector3_t gravity_vector(0, 0, -1);
    quaternion_t quat;
    quat.x() = imu_data_.orientation.x;
    quat.y() = imu_data_.orientation.y;
    quat.z() = imu_data_.orientation.z;
    quat.w() = imu_data_.orientation.w;
    matrix_t inverse_rot = GetRotationMatrixFromZyxEulerAngles(QuatToZyx(quat)).inverse();
    propri_.projected_gravity = inverse_rot * gravity_vector;
    propri_.base_euler_xyz = QuatToXyz(quat);
  }
}

void RLController::ComputeObservation() {
  double phase = duration<double>(high_resolution_clock::now().time_since_epoch()).count();
  if (control_conf_.walk_step_conf.sw_mode) {
    double cmd_norm = std::sqrt(Square(joy_data_.linear.x) + Square(joy_data_.linear.y) +
                                Square(joy_data_.angular.z));
    if (cmd_norm <= control_conf_.walk_step_conf.cmd_threshold) {
      phase = 0;
    }
  }
  phase = phase / control_conf_.walk_step_conf.cycle_time;

  // actions
  ControlCfg::OnnxCfg& onnx_conf = control_conf_.onnx_conf;
  vector_t propri_obs(onnx_conf.obs_size);

  propri_obs << sin(2 * M_PI * phase), cos(2 * M_PI * phase),
      joy_data_.linear.x * control_conf_.obs_scales.lin_vel,
      joy_data_.linear.y * control_conf_.obs_scales.lin_vel, joy_data_.angular.z,
      (propri_.joint_pos - init_joint_angles_) * control_conf_.obs_scales.dof_pos,
      propri_.joint_vel * control_conf_.obs_scales.dof_vel, last_actions_,
      propri_.base_ang_vel * control_conf_.obs_scales.ang_vel,
      propri_.base_euler_xyz * control_conf_.obs_scales.quat;

  if (is_first_rec_obs_) {
    for (size_t j = 0; j < control_conf_.ordered_joint_name.size(); ++j) {
      int index = joint_name_index_[control_conf_.ordered_joint_name[j]];
      if (j == 4 || j == 5 || j == 10 || j == 11) {
        low_pass_filters_[j].init(0);
      } else {
        low_pass_filters_[j].init(propri_.joint_pos[j]);
      }
    }

    // Magic number: set last_actions_ to 0
    for (int i = 29; i < 41; ++i) {
      propri_obs(i, 0) = 0.0;
    }
    for (int i = 0; i < onnx_conf.num_hist; ++i) {
      propri_history_buffer_.segment(i * onnx_conf.obs_size, onnx_conf.obs_size) =
          propri_obs.cast<tensor_element_t>();
    }
    is_first_rec_obs_ = false;
  }

  propri_history_buffer_.head(propri_history_buffer_.size() - onnx_conf.obs_size) =
      propri_history_buffer_.tail(propri_history_buffer_.size() - onnx_conf.obs_size);
  propri_history_buffer_.tail(onnx_conf.obs_size) = propri_obs.cast<tensor_element_t>();

  for (int i = 0; i < (onnx_conf.obs_size * onnx_conf.num_hist); ++i) {
    observations_[i] = static_cast<tensor_element_t>(propri_history_buffer_[i]);
  }
  // limit observations range
  scalar_t obs_min = -onnx_conf.obs_clip;
  scalar_t obs_max = onnx_conf.obs_clip;
  std::transform(
      observations_.begin(), observations_.end(), observations_.begin(),
      [obs_min, obs_max](scalar_t x) { return std::max(obs_min, std::min(obs_max, x)); });
}

void RLController::ComputeActions() {
  // create input tensor object
  std::vector<Ort::Value> input_tensor;
  input_tensor.push_back(Ort::Value::CreateTensor<tensor_element_t>(
      memory_info_, observations_.data(), observations_.size(), input_shapes_[0].data(),
      input_shapes_[0].size()));

  std::vector<Ort::Value> output_values = session_ptr_->Run(
      Ort::RunOptions{}, input_names_.data(), input_tensor.data(), 1, output_names_.data(), 1);

  for (int i = 0; i < control_conf_.onnx_conf.actions_size; ++i) {
    actions_[i] = *(output_values[0].GetTensorMutableData<tensor_element_t>() + i);
  }
  // limit action range
  scalar_t action_min = -control_conf_.onnx_conf.actions_clip;
  scalar_t action_max = control_conf_.onnx_conf.actions_clip;
  std::transform(actions_.begin(), actions_.end(), actions_.begin(),
                 [action_min, action_max](scalar_t x) {
                   return std::max(action_min, std::min(action_max, x));
                 });
}

}  // namespace xyber_x1_infer::rl_control_module