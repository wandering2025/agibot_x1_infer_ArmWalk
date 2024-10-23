// Copyright (c) 2023, AgiBot Inc.
// All rights reserved.
#pragma once

#include <chrono>
#include "aimrt_module_cpp_interface/module_base.h"
#include "rl_control_module/rl_controller.h"

using namespace std::chrono;

namespace xyber_x1_infer::rl_control_module {

class RlControlModule : public aimrt::ModuleBase {
 public:
  RlControlModule() = default;
  ~RlControlModule() override = default;
  [[nodiscard]] aimrt::ModuleInfo Info() const override {
    return aimrt::ModuleInfo{.name = "RlControlModule"};
  }
  bool Initialize(aimrt::CoreRef core) override;
  bool Start() override;
  void Shutdown() override;

 private:
  bool MainLoop();

 private:
  aimrt::CoreRef core_;
  aimrt::executor::ExecutorRef executor_;
  aimrt::channel::SubscriberRef joy_start_sub_;
  aimrt::channel::SubscriberRef joy_zero_sub_;
  aimrt::channel::SubscriberRef joy_stand_sub_;
  aimrt::channel::SubscriberRef joy_walk_sub_;
  aimrt::channel::SubscriberRef joy_vel_sub_;
  aimrt::channel::SubscriberRef imu_sub_;
  aimrt::channel::SubscriberRef joint_state_sub_;
  aimrt::channel::PublisherRef joint_cmd_pub_;

  // [zero]←→[stand]
  //   ↑    /   ↑
  //   |   /    |
  //   ↓  ↓     ↓
  // [idle]←—[walk]
  time_point<high_resolution_clock> last_set_start_time_;
  time_point<high_resolution_clock> last_set_zero_time_;
  time_point<high_resolution_clock> last_set_stand_time_;
  time_point<high_resolution_clock> last_set_walk_time_;

  std::unique_ptr<RLController> rl_controller_;
  bool use_sim_handles_;
  int32_t freq_;
  std::atomic_bool run_flag_{true};
};

}  // namespace xyber_x1_infer::rl_control_module
