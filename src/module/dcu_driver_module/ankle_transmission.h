// Copyright (c) 2023, AgiBot Inc.
// All rights reserved.

#pragma once

// project
#include "dcu_driver_module/transmission.h"

namespace xyber_x1_infer::dcu_driver_module {

class LeftAnkleParallelTransmission : public Transmission {
 public:
  LeftAnkleParallelTransmission(std::string name, std::string, ActuatorHandle actr_left,
                                ActuatorHandle actr_right, JointHandle joint_pitch,
                                JointHandle joint_roll)
      : Transmission(name),
        actr_left_(actr_left),
        actr_right_(actr_right),
        joint_roll_(joint_roll),
        joint_pitch_(joint_pitch) {}

  virtual void TransformActuatorToJoint() override;
  virtual void TransformJointToActuator() override;

 private:
  JointHandle joint_roll_;
  JointHandle joint_pitch_;
  ActuatorHandle actr_left_;
  ActuatorHandle actr_right_;

  const double kR = 25e-3;
  const double kL = (50.0) / 2. * 1e-3;
  const double kRL = kR / kL;
  const double kLR = kL / kR;
};

class RightAnkleParallelTransmission : public Transmission {
 public:
  RightAnkleParallelTransmission(std::string name, std::string, ActuatorHandle actr_left,
                                 ActuatorHandle actr_right, JointHandle joint_pitch,
                                 JointHandle joint_roll)
      : Transmission(name),
        actr_left_(actr_left),
        actr_right_(actr_right),
        joint_roll_(joint_roll),
        joint_pitch_(joint_pitch) {}

  virtual void TransformActuatorToJoint() override;
  virtual void TransformJointToActuator() override;

 private:
  JointHandle joint_roll_;
  JointHandle joint_pitch_;
  ActuatorHandle actr_left_;
  ActuatorHandle actr_right_;

  const double kR = 25e-3;
  const double kL = (50.0) / 2. * 1e-3;
  const double kRL = kR / kL;
  const double kLR = kL / kR;
};

}  // namespace xyber_x1_infer::dcu_driver_module
