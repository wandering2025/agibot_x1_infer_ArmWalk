// Copyright (c) 2023, AgiBot Inc.
// All rights reserved.

#pragma once

// linux
#include <math.h>

// c++
#include <string>
#include <vector>

#include <eigen3/Eigen/Dense>

namespace xyber_x1_infer::dcu_driver_module {

struct DataSpace {
  struct {
    double effort = 0;
    double velocity = 0;
    double position = 0;
    double kp = 0;
    double kd = 0;
  } cmd, state;
};

/**
 * \brief Contains pointers to raw data representing the position, velocity and
 * acceleration of a transmission's actuators.
 */
struct ActuatorHandle {
  double direction = 0;
  DataSpace* handle = nullptr;
};

/**
 * \brief Contains pointers to raw data representing the position, velocity and
 * acceleration of a transmission's joints.
 */
struct JointHandle {
  DataSpace* handle = nullptr;
};

class Transmission {
 public:
  Transmission(std::string name) : name_(name) {}
  virtual ~Transmission() {}

  std::string GetName() { return name_; }

  virtual void ActuatorToJointEffort() {};
  virtual void JointToActuatorEffort() {};

  virtual void ActuatorToJointVelocity() {};
  virtual void JointToActuatorVelocity() {};

  virtual void ActuatorToJointPosition() {};
  virtual void JointToActuatorPosition() {};

  virtual void TransformActuatorToJoint() {
    ActuatorToJointEffort();
    ActuatorToJointVelocity();
    ActuatorToJointPosition();
  }
  virtual void TransformJointToActuator() {
    JointToActuatorEffort();
    JointToActuatorVelocity();
    JointToActuatorPosition();
  }

 protected:
  std::string name_;
};

class SimpleTransmission : public Transmission {
 public:
  SimpleTransmission(std::string name, ActuatorHandle actuator, JointHandle joint)
      : Transmission(name), actuator_(actuator), joint_(joint) {}

  virtual void TransformActuatorToJoint() override {
    joint_.handle->state.effort = actuator_.handle->state.effort * actuator_.direction;
    joint_.handle->state.velocity = actuator_.handle->state.velocity * actuator_.direction;
    joint_.handle->state.position = actuator_.handle->state.position * actuator_.direction;
  }

  virtual void TransformJointToActuator() override {
    actuator_.handle->cmd.effort = joint_.handle->cmd.effort * actuator_.direction;
    actuator_.handle->cmd.velocity = joint_.handle->cmd.velocity * actuator_.direction;
    actuator_.handle->cmd.position = joint_.handle->cmd.position * actuator_.direction;

    actuator_.handle->cmd.kp = joint_.handle->cmd.kp;
    actuator_.handle->cmd.kd = joint_.handle->cmd.kd;
  }

 private:
  JointHandle joint_;
  ActuatorHandle actuator_;
};

class LeftAnkleParallelTransmission : public Transmission {
 public:
  LeftAnkleParallelTransmission(std::string name, std::string /*param*/, ActuatorHandle actr_left,
                                ActuatorHandle actr_right, JointHandle joint_pitch,
                                JointHandle joint_roll)
      : Transmission(name),
        actr_left_(actr_left),
        actr_right_(actr_right),
        joint_roll_(joint_roll),
        joint_pitch_(joint_pitch) {}

  virtual void TransformActuatorToJoint() override {
    // pos
    double qm5 = actr_right_.handle->state.position * actr_right_.direction;
    double qm6 = actr_left_.handle->state.position * actr_left_.direction;

    double tan_tr = kRL * std::sin(.5 * (qm6 - qm5));
    double q5, q6;
    std::tie(q5, q6) = std::tuple<double, double>(.5 * (qm5 + qm6), std::atan(tan_tr));

    // vel
    double qdm5 = actr_right_.handle->state.velocity * actr_right_.direction;
    double qdm6 = actr_left_.handle->state.velocity * actr_left_.direction;

    double qd5, qd6;
    double b = .5 * kRL * std::cos(.5 * (qm6 - qm5)) / (1 + tan_tr * tan_tr);
    std::tie(qd5, qd6) = std::tuple<double, double>(.5 * (qdm5 + qdm6), b * (qdm6 - qdm5));

    // tau
    double taum5 = actr_right_.handle->state.effort * actr_right_.direction;
    double taum6 = actr_left_.handle->state.effort * actr_left_.direction;

    static Eigen::Matrix<double, 2, 2> J =
        .5 * (Eigen::Matrix<double, 2, 2>() << 1., 1., -1., 1.).finished();
    J.row(1) << -b, b;
    Eigen::Matrix<double, 2, 1> tq_j =
        J.transpose().inverse() * (Eigen::Matrix<double, 2, 1>() << taum5, taum6).finished();
    double tauj5, tauj6;
    std::tie(tauj5, tauj6) = std::tuple<double, double>(tq_j(0), tq_j(1));

    joint_pitch_.handle->state.position = q5;
    joint_pitch_.handle->state.velocity = qd5;
    joint_pitch_.handle->state.effort = tauj5;

    joint_roll_.handle->state.position = q6;
    joint_roll_.handle->state.velocity = qd6;
    joint_roll_.handle->state.effort = tauj6;
  }

  virtual void TransformJointToActuator() override {
    double q6 = joint_roll_.handle->state.position;
    double mq5 = actr_right_.handle->state.position * actr_right_.direction;
    double mq6 = actr_left_.handle->state.position * actr_left_.direction;

    // pos
    double q5_cmd = joint_pitch_.handle->cmd.position;
    double q6_cmd = joint_roll_.handle->cmd.position;
    double qm5_cmd = 0;
    double qm6_cmd = 0;

    static Eigen::Matrix<double, 2, 2> m_inv =
        (Eigen::Matrix<double, 2, 2>() << 1., -1., 1., 1.).finished();
    Eigen::Matrix<double, 2, 1> q_m =
        m_inv * (Eigen::Matrix<double, 2, 1>() << q5_cmd,
                 std::asin(std::min(std::max(kLR * std::tan(q6_cmd), -1.0), 1.0)))
                    .finished();
    std::tie(qm5_cmd, qm6_cmd) = std::tuple<double, double>(q_m(0), q_m(1));

    // vel
    double qd5_cmd = joint_pitch_.handle->cmd.velocity;
    double qd6_cmd = joint_roll_.handle->cmd.velocity;
    double qdm5_cmd = 0;
    double qdm6_cmd = 0;

    static Eigen::Matrix<double, 2, 2> j =
        .5 * (Eigen::Matrix<double, 2, 2>() << 1., 1., -1., 1.).finished();
    double tan_qr = std::tan(q6);
    double b = .5 * kRL * std::cos(.5 * (mq6 - mq5)) / (1 + tan_qr * tan_qr);
    j.row(1) << -b, b;
    Eigen::Matrix<double, 2, 1> v_m =
        j.inverse() * (Eigen::Matrix<double, 2, 1>() << qd5_cmd, qd6_cmd).finished();
    std::tie(qdm5_cmd, qdm6_cmd) = std::tuple<double, double>(v_m(0), v_m(1));

    // tau
    double tau5_cmd = joint_pitch_.handle->cmd.effort;
    double tau6_cmd = joint_roll_.handle->cmd.effort;
    double taum5_cmd = 0;
    double taum6_cmd = 0;

    Eigen::Matrix<double, 2, 1> tq_m =
        j.transpose() * (Eigen::Matrix<double, 2, 1>() << tau5_cmd, tau6_cmd).finished();
    std::tie(taum5_cmd, taum6_cmd) = std::tuple<double, double>(tq_m(0), tq_m(1));

    actr_right_.handle->cmd.position = qm5_cmd * actr_right_.direction;
    actr_right_.handle->cmd.velocity = qdm5_cmd * actr_right_.direction;
    actr_right_.handle->cmd.effort = taum5_cmd * actr_right_.direction;
    actr_right_.handle->cmd.kp = joint_pitch_.handle->cmd.kp;
    actr_right_.handle->cmd.kd = joint_pitch_.handle->cmd.kd;

    actr_left_.handle->cmd.position = qm6_cmd * actr_left_.direction;
    actr_left_.handle->cmd.velocity = qdm6_cmd * actr_left_.direction;
    actr_left_.handle->cmd.effort = taum6_cmd * actr_left_.direction;
    actr_left_.handle->cmd.kp = joint_roll_.handle->cmd.kp;
    actr_left_.handle->cmd.kd = joint_roll_.handle->cmd.kd;
  }

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
  RightAnkleParallelTransmission(std::string name, std::string /*param*/, ActuatorHandle actr_left,
                                 ActuatorHandle actr_right, JointHandle joint_pitch,
                                 JointHandle joint_roll)
      : Transmission(name),
        actr_left_(actr_left),
        actr_right_(actr_right),
        joint_roll_(joint_roll),
        joint_pitch_(joint_pitch) {}

  virtual void TransformActuatorToJoint() override {
    // pos
    double qm5 = actr_left_.handle->state.position * actr_left_.direction;
    double qm6 = actr_right_.handle->state.position * actr_right_.direction;

    double tan_tr = kRL * std::sin(.5 * (qm5 - qm6));
    double q5, q6;
    std::tie(q5, q6) = std::tuple<double, double>(.5 * (qm5 + qm6), std::atan(tan_tr));

    // vela
    double qdm5 = actr_left_.handle->state.velocity * actr_left_.direction;
    double qdm6 = actr_right_.handle->state.velocity * actr_right_.direction;

    double qd5, qd6;
    double b = .5 * kRL * std::cos(.5 * (qm5 - qm6)) / (1 + tan_tr * tan_tr);
    std::tie(qd5, qd6) = std::tuple<double, double>(.5 * (qdm5 + qdm6), b * (qdm5 - qdm6));

    // tau
    double taum5 = actr_left_.handle->state.effort * actr_left_.direction;
    double taum6 = actr_right_.handle->state.effort * actr_right_.direction;

    static Eigen::Matrix<double, 2, 2> j =
        .5 * (Eigen::Matrix<double, 2, 2>() << 1., 1., 1., -1.).finished();
    j.row(1) << b, -b;
    Eigen::Matrix<double, 2, 1> tq_j =
        j.transpose().inverse() * (Eigen::Matrix<double, 2, 1>() << taum5, taum6).finished();
    double tauj5, tauj6;
    std::tie(tauj5, tauj6) = std::tuple<double, double>(tq_j(0), tq_j(1));

    joint_pitch_.handle->state.position = q5;
    joint_pitch_.handle->state.velocity = qd5;
    joint_pitch_.handle->state.effort = tauj5;

    joint_roll_.handle->state.position = q6;
    joint_roll_.handle->state.velocity = qd6;
    joint_roll_.handle->state.effort = tauj6;
  }

  virtual void TransformJointToActuator() override {
    double q6 = joint_roll_.handle->state.position;
    double mq5 = actr_left_.handle->state.position * actr_left_.direction;
    double mq6 = actr_right_.handle->state.position * actr_right_.direction;

    // pos
    double q5_cmd = joint_pitch_.handle->cmd.position;
    double q6_cmd = joint_roll_.handle->cmd.position;
    double qm5_cmd = 0;
    double qm6_cmd = 0;

    static Eigen::Matrix<double, 2, 2> m_inv =
        (Eigen::Matrix<double, 2, 2>() << 1., 1., 1., -1.).finished();
    Eigen::Matrix<double, 2, 1> q_m =
        m_inv * (Eigen::Matrix<double, 2, 1>() << q5_cmd,
                 std::asin(std::min(std::max(kLR * std::tan(q6_cmd), -1.0), 1.0)))
                    .finished();
    std::tie(qm5_cmd, qm6_cmd) = std::tuple<double, double>(q_m(0), q_m(1));

    // vel
    double qd5_cmd = joint_pitch_.handle->cmd.velocity;
    double qd6_cmd = joint_roll_.handle->cmd.velocity;
    double qdm5_cmd = 0;
    double qdm6_cmd = 0;

    static Eigen::Matrix<double, 2, 2> j =
        .5 * (Eigen::Matrix<double, 2, 2>() << 1., 1., 1., -1.).finished();
    double tan_qr = std::tan(q6);
    double b = .5 * kRL * std::cos(.5 * (mq5 - mq6)) / (1 + tan_qr * tan_qr);
    j.row(1) << b, -b;
    Eigen::Matrix<double, 2, 1> v_m =
        j.inverse() * (Eigen::Matrix<double, 2, 1>() << qd5_cmd, qd6_cmd).finished();
    std::tie(qdm5_cmd, qdm6_cmd) = std::tuple<double, double>(v_m(0), v_m(1));

    // tau
    double tau5_cmd = joint_pitch_.handle->cmd.effort;
    double tau6_cmd = joint_roll_.handle->cmd.effort;
    double taum5_cmd = 0;
    double taum6_cmd = 0;

    Eigen::Matrix<double, 2, 1> tq_m =
        j.transpose() * (Eigen::Matrix<double, 2, 1>() << tau5_cmd, tau6_cmd).finished();
    std::tie(taum5_cmd, taum6_cmd) = std::tuple<double, double>(tq_m(0), tq_m(1));

    actr_left_.handle->cmd.position = qm5_cmd * actr_left_.direction;
    actr_left_.handle->cmd.velocity = qdm5_cmd * actr_left_.direction;
    actr_left_.handle->cmd.effort = taum5_cmd * actr_left_.direction;
    actr_left_.handle->cmd.kp = joint_pitch_.handle->cmd.kp;
    actr_left_.handle->cmd.kd = joint_pitch_.handle->cmd.kd;

    actr_right_.handle->cmd.position = qm6_cmd * actr_right_.direction;
    actr_right_.handle->cmd.velocity = qdm6_cmd * actr_right_.direction;
    actr_right_.handle->cmd.effort = taum6_cmd * actr_right_.direction;
    actr_right_.handle->cmd.kp = joint_roll_.handle->cmd.kp;
    actr_right_.handle->cmd.kd = joint_roll_.handle->cmd.kd;
  }

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

class TransimissionManager {
 public:
  ~TransimissionManager() {
    for (auto it : transmissions_) {
      delete it;
    }
  }

  void RegisterTransmission(Transmission* trans) { transmissions_.push_back(trans); }

  void TransformActuatorToJoint() {
    for (const auto& it : transmissions_) {
      it->TransformActuatorToJoint();
    }
  }
  void TransformJointToActuator() {
    for (const auto& it : transmissions_) {
      it->TransformJointToActuator();
    }
  }

 private:
  std::vector<Transmission*> transmissions_;
};

}  // namespace xyber_x1_infer::dcu_driver_module