// Copyright (c) 2023, AgiBot Inc.
// All rights reserved.
#include "rl_control_module/utilities.h"

bool Throttler(const time_point<high_resolution_clock> now, time_point<high_resolution_clock> &last,
               const milliseconds interval) {
  auto elapsed = now - last;

  if (elapsed >= interval) {
    last = now;
    return true;
  }
  return false;
}

template <typename T>
digital_lp_filter<T>::digital_lp_filter(T w_c, T t_s) {
  Lpf_in_prev[0] = Lpf_in_prev[1] = 0;
  Lpf_out_prev[0] = Lpf_out_prev[1] = 0;
  Lpf_in1 = 0, Lpf_in2 = 0, Lpf_in3 = 0, Lpf_out1 = 0, Lpf_out2 = 0;
  wc = w_c;
  ts = t_s;
  update();
}

template <typename T>
void digital_lp_filter<T>::update(void) {
  double den = 2500 * ts * ts * wc * wc + 7071 * ts * wc + 10000;

  Lpf_in1 = 2500 * ts * ts * wc * wc / den;
  Lpf_in2 = 5000 * ts * ts * wc * wc / den;
  Lpf_in3 = 2500 * ts * ts * wc * wc / den;
  Lpf_out1 = -(5000 * ts * ts * wc * wc - 20000) / den;
  Lpf_out2 = -(2500 * ts * ts * wc * wc - 7071 * ts * wc + 10000) / den;
}

template <typename T>
digital_lp_filter<T>::~digital_lp_filter(void) {}

template <typename T>
void digital_lp_filter<T>::input(T lpf_in) {
  lpf_out = Lpf_in1 * lpf_in + Lpf_in2 * Lpf_in_prev[0] +
            Lpf_in3 * Lpf_in_prev[1] +                                // input component
            Lpf_out1 * Lpf_out_prev[0] + Lpf_out2 * Lpf_out_prev[1];  // output component
  Lpf_in_prev[1] = Lpf_in_prev[0];
  Lpf_in_prev[0] = lpf_in;
  Lpf_out_prev[1] = Lpf_out_prev[0];
  Lpf_out_prev[0] = lpf_out;
}

template <typename T>
T digital_lp_filter<T>::output(void) {
  return lpf_out;
}

template <typename T>
void digital_lp_filter<T>::set_ts(T t_s) {
  ts = t_s;
  update();
}

template <typename T>
void digital_lp_filter<T>::set_wc(T w_c) {
  wc = w_c;
  update();
}

template <typename T>
void digital_lp_filter<T>::clear(void) {
  Lpf_in_prev[1] = 0;
  Lpf_in_prev[0] = 0;
  Lpf_out_prev[1] = 0;
  Lpf_out_prev[0] = 0;
}

template <typename T>
void digital_lp_filter<T>::init(T init_data) {
  Lpf_in_prev[1] = init_data;
  Lpf_in_prev[0] = init_data;
  Lpf_out_prev[1] = init_data;
  Lpf_out_prev[0] = init_data;
}

template class digital_lp_filter<double>;
template class digital_lp_filter<float>;
