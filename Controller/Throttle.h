#pragma once
#include <Arduino.h>
#include <VescUart.h>

class Throttle {
public:
  Throttle();

  void setup();
  void update();

private:
  double ramp(double current, double target, double step);

  VescUart vesc;

  const double inputMin;
  const double inputMax;
  const double rampStep;

  const double brakeThreshold;
  const double brakeCurrent;
  const double coastThreshold;

  double currentDuty;
  double targetDuty;
};