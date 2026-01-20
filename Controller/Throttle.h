#pragma once
#include <Arduino.h>
#include <VescUart.h>

struct VescData {
  bool vesc_ok = false;
  float rpm = 0.0;
  float volt = 0.0;
  float amps = 0.0;
  float duty = 0.0;
  float vesc_temp = 0.0;
  float motor_temp = 0.0;
  double throttle = 0.0;
};

class Throttle {
public:
  Throttle(int pot, double min, double max, double step, double bThreshold, double bCurrent, double cThreshold);

  void setup();
  void update();
  VescData getData();
  void printData();

private:
  double ramp(double current, double target, double step);

  VescUart vesc;

  int potPin;

  double inputMin;
  double inputMax;
  double rampStep;

  double brakeThreshold;
  double brakeCurrent;
  double coastThreshold;

  double filteredThrottle;
  bool throttleEnabled;

  double currentDuty;
  double targetDuty;

  static const int rpmTargets[6];
  int stages;
  int currentStage;

  VescData data;
};