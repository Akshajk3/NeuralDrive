#include "Arduino.h"
#include "HardwareSerial.h"
#include "Throttle.h"

const int Throttle::rpmTargets[6] = {500, 800, 1100, 1500, 1800, 2200};

Throttle::Throttle(int pot, double min, double max, double step, double bThreshold, double bCurrent, double cThreshold) {
  potPin = pot;
  inputMin = min;
  inputMax = max;
  rampStep = step;
  brakeThreshold = bThreshold;
  brakeCurrent = bCurrent;
  coastThreshold = cThreshold;

  currentDuty = 0.0;
  targetDuty  = 0.0;
  filteredThrottle = 0.0;
  throttleEnabled = false;
  stages = sizeof(rpmTargets) / sizeof(rpmTargets[0]);
  currentStage = 0;
}

void Throttle::setup() {
  Serial1.begin(115200);
  vesc.setSerialPort(&Serial1);
  pinMode(potPin, INPUT);
}

void Throttle::update() {
  double raw = analogRead(potPin);
  data.throttle = raw;

  targetDuty = (raw - inputMin) / (inputMax - inputMin);
  targetDuty = constrain(targetDuty, 0.0, 1.0);

  if (!throttleEnabled && targetDuty > 0.15) throttleEnabled = true;  // must exceed 15% to turn on
  if (throttleEnabled && targetDuty < 0.12)  throttleEnabled = false; // must drop below 12% to turn off

  if (!throttleEnabled) {
    currentDuty = 0.0;
    vesc.setDuty(0.0);
    currentStage = 0;
    return;
  }

  currentDuty = ramp(currentDuty, targetDuty, rampStep);

  vesc.setDuty(currentDuty);

  delay(5);
}

VescData Throttle::getData() {
  if (vesc.getVescValues()) {
    data.rpm        = vesc.data.rpm;
    data.volt       = vesc.data.inpVoltage;
    data.amps       = vesc.data.avgMotorCurrent;
    data.duty       = vesc.data.dutyCycleNow;
    data.vesc_temp  = vesc.data.tempMosfet;
    data.motor_temp = vesc.data.tempMotor;
  }
  return data;
}

void Throttle::printData() {
  Serial.print(data.throttle, 0);   Serial.print(" | ");
  Serial.print(targetDuty, 3);      Serial.print(" | ");
  Serial.print(currentDuty, 3);     Serial.print(" | ");
  Serial.print(data.rpm, 0);        Serial.print(" | ");
  Serial.print(data.volt, 1);       Serial.print(" | ");
  Serial.print(data.amps, 1);       Serial.print(" | ");
  Serial.print(data.duty, 3);       Serial.print(" | ");
  Serial.print(data.vesc_temp, 2);  Serial.print(" | ");
  Serial.print(data.motor_temp, 2);

  if (!data.vesc_ok) Serial.print("  [NO VESC]");
  Serial.println();
}

double Throttle::ramp(double current, double target, double step) {
  if (fabs(target - current) < step) {
    return target;
  }

  if (target > current) {
    current += step;
    if (current > target) current = target;
  } else {
    current -= step;
    if (current < target) current = target;
  }

  current = constrain(current, 0.0, 0.95);

  return current;
}