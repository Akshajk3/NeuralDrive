#include "Arduino.h"
#include "HardwareSerial.h"
#include "Throttle.h"

Throttle::Throttle(int pot, double min, double max, double step, double bThreshold, double bCurrent, double cThreshold) {
  potPin = pot;
  inputMin = min;
  inputMax = max;
  rampStep = step;
  brakeThreshold = bThreshold;
  brakeCurrent = bCurrent;
  coastThreshold = cThreshold;
}

void Throttle::setup() {
  Serial1.begin(115200);
  vesc.setSerialPort(&Serial1);
  pinMode(potPin, INPUT);

  Serial.println("Data Output Format: Throttle | Target | Current | RPM | Voltage | Motor Duty Cycle | Vesc Temp | Motor Temp");
}

void Throttle::update() {
  data.throttle = analogRead(potPin);
  targetDuty = (data.throttle - inputMin) / (inputMax - inputMin);
  targetDuty = constrain(targetDuty, 0.0, 1.0);

  currentDuty = ramp(currentDuty, targetDuty, rampStep);

  if (targetDuty > brakeThreshold) {
    vesc.setDuty(-currentDuty);
  } else if (targetDuty <= brakeThreshold && targetDuty > coastThreshold) {
    vesc.setBrakeCurrent(brakeCurrent);
  } else {
    vesc.setDuty(0.0);
  }

  delay(5);
}

VescData Throttle::getData() {
  data.vesc_ok = vesc.getVescValues();
  data.rpm = data.vesc_ok ? vesc.data.rpm : 0.0;
  data.volt = data.vesc_ok ? vesc.data.inpVoltage : 0.0;
  data.amps = data.vesc_ok ? vesc.data.avgMotorCurrent : 0.0;
  data.duty = data.vesc_ok ? vesc.data.dutyCycleNow : 0.0;
  data.vesc_temp = data.vesc_ok ? vesc.data.tempMosfet : 0.0;
  data.motor_temp = data.vesc_ok ? vesc.data.tempMotor : 0.0;

  return data;
}

void Throttle::printData() {
  if (data.vesc_ok) {
    Serial.print(data.throttle, 0);   Serial.print(" | ");
    Serial.print(targetDuty, 3);      Serial.print(" | ");
    Serial.print(currentDuty, 3);     Serial.print(" | ");
    Serial.print(data.rpm, 0);        Serial.print(" | ");
    Serial.print(data.volt, 1);       Serial.print(" | ");
    Serial.print(data.amps, 1);       Serial.print(" | ");
    Serial.print(data.duty, 3);       Serial.print(" | ");
    Serial.print(data.vesc_temp, 2);  Serial.print(" | ");
    Serial.print(data.motor_temp, 2);
    Serial.println();
  } else {
    Serial.println("[NO VESC]!");
  }
}

double Throttle::ramp(double current, double target, double step) {
  if (current < target) {
    current += step;
    current = (current > target) ? target : current; 
  }
  else if (current > target) {
    current -= step;
    current = (current < target) ? target : current;
  }

  return current;
}