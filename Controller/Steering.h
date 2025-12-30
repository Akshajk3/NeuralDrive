#pragma once
#include <Arduino.h>
#include <Servo.h>

class Steering {
public:
  Steering(int pin1, int pin2, int servoPin);

  void setup();
  void update();

private:
  static void ai0();
  static void ai1();

  static volatile long temp, counter;
  static uint8_t pinA;
  static uint8_t pinB;
  static uint8_t servoPin;
  static float gearRatio;
  Servo steeringServo;
};