#include "Steering.h"

volatile long Steering::counter = 0;
volatile long Steering::temp = 0;
uint8_t Steering::pinA = 0;
uint8_t Steering::pinB = 0;
uint8_t Steering::servoPin = 0;
float Steering::gearRatio = 1;

Steering::Steering(int pin1, int pin2, int servPin) {
  pinA = pin1;
  pinB = pin2;
  servoPin = servPin;
}

void Steering::setup() {
  pinMode(pinA, INPUT_PULLUP);
  pinMode(pinB, INPUT_PULLUP);

  attachInterrupt(0, Steering::ai0, RISING);
  attachInterrupt(1, Steering::ai1, RISING);

  steeringServo.attach(servoPin);

  steeringServo.write(270);
  delay(100);
  steeringServo.write(0);
  delay(100);
  steeringServo.write(135);
}

void Steering::update() {
  if (counter != temp) {
    Serial.println(counter);
    temp = counter;
  }

  float countsPerDeg = 1200 / 360;
  float wheelDeg = counter / countsPerDeg;

  wheelDeg = constrain(wheelDeg, -135, 135);

  float servoDeg = (wheelDeg + 135) * gearRatio;
  steeringServo.write(servoDeg);
}

void Steering::ai0() {
  if (digitalRead(pinB) == LOW) {
    counter--;
  } else {
    counter++;
  }
}

void Steering::ai1() {
  if (digitalRead(pinA) == LOW) {
    counter++;
  } else {
    counter--;
  }
}