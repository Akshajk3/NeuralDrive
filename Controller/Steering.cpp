#include "Steering.h"

volatile long Steering::counter = 0;
volatile long Steering::temp = 0;
uint8_t Steering::pinA = 0; // Green Wire
uint8_t Steering::pinB = 0; // White Wire
uint8_t Steering::servoPin = 0;
float Steering::rackRatio = 0;
float Steering::wheelRatio = 0;

Steering::Steering(int pin1, int pin2, int servPin, float Rratio = 1, float Wratio = 1) {
  pinA = pin1;
  pinB = pin2;
  servoPin = servPin;
  rackRatio = Rratio;
  wheelRatio = Wratio;
}

void Steering::setup() {
  pinMode(pinA, INPUT_PULLUP);
  pinMode(pinB, INPUT_PULLUP);

  attachInterrupt(0, Steering::ai0, CHANGE);
  attachInterrupt(1, Steering::ai1, CHANGE);

  steeringServo.attach(servoPin);

  steeringServo.write(135);
}

void Steering::update() {
  if (counter != temp) {
    Serial.println(counter);
    temp = counter;
  }

  float countsPerDeg = 1200.0f / 360.0f;
  float wheelDeg = (counter / countsPerDeg) * wheelRatio;

  wheelDeg = constrain(wheelDeg, -135, 135);

  float servoDeg = (wheelDeg + 135) / rackRatio;
  servoDeg = constrain(servoDeg, 0.0f, 270.0f);
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