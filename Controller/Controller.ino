#include "Steering.h"
#include "Throttle.h"

Steering steering(2, 3, 4);
Throttle throttle(A12, 185.0, 875.0, 0.01, 0.10, 2.0, 0.01);

// Purple : Tx
// Orange : Rx

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  steering.setup();
  throttle.setup();
}

void loop() {
  // put your main code here, to run repeatedly:
  steering.update();
  throttle.update();
  throttle.printData();
}
