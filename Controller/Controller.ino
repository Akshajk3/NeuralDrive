#include "Steering.h"

Steering steering(2, 3, 4);

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  steering.setup();
}

void loop() {
  // put your main code here, to run repeatedly:
  steering.update();
}
