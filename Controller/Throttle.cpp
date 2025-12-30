#include "HardwareSerial.h"
#include "Throttle.h"

Throttle::Throttle(Serial vescSerial, int pot)
  : {

}

void Throttle::setup() {
  Serial1.begin(115200);

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