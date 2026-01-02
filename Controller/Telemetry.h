#pragma once

#include <Arduino.h>
#include <MsgPacketizer.h>

class Telemetry{
public:
  Telemetry(uint8_t send_index, uint8_t recv_index);

  void send_int();
  void send_float();
  void send_string();
  void send_array();
  void send_map();

  int get_int();
  float get_float();
  String get_string();
  

private:
  int i;
  float f;
  String s;

};