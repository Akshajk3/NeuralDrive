#pragma once

#include <Arduino.h>
#include <MsgPacketizer.h>

class Telemetry{
public:
  Telemetry(uint8_t send_index, uint8_t recv_index);

  void setup();
  void update();

  void send_int(int p_i);
  void send_float(float p_f);
  void send_string(MsgPack::str_t p_s);
  void send_vector(MsgPack::arr_t<int> p_v);
  void send_map(MsgPack::map_t<String, float> p_m);

  int get_int();
  float get_float();
  MsgPack::str_t get_string();
  MsgPack::arr_t<int> get_vector();
  MsgPack::map_t<String, float> get_map();

private:
  uint8_t send_index;
  uint8_t recv_index;
  int i;
  float f;
  MsgPack::str_t s;
  MsgPack::arr_t<int> v;
  MsgPack::map_t<String, float> m;
};