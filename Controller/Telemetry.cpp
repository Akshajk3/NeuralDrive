#include "Telemetry.h"

Telemetry::Telemetry(uint8_t send_ind, uint8_t recv_ind)
  : send_index(send_ind), recv_index(recv_ind) {
  
}

void Telemetry::setup() {
  MsgPacketizer::subscribe(Serial, recv_index, i, f, s, v, m);
}

void Telemetry::update() {
  MsgPacketizer::update();
}

void Telemetry::send_int(int p_i) {
  MsgPacketizer::send(Serial, send_index, p_i);
}

void Telemetry::send_float(float p_f) {
  MsgPacketizer::send(Serial, send_index, p_f);
}

void Telemetry::send_string(MsgPack::str_t p_s) {
  MsgPacketizer::send(Serial, send_index, p_s);
}

void Telemetry::send_vector(MsgPack::arr_t<int> p_v) {
  MsgPacketizer::send(Serial, send_index, p_v);
}

void Telemetry::send_map(MsgPack::map_t<String, float> p_m) {
  MsgPacketizer::send(Serial, send_index, p_m);
}

int Telemetry::get_int() {
  return i;
}

float Telemetry::get_float() {
  return f;
}

MsgPack::str_t Telemetry::get_string() {
  return s;
}

MsgPack::arr_t<int> Telemetry::get_vector() {
  return v;
}

MsgPack::map_t<String, float> Telemetry::get_map() {
  return m;
}