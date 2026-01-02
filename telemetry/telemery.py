import serial
from cobs import cobs
import serial.tools.list_ports
import msgpack

class Telemetry:
    def __init__(self, baudrate):
        self.baud_rate = baudrate
        self.port = self.list_serial_ports()
        self.ser = serial.Serial(self.port, self.baud_rate)

    def read_packet(self):
        buffer = self.ser.read_until(b'\x00')
        if len(buffer) < 2:
            return None

        try:
            decoded = cobs.decode(buffer[:-1])

            index = decoded[0]
            payload = msgpack.unpackb(decoded[1:])
            return index, payload
        except Exception as e:
            print(f"Decode error: {e}")
            return None

    def send_packet(self, index, data):
        payload = msgpack.packb(data)
        full_data = bytes([index]) + payload
        packet = cobs.encode(full_data) + b'\x00'
        self.ser.write(packet)

    def list_serial_ports():
        ports = list(serial.tools.list_ports.comports())
        if not ports:
            print("No Serial Devices Found")
            return None
        
        print("\nAvailable Serial Ports")
        for i, port in enumerate(ports):
            print(f"{i}: {port.device} - {port.description}")

        while True:
            try:
                choice = int(input("\nEnter Serial Port Number: "))
                if 0 <= choice < len(ports):
                    return ports[choice].device
                else:
                    print("Invalid Selection, Please Try Again")
            except ValueError:
                print("Please enter a number.")