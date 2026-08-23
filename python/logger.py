import serial
import csv
import os

SERIAL_PORT = "/dev/cu.usbmodem202636001"
BAUD_RATE = 115200

GESTURE = 'fist'
CLASS = 2

SUBJECT = 1
TRIAL = 14
ARM = 'left'
OUTPUT_PATH = os.path.join('..', 'data', 'my_data', 'subject_one', 'individual_files', f'{GESTURE}_subject_{SUBJECT}_{ARM}_trial_{TRIAL:02d}b.csv')

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

print(f'Connecting to Teensy at port: {SERIAL_PORT}')
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

with open(OUTPUT_PATH, mode='w', newline='') as file:
    writer = csv.writer(file)
    print(f"Writing data to file {OUTPUT_PATH}. Press ctrl+c to stop program")

    try:

        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()

            if not line:
                continue

            data = line.split(',')
            if data[0] =='sample':
                writer.writerow(data + ['gesture', 'class', 'subject', 'trial'])
                print(data + ['gesture', 'class', 'subject', 'trial'])
                continue

            if len(data) != 3:
                print(f"Malformed data. Skipping: {line}")
                continue 

            print(data)
            writer.writerow(data + [GESTURE, CLASS, SUBJECT, TRIAL])


    except KeyboardInterrupt:
        print('\n Logging stopped. Filed saved')
    finally:
        ser.close()