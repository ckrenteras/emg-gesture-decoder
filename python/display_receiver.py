import json
import os
import socket
import tkinter as tk

MODEL_DIR = os.path.join('..', 'models', 'v4')
CLASS_MAPPING_PATH = os.path.join(MODEL_DIR, 'class_mapping.json')

UDP_PORT = 4210

GESTURE_DISPLAY_MAPPING = {
    "rest": "Rest",
    "open_hand": "Open hand",
    "pinch": "Pinch",
    "chaka": "Chaka",
}

with open(CLASS_MAPPING_PATH) as f:
    class_mapping = json.load(f)


def gesture_name_for(gesture_id):
    gesture = class_mapping.get(str(gesture_id))
    if gesture is None:
        return f"Unknown ({gesture_id})"
    return GESTURE_DISPLAY_MAPPING.get(gesture, gesture)


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("", UDP_PORT))
    sock.setblocking(False)

    root = tk.Tk()
    root.title("Gesture Display")
    root.geometry("1200x800")
    root.configure(bg="black")
    label = tk.Label(root, text="Waiting for prediction...", font=("Helvetica", 90, "bold"),
                      relief="flat", bg="black", fg="white")
    label.pack(expand=True)

    def poll():
        try:
            while True:
                data, _ = sock.recvfrom(64)
                if data:
                    label.config(text=gesture_name_for(data[0]))
        except BlockingIOError:
            pass
        root.after(20, poll)

    root.after(20, poll)
    root.mainloop()


if __name__ == "__main__":
    main()
