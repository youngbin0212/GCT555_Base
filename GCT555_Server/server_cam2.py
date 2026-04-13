import cv2
import threading
import time
from flask import Flask, Response

# Configuration
WEB_PORT = 5004
CAMERA_INDEX = 3  # 다른 공간
DEBUG_MODE = True

current_frame = None
lock = threading.Lock()

app = Flask(__name__)

def generate_frames():
    while True:
        with lock:
            if current_frame is None:
                time.sleep(0.01)
                continue
            ret, buffer = cv2.imencode('.jpg', current_frame)
            frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/snapshot')
def snapshot():
    with lock:
        if current_frame is None:
            return "No frame", 504
        ret, buffer = cv2.imencode('.jpg', current_frame)
        return Response(buffer.tobytes(), mimetype='image/jpeg')

@app.route('/')
def index():
    return "<h1>Camera Stream Server</h1><p><a href='/video_feed'>View Stream</a></p>"

def main():
    global current_frame

    t_flask = threading.Thread(
        target=lambda: app.run(host='0.0.0.0', port=WEB_PORT, debug=False, use_reloader=False),
        daemon=False
    )
    t_flask.start()
    print(f"[Web] Server running on http://localhost:{WEB_PORT}")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Starting Main Loop...")
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        with lock:
            current_frame = image

        if DEBUG_MODE:
            cv2.imshow('Camera Stream', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()