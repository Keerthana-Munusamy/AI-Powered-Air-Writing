from flask import Flask, render_template, Response, request, send_file
import cv2
import numpy as np
import mediapipe as mp
import math
import os
from datetime import datetime

app = Flask(__name__)
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255
prev_x, prev_y = None, None
motion_threshold = 5

def generate_frames():
    global prev_x, prev_y, canvas
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            lm = hand_landmarks.landmark
            x_tip, y_tip = int(lm[8].x * w), int(lm[8].y * h)
            x_base, y_base = int(lm[5].x * w), int(lm[5].y * h)

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.circle(frame, (x_tip, y_tip), 8, (0, 255, 0), -1)

            if y_tip < y_base - 20:
                if prev_x is not None:
                    dist = math.hypot(x_tip - prev_x, y_tip - prev_y)
                    if dist > motion_threshold:
                        cv2.line(canvas, (prev_x, prev_y), (x_tip, y_tip), (0, 0, 0), 3)
                prev_x, prev_y = x_tip, y_tip
            else:
                prev_x, prev_y = None, None
        else:
            prev_x, prev_y = None, None

        combined = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
        ret, buffer = cv2.imencode('.jpg', combined)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/clear_canvas', methods=['POST'])
def clear_canvas():
    global canvas
    canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255
    return ('', 204)

@app.route('/save_canvas', methods=['POST'])
def save_canvas():
    filename = f"static/output/drawing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    cv2.imwrite(filename, canvas)
    return send_file(filename, mimetype='image/png', as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
