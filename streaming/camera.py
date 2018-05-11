#!/usr/bin/env python3

import numpy as np
import cv2
import time
import os
from threading import Thread

from flask import Flask, render_template, Response

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        # self.video = cv2.VideoCapture('video.mp4')
        self.fourcc = cv2.VideoWriter_fourcc(*'AVC1') # TODO x264
        self.name = time.asctime().replace(' ', '_')
        self.out = cv2.VideoWriter('static/' + self.name + '.mp4', self.fourcc, 20.0, (640,480))
        self.t = time.time()
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        # time.sleep(0.1)
        success, image = self.video.read()
        self.out.write(image)
        if time.time() - self.t > 10:
            self.t = time.time()
            self.out.release()
            for f in os.listdir('static')[:-10]:
                print('removing ' + f)
                os.remove('static/' + f)
            self.name = time.asctime().replace(' ', '_')
            self.out = cv2.VideoWriter('static/' + self.name + '.mp4', self.fourcc, 20.0, (640,480))
        # record video
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()     

app = Flask(__name__)

@app.route('/')
def index():
    # get all files
    files = os.listdir('static')
    return render_template('index.html', files=files)

@app.route('/playback')
def playback():
    files = os.listdir('static')
    return render_template('playback.html', files=files)

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True, debug=True)

# def segment(cap):
#     # 10 sec
#     t = time.time()
#     # TODO make sure there is less than 10 video files
#     for f in os.listdir('static')[:-10]:
#         print('removing ' + f)
#         os.remove('static/' + f)
#     # for dir in os.listdir('output'):
#     #     print(dir)
#     while (True):
#         ret, frame = cap.read()

#         out.write(frame)
        
#         # cv2.imshow('frame', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             out.release()
#             return False
#         if time.time() - t > 3:
#             out.release()
#             return True

# def foo():
#     cap = cv2.VideoCapture(0)

#     while segment(cap): pass

#     cap.release()
#     cv2.destroyAllWindows()

def play(f):
    cap = cv2.VideoCapture(f)
    while cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__hebi__':
    foo()
    play('./static/Fri_Mar_30_13:17:50_2018.avi')
