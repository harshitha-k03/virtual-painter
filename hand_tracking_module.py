# We will use it as a module anywhere as well
import cv2 as cv
import numpy as np
import mediapipe as mp
import time


class detector:
    def __init__(self, mode=False, max_hands=2, detection_con=0.5, tracking_con=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.tracking_con = tracking_con

        """# Here we will use the module midea pipe by google to use the built in hand tracker
        # Firse we have to intialize the class"""
        self.mphands = mp.solutions.hands
        """# Next we have to create an object instance of that class to use it"""
        self.hands = self.mphands.Hands(self.mode, self.max_hands, self.detection_con, self.tracking_con)
        "# We have to intiate another class which helps in drawing the dots and lines representing our hand"
        self.mpdraw = mp.solutions.drawing_utils
        self.drawspec2 = self.mpdraw.DrawingSpec(color=(0, 255, 0))
        self.drawspec1 = self.mpdraw.DrawingSpec(color=(255, 0, 255))
        self.track_id = [8,12,16,20]
    def draw_hands(self, frame, draw=True):
        RGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(RGB)
        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpdraw.draw_landmarks(frame, handlms, self.mphands.HAND_CONNECTIONS, self.drawspec1,
                                               self.drawspec2)

        return frame

    def find_pos(self, frame, draw=False):
        self.lm_list = []
        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                for id, lm in enumerate(handlms.landmark):
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    self.lm_list.append([id, cx, cy])
                    if draw:
                        cv.circle(frame, (cx, cy), 25, (255, 0, 255), thickness=4)
        return self.lm_list

    def fingers_up(self,thumb=False):
        fingers = []
        if len(self.lm_list) != 0:
            if thumb:
                if self.lm_list[4][1] > self.lm_list[4 - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            for id in self.track_id:
                if self.lm_list[id][2] < self.lm_list[id - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers

def main():
    # Reading videos
    capture = cv.VideoCapture(0)
    # Now we will display the frame rate on the window
    ptime = 0
    ctime = 0
    hand_track = detector()
    while True:
        _, frame = capture.read()
        frame = hand_track.draw_hands(frame)
        lm_list = hand_track.find_pos(frame)
        print(lm_list)
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        cv.putText(frame, str(int(fps)), (30, 70), cv.FONT_ITALIC, 1, (0, 255, 0), 2)
        cv.imshow('video', frame)
        if cv.waitKey(1) & 0xFF == ord('d'):
            break
    capture.release()

    cv.destroyAllWindows()
    pass


if __name__ == "__main__":
    main()
