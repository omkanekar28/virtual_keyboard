import logging
import time
import cv2
import mediapipe as mp
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("app.log"),
                              logging.StreamHandler()])

logger = logging.getLogger(__name__)


class VirtualKeyboard:

    def __init__(self, webcam = 0, resolution = (1280, 720)) -> None:
        self.start_time = time.time()
        self.frame_count = 0
        self.resolution = resolution
        
        # WEBCAM CONFIG
        self.webcam = cv2.VideoCapture(webcam)
        
        # DETECTING AND DRAWING HANDS
        self.hands = mp.solutions.hands.Hands()
        self.hands_drawing_util = mp.solutions.drawing_utils

        # KEYBOARD
        self.top_row_keys = "qwertyuiop"
        self.middle_row_keys = "asdfghjkl"
        self.second_middle_row_keys = "zxcvbnm,."
        self.numbers_row_keys = "1234567890"
        self.operators = "+-*/"
        self.is_numbers_row_enabled = True

        # KEY_POSITIONS
        self.key_positions = list()

    def show_webcam_with_hands(self):
        while True:
            _, frame = self.webcam.read()
            frame = cv2.resize(frame, self.resolution)
            
            # SHOW FPS
            self.frame_count += 1
            elapsed_time = time.time() - self.start_time
            fps = str(int(self.frame_count / elapsed_time))
            cv2.putText(frame, fps, (int(frame.shape[1] / 1.1), int(frame.shape[0] / 20)), cv2.FONT_HERSHEY_TRIPLEX, 
                        1, (0, 255, 255), 2)

            # VIRTUAL KEYBOARD (1ST ROW)
            top_left = [int(frame.shape[1] / 20), int(frame.shape[0] / 6)]
            bottom_right = [int(frame.shape[1] / 9), int(frame.shape[0] / 4)]

            for key_top_row in range(11):
                if key_top_row == 10:
                    bottom_right[0] += int(bottom_right[0] / 25)
                    pos, size = self.draw_key(frame, top_left, bottom_right, 
                                              '<-')
                    if self.frame_count == 1:
                        print(pos, size)
                        #TODO: the print for 1st row gives different values 
                        #TODO: than the one in exit(q) block, find out why.
                        self.key_positions.append([pos, size])
                    continue

                pos, size = self.draw_key(frame, top_left, bottom_right, 
                                          self.top_row_keys[key_top_row])
                if self.frame_count == 1:
                        print(pos, size)
                        self.key_positions.append([pos, size])

                top_left[0] += int(frame.shape[1] / 12)
                bottom_right[0] += int(frame.shape[1] / 12)

            # 2ND ROW
            top_left = [int(frame.shape[1] / 20), int(frame.shape[0] / 6)]
            bottom_right = [int(frame.shape[1] / 9), int(frame.shape[0] / 4)]
            top_left[1] += int(frame.shape[0] / 8)
            bottom_right[1] += int(frame.shape[0] / 8)

            for key_middle_row in range(10):
                if key_middle_row==9:
                    bottom_right[0] += int(bottom_right[0] / 7)
                    bottom_right[1] = int(frame.shape[0] / 4)
                    bottom_right[1] += 2 * int(frame.shape[0] / 8)
                    pos, size = self.draw_key(frame, top_left, bottom_right, 
                                              '<--')
                    if self.frame_count == 1:
                        self.key_positions.append([pos, size])
                    continue

                pos, size = self.draw_key(frame, top_left, bottom_right, 
                                          self.middle_row_keys[key_middle_row])
                if self.frame_count == 1:
                        self.key_positions.append([pos, size])

                top_left[0] += int(frame.shape[1] / 12)
                bottom_right[0] += int(frame.shape[1] / 12)
            
            # 3RD ROW
            top_left = [int(frame.shape[1] / 20), int(frame.shape[0] / 6)]
            bottom_right = [int(frame.shape[1] / 9), int(frame.shape[0] / 4)]
            top_left[1] += 2 * int(frame.shape[0] / 8)
            bottom_right[1] += 2 * int(frame.shape[0] / 8)

            for key_second_middle_row in range(9):

                pos, size = self.draw_key(frame, top_left, bottom_right, 
                                          self.second_middle_row_keys[key_second_middle_row])
                if self.frame_count == 1:
                        self.key_positions.append([pos, size])

                top_left[0] += int(frame.shape[1] / 12)
                bottom_right[0] += int(frame.shape[1] / 12)

            # 4TH ROW
            top_left = [int(frame.shape[1] / 20), int(frame.shape[0] / 6)]
            bottom_right = [int(frame.shape[1] / 9), int(frame.shape[0] / 4)]
            top_left[1] += 3 * int(frame.shape[0] / 8)
            bottom_right[1] += 3 * int(frame.shape[0] / 8)

            bottom_right[0] += int(frame.shape[1] / 1.5)
            pos, size = self.draw_key(frame, top_left, bottom_right, "spacebar")
            if self.frame_count == 1:
                        self.key_positions.append([pos, size])

            top_left[0] = bottom_right[0] + int(frame.shape[1] / 50)
            bottom_right[0] += int(frame.shape[1] / 5)
            pos, size = self.draw_key(frame, top_left, bottom_right, "123")
            if self.frame_count == 1:
                        self.key_positions.append([pos, size])

            # NUMBERS ROW
            if self.is_numbers_row_enabled:
                top_left = [int(frame.shape[1] / 20), int(frame.shape[0] / 6)]
                bottom_right = [int(frame.shape[1] / 9), int(frame.shape[0] / 4)]
                top_left[1] += 2 * int(frame.shape[0] / 4)
                bottom_right[1] += 2 * int(frame.shape[0] / 4)

                for key_numbers_row in range(10):

                    pos, size = self.draw_key(frame, top_left, bottom_right, 
                                              self.numbers_row_keys[key_numbers_row])
                    if self.frame_count == 1:
                        self.key_positions.append([pos, size])

                    top_left[0] += int(frame.shape[1] / 12)
                    bottom_right[0] += int(frame.shape[1] / 12)
                
                top_left = [int(frame.shape[1] / 20), int(frame.shape[0] / 6)]
                bottom_right = [int(frame.shape[1] / 9), int(frame.shape[0] / 4)]
                top_left[1] += int(2.5 * int(frame.shape[0] / 4))
                bottom_right[1] += int(2.5 * int(frame.shape[0] / 4))

                for operator in self.operators:
                    pos, size = self.draw_key(frame, top_left, bottom_right, 
                                              operator)
                    if self.frame_count == 1:
                        self.key_positions.append([pos, size])

                    top_left[0] += int(frame.shape[1] / 12)
                    bottom_right[0] += int(frame.shape[1] / 12)
            
            # HAND DETECTION
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hands = self.hands.process(frame_rgb).multi_hand_landmarks
            if hands:                
                # ITERATE THROUGH ALL THE HANDS BEING DETECTED
                for hand in hands:
                    self.hands_drawing_util.draw_landmarks(
                        frame, hand, mp.solutions.hands.HAND_CONNECTIONS
                    )

                    for id, landmark in enumerate(hand.landmark):
                        # ENDS OF INDEX AND MIDDLE FINGERS
                        if id in (8, 12):
                            x = int(landmark.x * frame.shape[1])
                            y = int(landmark.y * frame.shape[0])
                            cv2.circle(frame, (x, y), 5, (255, 0, 255), cv2.FILLED)
            
            for key_position in self.key_positions:
                point1 = key_position[0]
                point2 = key_position[1]
                cv2.line(frame, point1, point2, (255, 255, 255), 2)

            cv2.imshow("Webcam", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                    for i in self.key_positions:
                         print(i)
                    break

        self.webcam.release()
        cv2.destroyAllWindows()

    def draw_key(self, frame, top_left, bottom_right, key):
        # DRAWING RECTANGLE
        cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)

        # PUTTING KEY INSIDE IT
        center_x = int((top_left[0] + bottom_right[0]) / 2)
        center_y = int((top_left[1] + bottom_right[1]) / 2)

        cv2.putText(
            frame, key, 
            (center_x - 10, center_y + 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )

        # RETURNS POSITION
        return [top_left, bottom_right]

    def start_process(self):
        try:
            logger.info("Starting virtual keyboard")
            start = time.time()
            self.show_webcam_with_hands()
            logger.info(f"Process finished with execution time of {time.time() - start} seconds")
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(str(e))


if __name__ == '__main__':
    vkeyboard = VirtualKeyboard()
    vkeyboard.start_process()