import time
import cv2
import mediapipe as mp
from sklearn.metrics.pairwise import euclidean_distances

import warnings

warnings.filterwarnings("ignore")

import logging

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
        self.is_toggled = False
        
        # WEBCAM CONFIG
        self.webcam = cv2.VideoCapture(webcam)
        
        # DETECTING AND DRAWING HANDS
        self.hands = mp.solutions.hands.Hands()
        self.hands_drawing_util = mp.solutions.drawing_utils

        # KEYBOARD
        self.top_row_keys = "qwertyuiop"
        self.middle_row_keys = "asdfghjkl"
        self.second_middle_row_keys = "zxcvbnm,."
        self.output = ""

        # KEY_POSITIONS
        self.key_positions = dict()

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
                        self.key_positions['<-'] = [pos, size]
                    continue

                pos, size = self.draw_key(frame, top_left, bottom_right, 
                                          self.top_row_keys[key_top_row])
                if self.frame_count == 1:
                        self.key_positions[self.top_row_keys[key_top_row]] = [pos, size]

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
                        self.key_positions['<--'] = [pos, size]
                    continue

                pos, size = self.draw_key(frame, top_left, bottom_right, 
                                          self.middle_row_keys[key_middle_row])
                if self.frame_count == 1:
                        self.key_positions[self.middle_row_keys[key_middle_row]] = [pos, size]

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
                        self.key_positions[self.second_middle_row_keys[key_second_middle_row]] = [pos, size]

                top_left[0] += int(frame.shape[1] / 12)
                bottom_right[0] += int(frame.shape[1] / 12)

            # 4TH ROW (SPACEBAR)
            top_left = [int(frame.shape[1] / 20), int(frame.shape[0] / 6)]
            bottom_right = [int(frame.shape[1] / 9), int(frame.shape[0] / 4)]
            top_left[1] += 3 * int(frame.shape[0] / 8)
            bottom_right[1] += 3 * int(frame.shape[0] / 8)
            bottom_right[0] += int(frame.shape[1] / 1.5)
            bottom_right[0] += int(frame.shape[1] / 5)

            pos, size = self.draw_key(frame, top_left, bottom_right, "spacebar")
            if self.frame_count == 1:
                        self.key_positions['spacebar'] = [pos, size]

            top_left[0] = bottom_right[0] + int(frame.shape[1] / 50)
            
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
                            if id == 8:
                                self.index = (x, y)
                            if id == 12:
                                self.middle = (x, y)
                for key in list(self.key_positions.keys()):
                    top_left = self.key_positions[key][0]
                    bottom_right = self.key_positions[key][1]
                    index_on_key = self.is_point_inside_rectangle(
                         top_left,
                         bottom_right,
                         self.index
                    )

                    middle_on_key = self.is_point_inside_rectangle(
                        top_left,
                        bottom_right,
                        self.middle
                    )

                    if index_on_key and middle_on_key:
                        cv2.rectangle(
                             frame, 
                             self.key_positions[key][0],
                             self.key_positions[key][1], 
                             (225, 255, 255),
                             2
                        )

                        distance_between_fingers = euclidean_distances(
                            [(self.index)],
                            [(self.middle)]
                        )

                        if distance_between_fingers < 50:
                            if self.is_toggled:
                                continue
                            self.toggle_key(
                                frame, 
                                self.key_positions[key][0],
                                self.key_positions[key][1]
                            )
                            self.perform_action(key)
                            self.is_toggled = True
                        else:
                            self.is_toggled = False

            text_position = int((self.resolution[0])/2), int((self.resolution[1])/2)
            cv2.putText(
                frame, self.output, 
                text_position, 
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 255, 255), 
                3
            )                                       
            cv2.imshow("Webcam", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
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
        return [top_left.copy(), bottom_right.copy()]
    
    def is_point_inside_rectangle(self, top_left, bottom_right, point):
        x, y = point
        if (top_left[0] <= x <= bottom_right[0]) and (top_left[1] <= y <= bottom_right[1]):
            return True
        return False
    
    def toggle_key(self, frame, top_left, bottom_right):
        cv2.rectangle(
            frame,
            top_left,
            bottom_right,
            (0, 255, 0),
            cv2.FILLED
        )

    def perform_action(self, key):
        if key == '<-':
            if self.output == '':
                return
            else:
                self.output = self.output[:-1]
        elif key == '<--':
            # DO WHATEVER YOU WANT WITH THE TEXT AFTER PRESSING ENTER
            self.output = ''
        elif key == 'spacebar':
            self.output += ' '
        else:
            self.output += key

    def start_process(self):
        try:
            logger.info("Starting virtual keyboard")
            start = time.time()
            self.show_webcam_with_hands()
            logger.info(f"Process finished with execution time of {time.time() - start} seconds")
        except Exception as e:
            logger.error(str(e))


if __name__ == '__main__':
    vkeyboard = VirtualKeyboard()
    vkeyboard.start_process()