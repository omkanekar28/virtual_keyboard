import logging
import time
import cv2
import mediapipe as mp

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("app.log"),
                              logging.StreamHandler()])

logger = logging.getLogger(__name__)


class VirtualKeyboard:

    def __init__(self, webcam = 0, resolution = (800, 600)) -> None:
        self.start_time = time.time()
        self.frame_count = 0
        
        # WEBCAM CONFIG
        self.webcam = cv2.VideoCapture(webcam)
        self.webcam.set(3, resolution[0])   # WIDTH
        self.webcam.set(4, resolution[1])   # HEIGHT
        
        # DETECTING AND DRAWING HANDS
        self.hands = mp.solutions.hands.Hands()
        self.hands_drawing_util = mp.solutions.drawing_utils
        
    def show_webcam_with_hands(self):
        while True:
            _, frame = self.webcam.read()
            
            # SHOW FPS
            self.frame_count += 1
            elapsed_time = time.time() - self.start_time
            fps = str(int(self.frame_count / elapsed_time))
            cv2.putText(frame, fps, (575, 30), cv2.FONT_HERSHEY_TRIPLEX, 
                        1, (255, 255, 0), 2)
            
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
                        # ENDS OF THUMB AND INDEX FINGERS
                        if id in (4, 8):
                            x = int(landmark.x * frame.shape[0])
                            y = int(landmark.y * frame.shape[1])
                            # print(f"{id}: ({x}, {y})")
                            cv2.circle(frame, (x, y), 5, (255, 0, 255), cv2.FILLED)

            cv2.imshow("Webcam", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.webcam.release()
        cv2.destroyAllWindows()

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