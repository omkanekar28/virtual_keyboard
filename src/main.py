import time
import cv2


class VirtualKeyboard:

    def __init__(self, webcam = 0, resolution = (800, 600)) -> None:
        self.webcam = cv2.VideoCapture(webcam)
        self.webcam.set(3, resolution[0])   # WIDTH
        self.webcam.set(4, resolution[1])   # HEIGHT
        self.start_time = time.time()
        self.frame_count = 0
        
    def show_webcam(self):
        while True:
            _, frame = self.webcam.read()
            
            # SHOW FPS
            self.frame_count += 1
            elapsed_time = time.time() - self.start_time
            fps = str(int(self.frame_count / elapsed_time))
            cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255, 0, 0), 2)

            cv2.imshow("Webcam", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.webcam.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    vkeyboard = VirtualKeyboard()
    vkeyboard.show_webcam()