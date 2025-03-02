import cv2
import mediapipe as mp
import time
import math
import numpy as np

class handDetector(): # Обьявляем класс и параметры
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode # Определяем обьект и далее параметры
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = 1
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img
    
    def findPosition(self, img, handNo = 0, draw = True):
        
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo] 
            for id, lm in enumerate(myHand.landmark):
                print(id, lm) # Выводим номер маркера и его координаты
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)
                lmList.append([id, cx,cy])
                #if id == 4: # Если закоментировать это условие, то будет отрисовываться окружности на всех маркерах
                if draw:
                    cv2.circle(img, (cx, cy), 7, (255, 0, 0), cv2.FILLED) # Отрисовка окружности по координатам 4 маркера
        
        return lmList    


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0) # Подгружаем видео с вебкамеры
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])
        
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
    
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, # Выводим частоту кадров на экран
            (255, 0, 255), 3)
    
        cv2.imshow("Image", img)
        cv2.waitKey(1)
        
    
if __name__ == "__main__":
    main()