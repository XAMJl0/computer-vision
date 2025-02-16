import cv2
import time
import numpy as np
import mediapipe as mp
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import pyaudio
import audioop

wCam, hCam = 1280, 720

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Ошибка: Не удалось открыть веб-камеру.")
    exit()

cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

# Инициализация hand detection из mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Инициализация аудио управления
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVol, maxVol = volRange[0], volRange[1]

# Инициализация управления параметром звука
pyaudio_instance = pyaudio.PyAudio()
stream = pyaudio_instance.open(format=pyaudio.paInt16, channels=2, rate=44100, input=True, output=True, frames_per_buffer=1024)

def apply_white_noise(audio_data, noise_level):
    try:
        audio_data = np.frombuffer(audio_data, dtype=np.int16)

        # Генерация белого шума
        noise = np.random.normal(0, 1, len(audio_data)) * noise_level * 32767
        noise = noise.astype(np.int16)

        # Смешивание оригинала и белого шума
        signal_with_noise = audioop.add(audio_data.tobytes(), noise.tobytes(), 2)
        return signal_with_noise

    except Exception as e:
        print(f"Ошибка в функции apply_white_noise: {e}")
        return audio_data
    
    vol = minVol  # Текущий уровень громкости
noise_level = 0.0  # Процент белого шума по умолчанию

while True:
    success, img = cap.read()
    if not success:
        print("Не удалось получить изображение с камеры")
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    try:
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

                lmList = []
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])

                if len(lmList) != 0:
                    x1, y1 = lmList[4][1], lmList[4][2]
                    x2, y2 = lmList[8][1], lmList[8][2]
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
                    cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
                    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

                    length = math.hypot(x2 - x1, y2 - y1)

                    if handLms.landmark[0].x < handLms.landmark[5].x:  # Левая рука
                        vol = np.interp(length, [50, 300], [minVol, maxVol])
                        volBar = np.interp(length, [50, 300], [400, 150])
                        volPer = np.interp(length, [50, 300], [0, 100])
                        volume.SetMasterVolumeLevel(vol, None)
                        cv2.rectangle(img, (50, 150), (80, 400), (255, 0, 0), 3)
                        cv2.rectangle(img, (50, int(volBar)), (80, 400), (255, 0, 0), cv2.FILLED)
                        cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
                        cv2.putText(img, 'Громкость', (x1, y1 - 20), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)

                    else:  # Правая рука
                        noise_level = np.interp(length, [50, 300], [0.0, 1.0])
                        effectBar = np.interp(length, [50, 300], [400, 150])
                        if length < 50:
                            cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
                        else:
                            cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
                        cv2.rectangle(img, (100, 150), (130, 400), (0, 255, 0), 3)
                        cv2.rectangle(img, (100, int(effectBar)), (130, 400), (0, 255, 0), cv2.FILLED)
                        cv2.putText(img, 'Шум', (cx, cy - 20), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)

    except Exception as e:
        print(f"Ошибка при обработке руки: {e}")

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
    
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

    try:
        data = stream.read(1024)
        modified_data = apply_white_noise(data, noise_level)  # Применение белого шума
        stream.write(modified_data)
    except Exception as e:
        print(f"Ошибка при обработке аудио: {e}")

cap.release()
cv2.destroyAllWindows()
stream.stop_stream()
stream.close()
pyaudio_instance.terminate()