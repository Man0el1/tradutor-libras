import mediapipe as mp
import pickle
import cv2
import os

DATA_DIR = './data'
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

data = []
labels = []

for letter in os.listdir(DATA_DIR):
  for img_path in os.listdir(os.path.join(DATA_DIR, letter)):
    
    data_aux = []
    
    img = cv2.imread(os.path.join(DATA_DIR, letter, img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    print(letter + img_path)
    
    if results.multi_hand_landmarks:
      for hand in results.multi_hand_landmarks:
        for dot in range(len(hand.landmark)):
          data_aux.append(hand.landmark[dot].x)
          data_aux.append(hand.landmark[dot].y)
          
      data.append(data_aux)
      labels.append(letter)
      
f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()