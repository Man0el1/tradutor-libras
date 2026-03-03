import mediapipe as mp
import numpy as np
import pickle
import cv2

model_dictionary = pickle.load(open('./model.pickle', 'rb'))
model = model_dictionary['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

while True:
  data_aux = []
  x_ = []
  y_ = []
  
  ret, frame = cap.read()
  frame = cv2.flip(frame, 1)
  frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  results = hands.process(frame_rgb)
  
  H, W, _ = frame.shape
  
  if results.multi_hand_landmarks:
    for hand in results.multi_hand_landmarks:
      mp_drawing.draw_landmarks(
        frame,
        hand,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style()
      )
      
    for hand in results.multi_hand_landmarks:
      for dot in range(len(hand.landmark)):
        data_aux.append(hand.landmark[dot].x)
        data_aux.append(hand.landmark[dot].y)
        x_.append(hand.landmark[dot].x)
        y_.append(hand.landmark[dot].y)
        
    x1 = int(min(x_) * W)
    y1 = int(min(y_) * H)
    
    x2 = int(max(x_) * W)
    y2 = int(max(y_) * H)
    
    prediction = model.predict([np.asarray(data_aux)])
    predicted_letter = prediction[0]
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
    cv2.putText(frame, predicted_letter, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
  
  cv2.imshow('frame', frame)
  key = cv2.waitKey(5)
  if key == ord('q'):
    break
  
cap.release()
cv2.destroyAllWindows()