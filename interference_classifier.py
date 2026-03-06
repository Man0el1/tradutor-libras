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
      for dot in range(len(hand.landmark)):
        x_.append(hand.landmark[dot].x)
        y_.append(hand.landmark[dot].y)

      min_x = min(x_)
      min_y = min(y_)
      max_x = max(x_)
      max_y = max(y_)

      for dot in range(len(hand.landmark)):
        data_aux.append((hand.landmark[dot].x - min_x) / (max_x - min_x))
        data_aux.append((hand.landmark[dot].y - min_y) / (max_y - min_y))
    
    prediction = model.predict([np.asarray(data_aux)])
    prediction_prob = model.predict_proba([np.asarray(data_aux)])
    
    predicted_letter = prediction[0]
    confidence = str(int(np.max(prediction_prob) * 100))

    cv2.putText(frame, predicted_letter, (150, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, confidence + "%", (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
  
  cv2.imshow('frame', frame)
  key = cv2.waitKey(5)
  if key == ord('q'):
    break
  
cap.release()
cv2.destroyAllWindows()