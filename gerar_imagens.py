import mediapipe as mp
import cv2
import os

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
  os.makedirs(DATA_DIR)
  
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

quant_images = 50

alphabet = []
for letter in range(65,91):
  alphabet.append(chr(letter))

cap = cv2.VideoCapture(0)

for letter in alphabet:
  if not os.path.exists(os.path.join(DATA_DIR, letter)): 
    os.makedirs(os.path.join(DATA_DIR, letter)) # exemple : ./data/A
    
  complete = False
  skip = False
  while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    cv2.putText(
      frame,
      "Collecting data for " + letter + ": Q to continue, S to skip letter, E to exit.",
      (10, 50),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.55,
      (0, 0, 0),
      1,
      cv2.LINE_AA
    )
    
    if results.multi_hand_landmarks:
      for hand in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
          frame,
          hand,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style()
        )

    cv2.imshow('frame', frame)
    key = cv2.waitKey(5)
    if key == ord('q'):
      break
    if key == ord('s'):
      skip = True
      break
    if key == ord('e'):
      complete = True
      break
    
  if skip:
    continue
  if complete:
    break
  
  counter = 0
  while counter < quant_images * 2:
    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
      cv2.imshow('frame', frame)
      cv2.waitKey(50)
      cv2.imwrite(os.path.join(DATA_DIR, letter, '{}.jpg'.format(counter + 1)), frame)
      cv2.imwrite(os.path.join(DATA_DIR, letter, '{}.jpg'.format(counter + 2)), cv2.flip(frame, 1))
      counter += 2
    else:
      cv2.imshow('frame', frame)

cap.release()
cv2.destroyAllWindows()
  