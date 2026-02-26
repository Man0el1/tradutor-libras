import cv2
import os

PASTA_DATA = './data'
if not os.path.exists(PASTA_DATA):
  os.makedirs(PASTA_DATA)

num_pastas = 26
qunt_fotos = 50

alfabeto = []
for letra in range(65,91):
  alfabeto.append(chr(letra))

cap = cv2.VideoCapture(0)

for letra in alfabeto:
  if not os.path.exists(os.path.join(PASTA_DATA, letra)): 
    os.makedirs(os.path.join(PASTA_DATA, letra)) # exemplo : ./data/A
    
  concluido = False
  pular = False
  
  while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2.putText(
      frame,
      "Colhendo dados para " + letra + ": Q para continuar, P para pular e S para sair.",
      (10, 50),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.55,
      (0, 0, 0),
      1,
      cv2.LINE_AA
    )
    cv2.imshow('frame', frame)
    key = cv2.waitKey(5)
    if key == ord('q'):
      break
    if key == ord('p'):
      pular = True
      break
    if key == ord('s'):
      concluido = True
      break
    
  if pular:
    continue
  if concluido:
    break
  
  contador = 0
  
  while contador < qunt_fotos:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2.imshow('frame', frame)
    cv2.waitKey(25)
    cv2.imwrite(os.path.join(PASTA_DATA, letra, '{}.jpg'.format(contador + 1)), frame)
    contador += 1

cap.release()
cv2.destroyAllWindows()
  