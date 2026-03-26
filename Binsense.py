from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import cv2

np.set_printoptions(suppress=True)

# Cargar modelo y etiquetas
model = load_model("keras_model.h5", compile=False)
class_names = open("etiquetas.txt", "r").readlines()

# Iniciar cámara
cap = cv2.VideoCapture(0)

print("Presiona 'q' para salir")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preparar imagen para el modelo
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized

    # Predecir
    prediction = model.predict(data, verbose=0)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()[2:]  # quita el "0 " o "1 "
    confidence = prediction[0][index]

    # Mostrar resultado en pantalla
    color = (0, 255, 0) if "Reciclable" in class_name and "No" not in class_name else (0, 0, 255)
    label = f"{class_name}: {confidence * 100:.1f}%"
    cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.imshow("Binsense - Deteccion de residuos", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
