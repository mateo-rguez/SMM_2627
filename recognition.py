from ultralytics import YOLO
import numpy as np
import cv2
import time

# ---------------- CONFIGURACIÓN ----------------
VIDEO = "video.mp4"
MODELO = "yolo11n.pt"
PERSONA = 0
VEHICULOS = [2, 3, 5, 7]       # car, motorcycle, bus, truck
VENTANA = "Deteccion"
# ------------------------------------------------

AMARILLO, BLANCO, NEGRO, GRIS = (0, 255, 255), (255, 255, 255), (0, 0, 0), (60, 60, 60)


def puntos_cercanos(a, b):
    """Puntos más cercanos entre los bordes de dos cajas [x1, y1, x2, y2] y su distancia."""
    def eje(a1, a2, b1, b2):
        if a2 < b1:
            return a2, b1            # a está antes que b
        if b2 < a1:
            return a1, b2            # b está antes que a
        m = (max(a1, b1) + min(a2, b2)) / 2
        return m, m                  # se solapan en este eje
    xa, xb = eje(a[0], a[2], b[0], b[2])
    ya, yb = eje(a[1], a[3], b[1], b[3])
    return (int(xa), int(ya)), (int(xb), int(yb)), float(np.hypot(xa - xb, ya - yb))


def procesar(img):
    """Detecta personas y vehículos y une con una línea los bordes más cercanos de cada pareja."""
    r = model(img, classes=[PERSONA, *VEHICULOS], verbose=False)[0]
    frame = r.plot()
    cajas = r.boxes.xyxy.cpu().numpy()
    clases = r.boxes.cls.int().tolist()
    vehiculos = [b for b, c in zip(cajas, clases) if c in VEHICULOS]
    personas = [b for b, c in zip(cajas, clases) if c == PERSONA]

    for v in vehiculos:
        for p in personas:
            pv, pp, d = puntos_cercanos(v, p)
            cv2.line(frame, pv, pp, AMARILLO, 2)
            medio = ((pv[0] + pp[0]) // 2, (pv[1] + pp[1]) // 2)
            cv2.putText(frame, f"{d:.0f}px", medio, cv2.FONT_HERSHEY_SIMPLEX, 0.6, NEGRO, 4)
            cv2.putText(frame, f"{d:.0f}px", medio, cv2.FONT_HERSHEY_SIMPLEX, 0.6, AMARILLO, 2)
    return frame


# ---------------- VÍDEO ----------------
cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    raise SystemExit(f"No se puede abrir {VIDEO}")
fps = cap.get(cv2.CAP_PROP_FPS) or 30
ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
model = YOLO(MODELO)

pos = 0              # fotograma mostrado
salto = None         # fotograma al que saltar (lo pone la barra)
pausado = False
BOTON = (ancho - 150, 15, ancho - 15, 60)   # x1, y1, x2, y2


def dibujar_boton(frame):
    x1, y1, x2, y2 = BOTON
    cv2.rectangle(frame, (x1, y1), (x2, y2), GRIS, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), BLANCO, 2)
    cv2.putText(frame, "PLAY" if pausado else "PAUSA", (x1 + 25, y2 - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, BLANCO, 2)


def on_raton(evento, x, y, flags, param):
    global pausado
    x1, y1, x2, y2 = BOTON
    if evento == cv2.EVENT_LBUTTONDOWN and x1 <= x <= x2 and y1 <= y <= y2:
        pausado = not pausado


def on_barra(valor):
    global salto
    if valor != pos:       # ignorar cuando la movemos nosotros al reproducir
        salto = valor


cv2.namedWindow(VENTANA, cv2.WINDOW_NORMAL)
cv2.createTrackbar("Tiempo", VENTANA, 0, max(1, total - 1), on_barra)
cv2.setMouseCallback(VENTANA, on_raton)

frame = None
leer = True
t_anterior = time.time()

while True:
    if salto is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, salto)
        salto = None
        leer = True

    if not pausado or leer:
        ok, img = cap.read()
        if ok:
            pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            frame = procesar(img)
        else:
            pausado = True     # fin del vídeo: se queda en pausa
        leer = False

    if frame is None:
        break

    vista = frame.copy()
    dibujar_boton(vista)
    cv2.imshow(VENTANA, vista)
    cv2.setTrackbarPos("Tiempo", VENTANA, pos)

    # Esperar lo justo para reproducir a velocidad real
    espera = 30 if pausado else max(1, int((1 / fps - (time.time() - t_anterior)) * 1000))
    tecla = cv2.waitKey(espera) & 0xFF
    t_anterior = time.time()

    if tecla == ord("q") or cv2.getWindowProperty(VENTANA, cv2.WND_PROP_VISIBLE) < 1:
        break
    if tecla == ord(" "):
        pausado = not pausado

cap.release()
cv2.destroyAllWindows()