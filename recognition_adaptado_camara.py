#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ultralytics import YOLO

import numpy as np
import cv2
import time

from receptor import ReceptorVideo, argumentos_receptor


# ---------------- CONFIGURACIÓN ----------------

MODELO = "yolo11n.pt"

PERSONA = 0
VEHICULOS = [2, 3, 5, 7]

VENTANA = "Deteccion en directo"

TIMEOUT_VIDEO = 1.0

AMARILLO = (0, 255, 255)
BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)
GRIS = (60, 60, 60)
VERDE = (0, 255, 0)
ROJO = (0, 0, 255)

# ------------------------------------------------


def puntos_cercanos(a, b):
    """
    Calcula los puntos más cercanos entre los bordes
    de dos cajas [x1, y1, x2, y2] y su distancia.
    """

    def eje(a1, a2, b1, b2):

        if a2 < b1:
            return a2, b1

        if b2 < a1:
            return a1, b2

        m = (max(a1, b1) + min(a2, b2)) / 2

        return m, m

    xa, xb = eje(a[0], a[2], b[0], b[2])
    ya, yb = eje(a[1], a[3], b[1], b[3])

    return (
        (int(xa), int(ya)),
        (int(xb), int(yb)),
        float(np.hypot(xa - xb, ya - yb))
    )


def procesar(img, model):
    """
    Detecta personas y vehículos y une mediante una
    línea los bordes más cercanos de cada pareja.
    """

    r = model(
        img,
        classes=[PERSONA, *VEHICULOS],
        verbose=False
    )[0]

    frame = r.plot()

    cajas = r.boxes.xyxy.cpu().numpy()
    clases = r.boxes.cls.int().tolist()

    vehiculos = [
        b for b, c in zip(cajas, clases)
        if c in VEHICULOS
    ]

    personas = [
        b for b, c in zip(cajas, clases)
        if c == PERSONA
    ]

    for v in vehiculos:

        for p in personas:

            pv, pp, d = puntos_cercanos(v, p)

            cv2.line(
                frame, pv, pp, AMARILLO, 2
            )

            medio = (
                (pv[0] + pp[0]) // 2,
                (pv[1] + pp[1]) // 2
            )

            cv2.putText(
                frame,
                f"{d:.0f}px",
                medio,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                NEGRO,
                4
            )

            cv2.putText(
                frame,
                f"{d:.0f}px",
                medio,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                AMARILLO,
                2
            )

    return frame


def dibujar_estado(frame, fps, tiempo_ia):

    cv2.rectangle(
        frame,
        (10, 10),
        (390, 105),
        NEGRO,
        -1
    )

    cv2.putText(
        frame,
        "STREAMING EN DIRECTO",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        VERDE,
        2
    )

    cv2.putText(
        frame,
        f"FPS procesamiento: {fps:.1f}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        BLANCO,
        2
    )

    cv2.putText(
        frame,
        f"Tiempo IA: {tiempo_ia:.1f} ms",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        BLANCO,
        2
    )


def dibujar_sin_senal(width, height):

    frame = np.zeros(
        (height, width, 3),
        dtype=np.uint8
    )

    cv2.putText(
        frame,
        "SIN SENAL DE VIDEO",
        (30, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        ROJO,
        3
    )

    cv2.putText(
        frame,
        "Sistema de deteccion no operativo",
        (30, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        BLANCO,
        2
    )

    return frame


def main():

    args = argumentos_receptor()

    print("Cargando modelo YOLO...")
    model = YOLO(MODELO)

    receptor = ReceptorVideo(
        port=args.port,
        width=args.width,
        height=args.height,
        latency=args.latency,
        bind=args.bind
    ).start()

    cv2.namedWindow(
        VENTANA,
        cv2.WINDOW_NORMAL
    )

    print("Esperando streaming de vídeo...")

    fps = 0.0
    contador_frames = 0
    inicio_fps = time.monotonic()

    try:

        while receptor.isOpened():

            ok, img = receptor.read(timeout=0.1)

            if ok:

                inicio_ia = time.perf_counter()

                frame = procesar(img, model)

                tiempo_ia = (
                    time.perf_counter() - inicio_ia
                ) * 1000

                contador_frames += 1

                ahora = time.monotonic()
                intervalo = ahora - inicio_fps

                if intervalo >= 1.0:

                    fps = contador_frames / intervalo

                    contador_frames = 0
                    inicio_fps = ahora

                if receptor.edad_frame() > TIMEOUT_VIDEO:

                    frame = dibujar_sin_senal(
                        args.width,
                        args.height
                    )

                else:

                    dibujar_estado(
                        frame,
                        fps,
                        tiempo_ia
                    )

                cv2.imshow(VENTANA, frame)

            elif receptor.edad_frame() > TIMEOUT_VIDEO:

                fps = 0.0
                contador_frames = 0
                inicio_fps = time.monotonic()

                frame = dibujar_sin_senal(
                    args.width,
                    args.height
                )

                cv2.imshow(VENTANA, frame)

            tecla = cv2.waitKey(1) & 0xFF

            if tecla == ord("q"):
                break

            if cv2.getWindowProperty(
                VENTANA,
                cv2.WND_PROP_VISIBLE
            ) < 1:
                break

    except KeyboardInterrupt:
        pass

    finally:

        receptor.release()
        cv2.destroyAllWindows()

        print("Detección finalizada")


if __name__ == "__main__":
    main()