#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Receptor de vídeo H.264 mediante RTP/UDP."""

import argparse
import subprocess
import threading
import shutil
import time

import numpy as np


class ReceptorVideo:

    def __init__(
        self,
        port=5004,
        width=1280,
        height=720,
        latency=50,
        bind="0.0.0.0"
    ):
        self.port = port
        self.width = width
        self.height = height
        self.latency = latency
        self.bind = bind

        # Cada píxel contiene tres componentes BGR.
        self.bytes_frame = width * height * 3

        # GStreamer puede añadir padding a cada fila.
        if width % 4 != 0:
            raise ValueError(
                "Utiliza una anchura múltiplo de 4"
            )

        self.proceso = None
        self.hilo = None

        self.activo = False
        self.frame = None
        self.timestamp = 0.0

        self.secuencia = 0
        self.ultima_leida = 0

        self.condicion = threading.Condition()

    def pipeline(self):
        """Construye el pipeline de recepción."""

        return [
            "gst-launch-1.0", "-q",

            "udpsrc",
            f"address={self.bind}",
            f"port={self.port}",
            (
                "caps=application/x-rtp,"
                "media=video,"
                "encoding-name=H264,"
                "payload=96,"
                "clock-rate=90000"
            ), "!",

            "rtpjitterbuffer",
            f"latency={self.latency}",
            "drop-on-latency=true",
            "do-lost=true", "!",

            "rtph264depay", "!",
            "h264parse", "!",
            "avdec_h264", "!",

            "queue",
            "max-size-buffers=1",
            "max-size-bytes=0",
            "max-size-time=0",
            "leaky=downstream", "!",

            "videoconvert", "!",
            "videoscale", "!",

            (
                f"video/x-raw,format=BGR,"
                f"width={self.width},"
                f"height={self.height}"
            ), "!",

            "fdsink",
            "fd=1",
            "sync=false"
        ]

    def start(self):
        """Inicia GStreamer y el hilo de recepción."""

        if self.activo:
            return self

        if shutil.which("gst-launch-1.0") is None:
            raise RuntimeError(
                "GStreamer no está instalado o no está en PATH"
            )

        self.proceso = subprocess.Popen(
            self.pipeline(),
            stdout=subprocess.PIPE,
            bufsize=0
        )

        self.activo = True

        self.hilo = threading.Thread(
            target=self._recibir,
            daemon=True
        )

        self.hilo.start()

        return self

    def _leer_bytes(self, cantidad):
        """Lee exactamente los bytes de un fotograma."""

        datos = bytearray(cantidad)
        vista = memoryview(datos)

        recibidos = 0

        while recibidos < cantidad and self.activo:

            n = self.proceso.stdout.readinto(
                vista[recibidos:]
            )

            if not n:
                return None

            recibidos += n

        if recibidos != cantidad:
            return None

        return datos

    def _recibir(self):
        """Recibe imágenes continuamente."""

        try:
            while self.activo:

                datos = self._leer_bytes(self.bytes_frame)

                if datos is None:
                    break

                img = np.frombuffer(
                    datos,
                    dtype=np.uint8
                ).reshape(
                    self.height,
                    self.width,
                    3
                )

                with self.condicion:

                    # Reemplaza la imagen anterior.
                    self.frame = img

                    self.timestamp = time.monotonic()
                    self.secuencia += 1

                    self.condicion.notify_all()

        finally:
            with self.condicion:
                self.activo = False
                self.condicion.notify_all()

    def read(self, timeout=1.0):
        """Devuelve una imagen nueva, si está disponible."""

        with self.condicion:

            self.condicion.wait_for(
                lambda: (
                    self.secuencia != self.ultima_leida
                    or not self.activo
                ),
                timeout=timeout
            )

            if (
                not self.activo
                or self.frame is None
                or self.secuencia == self.ultima_leida
            ):
                return False, None

            self.ultima_leida = self.secuencia

            return True, self.frame.copy()

    def edad_frame(self):
        """Tiempo desde la recepción del último frame."""

        with self.condicion:
            if self.timestamp == 0:
                return float("inf")

            return time.monotonic() - self.timestamp

    def isOpened(self):
        return (
            self.activo
            and self.proceso is not None
            and self.proceso.poll() is None
        )

    def release(self):
        """Detiene la recepción y libera los recursos."""

        with self.condicion:
            self.activo = False
            self.condicion.notify_all()

        if self.proceso is not None:

            if self.proceso.poll() is None:

                self.proceso.terminate()

                try:
                    self.proceso.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.proceso.kill()
                    self.proceso.wait()

        if self.hilo is not None:
            self.hilo.join(timeout=3)

        if self.proceso is not None:
            self.proceso.stdout.close()


def argumentos_receptor():

    parser = argparse.ArgumentParser(
        description="Receptor de streaming RTP/UDP"
    )

    parser.add_argument("--port", type=int, default=5004)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--latency", type=int, default=50)
    parser.add_argument("--bind", default="0.0.0.0")

    return parser.parse_args()


def main():
    """Permite probar el receptor sin utilizar YOLO."""

    import cv2

    args = argumentos_receptor()

    receptor = ReceptorVideo(
        port=args.port,
        width=args.width,
        height=args.height,
        latency=args.latency,
        bind=args.bind
    ).start()

    ventana = "Streaming RTP"

    cv2.namedWindow(ventana, cv2.WINDOW_NORMAL)

    print(f"Escuchando en UDP {args.port}")
    print("Esperando vídeo...")

    try:
        while receptor.isOpened():

            ok, img = receptor.read(timeout=0.1)

            if ok:
                cv2.imshow(ventana, img)

            elif receptor.edad_frame() > 1.0:
                # Si el receptor no recibe frames durante más de 1 segundo, muestra un aviso de "SIN SEÑAL".
                # aviso es una imagen negra con el texto "SIN SEÑAL" en rojo.
                aviso = np.zeros(
                    (args.height, args.width, 3),
                    dtype=np.uint8
                )

                cv2.putText(
                    aviso,
                    "SIN SEÑAL",
                    (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 0, 255),
                    3
                )

                cv2.imshow(ventana, aviso)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        pass

    finally:
        receptor.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()