#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Emisor de vídeo H.264 mediante RTP/UDP."""

import argparse
import subprocess
import shutil


def argumentos():
    parser = argparse.ArgumentParser(
        description="Emisor de streaming RTP/UDP"
    )

    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=5004)

    parser.add_argument(
        "--camera",
        choices=["usb", "csi", "test"],
        default="usb"
    )

    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--bitrate", type=int, default=3000)
    parser.add_argument("--gop", type=int, default=15)

    return parser.parse_args()


def pipeline_rtp(args):
    """Empaquetado H.264 en RTP y envío mediante UDP."""

    return [
        "h264parse", "!",
        "rtph264pay",
        "config-interval=-1",
        "pt=96",
        "mtu=1200", "!",
        "udpsink",
        f"host={args.host}",
        f"port={args.port}",
        "sync=false",
        "async=false"
    ]


def pipeline_codificador(args):
    """Codificador H.264 de baja latencia por software."""

    return [
        "videoconvert", "!",
        "video/x-raw,format=I420", "!",
        "x264enc",
        "tune=zerolatency",
        "speed-preset=ultrafast",
        f"bitrate={args.bitrate}",
        f"key-int-max={args.gop}",
        "bframes=0",
        "byte-stream=true", "!"
    ]


def pipeline_usb(args):
    """Pipeline para cámaras USB mediante V4L2."""

    return [
        "gst-launch-1.0", "-q",
        "v4l2src",
        f"device={args.device}",
        "do-timestamp=true", "!",
        (
            f"video/x-raw,width={args.width},"
            f"height={args.height},"
            f"framerate={args.fps}/1"
        ), "!",
        "queue",
        "max-size-buffers=1",
        "max-size-bytes=0",
        "max-size-time=0",
        "leaky=downstream", "!"
    ] + pipeline_codificador(args) + pipeline_rtp(args)


def pipeline_test(args):
    """Genera vídeo artificial para probar la comunicación."""

    return [
        "gst-launch-1.0", "-q",
        "videotestsrc",
        "is-live=true",
        "pattern=ball", "!",
        (
            f"video/x-raw,width={args.width},"
            f"height={args.height},"
            f"framerate={args.fps}/1"
        ), "!"
    ] + pipeline_codificador(args) + pipeline_rtp(args)


def comando_csi(args):
    """Captura CSI y codificación directa en H.264."""

    programa = (
        shutil.which("rpicam-vid")
        or shutil.which("libcamera-vid")
    )
    if programa is None:
        raise RuntimeError(
            "No se encuentra rpicam-vid ni libcamera-vid"
        )

    return [
        programa,
        "--timeout", "0",
        "--nopreview",
        "--width", str(args.width),
        "--height", str(args.height),
        "--framerate", str(args.fps),
        "--codec", "h264",
        "--profile", "baseline",
        "--bitrate", str(args.bitrate * 1000),
        "--intra", str(args.gop),
        "--inline",
        "--flush",
        "--output", "-"
    ]


def detener(proceso):
    """Finaliza un proceso correctamente."""

    if proceso is not None and proceso.poll() is None:
        proceso.terminate()

        try:
            proceso.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proceso.kill()
            proceso.wait()


def main():
    args = argumentos()

    if shutil.which("gst-launch-1.0") is None:
        raise SystemExit("GStreamer no está instalado")

    captura = None
    streaming = None

    print("========== EMISOR RTP ==========")
    print(f"Destino:    {args.host}:{args.port}")
    print(f"Cámara:     {args.camera}")
    print(f"Resolución: {args.width}x{args.height}")
    print(f"FPS:        {args.fps}")
    print(f"Bitrate:    {args.bitrate} kbps")
    print(f"GOP:        {args.gop}")
    print("Códec:      H.264")
    print("Protocolo:  RTP/UDP")
    print("================================")

    try:
        if args.camera == "csi":

            captura = subprocess.Popen(
                comando_csi(args),
                stdout=subprocess.PIPE
            )
            # la variable captura es un proceso que genera H.264 y hace pipe a la entrada de gst-launch-1.0. 
            # Pipe es un mecanismo de comunicación entre procesos.
            # La salida de captura se conecta a la entrada de gst-launch-1.0, que empaqueta el H.264 en RTP y lo envía por UDP al host y puerto especificados.
            # gst-launch-1.0 es una herramienta de línea de comandos que permite construir y ejecutar pipelines de GStreamer.

            #comando es una lista de strings que contiene el comando y sus argumentos para ejecutar gst-launch-1.0 con los elementos necesarios para empaquetar H.264 en RTP y enviarlo por UDP.
            comando = [
                "gst-launch-1.0", "-q",
                "fdsrc", "fd=0", "!",
                "video/x-h264,stream-format=byte-stream", "!"
            ] + pipeline_rtp(args)

            streaming = subprocess.Popen(
                comando,
                stdin=captura.stdout
            )

            captura.stdout.close()

        elif args.camera == "usb":
            streaming = subprocess.Popen(
                pipeline_usb(args)
            )

        else:
            streaming = subprocess.Popen(
                pipeline_test(args)
            )

        print("Transmitiendo vídeo...")
        print("Pulsa Ctrl+C para detenerlo")

        codigo = streaming.wait()

        if codigo != 0:
            print(f"GStreamer terminó con código {codigo}")

    except KeyboardInterrupt:
        print("\nDeteniendo transmisión...")

    finally:
        detener(streaming)
        detener(captura)

        print("Emisor finalizado")


if __name__ == "__main__":
    main()