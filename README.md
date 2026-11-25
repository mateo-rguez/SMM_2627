# Detección de vehículos y personas con YOLO

Reproductor de vídeo que detecta **personas y vehículos** con YOLO11 y dibuja, para cada pareja vehículo–persona, una línea entre los bordes más cercanos de sus cajas junto con la distancia en píxeles.

## Instalación

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install ultralytics
```

`ultralytics` instala también PyTorch, OpenCV y NumPy. El modelo `yolo11n.pt` se descarga automáticamente la primera vez.

Para comprobar si se usa la GPU:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Uso

1. Coloca un vídeo en la carpeta del proyecto con el nombre `video.mp4` (o cambia `VIDEO` en el script).
2. Ejecuta:

```bash
python recognition.py
```

### Controles

| Acción | Control |
|---|---|
| Pausar / reanudar | Botón **PAUSA/PLAY** o `Espacio` |
| Ir a otro momento | Barra **Tiempo** |
| Salir | `Q` o cerrar la ventana |


## Limitaciones

- Las distancias se miden en **píxeles**, no en metros: con perspectiva, dos objetos pueden parecer cercanos en la imagen estando lejos en la realidad. Funciona mejor con cámaras fijas y elevadas.
- Las personas tapadas por un vehículo pueden no detectarse.