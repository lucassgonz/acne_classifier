"""
Treina o YOLOv8n para detectar as 5 regioes faciais (testa/nariz/queixo/
bochecha esq/dir), usando as caixas do MediaPipe como alvo -- ver
utils/build_yolo_dataset.py. Roda em CPU deliberadamente para nao
competir pelo MPS com o treino de classificacao rodando em paralelo.

Uso:
    cd training
    python run_yolo_train.py
"""
import os

from ultralytics import YOLO

# Caminho ABSOLUTO deliberado: a config global do ultralytics nesta maquina
# (~/Library/Application Support/Ultralytics/settings.json) tem runs_dir
# apontando para outro projeto do usuario (contagem-cajueiros) -- um
# caminho relativo aqui seria juntado aquele runs_dir errado.
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(HERE, "results", "yolo")

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")
    model.train(
        data=os.path.join(HERE, "..", "data", "yolo_regions", "data.yaml"),
        epochs=80,
        imgsz=640,
        batch=16,
        device="cpu",
        project=PROJECT_DIR,
        name="region_detector",
        patience=15,
        verbose=True,
    )
    metrics = model.val(data=os.path.join(HERE, "..", "data", "yolo_regions", "data.yaml"), split="test", device="cpu")
    print("TEST METRICS:", metrics.results_dict)
