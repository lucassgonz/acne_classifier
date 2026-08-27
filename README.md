
# 🧪 AcneNet: Classificação de Gravidade da Acne com Segmentação Facial

Este projeto visa classificar a gravidade da acne em diferentes regiões do rosto (testa, bochechas, nariz e queixo) utilizando uma abordagem baseada em **YOLOv8 para segmentação facial** e **ResNet18 para classificação**.

---

## 📂 Estrutura do Projeto

```
acne_1024/              # Dataset original (Acne1024)
backups/                # Backups de modelos antigos
data/                   
  ├── crops/            # Imagens segmentadas por região com YOLO
  ├── final/            # Dados segmentados e splitados por região + gravidade
  └── raw/              # Imagens cruas consolidadas de várias fontes
data_test/              # Área opcional de testes
Dataset/                # Outra fonte de dados (organizada por pastas Level)
segment/
  ├── segment_face/     # Ambiente virtual para segmentação (YOLO)
  ├── requirements_segment.txt
  └── yolo_train_and_segment_v2.ipynb
training/
  ├── acne-training/    # Dados de treino (separados)
  ├── models/           # Modelos treinados
  ├── models_backup/    # Backup dos modelos antigos
  ├── notebooks/        # Notebooks para treino e avaliação
  ├── results/          # Resultados do treino
  ├── utils/            # Scripts utilitários
  └── requirements_training.txt
.gitignore
README.md
```

---

## ⚙️ Ambientes Virtuais

- `segment_face`: Para segmentação facial com YOLOv8
- `acne-training`: Para treinamento de classificadores (PyTorch/ResNet18)

> Use `requirements_segment.txt` e `requirements_training.txt` para instalar os pacotes em cada ambiente.

---

## 🧩 Pipeline Geral

1. **📥 Consolidação de Dados**
   - Script: `utils/prepare_dataset.py`
   - Copia imagens de `acne_1024` e `Dataset/Train|Validation` para `data/raw/`

2. **🧠 Segmentação Facial com YOLOv8**
   - Notebook: `segment/yolo_train_and_segment_v2.ipynb`
   - Treina YOLO com anotações de regiões faciais e gera `data/crops/{região}/{imagem}`

3. **🔀 Split e Classificação por Gravidade**
   - Script: `utils/distribute_unique_split.py`
   - Lê as imagens em `data/crops` e divide em `train`, `val` e `test`, baseando-se no nome da imagem (`levleX_`, etc.)

4. **📈 Treinamento de Modelos**
   - Notebooks:
     - `AcneNet_TrainAll_EvalTest.ipynb`: Treino global
     - `ResNet18_All_Regions_Training.ipynb`: Treina por região
     - `ResNet18_Comparison_Training.ipynb`: Avaliação comparativa
     - `train_focal_augment_earlystop.ipynb`: Versão com focal loss, augmentation e early stopping

5. **📊 Avaliação dos Resultados**
   - Métricas: F1-score, precision, recall por região e gravidade
   - Resultados salvos em `training/results/` e modelos em `training/models/`

---

## 🛠 Scripts Auxiliares

- `count_classes.py`: Conta o número de imagens por classe e região
- `segment_combined_dataset.py`: Segmenta e salva todas as regiões detectadas
- `augment_class_X_train.py`: Aplica augmentations seletivas por classe

---

## ✅ Requisitos

### Segmentação

```bash
# Ative o ambiente segment_face
pip install -r segment/requirements_segment.txt
```

### Treinamento

```bash
# Ative o ambiente acne-training
pip install -r training/requirements_training.txt
```

---

## 📝 Observações

- Algumas classes (por exemplo, gravidade 3 em forehead) são naturalmente desbalanceadas.
- Utiliza `EarlyStopping` e `Data Augmentation` para compensar o desequilíbrio.
- As imagens são redimensionadas para **224x224** em todos os pontos do pipeline.

---

## 🤝 Contribuição

Sugestões de melhoria, testes em novos datasets e ajustes de arquitetura são bem-vindos!

---

## 🧑‍💻 Autor

**Lucas Gonzaga** — Técnico em Informática Integrado (IFCE) | IA na Saúde | Fullstack | IA + Drones
