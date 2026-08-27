# AcneNet: Classificação de Gravidade da Acne por Região Facial

Classificação automática da gravidade da acne (níveis 0–3) em cinco regiões do rosto, combinando segmentação facial com YOLOv8 e classificação com ResNet18.

**Regiões**: testa, nariz, queixo, bochecha esquerda, bochecha direita  
**Classes**: 0 = sem acne, 1 = leve, 2 = moderada, 3 = severa

---

## Arquitetura

```
Imagem facial
     │
     ▼
YOLOv8 (segmentação facial)
     │ crops 224×224 por região
     ▼
OptimizedMultiRegionResNet18
  ├── ResNet18 backbone  →  feature vector (512-d)
  ├── Region embedding   →  embedding (64-d)
  └── MLP classifier     →  logits (4 classes)
```

A classe `OptimizedMultiRegionResNet18` está definida em [`training/model.py`](training/model.py). O modelo recebe o crop da região e o índice da região simultaneamente, permitindo que a mesma rede sirva as cinco regiões com embeddings distintos.

---

## Estrutura do Repositório

```
training/
  model.py                     Definição da arquitetura
  predict.py                   Script de inferência (linha de comando)
  run_ablation.py              Estudo de ablação: com vs. sem embedding de região
  models/                      Pesos finais treinados (um por região)
    resnet18_forehead_final.pth
    resnet18_nose_final.pth
    resnet18_chin_final.pth
    resnet18_left_cheek_final.pth
    resnet18_right_cheek_final.pth
  notebooks/                   Notebooks de treino e avaliação
    ResNet18_SingleModel.ipynb     Treino e avaliação principal
    FinalModel.ipynb               Pipeline completo
    AcneNet_TrainAll_EvalTest.ipynb
    ResNet18_All_Regions_Training.ipynb
    ResNet18_Comparison_Training.ipynb
    train_focal_augment_earlystop.ipynb
  utils/                       Scripts de preparação de dados
    prepare_acne04_split.py    Split treino/val/teste (70/15/15) do ACNE04
    augment_class_0_train.py
    augment_class_1_train.py
    augment_class_3_train.py
    count_classes.py
  results/
    resultado_comparativo_modelos.csv
    comparacao_f1_por_classe.png
  requirements_training.txt
segment/
  yolo_train_and_segment_v2.ipynb   Segmentação facial com YOLOv8
  requirements_segment.txt
acne_classifier_app/               App Flutter (protótipo mobile)
papers/                            Artigos científicos (CBEB 2026, SIBGRAPI 2026)
images/                            Figuras de documentação
```

> O dataset original (ACNE04) não está incluído por tamanho. Para re-treinar, baixe os dados e siga o pipeline abaixo.

---

## Requisitos de Hardware

- **GPU NVIDIA** com CUDA 12.1, ou
- **Apple Silicon** (M1 ou superior) via backend MPS do PyTorch
- Python 3.10+

O código detecta o dispositivo automaticamente (`cuda` → `mps` → `cpu`).

---

## Configuração do Ambiente

### 1. Inferência e Treinamento

**GPU NVIDIA (CUDA):**

```bash
python -m venv venv

# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install -r training/requirements_training.txt
```

**Apple Silicon (MPS):**

```bash
python -m venv venv
source venv/bin/activate
pip install -r training/requirements_mps.txt
```

### 2. Segmentação facial (YOLOv8)

```bash
python -m venv venv_segment

# Windows
venv_segment\Scripts\activate
# Linux / macOS
source venv_segment/bin/activate

pip install -r segment/requirements_segment.txt
```

---

## Inferência (modelo treinado)

Os pesos finais estão em `training/models/`. Cada arquivo corresponde a uma região facial.

```bash
cd training
python predict.py caminho/para/imagem.jpg --region forehead --weights models/resnet18_forehead_final.pth
```

**Regiões válidas:** `forehead`, `nose`, `chin`, `left_cheek`, `right_cheek`

Exemplo de saída:

```
Regiao: forehead
Classe prevista: nivel_1 (1)
  nivel_0: 0.0821
  nivel_1: 0.7134
  nivel_2: 0.1703
  nivel_3: 0.0342
```

O script detecta GPU automaticamente. Sem GPU, roda em CPU.

---

## Pipeline de Treinamento (do zero)

O split treino/val/teste é feito **antes** da segmentação, ao nível da imagem original — assim todos os crops regionais de uma mesma foto ficam no mesmo split, sem vazamento de dados.

1. **Dividir o ACNE04 em treino/val/teste (70/15/15)**
   ```bash
   python training/utils/prepare_acne04_split.py --acne04 acne_1024 --output data/acne04_images
   ```

2. **Segmentar regiões faciais**
   Execute o notebook `segment/yolo_train_and_segment_v2.ipynb` sobre `data/acne04_images/{train,val,test}/`, preservando os subdiretórios de split.
   Gera crops 224×224 em `data/final/{região}/{split}/{classe}/`.

3. **Treinar o classificador**
   Execute `training/notebooks/ResNet18_SingleModel.ipynb`.  
   Salva o melhor modelo em `training/models/`.

4. **Estudo de ablação (com vs. sem embedding de região)**
   ```bash
   python training/run_ablation.py --data-dir data/final --seeds 5
   ```
   Roda múltiplas seeds para cada configuração e reporta média ± desvio-padrão de accuracy e F1.

5. **Avaliar resultados**
   Execute `training/notebooks/AcneNet_TrainAll_EvalTest.ipynb`.  
   Gera métricas em `training/results/`.

---

## Modelo

| Entrada | Formato |
|---------|---------|
| Imagem crop | tensor `[batch, 3, 224, 224]` |
| Índice da região | tensor `[batch]` (int) |

| Saída | Formato |
|-------|---------|
| Logits por classe | tensor `[batch, 4]` |

Índices de região: `0=forehead`, `1=chin`, `2=nose`, `3=left_cheek`, `4=right_cheek`

---

## Resultados

Consulte `training/results/resultado_comparativo_modelos.csv` para métricas detalhadas (F1, precision, recall) por região e gravidade.

---

## Autor

Lucas Gonzaga — IFCE | PIBIC
