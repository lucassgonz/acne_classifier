# Region-Aware ResNet-18 for Facial Acne Severity Classification

Classificação automática da gravidade da acne (escala de Hayashi, níveis 0–3) a partir de imagens faciais, combinando segmentação anatômica por landmarks com uma ResNet-18 que incorpora embeddings de região aprendíveis.

**Regiões**: testa, nariz, queixo, bochecha esquerda, bochecha direita
**Classes**: 0 = leve, 1 = moderada, 2 = severa, 3 = muito severa

Artigo completo em [`papers/cbeb/CBEB_final.pdf`](papers/cbeb/CBEB_final.pdf) (fonte: [`CBEB_final.tex`](papers/cbeb/CBEB_final.tex)).

---

## Contexto e trabalho relacionado

Este projeto teve como uma de suas referências o trabalho *"AcneNet: A Deep CNN Based Classification Approach for Acne Classes"* ([IEEE Xplore](https://ieeexplore.ieee.org/document/8850935)), que reporta acurácia acima de 94% na classificação de classes de acne. Esse número serviu de referência inicial de expectativa de desempenho, mas não é diretamente comparável ao problema tratado aqui: escalas, protocolos de avaliação e datasets diferem. "AcneNet" é o nome do trabalho de referência, não deste projeto.

Os resultados finais deste trabalho foram obtidos sob um protocolo deliberadamente rigoroso (splits verificados sem vazamento de imagem entre treino/val/teste, ablation controlada, generalização cross-dataset), o que explica números mais conservadores do que os relatados por trabalhos que não reportam essas verificações.

---

## Arquitetura

```
Recorte de região facial (224×224) + índice da região
     │
     ▼
ResNet-18 backbone (ImageNet → SCIN → fine-tuning)  →  feature vector (512-d)
     │
     ▼
Embedding de região aprendível (5×64)  →  concatenado ao feature vector
     │
     ▼
MLP classificador (Dropout + BatchNorm)  →  logits (4 classes)
```

O modelo final combina Stochastic Weight Averaging (SWA), ensemble de 5 seeds independentes e test-time augmentation (7 views). As classes `ResNet18WithEmbed` e `ResNet18NoEmbed` (usada na ablation) estão definidas em [`training/run_ablation.py`](training/run_ablation.py).

Duas arquiteturas complementares também foram avaliadas (Seções IV-F e IV-G do artigo):
- **Whole-image baseline** ([`training/run_wholeimage.py`](training/run_wholeimage.py)): classifica a imagem inteira, sem recorte por região.
- **Multi-region fusion** ([`training/run_multiregion_fusion.py`](training/run_multiregion_fusion.py)): funde todas as regiões disponíveis da mesma imagem numa única predição, mantendo o embedding anatômico.

---

## Estrutura do Repositório

```
training/
  run_ablation.py                       Modelo principal (WithEmbed/NoEmbed) + treino + ablation
  run_wholeimage.py                     Baseline de imagem inteira
  run_multiregion_fusion.py             Fusão multi-região
  pretrain_backbone.py                  Pré-treino do backbone (ImageNet → SCIN)
  evaluate_ensemble_tta.py              Avaliação ensemble + TTA (modelo de região)
  evaluate_wholeimage_ensemble_tta.py   Avaliação ensemble + TTA (imagem inteira)
  evaluate_multiregion_ensemble_tta.py  Avaliação ensemble + TTA (fusão multi-região)
  evaluate_by_region.py                 Desempenho por região anatômica
  predict.py                            Inferência via linha de comando (um checkpoint por vez)
  model.py                              Arquitetura legada (não usada pelo pipeline atual; mantida por histórico)
  models/                               Backbones e checkpoints finais (não versionados; ver abaixo)
  results/                              Resultados de cada experimento (JSON)
  utils/
    segment_faces_mediapipe.py          Segmentação facial (MediaPipe Face Mesh, 478 pontos)
    prepare_acne04_split.py             Split treino/val/teste (70/15/15) do ACNE04
    merge_manual_data.py                Funde a coleção anotada com o ACNE04 (protocolo anti-vazamento)
    consolidate_new_sources.py          Consolidação de fontes públicas adicionais
    prepare_scin_pretrain.py            Preparação do SCIN para pré-treino do backbone
    prepare_wholeimage_data.py          Preparação do dataset de imagem inteira
segment/
  yolo_train_and_segment_v2.ipynb       Segmentação legada via YOLOv8 (histórico; substituída por MediaPipe)
acne_classifier_app/                    App Flutter (protótipo mobile)
papers/
  cbeb/CBEB_final.tex, CBEB_final.pdf   Artigo atual (CBEB)
  sibgrapi/                             Artigo SIBGRAPI relacionado
```

> O dataset (ACNE04 + coleção anotada pelos autores) não está incluído no repositório por tamanho e, no caso da coleção própria, por não ser um dataset público redistribuível. Veja a seção "Dados" abaixo.

---

## Dados

O modelo é treinado sobre a combinação de duas fontes de imagens:

| Fonte | Descrição | Link |
|---|---|---|
| ACNE04 | Dataset público, rótulos originais na escala Hayashi | [Kaggle](https://www.kaggle.com/datasets/manuelhettich/acne04) |
| Coleção anotada pelos autores | Imagens de múltiplas fontes públicas, anotadas e revisadas manualmente pelos autores na escala Hayashi | não redistribuível diretamente; ver nota abaixo |

Cada imagem é segmentada em até 5 recortes anatômicos via MediaPipe Face Mesh (`training/utils/segment_faces_mediapipe.py`). O split treino/val/teste é feito **antes** da segmentação, ao nível da imagem original, garantindo que todos os recortes de uma mesma foto fiquem no mesmo split.

**Dataset processado e checkpoints treinados**: disponibilizados à parte (fora do repositório, por tamanho) em `dataset.zip` (~620 MB) e `models.zip` (~1,1 GB). Para usar:

```
dataset.zip → extrair em data/        (cria data/final, data/wholeimage, data/manual_annotations, data/manual_crops, data/acne_1024)
models.zip  → extrair em training/models/   (cria os checkpoints finais e o backbone)
```

---

## Requisitos de Hardware

- **GPU NVIDIA** com CUDA 12.1, ou
- **Apple Silicon** (M1 ou superior) via backend MPS do PyTorch
- Python 3.9+

O código detecta o dispositivo automaticamente (`cuda` → `mps` → `cpu`).

---

## Configuração do Ambiente

**GPU NVIDIA (CUDA):**
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r training/requirements_training.txt
```

**Apple Silicon (MPS):**
```bash
python -m venv venv
source venv/bin/activate
pip install -r training/requirements_mps.txt
```
> `mediapipe` força upgrade do numpy para ≥2.0, o que quebra o `ColorJitter` do torchvision em Apple Silicon. Instale mediapipe separadamente, rode a segmentação, e reinstale `numpy<2.0,>=1.26.0` antes de treinar (ver comentário em `requirements_mps.txt`).

---

## Pipeline de Treinamento (do zero)

> Os passos 1–3 documentam como `data/final` e `data/wholeimage` foram originalmente construídos, mas o **`dataset.zip` compartilhado já contém esse resultado pronto**. Para retreinar ou reproduzir resultados, pule direto para o passo 4 (ou para "Reproduzir os resultados sem retreinar", abaixo, se também tiver o `models.zip`).

1. **Split do ACNE04 (70/15/15, por imagem)**
   ```bash
   python training/utils/prepare_acne04_split.py --acne04 acne_1024 --output data/acne04_images
   ```

2. **Segmentação facial (MediaPipe)**
   ```bash
   cd training
   python utils/segment_faces_mediapipe.py --input-dir ../data/manual_annotations --output-dir ../data/manual_crops
   ```

3. **Fundir a coleção anotada com o ACNE04** (sem vazamento entre splits)
   ```bash
   python training/utils/merge_manual_data.py --annotations-dir data/manual_annotations --crops-dir data/manual_crops --final-dir data/final
   ```

4. **Pré-treinar o backbone (ImageNet → SCIN)**
   ```bash
   python training/utils/prepare_scin_pretrain.py --output data/scin_pretrain --min-samples 30
   python training/pretrain_backbone.py --data-dir data/scin_pretrain --epochs 15
   ```

5. **Treinar o modelo principal (5 seeds, com SWA)**
   ```bash
   cd training
   for seed in 42 43 44 45 46; do
     python run_ablation.py --seed $seed --data-dir ../data/final --only-config with_embedding --swa \
       --pretrained-backbone models/backbone_scin_pretrained_v2.pth \
       --save-models-dir models/ablation_ckpt_clean_swa \
       --output results/ablation_clean_swa.json
   done
   ```

6. **Avaliar ensemble + TTA**
   ```bash
   python evaluate_ensemble_tta.py --data-dir ../data/final --checkpoints-dir models/ablation_ckpt_clean_swa --tag with_embedding --tta-views 7
   ```

---

## Reproduzir os resultados sem retreinar

O treino completo (5 seeds × 4 configurações) leva várias horas de GPU. Com `dataset.zip` e `models.zip` já extraídos (ver seção "Dados"), dá para reproduzir os números do artigo direto:

```bash
cd training

# Modelo principal (Region-Aware ResNet-18): 64,18% acc / QWK 0,718
python evaluate_ensemble_tta.py --data-dir ../data/final --checkpoints-dir models/ablation_ckpt_clean_swa --tag with_embedding --tta-views 7

# Ablation (ResNet-18 convencional, sem embedding): 64,18% acc / QWK 0,698
python evaluate_ensemble_tta.py --data-dir ../data/final --checkpoints-dir models/ablation_ckpt_clean_noembed_swa --tag no_embedding --tta-views 7

# Baseline de imagem inteira: 76,26% acc / QWK 0,829
python evaluate_wholeimage_ensemble_tta.py --data-dir ../data/wholeimage --checkpoints-dir models/wholeimage_clean_swa --tta-views 7

# Fusão multi-região: 69,06% acc / QWK 0,770
python evaluate_multiregion_ensemble_tta.py --data-dir ../data/final --checkpoints-dir models/multiregion_ckpt_v3 --tta-views 7

# Desempenho por região anatômica
python evaluate_by_region.py --data-dir ../data/final --checkpoints-dir models/ablation_ckpt_clean_swa --tag with_embedding --tta-views 7
```

Para classificar uma única imagem já recortada por região:
```bash
python predict.py caminho/para/imagem.jpg --region forehead --weights models/ablation_ckpt_clean_swa/with_embedding_seed42.pth
```
(usa um único seed, não o ensemble completo — para o resultado reportado no artigo, use os scripts `evaluate_*` acima.)

---

## Resultados

Validação cruzada agrupada em 5 folds no conjunto combinado ACNE04 + AcneLevel (2.863 imagens, sem a mesma foto em treino e teste), métricas agregadas dos 5 folds de teste, última época, sem seleção no teste:

| Modelo | Unidade de avaliação | Accuracy | Macro-F1 | QWK |
|---|---|---|---|---|
| Region-Aware ResNet-18 (WithEmbed), regiões fundidas por foto | 2.450 imagens | 70,45% | 0,663 | 0,706 |
| ResNet-18 convencional (NoEmbed), regiões fundidas por foto | 2.450 imagens | 66,94% | 0,630 | 0,674 |
| Region-Aware ResNet-18 (WithEmbed), por recorte | 8.295 recortes | 58,36% | 0,543 | 0,556 |
| Imagem inteira: ramo CBAM (SCIN + mixup + SWA + TTA, 5 seeds) | 2.863 imagens | 79,04% | 0,772 | 0,859 |
| Imagem inteira: ramo com supervisão de contagem de lesões (7 seeds) | 2.863 imagens | 77,26% | 0,745 | 0,853 |
| **Imagem inteira: média 50/50 dos dois ramos (modelo de referência final)** | 2.863 imagens | **80,20%** | **0,779** | **0,874** |

Split oficial de 5 folds do ACNE04 (1.460 imagens de teste), comparável ao JAGC (84,11%):

| Modelo | Accuracy | Macro-F1 | QWK |
|---|---|---|---|
| Ramo CBAM (1 seed) | 76,64% | 0,719 | 0,847 |
| Ramo com supervisão de contagem (5 seeds) | 81,64% | 0,788 | 0,885 |
| **Média 50/50 (mesma configuração do combinado)** | **80,55%** | **0,776** | **0,879** |

Todos os modelos usam backbone ResNet-18. O ramo com contagem segue a formulação de grading + counting do JAGC (`training/run_jagc_original_replica.py --arch resnet18`); os números finais saem de `training/evaluate_final_whole_ensemble.py` (gera `training/results/final_whole_ensemble.json` e a matriz de confusão do artigo). A classe moderada continua sendo o principal gargalo (sensibilidade de 57,6% no combinado, confundida sobretudo com leve).

Detalhes completos (ablation, generalização cross-dataset ACNE04 ↔ coleção própria, desempenho por região) no artigo em `papers/cbeb/CBEB_final.pdf`.

---

## Modelo (entrada/saída)

| Entrada | Formato |
|---------|---------|
| Recorte de região | tensor `[batch, 3, 224, 224]` |
| Índice da região | tensor `[batch]` (int, 0–4) |

| Saída | Formato |
|-------|---------|
| Logits por classe | tensor `[batch, 4]` |

Índices de região: `0=forehead`, `1=chin`, `2=nose`, `3=left_cheek`, `4=right_cheek`

---

## App Mobile

Protótipo em Flutter (`acne_classifier_app/`) que integra o modelo treinado, guiando o usuário por captura de imagem, pré-processamento, predição de gravidade por região e histórico de avaliações.

---

## Autores

Daniel C. Anselmo, Lucas G. Andrade, Enzo G. R. B. Moreira — Instituto Federal do Ceará (IFCE)
Orientação: Prof. Dr. Pedro P. Rebouças Filho, Prof. Dr. Roger M. Sarmento — PPGCC/IFCE
