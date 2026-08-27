# AcneNet - Classificacao de Gravidade da Acne

Projeto de classificacao da gravidade da acne (niveis 0 a 3) em regioes faciais, usando segmentacao por YOLOv8 e classificacao com ResNet18 multi-regiao.

## Arquitetura

- **Segmentacao**: YOLOv8 detecta regioes do rosto (testa, nariz, queixo, bochechas) e gera crops 224x224
- **Classificacao**: `OptimizedMultiRegionResNet18` combina features da imagem com embedding da regiao facial
- **Classes**: 0 (sem acne), 1 (leve), 2 (moderada), 3 (severa)

## Estrutura

```
models/
  best_robust_model.pth    Pesos treinados (PyTorch)
  acne_model.ptl           Modelo exportado para mobile
training/
  model.py                 Definicao da arquitetura
  predict.py               Script de inferencia
  notebooks/
    ResNet18_SingleModel.ipynb   Treino e avaliacao principal
    FinalModel.ipynb             Versao final do pipeline
  utils/                   Scripts de preparacao de dados
segment/
  yolo_train_and_segment_v2.ipynb   Segmentacao facial com YOLOv8
requirements_training.txt
requirements_segment.txt
```

## Requisitos

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements_training.txt
```

Para segmentacao:

```bash
pip install -r requirements_segment.txt
```

## Inferencia

```bash
cd training
python predict.py caminho/imagem.jpg --region forehead
```

Regioes validas: `forehead`, `chin`, `nose`, `left_cheek`, `right_cheek`

## Dados

O pipeline espera imagens organizadas em:

```
data/final/{regiao}/{train|val|test}/{0|1|2|3}/
```

Os datasets originais (Acne1024, Dataset/) nao estao incluidos neste pacote por tamanho. Para retreinar, prepare os dados com os scripts em `training/utils/` e o notebook de segmentacao.

## Pipeline de treino

1. Consolidar imagens brutas
2. Segmentar regioes faciais (`segment/yolo_train_and_segment_v2.ipynb`)
3. Dividir por gravidade (`training/utils/distribute_unique_split.py`)
4. Treinar modelo (`training/notebooks/ResNet18_SingleModel.ipynb`)
5. Exportar para mobile (celula final do notebook gera `acne_model.ptl`)

## Modelo

| Arquivo | Formato | Uso |
|---------|---------|-----|
| `best_robust_model.pth` | state_dict PyTorch | Inferencia e retreino em Python |
| `acne_model.ptl` | TorchScript Mobile | Deploy em dispositivos moveis |

Entrada: tensor `[batch, 3, 224, 224]` + indice da regiao `[batch]`
Saida: logits `[batch, 4]`

## Autor

Lucas Gonzaga
