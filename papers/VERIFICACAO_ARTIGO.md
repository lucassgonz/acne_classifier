# Verificacao do artigo CBEB 2026

Verificacao cruzada entre o texto recebido e o codigo/dados em `pibic_final`.

## Confirmado (condiz com o projeto)

| Afirmacao | Evidencia |
|-----------|-----------|
| Arquitetura Region-Aware ResNet-18 com embedding de 64 dimensoes para 5 regioes | `training/notebooks/ResNet18_SingleModel.ipynb`, `training/model.py` |
| 4 classes de gravidade (0-3) | Dataset em `data/final/{regiao}/{split}/{0..3}/` |
| 5 regioes: forehead, chin, nose, left_cheek, right_cheek | Constante `REGIONS` nos notebooks |
| Conjunto de teste com 1.090 imagens | Reavaliado em 26/06/2026: 1.090 amostras |
| Matriz de confusao da Figura 4 | Valores identicos ao modelo `best_robust_model.pth` |
| Metricas globais (~95% accuracy/F1) | Accuracy 95,23%; weighted F1 0,952 |
| Metricas por regiao (Tabela IV) | Forehead F1=0,99; Chin/Left cheek F1=0,93; demais conforme artigo |
| Hiperparametros: Adam, lr=0,0001, batch=16, 40 epocas, ReduceLROnPlateau | Notebook `ResNet18_SingleModel.ipynb` |
| Augmentations: flip, rotacao, color jitter, blur, random erasing | Mesmo notebook |
| Segmentacao facial por YOLOv8 antes da classificacao | `segment/yolo_train_and_segment_v2.ipynb` |
| Exportacao mobile `.ptl` | `training/notebooks/acne_model.ptl` (43 MB, carrega e infere) |
| Prototipo Flutter existe | `acne_classifier_app/` (nao integrado ao modelo) |

## Matriz de confusao (reprodutivel)

```
[[301  15   1   1]
 [ 13 386  13   0]
 [  0   9 101   0]
 [  0   0   0 250]]
```

Accuracy: **95,23%**

## Inconsistencias ou exageros (corrigir no texto)

### 1. Imagem facial completa como entrada
**Artigo diz:** "a imagem facial completa tambem foi utilizada como entrada".
**Codigo:** o modelo recebe apenas **crops de regiao** (224x224) + indice da regiao. Nao ha branch para face inteira.

### 2. Pre-treinamento ImageNet
**Versao SIBGRAPI em ingles diz:** "pretrained on ImageNet".
**Codigo:** `models.resnet18(weights=None)` — treinamento **from scratch**, sem pesos ImageNet.

### 3. Tabela I — contagem de imagens
**Artigo diz:** ACNE04 e AcneLevel com 1.457 imagens cada.
**Repositorio:** `Dataset/` tem 1.457 imagens; `acne_1024/` tem ~1.406. Os numeros nao sao identicos para ambas as bases.

### 4. Papel de cada dataset
**Artigo sugere:** AcneLevel para treino/val/test e ACNE04 para avaliacao.
**Pipeline real:** ambas sao consolidadas em `data/raw`, segmentadas por YOLO e divididas por gravidade (70/15/15). Nao ha protocolo separado ACNE04-only para teste.

### 5. Hardware
**Artigo CBEB:** NVIDIA RTX 3060.
**README/SIBGRAPI draft:** RTX 4070 em alguns trechos.
Confirmar qual GPU foi usada de fato e padronizar.

### 6. Aplicativo movel integrado
**Artigo diz:** pre-processamento automatico e classificacao no app.
**Codigo:** app captura fotos e simula resultado (`Future.delayed`); **nao executa o modelo** nem chama API de inferencia. Supabase e ML API sao placeholders.

### 7. Estabilidade (30 execucoes)
**Artigo diz:** acuracia media 94,41% ± 0,46%.
**Reexecucao em 26/06/2026:** media **93,28%**, desvio **0,46%** (30 runs com augmentations leves no teste). O desvio coincide; a media difere ~1,1 p.p. Recomenda-se usar 93,3% ou reexecutar a celula exata do notebook para confirmar.

### 8. Nomenclatura clinica das classes
**Artigo usa:** Mild / Moderate / Severe / Very Severe.
**Dataset usa:** Level 0-3 (`levle0`, `Level1`, etc.). A correspondencia com escala Hayashi deve ser explicitada; evitar tratar Level 0 como "Mild" sem justificativa clinica.

### 9. Deteccao facial no pre-processamento
**Artigo diz:** "deteccao da regiao facial".
**Pipeline:** YOLO detecta **sub-regioes** (testa, nariz, etc.), nao apenas bounding box da face inteira. Alternativa MediaPipe em `segment_combined_dataset.py` nao e o pipeline principal documentado.

## Itens ausentes no artigo

1. **Script de inferencia** (`training/predict.py`) — util para reproducao
2. **Detalhes do split** (70% train, 15% val, 15% test) e unificacao train+val no treino final
3. **Early stopping** e criterio de salvamento do melhor modelo (robust accuracy)
4. **Class-weighted cross-entropy** como resposta ao desbalanceamento
5. **Limitacao:** modelo nao faz deteccao/contagem de lesoes individuais
6. **Arquivo de pesos** usado nos resultados: `best_robust_model.pth`

## Conclusao

O **nucleo experimental e verdadeiro**: modelo treinado, metricas e matriz de confusao batem com o codigo. As correcoes principais sao: remover entrada de face inteira, remover pre-treinamento ImageNet, alinhar descricao do app (prototipo), revisar tabela de datasets e padronizar hardware.

Os artigos gerados em `papers/cbeb/` e `papers/sibgrapi/` ja incorporam essas correcoes.
