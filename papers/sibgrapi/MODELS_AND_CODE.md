# Modelos e códigos da ablação (artigo 118)

## Modelos (pesos `.pth`)

| Arquivo | Papel |
|---|---|
| [`results/best_withembed.pth`](results/best_withembed.pth) | **WithEmbed** (cópia de `pibic_final/training/notebooks/best_robust_model.pth`) — 95.2% |
| [`results/best_noembed.pth`](results/best_noembed.pth) | **NoEmbed from scratch** — 48.4% |
| [`results/best_noembed_transfer.pth`](results/best_noembed_transfer.pth) | NoEmbed com backbone transferido do WithEmbed — 96.5% |
| [`results/ablation_metrics.json`](results/ablation_metrics.json) | Métricas, deltas e matriz de embeddings |

Arquitetura de referência no projeto original:  
`/Users/GonLu/Documents/Projetos/pibic_final/training/model.py`

## Códigos

| Script | Função |
|---|---|
| [`scripts/run_embedding_ablation.py`](scripts/run_embedding_ablation.py) | Dataset, WithEmbed/NoEmbed, treino, avaliação, JSON |
| [`scripts/generate_sibgrapi_figures.py`](scripts/generate_sibgrapi_figures.py) | Figuras do artigo (ablation, similaridade, pipeline, app) |
| Notebook original WithEmbed | `/Users/GonLu/Documents/Projetos/pibic_final/training/notebooks/ResNet18_SingleModel.ipynb` |
| Dados | `/Users/GonLu/Documents/Projetos/pibic_final/data/final/{region}/{train,val,test}/{0..3}/` |

## Como reproduzir

```bash
cd /Users/GonLu/Documents/pessoal/artigo-118-revisao
source .venv/bin/activate

# Ablação (WithEmbed carrega pesos; treina NoEmbed)
python scripts/run_embedding_ablation.py --epochs 40 --batch-size 16

# Figuras
python scripts/generate_sibgrapi_figures.py

# PDF
export PATH="/Library/TeX/texbin:$PATH"
pdflatex 118.tex && pdflatex 118.tex
```

## Conclusão empírica (resumo)

- Embeddings são críticos no **treino do zero** (+46.8 pp de accuracy).
- Com backbone já treinado sob embeddings, NoEmbed por transferência continua forte (96.5%).
- Ganho maior em cheek/chin; embeddings aprendidos são quase ortogonais entre regiões.
