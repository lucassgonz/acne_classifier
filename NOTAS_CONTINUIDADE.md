# Notas de continuidade (28-31/10/2026)

Resumo do estado do projeto para retomar em outra máquina (RTX 5060 Ti 16 GB, 16 GB de RAM). Histórico completo e detalhado de decisões anteriores: arquivo de memória do projeto (`project_acne_state.md`), se for copiado; este arquivo basta para continuar.

## Objetivo
Subir **todas** as métricas (acurácia, macro-F1, QWK) acima de 80% no dataset do artigo: **Acne04 + AcneLevel combinados (n=2863)**, aplicando o que as melhores arquiteturas do ACNE04 fazem. O modelo final e o do app devem continuar **ResNet-18** (ResNet-50 só como comparação). Aumentar épocas é permitido.

## Regras de trabalho
- Nunca apresentar resultado inválido ou com vazamento; reportar o número real e explicar a causa.
- Não narrar experimentos que falharam no artigo.
- Entregas sem qualquer vestígio de ferramentas de IA (comentários, mensagens de commit, `Co-Authored-By`).
- Não fazer `git commit` sem o usuário pedir.
- Sempre dar estimativa de tempo em tarefas longas e confirmar com `ps`/`pgrep` que o treino realmente está rodando.

## Dados (a pasta `data/` é ignorada pelo git)
- **Split oficial ACNE04**: 5 folds publicados pelos autores (`NNEW_*`), 1457 imagens (as mesmas do AcneLevel/`Dataset/`). `training/results/kfold_splits_official_acne04.json`. Comparável com a literatura (JAGC 84,11%).
- **Combinado (artigo)**: 2863 imagens = 1406 Acne04 (`a04_*`) + 1457 AcneLevel tratado (`lvl_*`), folds agrupados pela mesma foto (zero vazamento). `training/results/kfold_splits_wholeimage_v2.json` + `data/final_wholeimage_v2/manifest.json`.
- **a04-only**: só as 1406 imagens `a04_*` dos mesmos folds do combinado (diagnóstico).
- Imagens 300x300 usadas pela réplica JAGC: `data/jagc_replica_300` (oficial) e `data/jagc_replica_300_combined` (combinado). Versão 384px: `training/utils/build_jagc_replica_hires.py`.

## Resultados (última época, sem seleção no teste)
| Configuração | n | Acc | Macro-F1 | QWK |
|---|---|---|---|---|
| Artigo: ResNet-18+CBAM+SCIN+mixup+SWA+TTA, ensemble 5 seeds, combinado | 2863 | 79,04% | 77,24% | 0,859 |
| Réplica JAGC R18, combinado (3 seeds, ensemble) | 2863 | 76,74% | 73,45% | 0,847 |
| Réplica JAGC R18, split oficial (3 seeds, ensemble) | 1460 | 82,12% | 79,63% | 0,888 |
| Réplica JAGC R18, a04-only (seed 42) | 1406 | 79,23% | 77,68% | 0,868 |
| **Mistura 50/50 (artigo + réplica) no combinado** | 2863 | 79,95% | 77,24% | 0,872 |
| **Mistura 50/50 + correção de viés (validada entre folds)** | 2863 | **80,48%** | **78,35%** | 0,876 |

Achados: a receita JAGC (ImageNet + BatchNorm congelado + SGD + 120 épocas fixas, 300px, sem mixup/SWA/CBAM) é melhor no split oficial; no combinado as cópias tratadas do AcneLevel prejudicam essa receita (`lvl_` ~75% contra `a04_` ~78%); a mistura com a receita do artigo ajuda porque os erros são diferentes. A classe moderada (~62% de sensibilidade) segura o macro-F1. Correção de viés sozinha dá só ~+0,3pp no split oficial.

## Receita da réplica JAGC
`training/run_jagc_original_replica.py` (código original em `training/third_party/jagc`): ResNet-18/50 ImageNet com BN congelado, 65 classes de contagem com distribuição gaussiana (sigma 3), 3 losses KL, lam 0,6, SGD lr 0,001 momentum 0,9 wd 5e-4, batch 32, 120 épocas, LR x0,5 nas épocas 30/60/90, resize 256 + crop 224 + flip + rotação 20. Flags úteis: `--manifest-v2`, `--splits`, `--img-dir`, `--only-source {a04,lvl}`, `--tta` (salva `probs_tta`), `--resize/--crop`, `--num-workers`, `--output-dir`.

## Scripts novos
- `training/colab_queue.py`: fila de treinos em paralelo na mesma GPU (jobs `comb45..48` = combinado completo, `off45..48` = split oficial, `a04_*` = a04-only, `hires42`). Pula folds já salvos.
- `training/analyze_bias_correction.py`: correção de viés por classe com validação cruzada entre folds.
- `training/analyze_mix_combined.py`: mistura de grupos de modelos alinhada por nome + viés + métricas por fonte.

## Sessão de 04/11/2026 (PC local, RTX 5060 Ti)

Ambiente: `venv/` na raiz com Python 3.15 + torch 2.14.1+cu130 (`pip install torch==2.14.1 torchvision --index-url https://download.pytorch.org/whl/cu130`, depois `numpy scikit-learn scipy pillow tqdm`). Dados: `data/jagc_replica_300_combined` + `data/final_wholeimage_v2/manifest.json` extraídos do commit `18b35b9` (`git show 18b35b9:jagc_combined_data.tar.gz | tar xzf - --exclude='._*'`); `data/jagc_replica_300` (split oficial) extraído de `Downloads/transfer_dados_pc.zip`.

Correções para rodar no Windows:
- `run_jagc_original_replica.py`: batches montados em numpy no worker e convertidos em tensor no processo principal (o envio de tensores entre processos derrubava os workers). Receita de treino inalterada.
- `colab_queue.py`: processos filhos com `OMP/OPENBLAS/MKL_NUM_THREADS=1` (o OpenBLAS com 24 threads por worker esgotava a memória virtual; pagefile de 5 GB).

Velocidade: 3 folds em paralelo (`--parallel 3 --workers 5`) levam ~39 min por lote.

Seeds da réplica no combinado: 45 e 46 completas; 47 com folds 0-1; 48 não iniciada. Faltam 8 folds (~2 h).

Mistura + viés (cross-fit) no combinado, n=2863:
| Seeds da réplica (grupo B) | Acc | Macro-F1 | QWK |
|---|---|---|---|
| 42-44 (anterior) | 80,48% | 78,35% | 0,876 |
| 42-45 | 80,86% | 78,98% | 0,878 |
| 42-46 | 80,16% | 78,25% | 0,869 |
Sem viés, 42-46: 80,23% / 77,73% / 0,873. A oscilação de ±0,5 pp entre 4 e 5 seeds é ruído; reportar o número com todas as seeds, não o melhor.

Pós-processamentos testados em cross-fit (pesos por fold, limiares ordinais sobre o grau esperado, stacking com regressão logística): ganho máximo de ~0,3 pp sobre mistura + viés. Não resolvem macro-F1 > 80%.

Split oficial (já salvo, sem treino novo): réplica R18 sozinha com 3 seeds = 82,12% / 79,63% / 0,888, melhor que a mistura com o modelo do artigo (80,48%). O artigo ainda reporta 76,64% nesse split.

## Sessão de 05/11/2026: treinos concluídos e números finais
Concluídas `comb47`, `comb48` (combinado, réplica com 7 seeds: 42-48) e `off45`, `off46` (split oficial, réplica com 5 seeds: 42-46). O fold 1 da `off46` falhou uma vez por WinError 1450 (recurso do sistema) e foi refeito sozinho.

Números finais (`training/evaluate_final_whole_ensemble.py` -> `training/results/final_whole_ensemble.json`); a configuração final é a média 50/50 sem ajuste. A correção de viés com 7 seeds só acrescentava +0,14 pp e foi descartada.
| Protocolo | Configuração | Acc | Macro-F1 | QWK | ECE |
|---|---|---|---|---|---|
| Combinado (2863) | ramo CBAM (5 seeds) | 79,04% | 77,24% | 0,859 | 0,067 |
| Combinado | ramo contagem (7 seeds) | 77,26% | 74,51% | 0,853 | 0,041 |
| Combinado | **média 50/50** | **80,20%** | **77,87%** | **0,874** | 0,080 |
| Oficial (1460) | ramo CBAM (1 seed) | 76,64% | 71,93% | 0,847 | 0,032 |
| Oficial | ramo contagem (5 seeds) | 81,64% | 78,83% | 0,885 | 0,088 |
| Oficial | **média 50/50** | **80,55%** | **77,62%** | **0,879** | 0,091 |

`papers/cbeb/CBEB_final.tex` atualizado (resumo, III-C, Tabela 2, III-D, IV-E com a nova Tabela e `figures/fig8_ensemble.png`, IV-F, Discussão, Limitações, Conclusão). O PDF não foi recompilado: não há LaTeX nesta máquina; subir o `.tex` e a figura nova no Overleaf. O ECE do ramo CBAM no combinado dá 0,067 com 15 bins, contra 0,061 no texto antigo (binning diferente); o novo texto só usa valores calculados pelo script novo. README atualizado.

## O que falta
1. Recompilar o PDF no Overleaf e revisar o texto novo das seções IV-E/IV-F.
2. Macro-F1 ainda abaixo de 80% no combinado: a classe moderada tem ~58% de sensibilidade. Ideias: cabeça auxiliar de densidade de lesões com as 18.983 caixas do ACNE04; ramo da réplica em resolução maior (exige refazer as imagens a partir das originais de 1024 px, que não estão nesta máquina).
