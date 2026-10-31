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

## O que falta
1. Mais seeds da réplica no combinado completo (`comb45`, `comb46`, `comb47`): cada fold ~50 min na A100 com 3 em paralelo; as seeds 45 (>=3/5 folds) e 46 (parcial) foram iniciadas no Colab, que foi abandonado. Resultados parciais podem estar no Google Drive em `jagc_results/` (copiar para `training/results/kfold_wholeimage/<nome do job>/`).
2. Recalcular a mistura com mais seeds da réplica: `python analyze_mix_combined.py --a <5 dirs whole_v2_cbam_swa_tta*> --b <dirs jagc_replica_combined_resnet18*> --bias`.
3. Para fechar macro-F1 > 80% no combinado: atacar a classe moderada (ideias: cabeça auxiliar de densidade de lesões com as 18.983 caixas do ACNE04; mais diversidade no ensemble).
4. Atualizar números e texto de `papers/cbeb/CBEB_final.tex` quando os resultados fecharem.
