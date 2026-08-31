# Guia de Anotação de Gravidade de Acne (Escala Hayashi, 0-3)

Guia prático para classificar novas imagens no Roboflow, usando como referência visual as imagens já rotuladas do ACNE04 (`acne_1024/`, `metadata.jsonl`).

## Antes de começar

**Não tente decorar critérios clínicos abstratos.** O jeito mais confiável de manter consistência é comparar cada foto nova lado a lado com exemplos já classificados do ACNE04. Abra 3-4 exemplos de cada nível (caminhos abaixo) e deixe abertos como referência enquanto anota.

## Os 4 níveis, com exemplos reais do dataset

### Nível 0 — Sem acne / mínima
Pele limpa ou quase limpa. Sem lesões inflamadas visíveis, no máximo 1-2 marcas isoladas muito pequenas.

Exemplos: `acne_1024/levle0_144.jpg`, `levle0_164.jpg`, `levle0_289.jpg`, `levle0_462.jpg`

### Nível 1 — Leve
Lesões inflamadas (pápulas/pústulas pequenas, vermelhas) espalhadas, mas em número reduzido — geralmente visíveis mas não dominam a foto. Pele majoritariamente livre entre as lesões.

Exemplos: `acne_1024/levle1_564.jpg`, `levle1_365.jpg`, `levle1_97.jpg`, `levle1_240.jpg`

### Nível 2 — Moderada
Lesões mais numerosas e densas, cobrindo áreas maiores (testa e/ou bochechas). Mistura de pápulas e pústulas, vermelhidão mais evidente.

Exemplos: `acne_1024/levle2_134.jpg`, `levle2_32.jpg`, `levle2_60.jpg`, `levle2_135.jpg`

### Nível 3 — Severa
Lesões extensas e densas cobrindo grande parte do rosto, inflamação intensa, vermelhidão generalizada. Pode incluir lesões maiores/nodulares.

Exemplos: `acne_1024/levle3_87.jpg`, `levle3_85.jpg`, `levle3_142.jpg`, `levle3_83.jpg`

## Dicas práticas

1. **Anote em ordem aleatória**, não sequencial — evita viés de "calibrar" sua régua gradualmente e depois ficar inconsistente com as primeiras fotos.
2. **Na dúvida entre dois níveis adjacentes**, prefira o nível mais comum no ACNE04 original (Nível 1 é o mais frequente — 623 de 1.406 imagens; Nível 0 tem 491; Nível 2 tem 177; Nível 3 tem 115). Isso mantém a proporção de classes parecida com o dataset original, o que ajuda o modelo.
3. **Se possível, peça pra uma segunda pessoa anotar uma amostra** (~10%) das mesmas fotos, sem ver sua classificação. Se vocês concordarem na maioria, a anotação está confiável. Divergências grandes indicam critério ambíguo — descarte essas fotos ou reveja juntos.
4. **Descarte fotos ruins**: fora de foco, iluminação muito escura/estourada, rosto parcialmente coberto, ou maquiagem/filtro cobrindo a pele — isso não ajuda o modelo, só adiciona ruído.
5. **Faça pausas.** Cansaço visual faz a régua "derivar" ao longo de uma sessão longa de anotação.

## Formato de saída esperado

Ao exportar do Roboflow, organize em pastas por classe (compatível com o pipeline atual):

```
novo_dataset/
  0/   (imagens nível 0)
  1/   (imagens nível 1)
  2/   (imagens nível 2)
  3/   (imagens nível 3)
```

Isso é o mesmo formato que `training/utils/prepare_acne04_split.py` já espera — quando tiver um lote pronto, é só rodar o script de preparação apontando pra essa nova pasta, ou me pedir pra adaptar o pipeline pra combinar com o ACNE04 existente (sempre respeitando split por imagem, sem vazamento).
