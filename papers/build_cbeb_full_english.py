# -*- coding: utf-8 -*-
"""Full direct PT->EN translation of CBEB article with factual corrections and English figures."""
import json
import os
import re
from docx import Document
from docx.shared import Inches, Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_JSON = os.path.join(ROOT, "papers", "cbeb", "source_blocks.json")
TRANS_JSON = os.path.join(ROOT, "papers", "cbeb", "translations_en.json")
FIGS = os.path.join(ROOT, "papers", "sibgrapi", "figs")
OUT = os.path.join(ROOT, "papers", "cbeb", "AcneNet_CBEB2026_English.docx")

# Factual corrections applied after translation (code-verified)
CORRECTIONS = [
    (r"95\.0\s*%", "95.2%"),
    (r"95,0\s*%", "95.2%"),
    (r"94\.41\s*%", "93.3%"),
    (r"94,41\s*%", "93.3%"),
    (r"Experimental evaluation", "Merged into unified pipeline"),
    (r"Training, validation and evaluation", "Merged into unified pipeline (70/15/15 split)"),
    (r"Very Severe", "Grade 3 (Severe)"),
]

# Full paragraph replacements for structurally wrong content (keep similar length)
FULL_REPLACEMENTS = {
    "Além das regiões individuais, a imagem facial completa também foi utilizada como entrada da arquitetura. Dessa forma, o modelo aprende simultaneamente características globais da face e informações específicas de cada região anatômica, aumentando sua capacidade de representar diferentes padrões de distribuição das lesões de acne.":
    "During classification, each region crop is paired with its anatomical identifier (forehead, left cheek, right cheek, nose, or chin), which is mapped to a learnable 64-dimensional embedding. The model receives only region-level crops resized to 224×224 pixels; full-face images are not used as network input. This design aligns inference with the region-level labels available in the training corpora and preserves anatomical context through the embedding layer rather than through a separate global branch.",
    "In addition to the individual regions, the full facial image was also used as input to the architecture. Thus, the model simultaneously learns global facial characteristics and region-specific information, increasing its ability to represent different acne lesion distribution patterns.":
    "During classification, each region crop is paired with its anatomical identifier (forehead, left cheek, right cheek, nose, or chin), which is mapped to a learnable 64-dimensional embedding. The model receives only region-level crops resized to 224×224 pixels; full-face images are not used as network input. This design aligns inference with the region-level labels available in the training corpora and preserves anatomical context through the embedding layer rather than through a separate global branch.",
    "A primeira etapa consistiu na detecção da região facial, removendo áreas externas ao rosto e preservando apenas a região de interesse.":
    "The first step consisted of applying a custom YOLOv8 detector trained to localize five anatomical facial sub-regions (forehead, left cheek, right cheek, nose, and chin), rather than relying solely on a global face bounding box.",
    "The first step consisted of detecting the facial region, removing areas outside the face and preserving only the region of interest.":
    "The first step consisted of applying a custom YOLOv8 detector trained to localize five anatomical facial sub-regions (forehead, left cheek, right cheek, nose, and chin), rather than relying solely on a global face bounding box.",
    "Por fim, os conjuntos de treinamento, validação e teste foram definidos de acordo com os protocolos experimentais adotados para cada base de dados, garantindo a independência entre as amostras utilizadas durante o treinamento e a avaliação do modelo e evitando vazamento de informações entre as diferentes etapas do processo experimental.":
    "Finally, after consolidating and segmenting both corpora, the data were split into training, validation, and test sets (70%, 15%, and 15%, respectively), preserving independence between samples used for training and evaluation and avoiding information leakage across experimental stages. For final model selection, the training and validation splits were merged according to the robust validation criterion implemented in the PyTorch notebook, and the best weights were saved as best_robust_model.pth.",
    "Finally, the training, validation, and test sets were defined according to the experimental protocols adopted for each dataset, ensuring independence between samples used during model training and evaluation and avoiding information leakage between different stages of the experimental process.":
    "Finally, after consolidating and segmenting both corpora, the data were split into training, validation, and test sets (70%, 15%, and 15%, respectively), preserving independence between samples used for training and evaluation and avoiding information leakage across experimental stages. For final model selection, the training and validation splits were merged according to the robust validation criterion implemented in the PyTorch notebook, and the best weights were saved as best_robust_model.pth.",
    "Para demonstrar a aplicabilidade da proposta, foi desenvolvido um protótipo móvel utilizando o framework Flutter, permitindo a captura da imagem, o pré-processamento automático e a classificação da gravidade da acne.":
    "To demonstrate the applicability of the proposal, a Flutter mobile prototype was developed, supporting login, guided image capture, and the intended teledermatology workflow for remote severity assessment. At the time of writing, on-device inference is not yet integrated; the application demonstrates interface and workflow design rather than validated clinical deployment.",
    "To demonstrate the applicability of the proposal, a mobile prototype was developed using the Flutter framework, allowing image capture, automatic preprocessing, and acne severity classification.":
    "To demonstrate the applicability of the proposal, a Flutter mobile prototype was developed, supporting login, guided image capture, and the intended teledermatology workflow for remote severity assessment. At the time of writing, on-device inference is not yet integrated; the application demonstrates interface and workflow design rather than validated clinical deployment.",
}

FIGURE_MAP = {
    "Figura 1": ("fig1_pipeline.jpg", "Figure 1 – Overview of the proposed methodology, including image acquisition, preprocessing, Region-Aware ResNet-18 architecture, and Flutter integration."),
    "Figure 1": ("fig1_pipeline.jpg", "Figure 1 – Overview of the proposed methodology, including image acquisition, preprocessing, Region-Aware ResNet-18 architecture, and Flutter integration."),
    "Figura 2": ("fig2_datasets.jpg", "Figure 2 – Representative Acne04 and Acne Level samples and facial regions used by the proposed model."),
    "Figure 2": ("fig2_datasets.jpg", "Figure 2 – Representative Acne04 and Acne Level samples and facial regions used by the proposed model."),
    "Figura 3": ("fig3_architecture.jpg", "Figure 3 – Region-Aware ResNet-18 architecture with learnable region embeddings fused before classification."),
    "Figure 3": ("fig3_architecture.jpg", "Figure 3 – Region-Aware ResNet-18 architecture with learnable region embeddings fused before classification."),
    "Figura 4": ("fig4_confusion_en.png", "Figure 4 – Confusion matrix of the Region-Aware ResNet-18 on the test set."),
    "Figure 4": ("fig4_confusion_en.png", "Figure 4 – Confusion matrix of the Region-Aware ResNet-18 on the test set."),
    "Figure 5": ("fig5_stability_en.png", "Figure 5 – Accuracy distribution obtained from 30 independent robustness evaluations."),
    "Figura 5": ("fig5_stability_en.png", "Figure 5 – Accuracy distribution obtained from 30 independent robustness evaluations."),
    "Figure 6": ("fig6_app.jpg", "Figure 6 – Mobile teledermatology prototype developed for automatic facial acne severity assessment."),
    "Figura 6": ("fig6_app.jpg", "Figure 6 – Mobile teledermatology prototype developed for automatic facial acne severity assessment."),
}

TABLE_I = [
    ["Characteristic", "Acne04", "Acne Level"],
    ["Objective", "Acne severity classification", "Acne severity classification"],
    ["Number of images", "1,406", "1,457"],
    ["Classes", "4", "4"],
    ["Severity scale", "Hayashi (grades 0–3)", "Hayashi (grades 0–3)"],
    ["Image type", "Frontal face", "Frontal face"],
    ["Organization", "Original corpus", "Folder-organized by severity"],
    ["Use in this work", "Merged into unified pipeline", "Merged into unified pipeline"],
]

TABLE_II = [
    ["Parameter", "Value"],
    ["Framework", "PyTorch"],
    ["Backbone", "ResNet-18 (trained from scratch, weights=None)"],
    ["Optimizer", "Adam"],
    ["Loss function", "Weighted Cross-Entropy"],
    ["Learning rate", "0.0001"],
    ["Scheduler", "ReduceLROnPlateau"],
    ["Batch size", "16"],
    ["Maximum epochs", "40"],
    ["Data augmentation", "Horizontal flip (p=0.5), rotation (±25°), color jitter (brightness=0.3, contrast=0.3, saturation=0.2), Gaussian blur, random erasing"],
    ["Regularization", "Batch Normalization and Dropout"],
    ["Hardware", "NVIDIA GeForce RTX 3060 (12 GB)"],
    ["Model weights", "best_robust_model.pth"],
    ["Inference script", "training/predict.py"],
]

TABLE_III = [
    ["Class", "Precision", "Recall", "F1-score", "Support"],
    ["Grade 0 (Clear)", "0.96", "0.95", "0.95", "318"],
    ["Grade 1 (Mild)", "0.94", "0.94", "0.94", "412"],
    ["Grade 2 (Moderate)", "0.88", "0.92", "0.90", "110"],
    ["Grade 3 (Severe)", "1.00", "1.00", "1.00", "250"],
    ["Weighted average", "0.95", "0.95", "0.95", "1090"],
    ["Overall accuracy", "95.2%", "—", "—", "—"],
]

TABLE_IV = [
    ["Facial region", "Precision", "Recall", "F1-score", "Support"],
    ["Forehead", "0.99", "0.99", "0.99", "93"],
    ["Chin", "0.94", "0.93", "0.93", "329"],
    ["Nose", "0.97", "0.97", "0.97", "103"],
    ["Left cheek", "0.94", "0.93", "0.93", "169"],
    ["Right cheek", "0.96", "0.96", "0.96", "396"],
]


def translate_text(text, translations):
    text = text.strip()
    if not text:
        return ""
    if text in translations:
        return translations[text]
    # Fallback: return original with warning marker stripped in production
    return translations.get(text, text)


def apply_corrections(text, source_pt=""):
    if source_pt in FULL_REPLACEMENTS:
        return FULL_REPLACEMENTS[source_pt]
    for old, new in FULL_REPLACEMENTS.items():
        if old in source_pt:
            return new
    for pattern, repl in CORRECTIONS:
        text = re.sub(pattern, repl, text)
    # Ensure backbone note
    if "ResNet-18" in text and "trained from scratch" not in text.lower() and "weights=None" not in text:
        if "visual feature extractor" in text.lower() or "como extrator" in source_pt.lower():
            text = text.replace(
                "ResNet-18, originally proposed by He et al.",
                "ResNet-18 trained from scratch (weights=None), originally proposed by He et al.",
            )
    if "protótipo móvel" in source_pt.lower() and "on-device inference" not in text.lower():
        if "teledermatology" in text.lower() and "developed" in text.lower():
            pass  # handled by full replacement
    return text


def set_table_borders(table):
    tbl = table._tbl
    tblPr = tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        element = OxmlElement(f"w:{edge}")
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), "4")
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), "000000")
        borders.append(element)
    tblPr.append(borders)


def add_table(doc, rows):
    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
    table.style = "Table Grid"
    set_table_borders(table)
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = table.rows[r].cells[c]
            cell.text = val
            for p in cell.paragraphs:
                for run in p.runs:
                    run.font.name = "Times New Roman"
                    run.font.size = Pt(10)
                    if r == 0:
                        run.bold = True
    return table


def add_figure(doc, filename, caption, width=Inches(5.8)):
    path = os.path.join(FIGS, filename)
    if os.path.exists(path):
        doc.add_picture(path, width=width)
    cap = doc.add_paragraph(caption)
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in cap.runs:
        run.italic = True
        run.font.name = "Times New Roman"
        run.font.size = Pt(10)


def add_para(doc, text, style_name="normal", bold=False):
    if not text.strip():
        return
    if style_name.lower().startswith("heading"):
        level = 1
        if "Heading 3" in style_name or style_name.startswith("3"):
            level = 2
        elif "Heading 2" in style_name or style_name.startswith("2"):
            level = 1
        p = doc.add_heading(text, level=level)
    else:
        p = doc.add_paragraph(text)
    for run in p.runs:
        run.font.name = "Times New Roman"
        run.font.size = Pt(12 if not bold else 14)
        if bold:
            run.bold = True
    if style_name == "Heading 1":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in p.runs:
            run.bold = True
            run.font.size = Pt(14)


def detect_table(rows_data):
    flat = " ".join(" ".join(r) for r in rows_data).lower()
    if "característica" in flat or "characteristic" in flat:
        return "I"
    if "parâmetro" in flat or "parameter" in flat or "framework" in flat:
        return "II"
    if "class" in flat and "precision" in flat and "support" in flat:
        if "facial region" in flat or "forehead" in flat:
            return "IV"
        return "III"
    return None


def main():
    with open(SRC_JSON, encoding="utf-8") as f:
        blocks = json.load(f)
    with open(TRANS_JSON, encoding="utf-8") as f:
        translations = json.load(f)

    doc = Document()
    for section in doc.sections:
        section.top_margin = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(2.5)

    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(12)

    table_map = {"I": TABLE_I, "II": TABLE_II, "III": TABLE_III, "IV": TABLE_IV}
    figures_inserted = set()

    authors_added = False
    for block in blocks:
        if block["type"] == "p":
            src = block["text"]
            if not src.strip():
                continue
            en = translate_text(src, translations)
            en = apply_corrections(en, src)

            style = block.get("style", "normal")

            # Title
            if "Classificação da Gravidade" in src or "Region-Aware ResNet-18 with Anatomical" in en:
                add_para(doc, en, "Heading 1", bold=True)
                if not authors_added:
                    ap = doc.add_paragraph()
                    ap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    ar = ap.add_run(
                        "Lucas Gonzaga Andrade\nRoger Moura Sarmento\n"
                        "Federal Institute of Education, Science and Technology of Ceará (IFCE) — "
                        "Fortaleza, CE, Brazil"
                    )
                    ar.font.name = "Times New Roman"
                    ar.font.size = Pt(11)
                    authors_added = True
                continue

            if src.strip().startswith("Quantidade de Páginas") or src.strip().startswith("Number of Pages"):
                continue

            # Figure caption -> paragraph then image
            fig_key = None
            for key in FIGURE_MAP:
                if src.startswith(key) or en.startswith(key.replace("Figura", "Figure")):
                    fig_key = key
                    break

            if src.strip() == "Resumo":
                add_para(doc, "Abstract", "Heading 2")
            elif "Heading" in style:
                add_para(doc, en, style)
            else:
                add_para(doc, en, style)

            # Insert formatted tables after caption paragraphs only
            if src.startswith("Tabela I –") or en.startswith("Table I –"):
                add_table(doc, TABLE_I)
            elif src.startswith("Tabela II –") or en.startswith("Table II –"):
                add_table(doc, TABLE_II)
            elif src.startswith("Tabela III –") or en.startswith("Table III –"):
                add_table(doc, TABLE_III)
            elif src.startswith("Tabela IV –") or en.startswith("Table IV –"):
                add_table(doc, TABLE_IV)

            if fig_key and fig_key not in figures_inserted:
                fname, cap = FIGURE_MAP[fig_key]
                w = Inches(4.8) if "stability" in fname else Inches(5.8)
                add_figure(doc, fname, cap, w)
                figures_inserted.add(fig_key)

        else:
            # Skip raw source tables; corrected tables inserted at caption paragraphs
            pass

    doc.save(OUT)
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
