# -*- coding: utf-8 -*-
"""Build moderately condensed English CBEB article (~10-12 pages) with CBEB formatting."""
import os
import shutil
import sys

from docx import Document
from docx.shared import Inches, Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, "papers", "sibgrapi", "figs")
OUT_DOCX = os.path.join(ROOT, "papers", "cbeb", "AcneNet_CBEB2026_English_CBEB_Short.docx")
OUT_PDF = os.path.join(ROOT, "papers", "cbeb", "AcneNet_CBEB2026_English_CBEB_Short.pdf")


def setup_doc():
    doc = Document()
    for section in doc.sections:
        section.top_margin = Cm(2.54)
        section.bottom_margin = Cm(2.54)
        section.left_margin = Cm(2.54)
        section.right_margin = Cm(2.54)
    style = doc.styles["Normal"]
    style.font.name = "Arimo"
    style.font.size = Pt(12)
    pf = style.paragraph_format
    pf.line_spacing = 1.15
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    return doc


def add_title(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    r.bold = True
    r.font.name = "Arimo"
    r.font.size = Pt(14)


def add_authors(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    r.font.name = "Arimo"
    r.font.size = Pt(11)


def add_heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    for r in h.runs:
        r.font.name = "Arimo"


def add_para(doc, text):
    p = doc.add_paragraph(text)
    for r in p.runs:
        r.font.name = "Arimo"
        r.font.size = Pt(12)


def add_figure(doc, filename, caption, width=Inches(5.6)):
    path = os.path.join(FIGS, filename)
    if os.path.exists(path):
        doc.add_picture(path, width=width)
    cap = doc.add_paragraph(caption)
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for r in cap.runs:
        r.italic = True
        r.font.name = "Arimo"
        r.font.size = Pt(10)


def set_table_borders(table):
    tbl = table._tbl
    tblPr = tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "4")
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "000000")
        borders.append(el)
    tblPr.append(borders)


def add_table(doc, rows):
    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
    try:
        table.style = "Table Grid"
    except KeyError:
        pass
    set_table_borders(table)
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = table.rows[r].cells[c]
            cell.text = val
            for p in cell.paragraphs:
                for run in p.runs:
                    run.font.name = "Arimo"
                    run.font.size = Pt(10)
                    if r == 0:
                        run.bold = True
    return table


def export_pdf(docx_path, pdf_path):
    try:
        import win32com.client

        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False
        doc = word.Documents.Open(os.path.abspath(docx_path))
        doc.ExportAsFixedFormat(
            OutputFileName=os.path.abspath(pdf_path),
            ExportFormat=17,
            OpenAfterExport=False,
        )
        doc.Close(False)
        word.Quit()
        return os.path.exists(pdf_path)
    except Exception as exc:
        print("win32com failed:", exc)
    try:
        from docx2pdf import convert

        convert(docx_path, pdf_path)
        return os.path.exists(pdf_path)
    except Exception as exc:
        print("docx2pdf failed:", exc)
    return False


def build():
    doc = setup_doc()

    add_title(
        doc,
        "Facial Acne Severity Classification Using a Region-Aware "
        "ResNet-18 with Anatomical Embeddings",
    )
    add_authors(
        doc,
        "Lucas Gonzaga Andrade\nRoger Moura Sarmento\n"
        "Federal Institute of Education, Science and Technology of Ceará (IFCE) — "
        "Fortaleza, CE, Brazil",
    )

    add_heading(doc, "Abstract", level=1)
    add_para(
        doc,
        "Acne vulgaris is a highly prevalent inflammatory skin disease whose severity "
        "assessment remains subjective and specialist-dependent, limiting access in "
        "teledermatology. This paper presents a Region-Aware ResNet-18 for automatic "
        "four-level facial acne severity classification from region crops. Anatomical "
        "context is encoded through learnable 64-dimensional region embeddings fused with "
        "visual features from a shared ResNet-18 backbone trained from scratch. Facial "
        "regions are extracted with YOLOv8 and resized to 224×224. Experiments on merged "
        "Acne04 (1,406 images) and Acne Level (1,457 images) corpora achieved 95.2% "
        "accuracy and weighted F1-score of 0.95 on 1,090 test crops, with regional F1 ≥ 0.93 "
        "and stability of 93.3% ± 0.46% over 30 runs. A Flutter prototype illustrates "
        "the intended mobile workflow, although on-device inference is not yet integrated.",
    )
    add_para(
        doc,
        "Keywords: acne vulgaris; acne severity classification; deep learning; ResNet-18; "
        "anatomical embeddings; teledermatology.",
    )

    add_heading(doc, "1. Introduction", level=1)
    add_para(
        doc,
        "Acne vulgaris is a chronic inflammatory disease of the pilosebaceous unit characterized "
        "by comedones, papules, pustules, and nodules, which may evolve into permanent scars in "
        "severe cases. The high variability in lesion quantity, distribution, and appearance "
        "makes clinical severity assessment challenging and subject to inter-observer variability "
        "[1], [2]. The disease affects approximately 85% of adolescents and young adults and "
        "remains frequent in adult populations, especially among women [3], [4], generating "
        "substantial public health costs and psychosocial impact including anxiety, depression, "
        "and reduced quality of life [5]–[9].",
    )
    add_para(
        doc,
        "Despite high demand for dermatological care, access to specialists remains limited in "
        "many regions. Teledermatology has expanded remote care through smartphone imaging, "
        "reducing geographic barriers and supporting longitudinal monitoring [10]–[12]. "
        "Parallel advances in deep learning and computer vision have driven automatic systems "
        "for dermatological image analysis, yet most acne solutions rely on multi-stage pipelines "
        "in which lesion detection precedes severity estimation, increasing complexity and error "
        "propagation [13]–[16].",
    )
    add_para(
        doc,
        "This work proposes a unified Region-Aware ResNet-18 that classifies severity directly "
        "from facial-region crops enriched with learnable anatomical embeddings. The main "
        "contributions are: (i) a lightweight region-aware architecture with 64-dimensional "
        "embeddings; (ii) a YOLOv8-based preprocessing pipeline for five facial zones; "
        "(iii) evaluation covering global, regional, stability, and confidence analyses; and "
        "(iv) a Flutter teledermatology prototype demonstrating the intended mobile workflow.",
    )

    add_heading(doc, "2. Related Work", level=1)
    add_heading(doc, "2.1 AI in Dermatology and Acne Classification", level=2)
    add_para(
        doc,
        "CNNs consolidated as the primary architecture for dermatological classification and "
        "detection, while Vision Transformers have shown strong representation capacity in "
        "medical imaging [13]–[16]. For acne, lesion-detection pipelines based on Faster R-CNN "
        "and YOLO estimate severity from detected lesions [14], [19], whereas direct classifiers "
        "using ResNet and EfficientNet simplify inference [15], [16], [18]. ResNet-18 offers a "
        "favorable balance between predictive performance and deployment cost for mobile "
        "teledermatology [15], [16].",
    )
    add_heading(doc, "2.2 Limitations and Motivation", level=2)
    add_para(
        doc,
        "Most prior work reports only global accuracy and devotes little attention to regional "
        "consistency, run-to-run stability, or confidence behavior [13]. Multi-stage methods "
        "also depend on lesion-level annotations and propagate detection errors. Motivated by "
        "these gaps, we propose a single-backbone Region-Aware ResNet-18 with explicit "
        "anatomical encoding, extended evaluation protocol, and integration with a mobile "
        "prototype for teledermatology scenarios.",
    )

    add_heading(doc, "3. Materials and Methods", level=1)
    add_para(
        doc,
        "The proposed methodology performs automatic facial acne severity classification using "
        "a lightweight ResNet-18 enriched with learned anatomical region representations. The "
        "pipeline comprises four stages: data acquisition and preparation, YOLOv8-based region "
        "extraction, Region-Aware ResNet-18 classification, and integration into a mobile "
        "teledermatology prototype (Figure 1).",
    )
    add_heading(doc, "3.1 Dataset and Preprocessing", level=2)
    add_para(
        doc,
        "Experiments used two public corpora: Acne04 (1,406 frontal-face images) and Acne "
        "Level (1,457 images), both with four Hayashi-aligned severity grades (0–3). After "
        "consolidation, a custom YOLOv8 detector localized five facial sub-regions—forehead, "
        "left cheek, right cheek, nose, and chin. Each crop was resized to 224×224 pixels. "
        "The model receives only region-level crops paired with a region index; full-face "
        "images are not used as network input. Data were split 70/15/15 for training, "
        "validation, and test. For final model selection, training and validation splits were "
        "merged according to the robust validation criterion, and the best weights were saved "
        "as best_robust_model.pth. Table I summarizes dataset characteristics; representative "
        "samples and extracted regions are shown in Figure 2.",
    )
    add_table(
        doc,
        [
            ["Characteristic", "Acne04", "Acne Level"],
            ["Number of images", "1,406", "1,457"],
            ["Classes", "4 (Hayashi grades 0–3)", "4 (Hayashi grades 0–3)"],
            ["Use in this work", "Merged unified pipeline", "Merged unified pipeline"],
        ],
    )
    add_para(doc, "Table I – Characteristics of the datasets used in this study.")
    add_para(
        doc,
        "Figure 1 summarizes the end-to-end workflow from image acquisition through YOLOv8 "
        "region extraction, Region-Aware ResNet-18 inference, severity prediction, and mobile "
        "reporting.",
    )
    add_figure(
        doc,
        "fig1_pipeline.jpg",
        "Figure 1 – Overview of the proposed methodology from acquisition to mobile reporting.",
    )
    add_figure(
        doc,
        "fig2_datasets.jpg",
        "Figure 2 – Representative Acne04 and Acne Level samples and extracted facial regions.",
    )

    add_heading(doc, "3.2 Region-Aware ResNet-18", level=2)
    add_para(
        doc,
        "The architecture incorporates anatomical information while maintaining low "
        "computational complexity. Each region crop is processed by a ResNet-18 backbone "
        "trained from scratch (weights=None) [23]. After global average pooling, the visual "
        "descriptor f ∈ ℝ^512 is concatenated with a learnable region embedding "
        "e_r ∈ ℝ^64, producing z = Concat(f, e_r) ∈ ℝ^576. A classifier with fully connected "
        "layers, Batch Normalization, ReLU, and Dropout outputs four severity logits via Softmax.",
    )
    add_para(
        doc,
        "Unlike conventional CNN classifiers, the embedding layer injects explicit anatomical "
        "identity, allowing one shared backbone to adapt its decision boundary per zone without "
        "training separate models (Figure 3). This adds only a small number of parameters while "
        "preserving low inference cost for smartphone-based teledermatology.",
    )
    add_figure(
        doc,
        "fig3_architecture.jpg",
        "Figure 3 – Region-Aware ResNet-18 architecture with learnable region embeddings.",
    )

    add_heading(doc, "3.3 Training Strategy", level=2)
    add_para(
        doc,
        "Training was conducted in a supervised multiclass setting using preprocessed region "
        "crops and region identifiers. Optimization used Adam with class-weighted cross-entropy "
        "to mitigate imbalance. Data augmentation included random horizontal flipping, rotation "
        "(±25°), color jitter, Gaussian blur, and Random Erasing. ReduceLROnPlateau scheduling "
        "and Early Stopping reduced overfitting. Experiments were implemented in PyTorch with "
        "GPU acceleration on an NVIDIA GeForce RTX 3060 (12 GB). Reproducibility is supported "
        "by training/predict.py for inference with the saved weights.",
    )
    add_table(
        doc,
        [
            ["Parameter", "Value"],
            ["Framework", "PyTorch"],
            ["Backbone", "ResNet-18 (trained from scratch, weights=None)"],
            ["Optimizer / loss", "Adam / weighted cross-entropy"],
            ["Learning rate / batch", "0.0001 / 16"],
            ["Maximum epochs / scheduler", "40 / ReduceLROnPlateau"],
            ["Data augmentation", "Flip, rotation (±25°), color jitter, blur, random erasing"],
            ["Hardware", "NVIDIA GeForce RTX 3060 (12 GB)"],
        ],
    )
    add_para(doc, "Table II – Main training hyperparameters.")

    add_heading(doc, "3.4 Experimental Setup", level=2)
    add_para(
        doc,
        "Performance was evaluated using Accuracy, Precision, Recall, and F1-score globally, "
        "per class, and per facial region. To avoid overinterpretation of a single favorable "
        "metric, analysis covered three perspectives: (i) global classification performance; "
        "(ii) consistency across facial zones; and (iii) operational robustness via stability "
        "over 30 perturbation runs and Softmax confidence analysis.",
    )

    add_heading(doc, "4. Experimental Results", level=1)
    add_heading(doc, "4.1 Overall Performance", level=2)
    add_para(
        doc,
        "On 1,090 held-out region crops, the model achieved 95.2% accuracy and weighted "
        "F1-score of 0.95 (Table III). Precision, recall, and F1 remained close across classes, "
        "indicating balanced learning despite uneven class frequencies. Grades 0 and 3 achieved "
        "the strongest results; grade 2 was most challenging due to visual overlap with adjacent "
        "severities.",
    )
    add_table(
        doc,
        [
            ["Class", "Precision", "Recall", "F1", "Support"],
            ["Grade 0 (Clear)", "0.96", "0.95", "0.95", "318"],
            ["Grade 1 (Mild)", "0.94", "0.94", "0.94", "412"],
            ["Grade 2 (Moderate)", "0.88", "0.92", "0.90", "110"],
            ["Grade 3 (Severe)", "1.00", "1.00", "1.00", "250"],
            ["Weighted avg.", "0.95", "0.95", "0.95", "1090"],
        ],
    )
    add_para(doc, "Table III – Global classification performance on the test set.")
    add_para(
        doc,
        "The confusion matrix (Figure 4) confirms that most errors occur between neighboring "
        "classes, especially grades 1 and 2, whereas confusion between grades 0 and 3 is "
        "negligible. This pattern mirrors clinical ambiguity in borderline cases and suggests "
        "systematic rather than random failure modes. Under dermatological assessment, "
        "borderline cases between consecutive grades frequently present high visual similarity.",
    )
    add_figure(
        doc,
        "fig4_confusion_en.png",
        "Figure 4 – Confusion matrix on the test set (counts and normalized percentages).",
    )

    add_heading(doc, "4.2 Regional and Robustness Analysis", level=2)
    add_para(
        doc,
        "Because region identity is explicitly encoded, we analyzed performance across facial "
        "zones (Table IV). All regions achieved weighted F1 ≥ 0.93. Forehead obtained F1 = 0.99, "
        "while chin and left cheek were comparatively harder (F1 = 0.93), possibly due to "
        "texture variability, facial hair, shadows, and less homogeneous lesion distribution. "
        "These results support the architectural claim that a single ResNet-18 with region "
        "embeddings learns zone-specific behavior without maintaining five independent models.",
    )
    add_table(
        doc,
        [
            ["Region", "Precision", "Recall", "F1", "Support"],
            ["Forehead", "0.99", "0.99", "0.99", "93"],
            ["Chin", "0.94", "0.93", "0.93", "329"],
            ["Nose", "0.97", "0.97", "0.97", "103"],
            ["L. cheek", "0.94", "0.93", "0.93", "169"],
            ["R. cheek", "0.96", "0.96", "0.96", "396"],
        ],
    )
    add_para(doc, "Table IV – Weighted performance by facial region.")
    add_para(
        doc,
        "Stability was assessed through 30 independent runs with light input perturbations. "
        "Mean accuracy was 93.3% with σ = 0.46% (Figure 5), indicating low run-to-run "
        "variability compatible with smartphone capture. Confidence analysis showed that "
        "correct predictions usually received very high Softmax scores, while some incorrect "
        "predictions were also overconfident, recommending calibration and selective referral "
        "before clinical deployment.",
    )
    add_figure(
        doc,
        "fig5_stability_en.png",
        "Figure 5 – Accuracy distribution over 30 robustness evaluations.",
        width=Inches(4.6),
    )

    add_heading(doc, "5. Discussion", level=1)
    add_para(
        doc,
        "The results demonstrate that explicit anatomical encoding enriches a lightweight CNN "
        "without multi-stage lesion pipelines. A single shared backbone reduces parameter "
        "redundancy relative to one-model-per-region strategies. Low variance across runs "
        "indicates operational reliability, whereas adjacent-class confusion reflects "
        "clinically plausible ambiguity.",
    )
    add_para(
        doc,
        "A Flutter teledermatology prototype was developed to demonstrate the intended workflow "
        "(Figure 6). At the time of writing, on-device inference is not integrated; the "
        "application illustrates interface and workflow design rather than validated clinical "
        "deployment. A hybrid policy—automatic triage for high-confidence non-borderline cases "
        "and specialist review for uncertain ones—aligns model strengths with safe clinical use.",
    )
    add_figure(
        doc,
        "fig6_app.jpg",
        "Figure 6 – Mobile teledermatology prototype screens.",
        width=Inches(5.2),
    )

    add_heading(doc, "6. Limitations and Conclusion", level=1)
    add_para(
        doc,
        "This study used merged public datasets and does not yet cover the full diversity of "
        "skin phototypes, devices, and real-world lighting. Labels are region-level severity "
        "grades rather than lesion-level annotations, and the mobile prototype was not evaluated "
        "prospectively in clinical teledermatology.",
    )
    add_para(
        doc,
        "We presented a Region-Aware ResNet-18 for four-level acne severity classification from "
        "facial-region crops, achieving 95.2% test accuracy with consistent cross-region behavior "
        "and low stability variance. Future work includes confidence calibration, external "
        "validation, on-device inference, and broader demographic representation. The system is "
        "intended as clinical decision support and does not replace professional diagnosis.",
    )

    add_heading(doc, "References", level=1)
    refs = [
        "[1] M. Vasam, A. Korivi, and K. M. Balli, Acne vulgaris review, Biomedicines, vol. 11, no. 12, 2023.",
        "[2] Y. Li et al., Acne treatment perspectives, Frontiers in Medicine, vol. 11, 2024.",
        "[3] G. D. Lenuța et al., Epidemiology of acne, Cosmetics, vol. 12, no. 3, 2025.",
        "[4] A. H. Heng and F. T. Chew, Systematic review of acne epidemiology, Scientific Reports, vol. 10, 2020.",
        "[5] A. M. Layton, Burden of acne on quality of life, Dermatology and Therapy, 2025.",
        "[6] D. V. Samuels et al., Acne and risk of depression and anxiety, J. Amer. Acad. Dermatol., vol. 83, 2020.",
        "[7] A. M. Layton et al., Global burden of acne, Brit. J. Dermatol., vol. 184, 2020.",
        "[10] D. Z. Eichenfield et al., Management of acne vulgaris, JAMA, vol. 326, 2020.",
        "[11] S. Wongvibulsin et al., Dermatology mobile apps with AI, JAMA Dermatology, vol. 160, 2024.",
        "[12] R. V. Reynolds et al., Acne care guidelines, J. Amer. Acad. Dermatol., vol. 90, 2024.",
        "[13] S. P. Choy et al., Systematic review of DL for skin disease, npj Digital Medicine, vol. 6, 2023.",
        "[14] Q. T. Huynh et al., Automatic acne detection and grading, Diagnostics, vol. 12, 2022.",
        "[15] Y. Lin et al., KIEGLFN acne grading framework, CMPB, vol. 221, 2022.",
        "[16] S. Liu et al., AcneGrader ensemble, Skin Res. Technol., vol. 28, 2022.",
        "[18] Y. Lin et al., DED acne severity grading, Expert Systems with Applications, vol. 228, 2023.",
        "[19] P. K. Nguyen et al., ACNE8M detection system, VNUHCM J. Sci. Technol. Dev., vol. 27, 2024.",
        "[23] K. He et al., Deep residual learning for image recognition, CVPR, 2016.",
    ]
    for r in refs:
        add_para(doc, r)

    doc.save(OUT_DOCX)
    print("Saved DOCX:", OUT_DOCX)

    if export_pdf(OUT_DOCX, OUT_PDF):
        print("Saved PDF:", OUT_PDF)

    dl = os.path.join(os.path.expanduser("~"), "Downloads")
    shutil.copy2(OUT_DOCX, os.path.join(dl, os.path.basename(OUT_DOCX)))
    if os.path.exists(OUT_PDF):
        shutil.copy2(OUT_PDF, os.path.join(dl, os.path.basename(OUT_PDF)))


if __name__ == "__main__":
    build()
