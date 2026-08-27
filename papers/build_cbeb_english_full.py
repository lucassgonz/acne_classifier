# -*- coding: utf-8 -*-
"""Build full English CBEB 2026 article (translated, reviewed, English figures)."""
from docx import Document
from docx.shared import Inches, Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(ROOT, "sibgrapi", "figs")
OUT = os.path.join(ROOT, "cbeb", "AcneNet_CBEB2026_English.docx")
os.makedirs(os.path.join(ROOT, "cbeb"), exist_ok=True)

doc = Document()
for section in doc.sections:
    section.top_margin = Cm(2.5)
    section.bottom_margin = Cm(2.5)
    section.left_margin = Cm(2.5)
    section.right_margin = Cm(2.5)

style = doc.styles["Normal"]
style.font.name = "Times New Roman"
style.font.size = Pt(12)
style._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")


def add_title(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.bold = True
    run.font.name = "Times New Roman"
    run.font.size = Pt(14)


def add_authors(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.font.name = "Times New Roman"
    run.font.size = Pt(11)


def add_heading(text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.name = "Times New Roman"


def add_para(text):
    p = doc.add_paragraph(text)
    for run in p.runs:
        run.font.name = "Times New Roman"
        run.font.size = Pt(12)


def add_figure(filename, caption, width=Inches(5.8)):
    path = os.path.join(FIGS, filename)
    if not os.path.exists(path):
        path = os.path.join(ROOT, "figs", filename)
    if os.path.exists(path):
        doc.add_picture(path, width=width)
        cap = doc.add_paragraph(caption)
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in cap.runs:
            run.italic = True
            run.font.name = "Times New Roman"
            run.font.size = Pt(10)
    else:
        add_para(f"[Figure not found: {filename}]")


def add_table_caption(text):
    p = doc.add_paragraph(text)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in p.runs:
        run.font.name = "Times New Roman"
        run.font.size = Pt(10)
        run.italic = True


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


def add_table(headers, rows):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    set_table_borders(table)
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = h
        for p in cell.paragraphs:
            for run in p.runs:
                run.bold = True
                run.font.name = "Times New Roman"
                run.font.size = Pt(10)
    for r, row in enumerate(rows, start=1):
        for c, val in enumerate(row):
            cell = table.rows[r].cells[c]
            cell.text = val
            for p in cell.paragraphs:
                for run in p.runs:
                    run.font.name = "Times New Roman"
                    run.font.size = Pt(10)
    return table


# --- Document content ---
add_title(
    "Facial Acne Severity Classification Using a Region-Aware "
    "ResNet-18 with Anatomical Embeddings"
)
add_authors(
    "Lucas Gonzaga Andrade\nRoger Moura Sarmento\n"
    "Federal Institute of Education, Science and Technology of Ceará (IFCE) — "
    "Fortaleza, CE, Brazil"
)

add_heading("Abstract", level=1)
add_para(
    "Acne vulgaris is one of the most prevalent dermatological diseases worldwide, "
    "and continuous monitoring of disease severity remains challenging, especially "
    "in teledermatology scenarios. This work proposes a Region-Aware ResNet-18 "
    "architecture for automatic four-level facial acne severity classification from "
    "facial-region crops. Anatomical identity is encoded through learnable region "
    "embeddings that enrich the visual representation without significantly "
    "increasing model complexity. Experiments were conducted on the public Acne04 "
    "and Acne Level corpora, considering four Hayashi-aligned severity grades and "
    "evaluating global performance, regional behavior, robustness, and prediction "
    "confidence. The architecture achieved 95.2% accuracy and a weighted F1-score "
    "of 0.95 on 1,090 held-out test crops, with consistent performance across facial "
    "regions (weighted F1 ≥ 0.93) and low run-to-run variability (93.3% ± 0.46% "
    "over 30 perturbation runs). A Flutter mobile prototype demonstrates the "
    "intended teledermatology workflow, although on-device inference has not yet "
    "been integrated. The results show that learned anatomical representations "
    "provide a simple and efficient strategy for adding spatial context to lightweight "
    "convolutional architectures, combining strong predictive performance, low "
    "computational cost, and potential for real-world Biomedical Engineering "
    "applications."
)
add_para(
    "Keywords: acne vulgaris; acne severity classification; deep learning; "
    "ResNet-18; anatomical embeddings; teledermatology."
)

add_heading("1. Introduction", level=1)
add_para(
    "Acne vulgaris is a chronic inflammatory skin disease characterized by comedones, "
    "papules, pustules, and nodules, which may evolve into permanent scars in severe "
    "cases. The high variability in lesion quantity, distribution, and appearance "
    "makes clinical severity assessment challenging, often depending on specialist "
    "experience and subject to inter-observer variability [1], [2]."
)
add_para(
    "Considered one of the most prevalent dermatological conditions worldwide, acne "
    "affects approximately 85% of adolescents and young adults aged 12–25 years and "
    "remains frequent in adult populations, especially among women [3], [4]. Due to "
    "its high incidence and long clinical follow-up, the disease represents an "
    "important public health problem, generating millions of dermatological "
    "consultations annually and substantial costs associated with diagnosis, "
    "treatment, and monitoring [5]."
)
add_para(
    "Although often perceived as predominantly aesthetic, acne can produce significant "
    "physical and psychosocial consequences. Moderate and severe cases are associated "
    "with permanent scarring, pigmentation changes, pain, and impacts on self-esteem, "
    "anxiety, depression, social isolation, and quality of life [6]–[9]. Early "
    "identification of disease severity is therefore fundamental for treatment "
    "planning and prevention of physical and psychological sequelae."
)
add_para(
    "Despite high demand for dermatological care, access to specialists remains "
    "limited in many regions, especially in low- and middle-income countries. "
    "Teledermatology has emerged as an alternative to expand access to specialized "
    "care through remote analysis of dermatological images captured by mobile "
    "devices, reducing geographic barriers and supporting longitudinal patient "
    "monitoring [10]–[12]."
)
add_para(
    "Recent advances in Artificial Intelligence (AI), Deep Learning (DL), and Computer "
    "Vision (CV) have driven the development of automatic systems for medical image "
    "analysis. Convolutional Neural Networks (CNNs) have achieved competitive "
    "performance in dermatological classification, segmentation, and lesion "
    "detection [13]–[16]. For acne, however, most proposed solutions rely on "
    "multi-stage pipelines in which individual lesion detection is followed by "
    "classification or quantification."
)
add_para(
    "Although effective in controlled settings, multi-stage approaches increase "
    "computational complexity, favor error propagation between processing stages, "
    "and hinder deployment on mobile devices where memory, processing, and energy "
    "constraints are critical. Moreover, many studies focus exclusively on global "
    "metrics and devote little attention to robustness, stability, and prediction "
    "confidence—factors essential for clinical decision support."
)
add_para(
    "This work proposes a unified approach for automatic four-level facial acne "
    "severity classification based on a ResNet-18 architecture enriched with facial "
    "region embeddings. Unlike lesion-centric methods, the proposed system classifies "
    "severity directly from facial-region crops extracted by YOLOv8. This strategy "
    "eliminates intermediate lesion-detection stages, reduces error propagation, "
    "preserves regional anatomical context, and simplifies deployment in smartphone-based "
    "teledermatology applications."
)
add_para("The main contributions of this work are:")
add_para(
    "(i) a Region-Aware ResNet-18 architecture with 64-dimensional learnable region "
    "embeddings; (ii) a single shared convolutional backbone for all facial zones, "
    "avoiding multiple specialized models; (iii) a comprehensive experimental "
    "evaluation including global, class-wise, and regional analyses; (iv) robustness "
    "and confidence analyses beyond traditional classification metrics; and (v) a "
    "Flutter teledermatology prototype demonstrating the intended mobile workflow."
)

add_heading("2. Related Work", level=1)
add_heading("2.1 Artificial Intelligence in Dermatology", level=2)
add_para(
    "Recent advances in AI, DL, and CV have significantly transformed medical image "
    "analysis, especially in dermatology. CNNs remain the primary architecture for "
    "lesion classification, segmentation, and detection, while attention-based models "
    "such as Vision Transformers (ViTs) have shown strong representation capacity in "
    "medical applications [13]–[16]. These technologies support assisted diagnosis "
    "of melanoma, psoriasis, eczema, and acne, achieving performance comparable to "
    "specialists in controlled experimental settings [13], [17]."
)
add_para(
    "Mobile devices and telemedicine have further accelerated teledermatology systems "
    "capable of remote assessment through smartphone images [11], [12]. However, "
    "systematic reviews highlight limitations related to external validation, "
    "population diversity, robustness under real acquisition conditions, and "
    "integration into clinical workflows. Research is gradually shifting from "
    "maximizing accuracy alone toward more robust, interpretable, and deployable "
    "solutions [13]."
)

add_heading("2.2 Automatic Acne Severity Classification", level=2)
add_para(
    "Automatic acne severity classification has become an important research line "
    "for teledermatology and remote patient monitoring. Existing approaches can be "
    "grouped into lesion-detection pipelines and direct severity classifiers. "
    "Detection-based methods use Faster R-CNN, YOLO, or hybrid architectures to "
    "identify individual lesions before estimating severity [14], [19], [21]. "
    "Although competitive, they require lesion-level annotations, increase "
    "computational cost, and make final grading dependent on detection quality."
)
add_para(
    "Direct classifiers estimate severity from facial images without identifying each "
    "lesion individually. ResNet, EfficientNet, and knowledge-distillation strategies "
    "have achieved competitive results while simplifying inference [15], [16], [18]. "
    "More recently, CNN–Transformer hybrids and ViTs have been explored to improve "
    "spatial representation [21], but often at higher computational cost and inference "
    "time, limiting mobile deployment."
)
add_para(
    "From a Biomedical Engineering perspective, lightweight CNNs remain attractive "
    "for mobile health. ResNet-18 offers a favorable balance between predictive "
    "performance, training stability, reduced computational complexity, and deployment "
    "feasibility on smartphones [15], [16]."
)

add_heading("2.3 Limitations of Current Approaches", level=2)
add_para(
    "Despite recent progress, several limitations persist. Most methods still depend "
    "on multi-stage pipelines, increasing architectural complexity and error "
    "propagation [14], [19], [21]. Evaluation protocols often report only global "
    "accuracy, precision, and F1-score, with limited analysis of robustness, "
    "cross-run stability, regional consistency, and confidence behavior [13]. "
    "Furthermore, few studies present solutions effectively oriented toward real "
    "teledermatology deployment."
)
add_para(
    "Motivated by these gaps, this work proposes a lightweight Region-Aware ResNet-18 "
    "in which anatomical context is encoded through learnable region embeddings rather "
    "than multiple specialized networks or complex attention mechanisms. The method "
    "eliminates intermediate lesion-detection stages, extends evaluation with regional, "
    "stability, and confidence analyses, and connects the model to a Flutter mobile "
    "prototype for teledermatology scenarios."
)

add_heading("3. Materials and Methods", level=1)
add_para(
    "The proposed methodology performs automatic facial acne severity classification "
    "using a lightweight ResNet-18 enriched with learned anatomical region "
    "representations. The complete pipeline comprises four main stages: (i) data "
    "acquisition and preparation, (ii) image preprocessing and facial-region "
    "extraction, (iii) severity classification with the proposed architecture, and "
    "(iv) integration into a mobile teledermatology prototype."
)
add_para(
    "The end-to-end workflow—from image acquisition through YOLOv8-based region "
    "extraction, Region-Aware ResNet-18 inference, severity prediction, and mobile "
    "reporting—is summarized in Figure 1."
)
add_figure("fig1_pipeline.jpg", "Figure 1 – Overview of the proposed methodology, including image acquisition, preprocessing, Region-Aware ResNet-18 architecture, and Flutter integration.")

add_heading("3.1 Dataset and Preprocessing", level=2)
add_para(
    "Experiments used two public corpora widely employed in acne severity research: "
    "Acne04 and Acne Level. Using both datasets allows evaluation under different "
    "organization protocols and acquisition conditions, supporting a more "
    "comprehensive analysis of generalization."
)
add_table_caption("Table I – Characteristics of the datasets used in this study.")
add_table(
    ["Characteristic", "Acne04", "Acne Level"],
    [
        ["Objective", "Acne severity classification", "Acne severity classification"],
        ["Number of images", "1,406", "1,457"],
        ["Classes", "4", "4"],
        ["Severity scale", "Hayashi (grades 0–3)", "Hayashi (grades 0–3)"],
        ["Image type", "Frontal face", "Frontal face"],
        ["Organization", "Original corpus", "Folder-organized by severity"],
        ["Use in this work", "Merged into unified pipeline", "Merged into unified pipeline"],
    ],
)
add_para(
    "Although both datasets target acne severity classification, they differ in "
    "organization, class distribution, illumination, pose, and patient appearance. "
    "Representative samples from both sources, together with the five facial regions "
    "used by the classifier—forehead, left cheek, right cheek, nose, and chin—are "
    "shown in Figure 2."
)
add_figure("fig2_datasets.jpg", "Figure 2 – Representative Acne04 and Acne Level samples and extracted facial regions used by the proposed model.")

add_para(
    "All images underwent standardized preprocessing to uniformize inputs and reduce "
    "the influence of irrelevant background information. A custom YOLOv8 detector "
    "trained to localize five anatomical facial regions was applied to each image. "
    "This step extracts sub-regional crops rather than relying solely on a global "
    "face bounding box."
)
add_para(
    "Each detected region was resized to 224×224 pixels and normalized consistently "
    "with training. During classification, each crop is paired with a region index "
    "(forehead, left cheek, right cheek, nose, or chin), which is mapped to a "
    "learnable embedding. The model receives only region-level crops; full-face images "
    "are not used as network input."
)
add_para(
    "After consolidation and segmentation, the data were split into training, "
    "validation, and test sets (70%, 15%, and 15%, respectively), preserving "
    "independence between training and evaluation samples."
)

add_heading("3.2 Region-Aware ResNet-18", level=2)
add_para(
    "The proposed architecture incorporates anatomical information from different "
    "facial regions while maintaining a computationally simple structure suitable "
    "for mobile applications. The model receives a single region crop together with "
    "its anatomical identifier, as illustrated in Figure 3."
)
add_figure("fig3_architecture.jpg", "Figure 3 – Region-Aware ResNet-18 architecture with learnable region embeddings fused before classification.")

add_para(
    "Unlike conventional CNN classifiers that rely exclusively on visual features, "
    "the proposed architecture explicitly encodes anatomical context through a "
    "learnable embedding layer. A single ResNet-18 backbone trained from scratch "
    "(without ImageNet pretraining) adapts its internal representation according to "
    "the processed region while sharing weights across all facial zones."
)
add_para(
    "ResNet-18 was used as the visual feature extractor [23]. After the residual "
    "blocks and Global Average Pooling (GAP), the network produces a visual descriptor "
    "f ∈ ℝ^512. A learnable embedding matrix E ∈ ℝ^(5×64) assigns a 64-dimensional "
    "vector e_r to each of the five facial regions. The enriched representation is "
    "obtained by concatenation: z = Concat(f, e_r) ∈ ℝ^576."
)
add_para(
    "The fused vector z is processed by a classifier composed of fully connected "
    "layers with Batch Normalization, ReLU activation, and Dropout, followed by "
    "Softmax to estimate the probability of each of the four severity grades. This "
    "design adds anatomical knowledge through a single embedding layer while "
    "preserving the simplicity of ResNet-18 and enabling deployment in mobile "
    "teledermatology applications."
)

add_heading("3.3 Training Strategy", level=2)
add_para(
    "Training was conducted in a supervised multiclass setting using preprocessed "
    "region crops and their corresponding region identifiers. Optimization used the "
    "Adam algorithm with class-weighted Cross-Entropy Loss to mitigate class "
    "imbalance. Data augmentation included random horizontal flipping, rotation "
    "(±25°), color jitter, Gaussian blur, and Random Erasing. ReduceLROnPlateau "
    "scheduling and Early Stopping were applied to reduce overfitting."
)
add_table_caption("Table II – Main hyperparameters used during training.")
add_table(
    ["Parameter", "Value"],
    [
        ["Framework", "PyTorch"],
        ["Backbone", "ResNet-18 (trained from scratch)"],
        ["Optimizer", "Adam"],
        ["Loss function", "Weighted Cross-Entropy"],
        ["Learning rate", "0.0001"],
        ["Scheduler", "ReduceLROnPlateau"],
        ["Batch size", "16"],
        ["Maximum epochs", "40"],
        ["Data augmentation", "Horizontal flip (p=0.5), rotation (±25°), color jitter, Gaussian blur, random erasing"],
        ["Regularization", "Batch Normalization and Dropout"],
        ["Hardware", "NVIDIA GeForce RTX 3060 (12 GB)"],
    ],
)
add_para(
    "Hyperparameters were defined empirically from preliminary experiments and kept "
    "constant throughout evaluation. The best model was selected using robust "
    "validation criteria and saved as best_robust_model.pth."
)

add_heading("3.4 Experimental Setup", level=2)
add_para(
    "Evaluation aimed to analyze not only predictive performance but also robustness, "
    "stability, and applicability in teledermatology scenarios. Performance was "
    "measured using Accuracy, Precision, Recall, and F1-score globally and per class. "
    "Regional weighted metrics assessed consistency across facial zones."
)
add_para(
    "Stability was evaluated through 30 independent test runs with light input "
    "perturbations compatible with smartphone capture variability. Confidence analysis "
    "examined Softmax score distributions for correct and incorrect predictions. "
    "This three-perspective protocol—global discrimination, regional consistency, "
    "and operational robustness—reduces overinterpretation of a single favorable metric."
)

add_heading("4. Experimental Results", level=1)
add_heading("4.1 Overall Classification Performance", level=2)
add_para(
    "Table III summarizes global performance on the test set, which contains 1,090 "
    "region crops. The Region-Aware ResNet-18 achieved 95.2% accuracy and a weighted "
    "F1-score of 0.95. Precision, Recall, and F1 remained close across classes, "
    "indicating balanced learning despite uneven class frequencies."
)
add_table_caption("Table III – Global classification performance on the test set.")
add_table(
    ["Grade", "Precision", "Recall", "F1-score", "Support"],
    [
        ["0 (Clear)", "0.96", "0.95", "0.95", "318"],
        ["1 (Mild)", "0.94", "0.94", "0.94", "412"],
        ["2 (Moderate)", "0.88", "0.92", "0.90", "110"],
        ["3 (Severe)", "1.00", "1.00", "1.00", "250"],
        ["Weighted avg.", "0.95", "0.95", "0.95", "1090"],
        ["Overall accuracy", "95.2%", "—", "—", "—"],
    ],
)
add_para(
    "Grades 0 and 3 achieved the strongest results, whereas grade 2 was the most "
    "challenging due to visual overlap with adjacent severities. Most errors occurred "
    "between neighboring classes, especially grades 1 and 2, whereas confusion "
    "between distant classes (0 and 3) was negligible. This pattern is confirmed by "
    "the confusion matrix in Figure 4."
)
add_figure("fig4_confusion_en.png", "Figure 4 – Confusion matrix of the Region-Aware ResNet-18 on the test set (absolute counts and normalized percentages).")

add_para(
    "The strong diagonal dominance indicates systematic rather than random failure "
    "modes, with residual errors concentrated where severity boundaries are clinically "
    "subjective. This behavior is consistent with dermatological assessment, in which "
    "borderline cases between consecutive grades often present high visual similarity."
)

add_heading("4.2 Regional Performance Analysis", level=2)
add_para(
    "Because region identity is explicitly encoded, we analyzed whether performance "
    "remains stable across facial zones. Table IV reports weighted metrics by region."
)
add_table_caption("Table IV – Weighted classification performance by facial region.")
add_table(
    ["Facial region", "Precision", "Recall", "F1-score", "Support"],
    [
        ["Forehead", "0.99", "0.99", "0.99", "93"],
        ["Chin", "0.94", "0.93", "0.93", "329"],
        ["Nose", "0.97", "0.97", "0.97", "103"],
        ["Left cheek", "0.94", "0.93", "0.93", "169"],
        ["Right cheek", "0.96", "0.96", "0.96", "396"],
    ],
)
add_para(
    "All regions achieved F1-scores ≥ 0.93. Forehead obtained the best result "
    "(F1 = 0.99), while chin and left cheek were comparatively harder yet still "
    "strong (F1 = 0.93). Lower performance in chin and cheek regions may reflect "
    "texture variability, facial hair, shadows, and less homogeneous lesion "
    "distribution—factors commonly observed in mobile image acquisition."
)
add_para(
    "These results support the main architectural claim: a single ResNet-18 with "
    "region embeddings can learn zone-specific behavior without maintaining five "
    "independent models."
)

add_heading("4.3 Robustness and Confidence Analysis", level=2)
add_para(
    "Beyond predictive performance, the architecture was evaluated for operational "
    "robustness and confidence behavior. Stability was assessed through 30 independent "
    "runs with light input perturbations. Mean accuracy was 93.3% with a standard "
    "deviation of 0.46%, indicating low run-to-run variability under conditions "
    "compatible with smartphone capture. The corresponding distribution is shown in "
    "Figure 5."
)
add_figure("fig5_stability_en.png", "Figure 5 – Accuracy distribution over 30 independent robustness evaluations.", width=Inches(4.8))

add_para(
    "Confidence analysis showed that correct predictions usually received very high "
    "Softmax scores, while a subset of incorrect predictions was also overconfident. "
    "Therefore, calibration techniques and selective referral policies are recommended "
    "before clinical deployment, especially for borderline cases between grades 1 and 2."
)

add_heading("5. Discussion", level=1)
add_para(
    "The experimental results demonstrate that explicit anatomical encoding provides "
    "a simple and effective way to enrich a lightweight CNN without multi-stage "
    "lesion pipelines. The unified backbone reduces parameter redundancy and "
    "simplifies maintenance relative to one-model-per-region strategies."
)
add_para(
    "To demonstrate practical integration, a Flutter teledermatology prototype was "
    "developed. The interface covers home, guided capture, region selection, analysis "
    "progress, severity result with confidence indicator, and analysis history, as "
    "illustrated in Figure 6."
)
add_figure("fig6_app.jpg", "Figure 6 – Mobile teledermatology prototype screens for capture, analysis, and history.", width=Inches(5.5))

add_para(
    "At the time of writing, on-device inference is not yet integrated; the "
    "application demonstrates interface and workflow design rather than validated "
    "clinical deployment. Separating the mobile client from the inference service "
    "facilitates future model updates and uncertainty-based triage rules."
)
add_para(
    "From a deployment perspective, low variance across runs indicates operational "
    "reliability, whereas adjacent-class confusion reflects clinically plausible "
    "ambiguity. A hybrid policy—automatic triage for high-confidence non-borderline "
    "cases and specialist review for uncertain ones—aligns model strengths with "
    "safe clinical use."
)

add_heading("6. Limitations", level=1)
add_para(
    "This study used merged public datasets and does not yet cover the full diversity "
    "of skin phototypes, devices, and real-world lighting. Labels are region-level "
    "severity grades rather than lesion-level annotations, and the mobile prototype "
    "was not evaluated prospectively in clinical teledermatology. Some incorrect "
    "predictions were associated with high confidence, indicating the need for "
    "probability calibration and selective prediction strategies."
)

add_heading("7. Conclusion", level=1)
add_para(
    "This work presented a Region-Aware ResNet-18 for automatic four-level facial "
    "acne severity classification from region crops, achieving 95.2% test accuracy "
    "with consistent cross-region behavior and low stability variance. The results "
    "show that learned anatomical embeddings provide a simple and efficient strategy "
    "for enriching lightweight convolutional architectures with spatial context."
)
add_para(
    "Future work includes confidence calibration, external validation on independent "
    "datasets, on-device inference integration, and broader demographic representation. "
    "The system is intended as clinical decision support and does not replace "
    "professional diagnosis."
)

add_heading("References", level=1)
refs = [
    "[1] M. Vasam, A. Korivi, and K. M. Balli, “Acne vulgaris: A review of the pathophysiology, treatment, and recent nanotechnology-based advances,” Biomedicines, vol. 11, no. 12, 2023.",
    "[2] Y. Li, X. Hu, G. Dong, X. Wang, and T. Liu, “Acne treatment: Research progress and new perspectives,” Frontiers in Medicine, vol. 11, 2024.",
    "[3] G. D. Lenuța et al., “The Epidemiology of Acne in the Current Era: Trends and Clinical Implications,” Cosmetics, vol. 12, no. 3, 2025.",
    "[4] A. H. Heng and F. T. Chew, “Systematic review of the epidemiology of acne vulgaris,” Scientific Reports, vol. 10, 2020.",
    "[5] A. M. Layton, “The burden of acne vulgaris on health-related quality of life,” Dermatology and Therapy, 2025.",
    "[6] D. V. Samuels et al., “Acne vulgaris and risk of depression and anxiety: A meta-analytic review,” J. Amer. Acad. Dermatol., vol. 83, no. 2, pp. 532–541, 2020.",
    "[7] A. M. Layton, D. Thiboutot, and J. Tan, “Reviewing the global burden of acne,” Brit. J. Dermatol., vol. 184, 2020.",
    "[8] A. S. Morshed et al., “Understanding the impact of acne vulgaris on self-esteem and quality of life,” Scientific Reports, vol. 13, 2023.",
    "[9] T. Tasneem et al., “Effects of acne severity on depressive symptoms among adolescents and young adults,” Frontiers in Psychology, vol. 14, 2023.",
    "[10] D. Z. Eichenfield, J. Sprague, and L. F. Eichenfield, “Management of acne vulgaris: A review,” JAMA, vol. 326, no. 20, pp. 2055–2067, 2021.",
    "[11] S. Wongvibulsin et al., “Current state of dermatology mobile applications with AI features,” JAMA Dermatology, vol. 160, no. 6, 2024.",
    "[12] R. V. Reynolds et al., “Guidelines of care for the management of acne vulgaris,” J. Amer. Acad. Dermatol., vol. 90, no. 5, 2024.",
    "[13] S. P. Choy et al., “Systematic review of deep learning image analyses for skin disease,” npj Digital Medicine, vol. 6, Art. no. 180, 2023.",
    "[14] Q. T. Huynh et al., “Automatic acne detection and severity grading using smartphone images and AI,” Diagnostics, vol. 12, no. 8, 2022.",
    "[15] Y. Lin et al., “KIEGLFN: A unified acne grading framework on face images,” Comput. Methods Programs Biomed., vol. 221, 2022.",
    "[16] S. Liu et al., “AcneGrader: An ensemble pruning of deep learning base models to grade acne,” Skin Research and Technology, vol. 28, no. 5, pp. 677–688, 2022.",
    "[17] H. Wen et al., “Acne detection and severity evaluation with interpretable CNN models,” Technology and Health Care, vol. 30, no. 1, pp. 143–153, 2022.",
    "[18] Y. Lin et al., “DED: Diagnostic Evidence Distillation for acne severity grading,” Expert Systems with Applications, vol. 228, 2023.",
    "[19] P. K. Nguyen et al., “ACNE8M: An acnes detection and differential diagnosis system using AI,” VNUHCM J. Sci. Technol. Dev., vol. 27, no. 3, pp. 3550–3561, 2024.",
    "[20] L. Gazeau et al., “AcneAI: A new acne severity assessment method using digital images and deep learning,” in Proc. MICCAI, 2024, pp. 68–78.",
    "[21] N. Gao et al., “Evaluation of an acne lesion detection and severity grading model,” Scientific Reports, vol. 15, Art. no. 1119, 2025.",
    "[22] T. A. S. Viana et al., “ClearFace: Facial Acne Detection and Classification Using YOLOv11 and EfficientNet-B0,” in SIBGRAPI Extended Papers, 2025.",
    "[23] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proc. IEEE CVPR, 2016, pp. 770–778.",
]
for r in refs:
    add_para(r)

doc.save(OUT)
print("Saved:", OUT)
