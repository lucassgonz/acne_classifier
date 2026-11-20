package br.edu.ifce.acneapp.data

/**
 * General, non-prescriptive guidance per severity level. It intentionally names no
 * medication or brand: that decision belongs to a dermatologist.
 */
class Recommendation(val summary: String, val suggestions: List<String>)

val RECOMMENDATIONS_BY_LEVEL = listOf(
    Recommendation(
        "Skin looks clear. Keep up your routine.",
        listOf(
            "Continue a gentle cleanser and daily sun protection.",
            "No treatment needed for this region right now.",
        ),
    ),
    Recommendation(
        "Mild acne detected.",
        listOf(
            "A gentle, non-comedogenic cleanser and moisturizer are usually enough.",
            "Avoid picking or scrubbing the area.",
            "Reassess in a few weeks; see a dermatologist if it persists or worsens.",
        ),
    ),
    Recommendation(
        "Moderate acne detected.",
        listOf(
            "Consult a dermatologist for a proper treatment plan.",
            "Maintain a consistent, gentle skincare routine.",
            "Avoid picking the acne, which increases the risk of scarring.",
        ),
    ),
    Recommendation(
        "Severe acne detected.",
        listOf(
            "Please see a dermatologist soon: early treatment reduces the risk of scarring.",
            "Avoid harsh scrubbing or picking in the meantime.",
            "Keep the area clean with a gentle, fragrance-free cleanser.",
        ),
    ),
)

const val PROFESSIONAL_DISCLAIMER =
    "This assessment is informational only and does not replace a diagnosis from a qualified " +
        "dermatologist. Always seek professional advice for treatment decisions."

val SEVERITY_LABELS = listOf("Clear", "Mild", "Moderate", "Severe")
