package br.edu.ifce.acneapp.data

import org.json.JSONArray
import org.json.JSONObject

/** Severity result for one anatomical region within an analysis. */
data class RegionRecord(
    val region: String, // display label, e.g. "Forehead"
    val severityLevel: Int, // 0=Clear .. 3=Severe
    val confidence: Float, // 0..1
) {
    fun toJson(): JSONObject = JSONObject()
        .put("region", region)
        .put("severityLevel", severityLevel)
        .put("confidence", confidence.toDouble())

    companion object {
        fun fromJson(o: JSONObject) = RegionRecord(
            region = o.getString("region"),
            severityLevel = o.getInt("severityLevel"),
            confidence = o.getDouble("confidence").toFloat(),
        )
    }
}

/** One complete analysis: the photo and a result for each detected region. */
data class AnalysisRecord(
    val id: String,
    val imagePath: String,
    val regions: List<RegionRecord>,
    val createdAtMillis: Long,
) {
    /** The overall grade follows the most affected region. */
    val overallSeverity: Int get() = regions.maxOfOrNull { it.severityLevel } ?: 0

    /** Mean confidence of the regions that share the overall grade. */
    val overallConfidence: Float
        get() {
            val matching = regions.filter { it.severityLevel == overallSeverity }
            return if (matching.isEmpty()) 0f else matching.map { it.confidence }.average().toFloat()
        }

    fun toJson(): JSONObject = JSONObject()
        .put("id", id)
        .put("imagePath", imagePath)
        .put("regions", JSONArray().apply { regions.forEach { put(it.toJson()) } })
        .put("createdAtMillis", createdAtMillis)

    companion object {
        fun fromJson(o: JSONObject): AnalysisRecord {
            val arr = o.getJSONArray("regions")
            return AnalysisRecord(
                id = o.getString("id"),
                imagePath = o.getString("imagePath"),
                regions = (0 until arr.length()).map { RegionRecord.fromJson(arr.getJSONObject(it)) },
                createdAtMillis = o.getLong("createdAtMillis"),
            )
        }
    }
}
