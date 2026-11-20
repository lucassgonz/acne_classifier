package br.edu.ifce.acneapp.data

import android.content.Context
import android.net.Uri
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import java.io.File

/**
 * Keeps analyses on the device only: the records as a JSON array in
 * SharedPreferences (newest first) and a private copy of each photo in the app's
 * files directory. Nothing is sent to a server.
 */
class HistoryRepository(context: Context) {
    private val appContext = context.applicationContext
    private val prefs = appContext.getSharedPreferences("history", Context.MODE_PRIVATE)
    private val photosDir = File(appContext.filesDir, "photos").apply { mkdirs() }

    suspend fun load(): List<AnalysisRecord> = withContext(Dispatchers.IO) {
        val arr = JSONArray(prefs.getString(KEY, "[]"))
        (0 until arr.length()).mapNotNull { i ->
            runCatching { AnalysisRecord.fromJson(arr.getJSONObject(i)) }.getOrNull()
        }
    }

    suspend fun find(id: String): AnalysisRecord? = load().firstOrNull { it.id == id }

    suspend fun add(record: AnalysisRecord) = withContext(Dispatchers.IO) {
        val old = JSONArray(prefs.getString(KEY, "[]"))
        val updated = JSONArray().put(record.toJson())
        for (i in 0 until old.length()) updated.put(old.get(i))
        prefs.edit().putString(KEY, updated.toString()).apply()
    }

    /** Copies a photo (camera file or gallery URI) into private storage and returns its path. */
    suspend fun importPhoto(source: Uri, id: String): String = withContext(Dispatchers.IO) {
        val target = File(photosDir, "$id.jpg")
        val input = if (source.scheme == "file") File(requireNotNull(source.path)).inputStream()
        else requireNotNull(appContext.contentResolver.openInputStream(source)) { "Could not open the image." }
        input.use { inp -> target.outputStream().use { inp.copyTo(it) } }
        target.absolutePath
    }

    fun newPhotoFile(id: String): File = File(photosDir, "$id.jpg")

    companion object {
        private const val KEY = "analysis_history_v1"
    }
}
