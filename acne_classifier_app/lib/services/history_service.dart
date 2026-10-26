import 'dart:convert';

import 'package:shared_preferences/shared_preferences.dart';

import '../models/analysis_record.dart';

/// Persists analysis results on-device. No account or network round-trip is
/// required to use the app — history simply lives in SharedPreferences as a
/// JSON array, newest first.
class HistoryService {
  static const _key = 'analysis_history_v1';

  Future<List<AnalysisRecord>> load() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getStringList(_key) ?? const [];
    return raw
        .map((e) => AnalysisRecord.fromJson(jsonDecode(e) as Map<String, dynamic>))
        .toList();
  }

  Future<void> add(AnalysisRecord record) async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getStringList(_key) ?? <String>[];
    raw.insert(0, jsonEncode(record.toJson()));
    await prefs.setStringList(_key, raw);
  }

  Future<void> clear() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_key);
  }
}
