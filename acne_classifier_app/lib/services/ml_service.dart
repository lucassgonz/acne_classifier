
import 'package:http/http.dart' as http;
import 'dart:convert';

class MLService {
  static const String _baseUrl = 'YOUR_ML_API_ENDPOINT';

  Future<Map<String, dynamic>?> analyzeImages({
    required String analysisId,
    required List<String> imageUrls,
  }) async {
    try {
      final response = await http.post(
        Uri.parse('$_baseUrl/analyze'),
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer YOUR_API_KEY',
        },
        body: json.encode({
          'analysis_id': analysisId,
          'images': {
            'front': imageUrls[0],
            'right_side': imageUrls[1],
            'left_side': imageUrls[2],
          },
        }),
      );

      if (response.statusCode == 200) {
        return json.decode(response.body);
      } else {
        print('Erro na API de ML: ${response.statusCode}');
        return null;
      }
    } catch (e) {
      print('Erro ao analisar imagens: $e');
      return null;
    }
  }
}