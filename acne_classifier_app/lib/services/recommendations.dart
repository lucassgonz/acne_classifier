/// General, non-prescriptive guidance keyed by severity level. Deliberately
/// avoids naming specific medications or brands — that decision belongs to a
/// dermatologist, not this app.
class Recommendation {
  final String summary;
  final List<String> suggestions;

  const Recommendation(this.summary, this.suggestions);
}

const List<Recommendation> kRecommendationsByLevel = [
  Recommendation(
    'Skin looks clear. Keep up your routine.',
    [
      'Continue a gentle cleanser and daily sun protection.',
      'No treatment needed for this region right now.',
    ],
  ),
  Recommendation(
    'Mild acne detected.',
    [
      'A gentle, non-comedogenic cleanser and moisturizer are usually enough.',
      'Avoid picking or scrubbing the area.',
      'Reassess in a few weeks; see a dermatologist if it persists or worsens.',
    ],
  ),
  Recommendation(
    'Moderate acne detected.',
    [
      'Consult a dermatologist for a proper treatment plan.',
      'Maintain a consistent, gentle skincare routine.',
      'Avoid picking the acne — it increases the risk of scarring.',
    ],
  ),
  Recommendation(
    'Severe acne detected.',
    [
      'Please see a dermatologist soon — early treatment reduces the risk of scarring.',
      'Avoid harsh scrubbing or picking in the meantime.',
      'Keep the area clean with a gentle, fragrance-free cleanser.',
    ],
  ),
];

const String kProfessionalDisclaimer =
    'This assessment is informational only and does not replace a diagnosis '
    'from a qualified dermatologist. Always seek professional advice for '
    'treatment decisions.';
