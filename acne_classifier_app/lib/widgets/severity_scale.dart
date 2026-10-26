import 'package:flutter/material.dart';

import '../theme/app_theme.dart';

/// The four-segment Clear/Mild/Moderate/Severe bar with a small pointer
/// under the active segment, as shown on the result screen.
class SeverityScale extends StatelessWidget {
  final int level;

  const SeverityScale({super.key, required this.level});

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        final segmentWidth = constraints.maxWidth / 4;
        return Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            SizedBox(
              width: constraints.maxWidth,
              height: 10,
              child: Stack(
                clipBehavior: Clip.none,
                children: [
                  AnimatedPositioned(
                    duration: AppDurations.medium,
                    curve: Curves.easeOutCubic,
                    left: segmentWidth * level + segmentWidth / 2 - 7,
                    top: 0,
                    child: CustomPaint(
                      size: const Size(14, 8),
                      painter: _TrianglePainter(AppColors.forSeverity(level)),
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 4),
            Row(
              children: List.generate(4, (i) {
                return Expanded(
                  child: Container(
                    height: 8,
                    margin: EdgeInsets.only(right: i == 3 ? 0 : 4),
                    decoration: BoxDecoration(
                      color: AppColors.forSeverity(i).withValues(alpha: i == level ? 1 : 0.28),
                      borderRadius: BorderRadius.circular(AppRadius.pill),
                    ),
                  ),
                );
              }),
            ),
            const SizedBox(height: 8),
            Row(
              children: List.generate(4, (i) {
                final active = i == level;
                return Expanded(
                  child: Text(
                    kSeverityLabels[i],
                    textAlign: TextAlign.center,
                    style: TextStyle(
                      fontSize: 12.5,
                      fontWeight: active ? FontWeight.w700 : FontWeight.w500,
                      color: active ? AppColors.forSeverity(i) : AppColors.textSecondary,
                    ),
                  ),
                );
              }),
            ),
          ],
        );
      },
    );
  }
}

class _TrianglePainter extends CustomPainter {
  final Color color;
  _TrianglePainter(this.color);

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()..color = color;
    final path = Path()
      ..moveTo(size.width / 2, size.height)
      ..lineTo(0, 0)
      ..lineTo(size.width, 0)
      ..close();
    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(covariant _TrianglePainter oldDelegate) => oldDelegate.color != color;
}
