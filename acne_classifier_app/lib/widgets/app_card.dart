import 'package:flutter/material.dart';

import '../theme/app_theme.dart';

/// A flat, outlined card matching the app's visual language. Styled
/// directly on the widget (rather than via ThemeData.cardTheme) so it does
/// not depend on which Card theme class a given Flutter version expects.
class AppCard extends StatelessWidget {
  final Widget child;
  final EdgeInsetsGeometry padding;

  const AppCard({
    super.key,
    required this.child,
    this.padding = const EdgeInsets.all(18),
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: padding,
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(AppRadius.md),
        border: Border.all(color: AppColors.outline),
      ),
      child: child,
    );
  }
}
