import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

import '../widgets/app_bottom_nav.dart';
import 'about_screen.dart';
import 'history_screen.dart';
import 'home_screen.dart';

/// Top-level tabs (Home / History / About). The capture → region → analyzing
/// → result flow is pushed on top of this shell as full-screen routes, so it
/// keeps its own back-arrow navigation as in the reference design.
class HomeShell extends StatefulWidget {
  final List<CameraDescription> cameras;

  const HomeShell({super.key, required this.cameras});

  @override
  State<HomeShell> createState() => _HomeShellState();
}

class _HomeShellState extends State<HomeShell> {
  int _index = 0;
  final _historyKey = GlobalKey<HistoryScreenState>();

  void _goTo(int index) {
    setState(() => _index = index);
    if (index == 1) _historyKey.currentState?.refresh();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: IndexedStack(
        index: _index,
        children: [
          HomeScreen(cameras: widget.cameras, onViewHistory: () => _goTo(1)),
          HistoryScreen(key: _historyKey),
          const AboutScreen(),
        ],
      ),
      bottomNavigationBar: AppBottomNav(currentIndex: _index, onTap: _goTo),
    );
  }
}
