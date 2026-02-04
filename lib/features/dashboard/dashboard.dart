import 'package:flutter/material.dart';
import 'package:iconsax_flutter/iconsax_flutter.dart';
import 'package:ai_voice_to_hand_signs_project/features/auth/repositories/auth.repositories.dart';
import 'package:ai_voice_to_hand_signs_project/util/constants/colors.dart';

class DashboardScreen extends StatelessWidget {
  const DashboardScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: TColors.darkBackground,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        title: Text("Dashboard", style: TextStyle(color: TColors.textPrimary)),
        actions: [
          IconButton(
            onPressed: () {
              AuthRepositories.instance.logout();
            },
            icon: const Icon(Iconsax.logout, color: TColors.white),
          ),
        ],
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(
              "Welcome to the Future of Learning",
              style: TextStyle(
                color: TColors.textPrimary,
                fontSize: 20,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 20),
            Container(
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                gradient: TColors.linearGradient,
                borderRadius: BorderRadius.circular(20),
              ),
              child: const Text(
                "Start Recording",
                style: TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
          ],
        ),
      ),
      bottomNavigationBar: NavigationBar(
        backgroundColor: TColors.darkContainer,
        indicatorColor: TColors.primary.withAlpha(20),
        destinations: const [
          NavigationDestination(
            icon: Icon(Iconsax.home, color: TColors.textSecondary),
            selectedIcon: Icon(Iconsax.home, color: TColors.primary),
            label: "Home",
          ),
          NavigationDestination(
            icon: Icon(Iconsax.camera, color: TColors.textSecondary),
            selectedIcon: Icon(Iconsax.camera, color: TColors.primary),
            label: "Sign",
          ),
          NavigationDestination(
            icon: Icon(Iconsax.user, color: TColors.textSecondary),
            selectedIcon: Icon(Iconsax.user, color: TColors.primary),
            label: "Profile",
          ),
        ],
      ),
    );
  }
}
