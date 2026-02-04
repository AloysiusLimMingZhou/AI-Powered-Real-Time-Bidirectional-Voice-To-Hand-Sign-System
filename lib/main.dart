import 'package:firebase_auth/firebase_auth.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:google_sign_in/google_sign_in.dart';
import 'package:ai_voice_to_hand_signs_project/features/auth/repositories/auth.repositories.dart';
import 'package:ai_voice_to_hand_signs_project/features/auth/screens/login/login.dart';
import 'package:ai_voice_to_hand_signs_project/firebase_options.dart';
import 'package:permission_handler/permission_handler.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp(
    options: DefaultFirebaseOptions.currentPlatform,
    demoProjectId: "demo-project-id",
  );

  await FirebaseAuth.instance.useAuthEmulator("localhost", 9099);

  // Initialize Google Sign-In with serverClientId (Web Client ID from google-services.json)
  await GoogleSignIn.instance.initialize(
    serverClientId:
        '37627043520-iarcpu2agpqqavqemu0suc0n87c64t0v.apps.googleusercontent.com',
  );

  // Initialize Authentication Controller
  Get.put(AuthRepositories());

  // Request Permissions
  await [Permission.camera, Permission.microphone].request();

  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return GetMaterialApp(
      title: 'Hackathon Project',
      theme: ThemeData(fontFamily: 'Poppins', brightness: Brightness.dark),
      home: const LoginScreen(),
    );
  }
}
