import 'package:flutter/material.dart';
import 'screens/welcome_screen.dart';
import 'screens/login_screen.dart';
import 'screens/prediction_screen.dart';
import 'theme/theme.dart';

void main() {
  runApp(AuraApp());
}

class AuraApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'AURA',
      theme: AURATheme.lightTheme,
      initialRoute: '/',
      routes: {
        '/': (context) => WelcomeScreen(),
        '/login': (context) => LoginScreen(),
        '/predict': (context) => PredictionScreen(),
      },
    );
  }
}
