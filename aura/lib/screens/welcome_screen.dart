import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:flutter_animate/flutter_animate.dart';

class WelcomeScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Theme.of(context).colorScheme.primary,  // Fondo violeta
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            // Ícono de bienvenida con animación de fade-in
            Icon(Icons.self_improvement, size: 100, color: Colors.white)
              .animate()
              .fadeIn(duration: Duration(milliseconds: 1200)),
            const SizedBox(height: 20),
            // Texto de bienvenida animado (fade + slight slide-up)
            Text(
              'Bienvenido a AURA',
              style: GoogleFonts.nunito(
                color: Colors.white, fontSize: 28, fontWeight: FontWeight.bold
              ),
            )
              .animate()
              .fadeIn(duration: Duration(milliseconds: 1200))
              .slide(duration: Duration(milliseconds: 600), begin: Offset(0, 0.2)),
            const SizedBox(height: 40),
            // Botón para continuar al Login
            ElevatedButton(
              onPressed: () {
                Navigator.pushNamed(context, '/login');
              },
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.white,
                foregroundColor: Theme.of(context).colorScheme.primary,
                textStyle: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
              ),
              child: const Text('Comenzar'),
            ),
          ],
        ),
      ),
    );
  }
}
