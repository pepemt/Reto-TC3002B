import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:url_launcher/url_launcher.dart';

class PredictionScreen extends StatefulWidget {
  @override
  _PredictionScreenState createState() => _PredictionScreenState();
}

class _PredictionScreenState extends State<PredictionScreen> {
  final TextEditingController _inputController = TextEditingController();

  final List<String> _modelos = [
    'logistic_regression',
    'random_forest',
    'k-nearest_neighbors',
    'svm',
    'decision_tree',
    'naive_bayes',
    'xgboost',
  ];

  String _modeloSeleccionado = 'xgboost';
  Map<String, dynamic>? _resultado;

  Future<void> _onPredict() async {
    final texto = _inputController.text.trim();
    if (texto.isEmpty) return;

    final uri = Uri.parse('http://127.0.0.1:8000/predict/$_modeloSeleccionado');

    try {
      final response = await http.post(
        uri,
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          "title": "usuario",
          "text": texto,
        }),
      );

      if (response.statusCode == 200) {
        final json = jsonDecode(response.body);
        setState(() {
          _resultado = json;
        });
      } else {
        setState(() {
          _resultado = {"error": "Error del servidor (${response.statusCode})"};
        });
      }
    } catch (e) {
      setState(() {
        _resultado = {"error": "No se pudo conectar con la API."};
      });
    }
  }

  Future<void> _llamarAyuda() async {
    const phoneNumber = 'tel:8002738255'; // Puedes cambiar por el de tu país
    final uri = Uri.parse(phoneNumber);
    if (await canLaunchUrl(uri)) {
      await launchUrl(uri);
    } else {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('No se pudo iniciar la llamada')),
      );
    }
  }

  @override
  void dispose() {
    _inputController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Predicción')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            Text(
              'Selecciona el modelo de IA:',
              style: Theme.of(context).textTheme.bodyMedium,
            ),
            const SizedBox(height: 8),
            DropdownButtonFormField<String>(
              value: _modeloSeleccionado,
              onChanged: (value) {
                if (value != null) {
                  setState(() => _modeloSeleccionado = value);
                }
              },
              decoration: InputDecoration(
                labelText: 'Modelo de IA',
                prefixIcon: const Icon(Icons.smart_toy),
                border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
                filled: true,
                fillColor: Theme.of(context).colorScheme.surfaceVariant.withOpacity(0.2),
              ),
              items: _modelos.map((modelo) {
                return DropdownMenuItem<String>(
                  value: modelo,
                  child: Text(
                    modelo.replaceAll('_', ' ').toUpperCase(),
                    style: const TextStyle(letterSpacing: 1),
                  ),
                );
              }).toList(),
            ),
            const SizedBox(height: 20),
            TextField(
              controller: _inputController,
              maxLines: 3,
              decoration: const InputDecoration(
                labelText: '¿Cómo te sientes hoy?',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 16),
            ElevatedButton(
              onPressed: _onPredict,
              child: const Text('Predecir'),
            ),
            const SizedBox(height: 20),
            if (_resultado != null)
              Expanded(
                child: Card(
                  margin: const EdgeInsets.only(top: 12),
                  elevation: 4,
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
                  child: Padding(
                    padding: const EdgeInsets.all(20.0),
                    child: _resultado!.containsKey('error')
                        ? Text(
                            _resultado!['error'],
                            style: const TextStyle(color: Colors.red, fontSize: 16),
                          )
                        : Column(
                            crossAxisAlignment: CrossAxisAlignment.center,
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              Text(
                                _modeloSeleccionado.replaceAll('_', ' ').toUpperCase(),
                                style: Theme.of(context)
                                    .textTheme
                                    .headlineMedium
                                    ?.copyWith(fontWeight: FontWeight.bold),
                                textAlign: TextAlign.center,
                              ),
                              const SizedBox(height: 20),
                              Icon(
                                _resultado![_modeloSeleccionado] == true
                                    ? Icons.warning_amber_rounded
                                    : Icons.check_circle_outline,
                                size: 60,
                                color: _resultado![_modeloSeleccionado] == true
                                    ? Colors.orange
                                    : Colors.green,
                              ),
                              const SizedBox(height: 16),
                              Text(
                                _resultado![_modeloSeleccionado] == true
                                    ? 'Texto detectado como suicida o depresivo'
                                    : 'Texto NO detectado como suicida o depresivo',
                                style: Theme.of(context).textTheme.bodyLarge,
                                textAlign: TextAlign.center,
                              ),
                            ],
                          ),
                  ),
                ).animate().fadeIn().slide(duration: 800.ms),
              ),
            const SizedBox(height: 20),
            ElevatedButton.icon(
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.redAccent,
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
              ),
              icon: const Icon(Icons.phone),
              label: const Text(
                'Hablar con alguien',
                style: TextStyle(fontSize: 16),
              ),
              onPressed: _llamarAyuda,
            ),
          ],
        ),
      ),
    );
  }
}
