// 0.161 sec

import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:benchmark_harness/benchmark_harness.dart';
import 'package:dart_ml/dart_ml.dart';
import 'package:csv/csv.dart' as csv;

List<Float32x4Vector> features;
Float32x4Vector labels;
GradientRegressor regressor;

class GDRegressorBenchmark extends BenchmarkBase {
  const GDRegressorBenchmark() : super('Gradient descent regressor');

  static void main() {
    const GDRegressorBenchmark().report();
  }

  @override
  void run() {
    regressor.fit(features, labels);
  }

  @override
  void setup() {
    regressor = GradientRegressor();
  }

  void tearDown() {}
}

Future main() async {
  final csvCodec = csv.CsvCodec();
  final input = File('example/datasets/advertising.csv').openRead();
  final fields = (await input.transform(utf8.decoder)
      .transform(csvCodec.decoder).toList() as List<List<num>>)
      .sublist(1);

  List<double> extractFeatures(List<num> item) => item.sublist(0, 3)
      .map((num feature) => feature.toDouble())
      .toList();

  features = fields
      .map((List<num> item) => Float32x4Vector.from(extractFeatures(item)))
      .toList(growable: false);

  labels = Float32x4Vector.from(fields.map((List<num> item) => item.last.toDouble()));

  GDRegressorBenchmark.main();
}