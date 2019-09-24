import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_algo/src/math/randomizer/randomizer.dart';
import 'package:ml_algo/src/math/randomizer/randomizer_factory.dart';
import 'package:ml_algo/src/math/randomizer/randomizer_factory_impl.dart';
import 'package:ml_algo/src/solver/linear/convergence_detector/convergence_detector.dart';
import 'package:ml_algo/src/solver/linear/convergence_detector/convergence_detector_factory.dart';
import 'package:ml_algo/src/solver/linear/convergence_detector/convergence_detector_factory_impl.dart';
import 'package:ml_algo/src/solver/linear/gradient/learning_rate_generator/learning_rate_generator.dart';
import 'package:ml_algo/src/solver/linear/gradient/learning_rate_generator/learning_rate_generator_factory.dart';
import 'package:ml_algo/src/solver/linear/gradient/learning_rate_generator/learning_rate_generator_factory_impl.dart';
import 'package:ml_algo/src/solver/linear/gradient/learning_rate_generator/learning_rate_type.dart';
import 'package:ml_algo/src/solver/linear/initial_weights_generator/initial_weights_generator.dart';
import 'package:ml_algo/src/solver/linear/initial_weights_generator/initial_weights_generator_factory.dart';
import 'package:ml_algo/src/solver/linear/initial_weights_generator/initial_weights_generator_factory_impl.dart';
import 'package:ml_algo/src/solver/linear/initial_weights_generator/initial_weights_type.dart';
import 'package:ml_algo/src/solver/linear/linear_optimizer.dart';
import 'package:ml_algo/src/utils/parameter_default_values.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:xrange/integers.dart';

class GradientOptimizer implements LinearOptimizer {
  GradientOptimizer(Matrix points, Matrix labels, {
    DType dtype = ParameterDefaultValues.dtype,

    CostFunction costFunction,

    RandomizerFactory randomizerFactory =
      const RandomizerFactoryImpl(),

    LearningRateGeneratorFactory learningRateGeneratorFactory =
      const LearningRateGeneratorFactoryImpl(),

    InitialWeightsGeneratorFactory initialWeightsGeneratorFactory =
      const InitialWeightsGeneratorFactoryImpl(),

    ConvergenceDetectorFactory convergenceDetectorFactory =
      const ConvergenceDetectorFactoryImpl(),

    LearningRateType learningRateType,
    InitialWeightsType initialWeightsType,
    double initialLearningRate = ParameterDefaultValues.initialLearningRate,
    double minCoefficientsUpdate = ParameterDefaultValues.minCoefficientsUpdate,
    int iterationLimit = ParameterDefaultValues.iterationsLimit,
    double lambda,
    int batchSize,
    int randomSeed,
  })  :
        _points = points,
        _labels = labels,
        _lambda = lambda ?? 0.0,
        _batchSize = batchSize,
        _initialWeightsGenerator =
            initialWeightsGeneratorFactory.fromType(initialWeightsType, dtype),
        _learningRateGenerator =
            learningRateGeneratorFactory.fromType(learningRateType),
        _costFunction = costFunction,
        _convergenceDetector = convergenceDetectorFactory.create(
            minCoefficientsUpdate, iterationLimit),
        _randomizer = randomizerFactory.create(randomSeed) {
    if (batchSize < 1 || batchSize > points.rowsNum) {
      throw RangeError.range(batchSize, 1, points.rowsNum, 'Invalid batch size '
          'value');
    }
    _learningRateGenerator.init(initialLearningRate ?? 1.0);
  }

  final Matrix _points;
  final Matrix _labels;
  final Randomizer _randomizer;
  final CostFunction _costFunction;
  final LearningRateGenerator _learningRateGenerator;
  final InitialWeightsGenerator _initialWeightsGenerator;
  final ConvergenceDetector _convergenceDetector;

  final double _lambda;
  final int _batchSize;

  @override
  Matrix findExtrema({Matrix initialWeights,
    bool isMinimizingObjective = true}) {
    final batchSize =
        _batchSize >= _points.rowsNum ? _points.rowsNum : _batchSize;

    Matrix coefficients = initialWeights ??
        Matrix.fromColumns(List.generate(_labels.columnsNum,
            (i) => _initialWeightsGenerator.generate(_points.columnsNum)));

    var iteration = 0;
    var coefficientsDiff = double.maxFinite;

    while (!_convergenceDetector.isConverged(coefficientsDiff, iteration)) {
      final learningRate = _learningRateGenerator.getNextValue();
      final newCoefficients = _generateCoefficients(
          coefficients, _labels, learningRate, batchSize,
          isMinimization: isMinimizingObjective);
      coefficientsDiff = (newCoefficients - coefficients).norm();
      iteration++;
      coefficients = newCoefficients;
    }

    _learningRateGenerator.stop();

    return coefficients;
  }

  Matrix _generateCoefficients(
      Matrix coefficients, Matrix labels, double eta, int batchSize,
      {bool isMinimization = true}) {
    final range = _getBatchRange(batchSize);
    final start = range.first;
    final end = range.last;
    final pointsBatch = _points
        .sample(rowIndices: integers(start, end, upperClosed: false));
    final labelsBatch = labels
        .sample(rowIndices: integers(start, end, upperClosed: false));

    return _makeGradientStep(coefficients, pointsBatch, labelsBatch, eta,
        isMinimization: isMinimization);
  }

  Iterable<int> _getBatchRange(int batchSize) => _randomizer
      .getIntegerInterval(0, _points.rowsNum, intervalLength: batchSize);

  Matrix _makeGradientStep(
      Matrix coefficients, Matrix points, Matrix labels, double eta,
      {bool isMinimization = true}) {
    final gradient = _costFunction.getGradient(points, coefficients, labels);
    final regularizedCoefficients = _regularize(eta, _lambda, coefficients);
    return isMinimization
        ? regularizedCoefficients - gradient * eta
        : regularizedCoefficients + gradient * eta;
  }

  Matrix _regularize(double eta, double lambda, Matrix coefficients) {
    if (lambda == 0) {
      return coefficients;
    } else {
      return coefficients * (1 - 2 * eta * lambda);
    }
  }
}
