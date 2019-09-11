import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_algo/src/math/randomizer/randomizer_factory.dart';
import 'package:ml_algo/src/solver/linear/gradient/learning_rate_generator/learning_rate_generator_factory.dart';
import 'package:ml_algo/src/solver/linear/gradient/learning_rate_generator/learning_rate_type.dart';
import 'package:ml_algo/src/solver/linear/initial_weights_generator/initial_weights_generator_factory.dart';
import 'package:ml_algo/src/solver/linear/initial_weights_generator/initial_weights_type.dart';
import 'package:ml_algo/src/solver/linear/linear_optimizer.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';

abstract class LinearOptimizerFactory {
  LinearOptimizer gradient(Matrix points, Matrix labels, {
    DType dtype,
    CostFunction costFunction,
    RandomizerFactory randomizerFactory,
    LearningRateGeneratorFactory learningRateGeneratorFactory,
    InitialWeightsGeneratorFactory initialWeightsGeneratorFactory,
    LearningRateType learningRateType,
    InitialWeightsType initialWeightsType,
    double initialLearningRate,
    double minCoefficientsUpdate,
    int iterationLimit,
    double lambda,
    int batchSize,
    int randomSeed,
  });

  LinearOptimizer coordinate(Matrix points, Matrix labels, {
    DType dtype,
    InitialWeightsGeneratorFactory initialWeightsGeneratorFactory,
    CostFunction costFunction,
    double minCoefficientsDiff,
    int iterationLimit,
    double lambda,
    InitialWeightsType initialWeightsType,
  });
}