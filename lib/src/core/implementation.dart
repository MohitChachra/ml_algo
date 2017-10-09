import 'dart:math';
import 'dart:typed_data' show Float32List;

import 'package:dart_ml/src/di/injector.dart';
import 'package:di/di.dart';
import 'package:simd_vector/vector.dart';

import 'interface.dart';

part 'package:dart_ml/src/core/data_splitter/factory.dart';
part 'package:dart_ml/src/core/data_splitter/k_fold.dart';
part 'package:dart_ml/src/core/data_splitter/leave_p_out.dart';
part 'package:dart_ml/src/core/loss_function/cross_entropy.dart';
part 'package:dart_ml/src/core/loss_function/logistic_loss.dart';
part 'package:dart_ml/src/core/loss_function/loss_function_factory.dart';
part 'package:dart_ml/src/core/loss_function/squared_loss.dart';
part 'package:dart_ml/src/core/math/math.dart';
part 'package:dart_ml/src/core/math/math_analysis/gradient_calculator_impl.dart';
part 'package:dart_ml/src/core/math/randomizer/randomizer_impl.dart';
part 'package:dart_ml/src/core/metric/classification/accuracy.dart';
part 'package:dart_ml/src/core/metric/classification/metric_factory.dart';
part 'package:dart_ml/src/core/metric/factory.dart';
part 'package:dart_ml/src/core/metric/regression/mape.dart';
part 'package:dart_ml/src/core/metric/regression/metric_factory.dart';
part 'package:dart_ml/src/core/metric/regression/rmse.dart';
part 'package:dart_ml/src/core/optimizer/gradient/base_impl.dart';
part 'package:dart_ml/src/core/optimizer/gradient/batch.dart';
part 'package:dart_ml/src/core/optimizer/gradient/factory.dart';
part 'package:dart_ml/src/core/optimizer/gradient/initial_weights_generator/initial_weights_generator_factory.dart';
part 'package:dart_ml/src/core/optimizer/gradient/initial_weights_generator/zero_weights_generator.dart';
part 'package:dart_ml/src/core/optimizer/gradient/learning_rate_generator/learning_rate_generator_factory.dart';
part 'package:dart_ml/src/core/optimizer/gradient/learning_rate_generator/simple_learning_rate_generator.dart';
part 'package:dart_ml/src/core/optimizer/gradient/mini_batch.dart';
part 'package:dart_ml/src/core/optimizer/gradient/stochastic.dart';
part 'package:dart_ml/src/core/predictor/base/classifier_base.dart';
part 'package:dart_ml/src/core/predictor/base/predictor_base.dart';
part 'package:dart_ml/src/core/predictor/base/classifier_impl.dart';
part 'package:dart_ml/src/core/predictor/linear/classifier/gradient/logistic_regression.dart';
part 'package:dart_ml/src/core/predictor/linear/regressor/gradient/batch.dart';
part 'package:dart_ml/src/core/predictor/linear/regressor/gradient/mini_batch.dart';
part 'package:dart_ml/src/core/predictor/linear/regressor/gradient/stochastic.dart';
part 'package:dart_ml/src/core/predictor/base/regressor_impl.dart';
part 'package:dart_ml/src/core/score_function/linear.dart';
part 'package:dart_ml/src/core/score_function/score_function_factory.dart';
part 'package:dart_ml/src/di/factory.dart';