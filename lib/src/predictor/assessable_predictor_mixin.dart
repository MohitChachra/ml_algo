import 'package:ml_algo/src/common/exception/invalid_metric_type_exception.dart';
import 'package:ml_algo/src/di/dependencies.dart';
import 'package:ml_algo/src/helpers/features_target_split.dart';
import 'package:ml_algo/src/metric/metric_factory.dart';
import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_algo/src/model_selection/assessable.dart';
import 'package:ml_algo/src/predictor/predictor.dart';
import 'package:ml_dataframe/ml_dataframe.dart';

mixin AssessablePredictorMixin implements
    Assessable,
    Predictor {

  @override
  double assess(
      DataFrame samples,
      Iterable<String> targetNames,
      MetricType metricType,
  ) {
    if (!allowedMetrics.contains(metricType)) {
      throw InvalidMetricTypeException(
          metricType, allowedMetrics);
    }

    final splits = featuresTargetSplit(
      samples,
      targetNames: targetNames,
    ).toList();
    final metric = dependencies
        .getDependency<MetricFactory>()
        .createByType(metricType);
    final predictedLabels = predict(splits[0])
        .toMatrix(dtype);
    final originalLabels = splits[1]
        .toMatrix(dtype);

    return metric
        .getScore(predictedLabels, originalLabels);
  }
}
