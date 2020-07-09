import 'package:ml_algo/ml_algo.dart';
import 'package:ml_algo/src/classifier/classifier.dart';
import 'package:ml_algo/src/common/exception/invalid_metric_type_exception.dart';
import 'package:ml_algo/src/di/dependencies.dart';
import 'package:ml_algo/src/helpers/features_target_split.dart';
import 'package:ml_algo/src/metric/metric_factory.dart';
import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_algo/src/model_selection/model_assessor/model_assessor.dart';
import 'package:ml_algo/src/predictor/predictor.dart';
import 'package:ml_dataframe/ml_dataframe.dart';

class RegressorAssessorImpl implements ModelAssessor<Predictor> {
  RegressorAssessorImpl(this._metricFactory);

  static const List<MetricType> _allowedMetricTypes = [
    MetricType.rmse,
    MetricType.mape,
  ];

  final MetricFactory _metricFactory;

  @override
  double assess(
      Predictor regressor,
      MetricType metricType,
      DataFrame samples,
      ) {

    if (!_allowedMetricTypes.contains(metricType)) {
      throw InvalidMetricTypeException(
          metricType, _allowedMetricTypes);
    }

    final splits = featuresTargetSplit(
      samples,
      targetNames: regressor.targetNames,
    ).toList();
    final featuresFrame = splits[0];
    final originalLabelsFrame = splits[1];
    final metric = _metricFactory
        .createByType(metricType);
    final predictedLabels = regressor
        .predict(featuresFrame)
        .toMatrix(regressor.dtype);
    final originalLabels = originalLabelsFrame
        .toMatrix(regressor.dtype);

    return metric
        .getScore(predictedLabels, originalLabels);
  }
}
