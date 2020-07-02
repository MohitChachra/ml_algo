import 'package:ml_algo/src/common/exception/invalid_metric_type_exception.dart';
import 'package:ml_algo/src/helpers/binarize_column_matrix.dart';
import 'package:ml_algo/src/helpers/features_target_split.dart';
import 'package:ml_algo/src/metric/factory.dart';
import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_algo/src/model_selection/assessable.dart';
import 'package:ml_algo/src/predictor/predictor.dart';
import 'package:ml_dataframe/ml_dataframe.dart';

mixin AssessableClassifierMixin implements
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
    final metric = MetricFactory
        .createByType(metricType);
    final predictedLabels = predict(splits[0])
        .toMatrix(dtype);
    final originalLabels = splits[1]
        .toMatrix(dtype);

    final processedPrediction = predictedLabels.columnsNum == 1
        ? binarizeColumnMatrix(predictedLabels)
        : predictedLabels;
    final processedOriginal = originalLabels.columnsNum == 1
        ? binarizeColumnMatrix(originalLabels)
        : originalLabels;

    return metric.getScore(
        processedPrediction,
        processedOriginal,
    );
  }
}
