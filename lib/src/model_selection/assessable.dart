import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_dataframe/ml_dataframe.dart';

/// An interface for a ML model's performance assessment
abstract class Assessable {
  /// Assesses model performance according to provided [metricType]
  double assess(DataFrame observations, Iterable<String> targetNames,
      MetricType metricType);

  /// Returns metrics, applicable for the model
  List<MetricType> get allowedMetrics;
}
