import 'package:ml_algo/src/metric/classification/accuracy.dart';
import 'package:ml_algo/src/metric/classification/precision.dart';
import 'package:ml_algo/src/metric/metric.dart';
import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_algo/src/metric/regression/mape.dart';
import 'package:ml_algo/src/metric/regression/rmse.dart';

class MetricFactory {
  static Metric createByType(MetricType type) {
    switch (type) {
      case MetricType.rmse:
        return const RmseMetric();

      case MetricType.mape:
        return const MapeMetric();

      case MetricType.accuracy:
        return const AccuracyMetric();

      case MetricType.precision:
        return const PrecisionMetric();

      default:
        throw UnsupportedError('Unsupported metric type $type');
    }
  }
}
