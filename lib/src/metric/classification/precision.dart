import 'package:ml_algo/src/metric/metric.dart';
import 'package:ml_linalg/matrix.dart';

class PrecisionMetric implements Metric {
  const PrecisionMetric();

  @override
  double getScore(Matrix predictedLabels, Matrix origLabels) {
    // TODO: potential performance breach - maybe we don't need to map every row
    // TODO: to a 1-0 vector?
    final allPredictedPositiveCounts = predictedLabels
        .reduceRows((counts, row) => counts + row.mapToVector(
            (count) => count > 0 ? 1 : 0));

    // Let's say we have the following data:
    //
    // orig labels | predicted labels
    // -------------------------------
    //     1       |        1
    //     1       |        0
    //     0       |        1
    //     0       |        0
    //     1       |        1
    //--------------------------------
    //
    // in order to count correctly predicted positive labels in matrix notation
    // we may just subtract these two matrices-columns:
    //
    // 1 - 1 =  0
    // 1 - 0 =  1
    // 0 - 1 = -1
    // 0 - 0 =  0
    // 1 - 1 =  0
    //
    // we see that matrices subtraction in case of original positive label and a
    // correctly predicted one gives 0, thus we need to count zeroes in the
    // resulting matrix
    final correctPositiveCounts = (origLabels - predictedLabels)
        .reduceRows((counts, row) => counts + row.mapToVector(
            (diff) => diff == 0 ? 1 : 0));

    return (correctPositiveCounts / allPredictedPositiveCounts).mean();
  }
}
