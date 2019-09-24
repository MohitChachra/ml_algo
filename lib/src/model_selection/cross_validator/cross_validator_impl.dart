import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_algo/src/model_selection/cross_validator/cross_validator.dart';
import 'package:ml_algo/src/model_selection/data_splitter/splitter.dart';
import 'package:ml_algo/src/model_selection/assessable.dart';
import 'package:ml_algo/src/utils/parameter_default_values.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

class CrossValidatorImpl implements CrossValidator {
  CrossValidatorImpl(DType dtype, this._splitter)
      : dtype = dtype ?? ParameterDefaultValues.dtype;

  final DType dtype;
  final Splitter _splitter;

  @override
  double evaluate(Assessable predictorFactory(Matrix features, Matrix outcomes),
      Matrix observations, Matrix labels, MetricType metric) {
    if (observations.rowsNum != labels.rowsNum) {
      throw Exception(
          'Number of feature objects must be equal to the number of labels!');
    }

    final allIndicesGroups = _splitter.split(observations.rowsNum);
    var score = 0.0;
    var folds = 0;

    for (final testIndices in allIndicesGroups) {
      final testIndicesAsSet = Set<int>.from(testIndices);
      final trainFeatures =
          List<Vector>(observations.rowsNum - testIndicesAsSet.length);
      final trainLabels =
          List<Vector>(observations.rowsNum - testIndicesAsSet.length);

      final testFeatures = List<Vector>(testIndicesAsSet.length);
      final testLabels = List<Vector>(testIndicesAsSet.length);

      int trainPointsCounter = 0;
      int testPointsCounter = 0;

      for (int index = 0; index < observations.rowsNum; index++) {
        if (testIndicesAsSet.contains(index)) {
          testFeatures[testPointsCounter] = observations.getRow(index);
          testLabels[testPointsCounter] = labels.getRow(index);
          testPointsCounter++;
        } else {
          trainFeatures[trainPointsCounter] = observations.getRow(index);
          trainLabels[trainPointsCounter] = labels.getRow(index);
          trainPointsCounter++;
        }
      }

      final predictor = predictorFactory(
        Matrix.fromRows(trainFeatures, dtype: dtype),
        Matrix.fromRows(trainLabels, dtype: dtype),
      );

      score += predictor.assess(
          Matrix.fromRows(testFeatures, dtype: dtype),
          Matrix.fromRows(testLabels, dtype: dtype),
          metric
      );
      folds++;
    }

    return score / folds;
  }
}
