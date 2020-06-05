import 'dart:math' as math;

import 'package:ml_algo/src/exception/logit_scores_matrix_dimension_exception.dart';
import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

class Float64InverseLogitLinkFunction implements LinkFunction {
  const Float64InverseLogitLinkFunction();

  static final upperBound = 10;
  static final lowerBound = -10;

  @override
  Matrix link(Matrix scores) {
    if (scores.columnsNum != 1) {
      throw LogitScoresMatrixDimensionException(scores.columnsNum);
    }

    return scores
        .mapColumns(
            (column) => Vector.fromList(
                column.map(scoreToProbability).toList(),
                dtype: DType.float64,
            ),
    );
  }

  double scoreToProbability(double score) {
    if (score >= upperBound) {
      return 1;
    }

    if (score <= lowerBound) {
      return 0;
    }

    final exponentToScore = math.exp(score);

    return exponentToScore / (1 + exponentToScore);
  }
}