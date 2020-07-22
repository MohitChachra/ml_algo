import 'package:ml_algo/src/di/dependencies.dart';
import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_algo/src/model_selection/assessable.dart';
import 'package:ml_algo/src/model_selection/model_assessor/regressor_assessor.dart';
import 'package:ml_algo/src/predictor/predictor.dart';
import 'package:ml_dataframe/ml_dataframe.dart';

mixin AssessableRegressorMixin implements
    Assessable,
    Predictor {

  @override
  double assess(
      DataFrame samples,
      MetricType metricType,
  ) => dependencies
      .getDependency<RegressorAssessor>()
      .assess(this, metricType, samples);
}
