import 'package:ml_algo/src/classifier/classifier.dart';
import 'package:ml_algo/src/common/exception/invalid_metric_type_exception.dart';
import 'package:ml_algo/src/di/dependencies.dart';
import 'package:ml_algo/src/helpers/features_target_split_interface.dart';
import 'package:ml_algo/src/helpers/normalize_class_labels_interface.dart';
import 'package:ml_algo/src/metric/metric_factory.dart';
import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_algo/src/model_selection/model_assessor/model_assessor.dart';
import 'package:ml_dataframe/ml_dataframe.dart';

class ClassifierAssessor implements ModelAssessor<Classifier> {
  ClassifierAssessor(
      this._metricFactory,
      this._encoderFactory,
      this._featuresTargetSplit,
      this._normalizeClassLabels,
  );

  static const List<MetricType> _allowedMetricTypes = [
    MetricType.precision,
    MetricType.accuracy,
  ];

  final MetricFactory _metricFactory;
  final EncoderFactory _encoderFactory;
  final FeaturesTargetSplit _featuresTargetSplit;
  final NormalizeClassLabels _normalizeClassLabels;

  @override
  double assess(
      Classifier classifier,
      MetricType metricType,
      DataFrame samples,
  ) {

    if (!_allowedMetricTypes.contains(metricType)) {
      throw InvalidMetricTypeException(
          metricType, _allowedMetricTypes);
    }

    final splits = _featuresTargetSplit(
      samples,
      targetNames: classifier.targetNames,
    ).toList();
    final featuresFrame = splits[0];
    final originalLabelsFrame = splits[1];
    final metric = _metricFactory
        .createByType(metricType);
    final labelEncoder = _encoderFactory(
        originalLabelsFrame,
        originalLabelsFrame.header
    );
    final isTargetEncoded = classifier.targetNames.length > 1;
    final predictedLabels = !isTargetEncoded
        ? labelEncoder
            .process(classifier.predict(featuresFrame))
            .toMatrix(classifier.dtype)
        : classifier
            .predict(featuresFrame)
            .toMatrix(classifier.dtype);
    final originalLabels = !isTargetEncoded
        ? labelEncoder
            .process(originalLabelsFrame)
            .toMatrix(classifier.dtype)
        : originalLabelsFrame
            .toMatrix(classifier.dtype);
    final predefinedClassLabelsExist = classifier.negativeLabel != null
        && classifier.positiveLabel != null;
    final normalizedPredictedLabels = predefinedClassLabelsExist
        ? _normalizeClassLabels(
            predictedLabels,
            classifier.positiveLabel,
            classifier.negativeLabel,
          )
        : predictedLabels;
    final normalizedOriginalLabels = predefinedClassLabelsExist
        ? _normalizeClassLabels(
            originalLabels,
            classifier.positiveLabel,
            classifier.negativeLabel,
          )
        : predictedLabels;

    return metric.getScore(
      normalizedPredictedLabels,
      normalizedOriginalLabels,
    );
  }
}