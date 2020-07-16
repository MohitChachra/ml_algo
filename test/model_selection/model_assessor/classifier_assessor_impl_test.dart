import 'package:ml_algo/src/common/exception/invalid_metric_type_exception.dart';
import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_algo/src/model_selection/model_assessor/classifier_assessor_impl.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import '../../mocks.dart';

void main() {
  group('ClassifierAssessorImpl', () {
    final metricFactoryMock = MetricFactoryMock();
    final metricMock = MetricMock();
    final encoderFactoryMock = EncoderFactoryMock();
    final encoderMock = EncoderMock();
    final featureTargetSplitter = FeatureTargetSplitterMock();
    final assessor = ClassifierAssessorImpl(
        metricFactoryMock,
        encoderFactoryMock.create,
        featureTargetSplitter.split,
    );
    final metricType = MetricType.precision;
    final classifier = ClassifierMock();
    final featuresNames = ['feature_1', 'feature_2', 'feature_3'];
    final targetNames = ['target_1', 'target_2', 'target_2'];
    final samplesHeader = [...featuresNames, ...targetNames];
    final samples = DataFrame([
      <num>[     1,  33,   -199, 1, 0, 0],
      <num>[-90002, 232, 889.20, 1, 0, 0],
      <num>[-12004,  19,    111, 0, 1, 0],
    ], headerExists: false, header: samplesHeader);
    final featuresMock = DataFrame([
      <num>[     1,  33,   -199],
      <num>[-90002, 232, 889.20],
      <num>[-12004,  19,    111],
    ], headerExists: false, header: featuresNames);
    final targetMock = DataFrame([
      <num>[1, 0, 0],
      <num>[1, 0, 0],
      <num>[0, 1, 0],
    ], headerExists: false, header: targetNames);
    final predictionMock = DataFrame([
      <num>[0, 0, 1],
      <num>[0, 0, 1],
      <num>[1, 0, 0],
    ], headerExists: false, header: targetNames);

    setUp(() {
      when(
          encoderFactoryMock.create(
            argThat(anything),
            argThat(anything),
          )
      ).thenReturn(encoderMock);
      when(
          classifier.targetNames,
      ).thenReturn(targetNames);
      when(
        classifier.dtype,
      ).thenReturn(DType.float64);
      when(
          featureTargetSplitter.split(
            argThat(anything),
            targetNames: anyNamed('targetNames'),
          )
      ).thenReturn([featuresMock, targetMock]);
      when(
          classifier.predict(
            argThat(anything),
          ),
      ).thenReturn(predictionMock);
      when(
          encoderMock.process(
            argThat(anything),
          ),
      ).thenReturn(predictionMock);
      when(
          metricFactoryMock.createByType(
            argThat(anything),
          ),
      ).thenReturn(metricMock);
    });

    tearDown(() {
      reset(metricFactoryMock);
      reset(metricMock);
      reset(encoderFactoryMock);
      reset(encoderMock);
      reset(featureTargetSplitter);
    });

    test('should throw an exception if improper metric type is provided', () {
      final metricTypes = [MetricType.mape, MetricType.rmse];

      metricTypes.forEach((metricType) {
        final actual = () => assessor.assess(classifier, metricType, samples);

        expect(actual, throwsA(isA<InvalidMetricTypeException>()));
      });
    });

    test('should create metric entity', () {
      assessor.assess(classifier, metricType, samples);

      verify(metricFactoryMock.createByType(metricType)).called(1);
    });

    test('should encode predicted target column if it is not encoded', () {
      when(classifier.targetNames).thenReturn(['target']);

      assessor.assess(classifier, metricType, samples);

      verify(encoderMock.process(predictionMock)).called(1);
    });

    test('should encode original target column if it is not encoded', () {
      when(classifier.targetNames).thenReturn(['target']);

      assessor.assess(classifier, metricType, samples);

      verify(encoderMock.process(targetMock)).called(1);
    });
  });
}
