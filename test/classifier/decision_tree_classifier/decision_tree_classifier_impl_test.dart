import 'package:injector/injector.dart';
import 'package:ml_algo/ml_algo.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier_impl.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_json_keys.dart';
import 'package:ml_algo/src/di/dependencies.dart';
import 'package:ml_algo/src/di/dependency_keys.dart';
import 'package:ml_algo/src/di/injector.dart';
import 'package:ml_algo/src/metric/metric_factory.dart';
import 'package:ml_algo/src/tree_trainer/leaf_label/leaf_label.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/tree_node.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/tree_node_json_keys.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/dtype_to_json.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:ml_tech/unit_testing/matchers/iterable_2d_almost_equal_to.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import '../../mocks.dart';

void main() {
  group('DecisionTreeClassifierImpl', () {
    final sample1 = Vector.fromList([1, 2, 3]);
    final sample2 = Vector.fromList([10, 20, 30]);
    final sample3 = Vector.fromList([100, 200, 300]);
    final label1 = 100;
    final label2 = 300;
    final label3 = 200;
    final sample1WithLabel = Vector.fromList([...sample1, label1]);
    final sample2WithLabel = Vector.fromList([...sample2, label2]);
    final sample3WithLabel = Vector.fromList([...sample3, label3]);
    final predictedBinarizedLabels = [
      [0, 0, 1],
      [1, 0, 0],
      [0, 1, 0],
    ];
    final originalBinarizedLabels = [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ];
    final learnedLeafLabel1 = TreeLeafLabel(label3, probability: 0.7);
    final learnedLeafLabel2 = TreeLeafLabel(label1, probability: 0.55);
    final learnedLeafLabel3 = TreeLeafLabel(label2, probability: 0.5);
    final features = Matrix.fromRows([
      sample1,
      sample2,
      sample3,
    ]);
    final labelledFeatures = Matrix.fromRows([
      sample1WithLabel,
      sample2WithLabel,
      sample3WithLabel,
    ]);
    final unlabelledFeaturesFrame = DataFrame.fromMatrix(features);
    final labelledFeaturesFrame = DataFrame.fromMatrix(labelledFeatures);
    final targetColumnName = labelledFeaturesFrame.header.last;
    final predictedLabelsFrame = DataFrame([
      <dynamic>[label3],
      <dynamic>[label1],
      <dynamic>[label2],
    ], headerExists: false, header: [targetColumnName]);
    final originalLabelsFrame = DataFrame([
      <dynamic>[label1],
      <dynamic>[label2],
      <dynamic>[label3],
    ], headerExists: false, header: [targetColumnName]);
    final predictedBinarizedLabelsFrame = DataFrame(predictedBinarizedLabels,
        headerExists: false, header: [targetColumnName]);
    final originalBinarizedLabelsFrame = DataFrame(originalBinarizedLabels,
        headerExists: false, header: [targetColumnName]);
    final rootNodeJson = {
      childrenJsonKey: <Map<String, dynamic>>[],
    };
    final classifier32Json = {
      dTypeJsonKey: dTypeToJson(DType.float32),
      targetColumnNameJsonKey: targetColumnName,
      treeRootNodeJsonKey: rootNodeJson,
    };
    final classifier64Json = {
      dTypeJsonKey: dTypeToJson(DType.float64),
      targetColumnNameJsonKey: targetColumnName,
      treeRootNodeJsonKey: rootNodeJson,
    };
    final treeRootMock = createRootNodeMock({
      sample1: learnedLeafLabel1,
      sample2: learnedLeafLabel2,
      sample3: learnedLeafLabel3,
    }, rootNodeJson);
    final metricFactoryMock = MetricFactoryMock();
    final metricMock = MetricMock();
    final encoderFactoryMock = EncoderFactoryMock();
    final encoderMock = EncoderMock();
    final encodedLabelsFrames = [
      predictedBinarizedLabelsFrame,
      originalBinarizedLabelsFrame,
    ];
    var encoderCallIteration = 0;

    DecisionTreeClassifierImpl classifier32;
    DecisionTreeClassifierImpl classifier64;

    setUp(() {
      injector = Injector();

      when(metricFactoryMock.createByType(argThat(isA<MetricType>())))
          .thenReturn(metricMock);
      when(encoderFactoryMock.create(any, any)).thenReturn(encoderMock);
      when(encoderMock.process(any)).thenAnswer(
              (_) => encodedLabelsFrames[encoderCallIteration++]);

      injector
          ..registerDependency<EncoderFactory>(
                  (_) => encoderFactoryMock.create,
              dependencyName: oneHotEncoderFactoryKey)
          ..registerSingleton<MetricFactory>((_) => metricFactoryMock);

      classifier32 = DecisionTreeClassifierImpl(
        treeRootMock,
        targetColumnName,
        DType.float32,
      );

      classifier64 = DecisionTreeClassifierImpl(
        treeRootMock,
        targetColumnName,
        DType.float64,
      );
    });

    tearDown(() {
      reset(metricFactoryMock);
      reset(metricMock);
      reset(encoderFactoryMock);
      reset(encoderMock);
      injector.clearAll();
      encoderCallIteration = 0;
    });

    test('should predict labels for passed unlabelled features dataframe', () {
      final actual = classifier32.predict(unlabelledFeaturesFrame);

      expect(actual.toMatrix(), predictedLabelsFrame.toMatrix());
    });

    test('should return predicted labels with a proper header', () {
      final actual = classifier32.predict(unlabelledFeaturesFrame);

      expect(actual.header, classifier32.targetNames);
    });

    test('should return data frame with empty header if input matrix is '
        'empty', () {
      final predictedClasses = classifier32.predict(DataFrame([<num>[]]));

      expect(predictedClasses.header, isEmpty);
    });

    test('should return data frame with empty matrix if input feature matrix is '
        'empty', () {
      final predictedClasses = classifier32.predict(DataFrame([<num>[]]));

      expect(predictedClasses.toMatrix(), isEmpty);
    });

    test('should return data frame with probabilities for each class label', () {
      final predictedLabels = classifier32.predictProbabilities(unlabelledFeaturesFrame);

      expect(
          predictedLabels.toMatrix(),
          iterable2dAlmostEqualTo([
            [learnedLeafLabel1.probability.toDouble()],
            [learnedLeafLabel2.probability.toDouble()],
            [learnedLeafLabel3.probability.toDouble()],
          ]),
      );
    });

    test('should serialize (dtype is float32)', () {
      final data = classifier32.toJson();
      expect(data, equals(classifier32Json));
      verify(treeRootMock.toJson()).called(1);
    });

    test('should serialize (dtype is float64)', () {
      final data = classifier64.toJson();
      expect(data, equals(classifier64Json));
      verify(treeRootMock.toJson()).called(1);
    });

    test('should restore dtype field from json (dtype is float32)', () {
      final classifier = DecisionTreeClassifierImpl.fromJson(classifier32Json);
      expect(classifier.dtype, equals(DType.float32));
    });

    test('should restore dtype field from json (dtype is float64)', () {
      final classifier = DecisionTreeClassifierImpl.fromJson(classifier64Json);
      expect(classifier.dtype, equals(DType.float64));
    });

    test('should be restored from json', () {
      final classifier = DecisionTreeClassifierImpl.fromJson(classifier32Json);
      expect(classifier.targetColumnName, equals(targetColumnName));
      expect(classifier.treeRootNode, isNotNull);
    });

    test('should call metric factory while assessing a model, '
        'dtype=DType.float32', () {
      final metricType = MetricType.precision;

      classifier32.assess(labelledFeaturesFrame, metricType);
      verify(metricFactoryMock.createByType(metricType)).called(1);
    });

    test('should call encoder factory while assessing a model, '
        'dtype=DType.float32', () {
      final metricType = MetricType.precision;

      classifier32.assess(labelledFeaturesFrame, metricType);

      final verificationResult = verify(
          encoderFactoryMock.create(captureAny, [targetColumnName]))
        ..called(1);
      final actualFrame = verificationResult.captured[0] as DataFrame;

      expect(
        actualFrame.toMatrix(),
        originalLabelsFrame.toMatrix(),
      );
    });

    test('should encode predicted labels while assessing a model, '
        'dtype=DType.float32', () {
      final metricType = MetricType.precision;

      classifier32.assess(labelledFeaturesFrame, metricType);

      final verificationResult = verify(encoderMock.process(captureAny));
      final actualProcessingFrame = verificationResult.captured[0] as DataFrame;

      expect(
        actualProcessingFrame.toMatrix(),
        predictedLabelsFrame.toMatrix(),
      );
    });

    test('should preserve dataframe header while encoding predicted labels '
        'during a model assessment, dtype=DType.float32', () {
      final metricType = MetricType.precision;

      classifier32.assess(labelledFeaturesFrame, metricType);

      final verificationResult = verify(encoderMock.process(captureAny));
      final actualProcessingFrame = verificationResult.captured[0] as DataFrame;

      expect(actualProcessingFrame.header, classifier32.targetNames);
    });

    test('should encode original labels while assessing a model, '
        'dtype=DType.float32', () {
      final metricType = MetricType.precision;

      classifier32.assess(labelledFeaturesFrame, metricType);

      final verificationResult = verify(encoderMock.process(captureAny));
      final actualProcessingFrame = verificationResult.captured[1] as DataFrame;

      expect(
        actualProcessingFrame.toMatrix(),
        originalLabelsFrame.toMatrix(),
      );
    });

    test('should preserve dataframe header while encoding original labels '
        'during a model assessment, dtype=DType.float32', () {
      final metricType = MetricType.precision;

      classifier32.assess(labelledFeaturesFrame, metricType);

      final verificationResult = verify(encoderMock.process(captureAny));
      final actualProcessingFrame = verificationResult.captured[1] as DataFrame;

      expect(actualProcessingFrame.header, classifier32.targetNames);
    });

    test('should calculate metric, dtype=DType.float32', () {
      final metricType = MetricType.precision;

      classifier32.assess(labelledFeaturesFrame, metricType);
      verify(metricMock.getScore(
        argThat(equals(predictedBinarizedLabels)),
        argThat(equals(originalBinarizedLabels)),
      )).called(1);
    });

    test('should return performance score, dtype=DType.float32', () {
      final score = 0.75;

      when(metricMock.getScore(any, any)).thenReturn(score);

      final metricType = MetricType.precision;
      final actualScore = classifier32.assess(labelledFeaturesFrame, metricType);

      expect(actualScore, score);
    });
  });
}

TreeNode createRootNodeMock(Map<Vector, TreeLeafLabel> samplesByLabel,
    [Map<String, dynamic> jsonMock = const <String, dynamic>{}]) {

  final rootMock = TreeNodeMock();
  final children = <TreeNode>[];

  when(rootMock.isLeaf).thenReturn(false);

  samplesByLabel.forEach((sample, leafLabel) {
    final node = TreeNodeMock();

    when(node.label).thenReturn(leafLabel);
    when(node.isLeaf).thenReturn(true);

    samplesByLabel
        .forEach((otherSample, _) =>
          when(node.isSamplePassed(otherSample))
              .thenReturn(sample == otherSample));

    children.add(node);
  });
  
  when(rootMock.children)
      .thenReturn(children);
  when(rootMock.toJson())
      .thenReturn(jsonMock);

  return rootMock;
}
