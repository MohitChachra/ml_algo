import 'package:injector/injector.dart';
import 'package:ml_algo/ml_algo.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier_impl.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_json_keys.dart';
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
    final sample1WithLabel = Vector.fromList([...sample1, label3]);
    final sample2WithLabel = Vector.fromList([...sample2, label1]);
    final sample3WithLabel = Vector.fromList([...sample3, label2]);
    final originalBinarizedLabels = [
      [0, 0, 1],
      [1, 0, 0],
      [0, 1, 0],
    ];
    final predictedBinarizedLabels = [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ];
    final leafLabel1 = TreeLeafLabel(0, probability: 0.7);
    final leafLabel2 = TreeLeafLabel(1, probability: 0.55);
    final leafLabel3 = TreeLeafLabel(2, probability: 0.5);
    final targetColumnName = 'class_name';
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
    final samples = DataFrame.fromMatrix(features);
    final labelledSamples = DataFrame.fromMatrix(labelledFeatures);
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
      sample1: leafLabel1,
      sample2: leafLabel2,
      sample3: leafLabel3,
    }, rootNodeJson);
    final metricFactoryMock = MetricFactoryMock();
    final metricMock = MetricMock();

    DecisionTreeClassifierImpl classifier32;
    DecisionTreeClassifierImpl classifier64;

    setUp(() {
      injector = Injector();

      when(metricFactoryMock.createByType(argThat(isA<MetricType>())))
          .thenReturn(metricMock);

      injector
          .registerSingleton<MetricFactory>((_) => metricFactoryMock);

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
      resetMockitoState();
      injector.clearAll();
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
      final predictedLabels = classifier32.predictProbabilities(samples);

      expect(
          predictedLabels.toMatrix(),
          iterable2dAlmostEqualTo([
            [leafLabel1.probability.toDouble()],
            [leafLabel2.probability.toDouble()],
            [leafLabel3.probability.toDouble()],
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

    test('should contain proper allowed metrics', () {
      final classifier = DecisionTreeClassifierImpl.fromJson(classifier32Json);

      expect(classifier.allowedMetrics, [
        MetricType.accuracy,
        MetricType.precision,
      ]);
    });

    test('should call metric factory while assessing a model, '
        'dtype=DType.float32', () {
      final metricType = MetricType.precision;

      classifier32.assess(labelledSamples,
        [labelledSamples.header.last],
        metricType,
      );
      verify(metricFactoryMock.createByType(metricType));
    });

    test('should call metric factory while assessing a model, '
        'dtype=DType.float64', () {
      final metricType = MetricType.precision;

      classifier64.assess(labelledSamples,
        [labelledSamples.header.last],
        metricType,
      );
      verify(metricFactoryMock.createByType(metricType));
    });

    test('should calculate metric, dtype=DType.float32', () {
      final metricType = MetricType.precision;

      classifier32.assess(labelledSamples,
        [labelledSamples.header.last],
        metricType,
      );
      verify(metricMock.getScore(
        argThat(equals(originalBinarizedLabels)),
        argThat(equals(predictedBinarizedLabels)),
      )).called(1);
    });

    test('should calculate metric, dtype=DType.float64', () {
      final metricType = MetricType.precision;

      classifier64.assess(labelledSamples,
        [labelledSamples.header.last],
        metricType,
      );
      verify(metricMock.getScore(
        argThat(equals(originalBinarizedLabels)),
        argThat(equals(predictedBinarizedLabels)),
      )).called(1);
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
