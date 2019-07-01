import 'dart:collection';

import 'package:ml_algo/src/common/sequence_elements_distribution_calculator/distribution_calculator.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/leaf_label_factory/majority_leaf_label_factory.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';
import 'package:xrange/zrange.dart';

import '../../../../test_utils/mocks.dart';

void main() {
  group('MajorityLeafLabelFactory', () {
    test('should create decision tree leaf label in case when the class '
        'labels are numbers - label should be a majority class label among the '
        'whole observation matrix', () {

      final observations = Matrix.fromList([
        [10, 20, 30, 0],
      ]);

      final outcomesColumnRange = ZRange.singleton(3);

      final firstClassLabel = 0.0;
      final firstClassProbability = 0.7;

      final secondClassLabel = 2.0;
      final secondClassProbability = 0.1;

      final thirdClassLabel = 3.0;
      final thirdClassProbability = 0.2;

      final distribution = HashMap<double, double>.from(<double, double>{
        firstClassLabel: firstClassProbability,
        secondClassLabel: secondClassProbability,
        thirdClassLabel: thirdClassProbability,
      });
      final distributionCalculator = createDistributionCalculator<double>(
          observations.getColumn(3),
          distribution,
      );
      final labelFactory = MajorityLeafLabelFactory(distributionCalculator);
      final label = labelFactory.create(observations, outcomesColumnRange,
          false);

      expect(label.numericalValue, equals(0));
      expect(label.nominalValue, isNull);
      expect(label.probability, equals(firstClassProbability));
    });

    test('should create decision tree leaf label in case when the class '
        'labels are numbers and the observations contain only one class '
        'label', () {
      final observations = Matrix.fromList([
        [10, 20, 30, 3],
        [17, 10, 32, 3],
        [70, 80, 90, 3],
      ]);
      final outcomesColumnRange = ZRange.singleton(3);

      final classLabel = 3.0;
      final classProbability = 1.0;

      final distribution = HashMap<double, double>.from(<double, double>{
        classLabel: classProbability,
      });

      final distributionCalculator = createDistributionCalculator<double>(
        observations.getColumn(3),
        distribution,
      );
      final labelFactory = MajorityLeafLabelFactory(distributionCalculator);
      final label = labelFactory.create(observations, outcomesColumnRange,
          false);

      expect(label.numericalValue, equals(classLabel));
      expect(label.nominalValue, isNull);
      expect(label.probability, equals(classProbability));
    });

    test('should create decision tree leaf label in case when the class '
        'labels are vectors (encoded categories) - label should be a majority '
        'class label among the whole observation matrix', () {

      final observations = Matrix.fromList([
        [10, 20, 30, 0, 0, 1],
      ]);

      final outcomesColumnRange = ZRange.closed(3, 5);

      final firstClassLabel = Vector.fromList([0, 0, 1]);
      final firstClassProbability = 0.09;

      final secondClassLabel = Vector.fromList([1, 0, 0]);
      final secondClassProbability = 0.6;

      final thirdClassLabel = Vector.fromList([0, 1, 0]);
      final thirdClassProbability = 0.31;

      final distribution = HashMap<Vector, double>.from(<Vector, double>{
        firstClassLabel: firstClassProbability,
        secondClassLabel: secondClassProbability,
        thirdClassLabel: thirdClassProbability,
      });
      final distributionCalculator = createDistributionCalculator<Vector>(
        [Vector.fromList([0, 0, 1])],
        distribution,
      );
      final labelFactory = MajorityLeafLabelFactory(distributionCalculator);
      final label = labelFactory.create(observations, outcomesColumnRange,
          true);

      expect(label.nominalValue, equals(secondClassLabel));
      expect(label.numericalValue, isNull);
      expect(label.probability, equals(secondClassProbability));
    });

    test('should create decision tree leaf label in case when the class '
        'labels are vectors and the observations contain only one class '
        'label', () {
      final observations = Matrix.fromList([
        [10, 20, 30, 0, 1, 0],
      ]);
      final outcomesColumnRange = ZRange.closed(3, 5);

      final classLabel = Vector.fromList([0, 0, 1]);
      final classProbability = 1.0;

      final distribution = HashMap<Vector, double>.from(<Vector, double>{
        classLabel: classProbability,
      });

      final distributionCalculator = createDistributionCalculator<Vector>(
        [Vector.fromList([0, 1, 0])],
        distribution,
      );
      final labelFactory = MajorityLeafLabelFactory(distributionCalculator);
      final label = labelFactory.create(observations, outcomesColumnRange,
          true);

      expect(label.nominalValue, equals(classLabel));
      expect(label.numericalValue, isNull);
      expect(label.probability, classProbability);
    });
  });
}

SequenceElementsDistributionCalculator createDistributionCalculator<T>(
    Iterable<T> values, HashMap<T, double> distribution) {
  final distributionCalculator = DistributionCalculatorMock();

  when(distributionCalculator.calculate<T>(argThat(equals(values)), any))
      .thenReturn(distribution);

  return distributionCalculator;
}