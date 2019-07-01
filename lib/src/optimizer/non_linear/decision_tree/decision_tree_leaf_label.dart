import 'package:ml_linalg/vector.dart';

class DecisionTreeLeafLabel {
  DecisionTreeLeafLabel.nominal(this.nominalValue, {this.probability})
      : numericalValue = null;

  DecisionTreeLeafLabel.numerical(this.numericalValue, {this.probability})
      : nominalValue = null;

  final Vector nominalValue;
  final double numericalValue;
  final double probability;
}
