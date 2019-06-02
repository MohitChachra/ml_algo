import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:xrange/zrange.dart';

class DecisionTreeNode {
  DecisionTreeNode(this.children);

  final Iterable<DecisionTreeNode> children;
}

class DecisionTreeOptimizer {
  DecisionTreeOptimizer(Matrix features, Matrix outcomes,
      [this._maxNodesCount, this._featuresRanges]) {
    _root = _createNode(features, outcomes, 0);
  }

  final int _maxNodesCount;
  final Iterable<ZRange> _featuresRanges;
  DecisionTreeNode _root;

  /// Builds a tree, where each node is a logical rule, that divides given data
  /// into several parts
  DecisionTreeNode _createNode(Matrix observations, int nodesCount) {
    if (_isLeaf(observations, nodesCount)) {
      return DecisionTreeNode([]);
    }
    final range = _findSplittingFeatureRange(observations);
    final children = _learnStump(observations, range)
        .map((selected) => _createNode(selected, nodesCount + 1));
    return DecisionTreeNode(children);
  }

  ZRange _findSplittingFeatureRange(Matrix observations) {
    final errors = <double, List<ZRange>>{};
    _featuresRanges.forEach((range) {
      final stump = _learnStump(observations, range);
      final error = _getErrorOnStump(stump);
      errors.putIfAbsent(error, () => []);
      errors[error].add(range);
    });
    final sorted = errors.keys.toList(growable: false)
      ..sort();
    final minError = sorted.first;
    return errors[minError].first;
  }

  Iterable<Matrix> _learnStump(Matrix observations, ZRange target,
      [List<Vector> categoricalValues]) =>
      categoricalValues?.isNotEmpty == true
          ? _getObservationsByCategoricalValues(observations, target,
          categoricalValues)
          : _getObservationsByRealValue(observations, target);

  List<Matrix> _getObservationsByCategoricalValues(Matrix observations,
      ZRange range, List<Vector> splittingValues) =>
    splittingValues.map((value) {
      final foundRows = observations.rows
          .where((row) => row.subvectorByRange(range) == value)
          .toList(growable: false);
      return Matrix.fromRows(foundRows);
    }).toList(growable: false);

  List<Matrix> _getObservationsByRealValue(Matrix observations, ZRange range) {
    final value = 20;
    final rows = observations.rows
        .where((row) => row.subvectorByRange(range) == value)
        .toList(growable: false);
    
  }

  bool _isLeaf(Matrix features, Matrix outcomes, int nodesCount) {
    if (nodesCount >= _maxNodesCount) {
      return true;
    }
    if (outcomes.uniqueRows().rowsNum == 1) {
      return true;
    }
    if (_isGoodQualityReached(outcomes)) {
      return true;
    }
    return false;
  }

  bool _isGoodQualityReached(Matrix outcomes) {

  }

  double _getErrorOnStump(Iterable<Matrix> data) {}
}
