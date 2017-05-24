import 'package:dart_ml/src/math/vector/norm.dart' show Norm;

abstract class Vector {
  int get length;
  void set length(int value);

  double operator [] (int index);
  void operator []= (int index, double value);

  Vector operator + (Vector vector);
  Vector operator - (Vector vector);
  Vector operator * (Vector vector);
  Vector operator / (Vector vector);

  Vector intPow(int exponent, {bool inPlace = false});

  Vector scalarMul(double value, {bool inPlace = false});
  Vector scalarDiv(double value, {bool inPlace = false});
  Vector scalarAdd(double value, {bool inPlace = false});
  Vector scalarSub(double value, {bool inPlace = false});

  Vector abs({bool inPlace = false});

  double norm([Norm normType = Norm.EUCLIDEAN]);
  double distanceTo(Vector vector, [Norm normType = Norm.EUCLIDEAN]);

  double dot(Vector vector);

  double mean();
  double sum();

  Vector cut(int start, [int end]);
  Vector copy();
  Vector createFrom(Iterable<double> iterable);

  void fill(double value);
  void concat(Vector vector);
}