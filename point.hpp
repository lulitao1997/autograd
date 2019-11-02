#pragma once
#include <iostream>

#include "autograd.hpp"

struct Point3D {
  double x, y, z;
  // broadcast add
  Point3D operator+(double r) { return (Point3D){x + r, y + r, z + r}; }
  Point3D operator+(const Point3D& r) { return (Point3D){x + r.x, y + r.y, z + r.z}; }

  // broadcast mul
  Point3D operator*(double r) { return (Point3D){x * r, y * r, z * r}; }
  Point3D operator*(const Point3D& r) { return (Point3D){x * r.x, y * r.y, z * r.z}; }

  friend std::ostream& operator<<(std::ostream& out, const Point3D& p) {
    return out << '(' << p.x << ',' << p.y << ',' << p.x << ')';
  }
};

namespace autograd {
template <class X, class Y, class Z>
struct MakePoint {
  template <class... Args>
  static Point3D value(const Args&... args) {
    return (Point3D){X::value(args...), Y::value(args...), Z::value(args...)};
  }
  // FIXME: this only holds if d is scaler
  template <class D>
  using DeriveType = MakePoint<DerivativeOf<X, D>, DerivativeOf<Y, D>, DerivativeOf<Z, D>>;
};

template <>
struct ZeroOf<Point3D> : MakePoint<Const<0>, Const<0>, Const<0>> {};
template <>
struct UnitOf<Point3D> : MakePoint<Const<1>, Const<1>, Const<1>> {};
}  // namespace autograd
