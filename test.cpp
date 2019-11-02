#include <iostream>

#include "autograd.hpp"
#include "point.hpp"
using namespace std;

int main() {
  using namespace autograd;
  using X = Var<0, double>;
  using Y = Var<1, double>;
  using Z = Var<2, double>;
#define values 3, 2, 1, (Point3D){-1, -2, -3}, Point3D{5, -9, 7}
  cout << "d(100)/dx: " << derivative_of_v<Const<100>, X>(values) << endl;
  cout << "x, dx/dx, dx/dy: " << X::value(values) << ", " << derivative_of_v<X, X>(values) << ", "
       << derivative_of_v<X, Y>(values) << endl;
  using X_add_Y = Add<X, Y>;
  cout << "x+y, d(x+y)/dx, d(x+y)/dy: " << X_add_Y::value(values) << ", "
       << derivative_of_v<X_add_Y, X>(values) << ", " << derivative_of_v<X_add_Y, Y>(values)
       << endl;
  using X_mul_Y = Mul<X, Y>;
  cout << "x*y, d(x*y)/dx, d(x*y)/dy: " << X_mul_Y::value(values) << ", "
       << derivative_of_v<X_mul_Y, X>(values) << ", " << derivative_of_v<X_mul_Y, Y>(values)
       << endl;

  using X_div_Y = Div<X, Y>;
  cout << "x/y, d(x/y)/dx, d(x/y)/dy: " << X_div_Y::value(values) << ", "
       << derivative_of_v<X_div_Y, X>(values) << ", " << derivative_of_v<X_div_Y, Y>(values)
       << endl;
  using Pt1 = Var<3, Point3D>;
  using Pt2 = Var<4, Point3D>;
  using Pt1_add_X = Add<Pt1, X>;
  using Pt1_mul_X = Mul<Pt1, X>;
  using Pt1_mul_Pt2 = Mul<Pt1, Pt2>;
  cout << "pt1+x, d(pt1+x)/dx: " << Pt1_add_X::value(values) << ", "
       << derivative_of_v<Pt1_add_X, X>(values) << endl;

  // Second Derivatives
  // using Poly = Mul<Add<Mul<X_add_Y, X>, Mul<Const<5>, X>>>;
  using Poly = Mul<Add<Mul<X_add_Y, X>, Mul<Const<5>, X>>, X>;  // ((x+y)*x + 5x)*x
  cout << "p(x), dp/dx, d^2p/dx^2, d(dp/dx)/dy: " << Poly::value(values) << ", "
       << derivative_of_v<Poly, X>(values) << ", "
       << derivative_of_v<DerivativeOf<Poly, X>, X>(values) << ", "
       << derivative_of_v<DerivativeOf<Poly, X>, Y>(values) << endl;

  // not supported.
  // cout << "d(pt1+x)/d(pt1): " << derivative_of_v<Pt1_add_X, Pt1>(values) << endl;
  // cout << "d(pt1*pt2)/d(pt1): " << derivative_of_v<Pt1_mul_Pt2, Pt1>(values) << endl;
}
