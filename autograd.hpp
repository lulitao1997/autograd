#pragma once
#include <tuple>
#include <type_traits>

namespace autograd {

// FIXME: C++ only supports int template parameter
template <int V>
struct Const {
  template <class... Args>
  static double value(const Args &... args) {
    return V;
  }
  template <class>
  using DeriveType = Const<0>;
};

template <class ValueType>
struct ZeroOf;
template <class ValueType>
struct UnitOf;

template <>
struct ZeroOf<double> : Const<0> {};
template <>
struct UnitOf<double> : Const<1> {};

template <int Id, class ValueType>
struct Var;

template <int Id, class ValueType>
struct Var {
  static const int id = Id;
  template <class... Args>
  static ValueType value(const Args &... args) {
    return std::get<Id>(std::tuple<Args...>(args...));
  }

  // take derivative of variable of X
  template <class X>
  using DeriveType = std::conditional_t<X::id == Id, UnitOf<ValueType>, ZeroOf<ValueType>>;
};

template <class V, class X>
using DerivativeOf = typename V::template DeriveType<X>;

template <class V, class X, class... Args>
auto derivative_of_v(const Args &... args) {
  static_assert(std::is_arithmetic_v<decltype(X::value(args...))>,
                "only support take derivative of scaler");
  return DerivativeOf<V, X>::value(args...);
}

// p + q
template <class P, class Q>
struct Add {
  template <class... Args>
  static auto value(const Args &... args) {
    return P::value(args...) + Q::value(args...);
  }
  // p' + q'
  template <class X>
  using DeriveType = Add<DerivativeOf<P, X>, DerivativeOf<Q, X>>;
};

// p * q
template <class P, class Q>
struct Mul {
  template <class... Args>
  static auto value(const Args &... args) {
    return P::value(args...) * Q::value(args...);
  }
  // p * q' + p' * q
  template <class X>
  using DeriveType = Add<Mul<P, DerivativeOf<Q, X>>, Mul<DerivativeOf<P, X>, Q>>;
};

// -p
template <class P>
using Neg = Mul<Const<-1>, P>;

// p - q
template <class P, class Q>
using Minus = Add<P, Neg<Q>>;

// 1 / p
template <class P>
struct Inv {
  template <class... Args>
  static auto value(const Args &... args) {
    return 1.0 / P::value(args...);
  }
  // - 1 / (p * p) * p'
  template <class X>
  using DeriveType = Mul<Neg<Inv<Mul<P, P>>>, DerivativeOf<P, X>>;
};

// p / q
template <class P, class Q>
using Div = Mul<P, Inv<Q>>;

}  // namespace autograd
