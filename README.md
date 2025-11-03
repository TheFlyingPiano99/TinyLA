
<div align="center">
<img src="media/TinyLA_logo.png" alt="TinyLA Logo" width="300">
</div>

A light-weight header-only C++ library for linear algebra with automatic differentiation, focusing on natural mathematical syntax and ease of use.

## Design Philosophy

TinyLA follows the idea that well-formed mathematical expressions should map directly to well-formed code. Code representing ill-formed expressions should fail to compile, with strict enforcement of dimensional consistency.

All expressions are matrices: column vectors are single-column matrices, and scalars are 1x1 matrices. Practical aliases help avoid verbosity when declaring variables.

The derivative of any expression with respect to any variable is another valid expression, even if the original expression does not depend on that variable (yielding zero).

Expressions remain unevaluated until a numeric result is explicitly required. Keeping analytical forms enables transformations such as differentiation and simplification while maintaining dependence on original variables.

Algebraic expressions are immutable: operations create new expressions. Variables can change value, allowing expressions to act as functions.


## Installation
Using CMake, you can fetch the content of this repository using FetchContent as follows
```cmake
include(FetchContent)
FetchContent_Declare(TinyLA_content
    GIT_REPOSITORY https://github.com/TheFlyingPiano99/TinyLA.git
    GIT_TAG main
    GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(TinyLA_content)

target_link_libraries(${YOUR_TARGET} PRIVATE TinyLA::TinyLA)
```
Alternatively, you can copy the `TinyLA.h` file into your project's own include folder. In this case make sure to define the `ENABLE_CUDA_SUPPORT` flag in your source code if you want to use the library with CUDA!

## Examples

### Basic Scalar Operations

```cpp
#include "TinyLA.h"
#include <iostream>

// Scalar variables
auto x = TinyLA::dscal<'x'>{5.0};   // Variable with ID 'x'
auto y = TinyLA::dscal<'y'>{3.0};   // Variable with ID 'y'
const auto constant = TinyLA::dscal{2.0}; // Constant (no variable ID)

// Define an expression
auto expr = (x + y) * constant - x / y;

// Print the symbolic expression and the value
std::cout << "Expression: " << expr.to_string() << std::endl;
std::cout << "Value: " << expr.eval() << std::endl;
```

### Vector Operations

```cpp
// Create 3D vectors
auto v1 = TinyLA::dvec3<'v1'>{1.0, 2.0, 3.0};  // Variable vector with ID 'u'
auto v2 = TinyLA::dvec3<'v2'>{4.0, 5.0, 6.0};       // Constant vector

// Vector arithmetic
auto sum = v1 + v2;
auto scaled = v1 * 2.0;
auto cross_prod = cross(v1, v2);
auto dot_prod = dot(v1, v2);
```

### Matrix Operations

```cpp
// Create matrices
auto matA = TinyLA::dmat2<'A'>{{1.0, 2.0}, {3.0, 4.0}};
auto matB = TinyLA::dmat2<'B'>{{5.0, 6.0}, {7.0, 8.0}};
matC = TinyLA::dmat2<'C'>{{9.0, 10.0}, {11.0, 12.0}};
auto vec = TinyLA::dvec2<'v'>{1.0, 2.0};

// Matrix operations
auto matSum = matA + matB;          
auto matProd = matA * matB;         
auto elemProd = elementwiseProduct(matA, matB);
auto transposed = transpose(matA);
auto matVecProd = matA * vec;
```

### Automatic Differentiation with Matrices

```cpp
// Create variables
auto A = TinyLA::dmat2<'A'>{{2.0, 1.0}, {1.0, 3.0}};
auto x = TinyLA::dvec2<'x'>{{5.0}, {2.0}};

// Write an expression
auto expr = transpose(A) * A * x + x;

// Derivate
auto dA = expr.derivate<'A'>();  // Derivative with respect to matrix A
auto dx = expr.derivate<'x'>();  // Derivative with respect to vector x

std::cout << "d expr/dA at (0,0): " << dA.eval(0, 0) << std::endl;
std::cout << "d expr/dx at (0,0): " << dx.eval(0, 0) << std::endl;
```

### Complex Numbers

```cpp
// Complex-valued matrix
auto cmat = TinyLA::cmat2<'M'>{{std::complex<double>(1.0, 0.5), std::complex<double>(2.0, -1.0)},
                               {std::complex<double>(0.0, 1.0), std::complex<double>(3.0, 0.0)}};

// Complex operations
auto conjugated = conj(cmat);
auto adjoint_matrix = adj(cmat);  // Conjugate transpose
```

### Type Aliases for Convenience

```cpp
// The library provides convenient type aliases:
// Scalars: fscal, dscal, cscal (float, double, complex<double>)
// Vectors: fvec2, fvec3, fvec4, dvec2, dvec3, dvec4, cvec2, cvec3, cvec4
// Matrices: fmat2, fmat3, fmat4, dmat2, dmat3, dmat4, cmat2, cmat3, cmat4

using namespace TinyLA;

// Different data types with character-based variable IDs
auto float_matrix = fmat2<'F'>{{1.0f, 2.0f}, {3.0f, 4.0f}};
auto double_vector = dvec2<'D'>{1.0, 2.0};
auto complex_scalar = cscal<'C'>{std::complex<double>(1.0, 0.5)};

// All work together in expressions
auto mixed_expr = complex_scalar * float_matrix * double_vector;
```

### Mathematical Constants and Special Matrices

```cpp
// Mathematical constants
auto pi = TinyLA::Pi<double>;     // π constant
auto e = TinyLA::Euler<double>;   // Euler's number

// Special matrices
auto identity3 = TinyLA::Identity<double, 3>{};
auto zero23 = TinyLA::Zero<double, 2, 3>{}; // A matrix filled with 0
auto ones22 = TinyLA::Ones<double, 2, 2>{}; // A matrix filled with 1
```


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2025 Zoltán Simon

The MIT License is a permissive license that allows you to use, modify, and distribute this software freely, including for commercial purposes, as long as you include the original copyright notice and license text.

