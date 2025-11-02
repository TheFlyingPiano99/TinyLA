
<div align="center">
<img src="media/TinyLA_logo.png" alt="TinyLA Logo" width="300">
</div>

A header-only C++ template library for linear algebra operations with expression template based automatic differentiation, with emphasis on ease-of-use.


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

// Create scalar variables with character IDs for automatic differentiation
auto x = TinyLA::dscal<'x'>{5.0};   // Variable with ID 'x'
auto y = TinyLA::dscal<'y'>{3.0};   // Variable with ID 'y'
auto constant = TinyLA::dscal{2.0}; // Constant (no variable ID)

// Expression templates allow lazy evaluation and automatic differentiation
auto expr = (x + y) * constant - x / y;

std::cout << "Expression: " << expr.to_string() << std::endl;
std::cout << "Value: " << expr.eval() << std::endl;
```

### Vector Operations

```cpp
// Create 3D vectors
auto v1 = TinyLA::dvec3<'u'>{{1.0}, {2.0}, {3.0}};  // Variable vector with ID 'u'
auto v2 = TinyLA::dvec3{{4.0}, {5.0}, {6.0}};       // Constant vector

// Vector arithmetic with expression templates
auto sum = v1 + v2;           // Lazy addition expression
auto scaled = v1 * 2.0;       // Scalar multiplication
auto cross_prod = cross(v1, v2);  // Cross product
auto dot_prod = dot(v1, v2);      // Dot product
```

### Matrix Operations

```cpp
// Create matrices
auto matA = TinyLA::dmat2{{1.0, 2.0}, {3.0, 4.0}};
auto matB = TinyLA::dmat2{{5.0, 6.0}, {7.0, 8.0}};
matC = TinyLA::dmat2{{9.0, 10.0}, {11.0, 12.0}};
auto vec = TinyLA::dvec2{1.0, 2.0};

// Matrix operations
auto matSum = matA + matB;          // Matrix addition
auto matProd = matA * matB;         // Matrix multiplication
auto elemProd = elementwiseProduct(matA, matB); // Elementwise multiplication
auto transposed = transpose(matA);  // Transpose
auto matVecProd = matA * vec;        // Matrix-vector multiplication        
```

### Automatic Differentiation with Matrices

```cpp
// Create variable matrices for automatic differentiation
auto A = TinyLA::dmat2<'A'>{{2.0, 1.0}, {1.0, 3.0}};
auto x = TinyLA::dvec2<'x'>{{5.0}, {2.0}};

// Write an expression
auto expr = transpose(A) * A * x + x;

// Compute gradients using character-based variable IDs
auto dA = expr.derivate<'A'>();  // Derivative with respect to matrix A
auto dx = expr.derivate<'x'>();  // Derivative with respect to vector x

std::cout << "d expr/dA at (0,0): " << dA.eval(0, 0) << std::endl;
std::cout << "d expr/dx at (0,0): " << dx.eval(0, 0) << std::endl;
```

### Complex Numbers

```cpp
// Complex matrix operations with character-based variable IDs
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
auto double_vector = dvec3<'D'>{{1.0}, {2.0}, {3.0}};
auto complex_scalar = cscal<'C'>{std::complex<double>(1.0, 0.5)};

// All work together in expressions with automatic differentiation
auto mixed_expr = float_matrix * double_vector.eval(0, 0) + complex_scalar;

// Differentiate with respect to each variable
auto d_mixed_dF = mixed_expr.derivate<'F'>();
auto d_mixed_dD = mixed_expr.derivate<'D'>();
auto d_mixed_dC = mixed_expr.derivate<'C'>();
```

### Mathematical Constants and Special Matrices

```cpp
// Mathematical constants as expression templates
auto pi_expr = TinyLA::Pi<double>;     // π constant
auto e_expr = TinyLA::Euler<double>;   // Euler's number

// Special matrices
auto identity3 = TinyLA::Identity<double, 3>{};
auto zero23 = TinyLA::Zero<double, 2, 3>{}; // A matrix filled with 0
auto ones22 = TinyLA::Ones<double, 2, 2>{}; // A matrix filled with 1
```


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2025 Zoltán Simon

The MIT License is a permissive license that allows you to use, modify, and distribute this software freely, including for commercial purposes, as long as you include the original copyright notice and license text.

