#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "../include/TinyLA.h"

using namespace tinyla;
using namespace Catch;

TEST_CASE("Basic Matrix Creation and Initialization", "[matrix][creation]") {
    SECTION("Scalar creation") {
        fscal s1{3.14f};
        REQUIRE(s1.eval() == Approx(3.14f));
        
        dscal s2{2.71};
        REQUIRE(s2.eval() == Approx(2.71));
    }
    
    SECTION("Vector creation") {
        fvec3 v1{1.0f, 2.0f, 3.0f};  // Single braces for vectors
        REQUIRE(v1.eval(0, 0) == Approx(1.0f));
        REQUIRE(v1.eval(1, 0) == Approx(2.0f));
        REQUIRE(v1.eval(2, 0) == Approx(3.0f));
        
        dvec2 v2{1.5, 2.5};  // Single braces for vectors
        REQUIRE(v2.eval(0, 0) == Approx(1.5));
        REQUIRE(v2.eval(1, 0) == Approx(2.5));
    }
    
    SECTION("Matrix creation") {
        fmat2 m1{{1.0f, 2.0f}, {3.0f, 4.0f}};
        REQUIRE(m1.eval(0, 0) == Approx(1.0f));
        REQUIRE(m1.eval(0, 1) == Approx(2.0f));
        REQUIRE(m1.eval(1, 0) == Approx(3.0f));
        REQUIRE(m1.eval(1, 1) == Approx(4.0f));
        
        dmat3 m2{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
        REQUIRE(m2.eval(0, 0) == Approx(1.0));
        REQUIRE(m2.eval(1, 1) == Approx(1.0));
        REQUIRE(m2.eval(2, 2) == Approx(1.0));
        REQUIRE(m2.eval(0, 1) == Approx(0.0));
    }
    
    SECTION("Complex matrix creation") {
        cfvec2 cv{std::complex<float>(1.0f, 2.0f), std::complex<float>(3.0f, 4.0f)};  // Single braces for vectors
        auto val1 = cv.eval(0, 0);
        auto val2 = cv.eval(1, 0);
        REQUIRE(val1.real() == Approx(1.0f));
        REQUIRE(val1.imag() == Approx(2.0f));
        REQUIRE(val2.real() == Approx(3.0f));
        REQUIRE(val2.imag() == Approx(4.0f));
    }
}

TEST_CASE("Special Matrices", "[matrix][special]") {
    SECTION("Zero matrices") {
        auto z1 = zero<float>{};
        REQUIRE(z1.eval(0, 0) == Approx(0.0f));
        REQUIRE(z1.eval(1, 1) == Approx(0.0f));
        REQUIRE(z1.to_string() == "0");
        
        auto z_scalar = zero<float>{};
        REQUIRE(z_scalar.eval() == Approx(0.0f));
    }
    
    SECTION("Identity matrices") {
        auto id2 = identity2{};
        REQUIRE(id2.eval(0, 0) == Approx(1.0f));
        REQUIRE(id2.eval(1, 1) == Approx(1.0f));
        REQUIRE(id2.eval(0, 1) == Approx(0.0f));
        REQUIRE(id2.eval(1, 0) == Approx(0.0f));
        
        auto id_unit = unit{};
        REQUIRE(id_unit.eval() == Approx(1.0f));
        REQUIRE(id_unit.to_string() == "1");
    }
    
    SECTION("Ones matrices") {
        auto ones_mat = ones<float, 2, 3>{};
        REQUIRE(ones_mat.eval(0, 0) == Approx(1.0f));
        REQUIRE(ones_mat.eval(1, 2) == Approx(1.0f));
        
        auto one_scalar = one{};
        REQUIRE(one_scalar.eval() == Approx(1.0f));
    }
    
    SECTION("Filled constant matrices") {
        auto filled = FilledConstant<double, 2, 2, 1, 1>{5.5};
        REQUIRE(filled.eval(0, 0) == Approx(5.5));
        REQUIRE(filled.eval(1, 1) == Approx(5.5));
    }
    
    SECTION("Math constants") {
        auto pi_val = pi<float>;
        REQUIRE(pi_val.eval() == Approx(3.14159f).epsilon(0.001f));
        
        auto e_val = euler<double>;
        REQUIRE(e_val.eval() == Approx(2.71828).epsilon(0.001));
    }
}

TEST_CASE("Arithmetic Operations", "[arithmetic]") {
    SECTION("Addition") {
        fvec2 v1{1.0f, 2.0f};  // Single braces for vectors
        fvec2 v2{3.0f, 4.0f};  // Single braces for vectors
        auto result = v1 + v2;
        
        REQUIRE(result.eval(0, 0) == Approx(4.0f));
        REQUIRE(result.eval(1, 0) == Approx(6.0f));
        
        // Scalar addition
        auto scalar_add = v1 + 5.0f;
        REQUIRE(scalar_add.eval(0, 0) == Approx(6.0f));
        REQUIRE(scalar_add.eval(1, 0) == Approx(7.0f));
    }
    
    SECTION("Subtraction") {
        fmat2 m1{{5.0f, 6.0f}, {7.0f, 8.0f}};
        fmat2 m2{{1.0f, 2.0f}, {3.0f, 4.0f}};
        auto result = m1 - m2;
        
        REQUIRE(result.eval(0, 0) == Approx(4.0f));
        REQUIRE(result.eval(0, 1) == Approx(4.0f));
        REQUIRE(result.eval(1, 0) == Approx(4.0f));
        REQUIRE(result.eval(1, 1) == Approx(4.0f));
    }
    
    SECTION("Negation") {
        fvec3 v{1.0f, -2.0f, 3.0f};  // Single braces for vectors
        auto neg = -v;
        
        REQUIRE(neg.eval(0, 0) == Approx(-1.0f));
        REQUIRE(neg.eval(1, 0) == Approx(2.0f));
        REQUIRE(neg.eval(2, 0) == Approx(-3.0f));
    }
    
    SECTION("Element-wise multiplication") {
        fvec2 v1{2.0f, 3.0f};  // Single braces for vectors
        fvec2 v2{4.0f, 5.0f};  // Single braces for vectors
        auto result = elementwiseProduct(v1, v2);
        
        REQUIRE(result.eval(0, 0) == Approx(8.0f));
        REQUIRE(result.eval(1, 0) == Approx(15.0f));
        
        // Scalar multiplication
        auto scalar_mult = v1 * 2.0f;
        REQUIRE(scalar_mult.eval(0, 0) == Approx(4.0f));
        REQUIRE(scalar_mult.eval(1, 0) == Approx(6.0f));
    }
    
    SECTION("Matrix multiplication") {
        fmat2 m1{{1.0f, 2.0f}, {3.0f, 4.0f}};
        fmat2 m2{{5.0f, 6.0f}, {7.0f, 8.0f}};
        auto result = m1 * m2;
        
        REQUIRE(result.eval(0, 0) == Approx(19.0f)); // 1*5 + 2*7
        REQUIRE(result.eval(0, 1) == Approx(22.0f)); // 1*6 + 2*8
        REQUIRE(result.eval(1, 0) == Approx(43.0f)); // 3*5 + 4*7
        REQUIRE(result.eval(1, 1) == Approx(50.0f)); // 3*6 + 4*8
    }
    
    SECTION("Division") {
        fvec2 v1{8.0f, 12.0f};  // Single braces for vectors
        fvec2 v2{2.0f, 3.0f};   // Single braces for vectors
        auto result = v1 / v2;
        
        REQUIRE(result.eval(0, 0) == Approx(4.0f));
        REQUIRE(result.eval(1, 0) == Approx(4.0f));
        
        // Scalar division
        auto scalar_div = v1 / 4.0f;
        REQUIRE(scalar_div.eval(0, 0) == Approx(2.0f));
        REQUIRE(scalar_div.eval(1, 0) == Approx(3.0f));
    }
}

TEST_CASE("Matrix Operations", "[matrix][operations]") {
    SECTION("Transpose") {
        fmat<2, 3> m{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
        auto t = transpose(m);
        
        REQUIRE(t.eval(0, 0) == Approx(1.0f));
        REQUIRE(t.eval(1, 0) == Approx(2.0f));
        REQUIRE(t.eval(2, 0) == Approx(3.0f));
        REQUIRE(t.eval(0, 1) == Approx(4.0f));
        REQUIRE(t.eval(1, 1) == Approx(5.0f));
        REQUIRE(t.eval(2, 1) == Approx(6.0f));
        
        // Alternative syntax
        auto t2 = T(m);
        REQUIRE(t2.eval(0, 0) == Approx(1.0f));
    }
    
    SECTION("Conjugate") {
        cfvec2 cv{std::complex<float>(1.0f, 2.0f), std::complex<float>(3.0f, -4.0f)};
        auto conj_result = conj(cv);
        
        auto val1 = conj_result.eval(0, 0);
        auto val2 = conj_result.eval(1, 0);
        REQUIRE(val1.real() == Approx(1.0f));
        REQUIRE(val1.imag() == Approx(-2.0f));
        REQUIRE(val2.real() == Approx(3.0f));
        REQUIRE(val2.imag() == Approx(4.0f));
    }
    
    SECTION("Adjoint") {
        cfmat2 cm{{std::complex<float>(1.0f, 1.0f), std::complex<float>(2.0f, -1.0f)},
                  {std::complex<float>(3.0f, 2.0f), std::complex<float>(4.0f, 0.0f)}};
        auto adj_result = adjoint(cm);
        
        auto val = adj_result.eval(0, 0);
        REQUIRE(val.real() == Approx(1.0f));
        REQUIRE(val.imag() == Approx(-1.0f));
        
        val = adj_result.eval(1, 0);
        REQUIRE(val.real() == Approx(2.0f));
        REQUIRE(val.imag() == Approx(1.0f));
    }
    
    SECTION("Dot product") {
        // Create row vector and column vector for dot product
        auto row_vec = transpose(fvec3{1.0f, 2.0f, 3.0f});  // Single braces
        fvec3 col_vec{4.0f, 5.0f, 6.0f};  // Single braces
        auto dot_result = dot(row_vec, col_vec);
        
        REQUIRE(dot_result.eval() == Approx(32.0f)); // 1*4 + 2*5 + 3*6
    }
}

TEST_CASE("Mathematical Functions", "[math][functions]") {
    SECTION("Natural logarithm") {
        fscal s{2.71828f};
        auto log_result = log(s);
        REQUIRE(log_result.eval() == Approx(1.0f).epsilon(0.001f));
        
        fvec2 v{1.0f, 2.71828f};  // Single braces for vectors
        auto log_vec = log(v);
        REQUIRE(log_vec.eval(0, 0) == Approx(0.0f).epsilon(0.001f));
        REQUIRE(log_vec.eval(1, 0) == Approx(1.0f).epsilon(0.001f));
    }
    
    SECTION("Natural exponential") {
        fscal s{1.0f};
        auto exp_result = exp(s);
        REQUIRE(exp_result.eval() == Approx(2.71828f).epsilon(0.001f));
        
        fvec2 v{0.0f, 1.0f};  // Single braces for vectors
        auto exp_vec = exp(v);
        REQUIRE(exp_vec.eval(0, 0) == Approx(1.0f).epsilon(0.001f));
        REQUIRE(exp_vec.eval(1, 0) == Approx(2.71828f).epsilon(0.001f));
        
        // Test exp(0) = 1
        fscal zero{0.0f};
        auto exp_zero = exp(zero);
        REQUIRE(exp_zero.eval() == Approx(1.0f));
        
        // Test exp(ln(x)) = x
        fscal x{5.0f};
        auto exp_log_x = exp(log(x));
        REQUIRE(exp_log_x.eval() == Approx(5.0f).epsilon(0.001f));
    }
    
    SECTION("Power function") {
        fvec2 base{2.0f, 3.0f};   // Single braces for vectors
        fvec2 exp{3.0f, 2.0f};    // Single braces for vectors
        auto pow_result = pow(base, exp);
        
        REQUIRE(pow_result.eval(0, 0) == Approx(8.0f));  // 2^3
        REQUIRE(pow_result.eval(1, 0) == Approx(9.0f));  // 3^2
        
        // Scalar exponent
        auto pow_scalar = pow(base, 2.0f);
        REQUIRE(pow_scalar.eval(0, 0) == Approx(4.0f));  // 2^2
        REQUIRE(pow_scalar.eval(1, 0) == Approx(9.0f));  // 3^2
    }
    
    SECTION("Exponential derivative") {
        constexpr VarIDType x_id = U'x';
        fscal_var<x_id> x{2.0f};
        
        // d/dx(exp(x)) = exp(x)
        auto exp_x = exp(x);
        auto d_exp_x = derivate<x_id>(exp_x);
        REQUIRE(d_exp_x.eval() == Approx(exp_x.eval()).epsilon(0.001f));
        
        // d/dx(exp(2x)) = 2*exp(2x)
        auto exp_2x = exp(x * 2.0f);
        auto d_exp_2x = derivate<x_id>(exp_2x);
        REQUIRE(d_exp_2x.eval() == Approx(2.0f * exp_2x.eval()).epsilon(0.001f));
    }
}

TEST_CASE("Automatic Differentiation", "[autodiff]") {
    SECTION("Basic scalar differentiation") {
        constexpr VarIDType x_id = U'x';
        fscal_var<x_id> x{2.0f};
        
        // d/dx(x) = 1
        auto dx_dx = derivate<x_id>(x);
        REQUIRE(dx_dx.eval(0, 0, 0, 0) == Approx(1.0f));
        
        // d/dx(x^2) = 2x
        auto x_squared = pow(x, 2.0f);
        auto d_x_squared = derivate<x_id>(x_squared);
        REQUIRE(d_x_squared.eval(0, 0, 0, 0) == Approx(4.0f)); // 2 * 2
    }
    
    SECTION("Chain rule") {
        constexpr VarIDType x_id = U'x';
        fscal_var<x_id> x{3.0f};
        
        // d/dx(log(x)) = 1/x
        auto log_x = log(x);
        auto d_log_x = derivate<x_id>(log_x);
        REQUIRE(d_log_x.eval() == Approx(1.0f/3.0f));
    }
    
    SECTION("Product rule") {
        constexpr VarIDType x_id = U'x';
        constexpr VarIDType y_id = U'y';
        fscal_var<x_id> x{2.0f};
        fscal_var<y_id> y{3.0f};
        
        // d/dx(x * y) = y (treating y as constant)
        auto product = x * y;
        auto d_product_dx = derivate<x_id>(product);
        REQUIRE(d_product_dx.eval() == Approx(3.0f));
        
        // d/dy(x * y) = x (treating x as constant)
        auto d_product_dy = derivate<y_id>(product);
        REQUIRE(d_product_dy.eval() == Approx(2.0f));
    }
    
    SECTION("Vector differentiation") {
        constexpr VarIDType x_id = U'x';
        fvec2_var<x_id> x{1.0f, 2.0f};  // Single braces for vectors
        
        // d/dx(x) should be ones vector
        auto dx_dx = derivate<x_id>(x);
        REQUIRE(dx_dx.eval(0, 0, 0, 0) == Approx(1.0f));
        REQUIRE(dx_dx.eval(1, 0, 1, 0) == Approx(1.0f));
    }
}

TEST_CASE("Broadcasting and Shape Compatibility", "[broadcasting]") {
    SECTION("Scalar-vector broadcasting") {
        fvec3 v{1.0f, 2.0f, 3.0f};  // Single braces for vectors
        fscal s{5.0f};
        
        auto result = v + s;
        REQUIRE(result.eval(0, 0) == Approx(6.0f));
        REQUIRE(result.eval(1, 0) == Approx(7.0f));
        REQUIRE(result.eval(2, 0) == Approx(8.0f));
    }
    
    SECTION("Shape validation at compile time") {
        // These should compile successfully
        fmat2 m1{{1.0f, 2.0f}, {3.0f, 4.0f}};
        fmat2 m2{{5.0f, 6.0f}, {7.0f, 8.0f}};
        auto valid_add = m1 + m2;
        REQUIRE(valid_add.eval(0, 0) == Approx(6.0f));
        
        // Matrix multiplication compatibility
        fmat<2, 3> ma{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
        fmat<3, 2> mb{{7.0f, 8.0f}, {9.0f, 10.0f}, {11.0f, 12.0f}};
        auto mat_mult = ma * mb;
        REQUIRE(mat_mult.eval(0, 0) == Approx(58.0f)); // 1*7 + 2*9 + 3*11
    }
}

TEST_CASE("String Representation", "[string]") {
    SECTION("Scalar string representation") {
        fscal s{3.14f};
        auto str = s.to_string();
        REQUIRE_FALSE(str.empty());
        
        auto zero_str = zero<float>{}.to_string();
        REQUIRE(zero_str == "0");
        
        auto unit_str = unit{}.to_string();
        REQUIRE(unit_str == "1");
    }
    
    SECTION("Expression string representation") {
        fscal x{2.0f};
        fscal y{3.0f};
        
        auto sum = x + y;
        auto sum_str = sum.to_string();
        REQUIRE_FALSE(sum_str.empty());
        
        auto product = x * y;
        auto product_str = product.to_string();
        REQUIRE_FALSE(product_str.empty());
    }
    
    SECTION("Mathematical function strings") {
        fscal x{2.0f};
        auto log_x = log(x);
        auto log_str = log_x.to_string();
        REQUIRE(log_str.find("log") != std::string::npos);
        
        auto pow_x = pow(x, 2.0f);
        auto pow_str = pow_x.to_string();
        REQUIRE(pow_str.find("^") != std::string::npos);
    }
}

TEST_CASE("Complex Number Operations", "[complex]") {
    SECTION("Complex arithmetic") {
        cfvec2 c1{std::complex<float>(1.0f, 2.0f), std::complex<float>(3.0f, 4.0f)};  // Single braces
        cfvec2 c2{std::complex<float>(5.0f, 6.0f), std::complex<float>(7.0f, 8.0f)};  // Single braces
        
        auto sum = c1 + c2;
        auto val = sum.eval(0, 0);
        REQUIRE(val.real() == Approx(6.0f));
        REQUIRE(val.imag() == Approx(8.0f));
        
        val = sum.eval(1, 0);
        REQUIRE(val.real() == Approx(10.0f));
        REQUIRE(val.imag() == Approx(12.0f));
    }
    
    SECTION("Complex conjugate") {
        cfscal c{std::complex<float>(3.0f, 4.0f)};
        auto conj_c = conj(c);
        auto val = conj_c.eval();
        REQUIRE(val.real() == Approx(3.0f));
        REQUIRE(val.imag() == Approx(-4.0f));
    }
}

TEST_CASE("Type Safety and Concepts", "[types]") {
    SECTION("Scalar type concept validation") {
        static_assert(ScalarType<float>);
        static_assert(ScalarType<double>);
        static_assert(ScalarType<std::complex<float>>);
        static_assert(ScalarType<int>);
        static_assert(!ScalarType<std::string>);
    }
    
    SECTION("Expression type concept validation") {
        fvec2 v{1.0f, 2.0f};  // Single braces for vectors
        static_assert(ExprType<decltype(v)>);
        static_assert(ExprType<decltype(v + v)>);
        static_assert(ExprType<decltype(transpose(v))>);
    }
}

TEST_CASE("Performance and Memory", "[performance]") {
    SECTION("Expression templates don't store intermediate results") {
        fvec3 v1{1.0f, 2.0f, 3.0f};  // Single braces for vectors
        fvec3 v2{4.0f, 5.0f, 6.0f};  // Single braces for vectors
        fvec3 v3{7.0f, 8.0f, 9.0f};  // Single braces for vectors
        
        // Complex expression should evaluate lazily
        auto complex_expr = v1 + elementwiseProduct(v2, v3) - v1;
        REQUIRE(complex_expr[0] == Approx(28.0f)); // 1 + 4*7 - 1
        REQUIRE(complex_expr[1] == Approx(40.0f)); // 2 + 5*8 - 2
        REQUIRE(complex_expr[2] == Approx(54.0f)); // 3 + 6*9 - 3
    }
}

TEST_CASE("Edge Cases and Error Handling", "[edge_cases]") {
    SECTION("Zero handling in operations") {
        auto z = zero<float>{};
        fmat2 m{{1.0f, 2.0f}, {3.0f, 4.0f}};
        
        auto zero_add = z + m;
        REQUIRE(zero_add.eval(0, 0) == Approx(1.0f));
        REQUIRE(zero_add.eval(1, 1) == Approx(4.0f));
        
        auto zero_mult = z * m;
        REQUIRE(zero_mult.eval(0, 0) == Approx(0.0f));
        REQUIRE(zero_mult.eval(1, 1) == Approx(0.0f));
    }
    
    SECTION("Identity matrix operations") {
        auto id = identity2{};
        fmat2 m{{1.0f, 2.0f}, {3.0f, 4.0f}};
        
        auto id_mult = id * m;
        REQUIRE(id_mult.at(0, 0) == Approx(1.0f));
        REQUIRE(id_mult.at(0, 1) == Approx(2.0f));
        REQUIRE(id_mult.at(1, 0) == Approx(3.0f));
        REQUIRE(id_mult.at(1, 1) == Approx(4.0f));
    }
}

TEST_CASE("Indexing", "[indexing]") {
    SECTION("Matrix and vector indexing") {
        auto m = fmat2{{10.0f, 20.0f}, {30.0f, 40.0f}};
        m[1][1] += 5.0f;
        
        float indexed00 = m[0][0];
        float indexed01 = m[0][1];
        auto indexed10 = m[1][0];
        auto indexed11 = m[1][1];

        auto v = fvec2{100.0f, 200.0f};  // Single braces for vectors
        
        v[1] += 50;
        v[0] -= 5;
        
        auto x = fscal{2};
        x += 5;
        REQUIRE(x == Approx(7.0f));

        auto indexedv0 = v[0] -= 5;
        auto indexedv1 = v[1];

        REQUIRE(indexedv0 == Approx(90.0f));
        REQUIRE(indexedv1 == Approx(250.0f));
        REQUIRE(indexed00 == Approx(10.0f));
        REQUIRE(indexed01 == Approx(20.0f));
        REQUIRE(indexed10 == Approx(30.0f));
        REQUIRE(indexed11 == Approx(45.0f));
    }
}

TEST_CASE("Array Access Operator", "[operator_bracket]") {
    SECTION("Vector element access") {
        fvec3 v{1.0f, 2.0f, 3.0f};
        
        // Test read access
        REQUIRE(v[0] == Approx(1.0f));
        REQUIRE(v[1] == Approx(2.0f));
        REQUIRE(v[2] == Approx(3.0f));
        
        // Test write access
        v[0] = 10.0f;
        v[1] = 20.0f;
        v[2] = 30.0f;
        
        REQUIRE(v[0] == Approx(10.0f));
        REQUIRE(v[1] == Approx(20.0f));
        REQUIRE(v[2] == Approx(30.0f));
    }
    
    SECTION("Matrix row access") {
        fmat3 m{{1.0f, 2.0f, 3.0f}, 
                {4.0f, 5.0f, 6.0f}, 
                {7.0f, 8.0f, 9.0f}};
        
        // Test accessing entire rows
        auto row0 = m[0];
        auto row1 = m[1];
        auto row2 = m[2];
        
        // Check first row
        REQUIRE(row0[0] == Approx(1.0f));
        REQUIRE(row0[1] == Approx(2.0f));
        REQUIRE(row0[2] == Approx(3.0f));
        
        // Check second row
        REQUIRE(row1[0] == Approx(4.0f));
        REQUIRE(row1[1] == Approx(5.0f));
        REQUIRE(row1[2] == Approx(6.0f));
        
        // Check third row
        REQUIRE(row2[0] == Approx(7.0f));
        REQUIRE(row2[1] == Approx(8.0f));
        REQUIRE(row2[2] == Approx(9.0f));
    }
    
    SECTION("Nested matrix indexing") {
        fmat2 m{{1.0f, 2.0f}, {3.0f, 4.0f}};
        
        // Test double indexing m[row][col]
        REQUIRE(m[0][0] == Approx(1.0f));
        REQUIRE(m[0][1] == Approx(2.0f));
        REQUIRE(m[1][0] == Approx(3.0f));
        REQUIRE(m[1][1] == Approx(4.0f));
        
        // Test write access via double indexing
        m[0][0] = 10.0f;
        m[0][1] = 20.0f;
        m[1][0] = 30.0f;
        m[1][1] = 40.0f;
        
        REQUIRE(m[0][0] == Approx(10.0f));
        REQUIRE(m[0][1] == Approx(20.0f));
        REQUIRE(m[1][0] == Approx(30.0f));
        REQUIRE(m[1][1] == Approx(40.0f));
    }
    
    SECTION("Compound assignment operators") {
        fvec3 v{10.0f, 20.0f, 30.0f};
        
        // Test += operator
        v[0] += 5.0f;
        v[1] += 10.0f;
        v[2] += 15.0f;
        
        REQUIRE(v[0] == Approx(15.0f));
        REQUIRE(v[1] == Approx(30.0f));
        REQUIRE(v[2] == Approx(45.0f));
        
        // Test -= operator
        v[0] -= 3.0f;
        v[1] -= 6.0f;
        v[2] -= 9.0f;
        
        REQUIRE(v[0] == Approx(12.0f));
        REQUIRE(v[1] == Approx(24.0f));
        REQUIRE(v[2] == Approx(36.0f));
        
        // Test *= operator
        v[0] *= 2.0f;
        v[1] *= 0.5f;
        v[2] *= 3.0f;
        
        REQUIRE(v[0] == Approx(24.0f));
        REQUIRE(v[1] == Approx(12.0f));
        REQUIRE(v[2] == Approx(108.0f));
        
        // Test /= operator
        v[0] /= 4.0f;
        v[1] /= 3.0f;
        v[2] /= 6.0f;
        
        REQUIRE(v[0] == Approx(6.0f));
        REQUIRE(v[1] == Approx(4.0f));
        REQUIRE(v[2] == Approx(18.0f));
    }
    
    SECTION("Matrix compound assignment") {
        fmat2 m{{8.0f, 12.0f}, {16.0f, 20.0f}};
        
        // Test compound assignment on matrix elements
        m[0][0] += 2.0f;
        m[0][1] -= 2.0f;
        m[1][0] *= 2.0f;
        m[1][1] /= 4.0f;
        
        REQUIRE(m[0][0] == Approx(10.0f));
        REQUIRE(m[0][1] == Approx(10.0f));
        REQUIRE(m[1][0] == Approx(32.0f));
        REQUIRE(m[1][1] == Approx(5.0f));
    }
    
    SECTION("Chain operations") {
        fvec4 v{1.0f, 2.0f, 3.0f, 4.0f};
        
        // Test chained operations
        auto result1 = (v[0] += 5.0f);
        auto result2 = (v[1] *= 3.0f);
        
        REQUIRE(result1 == Approx(6.0f));
        REQUIRE(result2 == Approx(6.0f));
        REQUIRE(v[0] == Approx(6.0f));
        REQUIRE(v[1] == Approx(6.0f));
    }
    
    SECTION("Expression indexing") {
        fvec3 v1{1.0f, 2.0f, 3.0f};
        fvec3 v2{4.0f, 5.0f, 6.0f};
        
        // Test indexing on expression results
        auto sum = v1 + v2;
        REQUIRE(sum[0] == Approx(5.0f));  // 1 + 4
        REQUIRE(sum[1] == Approx(7.0f));  // 2 + 5
        REQUIRE(sum[2] == Approx(9.0f));  // 3 + 6
        
        auto product = elementwiseProduct(v1, v2);
        REQUIRE(product[0] == Approx(4.0f));   // 1 * 4
        REQUIRE(product[1] == Approx(10.0f));  // 2 * 5
        REQUIRE(product[2] == Approx(18.0f));  // 3 * 6
    }
    
    SECTION("Complex number indexing") {
        cfvec2 cv{std::complex<float>(1.0f, 2.0f), std::complex<float>(3.0f, 4.0f)};
        
        // Test read access
        std::complex<float> val0 = cv[0];
        std::complex<float> val1 = cv[1];
        
        REQUIRE(val0.real() == Approx(1.0f));
        REQUIRE(val0.imag() == Approx(2.0f));
        REQUIRE(val1.real() == Approx(3.0f));
        REQUIRE(val1.imag() == Approx(4.0f));
        
        // Test write access
        cv[0] = std::complex<float>(5.0f, 6.0f);
        cv[1] = std::complex<float>(7.0f, 8.0f);
        
        REQUIRE(static_cast<std::complex<float>>(cv[0]).real() == Approx(5.0f));
        REQUIRE(static_cast<std::complex<float>>(cv[0]).imag() == Approx(6.0f));
        REQUIRE(static_cast<std::complex<float>>(cv[1]).real() == Approx(7.0f));
        REQUIRE(static_cast<std::complex<float>>(cv[1]).imag() == Approx(8.0f));
    }
    
    SECTION("Different numeric types") {
        // Integer vector
        ivec3 iv{10, 20, 30};
        REQUIRE(iv[0] == 10);
        REQUIRE(iv[1] == 20);
        REQUIRE(iv[2] == 30);
        
        iv[0] = 100;
        iv[1] += 50;
        iv[2] *= 2;
        
        REQUIRE(iv[0] == 100);
        REQUIRE(iv[1] == 70);
        REQUIRE(iv[2] == 60);
        
        // Double precision vector
        dvec2 dv{1.5, 2.5};
        REQUIRE(dv[0] == Approx(1.5));
        REQUIRE(dv[1] == Approx(2.5));
        
        dv[0] /= 2.0;
        dv[1] -= 0.5;
        
        REQUIRE(dv[0] == Approx(0.75));
        REQUIRE(dv[1] == Approx(2.0));
    }
}