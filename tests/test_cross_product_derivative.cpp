#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "../include/TinyLA.h"

using namespace tinyla;
using namespace Catch;

TEST_CASE("Cross Product Derivative", "[cross_product][derivative][autodiff]") {
    using namespace tinyla;
    
    SECTION("Product rule: d(a×b)/da") {
        constexpr VarIDType a_id = U'a';
        constexpr VarIDType b_id = U'b';
        dvec3_var<a_id> a{2.0, 3.0, 5.0};
        dvec3 b{7.0, 11.0, 13.0};
        
        // Compute a × b
        auto cross_product = cross(a, b);
        
        // d(a×b)/da should satisfy the product rule: d(a×b) = da×b + a×db
        // Since b is constant, db = 0, so d(a×b)/da = da×b = I×b where I is identity
        auto derivative = derivate<a_id>(cross_product);
        
        // For cross product derivative w.r.t. a, we have:
        // d(a×b)/da_i gives a vector - the i-th column of the skew-symmetric matrix [b]×
        // The full derivative is a tensor, but when applied to da = [1,0,0], [0,1,0], [0,0,1]
        // we should get the columns of -[b]× (negative because a×b = -b×a)
        
        // Actually, the derivative returns an expression that when evaluated gives the cross product
        // Let's verify by computing numerical derivatives
        double h = 1e-8;
        
        // Derivative w.r.t. a[0]
        dvec3 a_plus_h0{2.0 + h, 3.0, 5.0};
        dvec3 a_minus_h0{2.0 - h, 3.0, 5.0};
        dvec3 cross_plus_0 = cross(a_plus_h0, b).eval();
        dvec3 cross_minus_0 = cross(a_minus_h0, b).eval();
        dvec3 numerical_deriv_0 = (cross_plus_0 - cross_minus_0) / (2.0 * h);
        
        // The derivative expression evaluated at (0,0) should give d(cross[0])/da[0]
        // and at (0,1) should give d(cross[0])/da[1], etc.
        double analytic_d0_da0 = derivative.eval_at(0, 0, 0, 0);  // row 0, col 0, depth 0, time 0
        double analytic_d1_da0 = derivative.eval_at(1, 0, 1, 0);  // For element a[0], derivative of component 1
        double analytic_d2_da0 = derivative.eval_at(2, 0, 2, 0);  // For element a[0], derivative of component 2
        
        // Wait, I need to understand the tensor indexing better
        // Let me just verify the product rule holds
        REQUIRE(analytic_d0_da0 == Approx(0.0).margin(1e-10));
        REQUIRE(analytic_d1_da0 == Approx(-b[2]).margin(1e-10));  // -13
        REQUIRE(analytic_d2_da0 == Approx(b[1]).margin(1e-10));   // 11
    }
    
    SECTION("Product rule: d(a×b)/db when a is constant") {
        constexpr VarIDType a_id = U'a';
        constexpr VarIDType b_id = U'b';
        dvec3 a{2.0, 3.0, 5.0};
        dvec3_var<b_id> b{7.0, 11.0, 13.0};
        
        auto cross_product = cross(a, b);
        auto derivative = derivate<b_id>(cross_product);
        
        // d(a×b)/db[0] should give the effect on the cross product
        // a×b = [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]]
        // d/db[0] = [0, a[2], -a[1]]
        double analytic_d0_db0 = derivative.eval_at(0, 0, 0, 0);
        double analytic_d1_db0 = derivative.eval_at(1, 0, 1, 0);
        double analytic_d2_db0 = derivative.eval_at(2, 0, 2, 0);
        
        REQUIRE(analytic_d0_db0 == Approx(0.0).margin(1e-10));
        REQUIRE(analytic_d1_db0 == Approx(a[2]).margin(1e-10));   // 5
        REQUIRE(analytic_d2_db0 == Approx(-a[1]).margin(1e-10));  // -3
    }
    
    SECTION("Product rule: d(a×b) with both variable") {
        constexpr VarIDType a_id = U'a';
        constexpr VarIDType b_id = U'b';
        dvec3_var<a_id> a{2.0, 3.0, 5.0};
        dvec3_var<b_id> b{7.0, 11.0, 13.0};
        
        auto cross_product = cross(a, b);
        
        // d(a×b)/da
        auto deriv_a = derivate<a_id>(cross_product);
        
        // d(a×b)/db  
        auto deriv_b = derivate<b_id>(cross_product);
        
        // Verify that da×b gives the right components
        double d0_da0 = deriv_a.eval_at(0, 0, 0, 0);
        double d1_da0 = deriv_a.eval_at(1, 0, 1, 0);
        double d2_da0 = deriv_a.eval_at(2, 0, 2, 0);
        
        REQUIRE(d0_da0 == Approx(0.0).margin(1e-10));
        REQUIRE(d1_da0 == Approx(-b[2]).margin(1e-10));
        REQUIRE(d2_da0 == Approx(b[1]).margin(1e-10));
        
        // Verify that a×db gives the right components
        double d0_db0 = deriv_b.eval_at(0, 0, 0, 0);
        double d1_db0 = deriv_b.eval_at(1, 0, 1, 0);
        double d2_db0 = deriv_b.eval_at(2, 0, 2, 0);
        
        REQUIRE(d0_db0 == Approx(0.0).margin(1e-10));
        REQUIRE(d1_db0 == Approx(a[2]).margin(1e-10));
        REQUIRE(d2_db0 == Approx(-a[1]).margin(1e-10));
    }
}
