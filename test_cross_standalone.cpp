#include <iostream>
#include <cmath>
#include "../include/TinyLA.h"

using namespace tinyla;

int main() {
    std::cout << "Testing Cross Product Derivative...\n\n";
    
    // Test 1: d(a×b)/da where b is constant
    {
        std::cout << "Test 1: d(a×b)/da where b is constant\n";
        constexpr VarIDType a_id = U'a';
        dvec3_var<a_id> a{2.0, 3.0, 5.0};
        dvec3 b{7.0, 11.0, 13.0};
        
        auto cross_product = cross(a, b);
        std::cout << "a × b = " << cross_product.to_string() << "\n";
        
        auto derivative = derivate<a_id>(cross_product);
        std::cout << "d(a×b)/da type constructed\n";
        
        // Test the derivative components
        // d(a×b)/da[0] should be [0, -b[2], b[1]] = [0, -13, 11]
        double d0_da0 = derivative.eval_at(0, 0, 0, 0);
        double d1_da0 = derivative.eval_at(1, 0, 1, 0);
        double d2_da0 = derivative.eval_at(2, 0, 2, 0);
        
        std::cout << "d(a×b)/da[0] = [" << d0_da0 << ", " << d1_da0 << ", " << d2_da0 << "]\n";
        std::cout << "Expected:      [0, -13, 11]\n";
        
        bool pass1 = std::abs(d0_da0 - 0.0) < 1e-10;
        bool pass2 = std::abs(d1_da0 - (-b.eval_at(2, 0))) < 1e-10;
        bool pass3 = std::abs(d2_da0 - b.eval_at(1, 0)) < 1e-10;
        
        if (pass1 && pass2 && pass3) {
            std::cout << "✓ Test 1 PASSED\n\n";
        } else {
            std::cout << "✗ Test 1 FAILED\n\n";
            return 1;
        }
    }
    
    // Test 2: d(a×b)/db where a is constant
    {
        std::cout << "Test 2: d(a×b)/db where a is constant\n";
        constexpr VarIDType b_id = U'b';
        dvec3 a{2.0, 3.0, 5.0};
        dvec3_var<b_id> b{7.0, 11.0, 13.0};
        
        auto cross_product = cross(a, b);
        auto derivative = derivate<b_id>(cross_product);
        
        // d(a×b)/db[0] should be [0, a[2], -a[1]] = [0, 5, -3]
        double d0_db0 = derivative.eval_at(0, 0, 0, 0);
        double d1_db0 = derivative.eval_at(1, 0, 1, 0);
        double d2_db0 = derivative.eval_at(2, 0, 2, 0);
        
        std::cout << "d(a×b)/db[0] = [" << d0_db0 << ", " << d1_db0 << ", " << d2_db0 << "]\n";
        std::cout << "Expected:      [0, 5, -3]\n";
        
        bool pass1 = std::abs(d0_db0 - 0.0) < 1e-10;
        bool pass2 = std::abs(d1_db0 - a.eval_at(2, 0)) < 1e-10;
        bool pass3 = std::abs(d2_db0 - (-a.eval_at(1, 0))) < 1e-10;
        
        if (pass1 && pass2 && pass3) {
            std::cout << "✓ Test 2 PASSED\n\n";
        } else {
            std::cout << "✗ Test 2 FAILED\n\n";
            return 1;
        }
    }
    
    // Test 3: Both variables
    {
        std::cout << "Test 3: d(a×b) with both a and b variable\n";
        constexpr VarIDType a_id = U'a';
        constexpr VarIDType b_id = U'b';
        dvec3_var<a_id> a{2.0, 3.0, 5.0};
        dvec3_var<b_id> b{7.0, 11.0, 13.0};
        
        auto cross_product = cross(a, b);
        
        auto deriv_a = derivate<a_id>(cross_product);
        auto deriv_b = derivate<b_id>(cross_product);
        
        // Test d/da
        double d0_da0 = deriv_a.eval_at(0, 0, 0, 0);
        double d1_da0 = deriv_a.eval_at(1, 0, 1, 0);
        double d2_da0 = deriv_a.eval_at(2, 0, 2, 0);
        
        std::cout << "d(a×b)/da[0] = [" << d0_da0 << ", " << d1_da0 << ", " << d2_da0 << "]\n";
        std::cout << "Expected:      [0, -13, 11]\n";
        
        // Test d/db
        double d0_db0 = deriv_b.eval_at(0, 0, 0, 0);
        double d1_db0 = deriv_b.eval_at(1, 0, 1, 0);
        double d2_db0 = deriv_b.eval_at(2, 0, 2, 0);
        
        std::cout << "d(a×b)/db[0] = [" << d0_db0 << ", " << d1_db0 << ", " << d2_db0 << "]\n";
        std::cout << "Expected:      [0, 5, -3]\n";
        
        bool pass_a = std::abs(d0_da0 - 0.0) < 1e-10 && 
                      std::abs(d1_da0 - (-b.eval_at(2, 0))) < 1e-10 && 
                      std::abs(d2_da0 - b.eval_at(1, 0)) < 1e-10;
        bool pass_b = std::abs(d0_db0 - 0.0) < 1e-10 && 
                      std::abs(d1_db0 - a.eval_at(2, 0)) < 1e-10 && 
                      std::abs(d2_db0 - (-a.eval_at(1, 0))) < 1e-10;
        
        if (pass_a && pass_b) {
            std::cout << "✓ Test 3 PASSED\n\n";
        } else {
            std::cout << "✗ Test 3 FAILED\n\n";
            return 1;
        }
    }
    
    std::cout << "All tests PASSED!\n";
    return 0;
}
