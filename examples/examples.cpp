#include "TinyLA.h"
#include <iostream>
#include <print>
#include <complex>
#include <format>
#include <sstream>
#include <vector>
#include <algorithm>

void print_expr(const auto& expr) {
    if constexpr (expr.rows == 1 && expr.cols == 1) {
        // Scalar case: simple one-line output
        std::println("{} = {}\n", expr.to_string(), expr.eval(0, 0));
    } else {
        // VariableMatrix/Vector case: formatted multi-line output
        
        // First pass: calculate maximum width for each column
        std::vector<size_t> col_widths(expr.cols, 0);
        std::vector<std::vector<std::string>> formatted_values(expr.rows, std::vector<std::string>(expr.cols));
        
        for (uint32_t r = 0; r < expr.rows; ++r) {
            for (uint32_t c = 0; c < expr.cols; ++c) {
                auto value = expr.eval(r, c);
                
                // Format the value with appropriate precision
                std::string formatted;
                if constexpr (requires { value.real(); value.imag(); }) {
                    // Complex number
                    if (value.imag() >= 0) {
                        formatted = std::format("{:.6g}+{:.6g}i", value.real(), value.imag());
                    } else {
                        formatted = std::format("{:.6g}{:.6g}i", value.real(), value.imag());
                    }
                } else {
                    // Real number
                    formatted = std::format("{:.6g}", value);
                }
                
                formatted_values[r][c] = formatted;
                col_widths[c] = std::max(col_widths[c], formatted.length());
            }
        }
        
        // Calculate the expression name and equals sign offset
        std::string expr_str = expr.to_string();
        size_t max_name_length = 0;
        size_t current_line_length = 0;
        
        // Find the longest line in the expression string
        for (char c : expr_str) {
            if (c == '\n') {
                max_name_length = std::max(max_name_length, current_line_length);
                current_line_length = 0;
            } else {
                current_line_length++;
            }
        }
        // Don't forget the last line (if it doesn't end with \n)
        max_name_length = std::max(max_name_length, current_line_length);
        
        std::string indent(max_name_length + 3, ' '); // +3 for " = "
        
        // Print the matrix with proper alignment
        for (uint32_t r = 0; r < expr.rows; ++r) {
            if (r == 0) {
                // First row: include expression name
                std::print("{} = ", expr_str);
            } else {
                // Subsequent rows: indent to align with the opening bracket
                std::print("{}", indent);
            }
            
            // Choose bracket style based on matrix type and position
            char left_bracket, right_bracket;
            if (expr.rows == 1) {
                // Row vector - single row
                left_bracket = '['; right_bracket = ']';
            } else if (expr.cols == 1) {
                // Column vector - use vertical bars for multi-row
                left_bracket = '|'; right_bracket = '|';
            } else {
                // VariableMatrix - use vertical bars for multi-row
                left_bracket = '|'; right_bracket = '|';
            }
            
            std::print("{}", left_bracket);
            
            for (uint32_t c = 0; c < expr.cols; ++c) {
                if (c > 0) std::print("  "); // Column separator
                
                // Right-align numbers in their column width
                std::print("{:>{}}", formatted_values[r][c], col_widths[c]);
            }
            
            std::println("{}", right_bracket);
        }
        std::println(); // Extra line for spacing
    }
}



int main() {

// Scalar variables
auto x = TinyLA::dscal<'x'>{5.0};   // Variable with ID 'x'
auto y = TinyLA::dscal<'y'>{3.0};   // Variable with ID 'y'
const auto constant = TinyLA::dscal{2.0}; // Constant (no variable ID)

// Define an expression
auto expr = (x + y) * constant - x / y;

// Print the symbolic expression and the value
std::cout << "Expression: " << expr.to_string() << std::endl;
std::cout << "Value: " << expr.eval() << std::endl;


// Create 3D vectors
auto v1 = TinyLA::dvec3<'u'>{1.0, 2.0, 3.0};  // Variable vector with ID 'u'
auto v2 = TinyLA::dvec3<>{4.0, 5.0, 6.0};       // Constant vector

// Vector arithmetic
auto sum = v1 + v2;
auto scaled = v1 * 2.0;
//auto cross_prod = cross(v1, v2);
//auto dot_prod = dot(v1, v2);



// Create matrices
auto matA = TinyLA::dmat2<>{{1.0, 2.0}, {3.0, 4.0}};
auto matB = TinyLA::dmat2<>{{5.0, 6.0}, {7.0, 8.0}};
auto matC = TinyLA::dmat2<>{{9.0, 10.0}, {11.0, 12.0}};
auto vec = TinyLA::dvec2<>{{1.0, 2.0}};

// Matrix operations
auto matSum = matA + matB;          
auto matProd = matA * matB;         
auto elemProd = elementwiseProduct(matA, matB);
auto transposed = transpose(matA);
auto matVecProd = matA * vec;



// Create variables
auto A = TinyLA::dmat2<'A'>{{2.0, 1.0}, {1.0, 3.0}};
auto x2 = TinyLA::dvec2<'x'>{{5.0}, {2.0}};

// Write an expression
auto expr2 = transpose(A) * A * x2 + x2;

// Derivate
auto dA = expr2.derivate<'A'>();  // Derivative with respect to matrix A
auto dx = expr2.derivate<'x'>();  // Derivative with respect to vector x

std::cout << "d expr/dA at (0,0): " << dA.eval(0, 0) << std::endl;
std::cout << "d expr/dx at (0,0): " << dx.eval(0, 0) << std::endl;



// Complex-valued matrix
auto cmat = TinyLA::cmat2<'M'>{{std::complex<double>(1.0, 0.5), std::complex<double>(2.0, -1.0)},
                               {std::complex<double>(0.0, 1.0), std::complex<double>(3.0, 0.0)}};

// Complex operations
auto conjugated = conj(cmat);
auto adjoint_matrix = adj(cmat);  // Conjugate transpose


// The library provides convenient type aliases:
// Scalars: fscal, dscal, cscal (float, double, complex<double>)
// Vectors: fvec2, fvec3, fvec4, dvec2, dvec3, dvec4, cvec2, cvec3, cvec4
// Matrices: fmat2, fmat3, fmat4, dmat2, dmat3, dmat4, cmat2, cmat3, cmat4


// Different data types with character-based variable IDs
auto float_matrix = TinyLA::fmat2<'F'>{{1.0f, 2.0f}, {3.0f, 4.0f}};
auto double_vector = TinyLA::dvec2<'D'>{1.0, 2.0};
auto complex_scalar = TinyLA::cscal<'C'>{std::complex<double>(1.0, 0.5)};

// All work together in expressions
auto mixed_expr = complex_scalar * float_matrix * double_vector;

// Mathematical constants
auto pi = TinyLA::Pi<double>;     // π constant
auto e = TinyLA::Euler<double>;   // Euler's number

// Special matrices
auto identity3 = TinyLA::Identity<double, 3>{};
auto zero23 = TinyLA::Zero<double, 2, 3>{}; // A matrix filled with 0
auto ones22 = TinyLA::Ones<double, 2, 2>{}; // A matrix filled with 1

    return 0;
}
