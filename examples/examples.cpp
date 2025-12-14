#include "TinyLA.h"
#include <iostream>
#include <print>
#include <complex>
#include <format>
#include <sstream>
#include <vector>
#include <algorithm>


void print_2d_slice(const std::string& header, 
                   const std::vector<std::vector<std::string>>& slice_data,
                   const std::vector<size_t>& col_widths,
                   const std::string& indent,
                   bool is_first) {
    
    uint32_t rows = slice_data.size();
    uint32_t cols = slice_data.empty() ? 0 : slice_data[0].size();

    std::println("{} = ", header);
    for (uint32_t r = 0; r < rows; ++r) {
        std::print("{}", indent);
        
        // Choose bracket style based on matrix type and position
        char left_bracket, right_bracket;
        if (rows == 1) {
            // Row vector - single row
            left_bracket = '['; right_bracket = ']';
        } else if (cols == 1) {
            // Column vector - use vertical bars for multi-row
            left_bracket = '|'; right_bracket = '|';
        } else {
            // Matrix - use vertical bars for multi-row
            left_bracket = '|'; right_bracket = '|';
        }
        
        std::print("{}", left_bracket);
        
        for (uint32_t c = 0; c < cols; ++c) {
            if (c > 0) std::print("  "); // Column separator
            
            // Right-align numbers in their column width
            std::print("{:>{}}", slice_data[r][c], col_widths[c]);
        }
        
        std::println("{}", right_bracket);
    }
}


void print_expr(const auto& expr) {
    std::println("Expression shape: {}", expr.shape());
    if constexpr (expr.rows == 1 && expr.cols == 1 && expr.depth == 1 && expr.time == 1) {
        // Scalar case: simple one-line output
        std::println("{} = {}\n", expr.to_string(), expr.eval_at(0, 0, 0, 0));
    } else {
        // Matrix/Vector/Tensor case: formatted multi-line output
        
        // First pass: calculate maximum width for each column across all depth/time slices
        std::vector<size_t> col_widths(expr.cols, 0);
        std::vector<std::vector<std::vector<std::vector<std::string>>>> formatted_values(
            expr.time, std::vector<std::vector<std::vector<std::string>>>(
                expr.depth, std::vector<std::vector<std::string>>(
                    expr.rows, std::vector<std::string>(expr.cols)
                )
            )
        );
        
        // Calculate column widths and format all values
        for (uint32_t t = 0; t < expr.time; ++t) {
            for (uint32_t d = 0; d < expr.depth; ++d) {
                for (uint32_t r = 0; r < expr.rows; ++r) {
                    for (uint32_t c = 0; c < expr.cols; ++c) {
                        auto value = expr.eval_at(r, c, d, t);
                        
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
                        
                        formatted_values[t][d][r][c] = formatted;
                        col_widths[c] = std::max(col_widths[c], formatted.length());
                    }
                }
            }
        }

        // Calculate the expression name and equals sign offset
        std::string expr_str = expr.to_string();
        size_t expr_line_length = 0;
        
        // Find the longest line in the expression string
        std::istringstream iss(expr_str);
        std::string line;
        while (std::getline(iss, line)) {
            expr_line_length = std::max(expr_line_length, line.length());
        }
        
        std::string indent(expr_line_length + 3, ' '); // +3 for " = "
        
        // Print header with expression name
        bool first_output = true;
        
        // Handle different dimensionalities
        if (expr.time == 1 && expr.depth == 1) {
            // 2D case (matrix/vector)
            print_2d_slice(expr_str, formatted_values[0][0], col_widths, indent, first_output);
        } else if (expr.time == 1) {
            // 3D case (depth > 1)
            for (uint32_t d = 0; d < expr.depth; ++d) {
                if (d > 0) {
                    std::println(); // Extra spacing between depth slices
                }
                
                std::string slice_header = first_output ? 
                    std::format("{} = [:, :, {}]", expr_str, d) :
                    std::format("{}[:, :, {}]", indent, d);
                    
                print_2d_slice(slice_header, formatted_values[0][d], col_widths, indent, first_output);
                first_output = false;
            }
        } else {
            // 4D case (time > 1)
            for (uint32_t t = 0; t < expr.time; ++t) {
                if (t > 0) {
                    std::println(); // Extra spacing between time slices
                }
                
                for (uint32_t d = 0; d < expr.depth; ++d) {
                    if (d > 0 || t > 0) {
                        std::println(); // Spacing between slices
                    }
                    
                    std::string slice_header = first_output ? 
                        std::format("{} = [:, :, {}, {}]", expr_str, d, t) :
                        std::format("{}[:, :, {}, {}]", indent, d, t);
                        
                    print_2d_slice(slice_header, formatted_values[t][d], col_widths, indent, first_output);
                    first_output = false;
                }
            }
        }
        
        std::println(); // Final spacing
    }
}


int main() {

    auto t = tinyla::dscal_var<'t'>{1.0};
    auto p_t = pow(t, 2) + 3 * t + 5;
    auto v_t = p_t.derivate<'t'>();
    //auto a_t = v_t.derivate<'t'>();
    print_expr(p_t);
    print_expr(v_t);
    //print_expr(a_t);


    // Scalar variables
    auto x = tinyla::dscal_var<'x'>{5.0};   // Variable with ID 'x'
    auto y = tinyla::dscal_var<'y'>{3.0};   // Variable with ID 'y'
    const auto constant = tinyla::dscal{2.0}; // Constant (no variable ID)



    // Define an expression
    auto expr = (x + y) * constant - x / y;

    // Print the symbolic expression and the value
    std::cout << "Expression: " << expr.to_string() << std::endl;
    std::cout << "Value: " << expr.eval_at() << std::endl;


    // Create 3D vectors
    auto v1 = tinyla::dvec3_var<'u'>{1.0, 2.0, 3.0};  // Variable vector with ID 'u'
    auto v2 = tinyla::dvec3{4.0, 5.0, 6.0};       // Constant vector

    // Vector arithmetic
    auto sum = v1 + v2;
    auto scaled = v1 * 2.0;
    auto dot_prod = dot(transpose(v1), v2);
    auto cosine = tinyla::cos(v1);
    auto sine = tinyla::sin(v2);
    print_expr(cosine.derivate<'u'>());
    print_expr(sine.derivate<'u'>());
    //auto cross_prod = cross(v1, v2);

    print_expr(dot_prod);

    // Create matrices
    auto matA = tinyla::dmat2{{1.0, 2.0}, {3.0, 4.0}};
    auto matB = tinyla::dmat2{{5.0, 6.0}, {7.0, 8.0}};
    auto matC = tinyla::dmat2{{9.0, 10.0}, {11.0, 12.0}};
    auto vec = tinyla::dvec2{1.0, 2.0};

    // Matrix operations
    auto matSum = matA + matB;
    auto matProd = matA * matB;
    auto elemProd = elementwiseProduct(matA, matB);
    auto transposed = transpose(matA);
    auto matVecProd = matA * vec;


    // Create variables
    auto A = tinyla::dmat2_var<'A'>{{2.0, 1.0}, {1.0, 3.0}};
    auto x2 = tinyla::dvec2_var<'x'>{5.0, 2.0};

    auto d = dot(T(x2), x2);
    print_expr(d);
    auto g = gradient<'x'>(d);
    print_expr(g);

    // Write an expression
    auto expr2 = transpose(A) * A * x2 + x2;
    // Derivate
    auto dx = expr2.derivate<'x'>();  // Derivative with respect to vector x
    auto dA = expr2.derivate<'A'>();  // Derivative with respect to matrix A
    std::cout << "d expr/dx = " << dx.to_string() << std::endl;
    std::cout << "d expr/dx at (0,0): " << dx.eval_at(0, 0) << std::endl;
    std::cout << "d expr/dA = " << dA.to_string() << std::endl;
    std::cout << "d expr/dA at (0,0): " << dA.eval_at(0, 0) << std::endl;
    auto evaluated_mat = expr2.eval();
    std::cout << "expr2 evaluated = " << evaluated_mat.to_string() << std::endl;

    // Complex-valued matrix
    auto cmat = tinyla::cmat2_var<'M'>{{std::complex<double>(1.0, 0.5), std::complex<double>(2.0, -1.0)},
                                {std::complex<double>(0.0, 1.0), std::complex<double>(3.0, 0.0)}};

    auto cmat2 = tinyla::cmat2_var<'A'>{{std::complex<double>(1.0, 0.5), std::complex<double>(2.0, -1.0)},
                                {std::complex<double>(0.0, 1.0), std::complex<double>(3.0, 0.0)}};
    // Complex operations
    auto conjugated = conj(cmat);
    auto adjoint_matrix = adjoint(cmat);  // Conjugate transpose

    auto cdiff = (cmat * cmat2).derivate<'M'>();  // Derivative with respect to matrix M
    std::cout << "d cmat/dM = " << cdiff.to_string() << std::endl;
    std::cout << "d cmat/dM at (0,0): " << cdiff.eval_at(0, 0, 0, 0) << std::endl;

    
    auto float_matrix = tinyla::fmat2{{1.0f, 2.0f}, {3.0f, 4.0f}};
    auto double_vector = tinyla::dvec2{1.0, 2.0};
    auto complex_scalar = tinyla::cscal{std::complex<double>(1.0, 0.5)};

    // Different data types with character-based variable IDs
    auto float_matrix_variable = tinyla::fmat2_var<'M'>{{1.0f, 2.0f}, {3.0f, 4.0f}};
    auto double_vector_variable = tinyla::dvec2_var<'v'>{1.0, 2.0};

    // All work together in expressions
    auto mixed_expr = complex_scalar * float_matrix * double_vector;

    // Mathematical constants
    auto pi = tinyla::pi<double>;     // π constant
    auto e = tinyla::euler<double>;   // Euler's number
    
    // Special matrices
    auto identity3 = tinyla::identity<double, 3>{};
    auto zero = tinyla::zero<double>{}; // A matrix filled with 0
    auto ones23 = tinyla::ones<double, 2, 3>{}; // A matrix filled with 1

    auto Op = tinyla::mat_var<double, 3, 4, 'M'>{   {3.0, -3.0, -3.0, 0.0},
                                                    {-4.0, 4.0, 0.0, 0.0},
                                                    {0.0, 4.0, -4.0, 1.0}                                                    
                                                };

    auto qr_op = tinyla::QRDecomposition{Op};
    qr_op.solve();
    print_expr(Op);
    print_expr(qr_op.Q);
    print_expr(qr_op.R);

    auto identity_to_qr = tinyla::VariableMatrix<double, 5, 5>::random(-10.0, 10.0);
    auto qr_identity = tinyla::QRDecomposition{identity_to_qr};
    qr_identity.solve();
    print_expr(identity_to_qr);
    print_expr(qr_identity.Q);
    print_expr(qr_identity.R);
    std::cout << "Determinant of Identity: " << qr_identity.determinant() << std::endl;


    auto u = tinyla::dvec4_var<'u'>{3.0, -3.0, -3.0, 1.0};
    auto norm_expr = p_norm<3>(Op * u);

    auto A_lin = tinyla::VariableMatrix<double, 5, 3>::random(-5.0, 5.0);
    auto b_lin = tinyla::VariableMatrix<double, 5, 1>::random(-5.0, 5.0);
    auto lin_eq = tinyla::LinearEquation{A_lin, b_lin};
    lin_eq.solve();
    print_expr(lin_eq.x);                                           
    
    auto AM = tinyla::dmat2_var<'A'>{
        {4.0, 1.0},
        {2.5, 10.0}
    };
    auto BM = tinyla::dmat2_var<'B'>{
        {0.0, 0.0},
        {0.0, 0.0} 
    };
    auto ones  = tinyla::dvec2{1.0, 1.0};
    auto to_minim = norm(abs(AM - BM) * ones);
    tinyla::AdamOptimizer{to_minim, BM, -100.0, 100.0}.solve();
    print_expr(AM);
    print_expr(BM);

    return 0;
}
