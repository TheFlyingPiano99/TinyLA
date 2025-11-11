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
        std::println("{} = {}\n", expr.to_string(), expr.eval(0, 0, 0, 0));
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
                        auto value = expr.eval(r, c, d, t);
                        
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

    auto v = tinyla::dvec3_var<'v'>{ 5.0, 4.0, 6.0 };   // Variable with ID 'x'
    auto u = tinyla::dvec3{ 1.0, 2.0, 3.0 };       // Constant vector
    auto expr = /*tinyla::dot(tinyla::transpose(v), u) */ tinyla::sin(tinyla::dot(tinyla::transpose(u), v));
    print_expr(expr);
    auto g = tinyla::gradient<'v'>(expr);
    print_expr(g);

    return 0;
}
