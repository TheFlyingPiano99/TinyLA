#pragma once

/*
 * Linear Algebra Library
 * Automatic Differentiation Support
 * Author: Zoltan Simon
 * Date: October 2025
 * Description: A library for linear algebra operations with automatic differentiation support.
 * License: MIT License
 *
 */

#include <complex>
#include <concepts>
#include <initializer_list>
#include <type_traits>
#include <format>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <random>
#include <algorithm>

#ifdef ENABLE_CUDA_SUPPORT
#include <cuda/std/complex>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#define CUDA_COMPATIBLE __device__ __host__
#define CUDA_DEVICE __device__
#define CUDA_HOST __host__
#else
#define CUDA_COMPATIBLE
#define CUDA_DEVICE
#define CUDA_HOST
#endif



template <typename T>
struct std::formatter<std::complex<T>> : std::formatter<T> {
    template <typename FormatContext>
    auto format(const std::complex<T>& c, FormatContext& ctx) const {
        auto out = ctx.out();
        out = std::formatter<T>::format(c.real(), ctx);
        out = std::format_to(out, " + ");
        out = std::formatter<T>::format(std::abs(c.imag()), ctx);
        return std::format_to(out, "i");
    }
};


namespace tinyla {

    //-------------------------------------------------------------------------------

    // Trait to check if a type is a specialization of a template
    template <class, template<typename...> class>
    inline constexpr bool is_specialization_v = false;

    template <template<class...> class Template, class... Args>
    inline constexpr bool is_specialization_v<Template<Args...>, Template> = true;

    // Concepts:
#ifdef ENABLE_CUDA_SUPPORT
    template<class T>
    concept ComplexType = is_specialization_v<T, cuda::std::complex> || is_specialization_v<T, std::complex>;
#else
    template<class T>
    concept ComplexType = is_specialization_v<T, std::complex>;
#endif
    template<class T>
    concept RealType = std::is_floating_point_v<T>;

    template<class T>
    concept ScalarType = ComplexType<T> || RealType<T> || std::is_integral_v<T>;


    template<typename T>
    struct is_complex : std::false_type {};

    template<typename T>
    struct is_complex<std::complex<T>> : std::true_type {};

    template<typename T>
    constexpr bool is_complex_v = is_complex<T>::value;

    template<typename T>
    struct complex_value_type { using type = T; };

    template<typename T>
    struct complex_value_type<std::complex<T>> { using type = T; };

    template<typename T, typename U>
    struct common_arithmetic {
    private:
        using value_T = typename complex_value_type<T>::type;
        using value_U = typename complex_value_type<U>::type;
        using scalar_common = std::common_type_t<value_T, value_U>;
    public:
        using type = std::conditional_t<
            is_complex_v<T> || is_complex_v<U>,
            std::complex<scalar_common>,
            scalar_common>;
    };

    template<typename T, typename U>
    using common_arithmetic_t = typename common_arithmetic<T, U>::type;



    //---------------------------------------------------------------------------------------

    /*
        Variable ID type for automatic differentiation
    */
    using VarIDType = char32_t;

    static inline std::string char32_to_utf8(char32_t cp) {
        std::string out;
        if (cp <= 0x7F) {
            out.push_back(static_cast<char>(cp));
        }
        else if (cp <= 0x7FF) {
            out.push_back(static_cast<char>(0xC0 | ((cp >> 6) & 0x1F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
        else if (cp <= 0xFFFF) {
            out.push_back(static_cast<char>(0xE0 | ((cp >> 12) & 0x0F)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
        else if (cp <= 0x10FFFF) {
            out.push_back(static_cast<char>(0xF0 | ((cp >> 18) & 0x07)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
        else {
            throw std::runtime_error("Invalid Unicode code point");
        }
        return out;
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    template<class E>
    concept ExprType = requires(const E & e) {
        e.eval_at(0, 0, 0, 0);
        E::rows;
        E::cols;
        E::depth;
        E::time;
    };

template<ExprType E1, ExprType E2>
    struct is_eq_shape {
        static constexpr bool value = (E1::rows == E2::rows && E1::cols == E2::cols && E1::depth == E2::depth && E1::time == E2::time);
    };

    template<ExprType E1, ExprType E2>
    inline constexpr bool is_eq_shape_v = is_eq_shape<E1, E2>::value;

    template<ExprType E>
    struct is_scalar_shape {
        static constexpr bool value = E::rows == 1 && E::cols == 1 && E::depth == 1 && E::time == 1;
    };

    template<ExprType E>
    inline constexpr bool is_scalar_shape_v = is_scalar_shape<E>::value;

    template<ExprType E>
    struct is_variable {
        static constexpr bool value = E::__is_variable_data;
    };

    template<ExprType E>
    inline constexpr bool is_variable_v = is_variable<E>::value;

    template<ExprType E>
    struct is_matrix_shape {
        static constexpr bool value = E::depth == 1 && E::time == 1;
    };

    template<ExprType E>
    inline constexpr bool is_matrix_shape_v = is_matrix_shape<E>::value;

    template<ExprType E>
    struct is_square_matrix {
        static constexpr bool value = is_matrix_shape_v<E> && (E::rows == E::cols);
    };

    template<ExprType E>
    inline constexpr bool is_square_matrix_v = is_square_matrix<E>::value;

    template<ExprType E1, ExprType E2>
    struct is_elementwise_broadcastable {
        static constexpr bool value = is_eq_shape_v<E1, E2> || is_scalar_shape_v<E1> || is_scalar_shape_v<E2>;
    };

    template<ExprType E1, ExprType E2>
    inline constexpr bool is_elementwise_broadcastable_v = is_elementwise_broadcastable<E1, E2>::value;

    template<ExprType E1, ExprType E2>
    struct is_matrix_multiplicable {
        static constexpr bool value = (E1::cols == E2::rows);
    };

    template<ExprType E1, ExprType E2>
    inline constexpr bool is_matrix_multiplicable_v = is_matrix_multiplicable<E1, E2>::value;

    template<ExprType E>
    struct is_row_vector {
        static constexpr bool value = (E::rows == 1) && E::depth == 1 && E::time == 1;
    };

    template<ExprType E>
    inline constexpr bool is_row_vector_v = is_row_vector<E>::value;

    template<ExprType E>
    struct is_column_vector {
        static constexpr bool value = (E::cols == 1) && E::depth == 1 && E::time == 1;
    };

    template<ExprType E>
    inline constexpr bool is_column_vector_v = is_column_vector<E>::value;

    template<ExprType E>
    struct is_vector {
        static constexpr bool value = (E::rows == 1 || E::cols == 1) && E::depth == 1 && E::time == 1;
    };

    template<ExprType E>
    inline constexpr bool is_vector_v = is_vector<E>::value;

    template<ExprType E>
    struct is_tensor {
        static constexpr bool value = (E::depth > 1 || E::time > 1);
    };

    template<ExprType E>
    inline constexpr bool is_tensor_v = is_tensor<E>::value;

    template<ExprType E>
    struct is_3d_tensor {
        static constexpr bool value = (E::depth > 1 && E::time == 1);
    };

    template<ExprType E>
    inline constexpr bool is_3d_tensor_v = is_3d_tensor<E>::value;

    template<ScalarType S, ExprType E>
    [[nodiscard]]
    CUDA_COMPATIBLE inline constexpr auto operator==(S other, const E& expr) {
        static_assert(is_scalar_shape_v<E>, "Only scalar shaped matrices can be compared to scalar values.");
        return static_cast<S>(expr.eval_at(0, 0)) == other;
    }

    template<ScalarType S, ExprType E>
    [[nodiscard]]
    CUDA_COMPATIBLE inline constexpr auto operator==(const E& expr, S other) {
        static_assert(is_scalar_shape_v<E>, "Only scalar shaped matrices can be compared to scalar values.");
        return static_cast<S>(expr.eval_at(0, 0)) == other;
    }

    template<ScalarType S, ExprType E>
    [[nodiscard]]
    CUDA_COMPATIBLE inline constexpr auto operator<(S other, const E& expr) {
        static_assert(is_scalar_shape_v<E>, "Only scalar shaped matrices can be compared to scalar values.");
        return other < static_cast<S>(expr.eval_at(0, 0));
    }

    template<ScalarType S, ExprType E>
    [[nodiscard]]
    CUDA_COMPATIBLE inline constexpr auto operator>(S other, const E& expr) {
        static_assert(is_scalar_shape_v<E>, "Only scalar shaped matrices can be compared to scalar values.");
        return other > static_cast<S>(expr.eval_at(0, 0));
    }

    template<ScalarType S, ExprType E>
    [[nodiscard]]
    CUDA_COMPATIBLE inline constexpr auto operator<(const E& expr, S other) {
        static_assert(is_scalar_shape_v<E>, "Only scalar shaped matrices can be compared to scalar values.");
        return static_cast<S>(expr.eval_at(0, 0)) < other;
    }

    template<ScalarType S, ExprType E>
    [[nodiscard]]
    CUDA_COMPATIBLE inline constexpr auto operator>(const E& expr, S other) {
        static_assert(is_scalar_shape_v<E>, "Only scalar shaped matrices can be compared to scalar values.");
        return static_cast<S>(expr.eval_at(0, 0)) > other;
    }



    enum class StorageStrategy {
        ColumnMajor,
        RowMajor,
        Sparse
    };

    template<ScalarType T, uint32_t Row, uint32_t Col, VarIDType varId = 0, StorageStrategy Storage = StorageStrategy::ColumnMajor>
    class VariableMatrix;

    template<ExprType E, uint32_t Row, uint32_t Col, uint32_t Depth = 1, uint32_t Time = 1>
    class SubMatrixExpr;

    template<class E, uint32_t Row, uint32_t Col, uint32_t Depth = 1, uint32_t Time = 1>
    class AbstractExpr {
    public:
        static constexpr bool __is_variable_data = false;
        static constexpr bool __is_quaternion_valued = false;
        static constexpr uint32_t rows = Row;
        static constexpr uint32_t cols = Col;
        static constexpr uint32_t depth = Depth;
        static constexpr uint32_t time = Time;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto shape() const {
            return std::make_tuple(Row, Col, Depth, Time);
        }

        template<VarIDType varId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            return static_cast<const E&>(*this).derivate<varId>();
        }

        [[nodiscard]]
        CUDA_HOST constexpr inline std::string to_string() const {
            return static_cast<const E&>(*this).to_string();
        }

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval_at(uint32_t r, uint32_t c = 0, uint32_t dr = 0, uint32_t dc = 0) const {
            return (static_cast<const E&>(*this)).eval_at(r, c, dr, dc);
        }

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval() const {
            static_assert(is_matrix_shape_v<E>, "eval() call is not supported for tensor expressions.");
            auto mat = VariableMatrix<decltype(this->eval_at(0, 0, 0, 0)), (*this).rows, (*this).cols>{};
            for (int c = 0; c < (*this).cols; ++c) {
                for (int r = 0; r < (*this).rows; ++r) {
                    mat.at(r, c) = (static_cast<const E&>(*this)).eval_at(r, c);
                }
            }
            return mat;
        }

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto operator[](uint32_t i) requires(Row > 1 || Col > 1 || Depth > 1 || Time > 1);

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto operator[](uint32_t i) const requires(Row > 1 || Col > 1 || Depth > 1 || Time > 1);

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto at(uint32_t r, uint32_t c = 0, uint32_t dr = 0, uint32_t dc = 0);

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto at(uint32_t r, uint32_t c = 0, uint32_t dr = 0, uint32_t dc = 0) const;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto x() {
            static_assert(is_vector_v<E>, "x() can only be called on scalar or vector expressions.");
            return (static_cast<E&>(*this)).at(0, 0, 0, 0);
        }

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto x() const {
            static_assert(is_vector_v<E>, "x() can only be called on scalar or vector expressions.");
            return (static_cast<const E&>(*this)).at(0, 0, 0, 0);
        }

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto y() {
            static_assert(is_vector_v<E>, "y() can only be called on vector expressions.");
            if constexpr (E::rows == 1) {
                static_assert(E::cols >= 2, "y() called on row vector with less than 2 columns.");
                return (static_cast<E&>(*this)).at(0, 1, 0, 0);
            }
            else {
                static_assert(E::rows >= 2, "y() called on column vector with less than 2 rows.");
                return (static_cast<E&>(*this)).at(1, 0, 0, 0);
            }
        }

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto y() const {
            static_assert(is_vector_v<E>, "y() can only be called on vector expressions.");
            if constexpr (E::rows == 1) {
                static_assert(E::cols >= 2, "y() called on row vector with less than 2 columns.");
                return (static_cast<const E&>(*this)).at(0, 1, 0, 0);
            }
            else {
                static_assert(E::rows >= 2, "y() called on column vector with less than 2 rows.");
                return (static_cast<const E&>(*this)).at(1, 0, 0, 0);
            }
        }

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto z() {
            static_assert(is_vector_v<E>, "z() can only be called on vector expressions.");
            if constexpr (E::rows == 1) {
                static_assert(E::cols >= 3, "z() called on row vector with less than 3 columns.");
                return (static_cast<E&>(*this)).at(0, 2, 0, 0);
            }
            else {
                static_assert(E::rows >= 3, "z() called on column vector with less than 3 rows.");
                return (static_cast<E&>(*this)).at(2, 0, 0, 0);
            }
        }

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto z() const {
            static_assert(is_vector_v<E>, "z() can only be called on vector expressions.");
            if constexpr (E::rows == 1) {
                static_assert(E::cols >= 3, "z() called on row vector with less than 3 columns.");
                return (static_cast<const E&>(*this)).at(0, 2, 0, 0);
            }
            else {
                static_assert(E::rows >= 3, "z() called on column vector with less than 3 rows.");
                return (static_cast<const E&>(*this)).at(2, 0, 0, 0);
            }
        }

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto w() {
            static_assert(is_vector_v<E>, "w() can only be called on vector expressions.");
            if constexpr (E::rows == 1) {
                static_assert(E::cols >= 4, "w() called on row vector with less than 4 columns.");
                return (static_cast<E&>(*this)).at(0, 3, 0, 0);
            }
            else {
                static_assert(E::rows >= 4, "w() called on column vector with less than 4 rows.");
                return (static_cast<E&>(*this)).at(3, 0, 0, 0);
            }
        }

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto w() const {
            static_assert(is_vector_v<E>, "w() can only be called on vector expressions.");
            if constexpr (E::rows == 1) {
                static_assert(E::cols >= 4, "w() called on row vector with less than 4 columns.");
                return (static_cast<const E&>(*this)).at(0, 3, 0, 0);
            }
            else {
                static_assert(E::rows >= 4, "w() called on column vector with less than 4 rows.");
                return (static_cast<const E&>(*this)).at(3, 0, 0, 0);
            }
        }

        
        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto real();

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto real() const;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto imag();
        
        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto imag() const;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto i();

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto i() const;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto j();

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto j() const;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto k();

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto k() const;

        template<ScalarType S>
        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr operator S() const {
            static_assert(is_scalar_shape_v<E>, "Only scalar shaped matrices can be assigned to scalar variables.");
            return static_cast<S>(static_cast<const E&>(*this).eval_at(0, 0, 0, 0));
        }

        template<ScalarType S>
        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr operator S() {
            static_assert(is_scalar_shape_v<E>, "Only scalar shaped matrices can be assigned to scalar variables.");
            return static_cast<S>(static_cast<const E&>(*this).eval_at(0, 0, 0, 0));
        }

        protected:
        template<ScalarType S>
        CUDA_COMPATIBLE inline constexpr void __assign_at_if_applicable(S value, uint32_t r, uint32_t c) {
            static_assert(false, "Assignment to constant expressions is not allowed.");
        }

    };

    template<ExprType E, uint32_t Row, uint32_t Col, uint32_t Depth, uint32_t Time>
    class SubMatrixExpr : public AbstractExpr<SubMatrixExpr<E, Row, Col, Depth, Time>, Row, Col, Depth, Time> {
    public:
        // SubMatrixExpr is a view, not the actual variable data, so always false
        static constexpr bool __is_variable_data = false;

        CUDA_COMPATIBLE inline constexpr SubMatrixExpr(E& expr, uint32_t row_offset = 0, uint32_t col_offset = 0, uint32_t depth_offset = 0, uint32_t time_offset = 0)
            : m_expr(expr), m_row_offset(row_offset), m_col_offset(col_offset), m_depth_offset(depth_offset), m_time_offset(time_offset) {
        }

        template<ScalarType S>
        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto operator=(S value) {
            static_assert(is_scalar_shape_v<SubMatrixExpr>, "Assignment to submatrix is only supported for single element submatrices.");
            m_expr.__assign_at_if_applicable(value, m_row_offset, m_col_offset, m_depth_offset, m_time_offset);
            return *this;
        }

        template<ScalarType S>
        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto operator+=(S value) {
            static_assert(is_scalar_shape_v<SubMatrixExpr>, "Assignment to submatrix is only supported for single element submatrices.");
            m_expr.__assign_at_if_applicable(m_expr.eval_at(m_row_offset, m_col_offset) + value, m_row_offset, m_col_offset, m_depth_offset, m_time_offset);
            return *this;
        }

        template<ScalarType S>
        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto operator-=(S value) {
            static_assert(is_scalar_shape_v<SubMatrixExpr>, "Assignment to submatrix is only supported for single element submatrices.");
            m_expr.__assign_at_if_applicable(m_expr.eval_at(m_row_offset, m_col_offset) - value, m_row_offset, m_col_offset, m_depth_offset, m_time_offset);
            return *this;
        }

        template<ScalarType S>
        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto operator*=(S value) {
            static_assert(is_scalar_shape_v<SubMatrixExpr>, "Assignment to submatrix is only supported for single element submatrices.");
            m_expr.__assign_at_if_applicable(m_expr.eval_at(m_row_offset, m_col_offset) * value, m_row_offset, m_col_offset, m_depth_offset, m_time_offset);
            return *this;
        }

        template<ScalarType S>
        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto operator/=(S value) {
            static_assert(is_scalar_shape_v<SubMatrixExpr>, "Assignment to submatrix is only supported for single element submatrices.");
            m_expr.__assign_at_if_applicable(m_expr.eval_at(m_row_offset, m_col_offset) / value, m_row_offset, m_col_offset, m_depth_offset, m_time_offset);
            return *this;
        }

        template<ExprType SE> requires(is_scalar_shape_v<SE>)
        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto operator=(const SE& value) {
            static_assert(is_scalar_shape_v<SubMatrixExpr>, "Assignment to submatrix is only supported for single element submatrices.");
            m_expr.__assign_at_if_applicable(value.eval_at(0), m_row_offset, m_col_offset, m_depth_offset, m_time_offset);
            return *this;
        }

        template<ExprType SE> requires(is_scalar_shape_v<SE>)
        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto operator+=(const SE& value) {
            static_assert(is_scalar_shape_v<SubMatrixExpr>, "Assignment to submatrix is only supported for single element submatrices.");
            m_expr.__assign_at_if_applicable(this->eval_at(0) + value.eval_at(0), m_row_offset, m_col_offset, m_depth_offset, m_time_offset);
            return *this;
        }

        template<ExprType SE> requires(is_scalar_shape_v<SE>)
        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto operator-=(const SE& value) {
            static_assert(is_scalar_shape_v<SubMatrixExpr>, "Assignment to submatrix is only supported for single element submatrices.");
            m_expr.__assign_at_if_applicable(this->eval_at(0) - value.eval_at(0), m_row_offset, m_col_offset, m_depth_offset, m_time_offset);
            return *this;
        }

        template<ExprType SE> requires(is_scalar_shape_v<SE>)
        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto operator*=(const SE& value) {
            static_assert(is_scalar_shape_v<SubMatrixExpr>, "Assignment to submatrix is only supported for single element submatrices.");
            m_expr.__assign_at_if_applicable(this->eval_at(0) * value.eval_at(0), m_row_offset, m_col_offset, m_depth_offset, m_time_offset);
            return *this;
        }

        template<ExprType SE> requires(is_scalar_shape_v<SE>)
        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto operator/=(const SE& value) {
            static_assert(is_scalar_shape_v<SubMatrixExpr>, "Assignment to submatrix is only supported for single element submatrices.");
            m_expr.__assign_at_if_applicable(this->eval_at(0) / value.eval_at(0), m_row_offset, m_col_offset, m_depth_offset, m_time_offset);
            return *this;
        }

        template<VarIDType varId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert(varId > 0, "Variable ID for differentiation must be positive.");
            return SubMatrixExpr<decltype(m_expr.derivate<varId>()), Row, Col>(
                m_expr.derivate<varId>(), m_row_offset, m_col_offset, m_depth_offset, m_time_offset);
        }

        [[nodiscard]]
        CUDA_HOST constexpr inline std::string to_string() const {
            return std::format("SubMatrix[offset:{},{}, {}, {}]; size:{}x{}x{}x{}]({})", m_row_offset, m_col_offset, m_depth_offset, m_time_offset, Row, Col, Depth, Time, m_expr.to_string());
        }

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval_at(uint32_t r = 0, uint32_t c = 0, uint32_t d = 0, uint32_t t = 0) const {
            return m_expr.eval_at(r + m_row_offset, c + m_col_offset, d + m_depth_offset, t + m_time_offset);
        }

    private:
        template<ScalarType S>
        CUDA_COMPATIBLE inline constexpr void __assign_at_if_applicable(S value, uint32_t r, uint32_t c, uint32_t d, uint32_t t) {
            m_expr.__assign_at_if_applicable(value, r + m_row_offset, c + m_col_offset, d + m_depth_offset, t + m_time_offset);
        }

        template<ExprType E, uint32_t Row, uint32_t Col, uint32_t Depth, uint32_t Time>
        friend class SubMatrixExpr;

        // Store by reference if E itself is variable data (to allow mutation), otherwise by value
        std::conditional_t<E::__is_variable_data, E&, E> m_expr;
        uint32_t m_row_offset;
        uint32_t m_col_offset;
        uint32_t m_depth_offset;
        uint32_t m_time_offset;
    };

    template<uint32_t Row, uint32_t Col, uint32_t Depth, uint32_t Time, ExprType E>
    [[nodiscard]]
    CUDA_COMPATIBLE auto submatrix(const E& expr, uint32_t row_offset = 0, uint32_t col_offset = 0, uint32_t depth_offset = 0, uint32_t time_offset = 0) {
        return SubMatrixExpr<E, Row, Col, Depth, Time>{expr, row_offset, col_offset, depth_offset, time_offset};
    }

    template<class E, uint32_t Row, uint32_t Col, uint32_t Depth, uint32_t Time>
    [[nodiscard]]
    CUDA_COMPATIBLE inline constexpr auto AbstractExpr<E, Row, Col, Depth, Time>::operator[](uint32_t i) requires(Row > 1 || Col > 1 || Depth > 1 || Time > 1) {
        if constexpr (Row > 1) {
            return SubMatrixExpr<E, 1, Col, Depth, Time>{static_cast<E&>(*this), i, 0, 0, 0};
        }
        else if constexpr (Col > 1) {
            return SubMatrixExpr<E, 1, 1, Depth, Time>{static_cast<E&>(*this), 0, i, 0, 0};
        }
        else if constexpr (Depth > 1) {
            return SubMatrixExpr<E, 1, 1, 1, Time>{static_cast<E&>(*this), 0, 0, i, 0};
        }
        else { // Time > 1
            return SubMatrixExpr<E, 1, 1, 1, 1>{static_cast<E&>(*this), 0, 0, 0, i};
        }
    }

    
    template<class E, uint32_t Row, uint32_t Col, uint32_t Depth, uint32_t Time>
    [[nodiscard]]
    CUDA_COMPATIBLE inline constexpr auto AbstractExpr<E, Row, Col, Depth, Time>::operator[](uint32_t i) const requires(Row > 1 || Col > 1 || Depth > 1 || Time > 1) {
        if constexpr (Row > 1) {
            return SubMatrixExpr<const E, 1, Col, Depth, Time>{static_cast<const E&>(*this), i, 0, 0, 0};
        }
        else if constexpr (Col > 1) {
            return SubMatrixExpr<const E, 1, 1, Depth, Time>{static_cast<const E&>(*this), 0, i, 0, 0};
        }
        else if constexpr (Depth > 1) {
            return SubMatrixExpr<const E, 1, 1, 1, Time>{static_cast<const E&>(*this), 0, 0, i, 0};
        }
        else { // Time > 1
            return SubMatrixExpr<const E, 1, 1, 1, 1>{static_cast<const E&>(*this), 0, 0, 0, i};
        }
    }

    template<class E, uint32_t Row, uint32_t Col, uint32_t Depth, uint32_t Time>
    [[nodiscard]]
    CUDA_COMPATIBLE inline constexpr auto AbstractExpr<E, Row, Col, Depth, Time>::at(uint32_t r, uint32_t c, uint32_t d, uint32_t t) {
        return SubMatrixExpr<E, 1, 1, 1, 1>{static_cast<E&>(*this), r, c, d, t};
    }
    
    template<class E, uint32_t Row, uint32_t Col, uint32_t Depth, uint32_t Time>
    [[nodiscard]]
    CUDA_COMPATIBLE inline constexpr auto AbstractExpr<E, Row, Col, Depth, Time>::at(uint32_t r, uint32_t c, uint32_t d, uint32_t t) const {
        return SubMatrixExpr<const E, 1, 1, 1, 1>{static_cast<const E&>(*this), r, c, d, t};
    }

    template<VarIDType varId, ExprType E>
    [[nodiscard]]
    CUDA_COMPATIBLE constexpr auto derivate(const E& expr) {
        static_assert(varId > 0, "Variable ID for differentiation must be positive.");
        return expr.derivate<varId>();
    }

    template<class E, uint32_t Row, uint32_t Col, uint32_t Depth, uint32_t Time>
    [[nodiscard]]
    CUDA_COMPATIBLE inline constexpr auto AbstractExpr<E, Row, Col, Depth, Time>::real() {
        return SubMatrixExpr<E, 1, 1, 1, 1>{static_cast<E&>(*this), 0, 0, 0, 0};
    }

    template<class E, uint32_t Row, uint32_t Col, uint32_t Depth, uint32_t Time>
    [[nodiscard]]
    CUDA_COMPATIBLE inline constexpr auto AbstractExpr<E, Row, Col, Depth, Time>::real() const {
        return SubMatrixExpr<const E, 1, 1, 1, 1>{static_cast<const E&>(*this), 0, 0, 0, 0};
    }

    template<class E, uint32_t Row, uint32_t Col, uint32_t Depth, uint32_t Time>
    [[nodiscard]]
    CUDA_COMPATIBLE inline constexpr auto AbstractExpr<E, Row, Col, Depth, Time>::imag() {
            return SubMatrixExpr<E, 3, 1, 1, 1>{static_cast<E&>(*this), 1, 0, 0, 0};
    }

    template<class E, uint32_t Row, uint32_t Col, uint32_t Depth, uint32_t Time>
    [[nodiscard]]
    CUDA_COMPATIBLE inline constexpr auto AbstractExpr<E, Row, Col, Depth, Time>::imag() const {
        return SubMatrixExpr<const E, 3, 1, 1, 1>{static_cast<const E&>(*this), 1, 0, 0, 0};
    }

    template<class E, uint32_t Row, uint32_t Col, uint32_t Depth, uint32_t Time>
    [[nodiscard]]
    CUDA_COMPATIBLE inline constexpr auto AbstractExpr<E, Row, Col, Depth, Time>::i() {
        return SubMatrixExpr<E, 1, 1, 1, 1>{static_cast<E&>(*this), 1, 0, 0, 0};
    }

    template<class E, uint32_t Row, uint32_t Col, uint32_t Depth, uint32_t Time>
    [[nodiscard]]
    CUDA_COMPATIBLE inline constexpr auto AbstractExpr<E, Row, Col, Depth, Time>::i() const {
        return SubMatrixExpr<const E, 1, 1, 1, 1>{static_cast<const E&>(*this), 1, 0, 0, 0};
    }

    template<class E, uint32_t Row, uint32_t Col, uint32_t Depth, uint32_t Time>
    [[nodiscard]]
    CUDA_COMPATIBLE inline constexpr auto AbstractExpr<E, Row, Col, Depth, Time>::j() {
        return SubMatrixExpr<E, 1, 1, 1, 1>{static_cast<E&>(*this), 2, 0, 0, 0};
    }

    template<class E, uint32_t Row, uint32_t Col, uint32_t Depth, uint32_t Time>
    [[nodiscard]]
    CUDA_COMPATIBLE inline constexpr auto AbstractExpr<E, Row, Col, Depth, Time>::j() const {
        return SubMatrixExpr<const E, 1, 1, 1, 1>{static_cast<const E&>(*this), 2, 0, 0, 0};
    }

    template<class E, uint32_t Row, uint32_t Col, uint32_t Depth, uint32_t Time>
    [[nodiscard]]
    CUDA_COMPATIBLE inline constexpr auto AbstractExpr<E, Row, Col, Depth, Time>::k() {
        return SubMatrixExpr<E, 1, 1, 1, 1>{static_cast<E&>(*this), 3, 0, 0, 0};
    }

    template<class E, uint32_t Row, uint32_t Col, uint32_t Depth, uint32_t Time>
    [[nodiscard]]
    CUDA_COMPATIBLE inline constexpr auto AbstractExpr<E, Row, Col, Depth, Time>::k() const {
        return SubMatrixExpr<const E, 1, 1, 1, 1>{static_cast<const E&>(*this), 3, 0, 0, 0};
    }



    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////







    template<ScalarType T>
    class zero : public AbstractExpr<zero<T>, 1, 1, 1, 1> {
    public:

        static constexpr bool __is_variable_data = false;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr zero() {}

        template<VarIDType diffVarId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert(diffVarId > 0, "Variable IDs must be positive.");
            return zero<T>();
        }

        [[nodiscard]]
        CUDA_HOST constexpr inline std::string to_string() const {
            return std::string("0");
        }

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval_at(uint32_t r = 0, uint32_t c = 0, uint32_t d = 0, uint32_t t = 0) const {
            return T{};
        }
    };

    // Specialized trait to check if a type is a zero specialization
    template<ExprType T>
    inline constexpr bool is_zero_v = false;

    template<ScalarType T>
    inline constexpr bool is_zero_v<zero<T>> = true;

    using fzero = zero<float>;
    using dzero = zero<double>;






    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////






    template<ScalarType T, uint32_t N>
    class identity : public AbstractExpr<identity<T, N>, N, N> {
    public:

        static constexpr bool __is_variable_data = false;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr identity() {}

        template<VarIDType diffVarId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert(diffVarId > 0, "Variable IDs must be positive.");
            return zero<T>();
        }

        [[nodiscard]]
        CUDA_HOST constexpr inline std::string to_string() const {
            if constexpr (N == 1) {
                return std::string("1");
            }
            else {
                return std::string("I");
            }
        }

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval_at(uint32_t r = 0, uint32_t c = 0, uint32_t d = 0, uint32_t t = 0) const {
            return (r == c) ? T{ 1 } : T{};
        }
    };


    template<ScalarType T, uint32_t Row, uint32_t Col>
    class identityTensor : public AbstractExpr<identityTensor<T, Row, Col>, Row, Col, Row, Col> {
    public:

        static constexpr bool __is_variable_data = false;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr identityTensor() {}

        template<VarIDType diffVarId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert(diffVarId > 0, "Variable IDs must be positive.");
            return zero<T>();
        }

        [[nodiscard]]
        CUDA_HOST constexpr inline std::string to_string() const {
            if constexpr (Row == 1 && Col == 1) {
                return std::string("1");
            }
            else {
                return std::string("I");
            }
        }

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval_at(uint32_t r, uint32_t c, uint32_t dr, uint32_t dc) const {
            return (r == dr && c == dc) ? T{ 1 } : T{};
        }
    };

    // Specialized trait to check if a type is a Identity specialization
    template<class T>
    inline constexpr bool is_identity_v = false;

    template<ScalarType T, uint32_t N>
    inline constexpr bool is_identity_v<identity<T, N>> = true;

    template<ScalarType T, uint32_t R, uint32_t C>
    inline constexpr bool is_identity_v<identityTensor<T, R, C>> = true;

    using funit = identity<float, 1>;
    using fidentity1 = identity<float, 1>;
    using fidentity2 = identity<float, 2>;
    using fidentity3 = identity<float, 3>;
    using fidentity4 = identity<float, 4>;

    using dunit = identity<double, 1>;
    using didentity1 = identity<double, 1>;
    using didentity2 = identity<double, 2>;
    using didentity3 = identity<double, 3>;
    using didentity4 = identity<double, 4>;

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////











    template<ScalarType T, uint32_t Row, uint32_t Col>
    class ones : public AbstractExpr<ones<T, Row, Col>, Row, Col> {
    public:

        static constexpr bool __is_variable_data = false;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr ones() {}

        template<VarIDType diffVarId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert(diffVarId > 0, "Variable IDs must be positive.");
            return zero<T>();
        }

        [[nodiscard]]
        CUDA_HOST constexpr inline std::string to_string() const {
            if constexpr (Row == 1 && Col == 1) {
                return std::string("1");
            }
            else if constexpr (Row == 1) {
                std::stringstream strStream;
                strStream << "[ ";
                for (size_t c = 0; c < Col; ++c) {
                    strStream << "1";
                    if (c < Col - 1) {
                        strStream << " ";
                    }
                }
                strStream << " ]";
                return strStream.str();
            }
            else if constexpr (Col == 1) {
                std::stringstream strStream;
                for (size_t r = 0; r < Row; ++r) {
                    if (r < Row - 1) {
                        strStream << "| 1 |\n";
                    }
                    else {
                        strStream << "| 1 |";
                    }
                }
                return strStream.str();
            }
            else {
                std::stringstream strStream;
                for (size_t r = 0; r < Row; ++r) {
                    strStream << "| ";
                    for (size_t c = 0; c < Col; ++c) {
                        strStream << "1 ";
                    }
                    strStream << "|";
                    if (r < Row - 1) {
                        strStream << "\n";
                    }
                }
                return strStream.str();
            }
        }

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval_at(uint32_t r = 0, uint32_t c = 0) const {
            return T{ 1 };
        }

    };

    // Specialized trait to check if a type is a Ones specialization
    template<class T>
    inline constexpr bool is_ones_v = false;

    template<ScalarType T, uint32_t R, uint32_t C>
    inline constexpr bool is_ones_v<ones<T, R, C>> = true;


    using one = ones<float, 1, 1>;
    using ones2 = ones<float, 2, 2>;
    using ones3 = ones<float, 3, 3>;
    using ones4 = ones<float, 4, 4>;






    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////








    template<ScalarType T, uint32_t Row, uint32_t Col, uint32_t Depth, uint32_t Time>
    class FilledTensor : public AbstractExpr<FilledTensor<T, Row, Col, Depth, Time>, Row, Col, Depth, Time> {
    public:

        CUDA_COMPATIBLE inline constexpr FilledTensor(T value) : m_value(value) {}

        template<VarIDType varId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert(varId > 0, "Variable IDs must be positive.");
            return zero<T>{};
        }

        [[nodiscard]]
        CUDA_HOST constexpr inline std::string to_string() const {
            return std::format("{}", m_value);
        }

        static constexpr bool __is_variable_data = false;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval_at(uint32_t r = 0, uint32_t c = 0, uint32_t d = 0, uint32_t t = 0) const {
            return m_value;
        }

    private:
        T m_value;
    };

    template<ExprType E, uint32_t Row, uint32_t Col, uint32_t Depth, uint32_t Time> requires(is_scalar_shape_v<E>)
    class BroadcastScalarExpr : public AbstractExpr<BroadcastScalarExpr<E, Row, Col, Depth, Time>, Row, Col, Depth, Time> {
    public:

        CUDA_COMPATIBLE inline constexpr BroadcastScalarExpr(const E& expr) : m_expr(expr) {}

        template<VarIDType varId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert(varId > 0, "Variable IDs must be positive.");
            return BroadcastScalarExpr<decltype(m_expr.derivate<varId>()), Row, Col, Depth, Time>{ m_expr.derivate<varId>() };
        }

        [[nodiscard]]
        CUDA_HOST constexpr inline std::string to_string() const {
            return std::format("broadcast({})", m_expr.to_string());
        }

        static constexpr bool __is_variable_data = false;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval_at(uint32_t r = 0, uint32_t c = 0, uint32_t d = 0, uint32_t t = 0) const {
            return m_expr.eval_at(0, 0, 0, 0);
        }

    private:
        std::conditional_t< (E::__is_variable_data), const E&, const E> m_expr;
    };

    template<ExprType Like, ExprType E>
    class RepeatAlongExcessDimensionExpr : public AbstractExpr<RepeatAlongExcessDimensionExpr<Like, E>, Like::rows, Like::cols, Like::depth, Like::time> {
    public:

        CUDA_COMPATIBLE inline constexpr RepeatAlongExcessDimensionExpr(const E& expr) : m_expr(expr) {}

        template<VarIDType varId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert(varId > 0, "Variable IDs must be positive.");
            Like like{};
            return RepeatAlongExcessDimensionExpr<decltype(m_expr.derivate<varId>()), decltype(like.derivate<varId>())>{ m_expr.derivate<varId>() };
        }

        [[nodiscard]]
        CUDA_HOST constexpr inline std::string to_string() const {
            return std::format("repeat_along_excess_dimension({})", m_expr.to_string());
        }

        static constexpr bool __is_variable_data = false;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval_at(uint32_t r = 0, uint32_t c = 0, uint32_t d = 0, uint32_t t = 0) const {
            return m_expr.eval_at(r % m_expr.rows, c % m_expr.cols, d % m_expr.depth, t % m_expr.time);
        }

    private:
        std::conditional_t< (E::__is_variable_data), const E&, const E> m_expr;
    };

    template<ExprType Like, ExprType E>
    CUDA_COMPATIBLE
    [[nodiscard]]
    constexpr auto repeatAlongExcess(const Like& like, const E& expr) -> std::conditional_t<
        (is_scalar_shape_v<E> && is_scalar_shape_v<Like> || is_eq_shape_v<Like, E>),
            const E&,  
            decltype(RepeatAlongExcessDimensionExpr<Like, E>{ expr })
        > {
        static_assert(
            (Like::rows > E::rows && E::rows == 1 || Like::rows == E::rows)
            && (Like::cols > E::cols && E::cols == 1 || Like::cols == E::cols)
            && (Like::depth > E::depth && E::depth == 1 || Like::depth == E::depth)
            && (Like::time > E::time && E::time == 1 || Like::time == E::time),
            "The 'like' expression must have dimensions greater than or equal to the 'expr' expression."
        );
        if constexpr (is_scalar_shape_v<E> && is_scalar_shape_v<Like> || is_eq_shape_v<Like, E>) {
            return expr;
        }
        else {
            return RepeatAlongExcessDimensionExpr<Like, E>{ expr };
        }
    }


    //---------------------------------------
    // Math constants:

    /*
        Pi constant
    */
    template <ScalarType T>
    constexpr auto pi = FilledTensor<T, 1, 1, 1, 1>{ 3.14159265358979323846264338327950288419716939937510582097494459230781640628 };

    /*
        Euler number
    */
    template <ScalarType T>
    constexpr auto euler = FilledTensor<T, 1, 1, 1, 1>{ 2.718281828459045235360287471352662497757247093699959574966967627724076630353 };



    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////






    template<ScalarType T, uint32_t Row, uint32_t Col, VarIDType varId, StorageStrategy Storage>
    class VariableMatrix : public AbstractExpr<VariableMatrix<T, Row, Col, varId, Storage>, Row, Col> {
    public:

        static constexpr bool __is_variable_data = true;
        static constexpr VarIDType variable_id = varId;
        static constexpr StorageStrategy storage_strategy = Storage;

        [[nodiscard]]
        CUDA_COMPATIBLE static inline constexpr auto identity() {
            static_assert(Row == Col, "Identity matrix can only be created for square matrices.");
            auto M = VariableMatrix<T, Row, Col, varId>{};
            for (uint32_t c{}; c < Col; ++c) {
                    M.m_data[c][c] = static_cast<T>(1);
            }
            return M;
        }

        [[nodiscard]]
        CUDA_COMPATIBLE static inline constexpr auto filled(T valueToFillWith) {
            auto M = VariableMatrix<T, Row, Col, varId>{};
            if constexpr (storage_strategy == StorageStrategy::ColumnMajor) {
                if (valueToFillWith != T{}) {
                    for (uint32_t c{}; c < Col; ++c) {
                        for (uint32_t r{}; r < Row; ++r) {
                            M.m_data[c][r] = valueToFillWith;
                        }
                    }
                }
            }
            else if constexpr (storage_strategy == StorageStrategy::RowMajor) {
                for (uint32_t r{}; r < Row; ++r) {
                    for (uint32_t c{}; c < Col; ++c) {
                        M.m_data[r][c] = valueToFillWith;
                    }
                }
            }
            return M;
        }

        [[nodiscard]]
        CUDA_COMPATIBLE static inline constexpr auto ones() {
            auto M = VariableMatrix<T, Row, Col, varId>{};
            if constexpr (storage_strategy == StorageStrategy::ColumnMajor) {
                for (uint32_t c{}; c < Col; ++c) {
                    for (uint32_t r{}; r < Row; ++r) {
                        M.m_data[c][r] = static_cast<T>(1);
                    }
                }
            }
            else if constexpr (storage_strategy == StorageStrategy::RowMajor) {
                for (uint32_t r{}; r < Row; ++r) {
                    for (uint32_t c{}; c < Col; ++c) {
                        M.m_data[r][c] = static_cast<T>(1);
                    }
                }
            }
            return M;
        }

        [[nodiscard]]
        CUDA_HOST static inline constexpr auto random(T minValue, T maxValue) {
            static std::default_random_engine rng(42); // Fixed seed for reproducibility
            std::uniform_real_distribution<T> dist(minValue, maxValue);
            auto M = VariableMatrix<T, Row, Col, varId>{};
            if constexpr (storage_strategy == StorageStrategy::RowMajor) {
                for (uint32_t r{}; r < Row; ++r) {
                    for (uint32_t c{}; c < Col; ++c) {
                        M.m_data[r][c] = dist(rng);
                    }
                }
            }
            else if constexpr (storage_strategy == StorageStrategy::ColumnMajor) {
                for (uint32_t c{}; c < Col; ++c) {
                    for (uint32_t r{}; r < Row; ++r) {
                        M.m_data[c][r] = dist(rng);
                    }
                }
            }
            return M;
        }

        template<ExprType QE> requires(QE::__is_quaternion_valued && QE::rows == 4 && QE::cols == 1)
        CUDA_COMPATIBLE
        [[nodiscard]] static constexpr inline auto rotation_matrix(const QE& q) {
            static_assert(VariableMatrix::rows == 3 && VariableMatrix::cols == 3, "Rotation matrix can only be computed for 3x3 matrices.");
            VariableMatrix R;
            T w = q.real();
            T x = q.i();
            T y = q.j();
            T z = q.k();

            R.at(0,0) = 1 - 2 * (y * y + z * z);
            R.at(0,1) = 2 * (x * y - z * w);
            R.at(0,2) = 2 * (x * z + y * w);

            R.at(1,0) = 2 * (x * y + z * w);
            R.at(1,1) = 1 - 2 * (x * x + z * z);
            R.at(1,2) = 2 * (y * z - x * w);

            R.at(2,0) = 2 * (x * z - y * w);
            R.at(2,1) = 2 * (y * z + x * w);
            R.at(2,2) = 1 - 2 * (x * x + y * y);

            return R;
        }

        template<ExprType VE> requires(!VE::__is_quaternion_valued && VE::rows == 3 && VE::cols == 1)
        CUDA_HOST
        [[nodiscard]] static constexpr inline auto rotation_matrix(const VE& euler_xyz_rad) {
            static_assert(VariableMatrix::rows == 3 && VariableMatrix::cols == 3, "Rotation matrix can only be computed for 3x3 matrices.");
            T roll = euler_xyz_rad.at(0, 0);
            T pitch = euler_xyz_rad.at(1, 0);
            T yaw = euler_xyz_rad.at(2, 0);

            T cy = std::cos(yaw * T{0.5});
            T sy = std::sin(yaw * T{0.5});
            T cp = std::cos(pitch * T{0.5});
            T sp = std::sin(pitch * T{0.5});
            T cr = std::cos(roll * T{0.5});
            T sr = std::sin(roll * T{0.5});

            T w = cr * cp * cy + sr * sp * sy;
            T x = sr * cp * cy - cr * sp * sy;
            T y = cr * sp * cy + sr * cp * sy;
            T z = cr * cp * sy - sr * sp * cy;

            VariableMatrix R;

            R.at(0,0) = 1 - 2 * (y * y + z * z);
            R.at(0,1) = 2 * (x * y - z * w);
            R.at(0,2) = 2 * (x * z + y * w);

            R.at(1,0) = 2 * (x * y + z * w);
            R.at(1,1) = 1 - 2 * (x * x + z * z);
            R.at(1,2) = 2 * (y * z - x * w);

            R.at(2,0) = 2 * (x * z - y * w);
            R.at(2,1) = 2 * (y * z + x * w);
            R.at(2,2) = 1 - 2 * (x * x + y * y);

            return R;
        }

        template<ExprType QE> requires(QE::__is_quaternion_valued && QE::rows == 4 && QE::cols == 1)
        CUDA_HOST
        [[nodiscard]] static constexpr inline auto euler_angles(const QE& q) {
            static_assert(VariableMatrix::rows == 3 && VariableMatrix::cols == 1, "Euler angles can only be computed for 3x1 column vectors.");
            VariableMatrix euler_xyz_rad;
            T w = q.real();
            T x = q.i();
            T y = q.j();
            T z = q.k();

            // Roll (x-axis rotation)
            T sinr_cosp = 2 * (w * x + y * z);
            T cosr_cosp = 1 - 2 * (x * x + y * y);
            euler_xyz_rad.at(0, 0) = std::atan2(sinr_cosp, cosr_cosp);
            // Pitch (y-axis rotation)
            T sinp = 2 * (w * y - z * x);
            if (std::abs(sinp) >= 1)
                euler_xyz_rad.at(1, 0) = std::copysign(pi<T> / 2.0, sinp); // use 90 degrees if out of range
            else
                euler_xyz_rad.at(1, 0) = std::asin(sinp);
            // Yaw (z-axis rotation)
            T siny_cosp = 2 * (w * z + x * y);
            T cosy_cosp = 1 - 2 * (y * y + z * z);
            euler_xyz_rad.at(2, 0) = std::atan2(siny_cosp, cosy_cosp);

            return euler_xyz_rad;
        }

        template<ExprType ME> requires(is_matrix_shape_v<ME> && ME::rows == 3 && ME::cols == 3)
        CUDA_HOST
        [[nodiscard]] static constexpr inline auto euler_angles(const ME& rotation_matrix) {
            static_assert(VariableMatrix::rows == 3 && VariableMatrix::cols == 1, "Euler angles can only be computed for 3x1 column vectors.");
            VariableMatrix euler_xyz_rad;

            T sy = std::sqrt(rotation_matrix.eval_at(0,0) * rotation_matrix.eval_at(0,0) + rotation_matrix.eval_at(1,0) * rotation_matrix.eval_at(1,0));

            bool singular = sy < 1e-6; // If

            if (!singular) {
                euler_xyz_rad.at(0, 0) = std::atan2(rotation_matrix.eval_at(2,1), rotation_matrix.eval_at(2,2)); // Roll
                euler_xyz_rad.at(1, 0) = std::atan2(-rotation_matrix.eval_at(2,0), sy); // Pitch
                euler_xyz_rad.at(2, 0) = std::atan2(rotation_matrix.eval_at(1,0), rotation_matrix.eval_at(0,0)); // Yaw
            }
            else {
                euler_xyz_rad.at(0, 0) = std::atan2(-rotation_matrix.eval_at(1,2), rotation_matrix.eval_at(1,1)); // Roll
                euler_xyz_rad.at(1, 0) = std::atan2(-rotation_matrix.eval_at(2,0), sy); // Pitch
                euler_xyz_rad.at(2, 0) = 0; // Yaw
            }

            return euler_xyz_rad;
        }


        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr VariableMatrix() : m_data{} {}

        template<ExprType _SE>
        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr VariableMatrix(const AbstractExpr<_SE, Row, Col>& expr) {
            static_assert(!is_tensor_v<_SE>, "A matrix can not be initialized with a tensor.");
            if constexpr (storage_strategy == StorageStrategy::RowMajor) {
                for (size_t r = 0; r < (*this).rows; ++r) {
                    for (size_t c = 0; c < (*this).cols; ++c) {
                        m_data[r][c] = expr.eval_at(r, c, 0, 0);
                    }
                }
            }
            else if constexpr (storage_strategy == StorageStrategy::ColumnMajor) {
                for (size_t c = 0; c < (*this).cols; ++c) {
                    for (size_t r = 0; r < (*this).rows; ++r) {
                        m_data[c][r] = expr.eval_at(r, c, 0, 0);
                    }
                }
            }
        }

        // Constructor for matrices (not vectors) - nested initializer lists
        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr VariableMatrix(const std::initializer_list<std::initializer_list<T>>& values)
            requires(Col > 1 && Row > 1) : m_data{} {
            size_t r = 0;   // Despite the column-major storage, we fill row by row from the initializer list
            for (const auto& row : values) {
                size_t c = 0;
                for (const auto& val : row) {
                    if constexpr (storage_strategy == StorageStrategy::RowMajor) {
                        m_data[r][c] = val;
                    }
                    else if constexpr (storage_strategy == StorageStrategy::ColumnMajor) {
                        m_data[c][r] = val;
                    }
                    c++;
                    if (c >= Col) {
                        break;
                    }
                }
                for (; c < Col; ++c) {
                    if constexpr (storage_strategy == StorageStrategy::RowMajor) {
                        m_data[r][c] = T{};
                    }
                    else if constexpr (storage_strategy == StorageStrategy::ColumnMajor) {
                        m_data[c][r] = T{};
                    }
                }
                r++;
                if (r >= Row) {
                    break;
                }
            }
            for (; r < Row; ++r) {
                for (size_t c = 0; c < Col; ++c) {
                    m_data[c][r] = T{};
                }
            }
        }

        // Constructor for vectors - single initializer list
        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr VariableMatrix(const std::initializer_list<T>& values)
            requires(Col == 1 || Row == 1) : m_data{} {
            if constexpr (Col == 1) {
                size_t r = 0;
                for (const auto& val : values) {
                    if constexpr (storage_strategy == StorageStrategy::RowMajor) {
                        m_data[r][0] = val;
                    }
                    else if constexpr (storage_strategy == StorageStrategy::ColumnMajor) {
                        m_data[0][r] = val;
                    }
                    r++;
                    if (r >= Row) {
                        break;
                    }
                }
                for (; r < Row; ++r) {
                    if constexpr (storage_strategy == StorageStrategy::RowMajor) {
                        m_data[r][0] = T{};
                    }
                    else if constexpr (storage_strategy == StorageStrategy::ColumnMajor) {
                        m_data[0][r] = T{};
                    }
                }
            }
            else if constexpr (Row == 1) {
                size_t c = 0;
                for (const auto& val : values) {
                    if constexpr (storage_strategy == StorageStrategy::RowMajor) {
                        m_data[0][c] = val;
                    }
                    else if constexpr (storage_strategy == StorageStrategy::ColumnMajor) {
                        m_data[c][0] = val;
                    }
                    c++;
                    if (c >= Col) {
                        break;
                    }
                }
                for (; c < Col; ++c) {
                    if constexpr (storage_strategy == StorageStrategy::RowMajor) {
                        m_data[0][c] = T{};
                    }
                    else if constexpr (storage_strategy == StorageStrategy::ColumnMajor) {
                        m_data[c][0] = T{};
                    }
                }
            }
        }

        template<ExprType _SE>
        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto operator=(const AbstractExpr<_SE, Row, Col>& expr) {
            static_assert(!is_tensor_v<_SE>, "No tensor allowed.");
            for (uint32_t c = 0; c < Col; ++c) {
                for (uint32_t r = 0; r < Row; ++r) {
                    if constexpr (storage_strategy == StorageStrategy::RowMajor) {
                        m_data[r][c] = expr.eval_at(r, c, 0, 0);
                    }
                    else if constexpr (storage_strategy == StorageStrategy::ColumnMajor) {
                        m_data[c][r] = expr.eval_at(r, c, 0, 0);
                    }
                }
            }
            return *this;
        }

        template<ScalarType S>
        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto operator=(S value) requires(Row == 1 && Col == 1) {
            m_data[0][0] = value;
            return *this;
        }

        template<ExprType _SE>
        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto operator+=(const AbstractExpr<_SE, Row, Col>& expr) {
            static_assert(!is_tensor_v<_SE>, "No tensor allowed.");
            for (uint32_t c = 0; c < Col; ++c) {
                for (uint32_t r = 0; r < Row; ++r) {
                    if constexpr (storage_strategy == StorageStrategy::RowMajor) {
                        m_data[r][c] += expr.eval_at(r, c, 0, 0);
                    }
                    else if constexpr (storage_strategy == StorageStrategy::ColumnMajor) {
                        m_data[c][r] += expr.eval_at(r, c, 0, 0);
                    }
                }
            }
            return *this;
        }

        template<ExprType _SE>
        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto operator-=(const AbstractExpr<_SE, Row, Col>& expr) {
            static_assert(!is_tensor_v<_SE>, "No tensor allowed.");
            for (uint32_t c = 0; c < Col; ++c) {
                for (uint32_t r = 0; r < Row; ++r) {
                    if constexpr (storage_strategy == StorageStrategy::RowMajor) {
                        m_data[r][c] -= expr.eval_at(r, c, 0, 0);
                    }
                    else if constexpr (storage_strategy == StorageStrategy::ColumnMajor) {
                        m_data[c][r] -= expr.eval_at(r, c, 0, 0);
                    }
                }
            }
            return *this;
        }

        template<ScalarType S>
        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto operator+=(S value) requires(Row == 1 && Col == 1) {
            m_data[0][0] += value;
            return *this;
        }

        template<ScalarType S>
        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto operator-=(S value) requires(Row == 1 && Col == 1) {
            m_data[0][0] -= value;
            return *this;
        }

        template<ScalarType S>
        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto operator*=(S value) requires(Row == 1 && Col == 1) {
            m_data[0][0] *= value;
            return *this;
        }

        template<ScalarType S>
        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto operator/=(S value) requires(Row == 1 && Col == 1) {
            m_data[0][0] /= value;
            return *this;
        }

        template<VarIDType derivationVarId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert(derivationVarId > 0, "Variable IDs must be positive.");
            if constexpr (derivationVarId == varId) {
                return identityTensor<T, Row, Col>{};
            }
            else {
                return zero<T>{};
            }
        }

        [[nodiscard]]
        CUDA_HOST constexpr inline std::string to_string() const {
            if constexpr (is_scalar_shape_v<VariableMatrix>) {
                return (varId == 0) ? std::format("{}", m_data[0][0]) : std::format("scalar_{}", char32_to_utf8(varId));
            }
            else if constexpr (is_row_vector_v<VariableMatrix>) {
                std::stringstream strStream;
                strStream << "[ ";
                for (uint32_t c = 0; c < Col; ++c) {
                    if (c > 0) {
                        strStream << "  ";
                    }
                    strStream << m_data[c][0];
                }
                strStream << " ]";
                return (varId == 0) ? strStream.str() : std::format("row_vec{}_{}", (*this).cols, char32_to_utf8(varId));
            }
            else if constexpr (is_column_vector_v<VariableMatrix>) {
                std::stringstream strStream;
                strStream << std::endl;
                for (uint32_t r = 0; r < Row; ++r) {
                    strStream << "| " << m_data[0][r] << " |" << std::endl;
                }
                return (varId == 0) ? strStream.str() : std::format("col_vec{}_{}", (*this).rows, char32_to_utf8(varId));
            }
            else if constexpr (!is_tensor_v<VariableMatrix>) {
                std::stringstream strStream;
                strStream << std::endl;
                for (uint32_t r = 0; r < Row; ++r) {
                    strStream << "| ";
                    for (uint32_t c = 0; c < Col; ++c) {
                        if (c > 0) {
                            strStream << "  ";
                        }
                        if constexpr (storage_strategy == StorageStrategy::RowMajor) {
                            strStream << m_data[r][c];
                        }
                        else if constexpr (storage_strategy == StorageStrategy::ColumnMajor) {
                            strStream << m_data[c][r];
                        }
                    }
                    strStream << " |" << std::endl;
                }
                return (varId == 0) ? strStream.str() : std::format("mat{}x{}_{}", (*this).rows, (*this).cols, char32_to_utf8(varId));
            }
        }

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval_at(uint32_t r = 0, uint32_t c = 0, uint32_t dr = 0, uint32_t dc = 0) const {
            if constexpr (Row == 1 && Col == 1) {   // Behave as scalar
                return m_data[0][0];
            }
#ifndef ENABLE_CUDA_SUPPORT
            if (r >= Row || c >= Col) {
                throw std::out_of_range("Matrix index out of range.");
            }
#endif
            if constexpr (storage_strategy == StorageStrategy::RowMajor) {
                return m_data[r][c];
            }
            else if constexpr (storage_strategy == StorageStrategy::ColumnMajor) {
                return m_data[c][r];
            }
        }

        private:
        template<ScalarType S>
        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr void __assign_at_if_applicable(S value, uint32_t r, uint32_t c, uint32_t d = 0, uint32_t t = 0) {
            if constexpr (storage_strategy == StorageStrategy::RowMajor) {
                m_data[r][c] = value;
            }
            else if constexpr (storage_strategy == StorageStrategy::ColumnMajor) {
                m_data[c][r] = value;
            }
        }

        template<ExprType E, uint32_t Row, uint32_t Col, uint32_t Depth, uint32_t Time>
        friend class SubMatrixExpr;

        using StorageType = std::conditional_t<
            storage_strategy == StorageStrategy::ColumnMajor,
            T[Col][Row],  // Column-major storage: [col][row]
            std::conditional_t<
                storage_strategy == StorageStrategy::RowMajor,
                T[Row][Col],  // Row-major storage: [row][col]
                void*         // Placeholder for Sparse storage
            >
        >;
        StorageType m_data;
    };




    // Type alias:
    template<ScalarType T>
    using scal = VariableMatrix<T, 1, 1>;
    template<ScalarType T, uint32_t N>
    using vec = VariableMatrix<T, N, 1>;
    template<uint32_t N>
    using fvec = VariableMatrix<float, N, 1>;
    template<uint32_t N>
    using dvec = VariableMatrix<double, N, 1>;
    template<uint32_t N>
    using cfvec = VariableMatrix<std::complex<float>, N, 1>;
    template<uint32_t N>
    using cvec = VariableMatrix<std::complex<double>, N, 1>;
    template<ScalarType T, uint32_t R, uint32_t C>
    using mat = VariableMatrix<T, R, C>;
    template<uint32_t R, uint32_t C>
    using fmat = VariableMatrix<float, R, C>;
    template<uint32_t R, uint32_t C>
    using dmat = VariableMatrix<double, R, C>;
    template<uint32_t R, uint32_t C>
    using cfmat = VariableMatrix<std::complex<float>, R, C>;
    template<uint32_t R, uint32_t C>
    using cdmat = VariableMatrix<std::complex<double>, R, C>;
    using iscal = VariableMatrix<int32_t, 1, 1>;
    using ivec2 = VariableMatrix<int32_t, 2, 1>;
    using ivec3 = VariableMatrix<int32_t, 3, 1>;
    using ivec4 = VariableMatrix<int32_t, 4, 1>;
    using uscal = VariableMatrix<uint32_t, 1, 1>;
    using uvec2 = VariableMatrix<uint32_t, 2, 1>;
    using uvec3 = VariableMatrix<uint32_t, 3, 1>;
    using uvec4 = VariableMatrix<uint32_t, 4, 1>;
    using fscal = VariableMatrix<float, 1, 1>;
    using fvec2 = VariableMatrix<float, 2, 1>;
    using fvec3 = VariableMatrix<float, 3, 1>;
    using fvec4 = VariableMatrix<float, 4, 1>;
    using fmat2 = VariableMatrix<float, 2, 2>;
    using fmat3 = VariableMatrix<float, 3, 3>;
    using fmat4 = VariableMatrix<float, 4, 4>;
    using dscal = VariableMatrix<double, 1, 1>;
    using dvec2 = VariableMatrix<double, 2, 1>;
    using dvec3 = VariableMatrix<double, 3, 1>;
    using dvec4 = VariableMatrix<double, 4, 1>;
    using dvec2 = VariableMatrix<double, 2, 1>;
    using dmat2 = VariableMatrix<double, 2, 2>;
    using dmat3 = VariableMatrix<double, 3, 3>;
    using dmat4 = VariableMatrix<double, 4, 4>;
    using cfscal = VariableMatrix<std::complex<float>, 1, 1>;
    using cfvec2 = VariableMatrix<std::complex<float>, 2, 1>;
    using cfvec3 = VariableMatrix<std::complex<float>, 3, 1>;
    using cfvec4 = VariableMatrix<std::complex<float>, 4, 1>;
    using cfmat2 = VariableMatrix<std::complex<float>, 2, 2>;
    using cfmat3 = VariableMatrix<std::complex<float>, 3, 3>;
    using cfmat4 = VariableMatrix<std::complex<float>, 4, 4>;
    using cdscal = VariableMatrix<std::complex<double>, 1, 1>;
    using cdvec2 = VariableMatrix<std::complex<double>, 2, 1>;
    using cdvec3 = VariableMatrix<std::complex<double>, 3, 1>;
    using cdvec4 = VariableMatrix<std::complex<double>, 4, 1>;
    using cdmat2 = VariableMatrix<std::complex<double>, 2, 2>;
    using cdmat3 = VariableMatrix<std::complex<double>, 3, 3>;
    using cdmat4 = VariableMatrix<std::complex<double>, 4, 4>;
    using cscal = VariableMatrix<std::complex<double>, 1, 1>;
    using cscal = VariableMatrix<std::complex<double>, 1, 1>;
    using cvec2 = VariableMatrix<std::complex<double>, 2, 1>;
    using cvec3 = VariableMatrix<std::complex<double>, 3, 1>;
    using cmat2 = VariableMatrix<std::complex<double>, 2, 2>;
    using cmat3 = VariableMatrix<std::complex<double>, 3, 3>;
    using cmat4 = VariableMatrix<std::complex<double>, 4, 4>;

    template<ScalarType T, VarIDType varId>
    using scal_var = VariableMatrix<T, 1, 1, varId>;
    template<ScalarType T, uint32_t N, VarIDType varId>
    using vec_var = VariableMatrix<T, N, 1, varId>;
    template<ScalarType T, uint32_t R, uint32_t C, VarIDType varId>
    using mat_var = VariableMatrix<T, R, C, varId>;

    template<VarIDType varId>
    using fscal_var = VariableMatrix<float, 1, 1, varId>;
    template<VarIDType varId>
    using fvec2_var = VariableMatrix<float, 2, 1, varId>;
    template<VarIDType varId>
    using fvec3_var = VariableMatrix<float, 3, 1, varId>;
    template<VarIDType varId>
    using fvec4_var = VariableMatrix<float, 4, 1, varId>;
    template<VarIDType varId>
    using fmat2_var = VariableMatrix<float, 2, 2, varId>;
    template<VarIDType varId>
    using fmat3_var = VariableMatrix<float, 3, 3, varId>;
    template<VarIDType varId>
    using fmat4_var = VariableMatrix<float, 4, 4, varId>;
    template<VarIDType varId>
    using dscal_var = VariableMatrix<double, 1, 1, varId>;
    template<VarIDType varId>
    using dvec2_var = VariableMatrix<double, 2, 1, varId>;
    template<VarIDType varId>
    using dvec3_var = VariableMatrix<double, 3, 1, varId>;
    template<VarIDType varId>
    using dvec4_var = VariableMatrix<double, 4, 1, varId>;
    template<VarIDType varId>
    using dvec2_var = VariableMatrix<double, 2, 1, varId>;
    template<VarIDType varId>
    using dmat2_var = VariableMatrix<double, 2, 2, varId>;
    template<VarIDType varId>
    using dmat3_var = VariableMatrix<double, 3, 3, varId>;
    template<VarIDType varId>
    using dmat4_var = VariableMatrix<double, 4, 4, varId>;
    template<VarIDType varId>
    using cfscal_var = VariableMatrix<std::complex<float>, 1, 1, varId>;
    template<VarIDType varId>
    using cfvec2_var = VariableMatrix<std::complex<float>, 2, 1, varId>;
    template<VarIDType varId>
    using cfvec3_var = VariableMatrix<std::complex<float>, 3, 1, varId>;
    template<VarIDType varId>
    using cfvec4_var = VariableMatrix<std::complex<float>, 4, 1, varId>;
    template<VarIDType varId>
    using cfmat2_var = VariableMatrix<std::complex<float>, 2, 2, varId>;
    template<VarIDType varId>
    using cfmat3_var = VariableMatrix<std::complex<float>, 3, 3, varId>;
    template<VarIDType varId>
    using cfmat4_var = VariableMatrix<std::complex<float>, 4, 4, varId>;
    template<VarIDType varId>
    using cdscal_var = VariableMatrix<std::complex<double>, 1, 1, varId>;
    template<VarIDType varId>
    using cdvec2_var = VariableMatrix<std::complex<double>, 2, 1, varId>;
    template<VarIDType varId>
    using cdvec3_var = VariableMatrix<std::complex<double>, 3, 1, varId>;
    template<VarIDType varId>
    using cdvec4_var = VariableMatrix<std::complex<double>, 4, 1, varId>;
    template<VarIDType varId>
    using cdmat2_var = VariableMatrix<std::complex<double>, 2, 2, varId>;
    template<VarIDType varId>
    using cdmat3_var = VariableMatrix<std::complex<double>, 3, 3, varId>;
    template<VarIDType varId>
    using cdmat4_var = VariableMatrix<std::complex<double>, 4, 4, varId>;
    template<VarIDType varId>
    using cscal_var = VariableMatrix<std::complex<double>, 1, 1, varId>;
    template<VarIDType varId>
    using cscal_var = VariableMatrix<std::complex<double>, 1, 1, varId>;
    template<VarIDType varId>
    using cvec2_var = VariableMatrix<std::complex<double>, 2, 1, varId>;
    template<VarIDType varId>
    using cvec3_var = VariableMatrix<std::complex<double>, 3, 1, varId>;
    template<VarIDType varId>
    using cmat2_var = VariableMatrix<std::complex<double>, 2, 2, varId>;
    template<VarIDType varId>
    using cmat3_var = VariableMatrix<std::complex<double>, 3, 3, varId>;
    template<VarIDType varId>
    using cmat4_var = VariableMatrix<std::complex<double>, 4, 4, varId>;




    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////




    template<RealType T, VarIDType varId = 0>
    class Quaternion : public AbstractExpr<Quaternion<T, varId>, 4, 1> {
    public:

        // NOTE: Set to false so expression templates store quaternions by VALUE instead of by REFERENCE.
        // Quaternions are small (4 * sizeof(T)) and copying is cheap.
        // Storing by reference causes dangling reference issues when quaternions are temporaries
        // (e.g., from Quaternion::rotation()), especially in functions like rotate_vector_by_quaternion
        // where SubMatrixExpr from .real()/.imag() would reference destroyed temporaries.
        static constexpr bool __is_variable_data = true;
        static constexpr bool __is_quaternion_valued = true;
        static constexpr VarIDType variable_id = varId;


        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr Quaternion(T scalar = T{}, T i = T{}, T j = T{}, T k = T{}) : m_data{scalar, i, j, k} {}


        template<ExprType SE, ExprType VE>
        requires(is_scalar_v<SE> && is_column_vector_v<VE> && VE::rows == 3)
        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr Quaternion(const SE& scalar, const VE& vec) : m_data{scalar.eval_at(0, 0), vec.eval_at(0, 0), vec.eval_at(1, 0), vec.eval_at(2, 0)} {}


        template<ExprType VE>
        requires(is_column_vector_v<VE> && VE::rows == 3)
        [[nodiscard]]
        CUDA_HOST static inline constexpr auto rotation_around_axis(T angle_rad, const VE& axis) {
            T half_angle = angle_rad * static_cast<T>(0.5);
            T sin_half = std::sin(half_angle);
            T cos_half = std::cos(half_angle);
            T ax = axis.eval_at(0, 0);
            T ay = axis.eval_at(1, 0);
            T az = axis.eval_at(2, 0);
            T norm = std::sqrt(ax * ax + ay * ay + az * az);
            if (norm == T{}) {
                throw std::invalid_argument("Rotation axis cannot be the zero vector.");
            }
            ax /= norm;
            ay /= norm;
            az /= norm;
            return Quaternion<T, varId>{
                cos_half,
                ax * sin_half,
                ay * sin_half,
                az * sin_half
            };
        }


        template<ExprType VE> requires(is_column_vector_v<VE> && VE::rows == 3)
        [[nodiscard]]
        CUDA_COMPATIBLE static inline constexpr auto rotation_from_euler_angles(const VE& euler_xyz_rad) {
            auto ex = euler_xyz_rad.eval_at(0, 0);
            auto ey = euler_xyz_rad.eval_at(1, 0);
            auto ez = euler_xyz_rad.eval_at(2, 0);
            auto cy = std::cos(ez * static_cast<T>(0.5));
            auto sy = std::sin(ez * static_cast<T>(0.5));
            auto cp = std::cos(ey * static_cast<T>(0.5));
            auto sp = std::sin(ey * static_cast<T>(0.5));
            auto cr = std::cos(ex * static_cast<T>(0.5));
            auto sr = std::sin(ex * static_cast<T>(0.5));

            return Quaternion<T, varId>{
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy
            };
        }

        template<ExprType ME>
        requires(is_matrix_shape_v<ME> && ME::rows == 3 && ME::cols == 3)
        [[nodiscard]]
        CUDA_COMPATIBLE static inline constexpr auto rotation_from_rotation_matrix(const ME& R) {
            T trace = R.eval_at(0, 0) + R.eval_at(1, 1) + R.eval_at(2, 2);
            T w, x, y, z;
            if (trace > T{}) {
                T s = std::sqrt(trace + T{1}) * T{2};
                w = s * T{0.25};
                x = (R.eval_at(2, 1) - R.eval_at(1, 2)) / s;
                y = (R.eval_at(0, 2) - R.eval_at(2, 0)) / s;
                z = (R.eval_at(1, 0) - R.eval_at(0, 1)) / s;
            }
            else if ((R.eval_at(0, 0) > R.eval_at(1, 1)) && (R.eval_at(0, 0) > R.eval_at(2, 2))) {
                T s = std::sqrt(T{1} + R.eval_at(0, 0) - R.eval_at(1, 1) - R.eval_at(2, 2)) * T{2};
                w = (R.eval_at(2, 1) - R.eval_at(1, 2)) / s;
                x = s * T{0.25};
                y = (R.eval_at(0, 1) + R.eval_at(1, 0)) / s;
                z = (R.eval_at(0, 2) + R.eval_at(2, 0)) / s;
            }
            else if (R.eval_at(1, 1) > R.eval_at(2, 2)) {
                T s = std::sqrt(T{1} + R.eval_at(1, 1) - R.eval_at(0, 0) - R.eval_at(2, 2)) * T{2};
                w = (R.eval_at(0, 2) - R.eval_at(2, 0)) / s;
                x = (R.eval_at(0, 1) + R.eval_at(1, 0)) / s;
                y = s * T{0.25};
                z = (R.eval_at(1, 2) + R.eval_at(2, 1)) / s;
            }
            else {
                T s = std::sqrt(T{1} + R.eval_at(2, 2) - R.eval_at(0, 0) - R.eval_at(1, 1)) * T{2};
                w = (R.eval_at(1, 0) - R.eval_at(0, 1)) / s;
                x = (R.eval_at(0, 2) + R.eval_at(2, 0)) / s;
                y = (R.eval_at(1, 2) + R.eval_at(2, 1)) / s;
                z = s * T{0.25};
            }
            return Quaternion<T, varId>{w, x, y, z};
        }

        template<VarIDType derivationVarId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert(derivationVarId > 0, "Variable IDs must be positive.");
            if constexpr (derivationVarId == varId) {
                return identityTensor<T, 4, 1>{};
            }
            else {
                return zero<T>{};
            }
        }

        [[nodiscard]]
        CUDA_HOST constexpr inline std::string to_string() const {
            return (varId == 0) ? std::format("({}, {}i, {}j, {}k)", m_data[0], m_data[1], m_data[2], m_data[3])
                               : std::format("quat{}_({}, {}i, {}j, {}k)", char32_to_utf8(varId), m_data[0], m_data[1], m_data[2], m_data[3]);
        }

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval_at(uint32_t r = 0, uint32_t c = 0, uint32_t d = 0, uint32_t t = 0) const {
            if (r > 3 || c != 0 || d != 0 || t != 0) {
                throw std::out_of_range("Quaternion index out of range.");
            }
            return m_data[r];
        }


        template<ExprType VE> requires(is_column_vector_v<VE> && VE::rows == 4)
        CUDA_COMPATIBLE inline constexpr auto operator=(const VE& expr) {
            m_data[0] = expr.eval_at(0, 0, 0, 0);
            m_data[1] = expr.eval_at(1, 0, 0, 0);
            m_data[2] = expr.eval_at(2, 0, 0, 0);
            m_data[3] = expr.eval_at(3, 0, 0, 0);
            return *this;
        }

        template<ScalarType S>
        CUDA_COMPATIBLE inline constexpr auto operator=(S value) {
            m_data[0] = value;
            m_data[1] = 0;
            m_data[2] = 0;
            m_data[3] = 0;
            return *this;
        }

        template<ExprType VE> requires(is_column_vector_v<VE> && VE::rows == 4)
        CUDA_COMPATIBLE inline constexpr auto operator+=(const VE& expr) {
            m_data[0] += expr.eval_at(0, 0, 0, 0);
            m_data[1] += expr.eval_at(1, 0, 0, 0);
            m_data[2] += expr.eval_at(2, 0, 0, 0);
            m_data[3] += expr.eval_at(3, 0, 0, 0);
            return *this;
        }

        template<ExprType VE> requires(is_column_vector_v<VE> && VE::rows == 4)
        CUDA_COMPATIBLE inline constexpr auto operator-=(const VE& expr) {
            m_data[0] -= expr.eval_at(0, 0, 0, 0);
            m_data[1] -= expr.eval_at(1, 0, 0, 0);
            m_data[2] -= expr.eval_at(2, 0, 0, 0);
            m_data[3] -= expr.eval_at(3, 0, 0, 0);
            return *this;
        }


        private:
        template<ScalarType S>
        CUDA_COMPATIBLE inline constexpr void __assign_at_if_applicable(S value, uint32_t r, uint32_t c, uint32_t d = 0, uint32_t t = 0) {
            if (r > 3 || c != 0 || d != 0 || t != 0) {
                throw std::out_of_range("Quaternion index out of range.");
            }
            m_data[r] = value;
        }


        T m_data[4]; // [r, i, j, k]
    };

    using dquat = Quaternion<double, 0>;
    using fquat = Quaternion<float, 0>;
    template<VarIDType varId>
    using dquat_var = Quaternion<double, varId>;
    template<VarIDType varId>
    using fquat_var = Quaternion<float, varId>;





    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////









    template<ExprType E1, ExprType E2> requires(is_column_vector_v<E1> && is_column_vector_v<E2>
        && E1::rows == 4 && E2::rows == 4)
    class HamiltonProductExpr : public AbstractExpr<HamiltonProductExpr<E1, E2>, 4, 1> {
        public:

            CUDA_COMPATIBLE inline constexpr HamiltonProductExpr(const E1& expr1, const E2& expr2) : m_expr1(expr1), m_expr2(expr2) {
            }
            static constexpr bool __is_quaternion_valued = true;

            template<VarIDType varId>
            [[nodiscard]]
            CUDA_COMPATIBLE constexpr auto derivate() const {
                static_assert(false, "Differentitation of HamiltonProductExpr is unimplemented!");
                /*
                static_assert(varId > 0, "Variable ID for differentiation must be positive.");
                auto expr1_derivative = m_expr1.derivate<varId>();
                auto expr2_derivative = m_expr2.derivate<varId>();
                if constexpr (is_zero_v<decltype(expr1_derivative)>) {
                    return HamiltonProductExpr<E1, decltype(expr2_derivative)>{m_expr1, expr2_derivative};
                }
                else if constexpr (is_zero_v<decltype(expr2_derivative)>) {
                    return HamiltonProductExpr<decltype(expr1_derivative), E2>{expr1_derivative, m_expr2};
                }
                else {
                    return HamiltonProductExpr<decltype(expr1_derivative), E2>{expr1_derivative, m_expr2} +
                        HamiltonProductExpr<E1, decltype(expr2_derivative)>{m_expr1, expr2_derivative};
                }
                */
            }

            [[nodiscard]]
            CUDA_HOST constexpr inline std::string to_string() const {
                if constexpr (is_zero_v<E1> || is_zero_v<E2>) {
                    return "";
                }
                else {
                    auto str1 = std::string(m_expr1.to_string());
                    auto str2 = std::string(m_expr2.to_string());
                    if (!str1.empty() && !str2.empty()) {
                        return std::format("{} * {}", str1, str2);
                    }
                    else if (!str1.empty()) {
                        return str1;
                    }
                    else {
                        return str2;
                    }
                }
            }

            static constexpr bool __is_variable_data = false;

            [[nodiscard]]
            CUDA_COMPATIBLE inline constexpr auto eval_at(uint32_t r = 0, uint32_t c = 0, uint32_t d = 0, uint32_t t = 0) const {
                using common_type = common_arithmetic_t<decltype(m_expr1.eval_at(r, c, d, t)), decltype(m_expr2.eval_at(r, c, d, t))>;
                if constexpr (is_zero_v<E1> || is_zero_v<E2>) {
                    return common_type{};
                }
                else {
                    if (0 == r) {
                        return static_cast<common_type>(m_expr1.eval_at(0, c, d, t)) * static_cast<common_type>(m_expr2.eval_at(0, c, d, t))
                             - static_cast<common_type>(m_expr1.eval_at(1, c, d, t)) * static_cast<common_type>(m_expr2.eval_at(1, c, d, t))
                             - static_cast<common_type>(m_expr1.eval_at(2, c, d, t)) * static_cast<common_type>(m_expr2.eval_at(2, c, d, t))
                             - static_cast<common_type>(m_expr1.eval_at(3, c, d, t)) * static_cast<common_type>(m_expr2.eval_at(3, c, d, t));
                    }
                    else if (1 == r) {
                        return static_cast<common_type>(m_expr1.eval_at(0, c, d, t)) * static_cast<common_type>(m_expr2.eval_at(1, c, d, t))
                             + static_cast<common_type>(m_expr1.eval_at(1, c, d, t)) * static_cast<common_type>(m_expr2.eval_at(0, c, d, t))
                             + static_cast<common_type>(m_expr1.eval_at(2, c, d, t)) * static_cast<common_type>(m_expr2.eval_at(3, c, d, t))
                             - static_cast<common_type>(m_expr1.eval_at(3, c, d, t)) * static_cast<common_type>(m_expr2.eval_at(2, c, d, t));
                    }
                    else if (2 == r) {
                        return static_cast<common_type>(m_expr1.eval_at(0, c, d, t)) * static_cast<common_type>(m_expr2.eval_at(2, c, d, t))
                             - static_cast<common_type>(m_expr1.eval_at(1, c, d, t)) * static_cast<common_type>(m_expr2.eval_at(3, c, d, t))
                             + static_cast<common_type>(m_expr1.eval_at(2, c, d, t)) * static_cast<common_type>(m_expr2.eval_at(0, c, d, t))
                             + static_cast<common_type>(m_expr1.eval_at(3, c, d, t)) * static_cast<common_type>(m_expr2.eval_at(1, c, d, t));
                    }
                    else if (3 <= r) {
                        return static_cast<common_type>(m_expr1.eval_at(0, c, d, t)) * static_cast<common_type>(m_expr2.eval_at(3, c, d, t))
                             + static_cast<common_type>(m_expr1.eval_at(1, c, d, t)) * static_cast<common_type>(m_expr2.eval_at(2, c, d, t))
                             - static_cast<common_type>(m_expr1.eval_at(2, c, d, t)) * static_cast<common_type>(m_expr2.eval_at(1, c, d, t))
                             + static_cast<common_type>(m_expr1.eval_at(3, c, d, t)) * static_cast<common_type>(m_expr2.eval_at(0, c, d, t));       
                    }
                }
            }

        private:
            std::conditional_t< (E1::__is_variable_data), const E1&, const E1> m_expr1;
            std::conditional_t< (E2::__is_variable_data), const E2&, const E2> m_expr2;
    };

    

    template<ExprType E1, ExprType E2> requires(is_column_vector_v<E1> && is_column_vector_v<E2> && E1::rows == 4 && E2::rows == 4)
    CUDA_COMPATIBLE
    [[nodiscard]] constexpr auto operator*(const E1& expr1, const E2& expr2) {
        static_assert(is_column_vector_v<E1> && is_column_vector_v<E2> && E1::rows == 4 && E2::rows == 4, "Hamilton product is only defined for 4D column vectors (quaternions).");
        return HamiltonProductExpr<E1, E2>{expr1, expr2};
    }

    // Named function to explicitly perform Hamilton product (quaternion multiplication)
    // Use this to avoid operator* ambiguity in some contexts
    template<ExprType E1, ExprType E2> requires(is_column_vector_v<E1> && is_column_vector_v<E2> && E1::rows == 4 && E2::rows == 4)
    CUDA_COMPATIBLE
    [[nodiscard]] constexpr auto quat_mult(const E1& expr1, const E2& expr2) {
        return HamiltonProductExpr<E1, E2>{expr1, expr2};
    }





    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////







    template<ExprType E1, ExprType E2> requires(is_elementwise_broadcastable_v<E1, E2>)
        class AdditionExpr : public AbstractExpr<AdditionExpr<E1, E2>,
        std::conditional_t< (E1::rows > E2::rows), E1, E2>::rows,
        std::conditional_t< (E1::cols > E2::cols), E1, E2>::cols,
        std::conditional_t< (E1::depth > E2::depth), E1, E2>::depth,
        std::conditional_t< (E1::time > E2::time), E1, E2>::time
    > {
        public:

            CUDA_COMPATIBLE inline constexpr AdditionExpr(const E1& expr1, const E2& expr2) : m_expr1(expr1), m_expr2(expr2) {
            }

            template<VarIDType varId>
            [[nodiscard]]
            CUDA_COMPATIBLE constexpr auto derivate() const {
                static_assert(varId > 0, "Variable ID for differentiation must be positive.");
                auto expr1_derivative = m_expr1.derivate<varId>();
                auto expr2_derivative = m_expr2.derivate<varId>();
                if constexpr (is_zero_v<decltype(expr1_derivative)> && is_zero_v<decltype(expr2_derivative)>) {
                    return expr1_derivative;
                }
                else if constexpr (is_zero_v<decltype(expr1_derivative)>) {
                    return expr2_derivative;
                }
                else if constexpr (is_zero_v<decltype(expr2_derivative)>) {
                    return expr1_derivative;
                }
                else {
                    return AdditionExpr<
                        decltype(expr1_derivative),
                        decltype(expr2_derivative)
                    >{
                        expr1_derivative,
                        expr2_derivative
                    };
                }
            }

            [[nodiscard]]
            CUDA_HOST constexpr inline std::string to_string() const {
                if constexpr (is_zero_v<E1> && is_zero_v<E2>) {
                    return "";
                }
                else if constexpr (is_zero_v<E1>) {
                    return m_expr2.to_string();
                }
                else if constexpr (is_zero_v<E2>) {
                    return m_expr1.to_string();
                }
                else {
                    auto str1 = std::string(m_expr1.to_string());
                    auto str2 = std::string(m_expr2.to_string());
                    if (!str1.empty() && !str2.empty()) {
                        return std::format("{} + {}", str1, str2);
                    }
                    else if (!str1.empty()) {
                        return str1;
                    }
                    else {
                        return str2;
                    }
                }
            }

            static constexpr bool __is_variable_data = false;

            [[nodiscard]]
            CUDA_COMPATIBLE inline constexpr auto eval_at(uint32_t r = 0, uint32_t c = 0, uint32_t d = 0, uint32_t t = 0) const {
                using common_type = common_arithmetic_t<decltype(m_expr1.eval_at(r, c, d, t)), decltype(m_expr2.eval_at(r, c, d, t))>;
                if constexpr (is_zero_v<E1> && is_zero_v<E2>) {
                    return common_type{};
                }
                else if constexpr (is_zero_v<E1>) {
                    return static_cast<common_type>(m_expr2.eval_at(r, c, d, t));
                }
                else if constexpr (is_zero_v<E2>) {
                    return static_cast<common_type>(m_expr1.eval_at(r, c, d, t));
                }
                else {
                    return static_cast<common_type>(m_expr1.eval_at(r, c, d, t)) + static_cast<common_type>(m_expr2.eval_at(r, c, d, t));
                }
            }

        private:
            std::conditional_t< (E1::__is_variable_data), const E1&, const E1> m_expr1;
            std::conditional_t< (E2::__is_variable_data), const E2&, const E2> m_expr2;
    };

    template<ExprType E1, ExprType E2>
    CUDA_COMPATIBLE
    [[nodiscard]] constexpr auto operator+(const E1& expr1, const E2& expr2) {
        static_assert(is_elementwise_broadcastable_v<E1, E2>, "Incompatible matrix dimensions for element-wise addition.\nMatrices must have the same shape or one of them must be a scalar.");
        return AdditionExpr<E1, E2>{expr1, expr2};
    }

    template<ExprType E, ScalarType S>
    CUDA_COMPATIBLE
        [[nodiscard]] constexpr auto operator+(const E& expr, S a) {
        return AdditionExpr<E, FilledTensor<S, E::rows, E::cols, E::depth, E::time>>{expr, FilledTensor<S, E::rows, E::cols, E::depth, E::time>{a}};
    }

    template<ScalarType S, ExprType E>
    CUDA_COMPATIBLE
        [[nodiscard]] constexpr auto operator+(S a, const E& expr) {
        return AdditionExpr<FilledTensor<S, E::rows, E::cols, E::depth, E::time>, E>{FilledTensor<S, E::rows, E::cols, E::depth, E::time>{a}, expr};
    }





    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////








    template<ExprType E>
    class NegationExpr : public AbstractExpr<NegationExpr<E>, E::rows, E::cols, E::depth, E::time> {
    public:

        CUDA_COMPATIBLE inline constexpr NegationExpr(const E& expr) : m_expr(expr) {}

        template<VarIDType varId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert(varId > 0, "Variable ID for differentiation must be positive.");
            auto expr_derivative = m_expr.derivate<varId>();
            if constexpr (is_zero_v<decltype(expr_derivative)>) {
                return expr_derivative;
            }
            else {
                return NegationExpr<
                    decltype(expr_derivative)
                >{
                    expr_derivative
                };
            }
        }

        [[nodiscard]]
        CUDA_HOST constexpr inline std::string to_string() const {
            std::string str_expr = m_expr.to_string();
            if (str_expr.empty()) {
                return std::format("");
            }
            else {
                return std::format("-{}", str_expr);
            }
        }

        static constexpr bool __is_variable_data = false;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval_at(uint32_t r = 0, uint32_t c = 0, uint32_t d = 0, uint32_t t = 0) const {
            if constexpr (is_zero_v<E>) {
                return 0;
            }
            else {
                return -m_expr.eval_at(r, c, d, t);
            }
        }

    private:
        std::conditional_t< (E::__is_variable_data), const E&, const E> m_expr;
    };

    template<ExprType E>
    CUDA_COMPATIBLE
        [[nodiscard]] constexpr auto operator-(const E& expr) {
        return NegationExpr<E>{expr};
    }





    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////








    template<ExprType E>
    class TransposeExpr : public AbstractExpr<TransposeExpr<E>, E::cols, E::rows, E::depth, E::time> {
    public:

        CUDA_COMPATIBLE inline constexpr TransposeExpr(const E& expr) : m_expr(expr) {}

        template<VarIDType varId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert(varId > 0, "Variable ID for differentiation must be positive.");
            auto expr_derivative = m_expr.derivate<varId>();
            if constexpr (is_zero_v<decltype(expr_derivative)>) {
                return expr_derivative;
            }
            else {
                return TransposeExpr<
                    decltype(expr_derivative)
                >{
                    expr_derivative
                };
            }
        }

        [[nodiscard]]
        CUDA_HOST constexpr inline std::string to_string() const {
            std::string str_expr = m_expr.to_string();
            if (str_expr.empty()) {
                return std::format("");
            }
            else {
                return std::format("({})^T", str_expr);
            }
        }

        static constexpr bool __is_variable_data = false;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval_at(uint32_t r, uint32_t c, uint32_t d = 0, uint32_t t = 0) const {
            return m_expr.eval_at(c, r, d, t);
        }

    private:
        std::conditional_t< (E::__is_variable_data), const E&, const E> m_expr;
    };

    template<ExprType E>
    CUDA_COMPATIBLE
        [[nodiscard]] constexpr auto transpose(const E& expr) {
        return TransposeExpr<E>{expr};
    }

    template<ExprType E>
    CUDA_COMPATIBLE
        [[nodiscard]] constexpr auto T(const E& expr) {
        return TransposeExpr<E>{expr};
    }






    /////////////////////////////////////////////////////////////////////////////////////////////





    template<ExprType E>
    class SwapRowWithDepthExpr : public AbstractExpr<SwapRowWithDepthExpr<E>, E::depth, E::cols, E::rows, E::time> {
    public:

        CUDA_COMPATIBLE inline constexpr SwapRowWithDepthExpr(const E& expr) : m_expr(expr) {}

        template<VarIDType varId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert(varId > 0, "Variable ID for differentiation must be positive.");
            auto expr_derivative = m_expr.derivate<varId>();
            if constexpr (is_zero_v<decltype(expr_derivative)> || is_identity_v<decltype(expr_derivative)>) {
                return expr_derivative;
            }
            else {
                return SwapRowWithDepthExpr<
                    decltype(expr_derivative)
                >{
                    expr_derivative
                };
            }
        }

        [[nodiscard]]
        CUDA_HOST constexpr inline std::string to_string() const {
            std::string str_expr = m_expr.to_string();
            if (str_expr.empty()) {
                return std::format("");
            }
            else {
                return std::format("({})[r<->d]", str_expr);
            }
        }

        static constexpr bool __is_variable_data = false;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval_at(uint32_t r, uint32_t c, uint32_t d = 0, uint32_t t = 0) const {            
            return m_expr.eval_at(d, c, r, t);
        }

    private:
        std::conditional_t< (E::__is_variable_data), const E&, const E> m_expr;
    };

    template<ExprType E>
    CUDA_COMPATIBLE
        [[nodiscard]] constexpr auto swapRowWithDepth(const E& expr) {
        return SwapRowWithDepthExpr{expr};
    }

    template<ExprType E>
    class SwapColsWithDepthExpr : public AbstractExpr<SwapColsWithDepthExpr<E>, E::rows, E::depth, E::cols, E::time> {
    public:

        CUDA_COMPATIBLE inline constexpr SwapColsWithDepthExpr(const E& expr) : m_expr(expr) {}

        template<VarIDType varId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert(varId > 0, "Variable ID for differentiation must be positive.");
            auto expr_derivative = m_expr.derivate<varId>();
            if constexpr (is_zero_v<decltype(expr_derivative)> || is_identity_v<decltype(expr_derivative)>) {
                return expr_derivative;
            }
            else {
                return SwapRowWithDepthExpr<
                    decltype(expr_derivative)
                >{
                    expr_derivative
                };
            }
        }

        [[nodiscard]]
        CUDA_HOST constexpr inline std::string to_string() const {
            std::string str_expr = m_expr.to_string();
            if (str_expr.empty()) {
                return std::format("");
            }
            else {
                return std::format("({})[c<->d]", str_expr);
            }
        }

        static constexpr bool __is_variable_data = false;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval_at(uint32_t r, uint32_t c, uint32_t d = 0, uint32_t t = 0) const {            
            return m_expr.eval_at(r, d, c, t);
        }

    private:
        std::conditional_t< (E::__is_variable_data), const E&, const E> m_expr;
    };

    template<ExprType E>
    CUDA_COMPATIBLE
        [[nodiscard]] constexpr auto swapColsWithDepth(const E& expr) {
        return SwapColsWithDepthExpr{expr};
    }

    template<ExprType E>
    class SwapColsWithTimeExpr : public AbstractExpr<SwapColsWithTimeExpr<E>, E::rows, E::depth, E::cols, E::time> {
    public:

        CUDA_COMPATIBLE inline constexpr SwapColsWithTimeExpr(const E& expr) : m_expr(expr) {}

        template<VarIDType varId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert(varId > 0, "Variable ID for differentiation must be positive.");
            auto expr_derivative = m_expr.derivate<varId>();
            if constexpr (is_zero_v<decltype(expr_derivative)> || is_identity_v<decltype(expr_derivative)>) {
                return expr_derivative;
            }
            else {
                return SwapRowWithDepthExpr<
                    decltype(expr_derivative)
                >{
                    expr_derivative
                };
            }
        }

        [[nodiscard]]
        CUDA_HOST constexpr inline std::string to_string() const {
            std::string str_expr = m_expr.to_string();
            if (str_expr.empty()) {
                return std::format("");
            }
            else {
                return std::format("({})[c<->d]", str_expr);
            }
        }

        static constexpr bool __is_variable_data = false;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval_at(uint32_t r, uint32_t c, uint32_t d = 0, uint32_t t = 0) const {            
            return m_expr.eval_at(r, t, d, c);
        }

    private:
        std::conditional_t< (E::__is_variable_data), const E&, const E> m_expr;
    };

    template<ExprType E>
    class SwapRowsAndColsWithDepthAndTimeExpr : public AbstractExpr<SwapRowsAndColsWithDepthAndTimeExpr<E>, E::depth, E::time, E::rows, E::cols> {
    public:

        CUDA_COMPATIBLE inline constexpr SwapRowsAndColsWithDepthAndTimeExpr(const E& expr) : m_expr(expr) {}

        template<VarIDType varId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert(varId > 0, "Variable ID for differentiation must be positive.");
            auto expr_derivative = m_expr.derivate<varId>();
            if constexpr (is_zero_v<decltype(expr_derivative)> || is_identity_v<decltype(expr_derivative)>) {
                return expr_derivative;
            }
            else {
                return SwapRowsAndColsWithDepthAndTimeExpr<
                    decltype(expr_derivative)
                >{
                    expr_derivative
                };
            }
        }

        [[nodiscard]]
        CUDA_HOST constexpr inline std::string to_string() const {
            std::string str_expr = m_expr.to_string();
            if (str_expr.empty()) {
                return std::format("");
            }
            else {
                return std::format("({})[rc<->dt]", str_expr);
            }
        }

        static constexpr bool __is_variable_data = false;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval_at(uint32_t r, uint32_t c, uint32_t d = 0, uint32_t t = 0) const {            
            return m_expr.eval_at(d, t, r, c);
        }

    private:
        std::conditional_t< (E::__is_variable_data), const E&, const E> m_expr;
    };

    template<ExprType E>
    CUDA_COMPATIBLE
        [[nodiscard]] constexpr auto swapColsWithTime(const E& expr) {
        return SwapColsWithTimeExpr{expr};
    }

    template<VarIDType varId, ExprType E>
    CUDA_COMPATIBLE
    [[nodiscard]]
    constexpr auto gradient(const E& expr) {
        static_assert(is_scalar_shape_v<E>, "Gradient can only be computed for scalar expressions.");
        auto expr_derivative = expr.derivate<varId>();
        using derivative_type = decltype(expr_derivative);
        static_assert(expr_derivative.rows == 1 && expr_derivative.cols == 1 && expr_derivative.depth > 1 && expr_derivative.time == 1,
            "The derivative of the scaler expression must be a column vector along the depth dimension."
        );
        if constexpr (is_column_vector_v<derivative_type>) {
            return expr_derivative;
        }
        else if constexpr (is_row_vector_v<derivative_type>) {
            return transpose(expr_derivative);
        }
        else if constexpr (is_column_vector_v<E>)
            return SwapRowWithDepthExpr<derivative_type>{expr_derivative};
        else {
            return SwapColsWithDepthExpr<derivative_type>{expr_derivative};
        }

    }

    template<VarIDType varId, ExprType E>
    CUDA_COMPATIBLE
    [[nodiscard]]
    constexpr auto jacobian(const E& expr) {
        static_assert(is_vector_v<E>, "Jacobian can only be computed for vector expressions.");
        auto expr_derivative = expr.derivate<varId>();
        using derivative_type = decltype(expr_derivative);
        return SwapColsWithDepthExpr<derivative_type>{expr_derivative};
    }





    /////////////////////////////////////////////////////////////////////////////////////////////










    template<ExprType E> requires( is_vector_v<E> )
    class DiagonalMatrix : public AbstractExpr<DiagonalMatrix<E>,
        E::rows * E::cols,
        E::rows * E::cols
    > {
    public:

        CUDA_COMPATIBLE inline constexpr DiagonalMatrix(const E& expr, double nondiagonal_filler = 0.0)
            : m_expr{expr}, m_nondiagonal_filler{static_cast<decltype(m_nondiagonal_filler)>(nondiagonal_filler)} {}

        template<VarIDType varId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert(varId > 0, "Variable ID for differentiation must be positive.");
            auto expr_derivative = m_expr.derivate<varId>();
            if constexpr (is_zero_v<decltype(expr_derivative)>) {
                return expr_derivative;
            }
            else {
                return DiagonalMatrix<
                    decltype(expr_derivative)
                >{
                    expr_derivative
                };
            }
        }

        [[nodiscard]]
        CUDA_HOST constexpr inline std::string to_string() const {
            std::string str_expr = m_expr.to_string();
            if (str_expr.empty()) {
                return std::format("");
            }
            else {
                if (0 == m_nondiagonal_filler) {
                    return std::format("diagonal({})", str_expr);
                }
                else {
                    return std::format("(diagonal({} - {}) + {})", str_expr, m_nondiagonal_filler, m_nondiagonal_filler);
                }
            }
        }

        static constexpr bool __is_variable_data = false;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval_at(uint32_t r, uint32_t c, uint32_t d = 0, uint32_t t = 0) const {
            if (r == c) {
                if constexpr (is_column_vector_v<E>) {
                    return m_expr.eval_at(r, 0, d, t);
                }
                else {
                    return m_expr.eval_at(0, c, d, t);
                }
            }
            else {
                return m_nondiagonal_filler;  // Zero
            }
        }

    private:
        std::conditional_t< (E::__is_variable_data), const E&, const E> m_expr;
        decltype(m_expr.eval_at(0, 0)) m_nondiagonal_filler = decltype(m_expr.eval_at(0, 0)){};
    };






    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////








        template<ExprType E> requires( is_vector_v<E> )
    class DiagonalTensor : public AbstractExpr<DiagonalTensor<E>,
        E::rows,
        E::cols,
        E::rows,
        E::cols
    > {
    public:

        CUDA_COMPATIBLE inline constexpr DiagonalTensor(const E& expr, double nondiagonal_filler = 0.0)
            : m_expr{expr}, m_nondiagonal_filler{static_cast<decltype(m_nondiagonal_filler)>(nondiagonal_filler)} {}

        template<VarIDType varId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert(varId > 0, "Variable ID for differentiation must be positive.");
            auto expr_derivative = m_expr.derivate<varId>();
            if constexpr (is_zero_v<decltype(expr_derivative)>) {
                return expr_derivative;
            }
            else {
                static_assert(false, "Derivative of DiagonalTensor is not implemented.");
            }
        }

        [[nodiscard]]
        CUDA_HOST constexpr inline std::string to_string() const {
            std::string str_expr = m_expr.to_string();
            if (str_expr.empty()) {
                return std::format("");
            }
            else {
                if (0 == m_nondiagonal_filler) {
                    return std::format("diagonal({})", str_expr);
                }
                else {
                    return std::format("(diagonal({} - {}) + {})", str_expr, m_nondiagonal_filler, m_nondiagonal_filler);
                }
            }
        }

        static constexpr bool __is_variable_data = false;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval_at(uint32_t r, uint32_t c, uint32_t d = 0, uint32_t t = 0) const {
            if (r == d && c == t) {
                return m_expr.eval_at(r, c, 0, 0);
            }
            else {
                return m_nondiagonal_filler;  // Zero
            }
        }

    private:
        std::conditional_t< (E::__is_variable_data), const E&, const E> m_expr;
        decltype(m_expr.eval_at(0, 0)) m_nondiagonal_filler = decltype(m_expr.eval_at(0, 0)){};
    };






    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////








    template<ExprType E>
    class ConjugateExpr : public AbstractExpr<ConjugateExpr<E>, E::rows, E::cols, E::depth, E::time> {
    public:

        CUDA_COMPATIBLE inline constexpr ConjugateExpr(const E& expr) : m_expr(expr) {}

        template<VarIDType varId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert(varId > 0, "Variable ID for differentiation must be positive.");
            auto expr_derivative = m_expr.derivate<varId>();
            if constexpr (is_zero_v<decltype(expr_derivative)> || is_identity_v<decltype(expr_derivative)> || is_ones_v<decltype(expr_derivative)>) {
                return expr_derivative;
            }
            else {
                return ConjugateExpr<
                    decltype(expr_derivative)
                >{
                    expr_derivative
                };
            }
        }

        [[nodiscard]]
        CUDA_HOST constexpr inline std::string to_string() const {
            std::string str_expr = m_expr.to_string();
            if (str_expr.empty()) {
                return std::format("");
            }
            else {
                return std::format("conj({})", str_expr);
            }
        }

        static constexpr bool __is_variable_data = false;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval_at(uint32_t r = 0, uint32_t c = 0, uint32_t d = 0, uint32_t t = 0) const {
            if constexpr (ComplexType<decltype(m_expr.eval_at(r, c, d, t))>) {
                return conj(m_expr.eval_at(r, c, d, t));
            }
            else if constexpr (m_expr.__is_quaternion_valued) { // Works as quaternion conjugate 
                if (r > 0) {
                    return -m_expr.eval_at(r, c, d, t);
                }
                else {
                    return m_expr.eval_at(r, c, d, t);
                }
            }
            else {
                return m_expr.eval_at(r, c, d, t);
            }
        }

    private:
        std::conditional_t< (E::__is_variable_data), const E&, const E> m_expr;
    };

    template<ExprType E>
    CUDA_COMPATIBLE
        [[nodiscard]] constexpr auto conjugate(const E& expr) {
        return ConjugateExpr<E>{expr};
    }

    template<ExprType E>
    CUDA_COMPATIBLE
        [[nodiscard]] constexpr auto conj(const E& expr) {
        return ConjugateExpr<E>{expr};
    }





    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////







    template<ExprType E>
    class AdjointExpr : public AbstractExpr<AdjointExpr<E>, E::cols, E::rows, E::time, E::depth> {
    public:

        CUDA_COMPATIBLE inline constexpr AdjointExpr(const E& expr) : m_expr(expr) {}

        template<VarIDType varId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert(varId > 0, "Variable ID for differentiation must be positive.");
            auto expr_derivative = m_expr.derivate<varId>();
            if constexpr (is_zero_v<decltype(expr_derivative)> || is_identity_v<decltype(expr_derivative)>) {
                return expr_derivative;
            }
            else {
                return AdjointExpr<
                    decltype(expr_derivative)
                >{
                    expr_derivative
                };
            }
        }

        [[nodiscard]]
        CUDA_HOST constexpr inline std::string to_string() const {
            std::string str_expr = m_expr.to_string();
            if (str_expr.empty()) {
                return std::format("");
            }
            else {
                return std::format("({})†", str_expr);
            }
        }

        static constexpr bool __is_variable_data = false;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval_at(uint32_t r, uint32_t c, uint32_t d = 0, uint32_t t = 0) const {
            if constexpr (ComplexType<decltype(m_expr.eval_at(c, r, t, d))>) {
                return conj(m_expr.eval_at(c, r, t, d));
            }
            else {
                return m_expr.eval_at(c, r, t, d);
            }
        }

    private:
        std::conditional_t< (E::__is_variable_data), const E&, const E> m_expr;
    };

    template<ExprType E>
    CUDA_COMPATIBLE
        [[nodiscard]] constexpr auto adjoint(const E& expr) {
        return AdjointExpr<E>{expr};
    }

    template<ExprType E>
    CUDA_COMPATIBLE
        [[nodiscard]] constexpr auto adj(const E& expr) {
        return AdjointExpr<E>{expr};
    }




    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////








    template<ExprType E1, ExprType E2> requires(is_elementwise_broadcastable_v<E1, E2>)
        class SubtractionExpr : public AbstractExpr<SubtractionExpr<E1, E2>,
        std::conditional_t< (E1::rows > E2::rows), E1, E2>::rows,
        std::conditional_t< (E1::cols > E2::cols), E1, E2>::cols,
        std::conditional_t< (E1::depth > E2::depth), E1, E2>::depth,
        std::conditional_t< (E1::time > E2::time), E1, E2>::time
            > {
        public:

            CUDA_COMPATIBLE inline constexpr SubtractionExpr(const E1& expr1, const E2& expr2) : m_expr1(expr1), m_expr2(expr2) {
            }

            template<VarIDType varId>
            [[nodiscard]]
            CUDA_COMPATIBLE constexpr inline auto derivate() const {
                static_assert(varId > 0, "Variable ID for differentiation must be positive.");
                auto expr1_derivative = m_expr1.derivate<varId>();
                auto expr2_derivative = m_expr2.derivate<varId>();
                if constexpr (is_zero_v<decltype(expr1_derivative)> && is_zero_v<decltype(expr2_derivative)>) {
                    return expr1_derivative;
                }
                else if constexpr (is_zero_v<decltype(expr1_derivative)>) {
                    return NegationExpr<decltype(expr2_derivative)>{expr2_derivative};
                }
                else if constexpr (is_zero_v<decltype(expr2_derivative)>) {
                    return expr1_derivative;
                }
                else {
                    return SubtractionExpr<
                        decltype(expr1_derivative),
                        decltype(expr2_derivative)
                    >{
                        expr1_derivative,
                        expr2_derivative
                    };
                }
            }

            [[nodiscard]]
            CUDA_HOST constexpr inline std::string to_string() const {
                if constexpr (is_zero_v<E1> && is_zero_v<E2>) {
                    return "";
                }
                else if constexpr (is_zero_v<E1>) {
                    return std::format("-{}", m_expr2.to_string());
                }
                else if constexpr (is_zero_v<E2>) {
                    return m_expr1.to_string();
                }
                else {
                    auto str1 = std::string(m_expr1.to_string());
                    auto str2 = std::string(m_expr2.to_string());
                    if (!str1.empty() && !str2.empty()) {
                        return std::format("{} - {}", str1, str2);
                    }
                    else if (!str1.empty()) {
                        return std::format("{}", str1);
                    }
                    else {
                        return std::format("-{}", str2);
                    }
                }
            }

            static constexpr bool __is_variable_data = false;

            [[nodiscard]]
            CUDA_COMPATIBLE inline constexpr auto eval_at(uint32_t r = 0, uint32_t c = 0, uint32_t d = 0, uint32_t t = 0) const {
                using common_type = common_arithmetic_t<decltype(m_expr1.eval_at(r, c, d, t)), decltype(m_expr2.eval_at(r, c, d, t))>;
                if constexpr (is_zero_v<E1> && is_zero_v<E2>) {
                    return common_type{};
                }
                else if constexpr (is_zero_v<E1>) {
                    return static_cast<common_type>(-m_expr2.eval_at(r, c, d, t));
                }
                else if constexpr (is_zero_v<E2>) {
                    return static_cast<common_type>(m_expr1.eval_at(r, c, d, t));
                }
                else {
                    return static_cast<common_type>(m_expr1.eval_at(r, c, d, t)) - static_cast<common_type>(m_expr2.eval_at(r, c, d, t));
                }
            }

        private:
            std::conditional_t< (E1::__is_variable_data), const E1&, const E1> m_expr1;
            std::conditional_t< (E2::__is_variable_data), const E2&, const E2> m_expr2;
    };

    template<ExprType E1, ExprType E2>
    CUDA_COMPATIBLE
        [[nodiscard]] constexpr auto operator-(const E1& expr1, const E2& expr2) {
        static_assert(is_elementwise_broadcastable_v<E1, E2>, "Incompatible matrix dimensions for element-wise subtraction.\nMatrices must have the same shape or one of them must be a scalar.");
        return SubtractionExpr<E1, E2>{expr1, expr2};
    }

    template<ExprType E, ScalarType S>
    CUDA_COMPATIBLE
        [[nodiscard]] constexpr auto operator-(const E& expr, S a) {
        return SubtractionExpr<E, FilledTensor<S, E::rows, E::cols, E::depth, E::time>>{expr, FilledTensor<S, E::rows, E::cols, E::depth, E::time>{a}};
    }

    template<ScalarType S, ExprType E>
    CUDA_COMPATIBLE
        [[nodiscard]] constexpr auto operator-(S a, const E& expr) {
        return SubtractionExpr<FilledTensor<S, E::rows, E::cols, E::depth, E::time>, E>{FilledTensor<S, E::rows, E::cols, E::depth, E::time>{a}, expr};
    }




    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////








    template<ExprType E1, ExprType E2> requires(is_elementwise_broadcastable_v<E1, E2>)
        class elementwise_prodExpr : public AbstractExpr<elementwise_prodExpr<E1, E2>,
        std::conditional_t<(E1::rows > E2::rows), E1, E2>::rows,
        std::conditional_t<(E1::cols > E2::cols), E1, E2>::cols,
        std::conditional_t<(E1::depth > E2::depth), E1, E2>::depth,
        std::conditional_t<(E1::time > E2::time), E1, E2>::time
    > {
        public:

            CUDA_COMPATIBLE inline constexpr elementwise_prodExpr(const E1& expr1, const E2& expr2) : m_expr1(expr1), m_expr2(expr2) {
            }

            template<VarIDType varId>
            [[nodiscard]]
            CUDA_COMPATIBLE constexpr inline auto derivate() const {
                static_assert((varId > 0), "Variable ID for differentiation must be positive.");
                auto expr1_derivative = m_expr1.derivate<varId>();
                auto expr2_derivative = m_expr2.derivate<varId>();
                if constexpr (is_identity_v<decltype(expr1_derivative)> && is_identity_v<decltype(expr2_derivative)>) {
                    return AdditionExpr<std::remove_cvref_t<decltype(m_expr1)>, std::remove_cvref_t<decltype(m_expr2)>>{
                        m_expr1,
                        m_expr2
                    };
                }
                else if constexpr (is_zero_v<decltype(expr1_derivative)> && is_zero_v<decltype(expr2_derivative)>) {
                    return expr1_derivative;
                }
                else if constexpr (is_zero_v<decltype(expr1_derivative)>) {
                    return elementwise_prodExpr<
                        std::remove_cvref_t<decltype(repeatAlongExcess(expr2_derivative, m_expr1))>,
                        decltype(expr2_derivative)
                    >{
                        repeatAlongExcess(expr2_derivative, m_expr1),
                        expr2_derivative
                    };
                }
                else if constexpr (is_zero_v<decltype(expr2_derivative)>) {
                    return elementwise_prodExpr<
                        decltype(expr1_derivative),
                        std::remove_cvref_t<decltype(repeatAlongExcess(expr1_derivative, m_expr2))>
                    >{
                        expr1_derivative,
                        repeatAlongExcess(expr1_derivative, m_expr2)
                    };
                }
                else if constexpr (is_tensor_v<decltype(expr1_derivative)> || is_tensor_v<decltype(expr2_derivative)>) {
                    return AdditionExpr{
                        elementwise_prodExpr<
                            std::remove_cvref_t<decltype(repeatAlongExcess(expr2_derivative, m_expr1))>,
                            decltype(expr2_derivative)
                        > {
                            repeatAlongExcess(expr2_derivative, m_expr1),
                            expr2_derivative
                        },
                        elementwise_prodExpr<
                            decltype(expr1_derivative),
                            std::remove_cvref_t<decltype(repeatAlongExcess(expr1_derivative, m_expr2))>
                        > {
                            expr1_derivative,
                            repeatAlongExcess(expr1_derivative, m_expr2)
                        }
                    };
                }
                else {
                    return AdditionExpr{
                        elementwise_prodExpr<
                            DiagonalMatrix<std::remove_cvref_t<decltype(m_expr1)>>,
                            decltype(expr2_derivative)
                        > {
                            DiagonalMatrix{m_expr1},
                            expr2_derivative
                        },
                        elementwise_prodExpr<
                            decltype(expr1_derivative),
                            DiagonalMatrix<std::remove_cvref_t<decltype(m_expr2)>>
                        > {
                            expr1_derivative,
                            DiagonalMatrix{m_expr2}
                        }
                    };
                }
            }

            [[nodiscard]]
            CUDA_HOST constexpr inline std::string to_string() const {
                if constexpr (is_zero_v<E1> || is_zero_v<E2>) {
                    return "";
                }
                else if constexpr (is_identity_v<E1> && is_identity_v<E2>) {
                    return "1";
                }
                else if constexpr (is_identity_v<E1>) {
                    return m_expr2.to_string();
                }
                else if constexpr (is_identity_v<E2>) {
                    return m_expr1.to_string();
                }
                else {
                    auto expr1_str = m_expr1.to_string();
                    auto expr2_str = m_expr2.to_string();

                    // Add parentheses around expressions that contain operators
                    bool expr1_needs_parens = expr1_str.find('+') != std::string::npos || expr1_str.find('-') != std::string::npos || expr1_str.find('/') != std::string::npos;
                    bool expr2_needs_parens = expr2_str.find('+') != std::string::npos || expr2_str.find('-') != std::string::npos || expr2_str.find('/') != std::string::npos;

                    if (expr1_needs_parens && expr2_needs_parens) {
                        return std::format("({}) * ({})", expr1_str, expr2_str);
                    }
                    else if (expr1_needs_parens) {
                        return std::format("({}) * {}", expr1_str, expr2_str);
                    }
                    else if (expr2_needs_parens) {
                        return std::format("{} * ({})", expr1_str, expr2_str);
                    }
                    else if (expr1_str.empty() || expr2_str.empty()) {
                        return std::format("");
                    }
                    else {
                        return std::format("{} * {}", expr1_str, expr2_str);
                    }
                }
            }

            static constexpr bool __is_variable_data = false;

            [[nodiscard]]
            CUDA_COMPATIBLE inline constexpr auto eval_at(uint32_t r = 0, uint32_t c = 0, uint32_t d = 0, uint32_t t = 0) const {
                using common_type = common_arithmetic_t<decltype(m_expr1.eval_at(r, c, d, t)), decltype(m_expr2.eval_at(r, c, d, t))>;
                if constexpr (is_zero_v<E1> || is_zero_v<E2>) {
                    return common_type{};
                }
                else if constexpr (is_identity_v<E1> && is_identity_v<E2>) {
                    return common_type{ 1 };
                }
                else if constexpr (is_identity_v<E1>) {
                    return static_cast<common_type>(m_expr2.eval_at(r, c, d, t));
                }
                else if constexpr (is_identity_v<E2>) {
                    return static_cast<common_type>(m_expr1.eval_at(r, c, d, t));
                }
                else {
                    if constexpr (is_scalar_shape_v<E1>) {
                        return static_cast<common_type>(m_expr1.eval_at(0, 0, 0, 0)) * static_cast<common_type>(m_expr2.eval_at(r, c, d, t));
                    }
                    else if constexpr (is_scalar_shape_v<E2>) {
                        return static_cast<common_type>(m_expr1.eval_at(r, c, d, t)) * static_cast<common_type>(m_expr2.eval_at(0, 0, 0, 0));
                    }
                    else {
                        return static_cast<common_type>(m_expr1.eval_at(r, c, d, t)) * static_cast<common_type>(m_expr2.eval_at(r, c, d, t));
                    }
                }
            }

        private:
            std::conditional_t< (E1::__is_variable_data), const E1&, const E1> m_expr1;
            std::conditional_t< (E2::__is_variable_data), const E2&, const E2> m_expr2;
    };

    template<ExprType E1, ExprType E2> requires(is_scalar_shape_v<E1> || is_scalar_shape_v<E2>)
        CUDA_COMPATIBLE
        [[nodiscard]] constexpr auto operator*(const E1& expr1, const E2& expr2) {
        static_assert(is_elementwise_broadcastable_v<E1, E2>, "Incompatible matrix dimensions for element-wise multiplication.\nMatrices must have the same shape or one of them must be a scalar.");
        return elementwise_prodExpr<E1, E2>{expr1, expr2};
    }

    template<ExprType E1, ExprType E2>
    CUDA_COMPATIBLE
        [[nodiscard]] constexpr auto elementwise_prod(const E1& expr1, const E2& expr2) {
        static_assert(is_elementwise_broadcastable_v<E1, E2>, "Incompatible matrix dimensions for element-wise multiplication.\nMatrices must have the same shape or one of them must be a scalar.");
        return elementwise_prodExpr<E1, E2>{expr1, expr2};
    }

    template<ExprType E, ScalarType S>
    CUDA_COMPATIBLE
        [[nodiscard]] constexpr auto operator*(const E& expr, S a) {
        return elementwise_prodExpr<E, FilledTensor<S, E::rows, E::cols, E::depth, E::time>>{expr, FilledTensor<S, E::rows, E::cols, E::depth, E::time>{a}};
    }

    template<ScalarType S, ExprType E>
    CUDA_COMPATIBLE
        [[nodiscard]] constexpr auto operator*(S a, const E& expr) {
        return elementwise_prodExpr<FilledTensor<S, E::rows, E::cols, E::depth, E::time>, E>{FilledTensor<S, E::rows, E::cols, E::depth, E::time>{a}, expr};
    }




    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////








    template<ExprType E1, ExprType E2> requires(is_matrix_multiplicable_v<E1, E2>)
        class MatrixMultiplicationExpr : public AbstractExpr<MatrixMultiplicationExpr<E1, E2>,
        E1::rows,
        E2::cols,
        std::conditional_t< (E1::depth > E2::depth), E1, E2>::depth,
        std::conditional_t< (E1::time > E2::time), E1, E2>::time
        > {
        public:

            CUDA_COMPATIBLE inline constexpr MatrixMultiplicationExpr(const E1& expr1, const E2& expr2) : m_expr1(expr1), m_expr2(expr2) {
            }

            template<VarIDType varId>
            [[nodiscard]]
            CUDA_COMPATIBLE constexpr inline auto derivate() const {
                static_assert((varId > 0), "Variable ID for differentiation must be positive.");
                auto expr1_derivative = m_expr1.derivate<varId>();
                auto expr2_derivative = m_expr2.derivate<varId>();
                if constexpr (is_zero_v<decltype(expr1_derivative)> && is_zero_v<decltype(expr2_derivative)>) {
                    return expr1_derivative;
                }
                else if constexpr (is_zero_v<decltype(expr1_derivative)>) {
                    return MatrixMultiplicationExpr<
                        std::remove_cvref_t<decltype(m_expr1)>,
                        decltype(expr2_derivative)
                    > {
                        m_expr1,
                        expr2_derivative
                    };
                }
                else if constexpr (is_zero_v<decltype(expr2_derivative)>) {
                    return MatrixMultiplicationExpr<
                        decltype(expr1_derivative),
                        std::remove_cvref_t<decltype(m_expr2)>
                    > {
                            expr1_derivative,
                            m_expr2
                    };
                }
                else {
                    return AdditionExpr{
                        MatrixMultiplicationExpr<
                            decltype(expr1_derivative),
                            std::remove_cvref_t<decltype(m_expr2)>
                        > {
                            expr1_derivative,
                            m_expr2
                        },
                        MatrixMultiplicationExpr<
                            std::remove_cvref_t<decltype(m_expr1)>,
                            decltype(expr2_derivative)
                        > {
                            m_expr1,
                            expr2_derivative
                        }
                    };
                }
            }

            [[nodiscard]]
            CUDA_HOST constexpr inline std::string to_string() const {
                if constexpr (is_zero_v<E1> || is_zero_v<E2>) {
                    return "";
                }
                else if constexpr (is_identity_v<E1> && is_identity_v<E2>) {
                    return "1";
                }
                else if constexpr (is_identity_v<E1>) {
                    return m_expr2.to_string();
                }
                else if constexpr (is_identity_v<E2>) {
                    return m_expr1.to_string();
                }
                else {
                    auto expr1_str = m_expr1.to_string();
                    auto expr2_str = m_expr2.to_string();

                    // Add parentheses around expressions that contain operators
                    bool expr1_needs_parens = expr1_str.find('+') != std::string::npos || expr1_str.find('-') != std::string::npos || expr1_str.find('/') != std::string::npos;
                    bool expr2_needs_parens = expr2_str.find('+') != std::string::npos || expr2_str.find('-') != std::string::npos || expr2_str.find('/') != std::string::npos;

                    if (expr1_needs_parens && expr2_needs_parens) {
                        return std::format("({}) * ({})", expr1_str, expr2_str);
                    }
                    else if (expr1_needs_parens) {
                        return std::format("({}) * {}", expr1_str, expr2_str);
                    }
                    else if (expr2_needs_parens) {
                        return std::format("{} * ({})", expr1_str, expr2_str);
                    }
                    else if (expr1_str.empty() || expr2_str.empty()) {
                        return std::format("");
                    }
                    else {
                        return std::format("{} * {}", expr1_str, expr2_str);
                    }
                }
            }

            static constexpr bool __is_variable_data = false;

            [[nodiscard]]
            CUDA_COMPATIBLE inline constexpr auto eval_at(uint32_t r, uint32_t c, uint32_t d = 0, uint32_t t = 0) const {
                using common_type = common_arithmetic_t<decltype(m_expr1.eval_at(r, c, d, t)), decltype(m_expr2.eval_at(r, c, d, t))>;
                if constexpr (is_zero_v<E1> || is_zero_v<E2>) {
                    return common_type{};
                }
                auto sum = common_type{};
                for (uint32_t k = 0; k < E1::cols; ++k) {
                    auto a = static_cast<common_type>(m_expr1.eval_at(r, k, d, t));
                    auto b = static_cast<common_type>(m_expr2.eval_at(k, c, d, t));
                    sum += a * b;
                }
                return sum;
            }

        private:
            std::conditional_t< (E1::__is_variable_data), const E1&, const E1> m_expr1;
            std::conditional_t< (E2::__is_variable_data), const E2&, const E2> m_expr2;
    };

    template<ExprType E1, ExprType E2> requires(!is_scalar_shape_v<E1> && !is_scalar_shape_v<E2> && 
        !(is_column_vector_v<E1> && is_column_vector_v<E2> && E1::rows == 4 && E2::rows == 4))
        CUDA_COMPATIBLE
        [[nodiscard]] constexpr auto operator*(const E1& expr1, const E2& expr2) {
        static_assert(is_matrix_multiplicable_v<E1, E2>,
            "Incompatible matrix dimensions for matrix multiplication.\nNumber of columns of the first matrix must equal the number of rows of the second matrix."
            );
        return MatrixMultiplicationExpr<E1, E2>{expr1, expr2};
    }




    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////














    template<ExprType E1, ExprType E2> requires(E1::depth == E2::rows)
        class MatrixMultiplicationExpr_DepthContraction : public AbstractExpr<MatrixMultiplicationExpr_DepthContraction<E1, E2>,
        E1::rows,
        E2::cols,
        1,
        std::conditional_t< (E1::time > E2::time), E1, E2>::time
        > {
        public:

            CUDA_COMPATIBLE inline constexpr MatrixMultiplicationExpr_DepthContraction(const E1& expr1, const E2& expr2) : m_expr1(expr1), m_expr2(expr2) {
            }

            template<VarIDType varId>
            [[nodiscard]]
            CUDA_COMPATIBLE constexpr inline auto derivate() const {
                static_assert((varId > 0), "Variable ID for differentiation must be positive.");
                auto expr1_derivative = m_expr1.derivate<varId>();
                auto expr2_derivative = m_expr2.derivate<varId>();
                if constexpr (is_zero_v<decltype(expr1_derivative)> && is_zero_v<decltype(expr2_derivative)>) {
                    return expr1_derivative;
                }
                else if constexpr (is_zero_v<decltype(expr1_derivative)>) {
                    return MatrixMultiplicationExpr_DepthContraction<
                        std::remove_cvref_t<decltype(m_expr1)>,
                        decltype(expr2_derivative)
                    > {
                        m_expr1,
                        expr2_derivative
                    };
                }
                else if constexpr (is_zero_v<decltype(expr2_derivative)>) {
                    return MatrixMultiplicationExpr_DepthContraction<
                        decltype(expr1_derivative),
                        std::remove_cvref_t<decltype(m_expr2)>
                    > {
                            expr1_derivative,
                            m_expr2
                    };
                }
                else {
                    return AdditionExpr{
                        MatrixMultiplicationExpr_DepthContraction<
                            decltype(expr1_derivative),
                            std::remove_cvref_t<decltype(m_expr2)>
                        > {
                            expr1_derivative,
                            m_expr2
                        },
                        MatrixMultiplicationExpr_DepthContraction<
                            std::remove_cvref_t<decltype(m_expr1)>,
                            decltype(expr2_derivative)
                        > {
                            m_expr1,
                            expr2_derivative
                        }
                    };
                }
            }

            [[nodiscard]]
            CUDA_HOST constexpr inline std::string to_string() const {
                if constexpr (is_zero_v<E1> || is_zero_v<E2>) {
                    return "";
                }
                else if constexpr (is_identity_v<E1> && is_identity_v<E2>) {
                    return "1";
                }
                else if constexpr (is_identity_v<E1>) {
                    return m_expr2.to_string();
                }
                else if constexpr (is_identity_v<E2>) {
                    return m_expr1.to_string();
                }
                else {
                    auto expr1_str = m_expr1.to_string();
                    auto expr2_str = m_expr2.to_string();

                    // Add parentheses around expressions that contain operators
                    bool expr1_needs_parens = expr1_str.find('+') != std::string::npos || expr1_str.find('-') != std::string::npos || expr1_str.find('/') != std::string::npos;
                    bool expr2_needs_parens = expr2_str.find('+') != std::string::npos || expr2_str.find('-') != std::string::npos || expr2_str.find('/') != std::string::npos;

                    if (expr1_needs_parens && expr2_needs_parens) {
                        return std::format("({}) * ({})", expr1_str, expr2_str);
                    }
                    else if (expr1_needs_parens) {
                        return std::format("({}) * {}", expr1_str, expr2_str);
                    }
                    else if (expr2_needs_parens) {
                        return std::format("{} * ({})", expr1_str, expr2_str);
                    }
                    else if (expr1_str.empty() || expr2_str.empty()) {
                        return std::format("");
                    }
                    else {
                        return std::format("{} * {}", expr1_str, expr2_str);
                    }
                }
            }

            static constexpr bool __is_variable_data = false;

            [[nodiscard]]
            CUDA_HOST constexpr inline auto eval_at(uint32_t r, uint32_t c, uint32_t d = 0, uint32_t t = 0) const {
                using common_type = common_arithmetic_t<decltype(m_expr1.eval_at(r, c, d, t)), decltype(m_expr2.eval_at(r, c, d, t))>;
                if constexpr (is_zero_v<E1> || is_zero_v<E2>) {
                    return common_type{};
                }
                auto sum = common_type{};
                for (uint32_t k = 0; k < E1::depth; ++k) {
                    auto a = static_cast<common_type>(m_expr1.eval_at(r, c, k, t));
                    auto b = static_cast<common_type>(m_expr2.eval_at(k, c, d, t));
                    sum += a * b;
                }
                return sum;
            }

        private:
            std::conditional_t< (E1::__is_variable_data), const E1&, const E1> m_expr1;
            std::conditional_t< (E2::__is_variable_data), const E2&, const E2> m_expr2;
    };





    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////








    template<ExprType E1, ExprType E2> requires(is_matrix_multiplicable_v<E1, E2>&& is_row_vector_v<E1>&& is_column_vector_v<E2>)
        class DotProductExpr : public AbstractExpr<DotProductExpr<E1, E2>,
        1,
        1
        > {
        public:

            CUDA_COMPATIBLE inline constexpr DotProductExpr(const E1& expr1, const E2& expr2) : m_expr1(expr1), m_expr2(expr2) {
            }

            template<VarIDType varId>
            [[nodiscard]]
            CUDA_COMPATIBLE constexpr inline auto derivate() const {
                static_assert((varId > 0), "Variable ID for differentiation must be positive.");
                auto expr1_derivative = m_expr1.derivate<varId>();
                auto expr2_derivative = m_expr2.derivate<varId>();
                if constexpr (is_zero_v<decltype(expr1_derivative)> && is_zero_v<decltype(expr2_derivative)>) {
                    return expr1_derivative;
                }
                else if constexpr (is_zero_v<decltype(expr1_derivative)>) {
                    return MatrixMultiplicationExpr<
                        std::remove_cvref_t<decltype(m_expr1)>,
                        ConjugateExpr<decltype(expr2_derivative)>
                    > {
                        m_expr1,
                        ConjugateExpr{expr2_derivative}
                    };
                }
                else if constexpr (is_zero_v<decltype(expr2_derivative)>) {
                    return MatrixMultiplicationExpr<
                        decltype(expr1_derivative),
                        ConjugateExpr<std::remove_cvref_t<decltype(m_expr2)>>
                    > {
                            expr1_derivative,
                            ConjugateExpr{m_expr2}
                    };
                }
                else {
                    return AdditionExpr{
                        MatrixMultiplicationExpr<
                            decltype(expr1_derivative),
                            ConjugateExpr<std::remove_cvref_t<decltype(m_expr2)>>
                        > {
                            expr1_derivative,
                            ConjugateExpr{m_expr2}
                        },
                        MatrixMultiplicationExpr<
                            std::remove_cvref_t<decltype(m_expr1)>,
                            ConjugateExpr<decltype(expr2_derivative)>
                        > {
                            m_expr1,
                            ConjugateExpr{expr2_derivative}
                        }
                    };
                }
            }

            [[nodiscard]]
            CUDA_HOST constexpr inline std::string to_string() const {
                if constexpr (is_zero_v<E1> || is_zero_v<E2>) {
                    return "";
                }
                else if constexpr (is_identity_v<E1> && is_identity_v<E2>) {
                    return "1";
                }
                else if constexpr (is_identity_v<E1>) {
                    return m_expr2.to_string();
                }
                else if constexpr (is_identity_v<E2>) {
                    return m_expr1.to_string();
                }
                else {
                    auto expr1_str = m_expr1.to_string();
                    auto expr2_str = m_expr2.to_string();

                    // Add parentheses around expressions that contain operators
                    bool expr1_needs_parens = expr1_str.find('+') != std::string::npos || expr1_str.find('-') != std::string::npos || expr1_str.find('/') != std::string::npos;
                    bool expr2_needs_parens = expr2_str.find('+') != std::string::npos || expr2_str.find('-') != std::string::npos || expr2_str.find('/') != std::string::npos;

                    if (expr1_needs_parens && expr2_needs_parens) {
                        return std::format("({}) dot ({})", expr1_str, expr2_str);
                    }
                    else if (expr1_needs_parens) {
                        return std::format("({}) dot {}", expr1_str, expr2_str);
                    }
                    else if (expr2_needs_parens) {
                        return std::format("{} dot ({})", expr1_str, expr2_str);
                    }
                    else if (expr1_str.empty() || expr2_str.empty()) {
                        return std::format("");
                    }
                    else {
                        return std::format("{} dot {}", expr1_str, expr2_str);
                    }
                }
            }

            static constexpr bool __is_variable_data = false;

            [[nodiscard]]
            CUDA_COMPATIBLE inline constexpr auto eval_at(uint32_t r = 0, uint32_t c = 0, uint32_t d = 0, uint32_t t = 0) const {
                using common_type = common_arithmetic_t<decltype(m_expr1.eval_at(r, c, d, t)), decltype(m_expr2.eval_at(r, c, d, t))>;
                if constexpr (is_zero_v<E1> || is_zero_v<E2>) {
                    return common_type{};
                }
                else {
                    auto sum = common_type{};
                    for (uint32_t k = 0; k < E1::cols; ++k) {
                        if constexpr (ComplexType<decltype(m_expr2.eval_at(0, 0, d, t))>) {
                            sum += static_cast<common_type>(m_expr1.eval_at(r, k, d, t)) * static_cast<common_type>(std::conj(m_expr2.eval_at(k, c, d, t)));
                        }
                        else {
                            sum += static_cast<common_type>(m_expr1.eval_at(r, k, d, t)) * static_cast<common_type>(m_expr2.eval_at(k, c, d, t));
                        }
                    }
                    return sum;
                }
            }

        private:
            std::conditional_t< (E1::__is_variable_data), const E1&, const E1> m_expr1;
            std::conditional_t< (E2::__is_variable_data), const E2&, const E2> m_expr2;
    };

    template<ExprType E1, ExprType E2>
    CUDA_COMPATIBLE
        [[nodiscard]] constexpr auto dot(const E1& expr1, const E2& expr2) {
        static_assert(is_matrix_multiplicable_v<E1, E2> && is_row_vector_v<E1> && is_column_vector_v<E2>
            || is_matrix_multiplicable_v<TransposeExpr<E1>, E2> && is_row_vector_v<TransposeExpr<E1>> && is_column_vector_v<E2>,
            "Incompatible vector dimensions for dot product.\nNumber of elements of the first row-vector must equal the number of elements of the second column-vector."
            );
        if constexpr (is_matrix_multiplicable_v<E1, E2>) {
            return DotProductExpr<E1, E2>{expr1, expr2};
        }
        else {
            return DotProductExpr<TransposeExpr<E1>, E2>{TransposeExpr{ expr1 }, expr2};
        }
    }



    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Skew-symmetric matrix expression for representing cross product derivatives
    // For a 3D vector v, [v]× is the skew-symmetric matrix such that [v]× * w = v × w
    // [v]× = [ 0   -v[2]  v[1] ]
    //        [ v[2]   0   -v[0] ]
    //        [-v[1]  v[0]   0   ]
    template<ExprType E> requires(is_column_vector_v<E> && E::rows == 3)
    class SkewSymmetricMatrixExpr : public AbstractExpr<SkewSymmetricMatrixExpr<E>, 3, 3> {
    public:
        CUDA_COMPATIBLE inline constexpr SkewSymmetricMatrixExpr(const E& expr) : m_expr(expr) {}

        static constexpr bool __is_variable_data = false;

        template<VarIDType varId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr auto derivate() const {
            // d[v]×/dv is a 3×3×3 tensor, which is complex
            // For now, return the derivative of the underlying expression
            auto expr_derivative = m_expr.derivate<varId>();
            if constexpr (is_zero_v<decltype(expr_derivative)>) {
                return zero<decltype(m_expr.eval_at(0, 0))>{};
            } else {
                return SkewSymmetricMatrixExpr<decltype(expr_derivative)>(expr_derivative);
            }
        }

        [[nodiscard]]
        CUDA_HOST constexpr inline std::string to_string() const {
            return std::format("[{}]×", m_expr.to_string());
        }

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval_at(uint32_t r = 0, uint32_t c = 0, uint32_t d = 0, uint32_t t = 0) const {
            using value_type = decltype(m_expr.eval_at(0, 0, d, t));
            if (r == c) {
                return value_type(0);
            } else if (r == 0 && c == 1) {
                return -m_expr.eval_at(2, 0, d, t);
            } else if (r == 0 && c == 2) {
                return m_expr.eval_at(1, 0, d, t);
            } else if (r == 1 && c == 0) {
                return m_expr.eval_at(2, 0, d, t);
            } else if (r == 1 && c == 2) {
                return -m_expr.eval_at(0, 0, d, t);
            } else if (r == 2 && c == 0) {
                return -m_expr.eval_at(1, 0, d, t);
            } else { // r == 2 && c == 1
                return m_expr.eval_at(0, 0, d, t);
            }
        }

    private:
        std::conditional_t<E::__is_variable_data, const E&, const E> m_expr;
    };

    template<ExprType E1, ExprType E2> requires(is_column_vector_v<E1> && is_column_vector_v<E2> && E1::rows == 3 && E2::rows == 3)
        class CrossProductExpr : public AbstractExpr<CrossProductExpr<E1, E2>,
        3,
        1
        > {
        public:

            CUDA_COMPATIBLE inline constexpr CrossProductExpr(const E1& expr1, const E2& expr2) : m_expr1(expr1), m_expr2(expr2) {
            }

            template<VarIDType varId>
            [[nodiscard]]
            CUDA_COMPATIBLE constexpr inline auto derivate() const {
                static_assert((varId > 0), "Variable ID for differentiation must be positive.");
                auto expr1_derivative = m_expr1.derivate<varId>();
                auto expr2_derivative = m_expr2.derivate<varId>();
                if constexpr (is_zero_v<decltype(expr1_derivative)> && is_zero_v<decltype(expr2_derivative)>) {
                    return zero<decltype(m_expr1.eval_at(0, 0))>{};
                }
                else if constexpr (is_zero_v<decltype(expr1_derivative)>) {
                    // d(a × b)/db when a is constant
                    if constexpr (is_identity_v<decltype(expr2_derivative)>) {
                        // If db is identity tensor, return [a]× (skew-symmetric matrix of a)
                        return SkewSymmetricMatrixExpr<E1>(m_expr1);
                    } else {
                        return CrossProductExpr<E1, decltype(expr2_derivative)>{m_expr1, expr2_derivative};
                    }
                }
                else if constexpr (is_zero_v<decltype(expr2_derivative)>) {
                    // d(a × b)/da when b is constant
                    if constexpr (is_identity_v<decltype(expr1_derivative)>) {
                        // If da is identity tensor, return -[b]× (negative skew-symmetric matrix of b)
                        using scalar_type = decltype(m_expr1.eval_at(0, 0));
                        auto skew_b = SkewSymmetricMatrixExpr<E2>(m_expr2);
                        return scalar_type(-1) * skew_b;
                    } else {
                        return CrossProductExpr<decltype(expr1_derivative), E2>{expr1_derivative, m_expr2};
                    }
                }
                else {
                    // Product rule: d(a × b) = da × b + a × db = -[b]× + [a]×
                    // Need to handle identity tensor cases
                    if constexpr (is_identity_v<decltype(expr1_derivative)> && is_identity_v<decltype(expr2_derivative)>) {
                        using scalar_type = decltype(m_expr1.eval_at(0, 0));
                        auto skew_b = SkewSymmetricMatrixExpr<E2>(m_expr2);
                        auto skew_a = SkewSymmetricMatrixExpr<E1>(m_expr1);
                        return scalar_type(-1) * skew_b + skew_a;
                    } else if constexpr (is_identity_v<decltype(expr1_derivative)>) {
                        using scalar_type = decltype(m_expr1.eval_at(0, 0));
                        auto skew_b = SkewSymmetricMatrixExpr<E2>(m_expr2);
                        return scalar_type(-1) * skew_b + CrossProductExpr<E1, decltype(expr2_derivative)>{m_expr1, expr2_derivative};
                    } else if constexpr (is_identity_v<decltype(expr2_derivative)>) {
                        auto skew_a = SkewSymmetricMatrixExpr<E1>(m_expr1);
                        return CrossProductExpr<decltype(expr1_derivative), E2>{expr1_derivative, m_expr2} + skew_a;
                    } else {
                        return CrossProductExpr<decltype(expr1_derivative), E2>{expr1_derivative, m_expr2} +
                            CrossProductExpr<E1, decltype(expr2_derivative)>{m_expr1, expr2_derivative};
                    }
                }
            }

            [[nodiscard]]
            CUDA_HOST constexpr inline std::string to_string() const {
                if constexpr (is_zero_v<E1> || is_zero_v<E2>) {
                    return "";
                }
                else if constexpr (is_identity_v<E1> && is_identity_v<E2>) {
                    return "1";
                }
                else if constexpr (is_identity_v<E1>) {
                    return m_expr2.to_string();
                }
                else if constexpr (is_identity_v<E2>) {
                    return m_expr1.to_string();
                }
                else {
                    auto expr1_str = m_expr1.to_string();
                    auto expr2_str = m_expr2.to_string();

                    // Add parentheses around expressions that contain operators
                    bool expr1_needs_parens = expr1_str.find('+') != std::string::npos || expr1_str.find('-') != std::string::npos || expr1_str.find('/') != std::string::npos;
                    bool expr2_needs_parens = expr2_str.find('+') != std::string::npos || expr2_str.find('-') != std::string::npos || expr2_str.find('/') != std::string::npos;

                    if (expr1_needs_parens && expr2_needs_parens) {
                        return std::format("({}) x ({})", expr1_str, expr2_str);
                    }
                    else if (expr1_needs_parens) {
                        return std::format("({}) x {}", expr1_str, expr2_str);
                    }
                    else if (expr2_needs_parens) {
                        return std::format("{} x ({})", expr1_str, expr2_str);
                    }
                    else if (expr1_str.empty() || expr2_str.empty()) {
                        return std::format("");
                    }
                    else {
                        return std::format("{} x {}", expr1_str, expr2_str);
                    }
                }
            }

            static constexpr bool __is_variable_data = false;

            [[nodiscard]]
            CUDA_COMPATIBLE inline constexpr auto eval_at(uint32_t r = 0, uint32_t c = 0, uint32_t d = 0, uint32_t t = 0) const {
                using common_type = common_arithmetic_t<decltype(m_expr1.eval_at(r, c, d, t)), decltype(m_expr2.eval_at(r, c, d, t))>;
                if constexpr (is_zero_v<E1> || is_zero_v<E2>) {
                    return common_type{};
                }
                else {
                    // Cross product: a x b = (a2b3 - a3b2, a3b1 - a1b3, a1b2 - a2b1)
                    
                    if (r == 0) {
                        auto a1 = static_cast<common_type>(m_expr1.eval_at(1, c, d, t));
                        auto a2 = static_cast<common_type>(m_expr1.eval_at(2, c, d, t));
                        auto b1 = static_cast<common_type>(m_expr2.eval_at(1, c, d, t));
                        auto b2 = static_cast<common_type>(m_expr2.eval_at(2, c, d, t));
                        return a1 * b2 - a2 * b1;
                    }
                    else if (r == 1) {
                        auto a0 = static_cast<common_type>(m_expr1.eval_at(0, c, d, t));
                        auto a2 = static_cast<common_type>(m_expr1.eval_at(2, c, d, t));
                        auto b0 = static_cast<common_type>(m_expr2.eval_at(0, c, d, t));
                        auto b2 = static_cast<common_type>(m_expr2.eval_at(2, c, d, t));
                        return a2 * b0 - a0 * b2;
                    }
                    else if (r == 2) {
                        auto a0 = static_cast<common_type>(m_expr1.eval_at(0, c, d, t));
                        auto b1 = static_cast<common_type>(m_expr2.eval_at(1, c, d, t));
                        auto a1 = static_cast<common_type>(m_expr1.eval_at(1, c, d, t));
                        auto b0 = static_cast<common_type>(m_expr2.eval_at(0, c, d, t));
                        return a0 * b1 - a1 * b0;
                    }
                    else {
                        return common_type{}; // Out of bounds
                    }
                }
            }

        private:
            std::conditional_t< (E1::__is_variable_data), const E1&, const E1> m_expr1;
            std::conditional_t< (E2::__is_variable_data), const E2&, const E2> m_expr2;
    };

    template<ExprType E1, ExprType E2>
    CUDA_COMPATIBLE
    [[nodiscard]] constexpr auto cross(const E1& expr1, const E2& expr2) {
        static_assert(is_column_vector_v<E1> && is_column_vector_v<E2> && E1::rows == 3 && E2::rows == 3,
            "Incompatible vector dimensions for cross product.\nNumber of elements of the first column-vector must equal the number of elements of the second column-vector."
            );
        return CrossProductExpr<E1, E2>{expr1, expr2};
    }






    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



    template<ExprType VE, ExprType QE> /*requires(is_column_vector_v<VE> && VE::rows == 3 && QE::__is_quaternion_valued)*/
    auto rotate_vector_by_quaternion(const VE& p, const QE& q) {
        using common_type = common_arithmetic_t<decltype(p.eval_at(0,0)), decltype(q.eval_at(0,0))>;
        auto w = q.real();
        auto r = q.imag();
        return p + static_cast<common_type>(2) * w * cross(r, p) + static_cast<common_type>(2) * cross(r, cross(r, p));
    }





    template<ExprType E1, ExprType E2> requires(is_elementwise_broadcastable_v<E1, E2>)
        class DivisionExpr : public AbstractExpr<DivisionExpr<E1, E2>,
        std::conditional_t<(E1::rows > E2::rows), E1, E2>::rows,
        std::conditional_t<(E1::cols > E2::cols), E1, E2>::cols,
        std::conditional_t<(E1::depth > E2::depth), E1, E2>::depth,
        std::conditional_t<(E1::time > E2::time), E1, E2>::time
    > {
        public:

            CUDA_COMPATIBLE inline constexpr DivisionExpr(const E1& expr1, const E2& expr2) : m_expr1(expr1), m_expr2(expr2) {
            }

            template<VarIDType varId>
            [[nodiscard]]
            CUDA_COMPATIBLE constexpr inline auto derivate() const {
                static_assert((varId > 0), "Variable ID for differentiation must be positive.");
                auto expr1_derivative = m_expr1.derivate<varId>();
                auto expr2_derivative = m_expr2.derivate<varId>();
                if constexpr (is_zero_v<decltype(expr1_derivative)> && is_zero_v<decltype(expr2_derivative)>) {
                    return expr1_derivative;
                }
                else if constexpr (is_zero_v<decltype(expr1_derivative)>) {
                    auto numerator = NegationExpr{
                        elementwise_prodExpr {
                            repeatAlongExcess(expr2_derivative, m_expr1),
                            expr2_derivative
                        }
                    };
                        auto denominator = repeatAlongExcess(
                            expr2_derivative,
                            elementwise_prodExpr {
                                m_expr2,
                                m_expr2
                            });
                    return DivisionExpr<decltype(numerator), decltype(denominator)>{
                        numerator,
                        denominator
                    };
                }
                else if constexpr (is_zero_v<decltype(expr2_derivative)>) {
                    auto numerator = elementwise_prodExpr {
                            expr1_derivative,
                            repeatAlongExcess(expr1_derivative, m_expr2)
                        };
                    auto denominator = elementwise_prodExpr {
                        m_expr2,
                        m_expr2
                    };
                    return DivisionExpr<decltype(numerator), decltype(denominator)>{
                        numerator,
                        denominator
                    };

                }
                else {
                    auto numerator = SubtractionExpr{
                        elementwise_prodExpr<
                            decltype(expr1_derivative),
                            DiagonalMatrix<std::remove_cvref_t<decltype(m_expr2)>>
                        > {
                            expr1_derivative,
                            DiagonalMatrix{m_expr2}
                        },
                        elementwise_prodExpr<
                            DiagonalMatrix<std::remove_cvref_t<decltype(m_expr1)>>,
                            decltype(expr2_derivative)
                        > {
                            DiagonalMatrix{m_expr1},
                            expr2_derivative
                        }
                    };
                    auto denominator = DiagonalMatrix{elementwise_prodExpr<
                        std::remove_cvref_t<decltype(m_expr2)>,
                        std::remove_cvref_t<decltype(m_expr2)>
                    >{
                        m_expr2,
                        m_expr2
                    }, 1};
                    return DivisionExpr<decltype(numerator), decltype(denominator)>{
                        numerator,
                        denominator
                    };
                }
            }

            [[nodiscard]]
            CUDA_HOST constexpr inline std::string to_string() const {
                if constexpr (is_zero_v<E1>) {
                    return "";
                }
                else if constexpr (is_zero_v<E2>) {
                    return std::string("[Division by zero in scalar expression.]");
                }
                else if constexpr (is_identity_v<E1> && is_identity_v<E2>) {
                    return "1";
                }
                else if constexpr (is_identity_v<E2>) {
                    return m_expr1.to_string();
                }
                else {
                    auto expr1_str = m_expr1.to_string();
                    auto expr2_str = m_expr2.to_string();

                    // Add parentheses around expressions that contain operators
                    bool expr1_needs_parens = expr1_str.find('+') != std::string::npos || expr1_str.find('-') != std::string::npos || expr1_str.find('*') != std::string::npos || expr1_str.find('/') != std::string::npos;
                    bool expr2_needs_parens = expr2_str.find('+') != std::string::npos || expr2_str.find('-') != std::string::npos || expr2_str.find('*') != std::string::npos || expr2_str.find('/') != std::string::npos;

                    if (expr1_needs_parens && expr2_needs_parens) {
                        return std::format("({}) / ({})", expr1_str, expr2_str);
                    }
                    else if (expr1_needs_parens) {
                        return std::format("({}) / {}", expr1_str, expr2_str);
                    }
                    else if (expr2_needs_parens) {
                        return std::format("{} / ({})", expr1_str, expr2_str);
                    }
                    else if (expr2_str.empty()) {
                        return std::format("[Division by zero in scalar expression.]");
                    }
                    else if (expr1_str.empty()) {
                        return std::format("");
                    }
                    else {
                        return std::format("{} / {}", expr1_str, expr2_str);
                    }
                }
            }

            static constexpr bool __is_variable_data = false;

            [[nodiscard]]
            CUDA_COMPATIBLE inline constexpr auto eval_at(uint32_t r = 0, uint32_t c = 0, uint32_t d = 0, uint32_t t = 0) const {
                static_assert(!is_zero_v<E2>, "Division by zero in scalar expression.");

                using common_type = common_arithmetic_t<decltype(m_expr1.eval_at(r, c, d, t)), decltype(m_expr2.eval_at(r, c, d, t))>;
                if constexpr (is_identity_v<E2>) {
                    return static_cast<common_type>(m_expr1.eval_at(r, c, d, t));
                }
                else {
                    return static_cast<common_type>(m_expr1.eval_at(r, c, d, t)) / static_cast<common_type>(m_expr2.eval_at(r, c, d, t));
                }
            }

        private:
            std::conditional_t< (E1::__is_variable_data), const E1&, const E1> m_expr1;
            std::conditional_t< (E2::__is_variable_data), const E2&, const E2> m_expr2;
    };

    template<ExprType E1, ExprType E2>
    CUDA_COMPATIBLE
    [[nodiscard]] constexpr auto operator/(const E1& expr1, const E2& expr2) {
        static_assert(is_elementwise_broadcastable_v<E1, E2>, "Incompatible matrix dimensions for element-wise division.\nMatrices must have the same shape or one of them must be a scalar.");
        return DivisionExpr<E1, E2>{expr1, expr2};
    }

    template<ExprType E, ScalarType S>
    CUDA_COMPATIBLE
        [[nodiscard]] constexpr auto operator/(const E& expr, S a) {
        return DivisionExpr<E, FilledTensor<S, E::rows, E::cols, E::depth, E::time>>{expr, FilledTensor<S, E::rows, E::cols, E::depth, E::time>{a}};
    }

    template<ScalarType S, ExprType E>
    CUDA_COMPATIBLE
        [[nodiscard]] constexpr auto operator/(S a, const E& expr) {
        return DivisionExpr<FilledTensor<S, E::rows, E::cols, E::depth, E::time>, E>{FilledTensor<S, E::rows, E::cols, E::depth, E::time>{a}, expr};
    }



    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////








    template<ExprType E>
    class NaturalLogExpr : public AbstractExpr<NaturalLogExpr<E>, E::rows, E::cols, E::depth, E::time> {
    public:

        CUDA_COMPATIBLE inline constexpr NaturalLogExpr(const E& expr) : m_expr(expr) {}

        template<VarIDType varId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert(varId > 0, "Variable ID for differentiation must be positive.");
            auto expr_derivative = m_expr.derivate<varId>();
            if constexpr (is_zero_v<decltype(expr_derivative)>) {
                return expr_derivative;
            }
            else {
                return DivisionExpr<
                    decltype(expr_derivative),
                    std::remove_cvref_t<decltype(repeatAlongExcess(expr_derivative, m_expr))>
                >{
                    expr_derivative,
                    repeatAlongExcess(expr_derivative, m_expr)
                };
            }
        }

        [[nodiscard]]
        CUDA_HOST constexpr inline std::string to_string() const {
            std::string str_expr = m_expr.to_string();
            if (str_expr.empty()) {
                return std::format("[]Log of zero expression.]");
            }
            else {
                return std::format("log({})", str_expr);
            }
        }

        static constexpr bool __is_variable_data = false;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval_at(uint32_t r = 0, uint32_t c = 0, uint32_t d = 0, uint32_t t = 0) const {
            auto expr_value = m_expr.eval_at(r, c, d, t);
#ifndef ENABLE_CUDA_SUPPORT
            if constexpr (is_zero_v<E>) {
                throw std::runtime_error("[Logarithm of zero in scalar expression.]");
            }
#endif
            return log(expr_value);
        }

    private:
        std::conditional_t<(E::__is_variable_data), const E&, const E> m_expr;
    };

    template<ExprType E>
    CUDA_COMPATIBLE
        [[nodiscard]] constexpr auto log(const E& expr) {
        return NaturalLogExpr<E>{expr};
    }




    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////






    template<ExprType E>
    class AbsoluteValueExpr : public AbstractExpr<AbsoluteValueExpr<E>, E::rows, E::cols, E::depth, E::time> {
    public:

        CUDA_COMPATIBLE inline constexpr AbsoluteValueExpr(const E& expr) : m_expr(expr) {}

        template<VarIDType varId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert(varId > 0, "Variable ID for differentiation must be positive.");
            auto expr_derivative = m_expr.derivate<varId>();
            if constexpr (is_zero_v<decltype(expr_derivative)>) {
                return expr_derivative;
            }
            else {
                return elementwise_prodExpr{
                    expr_derivative,
                    repeatAlongExcess(
                        expr_derivative,
                        DivisionExpr{
                            m_expr,
                            AbsoluteValueExpr{m_expr}
                        })
                    };
            }
        }

        [[nodiscard]]
        CUDA_HOST constexpr inline std::string to_string() const {
            std::string str_expr = m_expr.to_string();
            return std::format("abs({})", str_expr);
        }

        static constexpr bool __is_variable_data = false;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval_at(uint32_t r = 0, uint32_t c = 0, uint32_t d = 0, uint32_t t = 0) const {
            return abs(m_expr.eval_at(r, c, d, t));
        }

    private:
        std::conditional_t<(E::__is_variable_data), const E&, const E> m_expr;
    };




    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////








    template<ExprType E>
    class NaturalExpExpr : public AbstractExpr<NaturalExpExpr<E>, E::rows, E::cols, E::depth, E::time> {
    public:

        CUDA_COMPATIBLE inline constexpr NaturalExpExpr(const E& expr) : m_expr(expr) {}

        template<VarIDType varId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert(varId > 0, "Variable ID for differentiation must be positive.");
            auto expr_derivative = m_expr.derivate<varId>();
            if constexpr (is_zero_v<decltype(expr_derivative)>) {
                return expr_derivative;
            }
            else {
                return elementwise_prodExpr<
                    std::remove_cvref_t<decltype(repeatAlongExcess(expr_derivative, NaturalExpExpr<E>{m_expr}))>,
                    decltype(expr_derivative)
                > {
                    repeatAlongExcess(expr_derivative, NaturalExpExpr<E>{m_expr}),
                    expr_derivative
                };
            }
        }

        [[nodiscard]]
        CUDA_HOST constexpr inline std::string to_string() const {
            std::string str_expr = m_expr.to_string();
            if (str_expr.empty()) {
                return std::format("1");
            }
            else {
                return std::format("e^({})", str_expr);
            }
        }

        static constexpr bool __is_variable_data = false;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval_at(uint32_t r = 0, uint32_t c = 0, uint32_t d = 0, uint32_t t = 0) const {
            return exp(m_expr.eval_at(r, c, d, t));
        }

    private:
        std::conditional_t<(E::__is_variable_data), const E&, const E> m_expr;
    };

    template<ExprType E>
    CUDA_COMPATIBLE
        [[nodiscard]] constexpr auto exp(const E& expr) {
        return NaturalExpExpr<E>{expr};
    }





    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////








    template<ExprType E>
    class CosExpr;

    template<ExprType E>
    class SinExpr : public AbstractExpr<SinExpr<E>, E::rows, E::cols, E::depth, E::time> {
    public:

        CUDA_COMPATIBLE inline constexpr SinExpr(const E& expr) : m_expr(expr) {}

        template<VarIDType varId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert(varId > 0, "Variable ID for differentiation must be positive.");
            auto expr_derivative = m_expr.derivate<varId>();
            if constexpr (is_zero_v<decltype(expr_derivative)>) {
                return expr_derivative;
            }
            else {
                return elementwise_prodExpr {
                    repeatAlongExcess(expr_derivative, CosExpr<E>{m_expr}),
                    expr_derivative
                };
            }
        }

        [[nodiscard]]
        CUDA_HOST constexpr inline std::string to_string() const {
            std::string str_expr = m_expr.to_string();
            if (str_expr.empty()) {
                return std::format("sin(0)");
            }
            else {
                return std::format("sin({})", str_expr);
            }
        }

        static constexpr bool __is_variable_data = false;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval_at(uint32_t r = 0, uint32_t c = 0, uint32_t d = 0, uint32_t t = 0) const {
            return sin(m_expr.eval_at(r, c, d, t));
        }

    private:
        std::conditional_t<(E::__is_variable_data), const E&, const E> m_expr;
    };

    template<ExprType E>
    CUDA_COMPATIBLE
        [[nodiscard]] constexpr auto sin(const E& expr) {
        return SinExpr<E>{expr};
    }




    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////







    template<ExprType E>
    class CosExpr : public AbstractExpr<CosExpr<E>, E::rows, E::cols, E::depth, E::time> {
    public:

        CUDA_COMPATIBLE inline constexpr CosExpr(const E& expr) : m_expr(expr) {}

        template<VarIDType varId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert(varId > 0, "Variable ID for differentiation must be positive.");
            auto expr_derivative = m_expr.derivate<varId>();
            if constexpr (is_zero_v<decltype(expr_derivative)>) {
                return expr_derivative;
            }
            else {
                return elementwise_prodExpr<
                    std::remove_cvref_t<decltype(repeatAlongExcess(expr_derivative, NegationExpr{SinExpr<E>{m_expr}}))>,
                    decltype(expr_derivative)
                > {
                    repeatAlongExcess(expr_derivative, NegationExpr{SinExpr<E>{m_expr}}),
                    expr_derivative
                };
            }
        }

        [[nodiscard]]
        CUDA_HOST constexpr inline std::string to_string() const {
            std::string str_expr = m_expr.to_string();
            if (str_expr.empty()) {
                return std::format("cos(0)");
            }
            else {
                return std::format("cos({})", str_expr);
            }
        }

        static constexpr bool __is_variable_data = false;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval_at(uint32_t r = 0, uint32_t c = 0, uint32_t d = 0, uint32_t t = 0) const {
            return cos(m_expr.eval_at(r, c, d, t));
        }

    private:
        std::conditional_t<(E::__is_variable_data), const E&, const E> m_expr;
    };

    template<ExprType E>
    CUDA_COMPATIBLE
        [[nodiscard]] constexpr auto cos(const E& expr) {
        return CosExpr<E>{expr};
    }




    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////









    template<ExprType E1, ExprType E2> requires(is_elementwise_broadcastable_v<E1, E2>)
        class ElementwisePowExpr : public AbstractExpr<ElementwisePowExpr<E1, E2>,
        std::conditional_t<(E1::rows > E2::rows), E1, E2>::rows,
        std::conditional_t<(E1::cols > E2::cols), E1, E2>::cols,
        std::conditional_t<(E1::depth > E2::depth), E1, E2>::depth,
        std::conditional_t<(E1::time > E2::time), E1, E2>::time>
    {
    public:

        CUDA_COMPATIBLE inline constexpr ElementwisePowExpr(const E1& expr1, const E2& expr2) : m_expr1(expr1), m_expr2(expr2) {
        }

        template<VarIDType varId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert((varId > 0), "Variable ID for differentiation must be positive.");
            auto expr1_derivative = m_expr1.derivate<varId>();
            auto expr2_derivative = m_expr2.derivate<varId>();
            if constexpr (is_identity_v<E2>) {
                return expr1_derivative;
            }
            else if constexpr (is_zero_v<decltype(expr1_derivative)>)
            {
                return expr1_derivative;
            }
            else {
                return elementwise_prodExpr{
                    ElementwisePowExpr<E1, E2> {
                        m_expr1,
                        m_expr2
                    },
                    AdditionExpr {
                        elementwise_prodExpr {
                            expr1_derivative,
                            DivisionExpr {
                                m_expr2,
                                m_expr1
                            }
                        },
                        elementwise_prodExpr {
                            expr2_derivative,
                            NaturalLogExpr {
                                m_expr1
                            }
                        }
                    }
                };
            }
        }

        [[nodiscard]]
        CUDA_HOST constexpr inline std::string to_string() const {
            if constexpr (is_zero_v<E1>) {
                return "";
            }
            else if constexpr (is_zero_v<E2>) {
                return "1";
            }
            else if constexpr (is_identity_v<E1>) {
                return "1";
            }
            else if constexpr (is_identity_v<E2>) {
                return m_expr1.to_string();
            }
            else {
                auto expr1_str = m_expr1.to_string();
                auto expr2_str = m_expr2.to_string();

                // Add parentheses around expressions that contain operators
                bool expr1_needs_parens = expr1_str.find('+') != std::string::npos || expr1_str.find('-') != std::string::npos || expr1_str.find('*') != std::string::npos || expr1_str.find('/') != std::string::npos;
                bool expr2_needs_parens = expr2_str.find('+') != std::string::npos || expr2_str.find('-') != std::string::npos || expr2_str.find('*') != std::string::npos || expr2_str.find('/') != std::string::npos;

                if (expr1_needs_parens && expr2_needs_parens) {
                    return std::format("({})^({})", expr1_str, expr2_str);
                }
                else if (expr1_needs_parens) {
                    return std::format("({})^{}", expr1_str, expr2_str);
                }
                else if (expr2_needs_parens) {
                    return std::format("{}^({})", expr1_str, expr2_str);
                }
                else if (expr2_str.empty()) {
                    return std::format("1");
                }
                else if (expr1_str.empty()) {
                    return std::format("");
                }
                else {
                    return std::format("{}^{}", expr1_str, expr2_str);
                }
            }
        }

        static constexpr bool __is_variable_data = false;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval_at(uint32_t r = 0, uint32_t c = 0, uint32_t d = 0, uint32_t t = 0) const {
            return pow(m_expr1.eval_at(r, c, d, t), m_expr2.eval_at(r, c, d, t));
        }

    private:
        std::conditional_t< (E1::__is_variable_data), const E1&, const E1> m_expr1;
        std::conditional_t< (E2::__is_variable_data), const E2&, const E2> m_expr2;
    };





    ///////////////////////////////////////////////////////////////////////////////////////////////////////////










    template<uint32_t P, ExprType E> requires (P >= 1 && is_vector_v<E>)
    class PNormExpr : public AbstractExpr<PNormExpr<P, E>, 1, 1, 1, 1> {
    public:

        CUDA_COMPATIBLE inline constexpr PNormExpr(const E& expr) : m_expr(expr) {
        }

        template<VarIDType varId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert((varId > 0), "Variable ID for differentiation must be positive.");
            auto expr_derivative = m_expr.derivate<varId>();
            if constexpr (is_zero_v<decltype(expr_derivative)>) {
                return expr_derivative;
            }
            else {
                if constexpr (P >= 2) {
                    return MatrixMultiplicationExpr{
                        TransposeExpr{expr_derivative},
                        DivisionExpr{
                            elementwise_prodExpr{
                                ElementwisePowExpr{
                                    AbsoluteValueExpr{m_expr},
                                    FilledTensor<uint32_t, 1, 1, 1, 1>{P - 2}
                                },
                                m_expr
                            },
                            ElementwisePowExpr{
                                PNormExpr<P, E>{m_expr},
                                FilledTensor<uint32_t, 1, 1, 1, 1>{P - 1}
                            }
                        }
                    };
                }
                else {
                    static_assert(false, "Unimplemented function");
                }
            }
        }

        [[nodiscard]]
        CUDA_HOST constexpr inline std::string to_string() const {
            return std::format("{}-norm({})", P, m_expr.to_string());
        }

        static constexpr bool __is_variable_data = false;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval_at(uint32_t r = 0, uint32_t c = 0, uint32_t d = 0, uint32_t t = 0) const {
            using val_type = decltype(m_expr.eval_at(0, 0, 0, 0));
            if constexpr (is_zero_v<E>) {
                return val_type{};
            }
            auto sum = val_type{};
            for (uint32_t c = 0; c < E::cols; ++c) {
                for (uint32_t r = 0; r < E::rows; ++r) {
                    auto val = m_expr.eval_at(r, c, d, t);
                    sum += pow(abs(val), static_cast<decltype(val)>(P));
                }
            }
            return pow(sum, 1.0 / static_cast<double>(P));
        }

        private:
            std::conditional_t<(E::__is_variable_data), const E&, const E> m_expr;
    };

    /*
    P-norm
    */
    template<uint32_t P, ExprType E>
    CUDA_COMPATIBLE
    [[nodiscard]] constexpr auto p_norm(const E& expr) {
        return PNormExpr<P, E>{expr};
    }

    /*
    2-norm
    */
    template<ExprType E>
    CUDA_COMPATIBLE
    [[nodiscard]] constexpr auto norm(const E& expr) {
        return PNormExpr<2, E>{expr};
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////







    // Operator and function overloads:

    template<ExprType E1, ExprType E2>
    CUDA_COMPATIBLE
        [[nodiscard]] constexpr auto pow(const E1& base, const E2& exponent) {
        static_assert(is_elementwise_broadcastable_v<E1, E2>,
            "Incompatible matrix dimensions for element-wise power operation.\nMatrices must have the same shape or one of them must be a scalar for elementwise broadcasting."
            );
        return ElementwisePowExpr<E1, E2>{base, exponent};
    }

    template<ExprType E, ScalarType S>
    CUDA_COMPATIBLE
        [[nodiscard]] constexpr auto pow(const E& base, S exponent) {
        return ElementwisePowExpr<E, FilledTensor<S, E::rows, E::cols, E::depth, E::time>>{base, exponent};
    }

    template<ScalarType S, ExprType E>
    CUDA_COMPATIBLE
        [[nodiscard]] constexpr auto pow(S base, const E& exponent) {
        return ElementwisePowExpr<FilledTensor<S, E::rows, E::cols, E::depth, E::time>, E>{base, exponent};
    }

    template<ExprType E>
    CUDA_COMPATIBLE
    [[nodiscard]] constexpr auto abs(const E& expr) {
        return AbsoluteValueExpr<E>{expr};
    }

    template<ExprType E>
    CUDA_COMPATIBLE
    [[nodiscard]] constexpr inline auto sqrt(const E& expr) {
        return pow(expr, 0.5f);
    }

    /*
    Initializes a variable matrix as an identity matrix with the same shape and type as the given expression.
    */
    template<ExprType E>
    CUDA_COMPATIBLE
    [[nodiscard]] constexpr inline auto identity_like(const E& expr) {
        return VariableMatrix<decltype(expr.eval_at(0,0,0,0)), E::rows, E::cols>::identity();
    }

    /*
    Initializes a variable matrix as a full zero matrix with the same shape and type as the given expression.
    */
    template<ExprType E>
    CUDA_COMPATIBLE
    [[nodiscard]] constexpr inline auto zeros_like(const E& expr) {
        return VariableMatrix<decltype(expr.eval_at(0,0,0,0)), E::rows, E::cols>{};
    }

    /*
    Initializes a variable matrix as a matrix filled with ones with the same shape and type as the given expression.
    */
    template<ExprType E>
    CUDA_COMPATIBLE
    [[nodiscard]] constexpr inline auto ones_like(const E& expr) {
        return VariableMatrix<decltype(expr.eval_at(0,0,0,0)), E::rows, E::cols>::ones();
    }

    /*
    Initializes a variable matrix as a matrix filled with random values.
    */
    template<ExprType E>
    CUDA_COMPATIBLE
    [[nodiscard]] constexpr inline auto random_like(const E& expr, decltype(expr.eval_at(0,0,0,0)) min, decltype(expr.eval_at(0,0,0,0)) max) {
        return VariableMatrix<decltype(expr.eval_at(0,0,0,0)), E::rows, E::cols>::random(min, max);
    }

    /*
    Initializes a copy.
    */
    template<ExprType E>
    CUDA_COMPATIBLE
    [[nodiscard]] constexpr inline auto like(const E& expr) {
        return expr.eval();
    }

    /*
    Initializes a copy.
    */
    template<ExprType E>
    CUDA_COMPATIBLE
    [[nodiscard]] constexpr inline auto copy(const E& expr) {
        return expr.eval();
    }





    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // SOLVERS
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////










    class Solver {
    protected:
        bool m_print_progress = false;

    public:
        CUDA_COMPATIBLE
        virtual void solve() = 0;
    };









    template<ExprType ErrorType, ExprType ParamType>
        requires(is_scalar_shape_v<ErrorType> && is_variable_v<ParamType> && is_matrix_shape_v<ParamType>)
    class AdamOptimizer : public Solver {
    public:

        struct Options {
            uint32_t max_iter_count = 50000;
            uint32_t initial_state_count = 200;
            double alpha = 0.001; // Learning rate
            double beta1 = 0.9; 
            double beta2 = 0.999;
            double epsilon = 1e-8;
        };

        AdamOptimizer(
            const ErrorType& _output,
            ParamType& _param,
            decltype(_param.eval_at(0,0,0,0)) _paramMin = static_cast<decltype(_param.eval_at(0,0,0,0))>(-1e+2),
            decltype(_param.eval_at(0,0,0,0)) _paramMax = static_cast<decltype(_param.eval_at(0,0,0,0))>(1e+2)
        ) : m_error(_output), m_param(_param), m_param_min(_paramMin), m_param_max(_paramMax) {
            m_print_progress = true;
        };



        CUDA_HOST
        void solve() override {
            auto gradient = SwapRowsAndColsWithDepthAndTimeExpr{m_error.derivate<m_param.variable_id>()};
            static_assert(is_eq_shape_v<decltype(gradient), ParamType>, "Gradient and variable shapes do not match in minimization problem.");

            auto best_error = m_error.eval_at(0,0,0,0);
            auto param_copy = m_param;

            for (uint32_t i = 0; i < m_options.initial_state_count; ++i) {
                m_param = random_like(m_param, m_param_min, m_param_max);
                auto m = zeros_like(m_param);
                auto v = zeros_like(m_param);
                for (int t = 0; t < m_options.max_iter_count; ++t) {
                    m = m_options.beta1 * m + (1.0 - m_options.beta1) * gradient;   // Update first moment estimate
                    v = m_options.beta2 * v + (1.0 - m_options.beta2) * pow(gradient, 2); // Update second moment estimate
                    auto m_hat = m / (1.0 - std::pow(m_options.beta1, t + 1));
                    auto v_hat = v / (1.0 - std::pow(m_options.beta2, t + 1));
                    m_param -= m_options.alpha * m_hat / (sqrt(v_hat) + m_options.epsilon);
                }
                // Evaluate current error:
                auto current_error = m_error.eval_at(0,0,0,0);
                if (current_error < best_error) {
                    best_error = current_error;
                    param_copy = m_param;
                }
                if (m_print_progress) {
                    std::cout << "Optimizing " << (i + 1) << "/" << m_options.initial_state_count << " initial state ... best error: " << best_error << " ... current error: " << current_error << "                    \r";
                }
            }
            if (m_print_progress) {
                std::cout << "\n";
            }
            m_param = param_copy;
        }

    private:
        Options m_options;
        const ErrorType& m_error;
        ParamType& m_param;

        decltype(m_param.eval_at(0,0,0,0)) m_param_min;
        decltype(m_param.eval_at(0,0,0,0)) m_param_max;
    };









//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Scalar helper functions
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    // Helper functions for complex-aware operations
    template<typename T>
    CUDA_COMPATIBLE constexpr auto conj_value(const T& val) {
        if constexpr (is_complex_v<T>) {
            return std::conj(val);
        } else {
            return val;
        }
    }

    template<typename T>
    CUDA_COMPATIBLE constexpr auto real_value(const T& val) {
        if constexpr (is_complex_v<T>) {
            return std::real(val);
        } else {
            return val;
        }
    }

    template<typename T>
    CUDA_COMPATIBLE constexpr auto abs_squared(const T& val) {
        if constexpr (is_complex_v<T>) {
            return std::real(val * std::conj(val));
        } else {
            return val * val;
        }
    }

    template<typename T>
    CUDA_COMPATIBLE constexpr auto magnitude(const T& val) {
        if constexpr (is_complex_v<T>) {
            return std::sqrt(std::real(val * std::conj(val)));
        } else {
            return std::abs(val);
        }
    }







    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////







    /*
        QR Decomposition using Householder reflections.
        Decomposes matrix A into Q and R such that A = Q * R,
        where Q is an orthogonal (or unitary) matrix and R is an upper triangular matrix.
    */
    template<ExprType E>
    class QRDecomposition : public Solver {
    private:
        const E& m_A;
        VariableMatrix<decltype(m_A.eval_at(0,0,0,0)), E::rows, E::rows, 'Q'> m_Q;
        VariableMatrix<decltype(m_A.eval_at(0,0,0,0)), E::rows, E::cols, 'R'> m_R;
        uint32_t m_reflections_count;  // Number of Householder reflections applied

    public:
        QRDecomposition(const E& _A) : m_A(_A), m_reflections_count(0) {
            static_assert(is_matrix_shape_v<E>, "QR Decomposition can only be performed on matrices.");
        }

        CUDA_HOST
        void solve() override {
            using T = decltype(m_A.eval_at(0,0,0,0));
            using RealT = decltype(real_value(T{}));
            constexpr RealT epsilon = std::numeric_limits<RealT>::epsilon();

            m_Q = VariableMatrix<T, E::rows, E::rows>::identity();
            m_R = m_A.eval();
            m_reflections_count = 0;

            for (uint32_t k = 0; k < E::cols; ++k) {
                // Compute the norm of the k-th column from k to the end
                RealT norm_sq = RealT{};
                for (uint32_t i = k; i < E::rows; ++i) {
                    T val = m_R.eval_at(i, k);
                    norm_sq += abs_squared(val);
                }
                RealT norm = std::sqrt(norm_sq);

                // Skip if the column is already zero
                if (norm < epsilon) {
                    continue;
                }

                // Form the k-th Householder vector
                VariableMatrix<T, m_A.rows, 1> v;
                T r_kk = m_R.eval_at(k, k);
                
                // Choose sign to avoid cancellation
                // For complex numbers, we use the phase of r_kk
                T sign_choice;
                if constexpr (is_complex_v<T>) {
                    RealT mag = magnitude(r_kk);
                    if (mag > epsilon) {
                        sign_choice = r_kk / mag;  // Phase of r_kk
                    } else {
                        sign_choice = T{1};
                    }
                } else {
                    sign_choice = (r_kk >= 0) ? T{1} : T{-1};
                }
                T u_k = r_kk + sign_choice * norm;
                
                for (uint32_t i = 0; i < m_A.rows; ++i) {
                    if (i < k) {
                        v.at(i, 0) = T{0};
                    }
                    else if (i == k) {
                        v.at(i, 0) = u_k;
                    }
                    else {
                        v.at(i, 0) = m_R.eval_at(i, k);
                    }
                }

                // Compute v_norm_sq for normalization (v^H * v)
                RealT v_norm_sq = RealT{0};
                for (uint32_t i = k; i < m_A.rows; ++i) {
                    T val = v.eval_at(i, 0);
                    v_norm_sq += abs_squared(val);
                }

                // Skip if v is essentially zero
                if (v_norm_sq < epsilon) {
                    continue;
                }

                m_reflections_count++;

                // Apply Householder transformation to R: H*R
                // H = I - 2*v*v^H / (v^H*v)
                // For column k, we only update the diagonal since below will be zero
                // For columns k+1 onwards, we update from row k to end
                T inv_v_norm_sq = T{1} / v_norm_sq;
                
                // Update column k: only the diagonal element, set rest to zero
                T vH_R_col_k = T{0};
                for (uint32_t i = k; i < E::rows; ++i) {
                    vH_R_col_k += conj_value(v.eval_at(i, 0)) * m_R.eval_at(i, k);
                }
                T factor_k = T{2} * vH_R_col_k * inv_v_norm_sq;
                m_R.at(k, k) -= factor_k * v.eval_at(k, 0);  // Update diagonal
                for (uint32_t i = k + 1; i < E::rows; ++i) {
                    m_R.at(i, k) = T{0};  // Explicitly zero below diagonal
                }
                
                // For columns k+1 to end, update all rows from k onwards
                for (uint32_t j = k + 1; j < m_A.cols; ++j) {
                    // Compute v^H * R_col[j]
                    T vH_R_col = T{0};
                    for (uint32_t i = k; i < E::rows; ++i) {
                        vH_R_col += conj_value(v.eval_at(i, 0)) * m_R.eval_at(i, j);
                    }
                    
                    // Update R_col[j] := R_col[j] - 2*(v^H*R_col[j])*v / (v^H*v)
                    T factor = T{2} * vH_R_col * inv_v_norm_sq;
                    for (uint32_t i = k; i < E::rows; ++i) {
                        m_R.at(i, j) -= factor * v.eval_at(i, 0);
                    }
                }

                // Apply Householder transformation to Q: Q := Q*H
                // H = I - 2*v*v^H / (v^H*v)
                // For Q*H, we update: Q_row[i] := Q_row[i] - 2*(Q_row[i]*v)*v^H / (v^H*v)
                // We only need to update columns k onwards since v has zeros before k
                
                // For each row i of Q
                for (uint32_t i = 0; i < E::rows; ++i) {
                    // Compute Q_row[i] * v (only from column k onwards where v is non-zero)
                    T Q_row_v = T{0};
                    for (uint32_t l = k; l < E::rows; ++l) {
                        Q_row_v += m_Q.eval_at(i, l) * v.eval_at(l, 0);
                    }
                    
                    // Update Q_row[i] := Q_row[i] - 2*(Q_row[i]*v)*v^H / (v^H*v)
                    // Only update columns k onwards
                    T factor = T{2} * Q_row_v * inv_v_norm_sq;
                    for (uint32_t j = k; j < E::rows; ++j) {
                        m_Q.at(i, j) -= factor * conj_value(v.eval_at(j, 0));
                    }
                }
            }
        }
        
        
        /*
        Calculate the determinant of A using the QR decomposition
        det(A) = det(Q) * det(R)
        det(Q) = (-1)^num_reflections
        det(R) = product of diagonal elements (upper triangular)
        */
        CUDA_HOST
        constexpr auto determinant() const {
            using T = decltype(m_A.eval_at(0,0,0,0));
            static_assert(E::rows == E::cols, "Determinant can only be computed for square matrices.");
            
            T det_Q = (m_reflections_count % 2 == 0) ? T{1} : T{-1};
            T det_R = T{1};
            for (uint32_t i = 0; i < E::rows; ++i) {
                det_R *= m_R.eval_at(i, i);
            }
            return det_Q * det_R;
        }
        
        /*
        Get the sign/phase of det(Q)
        */
        CUDA_HOST
        constexpr auto sign() const {
            using T = decltype(m_A.eval_at(0,0,0,0));
            return (m_reflections_count % 2 == 0) ? T{1} : T{-1};
        }
        
        /*
        Get the number of Householder reflections applied
        */
        CUDA_HOST
        constexpr uint32_t reflection_count() const {
            return m_reflections_count;
        }

        // Getters for Q and R
        CUDA_HOST
        const auto& Q() const {
            return m_Q;
        }

        CUDA_HOST
        const auto& R() const {
            return m_R;
        }
    };

    template<ExprType E>
    CUDA_HOST
    [[nodiscard]] auto determinant(const E& expr) {
        static_assert(is_square_matrix_v<E>, "Determinant can only be computed for square matrices.");
        QRDecomposition qr{expr};
        qr.solve();
        return qr.get_determinant();
    }






    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////










    /*
    Solver for the linear equation Ax = b using QR Decomposition.
    */
    template<ExprType AType, ExprType BType>
    class LinearEquation : public Solver {
    public:

        LinearEquation(
            const AType& _A,
            const BType& _b
        ) : m_QR_of_A(QRDecomposition<AType>{_A}), m_b(_b), m_x() {
            static_assert(is_matrix_shape_v<AType>, "Coefficient matrix A must be a matrix.");
            static_assert(is_vector_v<BType>, "Right-hand side b must be a vector.");
            static_assert(AType::rows == BType::rows, "Incompatible dimensions between A and b in linear equation Ax = b.");
            m_QR_of_A.solve();
        };

        LinearEquation(
            QRDecomposition<AType>& _QR_of_A,
            const BType& _b
        ) : m_QR_of_A(_QR_of_A), m_b(_b), m_x() {
            static_assert(is_matrix_shape_v<AType>, "Coefficient matrix A must be a matrix.");
            static_assert(is_vector_v<BType>, "Right-hand side b must be a vector.");
            static_assert(AType::rows == BType::rows, "Incompatible dimensions between A and b in linear equation Ax = b.");
        };
        
        CUDA_HOST
        void solve() override {
            using T = decltype(m_x.eval_at(0,0,0,0));
            
            // Step 1: Compute y = Q^H * b (or Q^T * b for real matrices)
            VariableMatrix<T, AType::cols, 1> y;
            for (uint32_t i = 0; i < AType::cols; ++i) {
                T y_i = T{0};
                for (uint32_t j = 0; j < AType::rows; ++j) {
                    if constexpr (is_complex_v<T>) {
                        y_i += conj_value(m_QR_of_A.Q().eval_at(j, i)) * m_b.eval_at(j, 0);
                    } else {
                        y_i += m_QR_of_A.Q().eval_at(j, i) * m_b.eval_at(j, 0);
                    }
                }
                y.at(i, 0) = y_i;
            }
            
            // Step 2: Solve Rx = y by back substitution
            // Since R is upper triangular, start from bottom row and work up
            for (int i = static_cast<int>(AType::cols) - 1; i >= 0; --i) {
                T sum = y.eval_at(i, 0);
                
                // Subtract known terms: sum = y[i] - sum(R[i,j] * x[j]) for j > i
                for (uint32_t j = i + 1; j < AType::cols; ++j) {
                    sum -= m_QR_of_A.R().eval_at(i, j) * m_x.eval_at(j, 0);
                }
                
                // Divide by diagonal element: x[i] = sum / R[i,i]
                T r_ii = m_QR_of_A.R().eval_at(i, i);
                constexpr auto epsilon = static_cast<decltype(real_value(T{}))>(1e-10);
                
                if (abs_squared(r_ii) < epsilon) {
                    // Singular matrix - set x[i] to zero or handle appropriately
                    m_x.at(i, 0) = T{0};
                } else {
                    m_x.at(i, 0) = sum / r_ii;
                }
            }
        }
        
        // Get the solution vector
        CUDA_HOST
        const auto& solution() const {
            return m_x;
        }

    private:
        QRDecomposition<AType> m_QR_of_A;
        const BType& m_b;
        VariableMatrix<decltype(m_b.eval_at(0,0,0,0)), AType::cols, 1, 'x'> m_x;
    };






    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////








    /*
    Solver for computing eigenvalues of a square matrix A.
    */
    template<ExprType E> requires(is_square_matrix_v<E>)
    class EigenValues : public Solver {
    public:
        EigenValues(const E& _A) : m_A(_A), m_eigenvalues(E::rows) {
        }

        CUDA_HOST
        void solve() override {
            using T = decltype(m_A.eval_at(0,0,0,0));
            using RealT = decltype(real_value(T{}));
            constexpr uint32_t maxIterations = 1000;
            constexpr RealT tolerance = 1e-8;
            constexpr RealT epsilon = 1e-10;
            VariableMatrix<T, E::rows, E::cols> A_current = m_A.eval();
            
            for (uint32_t iter = 0; iter < maxIterations; ++iter) {
                // Compute Wilkinson shift from bottom-right 2x2 submatrix
                // For matrix [[a, b], [c, d]], compute eigenvalue closer to d
                T shift = T{0};
                if constexpr (E::rows >= 2) {
                    uint32_t n = E::rows - 1;  // Last index
                    T a = A_current.eval_at(n-1, n-1);
                    T b = A_current.eval_at(n-1, n);
                    T c = A_current.eval_at(n, n-1);
                    T d = A_current.eval_at(n, n);
                    
                    // Compute shift: eigenvalue of [[a,b],[c,d]] closest to d
                    // delta = (a - d) / 2
                    T delta = (a - d) / T{2};
                    
                    // discriminant = delta^2 + b*c
                    T discriminant = delta * delta + b * c;
                    
                    // For complex numbers or if discriminant is negative, use sqrt
                    // sign = sign(delta) or 1 if delta == 0
                    T sign_delta;
                    if constexpr (is_complex_v<T>) {
                        // For complex, use sign based on real part
                        RealT delta_real = real_value(delta);
                        sign_delta = (delta_real >= 0) ? T{1} : T{-1};
                    } else {
                        sign_delta = (delta >= 0) ? T{1} : T{-1};
                    }
                    
                    // Compute sqrt of discriminant
                    T sqrt_disc;
                    if constexpr (is_complex_v<T>) {
                        sqrt_disc = std::sqrt(discriminant);
                    } else {
                        // For real numbers, handle negative discriminant
                        if (discriminant >= 0) {
                            sqrt_disc = std::sqrt(discriminant);
                        } else {
                            // This shouldn't happen for real symmetric matrices
                            sqrt_disc = T{0};
                        }
                    }
                    
                    // Wilkinson shift: d - sign(delta) * |discriminant|^(1/2)
                    // More stable form: d - b*c / (delta + sign(delta)*sqrt(discriminant))
                    if (magnitude(delta) > epsilon || magnitude(sqrt_disc) > epsilon) {
                        shift = d - (b * c) / (delta + sign_delta * sqrt_disc);
                    } else {
                        shift = d;
                    }
                }
                
                // Apply shift: A_shifted = A_current - shift * I
                VariableMatrix<T, E::rows, E::cols> A_shifted = A_current;
                for (uint32_t i = 0; i < E::rows; ++i) {
                    A_shifted.at(i, i) -= shift;
                }
                
                // QR decomposition of shifted matrix
                auto qr = QRDecomposition{A_shifted};
                qr.solve();
                
                // Update: A_current = R*Q + shift*I
                A_current = qr.R() * qr.Q();
                for (uint32_t i = 0; i < E::rows; ++i) {
                    A_current.at(i, i) += shift;
                }
                
                // Check for convergence:
                bool converged = true;
                for (uint32_t i = 0; i < E::rows; ++i) {
                    for (uint32_t j = 0; j < E::cols; ++j) {
                        if (i != j) {
                            if (magnitude(A_current.eval_at(i, j)) > tolerance) {
                                converged = false;
                                break;
                            }
                        }
                    }
                    if (!converged) {
                        break;
                    }
                }
                if (m_print_progress && (iter % 100 == 0)) {
                    std::cout << "Eigenvalue computation iteration " << iter << "/" << maxIterations << "     \r";
                }
                if (converged) {
                    break;
                }
            }
            
            // Extract eigenvalues from the diagonal of A_current
            for (uint32_t i = 0; i < E::rows; ++i) {
                m_eigenvalues.push_back(A_current.eval_at(i, i));
            }
            std::sort(
                m_eigenvalues.begin(),
                m_eigenvalues.end(),
                [](const T& a, const T& b) {
                    return magnitude(a) > magnitude(b);
                }
            );
        }

        // Get the computed eigenvalues as a vector
        CUDA_HOST
        [[nodiscard]] const auto get_eigenvalues() const {
            return m_eigenvalues;
        }

    private:
        const E& m_A;
        std::vector<decltype(m_A.eval_at(0,0,0,0))> m_eigenvalues;
    };

    template<ExprType E>
    CUDA_HOST
    [[nodiscard]] auto eigenvalues(const E& expr) {
        static_assert(is_square_matrix_v<E>, "Eigenvalues can only be computed for square matrices.");
        auto eigen = EigenValues{expr};
        eigen.solve();
        return eigen.get_eigenvalues();
    }



    /*
    Solver for computing eigenvectors of a square matrix A.
    Uses the QR algorithm with Wilkinson shifts to compute eigenvectors.
    
    Supports both Hermitian and non-Hermitian matrices:
    - For Hermitian matrices: converges to diagonal form with orthogonal eigenvectors
    - For non-Hermitian matrices: converges to Schur form (upper triangular) with eigenvectors
    
    Leverages the existing QRDecomposition solver for efficient computation.
    */
    template<ExprType E> requires(is_square_matrix_v<E>)
    class EigenVectors : public Solver {
    public:
        EigenVectors(const E& _A) : m_A(_A) {
        }

        CUDA_HOST
        void solve() override {
            using T = decltype(m_A.eval_at(0,0,0,0));
            using RealT = decltype(real_value(T{}));
            constexpr uint32_t maxIterations = 1000;
            constexpr RealT tolerance = 1e-8;
            constexpr RealT epsilon = 1e-10;
            
            VariableMatrix<T, E::rows, E::cols> A_current = m_A.eval();
            
            // Initialize eigenvectors as identity matrix (each column as a separate vector)
            m_eigenvectors.clear();
            m_eigenvectors.reserve(E::cols);
            for (uint32_t j = 0; j < E::cols; ++j) {
                VariableMatrix<T, E::rows, 1> col;
                for (uint32_t i = 0; i < E::rows; ++i) {
                    col.at(i, 0) = (i == j) ? T{1} : T{0};
                }
                m_eigenvectors.push_back(col);
            }
            
            for (uint32_t iter = 0; iter < maxIterations; ++iter) {
                // Compute Wilkinson shift from bottom-right 2x2 submatrix
                T shift = T{0};
                if constexpr (E::rows >= 2) {
                    uint32_t n = E::rows - 1;  // Last index
                    T a = A_current.eval_at(n-1, n-1);
                    T b = A_current.eval_at(n-1, n);
                    T c = A_current.eval_at(n, n-1);
                    T d = A_current.eval_at(n, n);
                    
                    T delta = (a - d) / T{2};
                    T discriminant = delta * delta + b * c;
                    
                    T sign_delta;
                    if constexpr (is_complex_v<T>) {
                        RealT delta_real = real_value(delta);
                        sign_delta = (delta_real >= 0) ? T{1} : T{-1};
                    } else {
                        sign_delta = (delta >= 0) ? T{1} : T{-1};
                    }
                    
                    T sqrt_disc;
                    if constexpr (is_complex_v<T>) {
                        sqrt_disc = std::sqrt(discriminant);
                    } else {
                        if (discriminant >= 0) {
                            sqrt_disc = std::sqrt(discriminant);
                        } else {
                            sqrt_disc = T{0};
                        }
                    }
                    
                    if (magnitude(delta) > epsilon || magnitude(sqrt_disc) > epsilon) {
                        shift = d - (b * c) / (delta + sign_delta * sqrt_disc);
                    } else {
                        shift = d;
                    }
                }
                
                // Apply shift: A_shifted = A_current - shift * I
                VariableMatrix<T, E::rows, E::cols> A_shifted = A_current;
                for (uint32_t i = 0; i < E::rows; ++i) {
                    A_shifted.at(i, i) -= shift;
                }
                
                // QR decomposition of shifted matrix
                auto qr = QRDecomposition{A_shifted};
                qr.solve();
                
                // Update: A_current = R*Q + shift*I
                A_current = qr.R() * qr.Q();
                for (uint32_t i = 0; i < E::rows; ++i) {
                    A_current.at(i, i) += shift;
                }
                
                // Accumulate eigenvectors: V = V * Q
                // For each column j of the new eigenvector matrix
                std::vector<VariableMatrix<T, E::rows, 1>> V_new;
                V_new.reserve(E::cols);
                for (uint32_t j = 0; j < E::cols; ++j) {
                    VariableMatrix<T, E::rows, 1> new_col;
                    for (uint32_t i = 0; i < E::rows; ++i) {
                        T sum = T{0};
                        for (uint32_t k = 0; k < E::cols; ++k) {
                            sum += m_eigenvectors[k].eval_at(i, 0) * qr.Q().eval_at(k, j);
                        }
                        new_col.at(i, 0) = sum;
                    }
                    V_new.push_back(new_col);
                }
                m_eigenvectors = std::move(V_new);
                
                // Check for convergence to Schur form (upper triangular)
                // For Hermitian matrices: converges to diagonal (special case)
                // For non-Hermitian matrices: converges to upper triangular
                bool converged = true;
                for (uint32_t i = 0; i < E::rows; ++i) {
                    for (uint32_t j = 0; j < E::cols; ++j) {
                        // Only check lower triangular part (i > j)
                        if (i > j) {
                            if (magnitude(A_current.eval_at(i, j)) > tolerance) {
                                converged = false;
                                break;
                            }
                        }
                    }
                    if (!converged) {
                        break;
                    }
                }
                
                if (m_print_progress && (iter % 100 == 0)) {
                    std::cout << "Eigenvector computation iteration " << iter << "/" << maxIterations << "     \r";
                }
                
                if (converged) {
                    if (m_print_progress) {
                        std::cout << std::endl;
                    }
                    
                    // Extract eigenvalues from diagonal before extracting eigenvectors
                    std::vector<T> eigenvalues_unsorted;
                    eigenvalues_unsorted.reserve(E::cols);
                    for (uint32_t i = 0; i < E::rows; ++i) {
                        eigenvalues_unsorted.push_back(A_current.eval_at(i, i));
                    }
                    
                    // Extract eigenvectors from Schur form
                    // For upper triangular matrix T with V such that A = V*T*V^-1,
                    // the eigenvectors of A are V times the eigenvectors of T
                    extract_eigenvectors_from_schur(A_current);
                    
                    // Sort eigenvectors by eigenvalue magnitude to match EigenValues solver
                    std::vector<uint32_t> indices(E::cols);
                    for (uint32_t i = 0; i < E::cols; ++i) {
                        indices[i] = i;
                    }
                    std::sort(indices.begin(), indices.end(), [&](uint32_t a, uint32_t b) {
                        return magnitude(eigenvalues_unsorted[a]) > magnitude(eigenvalues_unsorted[b]);
                    });
                    
                    // Reorder eigenvectors according to sorted eigenvalues
                    std::vector<VariableMatrix<T, E::rows, 1>> sorted_eigenvectors;
                    sorted_eigenvectors.reserve(E::cols);
                    for (uint32_t i = 0; i < E::cols; ++i) {
                        sorted_eigenvectors.push_back(m_eigenvectors[indices[i]]);
                    }
                    m_eigenvectors = std::move(sorted_eigenvectors);
                    
                    break;
                }
            }
        }

    private:
        const E& m_A;
        std::vector<VariableMatrix<decltype(m_A.eval_at(0,0,0,0)), E::rows, 1>> m_eigenvectors;
        
        CUDA_HOST
        void extract_eigenvectors_from_schur(const VariableMatrix<decltype(m_A.eval_at(0,0,0,0)), E::rows, E::cols>& T) {
            using T_type = decltype(m_A.eval_at(0,0,0,0));
            using RealT = decltype(real_value(T_type{}));
            constexpr RealT epsilon = 1e-12;
            
            // For upper triangular Schur form T, eigenvalues are on the diagonal
            // To find eigenvector for T[k,k], we solve (T - T[k,k]*I)*x = 0
            // Since T is upper triangular, we can use back-substitution
            
            std::vector<VariableMatrix<T_type, E::rows, 1>> schur_eigenvectors;
            schur_eigenvectors.reserve(E::cols);
            
            for (uint32_t k = 0; k < E::cols; ++k) {
                T_type lambda = T.eval_at(k, k);
                VariableMatrix<T_type, E::rows, 1> x;
                
                // Initialize all to zero
                for (uint32_t i = 0; i < E::rows; ++i) {
                    x.at(i, 0) = T_type{0};
                }
                
                // For an upper triangular matrix, if λ = T[k,k], then:
                // - Row k of (T-λI) has T[k,k]-λ = 0 on diagonal
                // - So x[k] is the free variable; set it to 1
                // - Rows k+1, k+2, ... also have (T[j,j]-λ)*x[j] + terms with x[j+1], x[j+2], ... = 0
                //   These can be solved directly as x[j] remains 0 (since T[j,j] != λ for j > k)
                // - Rows 0, 1, ..., k-1 need back-substitution
                
                x.at(k, 0) = T_type{1};
                
                // For rows i from k-1 down to 0:
                // (T[i,i]-λ)*x[i] + sum(T[i,j]*x[j] for j=i+1 to n-1) = 0
                for (int i = static_cast<int>(k) - 1; i >= 0; --i) {
                    T_type sum = T_type{0};
                    for (uint32_t j = i + 1; j < E::rows; ++j) {
                        sum += T.eval_at(i, j) * x.eval_at(j, 0);
                    }
                    
                    T_type denom = T.eval_at(i, i) - lambda;
                    if (magnitude(denom) > epsilon) {
                        x.at(i, 0) = -sum / denom;
                    } else {
                        // Degenerate case - this shouldn't happen for distinct eigenvalues
                        x.at(i, 0) = T_type{0};
                    }
                }
                
                // Normalize
                RealT norm_sq = RealT{0};
                for (uint32_t i = 0; i < E::rows; ++i) {
                    norm_sq += abs_squared(x.eval_at(i, 0));
                }
                RealT norm = std::sqrt(norm_sq);
                if (norm > epsilon) {
                    for (uint32_t i = 0; i < E::rows; ++i) {
                        x.at(i, 0) /= norm;
                    }
                }
                
                schur_eigenvectors.push_back(x);
            }
            
            // Transform to original basis: eigenvector = V * schur_eigenvector
            std::vector<VariableMatrix<T_type, E::rows, 1>> final_eigenvectors;
            final_eigenvectors.reserve(E::cols);
            
            for (uint32_t k = 0; k < E::cols; ++k) {
                VariableMatrix<T_type, E::rows, 1> ev;
                for (uint32_t i = 0; i < E::rows; ++i) {
                    T_type sum = T_type{0};
                    for (uint32_t j = 0; j < E::cols; ++j) {
                        sum += m_eigenvectors[j].eval_at(i, 0) * schur_eigenvectors[k].eval_at(j, 0);
                    }
                    ev.at(i, 0) = sum;
                }
                
                final_eigenvectors.push_back(ev);
            }
            
            m_eigenvectors = std::move(final_eigenvectors);
        }

    public:
        // Get the computed eigenvectors as a vector of column vectors
        CUDA_HOST
        [[nodiscard]] const auto& get_eigenvectors() const {
            return m_eigenvectors;
        }
        
        // Get a specific eigenvector by index
        CUDA_HOST
        [[nodiscard]] const auto& get_eigenvector(uint32_t index) const {
            return m_eigenvectors[index];
        }
    };

    template<ExprType E>
    CUDA_HOST
    [[nodiscard]] auto eigenvectors(const E& expr) {
        static_assert(is_square_matrix_v<E>, "Eigenvectors can only be computed for square matrices.");
        auto eigen = EigenValues{expr};
        eigen.solve();
        return eigen.get_eigenvectors();
    }



    template<ExprType E>
    CUDA_HOST
    [[nodiscard]] auto invert(const E& expr) {
        static_assert(is_square_matrix_v<E>, "Matrix inversion can only be performed on square matrices.");
        using T = decltype(expr.eval_at(0,0,0,0));
        constexpr uint32_t N = E::rows;
        
        // Create identity matrix of same size
        VariableMatrix<T, N, N> I = VariableMatrix<T, N, N>::identity();
        
        // Solve Ax = I for each column of I
        VariableMatrix<T, N, N, 'I'> A_inv;
        for (uint32_t col = 0; col < N; ++col) {
            // Extract column vector b
            VariableMatrix<T, N, 1> b;
            for (uint32_t row = 0; row < N; ++row) {
                b.at(row, 0) = I.eval_at(row, col);
            }
            
            // Solve Ax = b
            LinearEquation<E, decltype(b)> lin_eq{expr, b};
            lin_eq.solve();
            const auto& x = lin_eq.solution();
            
            // Set column col of A_inv
            for (uint32_t row = 0; row < N; ++row) {
                A_inv.at(row, col) = x.eval_at(row, 0);
            }
        }
        
        return A_inv;
    }







}





    




