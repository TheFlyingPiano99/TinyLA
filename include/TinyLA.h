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
        e.eval(0, 0, 0, 0);
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


    template<ScalarType S, ExprType E>
    [[nodiscard]]
    CUDA_COMPATIBLE inline constexpr auto operator==(S other, const E& expr) {
        static_assert(is_scalar_shape_v<E>, "Only scalar shaped matrices can be compared to scalar values.");
        return static_cast<S>(expr.eval(0, 0)) == other;
    }

    template<ScalarType S, ExprType E>
    [[nodiscard]]
    CUDA_COMPATIBLE inline constexpr auto operator==(const E& expr, S other) {
        static_assert(is_scalar_shape_v<E>, "Only scalar shaped matrices can be compared to scalar values.");
        return static_cast<S>(expr.eval(0, 0)) == other;
    }

    template<ScalarType S, ExprType E>
    [[nodiscard]]
    CUDA_COMPATIBLE inline constexpr auto operator<(S other, const E& expr) {
        static_assert(is_scalar_shape_v<E>, "Only scalar shaped matrices can be compared to scalar values.");
        return other < static_cast<S>(expr.eval(0, 0));
    }

    template<ScalarType S, ExprType E>
    [[nodiscard]]
    CUDA_COMPATIBLE inline constexpr auto operator>(S other, const E& expr) {
        static_assert(is_scalar_shape_v<E>, "Only scalar shaped matrices can be compared to scalar values.");
        return other > static_cast<S>(expr.eval(0, 0));
    }

    template<ScalarType S, ExprType E>
    [[nodiscard]]
    CUDA_COMPATIBLE inline constexpr auto operator<(const E& expr, S other) {
        static_assert(is_scalar_shape_v<E>, "Only scalar shaped matrices can be compared to scalar values.");
        return static_cast<S>(expr.eval(0, 0)) < other;
    }

    template<ScalarType S, ExprType E>
    [[nodiscard]]
    CUDA_COMPATIBLE inline constexpr auto operator>(const E& expr, S other) {
        static_assert(is_scalar_shape_v<E>, "Only scalar shaped matrices can be compared to scalar values.");
        return static_cast<S>(expr.eval(0, 0)) > other;
    }


    template<class E, uint32_t Row, uint32_t Col, uint32_t Depth = 1, uint32_t Time = 1>
    class AbstractExpr {
    public:
        static constexpr bool variable_data = false;
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
        CUDA_COMPATIBLE inline constexpr auto eval(uint32_t r, uint32_t c = 0, uint32_t dr = 0, uint32_t dc = 0) const {
            return (static_cast<const E&>(*this)).eval(r, c, dr, dc);
        }

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto operator[](uint32_t i) requires(Row > 1 || Col > 1 || Depth > 1 || Time > 1);

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto at(uint32_t r, uint32_t c = 0, uint32_t dr = 0, uint32_t dc = 0);

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto x() const {
            static_assert(is_vector_v<E>, "x() can only be called on scalar or vector expressions.");
            return at(0, 0, 0, 0);
        }

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto y() const {
            static_assert(is_vector_v<E>, "y() can only be called on vector expressions.");
            if constexpr (E::rows == 1) {
                static_assert(E::cols >= 2, "y() called on row vector with less than 2 columns.");
                return at(0, 1, 0, 0);
            }
            else {
                static_assert(E::rows >= 2, "y() called on column vector with less than 2 rows.");
                return at(1, 0, 0, 0);
            }
        }

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto z() const {
            static_assert(is_vector_v<E>, "z() can only be called on vector expressions.");
            if constexpr (E::rows == 1) {
                static_assert(E::cols >= 3, "z() called on row vector with less than 3 columns.");
                return at(0, 2, 0, 0);
            }
            else {
                static_assert(E::rows >= 3, "z() called on column vector with less than 3 rows.");
                return at(2, 0, 0, 0);
            }
        }

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto w() const {
            static_assert(is_vector_v<E>, "w() can only be called on vector expressions.");
            if constexpr (E::rows == 1) {
                static_assert(E::cols >= 4, "w() called on row vector with less than 4 columns.");
                return at(0, 3, 0, 0);
            }
            else {
                static_assert(E::rows >= 4, "w() called on column vector with less than 4 rows.");
                return at(3, 0, 0, 0);
            }
        }

        template<ScalarType S>
        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr operator S() const {
            static_assert(is_scalar_shape_v<E>, "Only scalar shaped matrices can be assigned to scalar variables.");
            return static_cast<S>(static_cast<const E&>(*this).eval(0, 0, 0, 0));
        }

        template<ScalarType S>
        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr operator S() {
            static_assert(is_scalar_shape_v<E>, "Only scalar shaped matrices can be assigned to scalar variables.");
            return static_cast<S>(static_cast<const E&>(*this).eval(0, 0, 0, 0));
        }

        protected:
        template<ScalarType S>
        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr void __assign_at_if_applicable(S value, uint32_t r, uint32_t c) {
            static_assert(false, "Assignment to constant expressions is not allowed.");
        }

    };

    template<ExprType E, uint32_t Row, uint32_t Col, uint32_t Depth = 1, uint32_t Time = 1>
    class SubMatrixExpr : public AbstractExpr<SubMatrixExpr<E, Row, Col, Depth, Time>, Row, Col, Depth, Time> {
    public:
        static constexpr bool variable_data = E::variable_data;

        CUDA_COMPATIBLE inline constexpr SubMatrixExpr(E* expr, uint32_t row_offset = 0, uint32_t col_offset = 0, uint32_t depth_offset = 0, uint32_t time_offset = 0)
            : m_expr(*expr), m_row_offset(row_offset), m_col_offset(col_offset), m_depth_offset(depth_offset), m_time_offset(time_offset) {
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
            static_assert(E::variable_data, "Assignment to constant expressions is not allowed.");
            m_expr.__assign_at_if_applicable(m_expr.eval(m_row_offset, m_col_offset) + value, m_row_offset, m_col_offset, m_depth_offset, m_time_offset);
            return *this;
        }

        template<ScalarType S>
        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto operator-=(S value) {
            static_assert(is_scalar_shape_v<SubMatrixExpr>, "Assignment to submatrix is only supported for single element submatrices.");
            static_assert(E::variable_data, "Assignment to constant expressions is not allowed.");
            m_expr.__assign_at_if_applicable(m_expr.eval(m_row_offset, m_col_offset) - value, m_row_offset, m_col_offset, m_depth_offset, m_time_offset);
            return *this;
        }

        template<ScalarType S>
        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto operator*=(S value) {
            static_assert(is_scalar_shape_v<SubMatrixExpr>, "Assignment to submatrix is only supported for single element submatrices.");
            static_assert(E::variable_data, "Assignment to constant expressions is not allowed.");
            m_expr.__assign_at_if_applicable(m_expr.eval(m_row_offset, m_col_offset) * value, m_row_offset, m_col_offset, m_depth_offset, m_time_offset);
            return *this;
        }

        template<ScalarType S>
        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto operator/=(S value) {
            static_assert(is_scalar_shape_v<SubMatrixExpr>, "Assignment to submatrix is only supported for single element submatrices.");
            static_assert(E::variable_data, "Assignment to constant expressions is not allowed.");
            m_expr.__assign_at_if_applicable(m_expr.eval(m_row_offset, m_col_offset) / value, m_row_offset, m_col_offset, m_depth_offset, m_time_offset);
            return *this;
        }

        template<VarIDType varId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert(varId >= 0, "Variable ID for differentiation must be non-negative.");
            return SubMatrixExpr<decltype(m_expr.derivate<varId>()), Row, Col>(
                m_expr.derivate<varId>(), m_row_offset, m_col_offset, m_depth_offset, m_time_offset);
        }

        [[nodiscard]]
        CUDA_HOST constexpr inline std::string to_string() const {
            return std::format("SubMatrix[offset:{},{}, {}, {}]; size:{}x{}x{}x{}]({})", m_row_offset, m_col_offset, m_depth_offset, m_time_offset, Row, Col, Depth, Time, m_expr.to_string());
        }

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval(uint32_t r = 0, uint32_t c = 0, uint32_t d = 0, uint32_t t = 0) const {
            return m_expr.eval(r + m_row_offset, c + m_col_offset, d + m_depth_offset, t + m_time_offset);
        }

    private:
        template<ScalarType S>
        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr void __assign_at_if_applicable(S value, uint32_t r, uint32_t c, uint32_t d, uint32_t t) {
            m_expr.__assign_at_if_applicable(value, r + m_row_offset, c + m_col_offset, d + m_depth_offset, t + m_time_offset);
        }

        template<ExprType E, uint32_t Row, uint32_t Col, uint32_t Depth, uint32_t Time>
        friend class SubMatrixExpr;

        std::conditional_t<E::variable_data, E&, E> m_expr;
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
        ;
        if constexpr (Row > 1) {
            return SubMatrixExpr<E, 1, Col, Depth, Time>{static_cast<E*>(this), i, 0, 0, 0};
        }
        else if constexpr (Col > 1) {
            return SubMatrixExpr<E, 1, 1, Depth, Time>{static_cast<E*>(this), 0, i, 0, 0};
        }
        else if constexpr (Depth > 1) {
            return SubMatrixExpr<E, 1, 1, 1, Time>{static_cast<E*>(this), 0, 0, i, 0};
        }
        else if constexpr (Time > 1) {
            return SubMatrixExpr<E, 1, 1, 1, 1>{static_cast<E*>(this), 0, 0, 0, i};
        }
    }

    template<class E, uint32_t Row, uint32_t Col, uint32_t Depth, uint32_t Time>
    [[nodiscard]]
    CUDA_COMPATIBLE inline constexpr auto AbstractExpr<E, Row, Col, Depth, Time>::at(uint32_t r, uint32_t c, uint32_t d, uint32_t t) {
        return SubMatrixExpr<E, 1, 1, 1, 1>{static_cast<E*>(this), r, c, d, t};
    }
    
    template<VarIDType varId, ExprType E>
    [[nodiscard]]
    CUDA_COMPATIBLE constexpr auto derivate(const E& expr) {
        static_assert(varId >= 0, "Variable ID for differentiation must be non-negative.");
        return expr.derivate<varId>();
    }


    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    




    template<ScalarType T>
    class zero : public AbstractExpr<zero<T>, 1, 1, 1, 1> {
    public:

        static constexpr bool variable_data = false;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr zero() {}

        template<VarIDType diffVarId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert(diffVarId >= 0, "Variable IDs must be non-negative.");
            return zero<T>();
        }

        [[nodiscard]]
        CUDA_HOST constexpr inline std::string to_string() const {
            return std::string("0");
        }

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval(uint32_t r = 0, uint32_t c = 0, uint32_t d = 0, uint32_t t = 0) const {
            return T{};
        }
    };

    // Specialized trait to check if a type is a zero specialization
    template<class T>
    inline constexpr bool is_zero_v = false;

    template<ScalarType T>
    inline constexpr bool is_zero_v<zero<T>> = true;

    using fzero = zero<float>;
    using dzero = zero<double>;






    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////






    template<ScalarType T, uint32_t N>
    class identity : public AbstractExpr<identity<T, N>, N, N> {
    public:

        static constexpr bool variable_data = false;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr identity() {}

        template<VarIDType diffVarId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert(diffVarId >= 0, "Variable IDs must be non-negative.");
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
        CUDA_COMPATIBLE inline constexpr auto eval(uint32_t r = 0, uint32_t c = 0, uint32_t d = 0, uint32_t t = 0) const {
            return (r == c) ? T{ 1 } : T{};
        }
    };


    template<ScalarType T, uint32_t Row, uint32_t Col>
    class identityTensor : public AbstractExpr<identityTensor<T, Row, Col>, Row, Col, Row, Col> {
    public:

        static constexpr bool variable_data = false;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr identityTensor() {}

        template<VarIDType diffVarId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert(diffVarId >= 0, "Variable IDs must be non-negative.");
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
        CUDA_COMPATIBLE inline constexpr auto eval(uint32_t r, uint32_t c, uint32_t dr, uint32_t dc) const {
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

    using unit = identity<float, 1>;
    using identity1 = identity<float, 1>;
    using identity2 = identity<float, 2>;
    using identity3 = identity<float, 3>;
    using identity4 = identity<float, 4>;



    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////











    template<ScalarType T, uint32_t Row, uint32_t Col>
    class ones : public AbstractExpr<ones<T, Row, Col>, Row, Col> {
    public:

        static constexpr bool variable_data = false;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr ones() {}

        template<VarIDType diffVarId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert(diffVarId >= 0, "Variable IDs must be non-negative.");
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
        CUDA_COMPATIBLE inline constexpr auto eval(uint32_t r = 0, uint32_t c = 0) const {
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
    class FilledConstant : public AbstractExpr<FilledConstant<T, Row, Col, Depth, Time>, Row, Col, Depth, Time> {
    public:

        CUDA_COMPATIBLE inline constexpr FilledConstant(T value) : m_value(value) {}

        template<VarIDType varId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert(varId >= 0, "Variable IDs must be non-negative.");
            return zero<T>{};
        }

        [[nodiscard]]
        CUDA_HOST constexpr inline std::string to_string() const {
            return std::format("{}", m_value);
        }

        static constexpr bool variable_data = false;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval(uint32_t r = 0, uint32_t c = 0, uint32_t d = 0, uint32_t t = 0) const {
            return m_value;
        }

    private:
        T m_value;
    };

    //---------------------------------------------------------------------------------------
    // Math constants:

    /*
        Pi constant
    */
    template <ScalarType T>
    constexpr auto pi = FilledConstant<T, 1, 1, 1, 1>{ 3.14159265358979323846264338327950288419716939937510582097494459230781640628 };

    /*
        Euler number
    */
    template <ScalarType T>
    constexpr auto euler = FilledConstant<T, 1, 1, 1, 1>{ 2.718281828459045235360287471352662497757247093699959574966967627724076630353 };



    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////







    template<ScalarType T, uint32_t Row, uint32_t Col, VarIDType varId = 0>
    class VariableMatrix : public AbstractExpr<VariableMatrix<T, Row, Col, varId>, Row, Col> {
    public:

        static constexpr bool variable_data = true;

        [[nodiscard]]
        CUDA_COMPATIBLE static inline constexpr auto filled(T valueToFillWith) {
            auto m = VariableMatrix<T, Row, Col, varId>{};
            for (uint32_t c{}; c < Col; ++c) {
                for (uint32_t r{}; r < Row; ++r) {
                    m.m_data[c][r] = valueToFillWith;
                }
            }
            return m;
        }

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr VariableMatrix() : m_data{} {}

        template<class _SE>
        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr VariableMatrix(const AbstractExpr<_SE, Row, Col>& expr) {
            static_assert(!is_tensor_v<_SE>, "A matrix can not be initialized with a tensor.");
            for (size_t c = 0; c < (*this).cols; ++c) {
                for (size_t r = 0; r < (*this).rows; ++r) {
                    m_data[c][r] = expr.eval(r, c, 0, 0);
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
                    m_data[c][r] = val;
                    c++;
                    if (c >= Col) {
                        break;
                    }
                }
                for (; c < Col; ++c) {
                    m_data[c][r] = T{};
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
                    m_data[0][r] = val;
                    r++;
                    if (r >= Row) {
                        break;
                    }
                }
                for (; r < Row; ++r) {
                    m_data[0][r] = T{};
                }
            }
            else if constexpr (Row == 1) {
                size_t c = 0;
                for (const auto& val : values) {
                    m_data[c][0] = val;
                    c++;
                    if (c >= Col) {
                        break;
                    }
                }
                for (; c < Col; ++c) {
                    m_data[c][0] = T{};
                }
            }
        }

        template<class _SE>
        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto operator=(const AbstractExpr<_SE, Row, Col>& expr) {
            static_assert(!is_tensor_v<_SE>, "No tensor allowed.");
            for (uint32_t c = 0; c < Col; ++c) {
                for (uint32_t r = 0; r < Row; ++r) {
                    m_data[c][r] = expr.eval(r, c, 0, 0);
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
            static_assert(derivationVarId > 0, "Variable IDs must be non-negative.");
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
                        strStream << m_data[c][r];
                    }
                    strStream << " |" << std::endl;
                }
                return (varId == 0) ? strStream.str() : std::format("mat{}x{}_{}", (*this).rows, (*this).cols, char32_to_utf8(varId));
            }
        }

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval(uint32_t r = 0, uint32_t c = 0, uint32_t dr = 0, uint32_t dc = 0) const {
            if constexpr (Row == 1 && Col == 1) {   // Behave as scalar
                return m_data[0][0];
            }
#ifndef ENABLE_CUDA_SUPPORT
            if (r >= Row || c >= Col) {
                throw std::out_of_range("Matrix index out of range.");
            }
#endif
            return m_data[c][r];
        }

        private:
        template<ScalarType S>
        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr void __assign_at_if_applicable(S value, uint32_t r, uint32_t c, uint32_t d = 0, uint32_t t = 0) {
            m_data[c][r] = static_cast<T>(value);
        }

        template<ExprType E, uint32_t Row, uint32_t Col, uint32_t Depth, uint32_t Time>
        friend class SubMatrixExpr;

        T m_data[Col][Row]; // Column-major storage
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
    using mat_var = VariableMatrix<T, R, C>;
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







    template<class E1, class E2> requires(is_elementwise_broadcastable_v<E1, E2>)
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
                static_assert(varId >= 0, "Variable ID for differentiation must be non-negative.");
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

            static constexpr bool variable_data = false;

            [[nodiscard]]
            CUDA_COMPATIBLE inline constexpr auto eval(uint32_t r = 0, uint32_t c = 0, uint32_t d = 0, uint32_t t = 0) const {
                using common_type = common_arithmetic_t<decltype(m_expr1.eval(r, c, d, t)), decltype(m_expr2.eval(r, c, d, t))>;
                if constexpr (is_zero_v<E1> && is_zero_v<E2>) {
                    return common_type{};
                }
                else if constexpr (is_zero_v<E1>) {
                    return static_cast<common_type>(m_expr2.eval(r, c, d, t));
                }
                else if constexpr (is_zero_v<E2>) {
                    return static_cast<common_type>(m_expr1.eval(r, c, d, t));
                }
                else {
                    return static_cast<common_type>(m_expr1.eval(r, c, d, t)) + static_cast<common_type>(m_expr2.eval(r, c, d, t));
                }
            }

        private:
            std::conditional_t< (E1::variable_data), const E1&, const E1> m_expr1;
            std::conditional_t< (E2::variable_data), const E2&, const E2> m_expr2;
    };

    template<ExprType E1, ExprType E2>
    CUDA_COMPATIBLE
        [[nodiscard]] constexpr auto operator+(const E1& expr1, const E2& expr2) {
        static_assert(is_elementwise_broadcastable_v<E1, E2>,
            "Incompatible matrix dimensions for element-wise addition.\nMatrices must have the same shape or one of them must be a scalar for elementwise broadcasting."
            );
        return AdditionExpr<E1, E2>{expr1, expr2};
    }

    template<ExprType E, ScalarType S>
    CUDA_COMPATIBLE
        [[nodiscard]] constexpr auto operator+(const E& expr, S a) {
        return AdditionExpr<E, FilledConstant<S, E::rows, E::cols, E::depth, E::time>>{expr, FilledConstant<S, E::rows, E::cols, E::depth, E::time>{a}};
    }

    template<ScalarType S, ExprType E>
    CUDA_COMPATIBLE
        [[nodiscard]] constexpr auto operator+(S a, const E& expr) {
        return AdditionExpr<FilledConstant<S, E::rows, E::cols, E::depth, E::time>, E>{FilledConstant<S, E::rows, E::cols, E::depth, E::time>{a}, expr};
    }





    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////








    template<class E>
    class NegationExpr : public AbstractExpr<NegationExpr<E>, E::rows, E::cols, E::depth, E::time> {
    public:

        CUDA_COMPATIBLE inline constexpr NegationExpr(const E& expr) : m_expr(expr) {}

        template<VarIDType varId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert(varId >= 0, "Variable ID for differentiation must be non-negative.");
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

        static constexpr bool variable_data = false;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval(uint32_t r = 0, uint32_t c = 0, uint32_t d = 0, uint32_t t = 0) const {
            if constexpr (is_zero_v<E>) {
                return 0;
            }
            else {
                return -m_expr.eval(r, c, d, t);
            }
        }

    private:
        std::conditional_t< (E::variable_data), const E&, const E> m_expr;
    };

    template<ExprType E>
    CUDA_COMPATIBLE
        [[nodiscard]] constexpr auto operator-(const E& expr) {
        return NegationExpr<E>{expr};
    }





    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////








    template<class E>
    class TransposeExpr : public AbstractExpr<TransposeExpr<E>, E::cols, E::rows, E::depth, E::time> {
    public:

        CUDA_COMPATIBLE inline constexpr TransposeExpr(const E& expr) : m_expr(expr) {}

        template<VarIDType varId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert(varId >= 0, "Variable ID for differentiation must be non-negative.");
            auto expr_derivative = m_expr.derivate<varId>();
            if constexpr (is_zero_v<decltype(expr_derivative)> || is_identity_v<decltype(expr_derivative)>) {
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

        static constexpr bool variable_data = false;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval(uint32_t r, uint32_t c, uint32_t d = 0, uint32_t t = 0) const {
            return m_expr.eval(c, r, t, d);
        }

    private:
        std::conditional_t< (E::variable_data), const E&, const E> m_expr;
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
            static_assert(varId >= 0, "Variable ID for differentiation must be non-negative.");
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

        static constexpr bool variable_data = false;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval(uint32_t r, uint32_t c, uint32_t d = 0, uint32_t t = 0) const {
            if (r == c) {
                if constexpr (is_column_vector_v<E>) {
                    return m_expr.eval(r, 0, d, t);
                }
                else {
                    return m_expr.eval(0, c, d, t);
                }
            }
            else {
                return m_nondiagonal_filler;  // Zero
            }
        }

    private:
        std::conditional_t< (E::variable_data), const E&, const E> m_expr;
        decltype(m_expr.eval(0, 0)) m_nondiagonal_filler = decltype(m_expr.eval(0, 0)){};
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
            static_assert(varId >= 0, "Variable ID for differentiation must be non-negative.");
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

        static constexpr bool variable_data = false;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval(uint32_t r, uint32_t c, uint32_t d = 0, uint32_t t = 0) const {
            if (r == d && c == t) {
                return m_expr.eval(r, c, 0, 0);
            }
            else {
                return m_nondiagonal_filler;  // Zero
            }
        }

    private:
        std::conditional_t< (E::variable_data), const E&, const E> m_expr;
        decltype(m_expr.eval(0, 0)) m_nondiagonal_filler = decltype(m_expr.eval(0, 0)){};
    };






    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////








    template<class E>
    class ConjugateExpr : public AbstractExpr<ConjugateExpr<E>, E::rows, E::cols, E::depth, E::time> {
    public:

        CUDA_COMPATIBLE inline constexpr ConjugateExpr(const E& expr) : m_expr(expr) {}

        template<VarIDType varId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert(varId >= 0, "Variable ID for differentiation must be non-negative.");
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

        static constexpr bool variable_data = false;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval(uint32_t r = 0, uint32_t c = 0, uint32_t d = 0, uint32_t t = 0) const {
            if constexpr (ComplexType<decltype(m_expr.eval(r, c, d, t))>) {
                return conj(m_expr.eval(r, c, d, t));
            }
            else {
                return m_expr.eval(r, c, d, t);
            }
        }

    private:
        std::conditional_t< (E::variable_data), const E&, const E> m_expr;
    };

    template<ExprType E>
    CUDA_COMPATIBLE
        [[nodiscard]] constexpr auto conj(const E& expr) {
        return ConjugateExpr<E>{expr};
    }





    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////







    template<class E>
    class AdjointExpr : public AbstractExpr<AdjointExpr<E>, E::cols, E::rows, E::time, E::depth> {
    public:

        CUDA_COMPATIBLE inline constexpr AdjointExpr(const E& expr) : m_expr(expr) {}

        template<VarIDType varId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert(varId >= 0, "Variable ID for differentiation must be non-negative.");
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

        static constexpr bool variable_data = false;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval(uint32_t r, uint32_t c, uint32_t d = 0, uint32_t t = 0) const {
            if constexpr (ComplexType<decltype(m_expr.eval(c, r, t, d))>) {
                return conj(m_expr.eval(c, r, t, d));
            }
            else {
                return m_expr.eval(c, r, t, d);
            }
        }

    private:
        std::conditional_t< (E::variable_data), const E&, const E> m_expr;
    };

    template<ExprType E>
    CUDA_COMPATIBLE
        [[nodiscard]] constexpr auto adjoint(const E& expr) {
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
                static_assert(varId >= 0, "Variable ID for differentiation must be non-negative.");
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

            static constexpr bool variable_data = false;

            [[nodiscard]]
            CUDA_COMPATIBLE inline constexpr auto eval(uint32_t r = 0, uint32_t c = 0, uint32_t d = 0, uint32_t t = 0) const {
                using common_type = common_arithmetic_t<decltype(m_expr1.eval(r, c, d, t)), decltype(m_expr2.eval(r, c, d, t))>;
                if constexpr (is_zero_v<E1> && is_zero_v<E2>) {
                    return common_type{};
                }
                else if constexpr (is_zero_v<E1>) {
                    return static_cast<common_type>(-m_expr2.eval(r, c, d, t));
                }
                else if constexpr (is_zero_v<E2>) {
                    return static_cast<common_type>(m_expr1.eval(r, c, d, t));
                }
                else {
                    return static_cast<common_type>(m_expr1.eval(r, c, d, t)) - static_cast<common_type>(m_expr2.eval(r, c, d, t));
                }
            }

        private:
            std::conditional_t< (E1::variable_data), const E1&, const E1> m_expr1;
            std::conditional_t< (E2::variable_data), const E2&, const E2> m_expr2;
    };

    template<ExprType E1, ExprType E2>
    CUDA_COMPATIBLE
        [[nodiscard]] constexpr auto operator-(const E1& expr1, const E2& expr2) {
        static_assert((is_elementwise_broadcastable_v<E1, E2>),
            "Incompatible matrix dimensions for element-wise subtraction.\nMatrices must have the same shape."
            );
        return SubtractionExpr<E1, E2>{expr1, expr2};
    }

    template<ExprType E, ScalarType S>
    CUDA_COMPATIBLE
        [[nodiscard]] constexpr auto operator-(const E& expr, S a) {
        return SubtractionExpr<E, FilledConstant<S, E::rows, E::cols>>{expr, FilledConstant<S, E::rows, E::cols>{a}};
    }

    template<ScalarType S, ExprType E>
    CUDA_COMPATIBLE
        [[nodiscard]] constexpr auto operator-(S a, const E& expr) {
        return SubtractionExpr<FilledConstant<S, E::rows, E::cols>, E>{FilledConstant<S, E::rows, E::cols>{a}, expr};
    }




    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////








    template<ExprType E1, ExprType E2> requires(is_elementwise_broadcastable_v<E1, E2>)
        class ElementwiseProductExpr : public AbstractExpr<ElementwiseProductExpr<E1, E2>,
        std::conditional_t<(E1::rows > E2::rows), E1, E2>::rows,
        std::conditional_t<(E1::cols > E2::cols), E1, E2>::cols,
        std::conditional_t<(E1::depth > E2::depth), E1, E2>::depth,
        std::conditional_t<(E1::time > E2::time), E1, E2>::time
    > {
        public:

            CUDA_COMPATIBLE inline constexpr ElementwiseProductExpr(const E1& expr1, const E2& expr2) : m_expr1(expr1), m_expr2(expr2) {
            }

            template<VarIDType varId>
            [[nodiscard]]
            CUDA_COMPATIBLE constexpr inline auto derivate() const {
                static_assert((varId >= 0), "Variable ID for differentiation must be non-negative.");
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
                    if constexpr (is_tensor_v<decltype(expr2_derivative)>) {
                        return ElementwiseProductExpr<
                            DiagonalTensor<std::remove_cvref_t<decltype(m_expr1)>>,
                            decltype(expr2_derivative)
                        >{
                            DiagonalTensor{m_expr1},
                            expr2_derivative
                        };
                    }
                    else {
                        return ElementwiseProductExpr<
                            DiagonalMatrix<std::remove_cvref_t<decltype(m_expr1)>>,
                            decltype(expr2_derivative)
                        >{
                            DiagonalMatrix{m_expr1},
                            expr2_derivative
                        };
                    }
                }
                else if constexpr (is_zero_v<decltype(expr2_derivative)>) {
                    if constexpr (is_tensor_v<decltype(expr1_derivative)>) {
                        return ElementwiseProductExpr<
                            decltype(expr1_derivative),
                            DiagonalTensor<std::remove_cvref_t<decltype(m_expr2)>>
                        >{
                            expr1_derivative,
                            DiagonalTensor{m_expr2}
                        };
                    }
                    return ElementwiseProductExpr<
                        decltype(expr1_derivative),
                        DiagonalMatrix<std::remove_cvref_t<decltype(m_expr2)>>
                    >{
                        expr1_derivative,
                        DiagonalMatrix{m_expr2}
                    };
                }
                else {
                    if constexpr (is_tensor_v<decltype(expr1_derivative)> || is_tensor_v<decltype(expr2_derivative)>) {
                        return AdditionExpr{
                            ElementwiseProductExpr<
                                DiagonalTensor<std::remove_cvref_t<decltype(m_expr1)>>,
                                decltype(expr2_derivative)
                            > {
                                DiagonalTensor{m_expr1},
                                expr2_derivative
                            },
                            ElementwiseProductExpr<
                                decltype(expr1_derivative),
                                DiagonalTensor<std::remove_cvref_t<decltype(m_expr2)>>
                            > {
                                expr1_derivative,
                                DiagonalTensor{m_expr2}
                            }
                        };
                    }
                    else {
                        return AdditionExpr{
                            ElementwiseProductExpr<
                                DiagonalMatrix<std::remove_cvref_t<decltype(m_expr1)>>,
                                decltype(expr2_derivative)
                            > {
                                DiagonalMatrix{m_expr1},
                                expr2_derivative
                            },
                            ElementwiseProductExpr<
                                decltype(expr1_derivative),
                                DiagonalMatrix<std::remove_cvref_t<decltype(m_expr2)>>
                            > {
                                expr1_derivative,
                                DiagonalMatrix{m_expr2}
                            }
                        };
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

            static constexpr bool variable_data = false;

            [[nodiscard]]
            CUDA_COMPATIBLE inline constexpr auto eval(uint32_t r = 0, uint32_t c = 0, uint32_t d = 0, uint32_t t = 0) const {
                using common_type = common_arithmetic_t<decltype(m_expr1.eval(r, c, d, t)), decltype(m_expr2.eval(r, c, d, t))>;
                if constexpr (is_zero_v<E1> || is_zero_v<E2>) {
                    return common_type{};
                }
                else if constexpr (is_identity_v<E1> && is_identity_v<E2>) {
                    return common_type{ 1 };
                }
                else if constexpr (is_identity_v<E1>) {
                    return static_cast<common_type>(m_expr2.eval(r, c, d, t));
                }
                else if constexpr (is_identity_v<E2>) {
                    return static_cast<common_type>(m_expr1.eval(r, c, d, t));
                }
                else {
                    return static_cast<common_type>(m_expr1.eval(r, c, d, t)) * static_cast<common_type>(m_expr2.eval(r, c, d, t));
                }
            }

        private:
            std::conditional_t< (E1::variable_data), const E1&, const E1> m_expr1;
            std::conditional_t< (E2::variable_data), const E2&, const E2> m_expr2;
    };

    template<ExprType E1, ExprType E2> requires(is_scalar_shape_v<E1> || is_scalar_shape_v<E2>)
        CUDA_COMPATIBLE
        [[nodiscard]] constexpr auto operator*(const E1& expr1, const E2& expr2) {
        static_assert(is_elementwise_broadcastable_v<E1, E2>,
            "Incompatible matrix dimensions for element-wise product.\nMatrices must have the same shape or one of them must be a scalar for elementwise broadcasting."
            );
        return ElementwiseProductExpr<E1, E2>{expr1, expr2};
    }

    template<ExprType E1, ExprType E2>
    CUDA_COMPATIBLE
        [[nodiscard]] constexpr auto elementwiseProduct(const E1& expr1, const E2& expr2) {
        static_assert(is_elementwise_broadcastable_v<E1, E2>,
            "Incompatible matrix dimensions for element-wise product.\nMatrices must have the same shape or one of them must be a scalar for elementwise broadcasting."
            );
        return ElementwiseProductExpr<E1, E2>{expr1, expr2};
    }

    template<ExprType E, ScalarType S>
    CUDA_COMPATIBLE
        [[nodiscard]] constexpr auto operator*(const E& expr, S a) {
        return ElementwiseProductExpr<E, FilledConstant<S, E::rows, E::cols, E::depth, E::time>>{expr, FilledConstant<S, E::rows, E::cols, E::depth, E::time>{a}};
    }

    template<ScalarType S, ExprType E>
    CUDA_COMPATIBLE
        [[nodiscard]] constexpr auto operator*(S a, const E& expr) {
        return ElementwiseProductExpr<FilledConstant<S, E::rows, E::cols, E::depth, E::time>, E>{FilledConstant<S, E::rows, E::cols, E::depth, E::time>{a}, expr};
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
                if constexpr (is_identity_v<decltype(expr1_derivative)> && is_identity_v<decltype(expr2_derivative)>) {
                    return AdditionExpr<std::remove_cvref_t<decltype(m_expr1)>, std::remove_cvref_t<decltype(m_expr2)>>{
                        TransposeExpr{m_expr1},
                        m_expr2
                    };
                }
                else if constexpr (is_zero_v<decltype(expr1_derivative)> && is_zero_v<decltype(expr2_derivative)>) {
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
                        TransposeExpr{MatrixMultiplicationExpr<
                            std::remove_cvref_t<decltype(m_expr1)>,
                            decltype(expr2_derivative)
                        > {
                            m_expr1,
                            expr2_derivative
                        }}
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

            static constexpr bool variable_data = false;

            [[nodiscard]]
            CUDA_COMPATIBLE inline constexpr auto eval(uint32_t r, uint32_t c, uint32_t d = 0, uint32_t t = 0) const {
                using common_type = common_arithmetic_t<decltype(m_expr1.eval(r, c, d, t)), decltype(m_expr2.eval(r, c, d, t))>;
                if constexpr (is_zero_v<E1> || is_zero_v<E2>) {
                    return common_type{};
                }
                else if constexpr (is_identity_v<E1> && is_identity_v<E2>) {
                    return identity<common_type, this->rows>{};
                }
                else if constexpr (is_identity_v<E1>) {
                    return static_cast<common_type>(m_expr2.eval(r, c, d, t));
                }
                else if constexpr (is_identity_v<E2>) {
                    return static_cast<common_type>(m_expr1.eval(r, c, d, t));
                }
                else {
                    auto sum = common_type{};
                    for (uint32_t k = 0; k < E1::cols; ++k)
                        sum += static_cast<common_type>(m_expr1.eval(r, k, d, t)) * static_cast<common_type>(m_expr2.eval(k, c, d, t));
                    return sum;
                }
            }

        private:
            std::conditional_t< (E1::variable_data), const E1&, const E1> m_expr1;
            std::conditional_t< (E2::variable_data), const E2&, const E2> m_expr2;
    };

    template<ExprType E1, ExprType E2> requires(!is_scalar_shape_v<E1> && !is_scalar_shape_v<E2>)
        CUDA_COMPATIBLE
        [[nodiscard]] constexpr auto operator*(const E1& expr1, const E2& expr2) {
        static_assert(is_matrix_multiplicable_v<E1, E2>,
            "Incompatible matrix dimensions for matrix multiplication.\nNumber of columns of the first matrix must equal the number of rows of the second matrix."
            );
        return MatrixMultiplicationExpr<E1, E2>{expr1, expr2};
    }




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
                if constexpr (is_identity_v<decltype(expr1_derivative)> && is_identity_v<decltype(expr2_derivative)>) {
                    return AdditionExpr<std::remove_cvref_t<decltype(m_expr1)>, std::remove_cvref_t<decltype(m_expr2)>>{
                        TransposeExpr{m_expr1},
                        m_expr2
                    };
                }
                else if constexpr (is_zero_v<decltype(expr1_derivative)> && is_zero_v<decltype(expr2_derivative)>) {
                    return expr1_derivative;
                }
                else if constexpr (is_zero_v<decltype(expr1_derivative)>) {
                    return TransposeExpr{MatrixMultiplicationExpr<
                        std::remove_cvref_t<decltype(m_expr1)>,
                        ConjugateExpr<decltype(expr2_derivative)>
                    > {
                        m_expr1,
                        ConjugateExpr{expr2_derivative}
                    }};
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
                        TransposeExpr{MatrixMultiplicationExpr<
                            std::remove_cvref_t<decltype(m_expr1)>,
                            ConjugateExpr<decltype(expr2_derivative)>
                        > {
                            m_expr1,
                            ConjugateExpr{expr2_derivative}
                        }}
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

            static constexpr bool variable_data = false;

            [[nodiscard]]
            CUDA_COMPATIBLE inline constexpr auto eval(uint32_t r = 0, uint32_t c = 0, uint32_t d = 0, uint32_t t = 0) const {
                using common_type = common_arithmetic_t<decltype(m_expr1.eval(r, c, d, t)), decltype(m_expr2.eval(r, c, d, t))>;
                if constexpr (is_zero_v<E1> || is_zero_v<E2>) {
                    return common_type{};
                }
                else {
                    auto sum = common_type{};
                    for (uint32_t k = 0; k < E1::cols; ++k) {
                        if constexpr (ComplexType<decltype(m_expr2.eval(0, 0, d, t))>) {
                            sum += static_cast<common_type>(m_expr1.eval(r, k, d, t)) * static_cast<common_type>(std::conj(m_expr2.eval(k, c, d, t)));
                        }
                        else {
                            sum += static_cast<common_type>(m_expr1.eval(r, k, d, t)) * static_cast<common_type>(m_expr2.eval(k, c, d, t));
                        }
                    }
                    return sum;
                }
            }

        private:
            std::conditional_t< (E1::variable_data), const E1&, const E1> m_expr1;
            std::conditional_t< (E2::variable_data), const E2&, const E2> m_expr2;
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
                static_assert(false, "Unimplemented function");
                //TODO
                return zero<float>{};
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

            static constexpr bool variable_data = false;

            [[nodiscard]]
            CUDA_COMPATIBLE inline constexpr auto eval(uint32_t r = 0, uint32_t c = 0, uint32_t d = 0, uint32_t t = 0) const {
                using common_type = common_arithmetic_t<decltype(m_expr1.eval(r, c, d, t)), decltype(m_expr2.eval(r, c, d, t))>;
                if constexpr (is_zero_v<E1> || is_zero_v<E2>) {
                    return common_type{};
                }
                else {
                    static_assert(false, "Unimplemented function");
                }
            }

        private:
            std::conditional_t< (E1::variable_data), const E1&, const E1> m_expr1;
            std::conditional_t< (E2::variable_data), const E2&, const E2> m_expr2;
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












    template<class E1, class E2> requires(is_elementwise_broadcastable_v<E1, E2>)
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
                static_assert((varId >= 0), "Variable ID for differentiation must be non-negative.");
                auto expr1_derivative = m_expr1.derivate<varId>();
                auto expr2_derivative = m_expr2.derivate<varId>();
                if constexpr (is_zero_v<decltype(expr1_derivative)> && is_zero_v<decltype(expr2_derivative)>) {
                    return expr1_derivative;
                }
                else if constexpr (is_zero_v<decltype(expr1_derivative)>) {
                    auto numerator = NegationExpr{
                        ElementwiseProductExpr{
                            DiagonalMatrix{m_expr1},
                            expr2_derivative
                        }
                    };
                        auto denominator = DiagonalMatrix{ElementwiseProductExpr<
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
                else if constexpr (is_zero_v<decltype(expr2_derivative)>) {
                    auto numerator = 
                        ElementwiseProductExpr<
                            decltype(expr1_derivative),
                            DiagonalMatrix<std::remove_cvref_t<decltype(m_expr2)>>
                        > {
                            expr1_derivative,
                            DiagonalMatrix{m_expr2}
                        };
                    auto denominator = DiagonalMatrix{ElementwiseProductExpr<
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
                else {
                    auto numerator = SubtractionExpr{
                        ElementwiseProductExpr<
                            decltype(expr1_derivative),
                            DiagonalMatrix<std::remove_cvref_t<decltype(m_expr2)>>
                        > {
                            expr1_derivative,
                            DiagonalMatrix{m_expr2}
                        },
                        ElementwiseProductExpr<
                            DiagonalMatrix<std::remove_cvref_t<decltype(m_expr1)>>,
                            decltype(expr2_derivative)
                        > {
                            DiagonalMatrix{m_expr1},
                            expr2_derivative
                        }
                    };
                    auto denominator = DiagonalMatrix{ElementwiseProductExpr<
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

            static constexpr bool variable_data = false;

            [[nodiscard]]
            CUDA_COMPATIBLE inline constexpr auto eval(uint32_t r = 0, uint32_t c = 0, uint32_t d = 0, uint32_t t = 0) const {
                static_assert(!is_zero_v<E2>, "Division by zero in scalar expression.");

                using common_type = common_arithmetic_t<decltype(m_expr1.eval(r, c, d, t)), decltype(m_expr2.eval(r, c, d, t))>;
                if constexpr (is_identity_v<E2>) {
                    return static_cast<common_type>(m_expr1.eval(r, c, d, t));
                }
                else {
                    return static_cast<common_type>(m_expr1.eval(r, c, d, t)) / static_cast<common_type>(m_expr2.eval(r, c, d, t));
                }
            }

        private:
            std::conditional_t< (E1::variable_data), const E1&, const E1> m_expr1;
            std::conditional_t< (E2::variable_data), const E2&, const E2> m_expr2;
    };

    template<ExprType E1, ExprType E2>
    CUDA_COMPATIBLE
        [[nodiscard]] constexpr auto operator/(const E1& expr1, const E2& expr2) {
        static_assert(is_elementwise_broadcastable_v<E1, E2>,
            "Incompatible matrix dimensions for element-wise division.\nMatrices must have the same shape or one of them must be a scalar for elementwise broadcasting."
            );
        return DivisionExpr<E1, E2>{expr1, expr2};
    }

    template<ExprType E, ScalarType S>
    CUDA_COMPATIBLE
        [[nodiscard]] constexpr auto operator/(const E& expr, S a) {
        return DivisionExpr<E, FilledConstant<S, E::rows, E::cols, E::depth, E::time>>{expr, FilledConstant<S, E::rows, E::cols, E::depth, E::time>{a}};
    }

    template<ScalarType S, ExprType E>
    CUDA_COMPATIBLE
        [[nodiscard]] constexpr auto operator/(S a, const E& expr) {
        return DivisionExpr<FilledConstant<S, E::rows, E::cols, E::depth, E::time>, E>{FilledConstant<S, E::rows, E::cols, E::depth, E::time>{a}, expr};
    }




    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////








    template<class E>
    class NaturalLogExpr : public AbstractExpr<NaturalLogExpr<E>, E::rows, E::cols, E::depth, E::time> {
    public:

        CUDA_COMPATIBLE inline constexpr NaturalLogExpr(const E& expr) : m_expr(expr) {}

        template<VarIDType varId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert(varId >= 0, "Variable ID for differentiation must be non-negative.");
            auto expr_derivative = m_expr.derivate<varId>();
            if constexpr (is_zero_v<decltype(expr_derivative)>) {
                return expr_derivative;
            }
            else {
                return DivisionExpr<
                    decltype(expr_derivative),
                    std::remove_cvref_t<decltype(m_expr)>
                >{
                    expr_derivative,
                        m_expr
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

        static constexpr bool variable_data = false;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval(uint32_t r = 0, uint32_t c = 0, uint32_t d = 0, uint32_t t = 0) const {
            auto expr_value = m_expr.eval(r, c, d, t);
#ifndef ENABLE_CUDA_SUPPORT
            if constexpr (is_zero_v<E>) {
                throw std::runtime_error("[Logarithm of zero in scalar expression.]");
            }
#endif
            return log(expr_value);
        }

    private:
        std::conditional_t<(E::variable_data), const E&, const E> m_expr;
    };

    template<ExprType E>
    CUDA_COMPATIBLE
        [[nodiscard]] constexpr auto log(const E& expr) {
        return NaturalLogExpr<E>{expr};
    }




    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////







    template<class E>
    class NaturalExpExpr : public AbstractExpr<NaturalExpExpr<E>, E::rows, E::cols, E::depth, E::time> {
    public:

        CUDA_COMPATIBLE inline constexpr NaturalExpExpr(const E& expr) : m_expr(expr) {}

        template<VarIDType varId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert(varId >= 0, "Variable ID for differentiation must be non-negative.");
            auto expr_derivative = m_expr.derivate<varId>();
            if constexpr (is_zero_v<decltype(expr_derivative)>) {
                return expr_derivative;
            }
            else {
                return ElementwiseProductExpr<
                    NaturalExpExpr<E>,
                    decltype(expr_derivative)
                > {
                    NaturalExpExpr<E>{
                        m_expr
                    },
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

        static constexpr bool variable_data = false;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval(uint32_t r = 0, uint32_t c = 0, uint32_t d = 0, uint32_t t = 0) const {
            return exp(m_expr.eval(r, c, d, t));
        }

    private:
        std::conditional_t<(E::variable_data), const E&, const E> m_expr;
    };

    template<ExprType E>
    CUDA_COMPATIBLE
        [[nodiscard]] constexpr auto exp(const E& expr) {
        return NaturalExpExpr<E>{expr};
    }




    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////




    template<class E>
    class CosExpr;

    template<class E>
    class SinExpr : public AbstractExpr<SinExpr<E>, E::rows, E::cols> {
    public:

        CUDA_COMPATIBLE inline constexpr SinExpr(const E& expr) : m_expr(expr) {}

        template<VarIDType varId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert(varId >= 0, "Variable ID for differentiation must be non-negative.");
            auto expr_derivative = m_expr.derivate<varId>();
            if constexpr (is_zero_v<decltype(expr_derivative)>) {
                return expr_derivative;
            }
            else {
                return ElementwiseProductExpr<
                    DiagonalTensor<CosExpr<E>>,
                    decltype(expr_derivative)
                > {
                    DiagonalTensor{CosExpr<E>{m_expr}},
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

        static constexpr bool variable_data = false;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval(uint32_t r = 0, uint32_t c = 0, uint32_t d = 0, uint32_t t = 0) const {
            return sin(m_expr.eval(r, c, d, t));
        }

    private:
        std::conditional_t<(E::variable_data), const E&, const E> m_expr;
    };

    template<ExprType E>
    CUDA_COMPATIBLE
        [[nodiscard]] constexpr auto sin(const E& expr) {
        return SinExpr<E>{expr};
    }




    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////







    template<class E>
    class CosExpr : public AbstractExpr<CosExpr<E>, E::rows, E::cols> {
    public:

        CUDA_COMPATIBLE inline constexpr CosExpr(const E& expr) : m_expr(expr) {}

        template<VarIDType varId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert(varId >= 0, "Variable ID for differentiation must be non-negative.");
            auto expr_derivative = m_expr.derivate<varId>();
            if constexpr (is_zero_v<decltype(expr_derivative)>) {
                return expr_derivative;
            }
            else {
                return ElementwiseProductExpr<
                    DiagonalTensor<NegationExpr<SinExpr<E>>>,
                    decltype(expr_derivative)
                > {
                    DiagonalTensor{NegationExpr{SinExpr{m_expr}}},
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

        static constexpr bool variable_data = false;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval(uint32_t r = 0, uint32_t c = 0, uint32_t d = 0, uint32_t t = 0) const {
            return cos(m_expr.eval(r, c, d, t));
        }

    private:
        std::conditional_t<(E::variable_data), const E&, const E> m_expr;
    };

    template<ExprType E>
    CUDA_COMPATIBLE
        [[nodiscard]] constexpr auto cos(const E& expr) {
        return CosExpr<E>{expr};
    }




    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////









    template<class E1, class E2> requires(is_elementwise_broadcastable_v<E1, E2>)
        class ElementwisePowExpr : public AbstractExpr<ElementwisePowExpr<E1, E2>,
        std::conditional_t<(E1::rows > E2::rows), E1, E2>::rows,
        std::conditional_t<(E1::cols > E2::cols), E1, E2>::cols>
    {
    public:

        CUDA_COMPATIBLE inline constexpr ElementwisePowExpr(const E1& expr1, const E2& expr2) : m_expr1(expr1), m_expr2(expr2) {
        }

        template<VarIDType varId>
        [[nodiscard]]
        CUDA_COMPATIBLE constexpr inline auto derivate() const {
            static_assert((varId >= 0), "Variable ID for differentiation must be non-negative.");
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
                return ElementwiseProductExpr{
                    ElementwisePowExpr<E1, E2> {
                        m_expr1,
                        m_expr2
                    },
                    AdditionExpr {
                        ElementwiseProductExpr {
                            expr1_derivative,
                            DivisionExpr {
                                m_expr2,
                                m_expr1
                            }
                        },
                        ElementwiseProductExpr {
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

        static constexpr bool variable_data = false;

        [[nodiscard]]
        CUDA_COMPATIBLE inline constexpr auto eval(uint32_t r = 0, uint32_t c = 0, uint32_t d = 0, uint32_t t = 0) const {
            return pow(m_expr1.eval(r, c, d, t), m_expr2.eval(r, c, d, t));
        }

    private:
        std::conditional_t< (E1::variable_data), const E1&, const E1> m_expr1;
        std::conditional_t< (E2::variable_data), const E2&, const E2> m_expr2;
    };

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
        return ElementwisePowExpr<E, FilledConstant<S, E::rows, E::cols, E::depth, E::time>>{base, exponent};
    }

    template<ScalarType S, ExprType E>
    CUDA_COMPATIBLE
        [[nodiscard]] constexpr auto pow(S base, const E& exponent) {
        return ElementwisePowExpr<FilledConstant<S, E::rows, E::cols, E::depth, E::time>, E>{base, exponent};
    }
}