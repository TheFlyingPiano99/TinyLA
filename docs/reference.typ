#set page("a4", margin: 2cm)
#set math.equation(numbering: "1. ")

#title[TinyLA Reference]

= Types

== VariableMatrix

The `VariableMatrix` class template represents a matrix with variable elements.
The elements can be of any scalar type, and the matrix dimensions are specified as template parameters.
The data is stored in a contiguous column-major array.

E.g.

#table(columns: 2, inset: 1em, align: horizon,
table.header([*C++ Syntax*], [*Mathematical Notation*]),
[```cpp
auto A = tinyla::VariableMatrix<float, 2, 2, '0'>{};
```],
[
$0_(2 times 2) = mat(0, 0; 0, 0)$
],

[```cpp
auto A = tinyla::VariableMatrix<double, 2, 3, 'A'>{
  {1.0, 2.0, 3.0},
  {4.0, 5.0, 6.0}
};
```],
[
$A_(2 times 3) = mat(1, 2, 3; 4, 5, 6)$
],
[```cpp
auto I = tinyla::VariableMatrix<int, 3, 3, 'I'>::identity();
```],
[
$I_(3 times 3)  = mat(1,0,0;0,1,0;0,0,1)$
]
)

= Operators

Given $A$, $B$, $v$ and $w$ are `VariableMatrix` objects of compatible dimensions.
```cpp
auto r = 2;   // Number of rows
auto c = 2;   // Number of columns
auto A = tinyla::VariableMatrix<double, r, c, 'A'>{};
auto B = tinyla::VariableMatrix<double, r, c, 'B'>{};
auto C = tinyla::VariableMatrix<std::complex<float>, r, c, 'C'>{};
auto v = tinyla::VariableMatrix<double, r, 1, 'v'>{};
auto w = tinyla::VariableMatrix<double, r, 1, 'w'>{};
```

#table(columns: 3, inset: 1em, align: horizon,
table.header([*Operation*], [*C++ Syntax*], [*Mathematical Notation*]),
[Assignment],
[```cpp
A = B;
```],
[
$A := B$
],
[Indexing],
[```cpp
auto a1 = A.at(i, j);
auto a2 = A[i][j];
```],
[
$A_(i,j)$
],
[Transposition],
[```cpp
auto A_trans1 = transpose(A);
auto A_trans2 = T(A);
```],
[
$A^T$
],
[Conjugation],
[```cpp
auto C_conj1 = conjugate(C);
auto C_conj2 = conj(C);
```],
[
$overline(C)$
],
[Adjoint (conjugation and transposition)],
[```cpp
auto C_adj1 = adjoint(C);
auto C_adj2 = adj(C);
```],
[
$C^(dagger)$
],
[Addition],
[```cpp
auto S = A + B;
```],
[$A + B$],
[Subtraction],
[```cpp
auto D = A - B;
```],
[$A - B$],
[Matrix-vector multiplication],
[```cpp
auto p = A * v;
```],
[$A v$],
[Matrix-matrix multiplication],
[```cpp
auto M = A * B;
```],
[$A B$],
[Dot product],
[```cpp
auto d = dot(v, w);
```],
[$v dot w$],

)



