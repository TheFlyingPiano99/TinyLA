#set page("a4")
#set math.equation(numbering: "1. ")

#title[TinyLA API Reference]

= VariableMatrix

The `VariableMatrix` class template represents a matrix with variable elements.
The elements can be of any scalar type, and the matrix dimensions are specified as template parameters.
The data is stored in a contiguous column-major array.

E.g.

#table(columns: 2, inset: 1em, align: horizon,
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

#table(columns: 3, inset: 1em, align: horizon,
[Assignment],
[```cpp
auto A = tinyla::VariableMatrix<float, 2, 2, 'A'>{};
const auto B = tinyla::VariableMatrix<float, 2, 2, 'B'>{};
A = B;
```],
[
$A := B$
],
[s], [s])



