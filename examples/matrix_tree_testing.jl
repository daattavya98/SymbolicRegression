using SymbolicRegression
using LinearAlgebra
using MLJBase: machine, fit!, predict, report
using Random

# Define custom wrapper for the datatype I need
struct MatrixArray{T} <: AbstractMatrix{T}
    data::Vector{Matrix{T}}
end

# Define the size function for the custom datatype
Base.size(A::MatrixArray) = (length(A.data), size(A.data[1], 1))

# Define the getindex function for the custom datatype
function Base.getindex(A::MatrixArray, i::Int, j::Int)
    n = size(A.data[1], 1)
    matrix_index = div(i - 1, n) + 1
    row_index = rem(i - 1, n) + 1
    return A.data[matrix_index][row_index, j]
end

Base.IndexStyle(::Type{<:MatrixArray}) = Base.IndexStyle(Matrix)

Base.eltype(::Type{MatrixArray{T}}) where T = T


nrows(::FullInterface, ::Val{:other}, X::MatrixArray) = size(X, 1)
nrows(::LightInterface, ::Val{:other}, X::MatrixArray) = size(X, 1)
nrows(mat_array::MatrixArray) = size(mat_array, 1)

# Create my data and wrap it with the custom datatype
X = Matrix{Int}[rand(0:10, 2, 2) for _ in 1:10]
# println(typeof(X))
X = MatrixArray(X)
y = zeros(size(X))

@which getindex(X, 1, 1)
# model = SRRegressor(
#     niterations=100,
#     binary_operators=[+, *, /, -],
#     unary_operators=[LinearAlgebra.rank, LinearAlgebra.det],
#     populations=30,
#     expression_type=Expression,
# );

mach = MLJBase.machine(model, X, y)
MLJBase.fit!(mach)
MLJBase.report(mach).equations[end]