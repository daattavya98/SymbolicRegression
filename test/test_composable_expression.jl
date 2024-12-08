@testitem "Integration Test with fit! and Performance Check" tags = [:part3] begin
    include("../examples/template_expression.jl")
end
@testitem "Test ComposableExpression" tags = [:part2] begin
    using SymbolicRegression: ComposableExpression, Node
    using DynamicExpressions: OperatorEnum

    operators = OperatorEnum(; binary_operators=(+, *, /, -), unary_operators=(sin, cos))
    variable_names = (i -> "x$i").(1:3)
    ex = ComposableExpression(Node(Float64; feature=1); operators, variable_names)
    x = randn(32)
    y = randn(32)

    @test ex(x, y) == x
end

@testitem "Test interface for ComposableExpression" tags = [:part2] begin
    using SymbolicRegression: ComposableExpression
    using DynamicExpressions.InterfacesModule: Interfaces, ExpressionInterface
    using DynamicExpressions: OperatorEnum

    operators = OperatorEnum(; binary_operators=(+, *, /, -), unary_operators=(sin, cos))
    variable_names = (i -> "x$i").(1:3)
    x1 = ComposableExpression(Node(Float64; feature=1); operators, variable_names)
    x2 = ComposableExpression(Node(Float64; feature=2); operators, variable_names)

    f = x1 * sin(x2)
    g = f(f, f)

    @test string_tree(f) == "x1 * sin(x2)"
    @test string_tree(g) == "(x1 * sin(x2)) * sin(x1 * sin(x2))"

    @test Interfaces.test(ExpressionInterface, ComposableExpression, [f, g])
end

@testitem "Test interface for TemplateExpression" tags = [:part2] begin
    using SymbolicRegression
    using SymbolicRegression: TemplateExpression
    using DynamicExpressions.InterfacesModule: Interfaces, ExpressionInterface
    using DynamicExpressions: OperatorEnum

    operators = OperatorEnum(; binary_operators=(+, *, /, -), unary_operators=(sin, cos))
    variable_names = (i -> "x$i").(1:3)
    x1 = ComposableExpression(Node(Float64; feature=1); operators, variable_names)
    x2 = ComposableExpression(Node(Float64; feature=2); operators, variable_names)

    structure = TemplateStructure{(:f, :g)}(
        ((; f, g), (x1, x2)) -> f(f(f(x1))) - f(g(x2, x1))
    )
    @test structure.num_features == (; f=1, g=2)

    expr = TemplateExpression((; f=x1, g=x2 * x2); structure, operators, variable_names)

    @test String(string_tree(expr)) == "f = #1; g = #2 * #2"
    @test String(string_tree(expr; pretty=true)) == "╭ f = #1\n╰ g = #2 * #2"
    @test string_tree(get_tree(expr), operators) == "x1 - (x1 * x1)"
    @test Interfaces.test(ExpressionInterface, TemplateExpression, [expr])
end

@testitem "Printing and evaluation of TemplateExpression" tags = [:part2] begin
    using SymbolicRegression

    structure = TemplateStructure{(:f, :g)}(
        ((; f, g), (x1, x2, x3)) -> sin(f(x1, x2)) + g(x3)^2
    )
    operators = Options().operators
    variable_names = ["x1", "x2", "x3"]

    x1, x2, x3 = [
        ComposableExpression(Node{Float64}(; feature=i); operators, variable_names) for
        i in 1:3
    ]
    f = x1 * x2
    g = x1
    expr = TemplateExpression((; f, g); structure, operators, variable_names)

    # Default printing strategy:
    @test String(string_tree(expr)) == "f = #1 * #2; g = #1"

    x1_val = randn(5)
    x2_val = randn(5)

    # The feature indicates the index passed as argument:
    @test x1(x1_val) ≈ x1_val
    @test x2(x1_val, x2_val) ≈ x2_val
    @test x1(x2_val) ≈ x2_val

    # Composing expressions and then calling:
    @test String(string_tree((x1 * x2)(x3, x3))) == "x3 * x3"

    # Can evaluate with `sin` even though it's not in the allowed operators!
    X = randn(3, 5)
    x1_val = X[1, :]
    x2_val = X[2, :]
    x3_val = X[3, :]
    @test expr(X) ≈ @. sin(x1_val * x2_val) + x3_val^2

    # This is even though `g` is defined on `x1` only:
    @test g(x3_val) ≈ x3_val
end

@testitem "Test error handling" tags = [:part2] begin
    using SymbolicRegression
    using SymbolicRegression: ComposableExpression, Node, ValidVector
    using DynamicExpressions: OperatorEnum

    operators = OperatorEnum(; binary_operators=(+, *, /, -), unary_operators=(sin, cos))
    variable_names = (i -> "x$i").(1:3)
    ex = ComposableExpression(Node{Float64}(; feature=1); operators, variable_names)

    # Test error for unsupported input type with specific message
    @test_throws "ComposableExpression does not support input of type String" ex(
        "invalid input"
    )

    # Test ValidVector operations with numbers
    x = ValidVector([1.0, 2.0, 3.0], true)

    # Test binary operations between ValidVector and Number
    @test (x + 2.0).x ≈ [3.0, 4.0, 5.0]
    @test (2.0 + x).x ≈ [3.0, 4.0, 5.0]
    @test (x * 2.0).x ≈ [2.0, 4.0, 6.0]
    @test (2.0 * x).x ≈ [2.0, 4.0, 6.0]

    # Test unary operations on ValidVector
    @test sin(x).x ≈ sin.([1.0, 2.0, 3.0])
    @test cos(x).x ≈ cos.([1.0, 2.0, 3.0])
    @test abs(x).x ≈ [1.0, 2.0, 3.0]
    @test (-x).x ≈ [-1.0, -2.0, -3.0]

    # Test propagation of invalid flag
    invalid_x = ValidVector([1.0, 2.0, 3.0], false)
    @test !((invalid_x + 2.0).valid)
    @test !((2.0 + invalid_x).valid)
    @test !(sin(invalid_x).valid)

    # Test that regular numbers are considered valid
    @test (x + 2).valid
    @test sin(x).valid
end
@testitem "Test validity propagation with NaN" tags = [:part2] begin
    using SymbolicRegression: ComposableExpression, Node, ValidVector
    using DynamicExpressions: OperatorEnum

    operators = OperatorEnum(; binary_operators=(+, *, /, -), unary_operators=(sin, cos))
    variable_names = (i -> "x$i").(1:3)
    x1 = ComposableExpression(Node{Float64}(; feature=1); operators, variable_names)
    x2 = ComposableExpression(Node{Float64}(; feature=2); operators, variable_names)
    x3 = ComposableExpression(Node{Float64}(; feature=3); operators, variable_names)

    ex = 1.0 + x2 / x1

    @test ex([1.0], [2.0]) ≈ [3.0]

    @test ex([1.0, 1.0], [2.0, 2.0]) |> Base.Fix1(count, isnan) == 0
    @test ex([1.0, 0.0], [2.0, 2.0]) |> Base.Fix1(count, isnan) == 2

    x1_val = ValidVector([1.0, 2.0], false)
    x2_val = ValidVector([1.0, 2.0], false)
    @test ex(x1_val, x2_val).valid == false
end

@testitem "Test nothing return and type inference for TemplateExpression" tags = [:part2] begin
    using SymbolicRegression
    using Test: @inferred

    # Create a template expression that divides by x1
    structure = TemplateStructure{(:f,)}(((; f), (x1, x2)) -> 1.0 + f(x1) / x1)
    operators = Options(; binary_operators=(+, -, *, /)).operators
    variable_names = ["x1", "x2"]

    x1 = ComposableExpression(Node{Float64}(; feature=1); operators, variable_names)
    x2 = ComposableExpression(Node{Float64}(; feature=2); operators, variable_names)
    expr = TemplateExpression((; f=x1); structure, operators, variable_names)

    # Test division by zero returns nothing
    X = [0.0 1.0]'
    @test expr(X) === nothing

    # Test type inference
    X_good = [1.0 2.0]'
    @test @inferred(Union{Nothing,Vector{Float64}}, expr(X_good)) ≈ [2.0]

    # Test type inference with ValidVector input
    x1_val = ValidVector([1.0], true)
    x2_val = ValidVector([2.0], true)
    @test @inferred(ValidVector{Vector{Float64}}, x1(x1_val, x2_val)).x ≈ [1.0]

    x2_val_false = ValidVector([2.0], false)
    @test @inferred(x1(x1_val, x2_val_false)).valid == false
end
@testitem "Test compatibility with power laws" tags = [:part3] begin
    using SymbolicRegression
    using DynamicExpressions: OperatorEnum

    operators = OperatorEnum(; binary_operators=(+, -, *, /, ^))
    variable_names = ["x1", "x2"]
    x1 = ComposableExpression(Node{Float64}(; feature=1); operators, variable_names)
    x2 = ComposableExpression(Node{Float64}(; feature=2); operators, variable_names)

    structure = TemplateStructure{(:f,)}(((; f), (x1, x2)) -> f(x1)^f(x2))
    expr = TemplateExpression((; f=x1); structure, operators, variable_names)

    # There shouldn't be an error when we evaluate with invalid
    # expressions, even though the source of the NaN comes from the structure
    # function itself:
    X = -rand(2, 32)
    @test expr(X) === nothing
end

@testitem "Test constraints checking in TemplateExpression" tags = [:part2] begin
    using SymbolicRegression
    using SymbolicRegression: CheckConstraintsModule as CC

    # Create a template expression with nested exponentials
    options = Options(;
        binary_operators=(+, -, *, /),
        unary_operators=(exp,),
        nested_constraints=[exp => [exp => 1]], # Only allow one nested exp
    )
    operators = options.operators
    variable_names = ["x1", "x2"]

    # Create a valid inner expression
    x1 = ComposableExpression(Node{Float64}(; feature=1); operators, variable_names)
    valid_expr = exp(x1)  # One exp is ok

    # Create an invalid inner expression with too many nested exp
    invalid_expr = exp(exp(exp(x1)))
    # Three nested exp's violates constraint

    @test CC.check_constraints(valid_expr, options, 20)
    @test !CC.check_constraints(invalid_expr, options, 20)
end

@testitem "Test feature constraints in TemplateExpression" tags = [:part1] begin
    using SymbolicRegression
    using DynamicExpressions: Node

    operators = Options(; binary_operators=(+, -, *, /)).operators
    variable_names = ["x1", "x2", "x3"]

    # Create a structure where f only gets access to x1, x2
    # and g only gets access to x3
    structure = TemplateStructure{(:f, :g)}(((; f, g), (x1, x2, x3)) -> f(x1, x2) + g(x3))

    x1 = ComposableExpression(Node{Float64}(; feature=1); operators, variable_names)
    x2 = ComposableExpression(Node{Float64}(; feature=2); operators, variable_names)
    x3 = ComposableExpression(Node{Float64}(; feature=3); operators, variable_names)

    # Test valid case - each function only uses allowed features
    valid_f = x1 + x2
    valid_g = x1
    valid_template = TemplateExpression(
        (; f=valid_f, g=valid_g); structure, operators, variable_names
    )
    @test valid_template([1.0 2.0 3.0]') ≈ [6.0]  # (1 + 2) + 3

    # Test invalid case - f tries to use x3 which it shouldn't have access to
    invalid_f = x1 + x3
    invalid_template = TemplateExpression(
        (; f=invalid_f, g=valid_g); structure, operators, variable_names
    )
    @test invalid_template([1.0 2.0 3.0]') === nothing

    # Test invalid case - g tries to use x2 which it shouldn't have access to
    invalid_g = x2
    invalid_template2 = TemplateExpression(
        (; f=valid_f, g=invalid_g); structure, operators, variable_names
    )
    @test invalid_template2([1.0 2.0 3.0]') === nothing
end
@testitem "Test invalid structure" tags = [:part3] begin
    using SymbolicRegression

    operators = Options(; binary_operators=(+, -, *, /)).operators
    variable_names = ["x1", "x2", "x3"]

    x1 = ComposableExpression(Node{Float64}(; feature=1); operators, variable_names)
    x2 = ComposableExpression(Node{Float64}(; feature=2); operators, variable_names)
    x3 = ComposableExpression(Node{Float64}(; feature=3); operators, variable_names)

    @test_throws ArgumentError TemplateStructure{(:f,)}(
        ((; f), (x1, x2)) -> f(x1) + f(x1, x2)
    )
    @test_throws "Inconsistent number of arguments passed to f" TemplateStructure{(:f,)}(
        ((; f), (x1, x2)) -> f(x1) + f(x1, x2)
    )

    @test_throws ArgumentError TemplateStructure{(:f, :g)}(((; f, g), (x1, x2)) -> f(x1))
    @test_throws "Failed to infer number of features used by (:g,)" TemplateStructure{(
        :f, :g
    )}(
        ((; f, g), (x1, x2)) -> f(x1)
    )
end

@testitem "Test argument-less template structure" tags = [:part2] begin
    using SymbolicRegression
    using DynamicExpressions: OperatorEnum

    operators = OperatorEnum(; binary_operators=(+, *, /, -), unary_operators=(sin, cos))
    variable_names = ["x1", "x2"]
    x1 = ComposableExpression(Node{Float64}(; feature=1); operators, variable_names)
    x2 = ComposableExpression(Node{Float64}(; feature=2); operators, variable_names)
    c1 = ComposableExpression(Node{Float64}(; val=3.0); operators, variable_names)

    # We can evaluate an expression with no arguments:
    @test c1() == 3.0
    @test typeof(c1()) === Float64

    # Create a structure where f takes no arguments and g takes two
    structure = TemplateStructure{(:f, :g)}(((; f, g), (x1, x2)) -> f() + g(x1, x2))

    @test structure.num_features == (; f=0, g=2)

    X = [1.0 2.0]'
    expr = TemplateExpression((; f=c1, g=x1 + x2); structure, operators, variable_names)
    @test expr(X) ≈ [6.0]  # 3 + (1 + 2)
end

@testitem "Test symbolic derivatives" tags = [:part2] begin
    using SymbolicRegression: ComposableExpression, Node, D
    using DynamicExpressions: OperatorEnum, @declare_expression_operator, AbstractExpression
    using Zygote: gradient

    # Basic setup
    operators = OperatorEnum(; binary_operators=(+, *, /, -), unary_operators=(sin, cos))
    variable_names = ["x1", "x2"]
    x1 = ComposableExpression(Node(Float64; feature=1); operators, variable_names)
    x2 = ComposableExpression(Node(Float64; feature=2); operators, variable_names)

    # Test constant derivative
    c = ComposableExpression(Node(Float64; val=3.0); operators, variable_names)
    @test string(D(c, 1)) == "0.0"
    @test D(c, 1)([1.0]) ≈ [0.0]
    @test D(x1 + x2, 1)([0.0], [0.0]) ≈ [1.0]
    @test D(x1 + x2 * x2, 2)([0.0], [2.0]) ≈ [4.0]

    @test D(x1 * x2, 1)([1.0], [2.0]) ≈ [2.0]

    # Second order!
    @test D(D(x1 * x2, 1), 2)([1.0], [2.0]) ≈ [1.0]
    @test D(D(3.0 * x1 * x2 - x2, 1), 2)([1.0], [2.0]) ≈ [3.0]
    @test D(D(x1 * x2, 1), 1)([1.0], [2.0]) ≈ [0.0]

    # Unary operators:
    @test D(sin(x1), 1)([1.0]) ≈ [cos(1.0)]
    @test D(cos(x1), 1)([1.0]) ≈ [-sin(1.0)]
    @test D(sin(x1) * cos(x2), 1)([1.0], [2.0]) ≈ [cos(1.0) * cos(2.0)]
    @test D(D(sin(x1) * cos(x2), 1), 2)([1.0], [2.0]) ≈ [cos(1.0) * -sin(2.0)]

    # Printing should also be nice:
    @test repr(D(x1 * x2, 1)) == "last(x1, x2)"

    # We also have special behavior when there is no dependence:
    @test repr(D(sin(x2), 1)) == "0.0"
    @test repr(D(x2 + sin(x2), 1)) == "0.0"
    @test repr(D(x2 + sin(x2) - x1, 1)) == "-1.0"

    # But still nice printing for things like -sin:
    @test repr(D(D(sin(x1), 1), 1)) == "-sin(x1)"

    # Without generating weird additional operators:
    @test repr(D(D(D(sin(x1), 1), 1), 1)) == "-cos(x1)"

    # Custom functions have nice printing:
    my_op(x) = sin(x)
    @declare_expression_operator(my_op, 1)
    my_bin_op(x, y) = x + y
    @declare_expression_operator(my_bin_op, 2)
    operators = OperatorEnum(;
        binary_operators=(+, -, *, /, my_bin_op), unary_operators=(my_op,)
    )

    x = ComposableExpression(Node(Float64; feature=1); operators, variable_names)
    y = ComposableExpression(Node(Float64; feature=2); operators, variable_names)

    @test repr(D(my_op(x), 1)) == "∂my_op(x1)"
    @test repr(D(D(my_op(x), 1), 1)) == "∂∂my_op(x1)"

    @test repr(D(my_bin_op(x, y), 1)) == "∂₁my_bin_op(x1, x2)"
    @test repr(D(my_bin_op(x, y), 2)) == "∂₂my_bin_op(x1, x2)"
    @test repr(D(my_bin_op(x, x - y), 2)) == "∂₂my_bin_op(x1, x1 - x2) * -1.0"
end

@testitem "Test template structure with derivatives" tags = [:part2] begin
    using SymbolicRegression:
        ComposableExpression, Node, D, TemplateStructure, TemplateExpression
    using DynamicExpressions: OperatorEnum

    # Basic setup
    operators = OperatorEnum(; binary_operators=(+, *, /, -), unary_operators=(sin, cos))
    variable_names = ["x1", "x2"]
    x1 = ComposableExpression(Node(Float64; feature=1); operators, variable_names)
    x2 = ComposableExpression(Node(Float64; feature=2); operators, variable_names)

    # Create a structure that computes f(x1, x2) and its derivative with respect to x1
    structure = TemplateStructure{(:f,)}(((; f), (x1, x2)) -> f(x1, x2) + D(f, 1)(x1, x2))
    # We pass the functions through:
    @test structure.num_features == (; f=2)

    # Test with a simple function and its derivative
    expr = TemplateExpression((; f=x1 * sin(x2)); structure, operators, variable_names)

    # Truth: x1 * sin(x2) + sin(x2)
    X = randn(2, 32)
    @test expr(X) ≈ X[1, :] .* sin.(X[2, :]) .+ sin.(X[2, :])
end
