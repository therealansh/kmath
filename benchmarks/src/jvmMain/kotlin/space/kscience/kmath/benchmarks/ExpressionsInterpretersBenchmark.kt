/*
 * Copyright 2018-2021 KMath contributors.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the license/LICENSE.txt file.
 */

package space.kscience.kmath.benchmarks

import kotlinx.benchmark.Benchmark
import kotlinx.benchmark.Blackhole
import kotlinx.benchmark.Scope
import kotlinx.benchmark.State
import space.kscience.kmath.asm.compileToExpression
import space.kscience.kmath.expressions.*
import space.kscience.kmath.operations.DoubleField
import space.kscience.kmath.operations.bindSymbol
import space.kscience.kmath.operations.invoke
import kotlin.math.sin
import kotlin.random.Random

@State(Scope.Benchmark)
internal class ExpressionsInterpretersBenchmark {
    @Benchmark
    fun functionalExpression(blackhole: Blackhole) = invokeAndSum(functional, blackhole)

    @Benchmark
    fun mstExpression(blackhole: Blackhole) = invokeAndSum(mst, blackhole)

    @Benchmark
    fun asmExpression(blackhole: Blackhole) = invokeAndSum(asm, blackhole)

    @Benchmark
    fun rawExpression(blackhole: Blackhole) = invokeAndSum(raw, blackhole)

    @Benchmark
    fun justCalculate(blackhole: Blackhole) {
        val random = Random(0)
        var sum = 0.0

        repeat(times) {
            val x = random.nextDouble()
            sum += x * 2.0 + 2.0 / x - 16.0 / sin(x)
        }

        blackhole.consume(sum)
    }

    private fun invokeAndSum(expr: Expression<Double>, blackhole: Blackhole) {
        val random = Random(0)
        var sum = 0.0

        repeat(times) {
            sum += expr(x to random.nextDouble())
        }

        blackhole.consume(sum)
    }

    private companion object {
        private val x by symbol
        private val algebra = DoubleField
        private const val times = 1_000_000

        private val functional = DoubleField.expressionInExtendedField {
            bindSymbol(x) * number(2.0) + number(2.0) / bindSymbol(x) - number(16.0) / sin(bindSymbol(x))
        }

        private val node = MstExtendedField {
            x * 2.0 + number(2.0) / x - number(16.0) / sin(x)
        }

        private val mst = node.toExpression(DoubleField)
        private val asm = node.compileToExpression(DoubleField)

        private val raw = Expression<Double> { args ->
            val x = args[x]!!
            x * 2.0 + 2.0 / x - 16.0 / sin(x)
        }
    }
}
