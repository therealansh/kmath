package scientifik.kmath.linear

import koma.matrix.ejml.EJMLMatrixFactory
import scientifik.kmath.operations.RealField
import scientifik.kmath.structures.Matrix
import kotlin.contracts.ExperimentalContracts
import kotlin.random.Random
import kotlin.system.measureTimeMillis

@ExperimentalContracts
fun main() {
    val random = Random(12224)
    val dim = 100
    //creating invertible matrix
    val u = Matrix.real(dim, dim) { i, j -> if (i <= j) random.nextDouble() else 0.0 }
    val l = Matrix.real(dim, dim) { i, j -> if (i >= j) random.nextDouble() else 0.0 }
    val matrix = l dot u

    val n = 5000 // iterations

    MatrixContext.real.run {

        repeat(50) {
            val res = inverse(matrix)
        }

        val inverseTime = measureTimeMillis {
            repeat(n) {
                val res = inverse(matrix)
            }
        }

        println("[kmath] Inversion of $n matrices $dim x $dim finished in $inverseTime millis")
    }

    //commons-math

    val commonsTime = measureTimeMillis {
        CMMatrixContext.run {
            val cm = matrix.toCM()             //avoid overhead on conversion
            repeat(n) {
                val res = inverse(cm)
            }
        }
    }


    println("[commons-math] Inversion of $n matrices $dim x $dim finished in $commonsTime millis")

    //koma-ejml

    val komaTime = measureTimeMillis {
        KomaMatrixContext(EJMLMatrixFactory(), RealField).run {
            val km = matrix.toKoma()      //avoid overhead on conversion
            repeat(n) {
                val res = inverse(km)
            }
        }
    }

    println("[koma-ejml] Inversion of $n matrices $dim x $dim finished in $komaTime millis")
}