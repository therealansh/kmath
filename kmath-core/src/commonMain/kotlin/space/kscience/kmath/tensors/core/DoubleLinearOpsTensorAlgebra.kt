package space.kscience.kmath.tensors.core

import space.kscience.kmath.tensors.LinearOpsTensorAlgebra
import space.kscience.kmath.nd.as1D
import space.kscience.kmath.nd.as2D

public class DoubleLinearOpsTensorAlgebra :
    LinearOpsTensorAlgebra<Double, DoubleTensor, IntTensor>,
    DoubleTensorAlgebra() {

    override fun DoubleTensor.inv(): DoubleTensor = invLU()

    override fun DoubleTensor.det(): DoubleTensor = detLU()

    override fun DoubleTensor.lu(): Pair<DoubleTensor, IntTensor> {

        checkSquareMatrix(shape)

        val luTensor = copy()

        val n = shape.size
        val m = shape.last()
        val pivotsShape = IntArray(n - 1) { i -> shape[i] }
        pivotsShape[n - 2] = m + 1

        val pivotsTensor = IntTensor(
            pivotsShape,
            IntArray(pivotsShape.reduce(Int::times)) { 0 }
        )

        for ((lu, pivots) in luTensor.matrixSequence().zip(pivotsTensor.vectorSequence()))
            luHelper(lu.as2D(), pivots.as1D(), m)

        return Pair(luTensor, pivotsTensor)

    }

    override fun luPivot(
        luTensor: DoubleTensor,
        pivotsTensor: IntTensor
    ): Triple<DoubleTensor, DoubleTensor, DoubleTensor> {
        //todo checks
        checkSquareMatrix(luTensor.shape)
        check(
            luTensor.shape.dropLast(2).toIntArray() contentEquals pivotsTensor.shape.dropLast(1).toIntArray() ||
                    luTensor.shape.last() == pivotsTensor.shape.last() - 1
        ) { "Bed shapes ((" } //todo rewrite

        val n = luTensor.shape.last()
        val pTensor = luTensor.zeroesLike()
        for ((p, pivot) in pTensor.matrixSequence().zip(pivotsTensor.vectorSequence()))
            pivInit(p.as2D(), pivot.as1D(), n)

        val lTensor = luTensor.zeroesLike()
        val uTensor = luTensor.zeroesLike()

        for ((pairLU, lu) in lTensor.matrixSequence().zip(uTensor.matrixSequence())
            .zip(luTensor.matrixSequence())) {
            val (l, u) = pairLU
            luPivotHelper(l.as2D(), u.as2D(), lu.as2D(), n)
        }

        return Triple(pTensor, lTensor, uTensor)

    }

    override fun DoubleTensor.cholesky(): DoubleTensor {
        // todo checks
        checkSquareMatrix(shape)

        val n = shape.last()
        val lTensor = zeroesLike()

        for ((a, l) in this.matrixSequence().zip(lTensor.matrixSequence()))
            for (i in 0 until n) choleskyHelper(a.as2D(), l.as2D(), n)

        return lTensor
    }

    private fun MutableStructure1D<Double>.dot(other: MutableStructure1D<Double>): Double {
        var res = 0.0
        for (i in 0 until size) {
            res += this[i] * other[i]
        }
        return res
    }

    private fun MutableStructure1D<Double>.l2Norm(): Double {
        var squareSum = 0.0
        for (i in 0 until size) {
            squareSum += this[i] * this[i]
        }
        return sqrt(squareSum)
    }

    fun qrHelper(
        matrix: MutableStructure2D<Double>,
        q: MutableStructure2D<Double>,
        r: MutableStructure2D<Double>
    ) {
        //todo check square
        val n = matrix.colNum
        for (j in 0 until n) {
            val v = matrix.columns[j]
            if (j > 0) {
                for (i in 0 until j) {
                    r[i, j] = q.columns[i].dot(matrix.columns[j])
                    for (k in 0 until n) {
                        v[k] = v[k] - r[i, j] * q.columns[i][k]
                    }
                }
            }
            r[j, j] = v.l2Norm()
            for (i in 0 until n) {
                q[i, j] = v[i] / r[j, j]
            }
        }
    }

    override fun DoubleTensor.qr(): Pair<DoubleTensor, DoubleTensor> {
        checkSquareMatrix(shape)
        val qTensor = zeroesLike()
        val rTensor = zeroesLike()
        val seq = matrixSequence().zip((qTensor.matrixSequence().zip(rTensor.matrixSequence())))
        for ((matrix, qr) in seq) {
            val (q, r) = qr
            qrHelper(matrix.as2D(), q.as2D(), r.as2D())
        }
        return Pair(qTensor, rTensor)
    }

    override fun DoubleTensor.svd(): Triple<DoubleTensor, DoubleTensor, DoubleTensor> {
        TODO("ALYA")
    }

    override fun DoubleTensor.symEig(eigenvectors: Boolean): Pair<DoubleTensor, DoubleTensor> {
        TODO("ANDREI")
    }

    public fun DoubleTensor.detLU(): DoubleTensor {
        val (luTensor, pivotsTensor) = lu()
        val n = shape.size

        val detTensorShape = IntArray(n - 1) { i -> shape[i] }
        detTensorShape[n - 2] = 1
        val resBuffer = DoubleArray(detTensorShape.reduce(Int::times)) { 0.0 }

        val detTensor = DoubleTensor(
            detTensorShape,
            resBuffer
        )

        luTensor.matrixSequence().zip(pivotsTensor.vectorSequence()).forEachIndexed { index, (lu, pivots) ->
            resBuffer[index] = luMatrixDet(lu.as2D(), pivots.as1D())
        }

        return detTensor
    }

    public fun DoubleTensor.invLU(): DoubleTensor {
        val (luTensor, pivotsTensor) = lu()
        val invTensor = luTensor.zeroesLike()

        val seq = luTensor.matrixSequence().zip(pivotsTensor.vectorSequence()).zip(invTensor.matrixSequence())
        for ((luP, invMatrix) in seq) {
            val (lu, pivots) = luP
            luMatrixInv(lu.as2D(), pivots.as1D(), invMatrix.as2D())
        }

        return invTensor
    }
}

public inline fun <R> DoubleLinearOpsTensorAlgebra(block: DoubleLinearOpsTensorAlgebra.() -> R): R =
    DoubleLinearOpsTensorAlgebra().block()