package space.kscience.kmath.tensors.core

import space.kscience.kmath.tensors.TensorAlgebra
import space.kscience.kmath.tensors.TensorStructure


internal inline fun <T, TensorType : TensorStructure<T>,
        TorchTensorAlgebraType : TensorAlgebra<T, TensorType>>
        TorchTensorAlgebraType.checkEmptyShape(shape: IntArray): Unit =
    check(shape.isNotEmpty()) {
        "Illegal empty shape provided"
    }

internal inline fun < TensorType : TensorStructure<Double>,
        TorchTensorAlgebraType : TensorAlgebra<Double, TensorType>>
        TorchTensorAlgebraType.checkEmptyDoubleBuffer(buffer: DoubleArray): Unit =
    check(buffer.isNotEmpty()) {
        "Illegal empty buffer provided"
    }

internal inline fun < TensorType : TensorStructure<Double>,
        TorchTensorAlgebraType : TensorAlgebra<Double, TensorType>>
        TorchTensorAlgebraType.checkBufferShapeConsistency(shape: IntArray, buffer: DoubleArray): Unit =
    check(buffer.size == shape.reduce(Int::times)) {
        "Inconsistent shape ${shape.toList()} for buffer of size ${buffer.size} provided"
    }


internal inline fun <T, TensorType : TensorStructure<T>,
        TorchTensorAlgebraType : TensorAlgebra<T, TensorType>>
        TorchTensorAlgebraType.checkShapesCompatible(a: TensorType, b: TensorType): Unit =
    check(a.shape contentEquals b.shape) {
        "Incompatible shapes ${a.shape.toList()} and ${b.shape.toList()} "
    }


internal inline fun <T, TensorType : TensorStructure<T>,
        TorchTensorAlgebraType : TensorAlgebra<T, TensorType>>
        TorchTensorAlgebraType.checkDot(a: TensorType, b: TensorType): Unit {
    val sa = a.shape
    val sb = b.shape
    val na = sa.size
    val nb = sb.size
    var status: Boolean
    if (nb == 1) {
        status = sa.last() == sb[0]
    } else {
        status = sa.last() == sb[nb - 2]
        if ((na > 2) and (nb > 2)) {
            status = status and
                    (sa.take(nb - 2).toIntArray() contentEquals sb.take(nb - 2).toIntArray())
        }
    }
    check(status) { "Incompatible shapes ${sa.toList()} and ${sb.toList()} provided for dot product" }
}

internal inline fun <T, TensorType : TensorStructure<T>,
        TorchTensorAlgebraType : TensorAlgebra<T, TensorType>>
        TorchTensorAlgebraType.checkTranspose(dim: Int, i: Int, j: Int): Unit =
    check((i < dim) and (j < dim)) {
        "Cannot transpose $i to $j for a tensor of dim $dim"
    }

internal inline fun <T, TensorType : TensorStructure<T>,
        TorchTensorAlgebraType : TensorAlgebra<T, TensorType>>
        TorchTensorAlgebraType.checkView(a: TensorType, shape: IntArray): Unit =
    check(a.shape.reduce(Int::times) == shape.reduce(Int::times))