package space.kscience.kmath.ojalgo

import org.ojalgo.structure.Access2D
import space.kscience.kmath.linear.Matrix

public class OjalgoMatrix<T : Comparable<T>>(public val origin: Access2D<T>) : Matrix<T> {
    public override val rowNum: Int
        get() = origin.countRows().toInt()

    public override val colNum: Int
        get() = origin.countColumns().toInt()

    public override fun get(i: Int, j: Int): T = origin.get(i.toLong(), j.toLong())
}
