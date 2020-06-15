package scientifik.memory

import org.khronos.webgl.ArrayBuffer
import org.khronos.webgl.DataView
import org.khronos.webgl.Int8Array

class DataViewMemory(val view: DataView) : Memory {

    override val size: Int get() = view.byteLength

    override fun view(offset: Int, length: Int): Memory {
        require(offset >= 0) { "offset shouldn't be negative: $offset" }
        require(length >= 0) { "length shouldn't be negative: $length" }

        if (offset + length > size)
            throw IndexOutOfBoundsException("offset + length > size: $offset + $length > $size")

        return DataViewMemory(DataView(view.buffer, view.byteOffset + offset, length))
    }


    override fun copy(): Memory = DataViewMemory(DataView(view.buffer.slice(0)))

    private val reader = object : MemoryReader {
        override val memory: Memory get() = this@DataViewMemory

        override fun readDouble(offset: Int): Double = view.getFloat64(offset, false)

        override fun readFloat(offset: Int): Float = view.getFloat32(offset, false)

        override fun readByte(offset: Int): Byte = view.getInt8(offset)

        override fun readShort(offset: Int): Short = view.getInt16(offset, false)

        override fun readInt(offset: Int): Int = view.getInt32(offset, false)

        override fun readLong(offset: Int): Long = (view.getInt32(offset, false).toLong() shl 32) or
                view.getInt32(offset + 4, false).toLong()

        override fun release() {
            // does nothing on JS because of GC
        }
    }

    override fun reader(): MemoryReader = reader

    private val writer = object : MemoryWriter {
        override val memory: Memory get() = this@DataViewMemory

        override fun writeDouble(offset: Int, value: Double) {
            view.setFloat64(offset, value, false)
        }

        override fun writeFloat(offset: Int, value: Float) {
            view.setFloat32(offset, value, false)
        }

        override fun writeByte(offset: Int, value: Byte) {
            view.setInt8(offset, value)
        }

        override fun writeShort(offset: Int, value: Short) {
            view.setUint16(offset, value, false)
        }

        override fun writeInt(offset: Int, value: Int) {
            view.setInt32(offset, value, false)
        }

        override fun writeLong(offset: Int, value: Long) {
            view.setInt32(offset, (value shr 32).toInt(), littleEndian = false)
            view.setInt32(offset + 4, (value and 0xffffffffL).toInt(), littleEndian = false)
        }

        override fun release() {
            //does nothing on JS
        }

    }

    override fun writer(): MemoryWriter = writer

}

/**
 * Allocate the most effective platform-specific memory
 */
actual fun Memory.Companion.allocate(length: Int): Memory {
    val buffer = ArrayBuffer(length)
    return DataViewMemory(DataView(buffer, 0, length))
}

actual fun Memory.Companion.wrap(array: ByteArray): Memory {
    @Suppress("CAST_NEVER_SUCCEEDS") val int8Array = array as Int8Array
    return DataViewMemory(DataView(int8Array.buffer, int8Array.byteOffset, int8Array.length))
}
