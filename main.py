from pathlib import Path
import numpy as np


def make_npy_file():
    """Create sample .npy files with a 2D array for testing different dtypes."""
    pth = Path("test-data")
    pth.mkdir(exist_ok=True)

    # Boolean
    arr_bool = np.array(
        [[True, False, True, False], [False, True, False, True]], dtype=np.dtypes.BoolDType
    )
    np.save("test-data/bool.npy", arr_bool)

    # Bit-sized signed integers
    arr_i8 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.dtypes.Int8DType)
    np.save("test-data/i8.npy", arr_i8)

    arr_i16 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.dtypes.Int16DType)
    np.save("test-data/i16.npy", arr_i16)

    arr_i32 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.dtypes.Int32DType)
    np.save("test-data/i32.npy", arr_i32)

    arr_i64 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.dtypes.Int64DType)
    np.save("test-data/i64.npy", arr_i64)

    # Bit-sized unsigned integers
    arr_u8 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.dtypes.UInt8DType)
    np.save("test-data/u8.npy", arr_u8)

    arr_u16 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.dtypes.UInt16DType)
    np.save("test-data/u16.npy", arr_u16)

    arr_u32 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.dtypes.UInt32DType)
    np.save("test-data/u32.npy", arr_u32)

    arr_u64 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.dtypes.UInt64DType)
    np.save("test-data/u64.npy", arr_u64)

    # C-named signed integers
    arr_byte = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.dtypes.ByteDType)
    np.save("test-data/byte.npy", arr_byte)

    arr_short = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.dtypes.ShortDType)
    np.save("test-data/short.npy", arr_short)

    arr_intc = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.dtypes.IntDType)
    np.save("test-data/intc.npy", arr_intc)

    arr_long = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.dtypes.LongDType)
    np.save("test-data/long.npy", arr_long)

    arr_longlong = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.dtypes.LongLongDType)
    np.save("test-data/longlong.npy", arr_longlong)

    # C-named unsigned integers
    arr_ubyte = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.dtypes.UByteDType)
    np.save("test-data/ubyte.npy", arr_ubyte)

    arr_ushort = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.dtypes.UShortDType)
    np.save("test-data/ushort.npy", arr_ushort)

    arr_uintc = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.dtypes.UIntDType)
    np.save("test-data/uintc.npy", arr_uintc)

    arr_ulong = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.dtypes.ULongDType)
    np.save("test-data/ulong.npy", arr_ulong)

    arr_ulonglong = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.dtypes.ULongLongDType)
    np.save("test-data/ulonglong.npy", arr_ulonglong)

    # Floating point
    arr_f16 = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], dtype=np.dtypes.Float16DType)
    np.save("test-data/f16.npy", arr_f16)

    arr_f32 = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], dtype=np.dtypes.Float32DType)
    np.save("test-data/f32.npy", arr_f32)

    arr_f64 = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], dtype=np.dtypes.Float64DType)
    np.save("test-data/f64.npy", arr_f64)

    arr_longdouble = np.array(
        [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], dtype=np.dtypes.LongDoubleDType
    )
    np.save("test-data/longdouble.npy", arr_longdouble)

    # Complex
    arr_c64 = np.array(
        [[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]], dtype=np.dtypes.Complex64DType    
    )
    np.save("test-data/c64.npy", arr_c64)

    arr_c128 = np.array(
        [[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]], dtype=np.dtypes.Complex128DType
    )
    np.save("test-data/c128.npy", arr_c128)

    arr_clongdouble = np.array(
        [[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]], dtype=np.dtypes.CLongDoubleDType
    )
    np.save("test-data/clongdouble.npy", arr_clongdouble)

def main():
    make_npy_file()


if __name__ == "__main__":
    main()
