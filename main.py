from pathlib import Path

import numpy as np


def make_dtype_arrays():
    """Create sample .npy files with a 2D array for testing different dtypes."""
    pth = Path("test-data/dtypes")
    pth.mkdir(parents=True, exist_ok=True)

    # Boolean
    arr_bool = np.array(
        [[True, False, True, False], [False, True, False, True]],
        dtype=np.bool_,
    )
    np.save(pth / "bool.npy", arr_bool)

    # Signed integers
    arr_i8 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int8)
    np.save(pth / "i8.npy", arr_i8)

    arr_i16 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int16)
    np.save(pth / "i16.npy", arr_i16)

    arr_i32 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32)
    np.save(pth / "i32.npy", arr_i32)

    arr_i64 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int64)
    np.save(pth / "i64.npy", arr_i64)

    # Unsigned integers
    arr_u8 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.uint8)
    np.save(pth / "u8.npy", arr_u8)

    arr_u16 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.uint16)
    np.save(pth / "u16.npy", arr_u16)

    arr_u32 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.uint32)
    np.save(pth / "u32.npy", arr_u32)

    arr_u64 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.uint64)
    np.save(pth / "u64.npy", arr_u64)

    # Floating point
    arr_f32 = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], dtype=np.float32)
    np.save(pth / "f32.npy", arr_f32)

    arr_f64 = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], dtype=np.float64)
    np.save(pth / "f64.npy", arr_f64)

    # Complex
    arr_c64 = np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]], dtype=np.complex64)
    np.save(pth / "c64.npy", arr_c64)

    arr_c128 = np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]], dtype=np.complex128)
    np.save(pth / "c128.npy", arr_c128)

    print(f"Created dtype test files in {pth}")


def make_empty_arrays():
    """Create .npy files with empty arrays for testing edge cases."""
    pth = Path("test-data/empty")
    pth.mkdir(parents=True, exist_ok=True)

    # Empty 0D array (scalar-like but empty)
    arr_0d = np.array(42.0, dtype=np.float64)
    np.save(pth / "empty_0d.npy", arr_0d)

    # Empty 1D array
    arr_1d = np.array([], dtype=np.float64)
    np.save(pth / "empty_1d.npy", arr_1d)

    # Empty 2D array with one dimension zero
    arr_2d_0_5 = np.zeros((0, 5), dtype=np.float64)
    np.save(pth / "empty_2d_0x5.npy", arr_2d_0_5)

    arr_2d_5_0 = np.zeros((5, 0), dtype=np.float64)
    np.save(pth / "empty_2d_5x0.npy", arr_2d_5_0)

    arr_2d_0_0 = np.zeros((0, 0), dtype=np.float64)
    np.save(pth / "empty_2d_0x0.npy", arr_2d_0_0)

    # Empty 3D array
    arr_3d = np.zeros((0, 3, 4), dtype=np.int32)
    np.save(pth / "empty_3d_0x3x4.npy", arr_3d)

    arr_3d_middle = np.zeros((2, 0, 4), dtype=np.int32)
    np.save(pth / "empty_3d_2x0x4.npy", arr_3d_middle)

    # Different dtypes with empty arrays
    arr_empty_bool = np.array([], dtype=np.bool_)
    np.save(pth / "empty_bool.npy", arr_empty_bool)

    arr_empty_i32 = np.array([], dtype=np.int32)
    np.save(pth / "empty_i32.npy", arr_empty_i32)

    arr_empty_c128 = np.array([], dtype=np.complex128)
    np.save(pth / "empty_c128.npy", arr_empty_c128)

    print(f"Created empty array test files in {pth}")


def make_endian_arrays():
    """Create .npy files with different endianness for testing."""
    pth = Path("test-data/endian")
    pth.mkdir(parents=True, exist_ok=True)

    data = [[1, 2, 3, 4], [5, 6, 7, 8]]

    # Little endian
    arr_i16_le = np.array(data, dtype="<i2")  # little endian int16
    np.save(pth / "i16_little.npy", arr_i16_le)

    arr_f32_le = np.array(data, dtype="<f4")  # little endian float32
    np.save(pth / "f32_little.npy", arr_f32_le)

    # Big endian
    arr_i16_be = np.array(data, dtype=">i2")  # big endian int16
    np.save(pth / "i16_big.npy", arr_i16_be)

    arr_f32_be = np.array(data, dtype=">f4")  # big endian float32
    np.save(pth / "f32_big.npy", arr_f32_be)

    # Native endian (should match little or big depending on system)
    arr_i16_native = np.array(data, dtype="=i2")
    np.save(pth / "i16_native.npy", arr_i16_native)

    print(f"Created endian test files in {pth}")


def make_fortran_order_arrays():
    """Create .npy files with Fortran (column-major) order."""
    pth = Path("test-data/fortran")
    pth.mkdir(parents=True, exist_ok=True)

    # 2D Fortran order
    arr_2d_f = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float64, order="F")
    np.save(pth / "f64_2d_fortran.npy", arr_2d_f)

    # 3D Fortran order
    arr_3d_f = np.arange(24, dtype=np.int32).reshape((2, 3, 4), order="F")
    np.save(pth / "i32_3d_fortran.npy", arr_3d_f)

    # Compare with C order (row-major)
    arr_2d_c = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float64, order="C")
    np.save(pth / "f64_2d_c.npy", arr_2d_c)

    arr_3d_c = np.arange(24, dtype=np.int32).reshape((2, 3, 4), order="C")
    np.save(pth / "i32_3d_c.npy", arr_3d_c)

    print(f"Created Fortran order test files in {pth}")


def make_shape_arrays():
    """Create .npy files with various shapes for testing."""
    pth = Path("test-data/shapes")
    pth.mkdir(parents=True, exist_ok=True)

    # 0D array (scalar)
    arr_0d = np.array(42, dtype=np.int32)
    np.save(pth / "scalar_i32.npy", arr_0d)

    arr_0d_f64 = np.array(3.14159, dtype=np.float64)
    np.save(pth / "scalar_f64.npy", arr_0d_f64)

    # 1D arrays
    arr_1d_small = np.array([1, 2, 3], dtype=np.int32)
    np.save(pth / "1d_small.npy", arr_1d_small)

    arr_1d_large = np.arange(1000, dtype=np.float64)
    np.save(pth / "1d_large.npy", arr_1d_large)

    # 2D arrays
    arr_2d_square = np.arange(16, dtype=np.int32).reshape((4, 4))
    np.save(pth / "2d_4x4.npy", arr_2d_square)

    arr_2d_rect = np.arange(20, dtype=np.float32).reshape((4, 5))
    np.save(pth / "2d_4x5.npy", arr_2d_rect)

    # 3D array
    arr_3d = np.arange(60, dtype=np.int16).reshape((3, 4, 5))
    np.save(pth / "3d_3x4x5.npy", arr_3d)

    # 4D array
    arr_4d = np.arange(120, dtype=np.uint8).reshape((2, 3, 4, 5))
    np.save(pth / "4d_2x3x4x5.npy", arr_4d)

    # 5D array
    arr_5d = np.arange(720, dtype=np.int8).reshape((2, 3, 4, 5, 6))
    np.save(pth / "5d_2x2x2x3x5.npy", arr_5d)

    # Single element arrays
    arr_1_elem = np.array([42], dtype=np.int32)
    np.save(pth / "1d_single.npy", arr_1_elem)

    arr_2d_1_elem = np.array([[42]], dtype=np.float64)
    np.save(pth / "2d_1x1.npy", arr_2d_1_elem)

    print(f"Created shape test files in {pth}")


def make_special_values_arrays():
    """Create .npy files with special floating point values."""
    pth = Path("test-data/special")
    pth.mkdir(parents=True, exist_ok=True)

    # NaN, Inf, -Inf
    arr_special = np.array(
        [[0.0, 1.0, -1.0, np.inf], [-np.inf, np.nan, 1e308, -1e308]], dtype=np.float64
    )
    np.save(pth / "special_f64.npy", arr_special)

    # Very small and very large values
    arr_extreme = np.array(
        [
            [np.finfo(np.float32).min, np.finfo(np.float32).max],
            [np.finfo(np.float32).eps, np.finfo(np.float32).tiny],
        ],
        dtype=np.float32,
    )
    np.save(pth / "extreme_f32.npy", arr_extreme)

    # Integer min/max values
    arr_i8_extremes = np.array(
        [np.iinfo(np.int8).min, -1, 0, 1, np.iinfo(np.int8).max], dtype=np.int8
    )
    np.save(pth / "extremes_i8.npy", arr_i8_extremes)

    arr_i64_extremes = np.array(
        [np.iinfo(np.int64).min, 0, np.iinfo(np.int64).max], dtype=np.int64
    )
    np.save(pth / "extremes_i64.npy", arr_i64_extremes)

    arr_u64_extremes = np.array([0, 1, np.iinfo(np.uint64).max], dtype=np.uint64)
    np.save(pth / "extremes_u64.npy", arr_u64_extremes)

    # Complex with special values
    arr_complex_special = np.array(
        [1 + 2j, np.inf + 1j, 1 + np.inf * 1j, np.nan + 0j, 0 + np.nan * 1j],
        dtype=np.complex128,
    )
    np.save(pth / "special_c128.npy", arr_complex_special)

    print(f"Created special values test files in {pth}")


def make_all():
    """Generate all test data files."""
    print("Generating all test data files...")
    make_dtype_arrays()
    make_empty_arrays()
    make_endian_arrays()
    make_fortran_order_arrays()
    make_shape_arrays()
    make_special_values_arrays()
    print("\nAll test data files generated successfully!")


def main():
    make_all()


if __name__ == "__main__":
    main()
