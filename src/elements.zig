//! Module for array element functions and types.
const std = @import("std");
const builtin = @import("builtin");

const boolean = @import("elements/boolean.zig");
const types = @import("elements/types.zig");
const array = @import("./array.zig");

const native_endian = builtin.cpu.arch.endian();

pub const ElementType = types.ElementType;

pub const ViewDataError = error{
    /// The provided element type does not match the expected type.
    TypeMismatch,
    /// The provided data contains invalid boolean values (not 0 or 1).
    InvalidBool,
    /// The endianness of the provided data does not match the native endianness.
    EndiannessMismatch,
    /// The requested length causes an overflow when calculating the total byte size.
    LengthOverflow,
    /// Not enough bytes provided to fulfill the requested length.
    MissingBytes,
    /// More bytes provided than necessary for the requested length.
    ExtraBytes,
    /// The provided byte slice is not properly aligned for the target type.
    Misaligned,
};

const WriteDataError = error{};

const ReadDataError = error{};

/// A generic element type representing a single element of type `T` in a numpy array.
pub fn Element(comptime T: type) type {
    const element_type = ElementType.fromZigType(T) catch @compileError("Unsupported type");

    return struct {
        const Self = @This();

        /// Interprets a byte slice as a slice of the specified element type `T`.
        /// The function checks for type compatibility, endianness, length, and alignment.
        pub fn bytesAsSlice(
            bytes: []const u8,
            len: usize,
            type_descr: ElementType,
        ) ViewDataError![]const T {
            // Element types must match
            if (std.meta.activeTag(type_descr) != std.meta.activeTag(element_type)) {
                return ViewDataError.TypeMismatch;
            }

            switch (type_descr) {
                .Bool => {
                    // All bytes have to be either 0 or 1
                    if (!boolean.isAllZeroOrOne(bytes)) {
                        return ViewDataError.InvalidBool;
                    }
                },
                .Int8, .UInt8 => {},
                .Int16, .Int32, .Int64, .UInt16, .UInt32, .UInt64, .Float32, .Float64, .Float128, .Complex64, .Complex128 => |endian| {
                    // Endianness must match native endianness for multi-byte types
                    if (endian) |e| {
                        if (e != native_endian) {
                            return ViewDataError.EndiannessMismatch;
                        }
                    }
                    // endian is null -> assume native endian
                },
            }

            // Get the number of bytes. Check for overflow
            const num_bytes: usize, const overflow: u1 = @mulWithOverflow(len, @sizeOf(T));
            if (overflow != 0) {
                return ViewDataError.LengthOverflow;
            }

            // Make sure we have the right number of bytes
            switch (std.math.order(bytes.len, num_bytes)) {
                .lt => return ViewDataError.MissingBytes,
                .gt => return ViewDataError.ExtraBytes,
                .eq => {},
            }

            // At this point, bytes.len is zero when:
            // 1. len is zero, or
            // 2. @sizeOf(T) is zero (though this should not happen for valid numpy types)
            // Either case, we can return an empty slice. But that empty slice needs to be properly aligned,
            // otherwise Zig just attaches an undefined pointer to the slice which may be misaligned.
            if (bytes.len == 0) {
                return array.danglingPtr(T)[0..0];
            }

            // Now bytes is non-empty, len is non-zero, and @sizeOf(T) is non-zero
            // Check alignment
            if (!std.mem.isAligned(@intFromPtr(bytes.ptr), @alignOf(T))) {
                return ViewDataError.Misaligned;
            }

            // Both length and alignment checks passed, we can:

            // 1. Now upgrade the alignment in the type system.
            const aligned_bytes = @as([]align(@alignOf(T)) const u8, @alignCast(bytes));

            // 2. Now bytesAsSlice will be happy because the input is []align(T) const u8
            return std.mem.bytesAsSlice(T, aligned_bytes);
        }

        pub fn writeSlice(slice: []const T, writer: std.io.Writer) WriteDataError!void {
            _ = slice;
            _ = writer;
        }

        pub fn readSlice(slice: []T, reader: *std.io.Reader, type_descr: ElementType) ReadDataError!void {
            _ = slice;
            _ = reader;
            _ = type_descr;
        }
    };
}

test {
    _ = types;
    _ = boolean;
}

// Helper to get the opposite endianness
fn oppositeEndian(endian: std.builtin.Endian) std.builtin.Endian {
    return switch (endian) {
        .little => .big,
        .big => .little,
    };
}

test "bytesAsSlice - Bool with null endian, valid data" {
    var bytes = [_]u8{ 0, 1, 1, 0, 1 };
    const result = try Element(bool).bytesAsSlice(&bytes, 5, .Bool);
    try std.testing.expectEqual(5, result.len);
    try std.testing.expectEqual(false, result[0]);
    try std.testing.expectEqual(true, result[1]);
    try std.testing.expectEqual(true, result[2]);
    try std.testing.expectEqual(false, result[3]);
    try std.testing.expectEqual(true, result[4]);
}

test "bytesAsSlice - Int8 with null endian" {
    var bytes = [_]u8{ 0, 127, 255, 128, 1 };
    const result = try Element(i8).bytesAsSlice(&bytes, 5, .Int8);
    try std.testing.expectEqual(5, result.len);
    try std.testing.expectEqual(@as(i8, 0), result[0]);
    try std.testing.expectEqual(@as(i8, 127), result[1]);
    try std.testing.expectEqual(@as(i8, -1), result[2]);
    try std.testing.expectEqual(@as(i8, -128), result[3]);
    try std.testing.expectEqual(@as(i8, 1), result[4]);
}

test "bytesAsSlice - UInt8 with null endian" {
    var bytes = [_]u8{ 0, 127, 255, 128, 1 };
    const result = try Element(u8).bytesAsSlice(&bytes, 5, .UInt8);
    try std.testing.expectEqual(5, result.len);
    try std.testing.expectEqual(@as(u8, 0), result[0]);
    try std.testing.expectEqual(@as(u8, 127), result[1]);
    try std.testing.expectEqual(@as(u8, 255), result[2]);
    try std.testing.expectEqual(@as(u8, 128), result[3]);
    try std.testing.expectEqual(@as(u8, 1), result[4]);
}

test "bytesAsSlice - Int16 with null endian" {
    var bytes align(@alignOf(i16)) = [_]u8{ 0, 0, 1, 0, 255, 255 };
    const result = try Element(i16).bytesAsSlice(&bytes, 3, .{ .Int16 = null });
    try std.testing.expectEqual(3, result.len);
    try std.testing.expectEqual(@as(i16, 0), result[0]);
    try std.testing.expectEqual(@as(i16, if (native_endian == .little) 1 else 256), result[1]);
    try std.testing.expectEqual(@as(i16, -1), result[2]);
}

test "bytesAsSlice - Int32 with null endian" {
    var bytes align(@alignOf(i32)) = [_]u8{ 0, 0, 0, 0, 1, 0, 0, 0 };
    const result = try Element(i32).bytesAsSlice(&bytes, 2, .{ .Int32 = null });
    try std.testing.expectEqual(2, result.len);
    try std.testing.expectEqual(@as(i32, 0), result[0]);
    try std.testing.expectEqual(@as(i32, if (native_endian == .little) 1 else 0x40), result[1]);
}

test "bytesAsSlice - Int64 with null endian" {
    var bytes align(@alignOf(i64)) = [_]u8{ 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255 };
    const result = try Element(i64).bytesAsSlice(&bytes, 2, .{ .Int64 = null });
    try std.testing.expectEqual(2, result.len);
    try std.testing.expectEqual(@as(i64, 0), result[0]);
    try std.testing.expectEqual(@as(i64, -1), result[1]);
}

test "bytesAsSlice - UInt16 with null endian" {
    var bytes align(@alignOf(u16)) = [_]u8{ 0, 0, 255, 255 };
    const result = try Element(u16).bytesAsSlice(&bytes, 2, .{ .UInt16 = null });
    try std.testing.expectEqual(2, result.len);
    try std.testing.expectEqual(@as(u16, 0), result[0]);
    try std.testing.expectEqual(@as(u16, 0xFF_FF), result[1]);
}

test "bytesAsSlice - UInt32 with null endian" {
    var bytes align(@alignOf(u32)) = [_]u8{ 0, 0, 0, 0, 255, 255, 255, 255 };
    const result = try Element(u32).bytesAsSlice(&bytes, 2, .{ .UInt32 = null });
    try std.testing.expectEqual(2, result.len);
    try std.testing.expectEqual(@as(u32, 0), result[0]);
    try std.testing.expectEqual(@as(u32, 0xFFFF_FFFF), result[1]);
}

test "bytesAsSlice - UInt64 with null endian" {
    var bytes align(@alignOf(u64)) = [_]u8{ 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255 };
    const result = try Element(u64).bytesAsSlice(&bytes, 2, .{ .UInt64 = null });
    try std.testing.expectEqual(2, result.len);
    try std.testing.expectEqual(@as(u64, 0), result[0]);
    try std.testing.expectEqual(@as(u64, 0xFFFF_FFFF_FFFF_FFFF), result[1]);
}

test "bytesAsSlice - Float32 with null endian" {
    var bytes align(@alignOf(f32)) = [_]u8{0} ** 8;
    // 0.0 and 1.0 in IEEE 754 binary32 format (native endian)
    std.mem.writeInt(u32, bytes[0..4], @as(u32, @bitCast(@as(f32, 0.0))), native_endian);
    std.mem.writeInt(u32, bytes[4..8], @as(u32, @bitCast(@as(f32, 1.0))), native_endian);
    const result = try Element(f32).bytesAsSlice(&bytes, 2, .{ .Float32 = null });
    try std.testing.expectEqual(2, result.len);
    try std.testing.expectEqual(@as(f32, 0.0), result[0]);
    try std.testing.expectEqual(@as(f32, 1.0), result[1]);
}

test "bytesAsSlice - Float64 with null endian" {
    var bytes align(@alignOf(f64)) = [_]u8{0} ** 16;
    std.mem.writeInt(u64, bytes[0..8], @as(u64, @bitCast(@as(f64, 0.0))), native_endian);
    std.mem.writeInt(u64, bytes[8..16], @as(u64, @bitCast(@as(f64, 3.14159))), native_endian);
    const result = try Element(f64).bytesAsSlice(&bytes, 2, .{ .Float64 = null });
    try std.testing.expectEqual(2, result.len);
    try std.testing.expectEqual(@as(f64, 0.0), result[0]);
    try std.testing.expectApproxEqRel(@as(f64, 3.14159), result[1], 0.00001);
}

test "bytesAsSlice - Float128 with null endian" {
    var bytes align(@alignOf(f128)) = [_]u8{0} ** 32;
    std.mem.writeInt(u128, bytes[0..16], @as(u128, @bitCast(@as(f128, 0.0))), native_endian);
    std.mem.writeInt(u128, bytes[16..32], @as(u128, @bitCast(@as(f128, 1.0))), native_endian);
    const result = try Element(f128).bytesAsSlice(&bytes, 2, .{ .Float128 = null });
    try std.testing.expectEqual(2, result.len);
    try std.testing.expectEqual(@as(f128, 0.0), result[0]);
    try std.testing.expectEqual(@as(f128, 1.0), result[1]);
}

test "bytesAsSlice - Complex64 with null endian" {
    var bytes align(@alignOf(std.math.Complex(f32))) = [_]u8{0} ** 16;
    // Complex(1.0, 2.0) and Complex(3.0, 4.0)
    std.mem.writeInt(u32, bytes[0..4], @as(u32, @bitCast(@as(f32, 1.0))), native_endian); // real part
    std.mem.writeInt(u32, bytes[4..8], @as(u32, @bitCast(@as(f32, 2.0))), native_endian); // imag part
    std.mem.writeInt(u32, bytes[8..12], @as(u32, @bitCast(@as(f32, 3.0))), native_endian); // real part
    std.mem.writeInt(u32, bytes[12..16], @as(u32, @bitCast(@as(f32, 4.0))), native_endian); // imag part
    const result = try Element(std.math.Complex(f32)).bytesAsSlice(&bytes, 2, .{ .Complex64 = null });
    try std.testing.expectEqual(2, result.len);
    try std.testing.expectEqual(@as(f32, 1.0), result[0].re);
    try std.testing.expectEqual(@as(f32, 2.0), result[0].im);
    try std.testing.expectEqual(@as(f32, 3.0), result[1].re);
    try std.testing.expectEqual(@as(f32, 4.0), result[1].im);
}

test "bytesAsSlice - Complex128 with null endian" {
    var bytes align(@alignOf(std.math.Complex(f64))) = [_]u8{0} ** 32;
    std.mem.writeInt(u64, bytes[0..8], @as(u64, @bitCast(@as(f64, 1.5))), native_endian);
    std.mem.writeInt(u64, bytes[8..16], @as(u64, @bitCast(@as(f64, 2.5))), native_endian);
    std.mem.writeInt(u64, bytes[16..24], @as(u64, @bitCast(@as(f64, 3.5))), native_endian);
    std.mem.writeInt(u64, bytes[24..32], @as(u64, @bitCast(@as(f64, 4.5))), native_endian);
    const result = try Element(std.math.Complex(f64)).bytesAsSlice(&bytes, 2, .{ .Complex128 = null });
    try std.testing.expectEqual(2, result.len);
    try std.testing.expectEqual(@as(f64, 1.5), result[0].re);
    try std.testing.expectEqual(@as(f64, 2.5), result[0].im);
    try std.testing.expectEqual(@as(f64, 3.5), result[1].re);
    try std.testing.expectEqual(@as(f64, 4.5), result[1].im);
}

test "bytesAsSlice - Int16 with explicit native endian" {
    var bytes align(@alignOf(i16)) = [_]u8{ 1, 0, 2, 0 };
    const result = try Element(i16).bytesAsSlice(&bytes, 2, .{ .Int16 = native_endian });
    try std.testing.expectEqual(2, result.len);
}

test "bytesAsSlice - Int32 with explicit native endian" {
    var bytes align(@alignOf(i32)) = [_]u8{ 1, 0, 0, 0, 2, 0, 0, 0 };
    const result = try Element(i32).bytesAsSlice(&bytes, 2, .{ .Int32 = native_endian });
    try std.testing.expectEqual(2, result.len);
}

test "bytesAsSlice - Int64 with explicit native endian" {
    var bytes align(@alignOf(i64)) = [_]u8{0} ** 16;
    const result = try Element(i64).bytesAsSlice(&bytes, 2, .{ .Int64 = native_endian });
    try std.testing.expectEqual(2, result.len);
}

test "bytesAsSlice - UInt16 with explicit native endian" {
    var bytes align(@alignOf(u16)) = [_]u8{ 1, 0, 2, 0 };
    const result = try Element(u16).bytesAsSlice(&bytes, 2, .{ .UInt16 = native_endian });
    try std.testing.expectEqual(2, result.len);
}

test "bytesAsSlice - UInt32 with explicit native endian" {
    var bytes align(@alignOf(u32)) = [_]u8{ 1, 0, 0, 0 };
    const result = try Element(u32).bytesAsSlice(&bytes, 1, .{ .UInt32 = native_endian });
    try std.testing.expectEqual(1, result.len);
}

test "bytesAsSlice - UInt64 with explicit native endian" {
    var bytes align(@alignOf(u64)) = [_]u8{0} ** 8;
    const result = try Element(u64).bytesAsSlice(&bytes, 1, .{ .UInt64 = native_endian });
    try std.testing.expectEqual(1, result.len);
}

test "bytesAsSlice - Float32 with explicit native endian" {
    var bytes align(@alignOf(f32)) = [_]u8{0} ** 4;
    std.mem.writeInt(u32, &bytes, @as(u32, @bitCast(@as(f32, 1.0))), native_endian);
    const result = try Element(f32).bytesAsSlice(&bytes, 1, .{ .Float32 = native_endian });
    try std.testing.expectEqual(1, result.len);
    try std.testing.expectEqual(@as(f32, 1.0), result[0]);
}

test "bytesAsSlice - Float64 with explicit native endian" {
    var bytes align(@alignOf(f64)) = [_]u8{0} ** 8;
    std.mem.writeInt(u64, &bytes, @as(u64, @bitCast(@as(f64, 2.5))), native_endian);
    const result = try Element(f64).bytesAsSlice(&bytes, 1, .{ .Float64 = native_endian });
    try std.testing.expectEqual(1, result.len);
    try std.testing.expectEqual(@as(f64, 2.5), result[0]);
}

test "bytesAsSlice - Float128 with explicit native endian" {
    var bytes align(@alignOf(f128)) = [_]u8{0} ** 16;
    std.mem.writeInt(u128, &bytes, @as(u128, @bitCast(@as(f128, 1.0))), native_endian);
    const result = try Element(f128).bytesAsSlice(&bytes, 1, .{ .Float128 = native_endian });
    try std.testing.expectEqual(1, result.len);
    try std.testing.expectEqual(@as(f128, 1.0), result[0]);
}

test "bytesAsSlice - Complex64 with explicit native endian" {
    var bytes align(@alignOf(std.math.Complex(f32))) = [_]u8{0} ** 8;
    std.mem.writeInt(u32, bytes[0..4], @as(u32, @bitCast(@as(f32, 1.0))), native_endian);
    std.mem.writeInt(u32, bytes[4..8], @as(u32, @bitCast(@as(f32, 2.0))), native_endian);
    const result = try Element(std.math.Complex(f32)).bytesAsSlice(&bytes, 1, .{ .Complex64 = native_endian });
    try std.testing.expectEqual(1, result.len);
    try std.testing.expectEqual(@as(f32, 1.0), result[0].re);
    try std.testing.expectEqual(@as(f32, 2.0), result[0].im);
}

test "bytesAsSlice - Complex128 with explicit native endian" {
    var bytes align(@alignOf(std.math.Complex(f64))) = [_]u8{0} ** 16;
    std.mem.writeInt(u64, bytes[0..8], @as(u64, @bitCast(@as(f64, 3.0))), native_endian);
    std.mem.writeInt(u64, bytes[8..16], @as(u64, @bitCast(@as(f64, 4.0))), native_endian);
    const result = try Element(std.math.Complex(f64)).bytesAsSlice(&bytes, 1, .{ .Complex128 = native_endian });
    try std.testing.expectEqual(1, result.len);
    try std.testing.expectEqual(@as(f64, 3.0), result[0].re);
    try std.testing.expectEqual(@as(f64, 4.0), result[0].im);
}

test "bytesAsSlice - InvalidBool error" {
    var bytes = [_]u8{ 0, 1, 2, 1, 0 }; // byte with value 2 is invalid
    const result = Element(bool).bytesAsSlice(&bytes, 5, .Bool);
    try std.testing.expectError(ViewDataError.InvalidBool, result);
}

test "bytesAsSlice - InvalidBool with 255" {
    var bytes = [_]u8{ 0, 1, 255, 1, 0 };
    const result = Element(bool).bytesAsSlice(&bytes, 5, .Bool);
    try std.testing.expectError(ViewDataError.InvalidBool, result);
}

test "bytesAsSlice - EndiannessMismatch for Int16" {
    var bytes align(@alignOf(i16)) = [_]u8{ 1, 0 };
    const result = Element(i16).bytesAsSlice(&bytes, 1, .{ .Int16 = oppositeEndian(native_endian) });
    try std.testing.expectError(ViewDataError.EndiannessMismatch, result);
}

test "bytesAsSlice - EndiannessMismatch for Int32" {
    var bytes align(@alignOf(i32)) = [_]u8{ 1, 0, 0, 0 };
    const result = Element(i32).bytesAsSlice(&bytes, 1, .{ .Int32 = oppositeEndian(native_endian) });
    try std.testing.expectError(ViewDataError.EndiannessMismatch, result);
}

test "bytesAsSlice - EndiannessMismatch for Int64" {
    var bytes align(@alignOf(i64)) = [_]u8{0} ** 8;
    const result = Element(i64).bytesAsSlice(&bytes, 1, .{ .Int64 = oppositeEndian(native_endian) });
    try std.testing.expectError(ViewDataError.EndiannessMismatch, result);
}

test "bytesAsSlice - EndiannessMismatch for UInt16" {
    var bytes align(@alignOf(u16)) = [_]u8{ 1, 0 };
    const result = Element(u16).bytesAsSlice(&bytes, 1, .{ .UInt16 = oppositeEndian(native_endian) });
    try std.testing.expectError(ViewDataError.EndiannessMismatch, result);
}

test "bytesAsSlice - EndiannessMismatch for UInt32" {
    var bytes align(@alignOf(u32)) = [_]u8{ 1, 0, 0, 0 };
    const result = Element(u32).bytesAsSlice(&bytes, 1, .{ .UInt32 = oppositeEndian(native_endian) });
    try std.testing.expectError(ViewDataError.EndiannessMismatch, result);
}

test "bytesAsSlice - EndiannessMismatch for UInt64" {
    var bytes align(@alignOf(u64)) = [_]u8{0} ** 8;
    const result = Element(u64).bytesAsSlice(&bytes, 1, .{ .UInt64 = oppositeEndian(native_endian) });
    try std.testing.expectError(ViewDataError.EndiannessMismatch, result);
}

test "bytesAsSlice - EndiannessMismatch for Float32" {
    var bytes align(@alignOf(f32)) = [_]u8{0} ** 4;
    const result = Element(f32).bytesAsSlice(&bytes, 1, .{ .Float32 = oppositeEndian(native_endian) });
    try std.testing.expectError(ViewDataError.EndiannessMismatch, result);
}

test "bytesAsSlice - EndiannessMismatch for Float64" {
    var bytes align(@alignOf(f64)) = [_]u8{0} ** 8;
    const result = Element(f64).bytesAsSlice(&bytes, 1, .{ .Float64 = oppositeEndian(native_endian) });
    try std.testing.expectError(ViewDataError.EndiannessMismatch, result);
}

test "bytesAsSlice - EndiannessMismatch for Float128" {
    var bytes align(@alignOf(f128)) = [_]u8{0} ** 16;
    const result = Element(f128).bytesAsSlice(&bytes, 1, .{ .Float128 = oppositeEndian(native_endian) });
    try std.testing.expectError(ViewDataError.EndiannessMismatch, result);
}

test "bytesAsSlice - EndiannessMismatch for Complex64" {
    var bytes align(@alignOf(std.math.Complex(f32))) = [_]u8{0} ** 8;
    const result = Element(std.math.Complex(f32)).bytesAsSlice(&bytes, 1, .{ .Complex64 = oppositeEndian(native_endian) });
    try std.testing.expectError(ViewDataError.EndiannessMismatch, result);
}

test "bytesAsSlice - EndiannessMismatch for Complex128" {
    var bytes align(@alignOf(std.math.Complex(f64))) = [_]u8{0} ** 16;
    const result = Element(std.math.Complex(f64)).bytesAsSlice(&bytes, 1, .{ .Complex128 = oppositeEndian(native_endian) });
    try std.testing.expectError(ViewDataError.EndiannessMismatch, result);
}

test "bytesAsSlice - MissingBytes for Bool" {
    var bytes = [_]u8{ 0, 1, 0 };
    const result = Element(bool).bytesAsSlice(&bytes, 5, .Bool); // Requesting 5, only 3 available
    try std.testing.expectError(ViewDataError.MissingBytes, result);
}

test "bytesAsSlice - MissingBytes for Int32" {
    var bytes align(@alignOf(i32)) = [_]u8{ 0, 0, 0, 0, 1, 1 };
    const result = Element(i32).bytesAsSlice(&bytes, 2, .{ .Int32 = null }); // Requesting 2*4=8 bytes, only 6 available
    try std.testing.expectError(ViewDataError.MissingBytes, result);
}

test "bytesAsSlice - MissingBytes for Float64" {
    var bytes align(@alignOf(f64)) = [_]u8{0} ** 7;
    const result = Element(f64).bytesAsSlice(&bytes, 1, .{ .Float64 = null }); // Requesting 8 bytes, only 7 available
    try std.testing.expectError(ViewDataError.MissingBytes, result);
}

test "bytesAsSlice - ExtraBytes for Bool" {
    var bytes = [_]u8{ 0, 1, 0, 1, 0 };
    const result = Element(bool).bytesAsSlice(&bytes, 3, .Bool); // Only need 3, have 5
    try std.testing.expectError(ViewDataError.ExtraBytes, result);
}

test "bytesAsSlice - ExtraBytes for Int16" {
    var bytes align(@alignOf(i16)) = [_]u8{ 0, 0, 1, 0, 2, 0 };
    const result = Element(i16).bytesAsSlice(&bytes, 2, .{ .Int16 = null }); // Need 4 bytes, have 6
    try std.testing.expectError(ViewDataError.ExtraBytes, result);
}

test "bytesAsSlice - ExtraBytes for Float32" {
    var bytes align(@alignOf(f32)) = [_]u8{0} ** 8;
    const result = Element(f32).bytesAsSlice(&bytes, 1, .{ .Float32 = null }); // Need 4 bytes, have 8
    try std.testing.expectError(ViewDataError.ExtraBytes, result);
}

test "bytesAsSlice - LengthOverflow" {
    var bytes = [_]u8{0};
    // Try to allocate max usize elements of u64, which should overflow
    const huge_len = std.math.maxInt(usize);
    const result = Element(u64).bytesAsSlice(&bytes, huge_len, .{ .UInt64 = null });
    try std.testing.expectError(ViewDataError.LengthOverflow, result);
}

test "bytesAsSlice - zero-length bytes with non-zero len" {
    var bytes = [_]u8{};
    const result = Element(bool).bytesAsSlice(&bytes, 1, .Bool);
    try std.testing.expectError(ViewDataError.MissingBytes, result);
}

test "bytesAsSlice - zero len with non-empty bytes for Bool" {
    var bytes = [_]u8{ 0, 1 };
    const result = Element(bool).bytesAsSlice(&bytes, 0, .Bool);
    try std.testing.expectError(ViewDataError.ExtraBytes, result);
}

test "bytesAsSlice - zero len with non-empty bytes for Int32" {
    var bytes align(@alignOf(i32)) = [_]u8{ 0, 0, 0, 0 };
    const result = Element(i32).bytesAsSlice(&bytes, 0, .{ .Int32 = null });
    try std.testing.expectError(ViewDataError.ExtraBytes, result);
}

test "bytesAsSlice - zero len with empty bytes returns empty slice" {
    var bytes = [_]u8{};
    const result = try Element(bool).bytesAsSlice(&bytes, 0, .Bool);
    try std.testing.expectEqual(0, result.len);
}

test "bytesAsSlice - zero len with empty bytes for Int32" {
    var bytes = [_]u8{};
    const result = try Element(i32).bytesAsSlice(&bytes, 0, .{ .Int32 = null });
    try std.testing.expectEqual(0, result.len);
}

test "bytesAsSlice - Misaligned bytes for Int16" {
    // Create misaligned buffer
    var buffer align(@alignOf(i16)) = [_]u8{0} ** 5;
    const misaligned_slice = buffer[1..3]; // This should be misaligned for i16
    const result = Element(i16).bytesAsSlice(misaligned_slice, 1, .{ .Int16 = null });
    try std.testing.expectError(ViewDataError.Misaligned, result);
}

test "bytesAsSlice - Misaligned bytes for Int32" {
    var buffer align(@alignOf(i32)) = [_]u8{0} ** 7;
    const misaligned_slice = buffer[1..5];
    const result = Element(i32).bytesAsSlice(misaligned_slice, 1, .{ .Int32 = null });
    try std.testing.expectError(ViewDataError.Misaligned, result);
}

test "bytesAsSlice - Misaligned bytes for Int64" {
    var buffer align(@alignOf(i64)) = [_]u8{0} ** 12;
    const misaligned_slice = buffer[1..9];
    const result = Element(i64).bytesAsSlice(misaligned_slice, 1, .{ .Int64 = null });
    try std.testing.expectError(ViewDataError.Misaligned, result);
}

test "bytesAsSlice - Misaligned bytes for Float32" {
    var buffer align(@alignOf(f32)) = [_]u8{0} ** 7;
    const misaligned_slice = buffer[1..5];
    const result = Element(f32).bytesAsSlice(misaligned_slice, 1, .{ .Float32 = null });
    try std.testing.expectError(ViewDataError.Misaligned, result);
}

test "bytesAsSlice - Misaligned bytes for Float64" {
    var buffer align(@alignOf(f64)) = [_]u8{0} ** 12;
    const misaligned_slice = buffer[1..9];
    const result = Element(f64).bytesAsSlice(misaligned_slice, 1, .{ .Float64 = null });
    try std.testing.expectError(ViewDataError.Misaligned, result);
}

test "bytesAsSlice - Misaligned bytes for Complex64" {
    var buffer align(@alignOf(std.math.Complex(f32))) = [_]u8{0} ** 12;
    const misaligned_slice = buffer[1..9];
    const result = Element(std.math.Complex(f32)).bytesAsSlice(misaligned_slice, 1, .{ .Complex64 = null });
    try std.testing.expectError(ViewDataError.Misaligned, result);
}

test "bytesAsSlice - Misaligned bytes for Complex128" {
    var buffer align(@alignOf(std.math.Complex(f64))) = [_]u8{0} ** 20;
    const misaligned_slice = buffer[1..17];
    const result = Element(std.math.Complex(f64)).bytesAsSlice(misaligned_slice, 1, .{ .Complex128 = null });
    try std.testing.expectError(ViewDataError.Misaligned, result);
}

test "bytesAsSlice - TypeMismatch Bool vs Int8" {
    var bytes = [_]u8{ 0, 1, 0 };
    const result = Element(bool).bytesAsSlice(&bytes, 3, .Int8);
    try std.testing.expectError(ViewDataError.TypeMismatch, result);
}

test "bytesAsSlice - TypeMismatch Int32 vs Float32" {
    var bytes align(@alignOf(i32)) = [_]u8{ 0, 0, 0, 0 };
    const result = Element(i32).bytesAsSlice(&bytes, 1, .{ .Float32 = null });
    try std.testing.expectError(ViewDataError.TypeMismatch, result);
}

test "bytesAsSlice - TypeMismatch UInt64 vs Int64" {
    var bytes align(@alignOf(u64)) = [_]u8{0} ** 8;
    const result = Element(u64).bytesAsSlice(&bytes, 1, .{ .Int64 = null });
    try std.testing.expectError(ViewDataError.TypeMismatch, result);
}
