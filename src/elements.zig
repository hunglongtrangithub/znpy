//! Module for array element functions and types.
const std = @import("std");
const builtin = @import("builtin");

const boolean = @import("elements/boolean.zig");
const types = @import("elements/types.zig");
const array = @import("./array.zig");
const pointer = @import("./pointer.zig");

const native_endian = builtin.cpu.arch.endian();

pub const ElementType = types.ElementType;

pub const TypeCompatibilityError = error{
    /// The provided element type does not match the expected type.
    TypeMismatch,
    /// The provided data contains invalid boolean values (not 0 or 1).
    InvalidBool,
    /// The endianness of the provided data does not match the native endianness.
    EndiannessMismatch,
};

pub const DataLayoutError = error{
    /// The requested length causes an overflow when calculating the total byte size.
    LengthOverflow,
    /// Not enough bytes provided to fulfill the requested length.
    MissingBytes,
    /// More bytes provided than necessary for the requested length.
    ExtraBytes,
    /// The provided byte slice is not properly aligned for the target type.
    Misaligned,
};

pub const ViewDataError = TypeCompatibilityError || DataLayoutError;

pub const ReadDataError = TypeCompatibilityError || std.io.Reader.Error;

/// A generic element type representing a single element of type `T` in a numpy array.
pub fn Element(comptime T: type) type {
    const element_type = ElementType.fromZigType(T) catch @compileError("Unsupported type");

    return struct {
        const Self = @This();

        /// Validates that bytes are compatible with the target type.
        /// - For bool: ensures all bytes are 0 or 1
        /// - For multi-byte types: ensures endianness matches native
        fn validateTypeCompatibility(bytes: []const u8, type_descr: ElementType) TypeCompatibilityError!void {
            switch (type_descr) {
                .Bool => {
                    // All bytes have to be either 0 or 1
                    if (!boolean.isAllZeroOrOne(bytes)) {
                        return TypeCompatibilityError.InvalidBool;
                    }
                },
                .Int8, .UInt8 => {},
                .Int16,
                .Int32,
                .Int64,
                .UInt16,
                .UInt32,
                .UInt64,
                .Float32,
                .Float64,
                .Float128,
                .Complex64,
                .Complex128,
                => |endian| {
                    // Endianness must match native endianness for multi-byte types
                    if (endian) |e| {
                        if (e != native_endian) {
                            return TypeCompatibilityError.EndiannessMismatch;
                        }
                    }
                    // endian is null -> assume native endian
                },
            }
        }

        /// Validates the provided byte slice against the expected type and length.
        /// Check for type compatibility and data layout issues.
        fn validateBytes(
            bytes: []const u8,
            len: usize,
            type_descr: ElementType,
        ) ViewDataError!void {
            // Check type match
            if (std.meta.activeTag(type_descr) != std.meta.activeTag(element_type)) {
                return ViewDataError.TypeMismatch;
            }

            // Validate type compatibility
            try validateTypeCompatibility(bytes, type_descr);

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

            // Check alignment. Skip if length is zero
            if (bytes.len > 0 and !std.mem.isAligned(@intFromPtr(bytes.ptr), @alignOf(T))) {
                return ViewDataError.Misaligned;
            }
        }

        /// Interprets a byte slice (`[]u8` or `[]const u8`) as either `[]T` or `[]const T`
        /// depending on input byte slice, for a given length and element type description.
        /// The function checks for type compatibility, length, and alignment.
        pub fn bytesAsSlice(
            bytes: anytype,
            len: usize,
            type_descr: ElementType,
        ) ViewDataError!if (pointer.isConstPtr(@TypeOf(bytes))) []const T else []T {
            // Validate the input bytes against the expected type and length
            try validateBytes(bytes, len, type_descr);

            // At this point, bytes.len is zero when:
            // 1. len is zero, or
            // 2. @sizeOf(T) is zero (though this should not happen for valid numpy types)
            // Either case, we can return an empty slice. But that empty slice needs to be properly aligned,
            // otherwise Zig just attaches an undefined pointer to the slice which may be misaligned.
            if (bytes.len == 0) {
                return pointer.danglingPtr(T)[0..0];
            }

            // Now bytes is non-empty, len is non-zero, and @sizeOf(T) is non-zero
            // Both length and alignment checks passed
            // Now upgrade the alignment in the type system and cast based on constness
            if (comptime pointer.isConstPtr(@TypeOf(bytes))) {
                const aligned_bytes = @as([]align(@alignOf(T)) const u8, @alignCast(bytes));
                return std.mem.bytesAsSlice(T, aligned_bytes);
            } else {
                const aligned_bytes = @as([]align(@alignOf(T)) u8, @alignCast(bytes));
                return std.mem.bytesAsSlice(T, aligned_bytes);
            }
        }

        /// Reads data from a reader into the provided slice of type `T`.
        /// Performs type compatibility check. If endianness mismatch is detected, byte swapping is performed.
        pub fn readSlice(slice: []T, reader: *std.io.Reader, type_descr: ElementType) ReadDataError!void {
            // Check type match
            if (std.meta.activeTag(type_descr) != std.meta.activeTag(element_type)) {
                return TypeCompatibilityError.TypeMismatch;
            }
            const bytes: []u8 = @ptrCast(slice);
            // Cast to a byte slice and read all bytes
            try reader.readSliceAll(bytes);
            // Validate type compatibility
            validateTypeCompatibility(bytes, type_descr) catch |e| {
                switch (e) {
                    TypeCompatibilityError.EndiannessMismatch => {
                        // Perform byte swap. Should work on Complex types as well
                        std.mem.byteSwapAllElements(T, slice);
                    },
                    else => return e,
                }
            };
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
    try std.testing.expectEqualSlices(bool, &[_]bool{ false, true, true, false, true }, result);
}

test "bytesAsSlice - Int8 with null endian" {
    var bytes = [_]u8{ 0, 127, 255, 128, 1 };
    const result = try Element(i8).bytesAsSlice(&bytes, 5, .Int8);
    try std.testing.expectEqualSlices(i8, &[_]i8{ 0, 127, -1, -128, 1 }, result);
}

test "bytesAsSlice - UInt8 with null endian" {
    var bytes = [_]u8{ 0, 127, 255, 128, 1 };
    const result = try Element(u8).bytesAsSlice(&bytes, 5, .UInt8);
    try std.testing.expectEqualSlices(u8, &[_]u8{ 0, 127, 255, 128, 1 }, result);
}

test "bytesAsSlice - Int16 with null endian" {
    var bytes align(@alignOf(i16)) = [_]u8{ 0, 0, 1, 0, 255, 255 };
    const result = try Element(i16).bytesAsSlice(&bytes, 3, .{ .Int16 = null });
    try std.testing.expectEqualSlices(i16, &[_]i16{
        0,
        if (native_endian == .little) 1 else 256,
        -1,
    }, result);
}

test "bytesAsSlice - Int32 with null endian" {
    var bytes align(@alignOf(i32)) = [_]u8{ 0, 0, 0, 0, 1, 0, 0, 0 };
    const result = try Element(i32).bytesAsSlice(&bytes, 2, .{ .Int32 = null });
    try std.testing.expectEqualSlices(i32, &[_]i32{
        0,
        if (native_endian == .little) 1 else 0b100_0000,
    }, result);
}

test "bytesAsSlice - Int64 with null endian" {
    var bytes align(@alignOf(i64)) = [_]u8{ 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255 };
    const result = try Element(i64).bytesAsSlice(&bytes, 2, .{ .Int64 = null });
    try std.testing.expectEqualSlices(i64, &[_]i64{
        0,
        -1,
    }, result);
}

test "bytesAsSlice - UInt16 with null endian" {
    var bytes align(@alignOf(u16)) = [_]u8{ 0, 0, 255, 255 };
    const result = try Element(u16).bytesAsSlice(&bytes, 2, .{ .UInt16 = null });
    try std.testing.expectEqualSlices(u16, &[_]u16{
        0,
        0xFFFF,
    }, result);
}

test "bytesAsSlice - UInt32 with null endian" {
    var bytes align(@alignOf(u32)) = [_]u8{ 0, 0, 0, 0, 255, 255, 255, 255 };
    const result = try Element(u32).bytesAsSlice(&bytes, 2, .{ .UInt32 = null });
    try std.testing.expectEqualSlices(u32, &[_]u32{
        0,
        0xFFFF_FFFF,
    }, result);
}

test "bytesAsSlice - UInt64 with null endian" {
    var bytes align(@alignOf(u64)) = [_]u8{ 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255 };
    const result = try Element(u64).bytesAsSlice(&bytes, 2, .{ .UInt64 = null });
    try std.testing.expectEqualSlices(u64, &[_]u64{
        0,
        0xFFFF_FFFF_FFFF_FFFF,
    }, result);
}

test "bytesAsSlice - Float32 with null endian" {
    var bytes align(@alignOf(f32)) = [_]u8{0} ** 8;
    // 0.0 and 1.0 in IEEE 754 binary32 format (native endian)
    std.mem.writeInt(u32, bytes[0..4], @as(u32, @bitCast(@as(f32, 0.0))), native_endian);
    std.mem.writeInt(u32, bytes[4..8], @as(u32, @bitCast(@as(f32, 1.0))), native_endian);
    const result = try Element(f32).bytesAsSlice(&bytes, 2, .{ .Float32 = null });
    try std.testing.expectEqualSlices(f32, &[_]f32{
        0.0,
        1.0,
    }, result);
}

test "bytesAsSlice - Float64 with null endian" {
    var bytes align(@alignOf(f64)) = [_]u8{0} ** 16;
    std.mem.writeInt(u64, bytes[0..8], @as(u64, @bitCast(@as(f64, 0.0))), native_endian);
    std.mem.writeInt(u64, bytes[8..16], @as(u64, @bitCast(@as(f64, 3.14159))), native_endian);
    const result = try Element(f64).bytesAsSlice(&bytes, 2, .{ .Float64 = null });
    try std.testing.expectEqualSlices(f64, &[_]f64{
        0.0,
        3.14159,
    }, result);
}

test "bytesAsSlice - Float128 with null endian" {
    var bytes align(@alignOf(f128)) = [_]u8{0} ** 32;
    std.mem.writeInt(u128, bytes[0..16], @as(u128, @bitCast(@as(f128, 0.0))), native_endian);
    std.mem.writeInt(u128, bytes[16..32], @as(u128, @bitCast(@as(f128, 1.0))), native_endian);
    const result = try Element(f128).bytesAsSlice(&bytes, 2, .{ .Float128 = null });
    try std.testing.expectEqualSlices(f128, &[_]f128{
        0.0,
        1.0,
    }, result);
}

test "bytesAsSlice - Complex64 with null endian" {
    var bytes align(@alignOf(std.math.Complex(f32))) = [_]u8{0} ** 16;
    // Complex(1.0, 2.0) and Complex(3.0, 4.0)
    std.mem.writeInt(u32, bytes[0..4], @as(u32, @bitCast(@as(f32, 1.0))), native_endian); // real part
    std.mem.writeInt(u32, bytes[4..8], @as(u32, @bitCast(@as(f32, 2.0))), native_endian); // imag part
    std.mem.writeInt(u32, bytes[8..12], @as(u32, @bitCast(@as(f32, 3.0))), native_endian); // real part
    std.mem.writeInt(u32, bytes[12..16], @as(u32, @bitCast(@as(f32, 4.0))), native_endian); // imag part
    const result = try Element(std.math.Complex(f32)).bytesAsSlice(&bytes, 2, .{ .Complex64 = null });

    try std.testing.expectEqualSlices(std.math.Complex(f32), &[_]std.math.Complex(f32){
        .{ .re = 1.0, .im = 2.0 },
        .{ .re = 3.0, .im = 4.0 },
    }, result);
}

test "bytesAsSlice - Complex128 with null endian" {
    var bytes align(@alignOf(std.math.Complex(f64))) = [_]u8{0} ** 32;
    std.mem.writeInt(u64, bytes[0..8], @as(u64, @bitCast(@as(f64, 1.5))), native_endian);
    std.mem.writeInt(u64, bytes[8..16], @as(u64, @bitCast(@as(f64, 2.5))), native_endian);
    std.mem.writeInt(u64, bytes[16..24], @as(u64, @bitCast(@as(f64, 3.5))), native_endian);
    std.mem.writeInt(u64, bytes[24..32], @as(u64, @bitCast(@as(f64, 4.5))), native_endian);
    const result = try Element(std.math.Complex(f64)).bytesAsSlice(&bytes, 2, .{ .Complex128 = null });

    try std.testing.expectEqualSlices(std.math.Complex(f64), &[_]std.math.Complex(f64){
        .{ .re = 1.5, .im = 2.5 },
        .{ .re = 3.5, .im = 4.5 },
    }, result);
}

test "bytesAsSlice - Int16 with explicit native endian" {
    var bytes align(@alignOf(i16)) = [_]u8{ 1, 0, 2, 0 };
    const result = try Element(i16).bytesAsSlice(&bytes, 2, .{ .Int16 = native_endian });

    try std.testing.expectEqualSlices(i16, &[_]i16{
        if (native_endian == .little) 1 else 0x1_0000,
        if (native_endian == .little) 2 else 0x2_0000,
    }, result);
}

test "bytesAsSlice - Int32 with explicit native endian" {
    var bytes align(@alignOf(i32)) = [_]u8{ 0xFE, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE };
    const result = try Element(i32).bytesAsSlice(&bytes, 2, .{ .Int32 = native_endian });
    try std.testing.expectEqualSlices(i32, &[_]i32{
        if (native_endian == .little) -2 else -0x01_00_00_01,
        if (native_endian == .little) -0x01_00_00_01 else -2,
    }, result);
}

test "bytesAsSlice - Int64 with explicit native endian" {
    var bytes align(@alignOf(i64)) = [_]u8{0} ** 16;
    const result = try Element(i64).bytesAsSlice(&bytes, 2, .{ .Int64 = native_endian });
    try std.testing.expectEqualSlices(i64, &[_]i64{
        0,
        0,
    }, result);
}

test "bytesAsSlice - UInt16 with explicit native endian" {
    var bytes align(@alignOf(u16)) = [_]u8{ 1, 0, 2, 0 };
    const result = try Element(u16).bytesAsSlice(&bytes, 2, .{ .UInt16 = native_endian });
    try std.testing.expectEqualSlices(u16, &[_]u16{
        if (native_endian == .little) 1 else 0x1_0000,
        if (native_endian == .little) 2 else 0x2_0000,
    }, result);
}

test "bytesAsSlice - UInt32 with explicit native endian" {
    var bytes align(@alignOf(u32)) = [_]u8{ 1, 0, 0, 0 };
    const result = try Element(u32).bytesAsSlice(&bytes, 1, .{ .UInt32 = native_endian });
    try std.testing.expectEqualSlices(u32, &[_]u32{
        if (native_endian == .little) 1 else 0x01_00_00_00,
    }, result);
}

test "bytesAsSlice - UInt64 with explicit native endian" {
    var bytes align(@alignOf(u64)) = [_]u8{0} ** 8;
    const result = try Element(u64).bytesAsSlice(&bytes, 1, .{ .UInt64 = native_endian });
    try std.testing.expectEqualSlices(u64, &[_]u64{
        0,
    }, result);
}

test "bytesAsSlice - Float32 with explicit native endian" {
    var bytes align(@alignOf(f32)) = [_]u8{0} ** 4;
    std.mem.writeInt(u32, &bytes, @as(u32, @bitCast(@as(f32, 1.0))), native_endian);
    const result = try Element(f32).bytesAsSlice(&bytes, 1, .{ .Float32 = native_endian });
    try std.testing.expectEqualSlices(f32, &[_]f32{1.0}, result);
}

test "bytesAsSlice - Float64 with explicit native endian" {
    var bytes align(@alignOf(f64)) = [_]u8{0} ** 8;
    std.mem.writeInt(u64, &bytes, @as(u64, @bitCast(@as(f64, 2.5))), native_endian);
    const result = try Element(f64).bytesAsSlice(&bytes, 1, .{ .Float64 = native_endian });
    try std.testing.expectEqualSlices(f64, &[_]f64{2.5}, result);
}

test "bytesAsSlice - Float128 with explicit native endian" {
    var bytes align(@alignOf(f128)) = [_]u8{0} ** 16;
    std.mem.writeInt(u128, &bytes, @as(u128, @bitCast(@as(f128, 1.0))), native_endian);
    const result = try Element(f128).bytesAsSlice(&bytes, 1, .{ .Float128 = native_endian });
    try std.testing.expectEqualSlices(f128, &[_]f128{1.0}, result);
}

test "bytesAsSlice - Complex64 with explicit native endian" {
    var bytes align(@alignOf(std.math.Complex(f32))) = [_]u8{0} ** 8;
    std.mem.writeInt(u32, bytes[0..4], @as(u32, @bitCast(@as(f32, 1.0))), native_endian);
    std.mem.writeInt(u32, bytes[4..8], @as(u32, @bitCast(@as(f32, 2.0))), native_endian);
    const result = try Element(std.math.Complex(f32)).bytesAsSlice(&bytes, 1, .{ .Complex64 = native_endian });
    try std.testing.expectEqualSlices(std.math.Complex(f32), &[_]std.math.Complex(f32){
        .{ .re = 1.0, .im = 2.0 },
    }, result);
}

test "bytesAsSlice - Complex128 with explicit native endian" {
    var bytes align(@alignOf(std.math.Complex(f64))) = [_]u8{0} ** 16;
    std.mem.writeInt(u64, bytes[0..8], @as(u64, @bitCast(@as(f64, 3.0))), native_endian);
    std.mem.writeInt(u64, bytes[8..16], @as(u64, @bitCast(@as(f64, 4.0))), native_endian);
    const result = try Element(std.math.Complex(f64)).bytesAsSlice(&bytes, 1, .{ .Complex128 = native_endian });
    try std.testing.expectEqualSlices(std.math.Complex(f64), &[_]std.math.Complex(f64){
        .{ .re = 3.0, .im = 4.0 },
    }, result);
}

test "bytesAsSlice - InvalidBool error" {
    var bytes = [_]u8{ 0, 1, 2, 1, 255 }; // byte with value 2 or 255 is invalid
    const result = Element(bool).bytesAsSlice(&bytes, 5, .Bool);
    try std.testing.expectError(ViewDataError.InvalidBool, result);
}

test "bytesAsSlice - EndiannessMismatch for multi-byte types" {
    {
        var bytes align(@alignOf(i16)) = [_]u8{ 1, 0 };
        const result = Element(i16).bytesAsSlice(&bytes, 1, .{ .Int16 = oppositeEndian(native_endian) });
        try std.testing.expectError(ViewDataError.EndiannessMismatch, result);
    }
    {
        var bytes align(@alignOf(i32)) = [_]u8{ 1, 0, 0, 0 };
        const result = Element(i32).bytesAsSlice(&bytes, 1, .{ .Int32 = oppositeEndian(native_endian) });
        try std.testing.expectError(ViewDataError.EndiannessMismatch, result);
    }
    {
        var bytes align(@alignOf(i64)) = [_]u8{0} ** 8;
        const result = Element(i64).bytesAsSlice(&bytes, 1, .{ .Int64 = oppositeEndian(native_endian) });
        try std.testing.expectError(ViewDataError.EndiannessMismatch, result);
    }
    {
        var bytes align(@alignOf(u16)) = [_]u8{ 1, 0 };
        const result = Element(u16).bytesAsSlice(&bytes, 1, .{ .UInt16 = oppositeEndian(native_endian) });
        try std.testing.expectError(ViewDataError.EndiannessMismatch, result);
    }
    {
        var bytes align(@alignOf(u32)) = [_]u8{ 1, 0, 0, 0 };
        const result = Element(u32).bytesAsSlice(&bytes, 1, .{ .UInt32 = oppositeEndian(native_endian) });
        try std.testing.expectError(ViewDataError.EndiannessMismatch, result);
    }
    {
        var bytes align(@alignOf(u64)) = [_]u8{0} ** 8;
        const result = Element(u64).bytesAsSlice(&bytes, 1, .{ .UInt64 = oppositeEndian(native_endian) });
        try std.testing.expectError(ViewDataError.EndiannessMismatch, result);
    }
    {
        var bytes align(@alignOf(f32)) = [_]u8{0} ** 4;
        const result = Element(f32).bytesAsSlice(&bytes, 1, .{ .Float32 = oppositeEndian(native_endian) });
        try std.testing.expectError(ViewDataError.EndiannessMismatch, result);
    }
    {
        var bytes align(@alignOf(f64)) = [_]u8{0} ** 8;
        const result = Element(f64).bytesAsSlice(&bytes, 1, .{ .Float64 = oppositeEndian(native_endian) });
        try std.testing.expectError(ViewDataError.EndiannessMismatch, result);
    }
    {
        var bytes align(@alignOf(f128)) = [_]u8{0} ** 16;
        const result = Element(f128).bytesAsSlice(&bytes, 1, .{ .Float128 = oppositeEndian(native_endian) });
        try std.testing.expectError(ViewDataError.EndiannessMismatch, result);
    }
    {
        var bytes align(@alignOf(std.math.Complex(f32))) = [_]u8{0} ** 8;
        const result = Element(std.math.Complex(f32)).bytesAsSlice(&bytes, 1, .{ .Complex64 = oppositeEndian(native_endian) });
        try std.testing.expectError(ViewDataError.EndiannessMismatch, result);
    }
    {
        var bytes align(@alignOf(std.math.Complex(f64))) = [_]u8{0} ** 16;
        const result = Element(std.math.Complex(f64)).bytesAsSlice(&bytes, 1, .{ .Complex128 = oppositeEndian(native_endian) });
        try std.testing.expectError(ViewDataError.EndiannessMismatch, result);
    }
    {
        var bytes = [_]u8{ 0, 1, 0 };
        const result = Element(bool).bytesAsSlice(&bytes, 5, .Bool); // Requesting 5, only 3 available
        try std.testing.expectError(ViewDataError.MissingBytes, result);
    }
    {
        var bytes align(@alignOf(i32)) = [_]u8{ 0, 0, 0, 0, 1, 1 };
        const result = Element(i32).bytesAsSlice(&bytes, 2, .{ .Int32 = null }); // Requesting 2*4=8 bytes, only 6 available
        try std.testing.expectError(ViewDataError.MissingBytes, result);
    }
    {
        var bytes align(@alignOf(f64)) = [_]u8{0} ** 7;
        const result = Element(f64).bytesAsSlice(&bytes, 1, .{ .Float64 = null }); // Requesting 8 bytes, only 7 available
        try std.testing.expectError(ViewDataError.MissingBytes, result);
    }
}

test "bytesAsSlice - ExtraBytes" {
    {
        var bytes = [_]u8{ 0, 1, 0, 1, 0 };
        const result = Element(bool).bytesAsSlice(&bytes, 3, .Bool); // Only need 3, have 5
        try std.testing.expectError(ViewDataError.ExtraBytes, result);
    }
    {
        var bytes align(@alignOf(i16)) = [_]u8{ 0, 0, 1, 0, 2, 0 };
        const result = Element(i16).bytesAsSlice(&bytes, 2, .{ .Int16 = null }); // Need 4 bytes, have 6
        try std.testing.expectError(ViewDataError.ExtraBytes, result);
    }
    {
        var bytes align(@alignOf(f32)) = [_]u8{0} ** 8;
        const result = Element(f32).bytesAsSlice(&bytes, 1, .{ .Float32 = null }); // Need 4 bytes, have 8
        try std.testing.expectError(ViewDataError.ExtraBytes, result);
    }
}

test "bytesAsSlice - LengthOverflow" {
    var bytes = [_]u8{0};
    // Try to allocate max usize elements of u64, which should overflow
    const huge_len = std.math.maxInt(usize);
    const result = Element(u64).bytesAsSlice(&bytes, huge_len, .{ .UInt64 = null });
    try std.testing.expectError(ViewDataError.LengthOverflow, result);
}

test "bytesAsSlice - zero-length bytes with non-zero len" {
    {
        var bytes = [_]u8{};
        const result = Element(bool).bytesAsSlice(&bytes, 1, .Bool);
        try std.testing.expectError(ViewDataError.MissingBytes, result);
    }
    {
        var bytes = [_]u8{ 0, 1 };
        const result = Element(bool).bytesAsSlice(&bytes, 0, .Bool);
        try std.testing.expectError(ViewDataError.ExtraBytes, result);
    }
    {
        var bytes align(@alignOf(i32)) = [_]u8{ 0, 0, 0, 0 };
        const result = Element(i32).bytesAsSlice(&bytes, 0, .{ .Int32 = null });
        try std.testing.expectError(ViewDataError.ExtraBytes, result);
    }
}

test "bytesAsSlice - zero len with empty bytes returns empty slice" {
    {
        var bytes = [_]u8{};
        const result = try Element(bool).bytesAsSlice(&bytes, 0, .Bool);
        try std.testing.expectEqual(0, result.len);
    }
    {
        var bytes = [_]u8{};
        const result = try Element(i32).bytesAsSlice(&bytes, 0, .{ .Int32 = null });
        try std.testing.expectEqual(0, result.len);
    }
}

test "bytesAsSlice - Misaligned bytes for multi-byte types" {
    {
        var buffer align(@alignOf(i16)) = [_]u8{0} ** 5;
        const misaligned_slice = buffer[1..3]; // This should be misaligned for i16
        const result = Element(i16).bytesAsSlice(misaligned_slice, 1, .{ .Int16 = null });
        try std.testing.expectError(ViewDataError.Misaligned, result);
    }
    {
        var buffer align(@alignOf(i32)) = [_]u8{0} ** 7;
        const misaligned_slice = buffer[1..5];
        const result = Element(i32).bytesAsSlice(misaligned_slice, 1, .{ .Int32 = null });
        try std.testing.expectError(ViewDataError.Misaligned, result);
    }
    {
        var buffer align(@alignOf(i64)) = [_]u8{0} ** 12;
        const misaligned_slice = buffer[1..9];
        const result = Element(i64).bytesAsSlice(misaligned_slice, 1, .{ .Int64 = null });
        try std.testing.expectError(ViewDataError.Misaligned, result);
    }
    {
        var buffer align(@alignOf(f32)) = [_]u8{0} ** 7;
        const misaligned_slice = buffer[1..5];
        const result = Element(f32).bytesAsSlice(misaligned_slice, 1, .{ .Float32 = null });
        try std.testing.expectError(ViewDataError.Misaligned, result);
    }
    {
        var buffer align(@alignOf(f64)) = [_]u8{0} ** 12;
        const misaligned_slice = buffer[1..9];
        const result = Element(f64).bytesAsSlice(misaligned_slice, 1, .{ .Float64 = null });
        try std.testing.expectError(ViewDataError.Misaligned, result);
    }
    {
        var buffer align(@alignOf(std.math.Complex(f32))) = [_]u8{0} ** 12;
        const misaligned_slice = buffer[1..9];
        const result = Element(std.math.Complex(f32)).bytesAsSlice(misaligned_slice, 1, .{ .Complex64 = null });
        try std.testing.expectError(ViewDataError.Misaligned, result);
    }
    {
        var buffer align(@alignOf(std.math.Complex(f64))) = [_]u8{0} ** 20;
        const misaligned_slice = buffer[1..17];
        const result = Element(std.math.Complex(f64)).bytesAsSlice(misaligned_slice, 1, .{ .Complex128 = null });
        try std.testing.expectError(ViewDataError.Misaligned, result);
    }
}

test "bytesAsSlice - TypeMismatch" {
    {
        var bytes = [_]u8{ 0, 1, 0 };
        const result = Element(bool).bytesAsSlice(&bytes, 3, .Int8);
        try std.testing.expectError(ViewDataError.TypeMismatch, result);
    }
    {
        var bytes align(@alignOf(i32)) = [_]u8{ 0, 0, 0, 0 };
        const result = Element(i32).bytesAsSlice(&bytes, 1, .{ .Float32 = null });
        try std.testing.expectError(ViewDataError.TypeMismatch, result);
    }
    {
        var bytes align(@alignOf(u64)) = [_]u8{0} ** 8;
        const result = Element(u64).bytesAsSlice(&bytes, 1, .{ .Int64 = null });
        try std.testing.expectError(ViewDataError.TypeMismatch, result);
    }
}
