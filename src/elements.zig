//! Module for array element functions and types.
const std = @import("std");
const builtin = @import("builtin");

const header = @import("header.zig");

const boolean = @import("elements/boolean.zig");

const native_endian = builtin.cpu.arch.endian();

const ViewDataError = error{
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

pub fn Element(comptime T: type) type {
    const element_type = header.ElementType.fromZigType(T) catch @compileError("Unsupported type");

    return struct {
        value: T,

        const Self = @This();

        /// Interprets a byte slice as a slice of the specified element type `T`.
        pub fn bytesAsSlice(bytes: []const u8, len: usize, type_descr: header.ElementType) ViewDataError![]const T {
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
            // Either case, we can return an empty slice
            if (bytes.len == 0) {
                return &[_]T{};
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

        pub fn writeSlice(slice: []const Self, writer: std.io.Writer) WriteDataError!void {
            _ = slice;
            _ = writer;
        }

        pub fn readSlice(slice: []Self, reader: *std.io.Reader, type_descr: header.TypeDescriptor) ReadDataError!void {
            _ = slice;
            _ = reader;
            _ = type_descr;
        }
    };
}

test {
    _ = boolean;
}
