const std = @import("std");
const builtin = @import("builtin");

const Endian = std.builtin.Endian;

pub const ParseDescrError = error{
    /// Descriptor string is too short to be valid.
    TooShort,
    /// Endianness character is invalid for the given type.
    InvalidEndianness,
    /// Value in descriptor is invalid.
    MissingSize,
    /// Type is either not a valid `numpy.dtype.descr`, or not supported.
    InvalidType,
};

/// Element type of the array data, parsed from the `descr` field in the npy file header.
/// Maps to NumPy dtype codes.
/// For multi-byte types, endianness can be specified. If endianness is null,
/// it indicates native endianness.
pub const ElementType = union(enum) {
    /// Boolean type - dtype codes: 'b1'
    Bool,
    /// 8-bit signed integer - dtype code: 'i1'
    Int8,
    /// 16-bit signed integer - dtype code: 'i2'
    Int16: ?Endian,
    /// 32-bit signed integer - dtype code: 'i4'
    Int32: ?Endian,
    /// 64-bit signed integer - dtype code: 'i8'
    Int64: ?Endian,
    /// 8-bit unsigned integer - dtype code: 'u1'
    UInt8,
    /// 16-bit unsigned integer - dtype code: 'u2'
    UInt16: ?Endian,
    /// 32-bit unsigned integer - dtype code: 'u4'
    UInt32: ?Endian,
    /// 64-bit unsigned integer - dtype code: 'u8'
    UInt64: ?Endian,
    /// 32-bit floating point - dtype code: 'f4'
    Float32: ?Endian,
    /// 64-bit floating point - dtype code: 'f8'
    Float64: ?Endian,
    /// 128-bit floating point - dtype code: 'f16'
    Float128: ?Endian,
    /// 32-bit floating point complex number - dtype code: 'c8'
    Complex64: ?Endian,
    /// 64-bit floating point complex number - dtype code: 'c16'
    Complex128: ?Endian,

    const Self = @This();

    pub const FromZigTypeError = error{UnsupportedType};

    /// Converts a Zig type to the corresponding ElementType, with endianness set to null (native).
    pub fn fromZigType(t: type) FromZigTypeError!Self {
        return switch (t) {
            bool => .Bool,
            i8 => .Int8,
            i16 => .{ .Int16 = null },
            i32 => .{ .Int32 = null },
            i64 => .{ .Int64 = null },
            u8 => .UInt8,
            u16 => .{ .UInt16 = null },
            u32 => .{ .UInt32 = null },
            u64 => .{ .UInt64 = null },
            f32 => .{ .Float32 = null },
            f64 => .{ .Float64 = null },
            f128 => .{ .Float128 = null },
            std.math.Complex(f32) => .{ .Complex64 = null },
            std.math.Complex(f64) => .{ .Complex128 = null },
            else => error.UnsupportedType,
        };
    }

    /// Converts the ElementType to the corresponding Zig type.
    pub fn toZigType(self: Self) type {
        return switch (self) {
            .Bool => bool,
            .Int8 => i8,
            .Int16 => i16,
            .Int32 => i32,
            .Int64 => i64,
            .UInt8 => u8,
            .UInt16 => u16,
            .UInt32 => u32,
            .UInt64 => u64,
            .Float32 => f32,
            .Float64 => f64,
            .Float128 => f128,
            .Complex64 => std.math.Complex(f32),
            .Complex128 => std.math.Complex(f64),
        };
    }

    /// Returns the size in bytes of the ElementType.
    pub fn byteSize(self: Self) usize {
        return switch (self) {
            .Bool => @sizeOf(bool),
            .Int8 => @sizeOf(i8),
            .UInt8 => @sizeOf(u8),
            .Int16 => @sizeOf(i16),
            .UInt16 => @sizeOf(u16),
            .Int32 => @sizeOf(i32),
            .UInt32 => @sizeOf(u32),
            .Int64 => @sizeOf(i64),
            .UInt64 => @sizeOf(u64),
            .Float32 => @sizeOf(f32),
            .Float64 => @sizeOf(f64),
            .Float128 => @sizeOf(f128),
            .Complex64 => @sizeOf(std.math.Complex(f32)),
            .Complex128 => @sizeOf(std.math.Complex(f64)),
        };
    }

    /// Parses the `descr` string the Python dictionary in npy file's header into a `ElementType`.
    /// All multi-byte types must have endianness specified.
    pub fn fromString(descr: []const u8) ParseDescrError!Self {
        if (descr.len < 3) {
            return ParseDescrError.TooShort;
        }

        const endian_char = descr[0];
        const type_char = descr[1];

        const EndianOrNA = union(enum) {
            Applicable: ?Endian,
            NotApplicable,
        };

        const endianness: EndianOrNA = switch (endian_char) {
            '<' => .{ .Applicable = .little },
            '>' => .{ .Applicable = .big },
            '=' => .{ .Applicable = null },
            '|' => .NotApplicable,
            else => return ParseDescrError.InvalidEndianness,
        };

        const element_type: ElementType = switch (type_char) {
            // Boolean type
            'b' => blk: {
                // Endianness must be NotApplicable
                if (endianness != EndianOrNA.NotApplicable) {
                    return ParseDescrError.InvalidEndianness;
                }
                // Descr must be '|b1'
                if (descr.len != 3 or descr[2] != '1') {
                    return ParseDescrError.InvalidType;
                }
                break :blk .Bool;
            },
            // Signed/unsigned integers
            'i', 'u' => blk: {
                const size_slice = descr[2..];

                if (size_slice.len != 1) {
                    return ParseDescrError.InvalidType;
                }
                const size_char = size_slice[0];

                // Endianness must not be NotApplicable if size is not 1, and vice versa
                if ((size_char == '1') != (endianness == EndianOrNA.NotApplicable)) {
                    return ParseDescrError.InvalidEndianness;
                }

                if (type_char == 'i') {
                    switch (size_char) {
                        '1' => break :blk .Int8,
                        '2' => break :blk .{ .Int16 = endianness.Applicable },
                        '4' => break :blk .{ .Int32 = endianness.Applicable },
                        '8' => break :blk .{ .Int64 = endianness.Applicable },
                        else => return ParseDescrError.InvalidType,
                    }
                } else { // type_char == 'u'
                    switch (size_char) {
                        '1' => break :blk .UInt8,
                        '2' => break :blk .{ .UInt16 = endianness.Applicable },
                        '4' => break :blk .{ .UInt32 = endianness.Applicable },
                        '8' => break :blk .{ .UInt64 = endianness.Applicable },
                        else => return ParseDescrError.InvalidType,
                    }
                }
            },
            // Floating point types
            'f' => blk: {
                const size_slice = descr[2..];

                // Endianness must not be NotApplicable
                if (endianness == EndianOrNA.NotApplicable) {
                    return ParseDescrError.InvalidEndianness;
                }

                switch (size_slice.len) {
                    1 => {
                        switch (size_slice[0]) {
                            '4' => break :blk .{ .Float32 = endianness.Applicable },
                            '8' => break :blk .{ .Float64 = endianness.Applicable },
                            else => return ParseDescrError.InvalidType,
                        }
                    },
                    2 => {
                        if (std.mem.eql(u8, size_slice, "16")) {
                            break :blk .{ .Float128 = endianness.Applicable };
                        } else {
                            return ParseDescrError.InvalidType;
                        }
                    },
                    else => return ParseDescrError.InvalidType,
                }
            },
            // Complex types
            'c' => blk: {
                const size_slice = descr[2..];

                // Endianness must not be NotApplicable
                if (endianness == EndianOrNA.NotApplicable) {
                    return ParseDescrError.InvalidEndianness;
                }

                if (std.mem.eql(u8, size_slice, "8")) {
                    break :blk .{ .Complex64 = endianness.Applicable };
                } else if (std.mem.eql(u8, size_slice, "16")) {
                    break :blk .{ .Complex128 = endianness.Applicable };
                } else {
                    return ParseDescrError.InvalidType;
                }
            },
            else => return ParseDescrError.InvalidType,
        };

        return element_type;
    }
};

test "parse bool dtype" {
    const result = try ElementType.fromString("|b1");
    try std.testing.expectEqual(ElementType.Bool, result);
}

test "parse signed integer dtypes" {
    // Int8 (|i1)
    const i8_result = try ElementType.fromString("|i1");
    try std.testing.expectEqual(ElementType.Int8, i8_result);

    // Int16 (<i2, >i2)
    const i16_little = try ElementType.fromString("<i2");
    try std.testing.expectEqual(Endian.little, i16_little.Int16.?);

    const i16_big = try ElementType.fromString(">i2");
    try std.testing.expectEqual(Endian.big, i16_big.Int16.?);

    // Int32 (<i4, >i4)
    const i32_little = try ElementType.fromString("<i4");
    try std.testing.expectEqual(Endian.little, i32_little.Int32.?);

    const i32_big = try ElementType.fromString(">i4");
    try std.testing.expectEqual(Endian.big, i32_big.Int32.?);

    // Int64 (<i8, >i8)
    const i64_little = try ElementType.fromString("<i8");
    try std.testing.expectEqual(Endian.little, i64_little.Int64.?);

    const i64_big = try ElementType.fromString(">i8");
    try std.testing.expectEqual(Endian.big, i64_big.Int64.?);
}

test "parse unsigned integer dtypes" {
    // UInt8 (|u1)
    const u8_result = try ElementType.fromString("|u1");
    try std.testing.expectEqual(ElementType.UInt8, u8_result);

    // UInt16 (<u2, >u2)
    const u16_little = try ElementType.fromString("<u2");
    try std.testing.expectEqual(Endian.little, u16_little.UInt16.?);

    const u16_big = try ElementType.fromString(">u2");
    try std.testing.expectEqual(Endian.big, u16_big.UInt16.?);

    // UInt32 (<u4, >u4)
    const u32_little = try ElementType.fromString("<u4");
    try std.testing.expectEqual(Endian.little, u32_little.UInt32.?);

    const u32_big = try ElementType.fromString(">u4");
    try std.testing.expectEqual(Endian.big, u32_big.UInt32.?);

    // UInt64 (<u8, >u8)
    const u64_little = try ElementType.fromString("<u8");
    try std.testing.expectEqual(Endian.little, u64_little.UInt64.?);

    const u64_big = try ElementType.fromString(">u8");
    try std.testing.expectEqual(Endian.big, u64_big.UInt64.?);
}

test "parse floating point dtypes" {
    // Float32 (<f4, >f4)
    const f32_little = try ElementType.fromString("<f4");
    try std.testing.expectEqual(Endian.little, f32_little.Float32.?);

    const f32_big = try ElementType.fromString(">f4");
    try std.testing.expectEqual(Endian.big, f32_big.Float32.?);

    // Float64 (<f8, >f8)
    const f64_little = try ElementType.fromString("<f8");
    try std.testing.expectEqual(Endian.little, f64_little.Float64.?);

    const f64_big = try ElementType.fromString(">f8");
    try std.testing.expectEqual(Endian.big, f64_big.Float64.?);

    // Float128 (<f16, >f16)
    const f128_little = try ElementType.fromString("<f16");
    try std.testing.expectEqual(Endian.little, f128_little.Float128.?);

    const f128_big = try ElementType.fromString(">f16");
    try std.testing.expectEqual(Endian.big, f128_big.Float128.?);
}

test "parse complex dtypes" {
    // Complex64 (<c8, >c8)
    const c64_little = try ElementType.fromString("<c8");
    try std.testing.expectEqual(Endian.little, c64_little.Complex64.?);

    const c64_big = try ElementType.fromString(">c8");
    try std.testing.expectEqual(Endian.big, c64_big.Complex64.?);

    // Complex128 (<c16, >c16)
    const c128_little = try ElementType.fromString("<c16");
    try std.testing.expectEqual(Endian.little, c128_little.Complex128.?);

    const c128_big = try ElementType.fromString(">c16");
    try std.testing.expectEqual(Endian.big, c128_big.Complex128.?);
}

test "parse with native endianness" {
    const i32_native = try ElementType.fromString("=i4");
    try std.testing.expect(i32_native.Int32 == null);

    const u64_native = try ElementType.fromString("=u8");
    try std.testing.expect(u64_native.UInt64 == null);

    const f64_native = try ElementType.fromString("=f8");
    try std.testing.expect(f64_native.Float64 == null);
}

test "error on too short descr" {
    try std.testing.expectError(ParseDescrError.TooShort, ElementType.fromString(""));
    try std.testing.expectError(ParseDescrError.TooShort, ElementType.fromString("<"));
}

test "error on invalid endianness" {
    try std.testing.expectError(ParseDescrError.InvalidEndianness, ElementType.fromString("@i4"));
    try std.testing.expectError(ParseDescrError.InvalidEndianness, ElementType.fromString("!f8"));
}

test "error on invalid endianness for bool" {
    try std.testing.expectError(ParseDescrError.InvalidEndianness, ElementType.fromString("<b1"));
    try std.testing.expectError(ParseDescrError.InvalidEndianness, ElementType.fromString(">b1"));
    try std.testing.expectError(ParseDescrError.InvalidEndianness, ElementType.fromString("=b1"));
}

test "error on invalid endianness for i1/u1" {
    try std.testing.expectError(ParseDescrError.InvalidEndianness, ElementType.fromString("<i1"));
    try std.testing.expectError(ParseDescrError.InvalidEndianness, ElementType.fromString(">u1"));
    try std.testing.expectError(ParseDescrError.InvalidEndianness, ElementType.fromString("=i1"));
}

test "error on invalid endianness for multi-byte integers" {
    try std.testing.expectError(ParseDescrError.InvalidEndianness, ElementType.fromString("|i2"));
    try std.testing.expectError(ParseDescrError.InvalidEndianness, ElementType.fromString("|u4"));
    try std.testing.expectError(ParseDescrError.InvalidEndianness, ElementType.fromString("|i8"));
}

test "error on invalid endianness for floats" {
    try std.testing.expectError(ParseDescrError.InvalidEndianness, ElementType.fromString("|f4"));
    try std.testing.expectError(ParseDescrError.InvalidEndianness, ElementType.fromString("|f8"));
    try std.testing.expectError(ParseDescrError.InvalidEndianness, ElementType.fromString("|f16"));
}

test "error on invalid bool size" {
    try std.testing.expectError(ParseDescrError.InvalidType, ElementType.fromString("|b2"));
    try std.testing.expectError(ParseDescrError.InvalidType, ElementType.fromString("|b4"));
    try std.testing.expectError(ParseDescrError.TooShort, ElementType.fromString("|b"));
}

test "error on unsupported integer sizes" {
    try std.testing.expectError(ParseDescrError.InvalidType, ElementType.fromString("<i3"));
    try std.testing.expectError(ParseDescrError.InvalidType, ElementType.fromString(">u5"));
    try std.testing.expectError(ParseDescrError.InvalidType, ElementType.fromString("=i16"));
}

test "error on unsupported float sizes" {
    try std.testing.expectError(ParseDescrError.InvalidType, ElementType.fromString("<f2"));
    try std.testing.expectError(ParseDescrError.InvalidType, ElementType.fromString(">f12"));
    try std.testing.expectError(ParseDescrError.InvalidType, ElementType.fromString("=f32"));
}

test "error on unsupported type characters" {
    try std.testing.expectError(ParseDescrError.InvalidType, ElementType.fromString(">S10"));
    try std.testing.expectError(ParseDescrError.InvalidType, ElementType.fromString("=U5"));
    try std.testing.expectError(ParseDescrError.TooShort, ElementType.fromString("|O"));
    try std.testing.expectError(ParseDescrError.InvalidType, ElementType.fromString("<M8"));
}

test "error on invalid integer descr length" {
    try std.testing.expectError(ParseDescrError.TooShort, ElementType.fromString("<i"));
    try std.testing.expectError(ParseDescrError.TooShort, ElementType.fromString(">u"));
    try std.testing.expectError(ParseDescrError.InvalidType, ElementType.fromString("<i44"));
}

test "error on invalid float descr length" {
    try std.testing.expectError(ParseDescrError.TooShort, ElementType.fromString("<f"));
    try std.testing.expectError(ParseDescrError.InvalidType, ElementType.fromString(">f123"));
}

test "error on invalid complex descr length" {
    try std.testing.expectError(ParseDescrError.TooShort, ElementType.fromString("<c"));
    try std.testing.expectError(ParseDescrError.InvalidType, ElementType.fromString(">c24"));
    try std.testing.expectError(ParseDescrError.InvalidType, ElementType.fromString(">c32"));
}
