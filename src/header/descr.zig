const std = @import("std");

/// Endianness of the data stored in the .npy file.
pub const Endianness = enum {
    /// Little-endian byte order (e.g., '<f8')
    Little,
    /// Big-endian byte order (e.g., '>f8')
    Big,
    /// Native byte order (e.g., '=f8')
    Native,
    /// Not applicable
    NotApplicable,
};

/// Element type of the array data.
/// Maps to NumPy dtype codes.
pub const ElementType = enum {
    /// Boolean type - dtype codes: 'b1'
    Bool,
    /// 8-bit signed integer - dtype code: 'i1'
    Int8,
    /// 16-bit signed integer - dtype code: 'i2'
    Int16,
    /// 32-bit signed integer - dtype code: 'i4'
    Int32,
    /// 64-bit signed integer - dtype code: 'i8'
    Int64,
    /// 8-bit unsigned integer - dtype code: 'u1'
    UInt8,
    /// 16-bit unsigned integer - dtype code: 'u2'
    UInt16,
    /// 32-bit unsigned integer - dtype code: 'u4'
    UInt32,
    /// 64-bit unsigned integer - dtype code: 'u8'
    UInt64,
    /// 32-bit floating point - dtype code: 'f4'
    Float32,
    /// 64-bit floating point - dtype code: 'f8'
    Float64,
    /// 128-bit floating point - dtype code: 'f16'
    Float128,
};

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

/// Type descriptor parsed from the 'descr' field in a .npy file header.
pub const TypeDescriptor = struct {
    endian: Endianness,
    element_type: ElementType,

    const Self = @This();

    pub fn fromString(descr: []const u8) ParseDescrError!Self {
        if (descr.len < 3) {
            return ParseDescrError.TooShort;
        }

        const endian_char = descr[0];
        const type_char = descr[1];

        const endianness: Endianness = switch (endian_char) {
            '<' => .Little,
            '>' => .Big,
            '=' => .Native,
            '|' => .NotApplicable,
            else => return ParseDescrError.InvalidEndianness,
        };

        const element_type: ElementType = switch (type_char) {
            // Boolean type
            'b' => blk: {
                // Endianness must be NotApplicable
                if (endianness != Endianness.NotApplicable) {
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
                if ((size_char == '1') != (endianness == Endianness.NotApplicable)) {
                    return ParseDescrError.InvalidEndianness;
                }

                if (type_char == 'i') {
                    switch (size_char) {
                        '1' => break :blk .Int8,
                        '2' => break :blk .Int16,
                        '4' => break :blk .Int32,
                        '8' => break :blk .Int64,
                        else => return ParseDescrError.InvalidType,
                    }
                } else { // type_char == 'u'
                    switch (size_char) {
                        '1' => break :blk .UInt8,
                        '2' => break :blk .UInt16,
                        '4' => break :blk .UInt32,
                        '8' => break :blk .UInt64,
                        else => return ParseDescrError.InvalidType,
                    }
                }
            },
            // Floating point types
            'f' => blk: {
                const size_slice = descr[2..];

                // Endianness must not be NotApplicable
                if (endianness == Endianness.NotApplicable) {
                    return ParseDescrError.InvalidEndianness;
                }

                switch (size_slice.len) {
                    1 => {
                        switch (size_slice[0]) {
                            '4' => break :blk .Float32,
                            '8' => break :blk .Float64,
                            else => return ParseDescrError.InvalidType,
                        }
                    },
                    2 => {
                        if (std.mem.eql(u8, size_slice, "16")) {
                            break :blk .Float128;
                        } else {
                            return ParseDescrError.InvalidType;
                        }
                    },
                    else => return ParseDescrError.InvalidType,
                }
            },
            else => return ParseDescrError.InvalidType,
        };

        return TypeDescriptor{
            .endian = endianness,
            .element_type = element_type,
        };
    }
};

test "parse bool dtype" {
    const result = try TypeDescriptor.fromString("|b1");
    try std.testing.expectEqual(Endianness.NotApplicable, result.endian);
    try std.testing.expectEqual(ElementType.Bool, result.element_type);
}

test "parse signed integer dtypes" {
    // Int8 (|i1)
    const i8_result = try TypeDescriptor.fromString("|i1");
    try std.testing.expectEqual(Endianness.NotApplicable, i8_result.endian);
    try std.testing.expectEqual(ElementType.Int8, i8_result.element_type);

    // Int16 (<i2, >i2)
    const i16_little = try TypeDescriptor.fromString("<i2");
    try std.testing.expectEqual(Endianness.Little, i16_little.endian);
    try std.testing.expectEqual(ElementType.Int16, i16_little.element_type);

    const i16_big = try TypeDescriptor.fromString(">i2");
    try std.testing.expectEqual(Endianness.Big, i16_big.endian);
    try std.testing.expectEqual(ElementType.Int16, i16_big.element_type);

    // Int32 (<i4, >i4)
    const i32_little = try TypeDescriptor.fromString("<i4");
    try std.testing.expectEqual(Endianness.Little, i32_little.endian);
    try std.testing.expectEqual(ElementType.Int32, i32_little.element_type);

    const i32_big = try TypeDescriptor.fromString(">i4");
    try std.testing.expectEqual(Endianness.Big, i32_big.endian);
    try std.testing.expectEqual(ElementType.Int32, i32_big.element_type);

    // Int64 (<i8, >i8)
    const i64_little = try TypeDescriptor.fromString("<i8");
    try std.testing.expectEqual(Endianness.Little, i64_little.endian);
    try std.testing.expectEqual(ElementType.Int64, i64_little.element_type);

    const i64_big = try TypeDescriptor.fromString(">i8");
    try std.testing.expectEqual(Endianness.Big, i64_big.endian);
    try std.testing.expectEqual(ElementType.Int64, i64_big.element_type);
}

test "parse unsigned integer dtypes" {
    // UInt8 (|u1)
    const u8_result = try TypeDescriptor.fromString("|u1");
    try std.testing.expectEqual(Endianness.NotApplicable, u8_result.endian);
    try std.testing.expectEqual(ElementType.UInt8, u8_result.element_type);

    // UInt16 (<u2, >u2)
    const u16_little = try TypeDescriptor.fromString("<u2");
    try std.testing.expectEqual(Endianness.Little, u16_little.endian);
    try std.testing.expectEqual(ElementType.UInt16, u16_little.element_type);

    const u16_big = try TypeDescriptor.fromString(">u2");
    try std.testing.expectEqual(Endianness.Big, u16_big.endian);
    try std.testing.expectEqual(ElementType.UInt16, u16_big.element_type);

    // UInt32 (<u4, >u4)
    const u32_little = try TypeDescriptor.fromString("<u4");
    try std.testing.expectEqual(Endianness.Little, u32_little.endian);
    try std.testing.expectEqual(ElementType.UInt32, u32_little.element_type);

    const u32_big = try TypeDescriptor.fromString(">u4");
    try std.testing.expectEqual(Endianness.Big, u32_big.endian);
    try std.testing.expectEqual(ElementType.UInt32, u32_big.element_type);

    // UInt64 (<u8, >u8)
    const u64_little = try TypeDescriptor.fromString("<u8");
    try std.testing.expectEqual(Endianness.Little, u64_little.endian);
    try std.testing.expectEqual(ElementType.UInt64, u64_little.element_type);

    const u64_big = try TypeDescriptor.fromString(">u8");
    try std.testing.expectEqual(Endianness.Big, u64_big.endian);
    try std.testing.expectEqual(ElementType.UInt64, u64_big.element_type);
}

test "parse floating point dtypes" {
    // Float32 (<f4, >f4)
    const f32_little = try TypeDescriptor.fromString("<f4");
    try std.testing.expectEqual(Endianness.Little, f32_little.endian);
    try std.testing.expectEqual(ElementType.Float32, f32_little.element_type);

    const f32_big = try TypeDescriptor.fromString(">f4");
    try std.testing.expectEqual(Endianness.Big, f32_big.endian);
    try std.testing.expectEqual(ElementType.Float32, f32_big.element_type);

    // Float64 (<f8, >f8)
    const f64_little = try TypeDescriptor.fromString("<f8");
    try std.testing.expectEqual(Endianness.Little, f64_little.endian);
    try std.testing.expectEqual(ElementType.Float64, f64_little.element_type);

    const f64_big = try TypeDescriptor.fromString(">f8");
    try std.testing.expectEqual(Endianness.Big, f64_big.endian);
    try std.testing.expectEqual(ElementType.Float64, f64_big.element_type);

    // Float128 (<f16, >f16)
    const f128_little = try TypeDescriptor.fromString("<f16");
    try std.testing.expectEqual(Endianness.Little, f128_little.endian);
    try std.testing.expectEqual(ElementType.Float128, f128_little.element_type);

    const f128_big = try TypeDescriptor.fromString(">f16");
    try std.testing.expectEqual(Endianness.Big, f128_big.endian);
    try std.testing.expectEqual(ElementType.Float128, f128_big.element_type);
}

test "parse with native endianness" {
    const i32_native = try TypeDescriptor.fromString("=i4");
    try std.testing.expectEqual(Endianness.Native, i32_native.endian);
    try std.testing.expectEqual(ElementType.Int32, i32_native.element_type);

    const u64_native = try TypeDescriptor.fromString("=u8");
    try std.testing.expectEqual(Endianness.Native, u64_native.endian);
    try std.testing.expectEqual(ElementType.UInt64, u64_native.element_type);

    const f64_native = try TypeDescriptor.fromString("=f8");
    try std.testing.expectEqual(Endianness.Native, f64_native.endian);
    try std.testing.expectEqual(ElementType.Float64, f64_native.element_type);
}

test "error on too short descr" {
    try std.testing.expectError(ParseDescrError.TooShort, TypeDescriptor.fromString(""));
    try std.testing.expectError(ParseDescrError.TooShort, TypeDescriptor.fromString("<"));
}

test "error on invalid endianness" {
    try std.testing.expectError(ParseDescrError.InvalidEndianness, TypeDescriptor.fromString("@i4"));
    try std.testing.expectError(ParseDescrError.InvalidEndianness, TypeDescriptor.fromString("!f8"));
}

test "error on invalid endianness for bool" {
    try std.testing.expectError(ParseDescrError.InvalidEndianness, TypeDescriptor.fromString("<b1"));
    try std.testing.expectError(ParseDescrError.InvalidEndianness, TypeDescriptor.fromString(">b1"));
    try std.testing.expectError(ParseDescrError.InvalidEndianness, TypeDescriptor.fromString("=b1"));
}

test "error on invalid endianness for i1/u1" {
    try std.testing.expectError(ParseDescrError.InvalidEndianness, TypeDescriptor.fromString("<i1"));
    try std.testing.expectError(ParseDescrError.InvalidEndianness, TypeDescriptor.fromString(">u1"));
    try std.testing.expectError(ParseDescrError.InvalidEndianness, TypeDescriptor.fromString("=i1"));
}

test "error on invalid endianness for multi-byte integers" {
    try std.testing.expectError(ParseDescrError.InvalidEndianness, TypeDescriptor.fromString("|i2"));
    try std.testing.expectError(ParseDescrError.InvalidEndianness, TypeDescriptor.fromString("|u4"));
    try std.testing.expectError(ParseDescrError.InvalidEndianness, TypeDescriptor.fromString("|i8"));
}

test "error on invalid endianness for floats" {
    try std.testing.expectError(ParseDescrError.InvalidEndianness, TypeDescriptor.fromString("|f4"));
    try std.testing.expectError(ParseDescrError.InvalidEndianness, TypeDescriptor.fromString("|f8"));
    try std.testing.expectError(ParseDescrError.InvalidEndianness, TypeDescriptor.fromString("|f16"));
}

test "error on invalid bool size" {
    try std.testing.expectError(ParseDescrError.InvalidValue, TypeDescriptor.fromString("|b2"));
    try std.testing.expectError(ParseDescrError.InvalidValue, TypeDescriptor.fromString("|b4"));
    try std.testing.expectError(ParseDescrError.TooShort, TypeDescriptor.fromString("|b"));
}

test "error on unsupported integer sizes" {
    try std.testing.expectError(ParseDescrError.UnsupportedType, TypeDescriptor.fromString("<i3"));
    try std.testing.expectError(ParseDescrError.UnsupportedType, TypeDescriptor.fromString(">u5"));
    try std.testing.expectError(ParseDescrError.UnsupportedType, TypeDescriptor.fromString("=i16"));
}

test "error on unsupported float sizes" {
    try std.testing.expectError(ParseDescrError.UnsupportedType, TypeDescriptor.fromString("<f2"));
    try std.testing.expectError(ParseDescrError.UnsupportedType, TypeDescriptor.fromString(">f12"));
    try std.testing.expectError(ParseDescrError.UnsupportedType, TypeDescriptor.fromString("=f32"));
}

test "error on unsupported type characters" {
    try std.testing.expectError(ParseDescrError.UnsupportedType, TypeDescriptor.fromString("<c8"));
    try std.testing.expectError(ParseDescrError.UnsupportedType, TypeDescriptor.fromString(">S10"));
    try std.testing.expectError(ParseDescrError.UnsupportedType, TypeDescriptor.fromString("=U5"));
    try std.testing.expectError(ParseDescrError.TooShort, TypeDescriptor.fromString("|O"));
    try std.testing.expectError(ParseDescrError.UnsupportedType, TypeDescriptor.fromString("<M8"));
}

test "error on invalid integer descr length" {
    try std.testing.expectError(ParseDescrError.TooShort, TypeDescriptor.fromString("<i"));
    try std.testing.expectError(ParseDescrError.TooShort, TypeDescriptor.fromString(">u"));
    try std.testing.expectError(ParseDescrError.UnsupportedType, TypeDescriptor.fromString("<i44"));
}

test "error on invalid float descr length" {
    try std.testing.expectError(ParseDescrError.TooShort, TypeDescriptor.fromString("<f"));
    try std.testing.expectError(ParseDescrError.UnsupportedType, TypeDescriptor.fromString(">f123"));
}
