const std = @import("std");

const slice_mod = @import("../slice.zig");
const range_mod = @import("./range.zig");

const Slice = slice_mod.Slice;
const All = slice_mod.All;
const Etc = slice_mod.Etc;
const Range = range_mod.Range;

/// Format a tuple or struct of slice specifications into an array of Slice.
/// Each field can be:
/// - A `Slice` instance
/// - An integer (for index slice)
/// - A struct with up to three fields (for range slice):
///   - 1 field: start
///   - 2 fields: start, end
///   - 3 fields: start, end, step
///   Fields can be integers or optional integers. `null` indicates default value
///   for the respective field.
///   - Enum literals like `.All`, `.Etc`, etc. defined in `Slice` union
/// Example usage:
/// ```zig
/// const slice = @import("znpy").slice;
/// const format_slice = slice.format_slice;
/// const Slice = slice.Slice;
/// const All = slice.All;
/// const Etc = slice.Etc;
/// const slices = format_slice(.{
///     .{}, // Default slice
///     .{1}, // Slice with only start
///     .{ 1, 10 }, // Slice with start and end
///     .{ 1, 10, 2 }, // Slice with start, end and step
///     .{null, null, 2}, // Slice with default start/end and step of 2
///     .{1, null, 2}, // Slice with start, null end, and step
///     5, // Index slice
///     All,
///     Etc,
///    Slice{ .Range = slice.Range{ .start = 2, .end = 8 } }, // Already a Slice
/// });
/// ```
pub fn format_slice(args: anytype) [args.len]Slice {
    const ArgsType = @TypeOf(args);
    const args_type_info = @typeInfo(ArgsType);
    if (args_type_info != .@"struct") {
        @compileError("expected tuple or struct argument, found " ++ @typeName(ArgsType));
    }

    const fields_info = args_type_info.@"struct".fields;

    var result: [fields_info.len]Slice = undefined;

    inline for (fields_info, 0..) |field, i| {
        result[i] = toSlice(@field(args, field.name));
    }

    return result;
}

fn toSlice(comptime value: anytype) Slice {
    const ValueType = @TypeOf(value);

    // If it's already a Slice, return it directly
    if (ValueType == Slice) {
        return value;
    }

    const value_type_info = @typeInfo(ValueType);

    switch (value_type_info) {
        // Single integer (comptime known)
        .comptime_int => {
            const v: isize = @intCast(value);
            return Slice{ .Index = v };
        },
        // Normal integers
        .int => {
            const v = value;
            return Slice{ .Index = v };
        },
        // A `struct` — treat as Range
        .@"struct" => {
            const fields = value_type_info.@"struct".fields;

            // Default range
            var range = Range{};

            // Populate fields by count
            switch (fields.len) {
                0 => {},
                1 => {
                    const f0 = @field(value, fields[0].name);
                    assignOptInt(&range.start, f0, "start");
                },
                2 => {
                    const f0 = @field(value, fields[0].name);
                    const f1 = @field(value, fields[1].name);
                    assignOptInt(&range.start, f0, "start");
                    assignOptInt(&range.end, f1, "end");
                },
                3 => {
                    const f0 = @field(value, fields[0].name);
                    const f1 = @field(value, fields[1].name);
                    const f2 = @field(value, fields[2].name);
                    assignOptInt(&range.start, f0, "start");
                    assignOptInt(&range.end, f1, "end");
                    assignInt(&range.step, f2, "step");
                },
                else => @compileError("Too many fields in range struct (max 3)"),
            }

            return Slice{ .Range = range };
        },
        .@"enum" => {
            // Handle enum literals like .All, .Etc, .NewAxis, .Ellipsis
            // These should be part of the Slice union
            return @as(Slice, value);
        },
        else => @compileError("unsupported type for toSlice: " ++ @typeName(ValueType)),
    }
}

// Assign to an optional integer field in Range struct
fn assignOptInt(field: *?isize, val: anytype, comptime field_name: []const u8) void {
    const T = @TypeOf(val);
    const info = @typeInfo(T);

    switch (info) {
        .comptime_int, .int => {
            field.* = @intCast(val);
        },
        .null => {
            field.* = null;
        },
        .optional => |opt| {
            switch (@typeInfo(opt.child)) {
                .comptime_int, .int => {
                    if (val) |v| {
                        field.* = @intCast(v);
                    } else {
                        field.* = null;
                    }
                },
                else => @compileError("Field '" ++ field_name ++ "': expected optional integer, found " ++ @typeName(T)),
            }
        },
        else => @compileError("Field '" ++ field_name ++ "': expected integer or optional integer, found " ++ @typeName(T)),
    }
}

// Assign to an integer field in Range struct
fn assignInt(field: *isize, val: anytype, comptime field_name: []const u8) void {
    const T = @TypeOf(val);
    const info = @typeInfo(T);

    switch (info) {
        .comptime_int, .int => {
            field.* = @intCast(val);
        },
        .null => {
            // Null passed to non-optional field - skip assignment (keep default)
        },
        .optional => |opt| {
            switch (@typeInfo(opt.child)) {
                .comptime_int, .int => {
                    if (val) |v| {
                        field.* = @intCast(v);
                    }
                    // If null, keep the default value (don't assign)
                },
                else => @compileError("Field '" ++ field_name ++ "': expected optional integer, found " ++ @typeName(T)),
            }
        },
        else => @compileError("Field '" ++ field_name ++ "': expected integer or optional integer, found " ++ @typeName(T)),
    }
}

test "format_slice comprehensive" {
    const inputs = .{
        .{},
        .{ null, null, null },
        .{1},
        .{ 1, 10 },
        .{ 1, 10, null },
        .{ 1, 10, 2 },
        .{ -5, null, 2 },
        .{ null, null, 2 },
        .{ null, null, 0 },
        5,
        All,
        Etc,
        Slice{ .Range = Range{ .start = 2, .end = 8 } },
    };

    const expected = .{
        Slice{ .Range = Range{} },
        Slice{ .Range = Range{} },
        Slice{ .Range = Range{ .start = 1 } },
        Slice{ .Range = Range{ .start = 1, .end = 10 } },
        Slice{ .Range = Range{ .start = 1, .end = 10 } },
        Slice{ .Range = Range{ .start = 1, .end = 10, .step = 2 } },
        Slice{ .Range = Range{ .start = -5, .step = 2 } },
        Slice{ .Range = Range{ .step = 2 } },
        Slice{ .Index = 5 },
        Slice{ .Index = 0 },
        All,
        Etc,
        Slice{ .Range = Range{ .start = 2, .end = 8 } },
    };

    const slices = format_slice(inputs);

    inline for (0..expected.len) |i| {
        try std.testing.expectEqualDeep(expected[i], slices[i]);
    }
}
