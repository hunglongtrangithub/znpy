const std = @import("std");

const header = @import("header.zig");
const elements = @import("elements.zig");
pub const dynamic = @import("shape/dynamic.zig");
pub const static = @import("shape/static.zig");

pub const DynamicShape = dynamic.DynamicShape;
pub const StaticShape = static.StaticShape;

/// Specifies how array data is laid out in memory
pub const Order = enum {
    /// Array data is in row-major order (C-contiguous)
    C,
    /// Array data is in column-major order (Fortran-contiguous)
    F,
};

/// Returns the number of elements in the array on success, or `null` if overflow occurs.
/// The number of bytes an array takes, given its shape and element type, must not exceed `std.math.maxInt(isize)`.
/// An empty shape (zero dimensions) is valid and has size 1 (a scalar).
/// A shape with any dimension of size zero is valid and has size 0.
pub fn shapeSizeChecked(T: elements.ElementType, shape: []const usize) ?usize {
    // NOTE: An empty shape (zero dimensions) is valid and has size 1 (a scalar).
    const num_elements = blk: {
        var prod: usize = 1;
        for (shape) |dim| {
            prod, const overflow = @mulWithOverflow(prod, dim);
            if (overflow != 0) {
                return null;
            }
        }
        break :blk prod;
    };

    const num_bytes: usize, const overflow: u1 = @mulWithOverflow(num_elements, T.byteSize());
    if ((overflow != 0) or (num_bytes > std.math.maxInt(isize))) {
        return null;
    }
    return num_elements;
}

test {
    _ = dynamic;
    _ = static;
}

test "shapeSizeChecked - normal shape" {
    // normal shape
    const shape = [_]usize{ 3, 4, 5 };
    const result = shapeSizeChecked(.{ .Float64 = null }, &shape);
    try std.testing.expectEqual(60, result.?);
}

test "shapeSizeChecked - empty shape" {
    // empty shape
    const shape = [_]usize{};
    const result = shapeSizeChecked(.{ .Float64 = null }, &shape);
    try std.testing.expectEqual(1, result.?);
}

test "shapeSizeChecked - shape with zero dimension" {
    const shape = [_]usize{ 3, 0, 5 };
    const result = shapeSizeChecked(.{ .Float64 = null }, &shape);
    try std.testing.expectEqual(0, result.?);
}

test "shapeSizeChecked - overflow in element count" {
    const shape = [_]usize{ std.math.maxInt(usize), 2 };
    const result = shapeSizeChecked(.{ .Float64 = null }, &shape);
    try std.testing.expectEqual(null, result);
}

test "shapeSizeChecked - overflow in byte count" {
    // Create a shape that doesn't overflow element count but does overflow byte count
    const huge = std.math.maxInt(usize) / 2;
    const shape = [_]usize{huge};
    const result = shapeSizeChecked(.{ .Float64 = null }, &shape); // 8 bytes per f64
    try std.testing.expectEqual(null, result);
}

test "shapeSizeChecked - exceeds isize max" {
    // Try to create a shape that exceeds maxInt(isize)
    const limit = @as(usize, std.math.maxInt(isize));
    const shape = [_]usize{limit / 7 + 1}; // Will exceed when multiplied by 8 (f64 size)
    const result = shapeSizeChecked(.{ .Float64 = null }, &shape);
    try std.testing.expectEqual(null, result);
}
