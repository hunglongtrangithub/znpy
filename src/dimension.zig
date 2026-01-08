const std = @import("std");

const header = @import("header.zig");

/// Returns the number of elements in the array on success, or `null` if overflow occurs.
/// The number of bytes an array takes, given its shape and element type, must not exceed `std.math.maxInt(isize)`
pub fn shapeSizeChecked(T: header.ElementType, shape: []const usize) ?usize {
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

/// A multi-dimensional shape of an array, with a dynamic number of dimensions.
const Shape = struct {
    /// The size of each dimension. The slice is owned by the caller.
    /// The total number of elements is the product of all dimensions,
    /// which must not overflow `std.math.maxInt(isize)`.
    dims: []const usize,
    /// The memory order of the array.
    order: header.Order,

    const Self = @This();

    const FromHeaderError = error{ShapeSizeOverflow};

    /// Create a Shape from a numpy header.
    /// Returns an error if the total number of elements overflows `isize`.
    pub fn fromHeader(npy_header: header.Header) FromHeaderError!Self {
        // Check that the shape length fits in isize
        _ = shapeSizeChecked(npy_header.descr, npy_header.shape) orelse {
            return FromHeaderError.ShapeSizeOverflow;
        };
        return Self{
            .dims = npy_header.shape,
            .order = npy_header.order,
        };
    }

    /// Compute the strides for this shape, allocating the result using the given allocator.
    /// Strides slice should have the same length as the number of dimensions in the shape.
    pub fn getStrides(self: *const Self, allocator: std.mem.Allocator) std.mem.Allocator.Error![]const isize {
        if (self.dims.len == 0) {
            // Scalar case: no dimensions, no strides
            return &[_]isize{};
        }

        var strides = try allocator.alloc(isize, self.dims.len);
        @memset(strides, 0);

        if (std.mem.indexOfScalar(usize, self.dims, 0)) |_| {
            // If any dimension is zero, all strides are zero
            return strides;
        }

        return switch (self.order) {
            .C => blk: {
                // Shape (a, b, c) => Give strides (b * c, c, 1)
                var stride: isize = 1;
                for (0..self.dims.len) |i_rev| {
                    // NOTE: self.dims is not empty, so this index is safe
                    const i = self.dims.len - 1 - i_rev;
                    strides[i] = stride;
                    // NOTE: this cast is safe because we have already verified that
                    // the total size does not overflow isize, which means the individual
                    // dimensions and strides must also fit in isize.
                    const dim = self.dims[i];
                    stride *= @intCast(dim);
                }
                break :blk strides;
            },
            .F => blk: {
                // Shape (a, b, c) => Give strides (1, a, a * b)
                var stride: isize = 1;
                for (self.dims, 0..) |dim, i| {
                    strides[i] = stride;
                    // NOTE: this cast is safe because we have already verified that
                    // the total size does not overflow isize, which means the individual
                    // dimensions and strides must also fit in isize.
                    stride *= @intCast(dim);
                }
                break :blk strides;
            },
        };
    }
};

test "shapeSizeChecked - normal shape" {
    const shape = [_]usize{ 3, 4, 5 };
    const result = shapeSizeChecked(.{ .Float64 = null }, &shape);
    try std.testing.expectEqual(@as(usize, 60), result.?);
}

test "shapeSizeChecked - empty shape" {
    const shape = [_]usize{};
    const result = shapeSizeChecked(.{ .Float64 = null }, &shape);
    try std.testing.expectEqual(@as(usize, 1), result.?);
}

test "shapeSizeChecked - shape with zero dimension" {
    const shape = [_]usize{ 3, 0, 5 };
    const result = shapeSizeChecked(.{ .Float64 = null }, &shape);
    try std.testing.expectEqual(@as(usize, 0), result.?);
}

test "shapeSizeChecked - overflow in element count" {
    const shape = [_]usize{ std.math.maxInt(usize), 2 };
    const result = shapeSizeChecked(.{ .Float64 = null }, &shape);
    try std.testing.expectEqual(@as(?usize, null), result);
}

test "shapeSizeChecked - overflow in byte count" {
    // Create a shape that doesn't overflow element count but does overflow byte count
    const huge = std.math.maxInt(usize) / 2;
    const shape = [_]usize{huge};
    const result = shapeSizeChecked(.{ .Float64 = null }, &shape); // 8 bytes per f64
    try std.testing.expectEqual(@as(?usize, null), result);
}

test "shapeSizeChecked - exceeds isize max" {
    // Try to create a shape that exceeds maxInt(isize)
    const limit = @as(usize, @intCast(std.math.maxInt(isize)));
    const shape = [_]usize{limit / 7 + 1}; // Will exceed when multiplied by 8 (f64 size)
    const result = shapeSizeChecked(.{ .Float64 = null }, &shape);
    try std.testing.expectEqual(@as(?usize, null), result);
}

test "Shape.fromHeader - valid shape" {
    const allocator = std.testing.allocator;
    var shape_data = [_]usize{ 2, 3, 4 };
    const npy_header = header.Header{
        .shape = &shape_data,
        .descr = .{ .Float64 = null },
        .order = .C,
    };
    const shape = try Shape.fromHeader(npy_header);
    try std.testing.expectEqual(@as(usize, 3), shape.dims.len);
    try std.testing.expectEqual(@as(usize, 2), shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 3), shape.dims[1]);
    try std.testing.expectEqual(@as(usize, 4), shape.dims[2]);
    try std.testing.expectEqual(header.Order.C, shape.order);
    _ = allocator;
}

test "Shape.fromHeader - overflow error" {
    const allocator = std.testing.allocator;
    var shape_data = [_]usize{ std.math.maxInt(usize), 2 };
    const npy_header = header.Header{
        .shape = &shape_data,
        .descr = .{ .Float64 = null },
        .order = .C,
    };
    const result = Shape.fromHeader(npy_header);
    try std.testing.expectError(Shape.FromHeaderError.ShapeSizeOverflow, result);
    _ = allocator;
}

test "Shape.getStrides - empty shape" {
    const allocator = std.testing.allocator;
    const shape_data = [_]usize{};
    const shape = Shape{
        .dims = &shape_data,
        .order = .C,
    };
    const strides = try shape.getStrides(allocator);
    defer allocator.free(strides);
    try std.testing.expectEqual(@as(usize, 0), strides.len);
}

test "Shape.getStrides - shape with zero dimension C order" {
    const allocator = std.testing.allocator;
    var shape_data = [_]usize{ 3, 0, 5 };
    const npy_header = header.Header{
        .shape = &shape_data,
        .descr = .{ .Float64 = null },
        .order = .C,
    };
    const shape = try Shape.fromHeader(npy_header);
    const strides = try shape.getStrides(allocator);
    defer allocator.free(strides);
    try std.testing.expectEqual(@as(usize, 3), strides.len);
    try std.testing.expectEqual(@as(isize, 0), strides[0]);
    try std.testing.expectEqual(@as(isize, 0), strides[1]);
    try std.testing.expectEqual(@as(isize, 0), strides[2]);
}

test "Shape.getStrides - shape with zero dimension F order" {
    const allocator = std.testing.allocator;
    var shape_data = [_]usize{ 3, 0, 5 };
    const npy_header = header.Header{
        .shape = &shape_data,
        .descr = .{ .Float64 = null },
        .order = .F,
    };
    const shape = try Shape.fromHeader(npy_header);
    const strides = try shape.getStrides(allocator);
    defer allocator.free(strides);
    try std.testing.expectEqual(@as(usize, 3), strides.len);
    try std.testing.expectEqual(@as(isize, 0), strides[0]);
    try std.testing.expectEqual(@as(isize, 0), strides[1]);
    try std.testing.expectEqual(@as(isize, 0), strides[2]);
}

test "Shape.getStrides - C order (2, 3, 4)" {
    const allocator = std.testing.allocator;
    const shape_data = [_]usize{ 2, 3, 4 };
    const shape = Shape{
        .dims = &shape_data,
        .order = .C,
    };
    const strides = try shape.getStrides(allocator);
    defer allocator.free(strides);
    try std.testing.expectEqual(@as(usize, 3), strides.len);
    // C order: strides are (3*4, 4, 1) = (12, 4, 1)
    try std.testing.expectEqual(@as(isize, 12), strides[0]);
    try std.testing.expectEqual(@as(isize, 4), strides[1]);
    try std.testing.expectEqual(@as(isize, 1), strides[2]);
}

test "Shape.getStrides - F order (2, 3, 4)" {
    const allocator = std.testing.allocator;
    const shape_data = [_]usize{ 2, 3, 4 };
    const shape = Shape{
        .dims = &shape_data,
        .order = .F,
    };
    const strides = try shape.getStrides(allocator);
    defer allocator.free(strides);
    try std.testing.expectEqual(@as(usize, 3), strides.len);
    // F order: strides are (1, 2, 2*3) = (1, 2, 6)
    try std.testing.expectEqual(@as(isize, 1), strides[0]);
    try std.testing.expectEqual(@as(isize, 2), strides[1]);
    try std.testing.expectEqual(@as(isize, 6), strides[2]);
}

test "Shape.getStrides - C order 1D array" {
    const allocator = std.testing.allocator;
    const shape_data = [_]usize{10};
    const shape = Shape{
        .dims = &shape_data,
        .order = .C,
    };
    const strides = try shape.getStrides(allocator);
    defer allocator.free(strides);
    try std.testing.expectEqual(@as(usize, 1), strides.len);
    try std.testing.expectEqual(@as(isize, 1), strides[0]);
}

test "Shape.getStrides - F order 1D array" {
    const allocator = std.testing.allocator;
    const shape_data = [_]usize{10};
    const shape = Shape{
        .dims = &shape_data,
        .order = .F,
    };
    const strides = try shape.getStrides(allocator);
    defer allocator.free(strides);
    try std.testing.expectEqual(@as(usize, 1), strides.len);
    try std.testing.expectEqual(@as(isize, 1), strides[0]);
}

test "Shape.getStrides - C order 2D array" {
    const allocator = std.testing.allocator;
    const shape_data = [_]usize{ 5, 7 };
    const shape = Shape{
        .dims = &shape_data,
        .order = .C,
    };
    const strides = try shape.getStrides(allocator);
    defer allocator.free(strides);
    try std.testing.expectEqual(@as(usize, 2), strides.len);
    // C order: strides are (7, 1)
    try std.testing.expectEqual(@as(isize, 7), strides[0]);
    try std.testing.expectEqual(@as(isize, 1), strides[1]);
}

test "Shape.getStrides - F order 2D array" {
    const allocator = std.testing.allocator;
    const shape_data = [_]usize{ 5, 7 };
    const shape = Shape{
        .dims = &shape_data,
        .order = .F,
    };
    const strides = try shape.getStrides(allocator);
    defer allocator.free(strides);
    try std.testing.expectEqual(@as(usize, 2), strides.len);
    // F order: strides are (1, 5)
    try std.testing.expectEqual(@as(isize, 1), strides[0]);
    try std.testing.expectEqual(@as(isize, 5), strides[1]);
}

test "Shape.getStrides - C order 4D array" {
    const allocator = std.testing.allocator;
    const shape_data = [_]usize{ 2, 3, 4, 5 };
    const shape = Shape{
        .dims = &shape_data,
        .order = .C,
    };
    const strides = try shape.getStrides(allocator);
    defer allocator.free(strides);
    try std.testing.expectEqual(@as(usize, 4), strides.len);
    // C order: strides are (3*4*5, 4*5, 5, 1) = (60, 20, 5, 1)
    try std.testing.expectEqual(@as(isize, 60), strides[0]);
    try std.testing.expectEqual(@as(isize, 20), strides[1]);
    try std.testing.expectEqual(@as(isize, 5), strides[2]);
    try std.testing.expectEqual(@as(isize, 1), strides[3]);
}

test "Shape.getStrides - F order 4D array" {
    const allocator = std.testing.allocator;
    const shape_data = [_]usize{ 2, 3, 4, 5 };
    const shape = Shape{
        .dims = &shape_data,
        .order = .F,
    };
    const strides = try shape.getStrides(allocator);
    defer allocator.free(strides);
    try std.testing.expectEqual(@as(usize, 4), strides.len);
    // F order: strides are (1, 2, 2*3, 2*3*4) = (1, 2, 6, 24)
    try std.testing.expectEqual(@as(isize, 1), strides[0]);
    try std.testing.expectEqual(@as(isize, 2), strides[1]);
    try std.testing.expectEqual(@as(isize, 6), strides[2]);
    try std.testing.expectEqual(@as(isize, 24), strides[3]);
}
