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

/// A `Shape` represents the dimensions and memory order of a numpy array.
/// It can be either statically (`Rank` is non-null) or dynamically ranked (`Rank` is null).
///
/// The shape does not own the dimension data; the caller is responsible for ensuring
/// that the `dims` slice remains valid for the lifetime of the `Shape`.
/// The total number of elements in the shape must not overflow `std.math.maxInt(isize)`.
pub fn Shape(comptime Rank: ?usize) type {
    return struct {
        /// The size of each dimension. The slice is owned by the caller.
        /// The total number of elements is the product of all dimensions,
        /// which must not overflow `std.math.maxInt(isize)`.
        dims: if (Rank) |R| [R]usize else []const usize,
        /// The memory order of the array.
        order: header.Order,

        const Self = @This();

        pub const FromHeaderError = error{ ShapeSizeOverflow, DimensionMismatch };

        pub const StridesType = if (Rank) |R| [R]isize else []isize;

        /// Create a `Shape` from a numpy header.
        /// Returns an error if the shape's total size in bytes overflows isize,
        /// or if the number of dimensions does not match the expected Rank (if static).
        /// On success, also returns the total number of elements.
        pub fn fromHeader(npy_header: header.Header) FromHeaderError!struct { Self, usize } {
            // Extract shape, checking rank if static
            const shape = blk: {
                if (Rank) |R| {
                    if (npy_header.shape.len != R) {
                        return error.DimensionMismatch;
                    }
                    var static_dims: [R]usize = undefined;
                    @memcpy(&static_dims, npy_header.shape[0..R]);
                    break :blk static_dims;
                } else {
                    break :blk npy_header.shape;
                }
            };

            // Check that the shape length fits in isize
            const num_elements = shapeSizeChecked(npy_header.descr, shape[0..]) orelse {
                return error.ShapeSizeOverflow;
            };
            return .{ Self{
                .dims = shape,
                .order = npy_header.order,
            }, num_elements };
        }

        /// Compute the strides for this shape, allocating the result using the given allocator (if dynamic).
        /// Strides slice has the same length as the number of dimensions in the shape.
        ///
        /// **Invariant**: For any valid multi-dimensional index (where `0 <= indices[i] < dims[i]` for all `i`),
        /// the offset calculation `Σ(indices[i] * strides[i])` is guaranteed to:
        /// 1. Be less than the total number of elements in the shape
        /// 2. Fit in isize without overflow
        pub fn getStrides(
            self: *const Self,
            // If static, this arg is 'void'. If dynamic, it's 'Allocator'.
            allocator: if (Rank == null) std.mem.Allocator else void,
        ) if (Rank == null) std.mem.Allocator.Error!StridesType else StridesType {
            var strides: StridesType = if (Rank) |R|
                [_]isize{0} ** R
            else blk: {
                const s = try allocator.alloc(isize, self.dims.len);
                @memset(s, 0);
                break :blk s;
            };

            const len = if (Rank) |R| R else self.dims.len;

            // Scalar case: no dimensions, no strides
            if (len == 0) return strides;

            if (std.mem.indexOfScalar(usize, self.dims[0..], 0)) |_| {
                // If any dimension is zero, all strides are zero
                return strides;
            }

            switch (self.order) {
                .C => {
                    // Shape (a, b, c) => Give strides (b * c, c, 1)
                    var stride: isize = 1;
                    for (0..self.dims.len) |i_rev| {
                        // NOTE: self.dims is not empty, so this index is safe
                        const i = self.dims.len - 1 - i_rev;
                        strides[i] = stride;
                        // SAFETY: this cast and multiplication is overflow-safe because we have already verified that
                        // the total size does not overflow isize, which means the individual
                        // dimensions and strides must also fit in isize.
                        const dim = self.dims[i];
                        stride *= @intCast(dim);
                    }
                },
                .F => {
                    // Shape (a, b, c) => Give strides (1, a, a * b)
                    var stride: isize = 1;
                    for (self.dims, 0..) |dim, i| {
                        strides[i] = stride;
                        // SAFETY: this cast and multiplication is overflow-safe because we have already verified that
                        // the total size does not overflow isize, which means the individual
                        // dimensions and strides must also fit in isize.
                        stride *= @intCast(dim);
                    }
                },
            }

            return strides;
        }
    };
}

fn StaticShape(comptime Rank: usize) type {
    return Shape(Rank);
}

const DynamicShape = Shape(null);

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
    const shape, _ = try DynamicShape.fromHeader(npy_header);
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
    const result = DynamicShape.fromHeader(npy_header);
    try std.testing.expectError(DynamicShape.FromHeaderError.ShapeSizeOverflow, result);
    _ = allocator;
}

test "Shape.getStrides - empty shape" {
    const allocator = std.testing.allocator;
    const shape_data = [_]usize{};
    const shape = DynamicShape{
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
    const shape, _ = try DynamicShape.fromHeader(npy_header);
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
    const shape, _ = try DynamicShape.fromHeader(npy_header);
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
    const shape = DynamicShape{
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
    const shape = DynamicShape{
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
    const shape = DynamicShape{
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
    const shape = DynamicShape{
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
    const shape = DynamicShape{
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
    const shape = DynamicShape{
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
    const shape = DynamicShape{
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
    const shape = DynamicShape{
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

test "StaticShape(0) - scalar shape" {
    const Shape0D = StaticShape(0);
    const shape = Shape0D{
        .dims = [_]usize{},
        .order = .C,
    };
    // Scalar shape has 1 element
    const strides = shape.getStrides({});
    try std.testing.expectEqual(@as(usize, 0), strides.len);
}

test "StaticShape(1) - 1D shape" {
    const Shape1D = StaticShape(1);
    const shape = Shape1D{
        .dims = [_]usize{10},
        .order = .C,
    };
    // Shape (10) has 10 elements
    const strides = shape.getStrides({});
    try std.testing.expectEqual(@as(usize, 1), strides.len);
    try std.testing.expectEqual(@as(isize, 1), strides[0]);
}

test "StaticShape(2) - 2D C order" {
    const Shape2D = StaticShape(2);
    const shape = Shape2D{
        .dims = [_]usize{ 5, 7 },
        .order = .C,
    };
    // Shape (5, 7) has 35 elements
    const strides = shape.getStrides({});
    try std.testing.expectEqual(@as(usize, 2), strides.len);
    // C order: strides are (7, 1)
    try std.testing.expectEqual(@as(isize, 7), strides[0]);
    try std.testing.expectEqual(@as(isize, 1), strides[1]);
}

test "StaticShape(2) - 2D F order" {
    const Shape2D = StaticShape(2);
    const shape = Shape2D{
        .dims = [_]usize{ 5, 7 },
        .order = .F,
    };
    // Shape (5, 7) has 35 elements
    const strides = shape.getStrides({});
    try std.testing.expectEqual(@as(usize, 2), strides.len);
    // F order: strides are (1, 5)
    try std.testing.expectEqual(@as(isize, 1), strides[0]);
    try std.testing.expectEqual(@as(isize, 5), strides[1]);
}

test "StaticShape(3) - 3D C order" {
    const Shape3D = StaticShape(3);
    const shape = Shape3D{
        .dims = [_]usize{ 2, 3, 4 },
        .order = .C,
    };
    // Shape (2, 3, 4) has 24 elements
    const strides = shape.getStrides({});
    try std.testing.expectEqual(@as(usize, 3), strides.len);
    // C order: strides are (12, 4, 1)
    try std.testing.expectEqual(@as(isize, 12), strides[0]);
    try std.testing.expectEqual(@as(isize, 4), strides[1]);
    try std.testing.expectEqual(@as(isize, 1), strides[2]);
}

test "StaticShape(3) - 3D F order" {
    const Shape3D = StaticShape(3);
    const shape = Shape3D{
        .dims = [_]usize{ 2, 3, 4 },
        .order = .F,
    };
    // Shape (2, 3, 4) has 24 elements
    const strides = shape.getStrides({});
    try std.testing.expectEqual(@as(usize, 3), strides.len);
    // F order: strides are (1, 2, 6)
    try std.testing.expectEqual(@as(isize, 1), strides[0]);
    try std.testing.expectEqual(@as(isize, 2), strides[1]);
    try std.testing.expectEqual(@as(isize, 6), strides[2]);
}

test "StaticShape(3) - shape with zero dimension" {
    const Shape3D = StaticShape(3);
    const shape = Shape3D{
        .dims = [_]usize{ 3, 0, 5 },
        .order = .C,
    };
    // Shape (3, 0, 5) has 0 elements (zero dimension)
    const strides = shape.getStrides({});
    try std.testing.expectEqual(@as(usize, 3), strides.len);
    // All strides should be zero
    try std.testing.expectEqual(@as(isize, 0), strides[0]);
    try std.testing.expectEqual(@as(isize, 0), strides[1]);
    try std.testing.expectEqual(@as(isize, 0), strides[2]);
}

test "StaticShape(4) - 4D C order" {
    const Shape4D = StaticShape(4);
    const shape = Shape4D{
        .dims = [_]usize{ 2, 3, 4, 5 },
        .order = .C,
    };
    // Shape (2, 3, 4, 5) has 120 elements
    const strides = shape.getStrides({});
    // C order: strides are (60, 20, 5, 1)
    try std.testing.expectEqual(@as(isize, 60), strides[0]);
    try std.testing.expectEqual(@as(isize, 20), strides[1]);
    try std.testing.expectEqual(@as(isize, 5), strides[2]);
    try std.testing.expectEqual(@as(isize, 1), strides[3]);
}

test "StaticShape(4) - 4D F order" {
    const Shape4D = StaticShape(4);
    const shape = Shape4D{
        .dims = [_]usize{ 2, 3, 4, 5 },
        .order = .F,
    };
    // Shape (2, 3, 4, 5) has 120 elements
    const strides = shape.getStrides({});
    // F order: strides are (1, 2, 6, 24)
    try std.testing.expectEqual(@as(isize, 1), strides[0]);
    try std.testing.expectEqual(@as(isize, 2), strides[1]);
    try std.testing.expectEqual(@as(isize, 6), strides[2]);
    try std.testing.expectEqual(@as(isize, 24), strides[3]);
}

test "StaticShape.fromHeader - valid 2D" {
    const Shape2D = StaticShape(2);
    var shape_data = [_]usize{ 3, 4 };
    const npy_header = header.Header{
        .shape = &shape_data,
        .descr = .{ .Float64 = null },
        .order = .C,
    };
    const shape, const num_elements = try Shape2D.fromHeader(npy_header);
    try std.testing.expectEqual(@as(usize, 12), num_elements);
    try std.testing.expectEqual(@as(usize, 3), shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 4), shape.dims[1]);
    try std.testing.expectEqual(header.Order.C, shape.order);
}

test "StaticShape.fromHeader - dimension mismatch" {
    const Shape2D = StaticShape(2);
    var shape_data = [_]usize{ 3, 4, 5 };
    const npy_header = header.Header{
        .shape = &shape_data,
        .descr = .{ .Float64 = null },
        .order = .C,
    };
    const result = Shape2D.fromHeader(npy_header);
    try std.testing.expectError(error.DimensionMismatch, result);
}

test "StaticShape.fromHeader - overflow error" {
    const Shape2D = StaticShape(2);
    var shape_data = [_]usize{ std.math.maxInt(usize), 2 };
    const npy_header = header.Header{
        .shape = &shape_data,
        .descr = .{ .Float64 = null },
        .order = .C,
    };
    const result = Shape2D.fromHeader(npy_header);
    try std.testing.expectError(error.ShapeSizeOverflow, result);
}
