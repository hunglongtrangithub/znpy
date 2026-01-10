const std = @import("std");

const header_mod = @import("../header.zig");
const shape_mod = @import("../shape.zig");

pub fn StaticShape(comptime rank: usize) type {
    // TODO: consider adding a check to reject ranks that are too large?
    return struct {
        /// The size of each dimension.
        /// The total number of elements is the product of all dimensions,
        /// which must not overflow `std.math.maxInt(isize)`.
        dims: [rank]usize,
        /// The strides for indexing into the array
        strides: [rank]isize,
        /// The total number of elements in the array
        num_elements: usize,
        /// The memory order of the array.
        order: header_mod.Order,

        const Self = @This();

        pub const FromHeaderError = error{
            ShapeSizeOverflow,
            DimensionMismatch,
        };

        pub const InitError = error{ShapeSizeOverflow};

        /// Initialize a `StaticShape` instance, with shape size overflow check and strides computation.
        pub fn init(
            dims: [rank]usize,
            order: header_mod.Order,
            descr: header_mod.ElementType,
        ) InitError!Self {
            // Check that the shape length fits in isize
            const num_elements = shape_mod.shapeSizeChecked(descr, dims[0..]) orelse {
                return InitError.ShapeSizeOverflow;
            };
            const strides = computeStrides(dims, order);
            return Self{
                .dims = dims,
                .order = order,
                .strides = strides,
                .num_elements = num_elements,
            };
        }

        /// Create a `StaticShape` from a numpy header.
        /// Returns an error if the shape's total size in bytes overflows isize,
        /// or if the number of dimensions does not match the expected rank.
        pub fn fromHeader(npy_header: header_mod.Header) FromHeaderError!Self {
            // Extract shape, checking rank if static
            const dims = blk: {
                if (npy_header.shape.len != rank) {
                    return error.DimensionMismatch;
                }
                var static_dims: [rank]usize = undefined;
                @memcpy(&static_dims, npy_header.shape[0..rank]);
                break :blk static_dims;
            };

            // Check that the shape length fits in isize
            const num_elements = shape_mod.shapeSizeChecked(npy_header.descr, dims[0..]) orelse {
                return error.ShapeSizeOverflow;
            };
            const strides = computeStrides(dims, npy_header.order);
            return Self{
                .dims = dims,
                .order = npy_header.order,
                .strides = strides,
                .num_elements = num_elements,
            };
        }

        /// Compute the strides for a given shape and order.
        /// Strides array has the same length as the number of dimensions in the shape.
        ///
        /// **Invariant**: For any valid multi-dimensional index (where `0 <= indices[i] < dims[i]` for all `i`),
        /// the offset calculation `Σ(indices[i] * strides[i])` is guaranteed to:
        /// 1. Be less than the total number of elements in the shape
        /// 2. Fit in isize without overflow
        fn computeStrides(dims: [rank]usize, order: header_mod.Order) [rank]isize {
            var strides = [_]isize{0} ** rank;

            // Scalar case: no dimensions, no strides
            if (rank == 0) return strides;

            // If any dimension is zero, all strides are zero
            inline for (dims) |dim| {
                if (dim == 0) {
                    return strides;
                }
            }

            switch (order) {
                .C => {
                    // Shape (a, b, c) => Give strides (b * c, c, 1)
                    var stride: isize = 1;
                    inline for (0..dims.len) |i_rev| {
                        // NOTE: dims is not empty, so this index is safe
                        const i = dims.len - 1 - i_rev;
                        strides[i] = stride;
                        // SAFETY: this cast and multiplication is overflow-safe because we have already verified that
                        // the total size does not overflow isize, which means the individual
                        // dimensions and strides must also fit in isize.
                        const dim = dims[i];
                        stride *= @intCast(dim);
                    }
                },
                .F => {
                    // Shape (a, b, c) => Give strides (1, a, a * b)
                    var stride: isize = 1;
                    inline for (dims, 0..) |dim, i| {
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

test "StaticShape(0) - scalar shape" {
    const Shape0D = StaticShape(0);
    const shape = try Shape0D.init([_]usize{}, .C, .{ .Float64 = null });
    // Scalar shape has 1 element
    try std.testing.expectEqual(@as(usize, 1), shape.num_elements);
    const strides = shape.strides;
    try std.testing.expectEqual(@as(usize, 0), strides.len);
}

test "StaticShape(1) - 1D shape" {
    const Shape1D = StaticShape(1);
    const shape = try Shape1D.init([_]usize{10}, .C, .{ .Float64 = null });
    // Shape (10) has 10 elements
    try std.testing.expectEqual(@as(usize, 10), shape.num_elements);
    const strides = shape.strides;
    try std.testing.expectEqual(@as(usize, 1), strides.len);
    try std.testing.expectEqual(@as(isize, 1), strides[0]);
}

test "StaticShape(2) - 2D C order" {
    const Shape2D = StaticShape(2);
    const shape = try Shape2D.init([_]usize{ 5, 7 }, .C, .{ .Float64 = null });
    // Shape (5, 7) has 35 elements
    try std.testing.expectEqual(@as(usize, 35), shape.num_elements);
    const strides = shape.strides;
    try std.testing.expectEqual(@as(usize, 2), strides.len);
    // C order: strides are (7, 1)
    try std.testing.expectEqual(@as(isize, 7), strides[0]);
    try std.testing.expectEqual(@as(isize, 1), strides[1]);
}

test "StaticShape(2) - 2D F order" {
    const Shape2D = StaticShape(2);
    const shape = try Shape2D.init([_]usize{ 5, 7 }, .F, .{ .Float64 = null });
    // Shape (5, 7) has 35 elements
    try std.testing.expectEqual(@as(usize, 35), shape.num_elements);
    const strides = shape.strides;
    try std.testing.expectEqual(@as(usize, 2), strides.len);
    // F order: strides are (1, 5)
    try std.testing.expectEqual(@as(isize, 1), strides[0]);
    try std.testing.expectEqual(@as(isize, 5), strides[1]);
}

test "StaticShape(3) - 3D C order" {
    const Shape3D = StaticShape(3);
    const shape = try Shape3D.init([_]usize{ 2, 3, 4 }, .C, .{ .Float64 = null });
    // Shape (2, 3, 4) has 24 elements
    try std.testing.expectEqual(@as(usize, 24), shape.num_elements);
    const strides = shape.strides;
    try std.testing.expectEqual(@as(usize, 3), strides.len);
    // C order: strides are (12, 4, 1)
    try std.testing.expectEqual(@as(isize, 12), strides[0]);
    try std.testing.expectEqual(@as(isize, 4), strides[1]);
    try std.testing.expectEqual(@as(isize, 1), strides[2]);
}

test "StaticShape(3) - 3D F order" {
    const Shape3D = StaticShape(3);
    const shape = try Shape3D.init([_]usize{ 2, 3, 4 }, .F, .{ .Float64 = null });
    // Shape (2, 3, 4) has 24 elements
    try std.testing.expectEqual(@as(usize, 24), shape.num_elements);
    const strides = shape.strides;
    try std.testing.expectEqual(@as(usize, 3), strides.len);
    // F order: strides are (1, 2, 6)
    try std.testing.expectEqual(@as(isize, 1), strides[0]);
    try std.testing.expectEqual(@as(isize, 2), strides[1]);
    try std.testing.expectEqual(@as(isize, 6), strides[2]);
}

test "StaticShape(3) - shape with zero dimension" {
    const Shape3D = StaticShape(3);
    const shape = try Shape3D.init([_]usize{ 3, 0, 5 }, .C, .{ .Float64 = null });
    // Shape (3, 0, 5) has 0 elements (zero dimension)
    try std.testing.expectEqual(@as(usize, 0), shape.num_elements);
    const strides = shape.strides;
    try std.testing.expectEqual(@as(usize, 3), strides.len);
    // All strides should be zero
    try std.testing.expectEqual(@as(isize, 0), strides[0]);
    try std.testing.expectEqual(@as(isize, 0), strides[1]);
    try std.testing.expectEqual(@as(isize, 0), strides[2]);
}

test "StaticShape(4) - 4D C order" {
    const Shape4D = StaticShape(4);
    const shape = try Shape4D.init([_]usize{ 2, 3, 4, 5 }, .C, .{ .Float64 = null });
    // Shape (2, 3, 4, 5) has 120 elements
    try std.testing.expectEqual(@as(usize, 120), shape.num_elements);
    const strides = shape.strides;
    // C order: strides are (60, 20, 5, 1)
    try std.testing.expectEqual(@as(isize, 60), strides[0]);
    try std.testing.expectEqual(@as(isize, 20), strides[1]);
    try std.testing.expectEqual(@as(isize, 5), strides[2]);
    try std.testing.expectEqual(@as(isize, 1), strides[3]);
}

test "StaticShape(4) - 4D F order" {
    const Shape4D = StaticShape(4);
    const shape = try Shape4D.init([_]usize{ 2, 3, 4, 5 }, .F, .{ .Float64 = null });
    // Shape (2, 3, 4, 5) has 120 elements
    try std.testing.expectEqual(@as(usize, 120), shape.num_elements);
    const strides = shape.strides;
    // F order: strides are (1, 2, 6, 24)
    try std.testing.expectEqual(@as(isize, 1), strides[0]);
    try std.testing.expectEqual(@as(isize, 2), strides[1]);
    try std.testing.expectEqual(@as(isize, 6), strides[2]);
    try std.testing.expectEqual(@as(isize, 24), strides[3]);
}

test "StaticShape.fromHeader - valid 2D" {
    const Shape2D = StaticShape(2);
    var shape_data = [_]usize{ 3, 4 };
    const npy_header = header_mod.Header{
        .shape = &shape_data,
        .descr = .{ .Float64 = null },
        .order = .C,
    };
    const shape = try Shape2D.fromHeader(npy_header);
    try std.testing.expectEqual(@as(usize, 12), shape.num_elements);
    try std.testing.expectEqual(@as(usize, 3), shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 4), shape.dims[1]);
    try std.testing.expectEqual(header_mod.Order.C, shape.order);
}

test "StaticShape.fromHeader - dimension mismatch" {
    const Shape2D = StaticShape(2);
    var shape_data = [_]usize{ 3, 4, 5 };
    const npy_header = header_mod.Header{
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
    const npy_header = header_mod.Header{
        .shape = &shape_data,
        .descr = .{ .Float64 = null },
        .order = .C,
    };
    const result = Shape2D.fromHeader(npy_header);
    try std.testing.expectError(error.ShapeSizeOverflow, result);
}
