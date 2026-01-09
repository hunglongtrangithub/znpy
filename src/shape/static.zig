const std = @import("std");

const header_mod = @import("../header.zig");
const shape_mod = @import("../shape.zig");

pub fn StaticShape(comptime rank: usize) type {
    return struct {
        /// The size of each dimension.
        /// The total number of elements is the product of all dimensions,
        /// which must not overflow `std.math.maxInt(isize)`.
        dims: [rank]usize,
        /// The memory order of the array.
        order: header_mod.Order,

        const Self = @This();

        pub const FromHeaderError = error{
            ShapeSizeOverflow,
            DimensionMismatch,
        };

        pub const InitError = error{
            ShapeSizeOverflow,
        };

        pub fn init(dims: [rank]usize, order: header_mod.Order, descr: header_mod.ElementType) InitError!struct { Self, usize } {
            // Check that the shape length fits in isize
            const num_elements = shape_mod.shapeSizeChecked(descr, dims[0..]) orelse {
                return InitError.ShapeSizeOverflow;
            };
            return .{ Self{
                .dims = dims,
                .order = order,
            }, num_elements };
        }

        /// Create a `StaticShape` from a numpy header.
        /// Returns an error if the shape's total size in bytes overflows isize,
        /// or if the number of dimensions does not match the expected rank.
        /// On success, also returns the total number of elements.
        pub fn fromHeader(npy_header: header_mod.Header) FromHeaderError!struct { Self, usize } {
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
            return .{ Self{
                .dims = dims,
                .order = npy_header.order,
            }, num_elements };
        }

        /// Compute the strides for this shape.
        /// Strides slice has the same length as the number of dimensions in the shape.
        ///
        /// **Invariant**: For any valid multi-dimensional index (where `0 <= indices[i] < dims[i]` for all `i`),
        /// the offset calculation `Σ(indices[i] * strides[i])` is guaranteed to:
        /// 1. Be less than the total number of elements in the shape
        /// 2. Fit in isize without overflow
        pub fn getStrides(self: *const Self) [rank]isize {
            var strides = [_]isize{0} ** rank;

            // Scalar case: no dimensions, no strides
            if (rank == 0) return strides;

            // If any dimension is zero, all strides are zero
            inline for (self.dims) |dim| {
                if (dim == 0) {
                    return strides;
                }
            }

            switch (self.order) {
                .C => {
                    // Shape (a, b, c) => Give strides (b * c, c, 1)
                    var stride: isize = 1;
                    inline for (0..self.dims.len) |i_rev| {
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
                    inline for (self.dims, 0..) |dim, i| {
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
    const shape = Shape0D{
        .dims = [_]usize{},
        .order = .C,
    };
    // Scalar shape has 1 element
    const strides = shape.getStrides();
    try std.testing.expectEqual(@as(usize, 0), strides.len);
}

test "StaticShape(1) - 1D shape" {
    const Shape1D = StaticShape(1);
    const shape = Shape1D{
        .dims = [_]usize{10},
        .order = .C,
    };
    // Shape (10) has 10 elements
    const strides = shape.getStrides();
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
    const strides = shape.getStrides();
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
    const strides = shape.getStrides();
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
    const strides = shape.getStrides();
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
    const strides = shape.getStrides();
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
    const strides = shape.getStrides();
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
    const strides = shape.getStrides();
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
    const strides = shape.getStrides();
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
    const shape, const num_elements = try Shape2D.fromHeader(npy_header);
    try std.testing.expectEqual(@as(usize, 12), num_elements);
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
