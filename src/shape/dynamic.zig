const std = @import("std");

const header_mod = @import("../header.zig");
const shape_mod = @import("../shape.zig");
const elements_mod = @import("../elements.zig");

pub const DynamicShape = struct {
    /// The size of each dimension. Slice is owned by the caller.
    /// The total number of elements is the product of all dimensions,
    /// which must not overflow `std.math.maxInt(isize)`.
    dims: []const usize,
    /// The strides for indexing into the array. Allocated with samelength as dims.
    strides: []const isize,
    /// The total number of elements in the array
    num_elements: usize,
    /// The memory order of the array.
    order: shape_mod.Order,

    const Self = @This();

    pub const Error = error{ShapeSizeOverflow} || std.mem.Allocator.Error;

    /// Initialize a `DynamicShape` instance, with shape size overflow check and strides computation.
    pub fn init(
        dims: []const usize,
        order: shape_mod.Order,
        descr: elements_mod.ElementType,
        allocator: std.mem.Allocator,
    ) Error!Self {
        // Check that the shape length fits in isize
        const num_elements = shape_mod.shapeSizeChecked(descr, dims[0..]) orelse {
            return Error.ShapeSizeOverflow;
        };
        const strides = try computeStrides(dims, order, allocator);

        std.debug.assert(strides.len == dims.len);

        return Self{
            .dims = dims,
            .order = order,
            .strides = strides,
            .num_elements = num_elements,
        };
    }

    /// Create a `DynamicShape` from a numpy header.
    /// Returns an error if the shape's total size in bytes overflows isize.
    /// Allocates memory for strides using the provided allocator.
    pub fn fromHeader(npy_header: header_mod.Header, allocator: std.mem.Allocator) Error!Self {
        // Extract shape
        const dims = npy_header.shape;

        // Check that the shape length fits in isize
        const num_elements = shape_mod.shapeSizeChecked(npy_header.descr, dims[0..]) orelse {
            return error.ShapeSizeOverflow;
        };
        const strides = try computeStrides(dims, npy_header.order, allocator);
        return Self{
            .dims = dims,
            .order = npy_header.order,
            .strides = strides,
            .num_elements = num_elements,
        };
    }

    /// Free the memory allocated for the strides.
    pub fn deinit(self: *const Self, allocator: std.mem.Allocator) void {
        allocator.free(self.strides);
    }

    /// Compute the strides for a given shape and order, allocating memory from the given allocator.
    /// Strides slice has the same length as the number of dimensions in the shape.
    ///
    /// **Invariant**: For any valid multi-dimensional index (where `0 <= indices[i] < dims[i]` for all `i`),
    /// the offset calculation `Σ(indices[i] * strides[i])` is guaranteed to:
    /// 1. Be less than the total number of elements in the shape
    /// 2. Fit in isize without overflow
    fn computeStrides(
        dims: []const usize,
        order: shape_mod.Order,
        allocator: std.mem.Allocator,
    ) std.mem.Allocator.Error![]isize {
        // Scalar case: no dimensions, no strides - return empty slice owned by allocator
        if (dims.len == 0) return try allocator.alloc(isize, 0);

        var strides = try allocator.alloc(isize, dims.len);

        // If any dimension is zero, all strides are zero
        if (std.mem.indexOfScalar(usize, dims, 0) != null) {
            @memset(strides, 0);
            return strides;
        }

        switch (order) {
            .C => {
                // Shape (a, b, c) => Give strides (b * c, c, 1)
                var stride: isize = 1;
                for (0..dims.len) |i_rev| {
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
                for (dims, 0..) |dim, i| {
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

test "DynamicShape.fromHeader - valid shape" {
    const allocator = std.testing.allocator;
    var shape_data = [_]usize{ 2, 3, 4 };
    const npy_header = header_mod.Header{
        .shape = &shape_data,
        .descr = .{ .Float64 = null },
        .order = .C,
    };
    const shape = try DynamicShape.fromHeader(npy_header, allocator);
    defer shape.deinit(allocator);

    try std.testing.expectEqualSlices(
        usize,
        &shape_data,
        shape.dims,
    );
    try std.testing.expectEqual(24, shape.num_elements);
    try std.testing.expectEqual(shape_mod.Order.C, shape.order);
}

test "DynamicShape.fromHeader - overflow error" {
    const allocator = std.testing.allocator;
    var shape_data = [_]usize{ std.math.maxInt(usize), 2 };
    const npy_header = header_mod.Header{
        .shape = &shape_data,
        .descr = .{ .Float64 = null },
        .order = .C,
    };
    const result = DynamicShape.fromHeader(npy_header, allocator);
    try std.testing.expectError(DynamicShape.Error.ShapeSizeOverflow, result);
}

test "DynamicShape.strides - empty shape" {
    const allocator = std.testing.allocator;
    const shape_data = [_]usize{};
    const shape = try DynamicShape.init(
        &shape_data,
        .C,
        .{ .Float64 = null },
        allocator,
    );
    defer shape.deinit(allocator);
    try std.testing.expectEqual(0, shape.strides.len);
}

test "DynamicShape.strides - shape with zero dimension" {
    const allocator = std.testing.allocator;
    var shape_data = [_]usize{ 3, 0, 5 };
    const npy_header = header_mod.Header{
        .shape = &shape_data,
        .descr = .{ .Float64 = null },
        .order = .C,
    };
    const shape = try DynamicShape.fromHeader(npy_header, allocator);
    defer shape.deinit(allocator);
    const strides = shape.strides;
    try std.testing.expectEqualSlices(
        isize,
        &[_]isize{ 0, 0, 0 },
        strides,
    );
}

test "DynamicShape.strides - C order (2, 3, 4)" {
    const allocator = std.testing.allocator;
    const shape_data = [_]usize{ 2, 3, 4 };
    const shape = try DynamicShape.init(&shape_data, .C, .{ .Float64 = null }, allocator);
    defer shape.deinit(allocator);
    const strides = shape.strides;
    // C order: strides are (3*4, 4, 1) = (12, 4, 1)
    try std.testing.expectEqualSlices(
        isize,
        &[_]isize{ 12, 4, 1 },
        strides,
    );
}

test "DynamicShape.strides - F order (2, 3, 4)" {
    const allocator = std.testing.allocator;
    const shape_data = [_]usize{ 2, 3, 4 };
    const shape = try DynamicShape.init(&shape_data, .F, .{ .Float64 = null }, allocator);
    defer shape.deinit(allocator);
    const strides = shape.strides;
    // F order: strides are (1, 2, 2*3) = (1, 2, 6)
    try std.testing.expectEqualSlices(
        isize,
        &[_]isize{ 1, 2, 6 },
        strides,
    );
}

test "DynamicShape.strides - C order 1D array" {
    const allocator = std.testing.allocator;
    const shape_data = [_]usize{10};
    const shape = try DynamicShape.init(&shape_data, .C, .{ .Float64 = null }, allocator);
    defer shape.deinit(allocator);
    const strides = shape.strides;
    try std.testing.expectEqualSlices(
        isize,
        &[_]isize{1},
        strides,
    );
}

test "DynamicShape.strides - F order 1D array" {
    const allocator = std.testing.allocator;
    const shape_data = [_]usize{10};
    const shape = try DynamicShape.init(&shape_data, .F, .{ .Float64 = null }, allocator);
    defer shape.deinit(allocator);
    const strides = shape.strides;
    try std.testing.expectEqualSlices(
        isize,
        &[_]isize{1},
        strides,
    );
}

test "DynamicShape.strides - C order 2D array" {
    const allocator = std.testing.allocator;
    const shape_data = [_]usize{ 5, 7 };
    const shape = try DynamicShape.init(&shape_data, .C, .{ .Float64 = null }, allocator);
    defer shape.deinit(allocator);
    const strides = shape.strides;
    // C order: strides are (7, 1)
    try std.testing.expectEqualSlices(
        isize,
        &[_]isize{ 7, 1 },
        strides,
    );
}

test "DynamicShape.strides - F order 2D array" {
    const allocator = std.testing.allocator;
    const shape_data = [_]usize{ 5, 7 };
    const shape = try DynamicShape.init(&shape_data, .F, .{ .Float64 = null }, allocator);
    defer shape.deinit(allocator);
    const strides = shape.strides;
    // F order: strides are (1, 5)
    try std.testing.expectEqualSlices(
        isize,
        &[_]isize{ 1, 5 },
        strides,
    );
}

test "DynamicShape.strides - C order 4D array" {
    const allocator = std.testing.allocator;
    const shape_data = [_]usize{ 2, 3, 4, 5 };
    const shape = try DynamicShape.init(&shape_data, .C, .{ .Float64 = null }, allocator);
    defer shape.deinit(allocator);
    const strides = shape.strides;
    // C order: strides are (3*4*5, 4*5, 5, 1) = (60, 20, 5, 1)
    try std.testing.expectEqualSlices(
        isize,
        &[_]isize{ 60, 20, 5, 1 },
        strides,
    );
}

test "DynamicShape.strides - F order 4D array" {
    const allocator = std.testing.allocator;
    const shape_data = [_]usize{ 2, 3, 4, 5 };
    const shape = try DynamicShape.init(&shape_data, .F, .{ .Float64 = null }, allocator);
    defer shape.deinit(allocator);
    const strides = shape.strides;
    // F order: strides are (1, 2, 2*3, 2*3*4) = (1, 2, 6, 24)
    try std.testing.expectEqualSlices(
        isize,
        &[_]isize{ 1, 2, 6, 24 },
        strides,
    );
}
