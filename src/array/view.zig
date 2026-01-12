const std = @import("std");

const array_mod = @import("../array.zig");
const slice_mod = @import("../slice.zig");

/// Compute the flat array offset for a given multi-dimensional index.
/// Returns:
///   - The computed offset as an `isize` if the index is valid
///   - `null` if the index is invalid (wrong number of dimensions or out of bounds)
fn strideOffset(dims: []const usize, strides: []const isize, index: []const usize) ?isize {
    // Dimension mismatch
    if (index.len != dims.len) return null;
    std.debug.assert(strides.len == dims.len);

    var offset: isize = 0;

    for (index, dims, strides) |idx, dim, stride| {
        if (idx >= dim) {
            // Index out of bounds
            return null;
        }
        // SAFETY: This cast is safe due to the bounds check above (dim fits in isize and idx < dim)
        const idx_isize: isize = @intCast(idx);
        offset += idx_isize * stride;
    }

    return offset;
}

/// Compute the flat array offset for a given multi-dimensional index without bounds checking.
///
/// SAFETY: The caller MUST ensure that all indices are within bounds.
/// Undefined behavior if any `index[i] >= dims[i]` or if `index.len != dims.len`.
///
/// When a dimension size is 0, any index for that dimension is out of bounds,
/// so this function must never be called in that case.
fn strideOffsetUnchecked(strides: []const isize, index: []const usize) isize {
    var offset: isize = 0;
    for (index, strides) |idx, stride| {
        const idx_isize: isize = @intCast(idx);
        offset += idx_isize * stride;
    }
    return offset;
}

test "strideOffset - valid index" {
    const dims = [_]usize{ 2, 3, 4 };
    const strides = [_]isize{ 12, 4, 1 }; // C-order strides

    const index = [_]usize{ 1, 2, 3 };
    const offset = strideOffset(&dims, &strides, &index);
    try std.testing.expectEqual(23, offset.?);
}

test "strideOffset - out of bounds index" {
    const dims = [_]usize{ 2, 3, 4 };
    const strides = [_]isize{ 12, 4, 1 };

    const index = [_]usize{ 2, 0, 0 }; // Out of bounds in first dimension
    const offset = strideOffset(&dims, &strides, &index);
    try std.testing.expectEqual(null, offset);
}

test "strideOffset - empty shape" {
    const dims = [_]usize{};
    const strides = [_]isize{};

    const index = [_]usize{};
    const offset = strideOffset(&dims, &strides, &index);
    try std.testing.expectEqual(0, offset.?);
}

test "strideOffset - empty array" {
    const dims = [_]usize{ 0, 3 };
    const strides = [_]isize{ 3, 1 };

    const index = [_]usize{ 0, 0 };
    const offset = strideOffset(&dims, &strides, &index);
    try std.testing.expectEqual(null, offset);
}

/// A mutable view into a multi-dimensional array.
/// The view does NOT own the underlying data buffer or shape metadata.
/// You can read and write elements through this view.
///
/// `T` is the element type.
pub fn ArrayView(comptime T: type) type {
    return struct {
        /// The dimensions of the array
        dims: []const usize,
        /// The strides for each dimension (in elements, not bytes)
        /// Should always have the same length as `dims`
        strides: []const isize,
        /// Pointer to "Logical Index 0" of the array view
        data_ptr: [*]T,

        const Self = @This();

        /// Get a pointer to the element at the given multi-dimensional index.
        ///
        /// Returns:
        /// - A pointer to the element if the index is valid
        /// - `null` if the index is out of bounds
        ///
        /// SAFETY:
        /// - The returned pointer is only valid as long as the underlying data exists
        /// - Do not store the pointer beyond the data's lifetime
        ///
        /// For safe value access, prefer `get()` and `set()` methods.
        pub fn at(self: *const Self, index: []const usize) ?*T {
            const offset = strideOffset(
                self.dims,
                self.strides,
                index,
            ) orelse return null;
            return array_mod.ptrFromOffset(T, self.data_ptr, offset);
        }

        /// Get a pointer to the element at the given multi-dimensional index without bounds checking.
        ///
        /// SAFETY: The caller MUST ensure that all indices are within bounds.
        /// Undefined behavior if any `index[i] >= dims[i]` or if `index.len != dims.len`.
        ///
        /// This function skips all bounds checking for maximum performance.
        /// Use only when you have already validated the indices.
        pub fn atUnchecked(self: *const Self, index: []const usize) *T {
            const offset = strideOffsetUnchecked(self.strides, index);
            return array_mod.ptrFromOffset(T, self.data_ptr, offset);
        }

        /// Get the element at the given multi-dimensional index.
        /// Returns null if the index is out of bounds.
        pub fn get(self: *const Self, index: []const usize) ?T {
            const ptr = self.at(index) orelse return null;
            return ptr.*;
        }

        /// Set the element at the given multi-dimensional index.
        /// Panics if the index is out of bounds.
        pub fn set(self: *const Self, index: []const usize, value: T) void {
            const ptr = self.at(index).?;
            ptr.* = value;
        }

        /// Create a sliced view from this view.
        /// The returned view has the same mutability as the original.
        ///
        /// The caller owns the returned view's dims and strides arrays.
        pub fn slice(
            self: *const Self,
            slices: []const slice_mod.Slice,
            allocator: std.mem.Allocator,
        ) (slice_mod.SliceError || std.mem.Allocator.Error)!Self {
            const new_dims, const new_strides, const offset = try slice_mod.applySlices(
                self.dims,
                self.strides,
                slices,
                allocator,
            );

            // Calculate new data pointer
            const new_data_ptr_single = array_mod.ptrFromOffset(
                T,
                self.data_ptr,
                offset,
            );
            const new_data_ptr: [*]T = @ptrCast(new_data_ptr_single);

            return Self{
                .dims = new_dims,
                .strides = new_strides,
                .data_ptr = new_data_ptr,
            };
        }

        pub fn deinit(self: Self, allocator: std.mem.Allocator) void {
            allocator.free(self.dims);
            allocator.free(self.strides);
        }
    };
}

test "ArrayView - standalone use" {
    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const dims = [_]usize{ 2, 3 };
    const strides = [_]isize{ 3, 1 }; // C-order strides

    const view = ArrayView(f32){
        .dims = &dims,
        .strides = &strides,
        .data_ptr = &data,
    };

    // Test get/set through view
    try std.testing.expectEqual(@as(?f32, 1.0), view.get(&[_]usize{ 0, 0 }));
    try std.testing.expectEqual(@as(?f32, 6.0), view.get(&[_]usize{ 1, 2 }));

    // Test mutation through view
    view.set(&[_]usize{ 1, 1 }, 42.0);
    try std.testing.expectEqual(@as(f32, 42.0), data[4]);

    // Test at() returns mutable pointer
    const ptr = view.at(&[_]usize{ 0, 1 });
    try std.testing.expect(ptr != null);
    try std.testing.expectEqual(@as(f32, 2.0), ptr.?.*);

    // Modify through pointer
    ptr.?.* = 99.0;
    try std.testing.expectEqual(@as(f32, 99.0), data[1]);

    // Test bounds checking
    try std.testing.expectEqual(@as(?f32, null), view.get(&[_]usize{ 2, 0 }));
    try std.testing.expectEqual(@as(?f32, null), view.get(&[_]usize{ 0, 3 }));
}

test "ArrayView - slice with Index" {
    const allocator = std.testing.allocator;
    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const dims = [_]usize{ 2, 3 };
    const strides = [_]isize{ 3, 1 }; // C-order: [[1,2,3], [4,5,6]]

    const view = ArrayView(f32){
        .dims = &dims,
        .strides = &strides,
        .data_ptr = &data,
    };

    // Slice first row: view[0, :]
    const slices = [_]slice_mod.Slice{
        .{ .Index = 0 },
        .{ .Range = .{} },
    };
    const sliced = try view.slice(&slices, allocator);
    defer sliced.deinit(allocator);

    try std.testing.expectEqual(1, sliced.dims.len);
    try std.testing.expectEqual(3, sliced.dims[0]);
    try std.testing.expectEqual(@as(?f32, 1.0), sliced.get(&[_]usize{0}));
    try std.testing.expectEqual(@as(?f32, 2.0), sliced.get(&[_]usize{1}));
    try std.testing.expectEqual(@as(?f32, 3.0), sliced.get(&[_]usize{2}));
}

test "ArrayView - slice with Range and step" {
    const allocator = std.testing.allocator;
    var data = [_]f32{ 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 };
    const dims = [_]usize{6};
    const strides = [_]isize{1};

    const view = ArrayView(f32){
        .dims = &dims,
        .strides = &strides,
        .data_ptr = &data,
    };

    // Slice with step: view[1:5:2] -> [1.0, 3.0]
    const slices = [_]slice_mod.Slice{
        .{ .Range = .{ .start = 1, .end = 5, .step = 2 } },
    };
    const sliced = try view.slice(&slices, allocator);
    defer sliced.deinit(allocator);

    try std.testing.expectEqual(1, sliced.dims.len);
    try std.testing.expectEqual(2, sliced.dims[0]);
    try std.testing.expectEqual(@as(?f32, 1.0), sliced.get(&[_]usize{0}));
    try std.testing.expectEqual(@as(?f32, 3.0), sliced.get(&[_]usize{1}));
}

test "ArrayView - slice with negative step" {
    const allocator = std.testing.allocator;

    // Create a 1D array [0, 1, 2, 3]
    var data = [_]f32{ 0, 1, 2, 3 };
    var dims = [_]usize{4};
    var strides = [_]isize{1};

    const view = ArrayView(f32){
        .dims = &dims,
        .strides = &strides,
        .data_ptr = &data,
    };

    // Test 1: should give [3, 2]
    // With negative step, we go from end-1 backwards to start
    {
        const slices = [_]slice_mod.Slice{
            .{ .Range = .{ .start = 3, .end = 1, .step = -1 } },
        };
        const sliced = try view.slice(&slices, allocator);
        defer sliced.deinit(allocator);

        try std.testing.expectEqual(2, sliced.dims[0]);
        try std.testing.expectEqual(@as(?f32, 3.0), sliced.get(&.{0}));
        try std.testing.expectEqual(@as(?f32, 2.0), sliced.get(&.{1}));
    }

    // Test 2: should give [1]
    // Range from 1 to end (-3), step -2, starts at index 1
    {
        const slices = [_]slice_mod.Slice{
            .{ .Range = .{ .start = 1, .end = null, .step = -2 } },
        };
        const sliced = try view.slice(&slices, allocator);
        defer sliced.deinit(allocator);

        try std.testing.expectEqual(1, sliced.dims[0]);
        try std.testing.expectEqual(@as(?f32, 1.0), sliced.get(&.{0}));
    }

    // Test 3: should give [1, 3]
    {
        const slices = [_]slice_mod.Slice{
            .{ .Range = .{ .start = 4, .end = 0, .step = -2 } },
        };
        const sliced = try view.slice(&slices, allocator);
        defer sliced.deinit(allocator);

        try std.testing.expectEqual(2, sliced.dims[0]);
        try std.testing.expectEqual(@as(?f32, 1.0), sliced.get(&.{1}));
        try std.testing.expectEqual(@as(?f32, 3.0), sliced.get(&.{0}));
    }

    // Test 4: should give [3, 1]
    {
        const slices = [_]slice_mod.Slice{
            .{ .Range = .{ .start = null, .end = 0, .step = -2 } },
        };
        const sliced = try view.slice(&slices, allocator);
        defer sliced.deinit(allocator);

        try std.testing.expectEqual(2, sliced.dims[0]);
        try std.testing.expectEqual(@as(?f32, 3.0), sliced.get(&.{0}));
        try std.testing.expectEqual(@as(?f32, 1.0), sliced.get(&.{1}));
    }

    // Test 5: should give [3, 1]
    {
        const slices = [_]slice_mod.Slice{
            .{ .Range = .{ .start = null, .end = 0, .step = -2 } },
        };
        const sliced = try view.slice(&slices, allocator);
        defer sliced.deinit(allocator);

        try std.testing.expectEqual(2, sliced.dims[0]);
        try std.testing.expectEqual(@as(?f32, 3.0), sliced.get(&.{0}));
        try std.testing.expectEqual(@as(?f32, 1.0), sliced.get(&.{1}));
    }
}

test "ArrayView - slice with NewAxis" {
    const allocator = std.testing.allocator;
    var data = [_]f32{ 1.0, 2.0, 3.0 };
    const dims = [_]usize{3};
    const strides = [_]isize{1};

    const view = ArrayView(f32){
        .dims = &dims,
        .strides = &strides,
        .data_ptr = &data,
    };

    // Add new axis: view[newaxis, :] -> shape (1, 3)
    const slices = [_]slice_mod.Slice{
        .NewAxis,
        .{ .Range = .{} },
    };
    const sliced = try view.slice(&slices, allocator);
    defer sliced.deinit(allocator);

    try std.testing.expectEqual(2, sliced.dims.len);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 1, 3 }, sliced.dims);
    try std.testing.expectEqual(@as(?f32, 1.0), sliced.get(&[_]usize{ 0, 0 }));
    try std.testing.expectEqual(@as(?f32, 2.0), sliced.get(&[_]usize{ 0, 1 }));
    try std.testing.expectEqual(@as(?f32, 3.0), sliced.get(&[_]usize{ 0, 2 }));
}

test "ArrayView - slice with NewAxis, Range, and Index" {
    const allocator = std.testing.allocator;
    var data = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    const dims = [_]usize{ 2, 2, 3 };
    const strides = [_]isize{ 6, 3, 1 }; // C-order: [[[1,2,3], [4,5,6]], [[7,8,9], [10,11,12]]]
    const view = ArrayView(u32){
        .dims = &dims,
        .strides = &strides,
        .data_ptr = &data,
    };

    // Test 1: NewAxis at position 0 (front)
    // [NewAxis, Index=0, Range, Range]
    // Result: [[[1,2,3], [4,5,6]]] with shape [1, 2, 3]
    const sliced1 = try view.slice(
        &[_]slice_mod.Slice{
            .NewAxis,
            .{ .Index = 0 },
            .{ .Range = .{} },
            .{ .Range = .{} },
        },
        allocator,
    );
    defer sliced1.deinit(allocator);
    try std.testing.expectEqual(3, sliced1.dims.len);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 1, 2, 3 }, sliced1.dims);
    try std.testing.expectEqual(@as(?u32, 1), sliced1.get(&[_]usize{ 0, 0, 0 }));
    try std.testing.expectEqual(@as(?u32, 2), sliced1.get(&[_]usize{ 0, 0, 1 }));
    try std.testing.expectEqual(@as(?u32, 3), sliced1.get(&[_]usize{ 0, 0, 2 }));
    try std.testing.expectEqual(@as(?u32, 4), sliced1.get(&[_]usize{ 0, 1, 0 }));
    try std.testing.expectEqual(@as(?u32, 5), sliced1.get(&[_]usize{ 0, 1, 1 }));
    try std.testing.expectEqual(@as(?u32, 6), sliced1.get(&[_]usize{ 0, 1, 2 }));

    // Test 2: NewAxis at position 1 (middle)
    // [Index=0, NewAxis, Range, Range]
    // Result: [[[1,2,3], [4,5,6]]] with shape [1, 2, 3]
    const sliced2 = try view.slice(
        &[_]slice_mod.Slice{
            .{ .Index = 0 },
            .NewAxis,
            .{ .Range = .{} },
            .{ .Range = .{} },
        },
        allocator,
    );
    defer sliced2.deinit(allocator);
    try std.testing.expectEqual(3, sliced2.dims.len);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 1, 2, 3 }, sliced2.dims);
    try std.testing.expectEqual(@as(?u32, 1), sliced2.get(&[_]usize{ 0, 0, 0 }));
    try std.testing.expectEqual(@as(?u32, 2), sliced2.get(&[_]usize{ 0, 0, 1 }));
    try std.testing.expectEqual(@as(?u32, 3), sliced2.get(&[_]usize{ 0, 0, 2 }));
    try std.testing.expectEqual(@as(?u32, 4), sliced2.get(&[_]usize{ 0, 1, 0 }));
    try std.testing.expectEqual(@as(?u32, 5), sliced2.get(&[_]usize{ 0, 1, 1 }));
    try std.testing.expectEqual(@as(?u32, 6), sliced2.get(&[_]usize{ 0, 1, 2 }));

    // Test 3: NewAxis at position 2 (after first two dims)
    // [Index=0, Range, NewAxis, Range]
    // Result: [[[1,2,3]], [[4,5,6]]] with shape [2, 1, 3]
    const sliced3 = try view.slice(
        &[_]slice_mod.Slice{
            .{ .Index = 0 },
            .{ .Range = .{} },
            .NewAxis,
            .{ .Range = .{} },
        },
        allocator,
    );
    defer sliced3.deinit(allocator);
    try std.testing.expectEqual(3, sliced3.dims.len);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 2, 1, 3 }, sliced3.dims);
    try std.testing.expectEqual(@as(?u32, 1), sliced3.get(&[_]usize{ 0, 0, 0 }));
    try std.testing.expectEqual(@as(?u32, 2), sliced3.get(&[_]usize{ 0, 0, 1 }));
    try std.testing.expectEqual(@as(?u32, 3), sliced3.get(&[_]usize{ 0, 0, 2 }));
    try std.testing.expectEqual(@as(?u32, 4), sliced3.get(&[_]usize{ 1, 0, 0 }));
    try std.testing.expectEqual(@as(?u32, 5), sliced3.get(&[_]usize{ 1, 0, 1 }));
    try std.testing.expectEqual(@as(?u32, 6), sliced3.get(&[_]usize{ 1, 0, 2 }));

    // Test 4: NewAxis at position 3 (end)
    // [Index=0, Range, Range, NewAxis]
    // Result: [[[1],[2],[3]], [[4],[5],[6]]] with shape [2, 3, 1]
    const sliced4 = try view.slice(
        &[_]slice_mod.Slice{
            .{ .Index = 0 },
            .{ .Range = .{} },
            .{ .Range = .{} },
            .NewAxis,
        },
        allocator,
    );
    defer sliced4.deinit(allocator);
    try std.testing.expectEqual(3, sliced4.dims.len);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 2, 3, 1 }, sliced4.dims);
    try std.testing.expectEqual(@as(?u32, 1), sliced4.get(&[_]usize{ 0, 0, 0 }));
    try std.testing.expectEqual(@as(?u32, 2), sliced4.get(&[_]usize{ 0, 1, 0 }));
    try std.testing.expectEqual(@as(?u32, 3), sliced4.get(&[_]usize{ 0, 2, 0 }));
    try std.testing.expectEqual(@as(?u32, 4), sliced4.get(&[_]usize{ 1, 0, 0 }));
    try std.testing.expectEqual(@as(?u32, 5), sliced4.get(&[_]usize{ 1, 1, 0 }));
    try std.testing.expectEqual(@as(?u32, 6), sliced4.get(&[_]usize{ 1, 2, 0 }));
}

test "ArrayView - slice mutability preserved" {
    const allocator = std.testing.allocator;
    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const dims = [_]usize{ 2, 2 };
    const strides = [_]isize{ 2, 1 };

    const view = ArrayView(f32){
        .dims = &dims,
        .strides = &strides,
        .data_ptr = &data,
    };

    // Slice and mutate
    const slices = [_]slice_mod.Slice{
        .{ .Index = 0 },
        .{ .Range = .{} },
    };
    const sliced = try view.slice(&slices, allocator);
    defer sliced.deinit(allocator);

    // Mutate through sliced view
    sliced.set(&[_]usize{0}, 99.0);
    try std.testing.expectEqual(@as(f32, 99.0), data[0]);
}

/// An immutable view into a multi-dimensional array.
/// The view does NOT own the underlying data buffer or shape metadata.
/// You can only read elements through this view.
///
/// `T` is the element type.
pub fn ConstArrayView(comptime T: type) type {
    return struct {
        /// The dimensions of the array
        dims: []const usize,
        /// The strides for each dimension (in elements, not bytes)
        /// Should always have the same length as `dims`
        strides: []const isize,
        /// Pointer to "Logical Index 0" of the array view
        data_ptr: [*]const T,

        const Self = @This();

        /// Get a const pointer to the element at the given multi-dimensional index.
        ///
        /// Returns:
        /// - A pointer to the element if the index is valid
        /// - `null` if the index is out of bounds
        ///
        /// SAFETY:
        /// - The returned pointer is only valid as long as the underlying data exists
        /// - Do not store the pointer beyond the data's lifetime
        ///
        /// For safe value access, prefer the `get()` method.
        pub fn at(self: *const Self, index: []const usize) ?*const T {
            const offset = strideOffset(
                self.dims,
                self.strides,
                index,
            ) orelse return null;
            return array_mod.ptrFromOffset(T, self.data_ptr, offset);
        }

        /// Get a const pointer to the element at the given multi-dimensional index without bounds checking.
        ///
        /// SAFETY: The caller MUST ensure that all indices are within bounds.
        /// Undefined behavior if any index[i] >= dims[i] or if index.len != dims.len.
        ///
        /// This function skips all bounds checking for maximum performance.
        /// Use only when you have already validated the indices.
        pub fn atUnchecked(self: *const Self, index: []const usize) *const T {
            const offset = strideOffsetUnchecked(self.strides, index);
            return array_mod.ptrFromOffset(T, self.data_ptr, offset);
        }

        /// Get the element at the given multi-dimensional index.
        /// Returns null if the index is out of bounds.
        pub fn get(self: *const Self, index: []const usize) ?T {
            const ptr = self.at(index) orelse return null;
            return ptr.*;
        }

        /// Create a sliced view from this view.
        /// The returned view has the same mutability as the original.
        ///
        /// The caller owns the returned view's dims and strides arrays.
        pub fn slice(
            self: *const Self,
            slices: []const slice_mod.Slice,
            allocator: std.mem.Allocator,
        ) (slice_mod.SliceError || std.mem.Allocator.Error)!Self {
            const new_dims, const new_strides, const offset = try slice_mod.applySlices(
                self.dims,
                self.strides,
                slices,
                allocator,
            );

            // Calculate new data pointer
            const new_data_ptr_single = array_mod.ptrFromOffset(
                T,
                self.data_ptr,
                offset,
            );
            const new_data_ptr: [*]const T = @ptrCast(new_data_ptr_single);

            return Self{
                .dims = new_dims,
                .strides = new_strides,
                .data_ptr = new_data_ptr,
            };
        }

        pub fn deinit(self: Self, allocator: std.mem.Allocator) void {
            allocator.free(self.dims);
            allocator.free(self.strides);
        }
    };
}

test "ConstArrayView - standalone use" {
    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const dims = [_]usize{ 2, 3 };
    const strides = [_]isize{ 3, 1 }; // C-order strides

    const view = ConstArrayView(f32){
        .dims = &dims,
        .strides = &strides,
        .data_ptr = &data,
    };

    // Test get through view
    try std.testing.expectEqual(@as(?f32, 1.0), view.get(&[_]usize{ 0, 0 }));
    try std.testing.expectEqual(@as(?f32, 6.0), view.get(&[_]usize{ 1, 2 }));
    try std.testing.expectEqual(@as(?f32, 5.0), view.get(&[_]usize{ 1, 1 }));

    // Test at() returns const pointer
    const ptr = view.at(&[_]usize{ 0, 1 });
    try std.testing.expect(ptr != null);
    try std.testing.expectEqual(@as(f32, 2.0), ptr.?.*);

    // Verify return type is *const f32
    const ptr_type = @TypeOf(ptr.?);
    try std.testing.expect(ptr_type == *const f32);

    // Test bounds checking
    try std.testing.expectEqual(@as(?f32, null), view.get(&[_]usize{ 2, 0 }));
    try std.testing.expectEqual(@as(?f32, null), view.get(&[_]usize{ 0, 3 }));
}

test "ConstArrayView - slice immutability preserved" {
    const allocator = std.testing.allocator;
    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const dims = [_]usize{ 2, 2 };
    const strides = [_]isize{ 2, 1 };

    const view = ConstArrayView(f32){
        .dims = &dims,
        .strides = &strides,
        .data_ptr = &data,
    };

    // Slice and mutate
    const slices = [_]slice_mod.Slice{
        .{ .Index = 0 },
        .{ .Range = .{} },
    };
    const sliced = try view.slice(&slices, allocator);
    defer allocator.free(sliced.dims);
    defer allocator.free(sliced.strides);

    // Verify return type is *const f32
    const ptr = sliced.at(&[_]usize{ 0, 1 });
    const ptr_type = @TypeOf(ptr.?);
    try std.testing.expect(ptr_type == *const f32);
}
