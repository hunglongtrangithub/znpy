const std = @import("std");

const array = @import("./array.zig");

const ArrayView = array.ArrayView;

const Range = struct {
    /// Start index (inclusive).
    /// Negatives are counted from the back of the axis.
    start: isize = 0,
    /// End index (exclusive). If null, goes to the end.
    /// Negatives are counted from the back of the axis.
    end: ?isize = null,
    /// Step size. Should never be zero.
    /// Negatives mean counting backwards.
    step: isize = 1,

    const Self = @This();

    /// Convert the possibly negative start and end indices to absolute (non-negative, non-null) indices
    /// based on the given dimension size.
    /// Return null when range values are invalid:
    ///
    /// 1. The start index (after converting negative indices) exceeds the dimension size
    /// 2. The end index (after converting negative indices) exceeds the dimension size
    /// 3. The step is zero
    ///
    /// On valid inputs, return { start, end, step }
    pub fn toAbsolute(self: Self, dim_size: usize) ?struct { usize, usize, isize } {
        const abs_start = asboluteIndex(self.start, dim_size);

        // Handle end index: can be dim_size (exclusive upper bound)
        const abs_end = if (self.end) |e| blk: {
            if (e >= 0) {
                const abs_e: usize = @intCast(e);
                // Allow end to equal dim_size (exclusive bound)
                if (abs_e > dim_size) break :blk dim_size + 1; // will fail validation
                break :blk abs_e;
            } else {
                break :blk asboluteIndex(e, dim_size);
            }
        } else dim_size;

        if (abs_start < dim_size and abs_end <= dim_size and self.step != 0) {
            return .{ abs_start, abs_end, self.step };
        }

        return null;
    }
};

/// Convert a possibly negative index into a non-negative index based on the dimension size.
/// `dim_size` must not be more than `std.math.MaxInt(isize)` to prevent overflow.
/// Example:
/// ```zig
/// const idx = asboluteIndex(-1, 5); // idx == 4
/// const idx2 = asboluteIndex(2, 5); // idx2 == 2
/// ```
/// The valid input index range is `[-dim_size, dim_size)`.
/// Returned output index range is `[0, dim_size)`.
fn asboluteIndex(index: isize, dim_size: usize) usize {
    if (index >= 0) {
        const abs_index: usize = @intCast(index);
        std.debug.assert(abs_index < dim_size);
        return abs_index;
    } else {
        const abs_index: usize = @intCast(-index);
        std.debug.assert(abs_index <= dim_size);
        return dim_size - abs_index;
    }
}

pub const Slice = union(enum) {
    /// Single index (collapses the dimension).
    Index: isize,
    /// A range with step size.
    Range: Range,
    /// New axis (adds a dimension of size 1).
    NewAxis,
    /// Ellipsis (expands to multiple Range slices).
    Ellipsis,
};

pub const SliceError = error{
    /// The expected number of dimensions from the slices
    /// differ from the number of dimensions.
    DimensionMismatch,
    /// The provided range values are invalid.
    InvalidRangeValues,
    /// Multiple ellipsis slices found. Only 1 is allowed.
    MultipleEllipsis,
};

/// Expand ellipsis slices into full Range slices based on the array rank.
/// The caller owns the returned slice array.
/// If no ellipsis is found, returns `null`.
/// Returns an error if multiple ellipses are found or if the slices
/// cannot match the array rank.
/// Example:
/// ```zig
/// const slices1 = &[_]Slice{
///     .{ .Index = 0 },
///     .Ellipsis,
///     .{ .Range = .{} },
/// };
/// const expanded1 = try expandEllipsis(slices, 4, allocator);
/// // expanded1 == &[_]Slice{
/// //     .{ .Index = 0 },
/// //     .{ .Range = .{} }, // expanded
/// //     .{ .Range = .{} }, // expanded
/// //     .{ .Range = .{} },
/// // }
/// const slices2 = &[_]Slice{
///     .{ .Range = .{} },
///     .{ .Index = 1 },
///     .Ellipsis,
///     .{ .NewAxis },
/// };
/// const expanded2 = try expandEllipsis(slices2, 2, allocator);
/// // expanded2 == &[_]Slice{
/// //     .{ .Range = .{} },
/// //     .{ .Index = 1 },
/// //     .{ .NewAxis },
/// // }
/// ```
fn expandEllipsis(
    slices: []const Slice,
    array_rank: usize,
    allocator: std.mem.Allocator,
) (SliceError || std.mem.Allocator.Error)!?[]const Slice {
    // Find ellipsis
    var ellipsis_idx: ?usize = null;
    var explicit_dims: usize = 0;

    for (slices, 0..) |slice, i| {
        switch (slice) {
            .Ellipsis => {
                if (ellipsis_idx != null) return SliceError.MultipleEllipsis;
                ellipsis_idx = i;
            },
            .Index, .Range => explicit_dims += 1,
            .NewAxis => {},
        }
    }

    // If no ellipsis, return null
    if (ellipsis_idx == null) return null;

    // One ellipsis exists. Calculate how many Range slices the ellipsis should expand to
    const ellipsis_expansion = if (array_rank >= explicit_dims)
        array_rank - explicit_dims
    else
        return SliceError.DimensionMismatch;

    // Create new slice array with ellipsis expanded
    const new_len = slices.len - 1 + ellipsis_expansion;
    const expanded_slices = try allocator.alloc(Slice, new_len);

    // Copy slices before ellipsis
    @memcpy(expanded_slices[0..ellipsis_idx.?], slices[0..ellipsis_idx.?]);
    // Fill in expanded ellipsis
    for (0..ellipsis_expansion) |i| {
        expanded_slices[ellipsis_idx.? + i] = .{ .Range = .{} };
    }
    // Copy slices after ellipsis
    @memcpy(expanded_slices[ellipsis_idx.? + ellipsis_expansion ..], slices[ellipsis_idx.? + 1 ..]);

    return expanded_slices;
}

test "expandEllipsis - Single ellipsis in the middle" {
    const allocator = std.testing.allocator;
    const slices = &[_]Slice{
        .{ .Index = 0 },
        .Ellipsis,
        .{ .Range = .{} },
    };
    const expanded = (try expandEllipsis(
        slices,
        4,
        allocator,
    )).?;
    defer allocator.free(expanded);
    try std.testing.expectEqualSlices(
        Slice,
        &[_]Slice{
            .{ .Index = 0 },
            .{ .Range = .{} },
            .{ .Range = .{} },
            .{ .Range = .{} },
        },
        expanded,
    );
}

test "expandEllipsis - no ellipsis" {
    const allocator = std.testing.allocator;
    const slices = &[_]Slice{
        .{ .Range = .{} },
        .{ .Index = 1 },
        .NewAxis,
    };
    const expanded = try expandEllipsis(
        slices,
        2,
        allocator,
    );
    // Should be null
    try std.testing.expectEqual(null, expanded);
}

test "expandEllipsis - Multiple ellipses (error)" {
    const allocator = std.testing.allocator;
    const slices = &[_]Slice{
        .Ellipsis,
        .{ .Index = 0 },
        .Ellipsis,
    };
    const result = expandEllipsis(
        slices,
        3,
        allocator,
    );
    try std.testing.expectEqual(SliceError.MultipleEllipsis, result);
}

test "expandEllipsis - Ellipsis expansion that doesn't fit (error)" {
    const allocator = std.testing.allocator;
    const slices = &[_]Slice{
        .{ .Index = 0 },
        .NewAxis,
        .Ellipsis,
        .{ .Range = .{} },
    };
    const result = expandEllipsis(
        slices,
        1,
        allocator,
    );
    try std.testing.expectEqual(SliceError.DimensionMismatch, result);
}

test "expandEllipsis - Ellipsis does not expand when exact fit" {
    const allocator = std.testing.allocator;
    const slices = &[_]Slice{
        .{ .Index = 0 },
        .Ellipsis,
        .{ .Range = .{} },
        .NewAxis,
    };
    const expanded = (try expandEllipsis(
        slices,
        2,
        allocator,
    )).?;
    defer allocator.free(expanded);
    try std.testing.expectEqualSlices(
        Slice,
        &[_]Slice{
            .{ .Index = 0 },
            .{ .Range = .{} },
            .NewAxis,
        },
        expanded,
    );
}

test "Ellipsis expansion at end" {
    const allocator = std.testing.allocator;
    const slices = &[_]Slice{
        .{ .Index = 1 },
        .Ellipsis,
    };
    const expanded = (try expandEllipsis(
        slices,
        3,
        allocator,
    )).?;
    defer allocator.free(expanded);
    try std.testing.expectEqualSlices(
        Slice,
        &[_]Slice{
            .{ .Index = 1 },
            .{ .Range = .{} },
            .{ .Range = .{} },
        },
        expanded,
    );
}

/// Calculate the new dims and strides arrays based on the given slices.
/// `dims` and `strides` are the original array's dimensions and strides,
/// and must have the same length.
/// Return the new dims and strides arrays along with the offset to the
///  first element of the sliced view.
/// The caller owns the returned view's dims and strides arrays.
pub fn applySlices(
    dims: []const usize,
    strides: []const isize,
    slices: []const Slice,
    allocator: std.mem.Allocator,
) (SliceError || std.mem.Allocator.Error)!struct { []usize, []isize, isize } {
    std.debug.assert(dims.len == strides.len);

    // Expand ellipsis first
    const maybe_expanded_slices = try expandEllipsis(
        slices,
        dims.len,
        allocator,
    );
    defer if (maybe_expanded_slices) |es| allocator.free(es);
    const expanded_slices = if (maybe_expanded_slices) |es| es else slices;

    // Calculate expected input rank
    const in_rank = blk: {
        var count: usize = 0;
        for (expanded_slices) |s| {
            switch (s) {
                .Index, .Range => count += 1,
                // NewAxis does not consume an input dimension
                .NewAxis => {},
                .Ellipsis => unreachable, // should have been expanded already
            }
        }
        break :blk count;
    };

    if (in_rank != dims.len) {
        return SliceError.DimensionMismatch;
    }

    // Calculate output rank
    const out_rank = blk: {
        var count: usize = 0;
        for (expanded_slices) |s| {
            switch (s) {
                // Index collapses dimension
                .Index => {},
                .Range, .NewAxis => count += 1,
                .Ellipsis => unreachable, // should have been expanded already
            }
        }
        break :blk count;
    };

    // Allocate new dims and strides
    const new_dims = try allocator.alloc(usize, out_rank);
    errdefer allocator.free(new_dims);
    const new_strides = try allocator.alloc(isize, out_rank);
    errdefer allocator.free(new_strides);

    // Calculate the offset to the first element and populate new dims/strides
    var offset: isize = 0;
    var in_axis: usize = 0;
    var out_axis: usize = 0;

    for (expanded_slices) |s| {
        switch (s) {
            .Index => |idx| {
                // Index: collapse this dimension
                const abs_idx = asboluteIndex(idx, dims[in_axis]);
                offset += @as(isize, @intCast(abs_idx)) * strides[in_axis];
                in_axis += 1;
            },
            .Range => |range| {
                // Range: keep this dimension with modified stride
                const start, const end, const step = range.toAbsolute(dims[in_axis]) orelse return SliceError.InvalidRangeValues;

                // Calculate new dimension size
                const range_size = if (end > start) end - start else 0;
                const abs_step = if (step > 0) @as(usize, @intCast(step)) else @as(usize, @intCast(-step));
                const new_size = if (range_size == 0) 0 else (range_size + abs_step - 1) / abs_step;

                new_dims[out_axis] = new_size;
                new_strides[out_axis] = strides[in_axis] * step;

                // Add offset to the start of the range
                // For negative steps, start from end-1 and go backwards
                const range_start_idx = if (step > 0) start else if (end > 0) end - 1 else 0;
                offset += @as(isize, @intCast(range_start_idx)) * strides[in_axis];

                in_axis += 1;
                out_axis += 1;
            },
            .NewAxis => {
                // NewAxis: add a dimension of size 1 with stride 0
                new_dims[out_axis] = 1;
                new_strides[out_axis] = 0;
                out_axis += 1;
            },
            .Ellipsis => unreachable, // should have been expanded already
        }
    }

    std.debug.assert(in_axis == in_rank);
    std.debug.assert(out_axis == out_rank);

    return .{ new_dims, new_strides, offset };
}

test "comprehensive slicing tests" {
    const allocator = std.testing.allocator;
    var data = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    const dims = [_]usize{ 2, 2, 3 };
    const strides = [_]isize{ 6, 3, 1 }; // C-order: [[[1,2,3], [4,5,6]], [[7,8,9], [10,11,12]]]
    const view = ArrayView(u32){
        .dims = &dims,
        .strides = &strides,
        .data_ptr = &data,
    };

    // Test 1: Single Index - get first 2x3 slice
    // [Index=0, Ellipsis]
    // -> [Index=0, Range, Range]
    // -> [[1,2,3], [4,5,6]] with shape [2, 3]
    {
        const sliced = try view.slice(
            &[_]Slice{
                .{ .Index = 0 },
                .Ellipsis,
            },
            allocator,
        );
        defer sliced.deinit(allocator);
        try std.testing.expectEqual(@as(usize, 2), sliced.dims.len);
        try std.testing.expectEqual(@as(usize, 2), sliced.dims[0]);
        try std.testing.expectEqual(@as(usize, 3), sliced.dims[1]);
        try std.testing.expectEqual(@as(?u32, 1), sliced.get(&[_]usize{ 0, 0 }));
        try std.testing.expectEqual(@as(?u32, 3), sliced.get(&[_]usize{ 0, 2 }));
        try std.testing.expectEqual(@as(?u32, 6), sliced.get(&[_]usize{ 1, 2 }));
    }

    // Test 2: Single Index - get second 2x3 slice
    // [Index=1, Sllipsis]
    // -> [Index=1, Range, Range]
    // -> [[7,8,9], [10,11,12]] with shape [2, 3]
    {
        const sliced = try view.slice(
            &[_]Slice{
                .{ .Index = 1 },
                .Ellipsis,
            },
            allocator,
        );
        defer sliced.deinit(allocator);
        try std.testing.expectEqual(@as(usize, 2), sliced.dims.len);
        try std.testing.expectEqual(@as(?u32, 7), sliced.get(&[_]usize{ 0, 0 }));
        try std.testing.expectEqual(@as(?u32, 12), sliced.get(&[_]usize{ 1, 2 }));
    }

    // Test 3: Double Index - get specific row
    // [Index=0, Index=1, Ellipsis]
    // -> [Index=0, Index=1, Range]
    // -> [4,5,6] with shape [3]
    {
        const sliced = try view.slice(
            &[_]Slice{
                .{ .Index = 0 },
                .{ .Index = 1 },
                .Ellipsis,
            },
            allocator,
        );
        defer sliced.deinit(allocator);
        try std.testing.expectEqual(@as(usize, 1), sliced.dims.len);
        try std.testing.expectEqual(@as(usize, 3), sliced.dims[0]);
        try std.testing.expectEqual(@as(?u32, 4), sliced.get(&[_]usize{0}));
        try std.testing.expectEqual(@as(?u32, 5), sliced.get(&[_]usize{1}));
        try std.testing.expectEqual(@as(?u32, 6), sliced.get(&[_]usize{2}));
    }

    // Test 4: Range with bounds - get middle column
    // [Ellipsis, Index=1]
    // -> [Range, Range, Index=1]
    // -> [[2,5], [8,11]] with shape [2, 2]
    {
        const sliced = try view.slice(
            &[_]Slice{
                .Ellipsis,
                .{ .Index = 1 },
            },
            allocator,
        );
        defer sliced.deinit(allocator);
        try std.testing.expectEqual(@as(usize, 2), sliced.dims.len);
        try std.testing.expectEqual(@as(usize, 2), sliced.dims[0]);
        try std.testing.expectEqual(@as(usize, 2), sliced.dims[1]);
        try std.testing.expectEqual(@as(?u32, 2), sliced.get(&[_]usize{ 0, 0 }));
        try std.testing.expectEqual(@as(?u32, 5), sliced.get(&[_]usize{ 0, 1 }));
        try std.testing.expectEqual(@as(?u32, 8), sliced.get(&[_]usize{ 1, 0 }));
        try std.testing.expectEqual(@as(?u32, 11), sliced.get(&[_]usize{ 1, 1 }));
    }

    // Test 5: NewAxis at front
    // [NewAxis, Index=0, Ellipsis]
    // -> [NewAxis, Index=0, Range, Range]
    // -> [[[1,2,3], [4,5,6]]] with shape [1, 2, 3]
    {
        const sliced = try view.slice(
            &[_]Slice{
                .NewAxis,
                .{ .Index = 0 },
                .Ellipsis,
            },
            allocator,
        );
        defer sliced.deinit(allocator);
        try std.testing.expectEqual(@as(usize, 3), sliced.dims.len);
        try std.testing.expectEqual(@as(usize, 1), sliced.dims[0]);
        try std.testing.expectEqual(@as(usize, 2), sliced.dims[1]);
        try std.testing.expectEqual(@as(usize, 3), sliced.dims[2]);
        try std.testing.expectEqual(@as(?u32, 1), sliced.get(&[_]usize{ 0, 0, 0 }));
        try std.testing.expectEqual(@as(?u32, 6), sliced.get(&[_]usize{ 0, 1, 2 }));
    }

    // Test 6: NewAxis in middle
    // [Index=0, NewAxis, Range, Range]
    // -> [Index=0, NewAxis, Range, Range]
    // -> [[[1,2,3], [4,5,6]]] with shape [1, 2, 3]
    {
        const sliced = try view.slice(
            &[_]Slice{
                .{ .Index = 0 },
                .NewAxis,
                .Ellipsis,
            },
            allocator,
        );
        defer sliced.deinit(allocator);
        try std.testing.expectEqual(@as(usize, 3), sliced.dims.len);
        try std.testing.expectEqual(@as(usize, 1), sliced.dims[0]);
        try std.testing.expectEqual(@as(usize, 2), sliced.dims[1]);
        try std.testing.expectEqual(@as(usize, 3), sliced.dims[2]);
        try std.testing.expectEqual(@as(?u32, 1), sliced.get(&[_]usize{ 0, 0, 0 }));
        try std.testing.expectEqual(@as(?u32, 6), sliced.get(&[_]usize{ 0, 1, 2 }));
    }

    // Test 7: NewAxis after two ranges
    // [Index=0, Range, NewAxis, Ellipsis]
    // -> [Index=0, Range, NewAxis, Range]
    // -> [[[1,2,3]], [[4,5,6]]] with shape [2, 1, 3]
    {
        const sliced = try view.slice(
            &[_]Slice{
                .{ .Index = 0 },
                .{ .Range = .{} },
                .NewAxis,
                .Ellipsis,
            },
            allocator,
        );
        defer sliced.deinit(allocator);
        try std.testing.expectEqual(@as(usize, 3), sliced.dims.len);
        try std.testing.expectEqual(@as(usize, 2), sliced.dims[0]);
        try std.testing.expectEqual(@as(usize, 1), sliced.dims[1]);
        try std.testing.expectEqual(@as(usize, 3), sliced.dims[2]);
        try std.testing.expectEqual(@as(?u32, 1), sliced.get(&[_]usize{ 0, 0, 0 }));
        try std.testing.expectEqual(@as(?u32, 3), sliced.get(&[_]usize{ 0, 0, 2 }));
        try std.testing.expectEqual(@as(?u32, 4), sliced.get(&[_]usize{ 1, 0, 0 }));
        try std.testing.expectEqual(@as(?u32, 6), sliced.get(&[_]usize{ 1, 0, 2 }));
    }

    // Test 8: NewAxis at end
    // [Index=0, Ellipsis, NewAxis]
    // -> [Index=0, Range, Range, NewAxis]
    // -> [[[1],[2],[3]], [[4],[5],[6]]] with shape [2, 3, 1]
    {
        const sliced = try view.slice(
            &[_]Slice{
                .{ .Index = 0 },
                .Ellipsis,
                .NewAxis,
            },
            allocator,
        );
        defer sliced.deinit(allocator);
        try std.testing.expectEqual(@as(usize, 3), sliced.dims.len);
        try std.testing.expectEqual(@as(usize, 2), sliced.dims[0]);
        try std.testing.expectEqual(@as(usize, 3), sliced.dims[1]);
        try std.testing.expectEqual(@as(usize, 1), sliced.dims[2]);
        try std.testing.expectEqual(@as(?u32, 1), sliced.get(&[_]usize{ 0, 0, 0 }));
        try std.testing.expectEqual(@as(?u32, 2), sliced.get(&[_]usize{ 0, 1, 0 }));
        try std.testing.expectEqual(@as(?u32, 3), sliced.get(&[_]usize{ 0, 2, 0 }));
        try std.testing.expectEqual(@as(?u32, 4), sliced.get(&[_]usize{ 1, 0, 0 }));
        try std.testing.expectEqual(@as(?u32, 5), sliced.get(&[_]usize{ 1, 1, 0 }));
        try std.testing.expectEqual(@as(?u32, 6), sliced.get(&[_]usize{ 1, 2, 0 }));
    }

    // Test 9: Multiple NewAxis
    // [NewAxis, Index=0, Index=0, Range, NewAxis]
    // -> [[[1,2,3]]] with shape [1, 3, 1]
    {
        const sliced = try view.slice(
            &[_]Slice{
                .NewAxis,
                .{ .Index = 0 },
                .{ .Index = 0 },
                .{ .Range = .{} },
                .NewAxis,
            },
            allocator,
        );
        defer sliced.deinit(allocator);
        try std.testing.expectEqual(@as(usize, 3), sliced.dims.len);
        try std.testing.expectEqual(@as(usize, 1), sliced.dims[0]);
        try std.testing.expectEqual(@as(usize, 3), sliced.dims[1]);
        try std.testing.expectEqual(@as(usize, 1), sliced.dims[2]);
        try std.testing.expectEqual(@as(?u32, 1), sliced.get(&[_]usize{ 0, 0, 0 }));
        try std.testing.expectEqual(@as(?u32, 2), sliced.get(&[_]usize{ 0, 1, 0 }));
        try std.testing.expectEqual(@as(?u32, 3), sliced.get(&[_]usize{ 0, 2, 0 }));
    }

    // Test 10: Get a specific element (all indices)
    // [Index=1, Index=1, Index=2]
    // -> scalar 12 with shape []
    {
        const sliced = try view.slice(
            &[_]Slice{
                .{ .Index = 1 },
                .{ .Index = 1 },
                .{ .Index = 2 },
            },
            allocator,
        );
        defer sliced.deinit(allocator);
        try std.testing.expectEqual(@as(usize, 0), sliced.dims.len);
        try std.testing.expectEqual(@as(?u32, 12), sliced.get(&[_]usize{}));
    }

    // Test 11: Negative indices
    // [Index=-1, Ellipsis]
    // -> [[7,8,9], [10,11,12]] (last slice)
    {
        const sliced = try view.slice(
            &[_]Slice{
                .{ .Index = -1 },
                .Ellipsis,
            },
            allocator,
        );
        defer sliced.deinit(allocator);
        try std.testing.expectEqual(@as(usize, 2), sliced.dims.len);
        try std.testing.expectEqual(@as(?u32, 7), sliced.get(&[_]usize{ 0, 0 }));
        try std.testing.expectEqual(@as(?u32, 12), sliced.get(&[_]usize{ 1, 2 }));
    }

    // Test 12: Range with explicit bounds
    // [Range{start=0, end=1}, Ellipsis]
    // -> [[[1,2,3], [4,5,6]]] (first slice only)
    {
        const sliced = try view.slice(
            &[_]Slice{
                .{ .Range = .{ .start = 0, .end = 1 } },
                .Ellipsis,
            },
            allocator,
        );
        defer sliced.deinit(allocator);
        try std.testing.expectEqual(@as(usize, 3), sliced.dims.len);
        try std.testing.expectEqual(@as(usize, 1), sliced.dims[0]);
        try std.testing.expectEqual(@as(usize, 2), sliced.dims[1]);
        try std.testing.expectEqual(@as(usize, 3), sliced.dims[2]);
        try std.testing.expectEqual(@as(?u32, 1), sliced.get(&[_]usize{ 0, 0, 0 }));
        try std.testing.expectEqual(@as(?u32, 6), sliced.get(&[_]usize{ 0, 1, 2 }));
    }

    // Test 13: Range with step
    // [Ellipsis, Range{start=0, end=3, step=2}]
    // -> [Range, Range, Range{start=0, end=3, step=2}]
    // -> every other element in last dim
    // -> [[[1,3], [4,6]], [[7,9], [10,12]]]
    {
        const sliced = try view.slice(
            &[_]Slice{
                .Ellipsis,
                .{ .Range = .{ .start = 0, .end = 3, .step = 2 } },
            },
            allocator,
        );
        defer sliced.deinit(allocator);
        try std.testing.expectEqual(@as(usize, 3), sliced.dims.len);
        try std.testing.expectEqual(@as(usize, 2), sliced.dims[0]);
        try std.testing.expectEqual(@as(usize, 2), sliced.dims[1]);
        try std.testing.expectEqual(@as(usize, 2), sliced.dims[2]);
        try std.testing.expectEqual(@as(?u32, 1), sliced.get(&[_]usize{ 0, 0, 0 }));
        try std.testing.expectEqual(@as(?u32, 3), sliced.get(&[_]usize{ 0, 0, 1 }));
        try std.testing.expectEqual(@as(?u32, 4), sliced.get(&[_]usize{ 0, 1, 0 }));
        try std.testing.expectEqual(@as(?u32, 6), sliced.get(&[_]usize{ 0, 1, 1 }));
    }

    // Test 14: Combination - get diagonal-like pattern
    // [Index=0, Index=1, NewAxis, Range]
    // -> [[4,5,6]] with shape [1, 3]
    {
        const sliced = try view.slice(
            &[_]Slice{
                .{ .Index = 0 },
                .{ .Index = 1 },
                .NewAxis,
                .{ .Range = .{} },
            },
            allocator,
        );
        defer sliced.deinit(allocator);
        try std.testing.expectEqual(@as(usize, 2), sliced.dims.len);
        try std.testing.expectEqual(@as(usize, 1), sliced.dims[0]);
        try std.testing.expectEqual(@as(usize, 3), sliced.dims[1]);
        try std.testing.expectEqual(@as(?u32, 4), sliced.get(&[_]usize{ 0, 0 }));
        try std.testing.expectEqual(@as(?u32, 5), sliced.get(&[_]usize{ 0, 1 }));
        try std.testing.expectEqual(@as(?u32, 6), sliced.get(&[_]usize{ 0, 2 }));
    }
}
