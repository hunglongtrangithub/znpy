//! Slicing utilities for multi-dimensional arrays.
//! Supports indexing, ranges with steps, new axes, and ellipses.
//! Example usage:
//! ```zig
//! const slice = Slice.Range{ .start = 0, .end = 10, .step = 2 };
//! const slices = &[_]Slice{
//!     .{ .Index = 0 },
//!     .Ellipsis,
//!     .{ .Range = .{ .start = 1, .end = 5 } },
//!     .NewAxis,
//! };
//! const sliced_view = try array_view.slice(slices, allocator);
//! ```
const std = @import("std");

const range_mod = @import("./slice/range.zig");
const format_mod = @import("./slice/format.zig");

const array_mod = @import("./array.zig");
const pointer_mod = @import("./pointer.zig");

const ArrayView = array_mod.ArrayView;
const Range = range_mod.Range;

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

pub const All = Slice{ .Range = .{} };
pub const Etc = Slice.Ellipsis;
pub const format_slice = format_mod.format_slice;

pub const SliceError = error{
    /// The expected number of dimensions from the slices
    /// differ from the number of dimensions.
    DimensionMismatch,
    /// The provided range values are invalid.
    InvalidRangeValues,
    /// The provided index value is invalid.
    InvalidIndexValue,
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

/// Calculate the new dims and strides arrays based on the given slices.
/// `dims` and `strides` are the original array's dimensions and strides,
/// and must have the same length.
/// When `dims` has at least one dimension of size 0, the strides are expected
/// to be all zeros as well.
/// Return the new dims and strides arrays along with the offset to the
///  first element of the sliced view: `.{ new_dims, new_strides, offset }`.
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
                const abs_idx = range_mod.absoluteIndex(idx, dims[in_axis]) orelse
                    return SliceError.InvalidIndexValue;
                // Now index is safe to cast
                offset += @as(isize, @intCast(abs_idx)) * strides[in_axis];
                in_axis += 1;
            },
            .Range => |range| {
                // Range: keep this dimension with modified stride and dimension size
                const start, const size = range.normalize(dims[in_axis]) orelse
                    return SliceError.InvalidRangeValues;

                new_dims[out_axis] = size;
                new_strides[out_axis] = strides[in_axis] * range.step;

                // Add offset to the start of the range
                offset += start * strides[in_axis];

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

test {
    _ = range_mod;
    _ = format_mod;
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

test "applySlices - basic slicing" {
    const allocator = std.testing.allocator;
    const dims = [_]usize{ 4, 5, 6 };
    const strides = [_]isize{ 30, 6, 1 }; // C-order

    const slices = &[_]Slice{
        .{ .Index = 1 },
        .{ .Range = .{ .start = 2, .end = 5, .step = 2 } },
        .NewAxis,
        .{ .Range = .{} },
    };

    const new_dims, const new_strides, const new_offset = try applySlices(
        &dims,
        &strides,
        slices,
        allocator,
    );
    defer allocator.free(new_dims);
    defer allocator.free(new_strides);

    try std.testing.expectEqualSlices(
        usize,
        &[_]usize{
            2, // from range
            1, // new axis
            6, // from full range
        },
        new_dims,
    );

    try std.testing.expectEqualSlices(
        isize,
        &[_]isize{
            12, // stride adjusted by step
            0, // new axis stride
            1, // original stride
        },
        new_strides,
    );

    try std.testing.expectEqual(42, new_offset); // offset = 1*30 + 2*6
}

test "applySlices - empty array's dimensions and strides" {
    const allocator = std.testing.allocator;
    const dims = [_]usize{ 0, 5, 6 };
    const strides = [_]isize{ 0, 0, 0 }; // empty array -> all strides are 0

    const slices = &[_]Slice{
        .{ .Range = .{ .start = 2, .end = 5, .step = 2 } },
        .NewAxis,
        .{ .Index = 1 },
        .{ .Range = .{} },
    };

    const new_dims, const new_strides, const new_offset = try applySlices(
        &dims,
        &strides,
        slices,
        allocator,
    );

    defer allocator.free(new_dims);
    defer allocator.free(new_strides);

    try std.testing.expectEqualSlices(
        usize,
        &[_]usize{
            0, // from range
            1, // new axis
            6, // from full range
        },
        new_dims,
    );

    try std.testing.expectEqualSlices(
        isize,
        &[_]isize{
            0, // from range
            0, // new axis stride
            0, // from full range
        },
        new_strides,
    );

    // Offset is always zero for empty arrays
    try std.testing.expectEqual(0, new_offset);
}

test "slicing tests - working cases" {
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
        try std.testing.expectEqualSlices(
            usize,
            &[_]usize{ 2, 3 },
            sliced.dims,
        );
        try std.testing.expectEqual(1, sliced.get(&[_]usize{ 0, 0 }));
        try std.testing.expectEqual(3, sliced.get(&[_]usize{ 0, 2 }));
        try std.testing.expectEqual(6, sliced.get(&[_]usize{ 1, 2 }));
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
        try std.testing.expectEqualSlices(
            usize,
            &[_]usize{ 2, 3 },
            sliced.dims,
        );
        try std.testing.expectEqual(7, sliced.get(&[_]usize{ 0, 0 }));
        try std.testing.expectEqual(12, sliced.get(&[_]usize{ 1, 2 }));
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
        try std.testing.expectEqualSlices(
            usize,
            &[_]usize{3},
            sliced.dims,
        );
        try std.testing.expectEqual(4, sliced.get(&[_]usize{0}));
        try std.testing.expectEqual(5, sliced.get(&[_]usize{1}));
        try std.testing.expectEqual(6, sliced.get(&[_]usize{2}));
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

        try std.testing.expectEqualSlices(
            usize,
            &[_]usize{ 2, 2 },
            sliced.dims,
        );
        try std.testing.expectEqual(2, sliced.get(&[_]usize{ 0, 0 }));
        try std.testing.expectEqual(5, sliced.get(&[_]usize{ 0, 1 }));
        try std.testing.expectEqual(8, sliced.get(&[_]usize{ 1, 0 }));
        try std.testing.expectEqual(11, sliced.get(&[_]usize{ 1, 1 }));
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
        try std.testing.expectEqualSlices(
            usize,
            &[_]usize{ 1, 2, 3 },
            sliced.dims,
        );
        try std.testing.expectEqual(1, sliced.get(&[_]usize{ 0, 0, 0 }));
        try std.testing.expectEqual(6, sliced.get(&[_]usize{ 0, 1, 2 }));
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
        try std.testing.expectEqualSlices(
            usize,
            &[_]usize{ 1, 2, 3 },
            sliced.dims,
        );
        try std.testing.expectEqual(1, sliced.get(&[_]usize{ 0, 0, 0 }));
        try std.testing.expectEqual(6, sliced.get(&[_]usize{ 0, 1, 2 }));
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

        try std.testing.expectEqualSlices(
            usize,
            &[_]usize{ 2, 1, 3 },
            sliced.dims,
        );
        try std.testing.expectEqual(1, sliced.get(&[_]usize{ 0, 0, 0 }));
        try std.testing.expectEqual(3, sliced.get(&[_]usize{ 0, 0, 2 }));
        try std.testing.expectEqual(4, sliced.get(&[_]usize{ 1, 0, 0 }));
        try std.testing.expectEqual(6, sliced.get(&[_]usize{ 1, 0, 2 }));
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

        try std.testing.expectEqualSlices(
            usize,
            &[_]usize{ 2, 3, 1 },
            sliced.dims,
        );
        try std.testing.expectEqual(1, sliced.get(&[_]usize{ 0, 0, 0 }));
        try std.testing.expectEqual(2, sliced.get(&[_]usize{ 0, 1, 0 }));
        try std.testing.expectEqual(3, sliced.get(&[_]usize{ 0, 2, 0 }));
        try std.testing.expectEqual(4, sliced.get(&[_]usize{ 1, 0, 0 }));
        try std.testing.expectEqual(5, sliced.get(&[_]usize{ 1, 1, 0 }));
        try std.testing.expectEqual(6, sliced.get(&[_]usize{ 1, 2, 0 }));
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

        try std.testing.expectEqualSlices(
            usize,
            &[_]usize{ 1, 3, 1 },
            sliced.dims,
        );
        try std.testing.expectEqual(1, sliced.get(&[_]usize{ 0, 0, 0 }));
        try std.testing.expectEqual(2, sliced.get(&[_]usize{ 0, 1, 0 }));
        try std.testing.expectEqual(3, sliced.get(&[_]usize{ 0, 2, 0 }));
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
        try std.testing.expectEqual(0, sliced.dims.len);
        try std.testing.expectEqual(12, sliced.get(&[_]usize{}));
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

        try std.testing.expectEqualSlices(
            usize,
            &[_]usize{ 2, 3 },
            sliced.dims,
        );
        try std.testing.expectEqual(7, sliced.get(&[_]usize{ 0, 0 }));
        try std.testing.expectEqual(12, sliced.get(&[_]usize{ 1, 2 }));
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

        try std.testing.expectEqualSlices(
            usize,
            &[_]usize{ 1, 2, 3 },
            sliced.dims,
        );
        try std.testing.expectEqual(1, sliced.get(&[_]usize{ 0, 0, 0 }));
        try std.testing.expectEqual(6, sliced.get(&[_]usize{ 0, 1, 2 }));
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
        try std.testing.expectEqualSlices(
            usize,
            &[_]usize{ 2, 2, 2 },
            sliced.dims,
        );
        try std.testing.expectEqual(1, sliced.get(&[_]usize{ 0, 0, 0 }));
        try std.testing.expectEqual(3, sliced.get(&[_]usize{ 0, 0, 1 }));
        try std.testing.expectEqual(4, sliced.get(&[_]usize{ 0, 1, 0 }));
        try std.testing.expectEqual(6, sliced.get(&[_]usize{ 0, 1, 1 }));
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
        try std.testing.expectEqualSlices(
            usize,
            &[_]usize{ 1, 3 },
            sliced.dims,
        );
        try std.testing.expectEqual(4, sliced.get(&[_]usize{ 0, 0 }));
        try std.testing.expectEqual(5, sliced.get(&[_]usize{ 0, 1 }));
        try std.testing.expectEqual(6, sliced.get(&[_]usize{ 0, 2 }));
    }

    // Test 15: Empty array
    {
        const empty_data: []u32 = pointer_mod.danglingPtr(u32)[0..0];
        const empty_dims = [_]usize{ 0, 1, 2 };
        const empty_strides = [_]isize{ 0, 0, 0 }; // empty array -> all strides are 0
        const empty_view = ArrayView(u32){
            .dims = &empty_dims,
            .strides = &empty_strides,
            .data_ptr = empty_data.ptr,
        };

        const sliced = try empty_view.slice(
            &[_]Slice{Slice.Ellipsis},
            allocator,
        );
        defer sliced.deinit(allocator);

        // There should be no pointer offset. Data pointer should be the same.
        try std.testing.expectEqual(@intFromPtr(empty_data.ptr), @intFromPtr(sliced.data_ptr));
    }
}

test "slicing tests - error cases" {
    const allocator = std.testing.allocator;
    var data = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    const dims = [_]usize{ 2, 2, 3 };
    const strides = [_]isize{ 6, 3, 1 }; // C-order: [[[1,2,3], [4,5,6]], [[7,8,9], [10,11,12]]]
    const view = ArrayView(u32){
        .dims = &dims,
        .strides = &strides,
        .data_ptr = &data,
    };

    // Test 1: Dimension mismatch
    {
        const result = view.slice(
            &[_]Slice{
                .{ .Index = 0 },
            },
            allocator,
        );
        try std.testing.expectEqual(SliceError.DimensionMismatch, result);
    }

    // Test 2: Invalid index value
    {
        const result = view.slice(
            &[_]Slice{
                .{ .Index = 2 },
                .Ellipsis,
            },
            allocator,
        );
        try std.testing.expectEqual(SliceError.InvalidIndexValue, result);
    }

    // Test 3: Invalid range values
    {
        const result = view.slice(
            &[_]Slice{
                .{ .Range = .{ .start = -3, .end = 5, .step = 0 } },
                .Ellipsis,
            },
            allocator,
        );
        try std.testing.expectEqual(SliceError.InvalidRangeValues, result);
    }

    // Test 4: Multiple ellipses
    {
        const result = view.slice(
            &[_]Slice{
                .Ellipsis,
                .{ .Index = 0 },
                .Ellipsis,
            },
            allocator,
        );
        try std.testing.expectEqual(SliceError.MultipleEllipsis, result);
    }
}
