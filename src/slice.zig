const std = @import("std");

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
pub fn asboluteIndex(index: isize, dim_size: usize) usize {
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

    const Self = @This();

    /// Get the number of dimensions required in the input array for applying these slices.
    pub fn inNdim(slices: []const Self) usize {
        var in_ndim: usize = 0;
        for (slices) |slice| {
            switch (slice) {
                // NewAxis slices don't consume a dimension
                .NewAxis => {},
                else => in_ndim += 1,
            }
        }
        return in_ndim;
    }

    /// Get the number of dimensions in the output array after applying these slices.
    pub fn outNdim(slices: []const Self) usize {
        var out_ndim: usize = 0;
        for (slices) |slice| {
            switch (slice) {
                // Index slices reduce dimensionality
                .Index => {},
                else => out_ndim += 1,
            }
        }
        return out_ndim;
    }
};

pub const SliceError = error{
    /// The expected number of dimensions from the slices
    /// differ from the number of dimensions.
    DimensionMismatch,
    /// The provided range values are invalid.
    InvalidRangeValues,
};

/// Calculate the new dims and strides arrays based on the given slices.
///
/// The caller owns the returned view's dims and strides arrays.
pub fn applySlices(
    dims: []const usize,
    strides: []const isize,
    slices: []const Slice,
    allocator: std.mem.Allocator,
) (SliceError || std.mem.Allocator.Error)!struct { []usize, []isize, isize } {
    std.debug.assert(dims.len == strides.len);

    const in_ndim = Slice.inNdim(slices);
    const out_ndim = Slice.outNdim(slices);

    // Validate that slices match input dimensions
    if (in_ndim != dims.len) {
        return SliceError.DimensionMismatch;
    }

    // Allocate new dims and strides
    const new_dims = try allocator.alloc(usize, out_ndim);
    errdefer allocator.free(new_dims);
    const new_strides = try allocator.alloc(isize, out_ndim);
    errdefer allocator.free(new_strides);

    // Calculate the offset to the first element and populate new dims/strides
    var offset: isize = 0;
    var in_axis: usize = 0;
    var out_axis: usize = 0;

    for (slices) |s| {
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
        }
    }

    std.debug.assert(in_axis == in_ndim);
    std.debug.assert(out_axis == out_ndim);

    return .{ new_dims, new_strides, offset };
}
