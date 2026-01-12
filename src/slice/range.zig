const std = @import("std");

/// Convert a possibly negative index into a non-negative index based on the dimension size.
/// Reuturn the absolute index if valid, otherwise `null`.
/// Example:
/// ```zig
/// const idx = asboluteIndex(-1, 5); // idx == 4
/// const idx2 = asboluteIndex(2, 5); // idx2 == 2
/// const idx3 = asboluteIndex(5, 5); // idx3 == null
/// ```
/// The valid input index range is `[-dim_size, dim_size)`.
/// Returned output index range is `[0, dim_size)`.
pub fn absoluteIndex(index: isize, dim_size: usize) ?usize {
    if (index >= 0) {
        const abs_index: usize = @intCast(index);
        return if (abs_index < dim_size)
            abs_index
        else
            null;
    } else {
        const abs_index: usize = @intCast(-index);
        return if (abs_index <= dim_size)
            dim_size - abs_index
        else
            null;
    }
}

test "absoluteIndex - non-negative index" {
    try std.testing.expectEqual(0, absoluteIndex(0, 10));
    try std.testing.expectEqual(5, absoluteIndex(5, 10));
    try std.testing.expectEqual(null, absoluteIndex(10, 10));
    try std.testing.expectEqual(null, absoluteIndex(11, 10));
}

test "absoluteIndex - negative index" {
    try std.testing.expectEqual(9, absoluteIndex(-1, 10));
    try std.testing.expectEqual(5, absoluteIndex(-5, 10));
    try std.testing.expectEqual(0, absoluteIndex(-10, 10));
    try std.testing.expectEqual(null, absoluteIndex(-11, 10));
}

test "absoluteIndex - 0 dimension size" {
    try std.testing.expectEqual(null, absoluteIndex(0, 0));
    try std.testing.expectEqual(null, absoluteIndex(1, 0));
    try std.testing.expectEqual(null, absoluteIndex(-1, 0));
}

pub const Range = struct {
    /// Start index (inclusive). If null, defaults based on step direction.
    /// Negatives are counted from the back of the axis.
    start: ?isize = null,
    /// End index (exclusive). If null, defaults based on step direction.
    /// Negatives are counted from the back of the axis.
    end: ?isize = null,
    /// Step size. Should never be zero.
    /// Negatives mean counting backwards.
    step: isize = 1,

    const Self = @This();

    /// Normalize range indicies based on dimension size.
    /// The given dimension size must be no greater than `std.math.maxInt(isize)`.
    /// Range of normalized indicies for start and end:
    /// - start: `[0, dim_size - 1]`
    /// - end: `[-1, dim_size]`
    /// Return `null` when step is 0.]
    /// Return value is a struct of (start, num_elements).
    pub fn normalize(self: Self, dim_size: usize) ?struct { isize, usize } {
        // step must not be zero
        if (self.step == 0) {
            return null;
        }

        // dim_size must fit in isize
        const n = @as(isize, @intCast(dim_size));

        if (n == 0) {
            // It's obvious number of elements is zero,
            // but start index should be zero so that
            // slicing works correctly, by not changing
            // the current pointer offset.
            return .{ 0, 0 };
        }

        // Get default indicies for start and end
        const default_start: isize = if (self.step > 0) 0 else -1;
        const default_end: isize = if (self.step > 0) n else -n - 1;

        // Fill default values
        var start = self.start orelse default_start;
        var end = self.end orelse default_end;

        // Normalize start and end
        start = if (start >= 0) start else n + start;
        end = if (end >= 0) end else n + end;

        // Clamp start and end
        // start: [0, n-1]
        // end: [-1, n]
        start = @min(@max(0, start), n - 1);
        end = @min(@max(-1, end), n);

        // Calculate number of elements from the processed start and end
        const num_elements: usize = blk: {
            if (self.step > 0) {
                if (end <= start) break :blk 0;
                const range_size = @as(usize, @intCast(end - start));
                const abs_step = @as(usize, @intCast(self.step));
                break :blk (range_size + abs_step - 1) / abs_step;
            } else {
                if (start <= end) break :blk 0;
                const range_size = @as(usize, @intCast(start - end));
                const abs_step = @as(usize, @intCast(-self.step));
                break :blk (range_size + abs_step - 1) / abs_step;
            }
        };

        return .{ start, num_elements };
    }
};

test "range normalization - default start" {
    const r = Range{ .end = 5, .step = 1 };
    const normalized = r.normalize(10).?;
    try std.testing.expectEqual(0, normalized.@"0");
    try std.testing.expectEqual(5, normalized.@"1");

    const r2 = Range{ .end = 5, .step = -1 };
    const normalized2 = r2.normalize(10).?;
    try std.testing.expectEqual(9, normalized2.@"0");
    try std.testing.expectEqual(4, normalized2.@"1");
}

test "range normalization - default end" {
    const r = Range{ .start = 3, .step = 1 };
    const normalized = r.normalize(10).?;
    try std.testing.expectEqual(3, normalized.@"0");
    try std.testing.expectEqual(7, normalized.@"1");

    const r2 = Range{ .start = 7, .step = -1 };
    const normalized2 = r2.normalize(10).?;
    try std.testing.expectEqual(7, normalized2.@"0");
    try std.testing.expectEqual(8, normalized2.@"1");
}

test "range normalization - negative and non-negative indicies" {
    const r = Range{ .start = -7, .end = 5, .step = 1 };
    const normalized = r.normalize(10).?;
    try std.testing.expectEqual(3, normalized.@"0");
    try std.testing.expectEqual(2, normalized.@"1");

    const r2 = Range{ .start = 2, .end = -10, .step = -1 };
    const normalized2 = r2.normalize(10).?;
    try std.testing.expectEqual(2, normalized2.@"0");
    try std.testing.expectEqual(2, normalized2.@"1");
}

test "range normalization - clamping" {
    const r = Range{ .start = -20, .end = 15, .step = 1 };
    const normalized = r.normalize(10).?;
    try std.testing.expectEqual(0, normalized.@"0");
    try std.testing.expectEqual(10, normalized.@"1");

    const r2 = Range{ .start = 15, .end = -20, .step = -1 };
    const normalized2 = r2.normalize(10).?;
    try std.testing.expectEqual(9, normalized2.@"0");
    try std.testing.expectEqual(10, normalized2.@"1");
}

test "range normalization - zero elements" {
    const r = Range{ .start = 5, .end = 5, .step = 1 };
    const normalized = r.normalize(10).?;
    try std.testing.expectEqual(5, normalized.@"0");
    try std.testing.expectEqual(0, normalized.@"1");

    const r2 = Range{ .start = 5, .end = 5, .step = -1 };
    const normalized2 = r2.normalize(10).?;
    try std.testing.expectEqual(5, normalized2.@"0");
    try std.testing.expectEqual(0, normalized2.@"1");

    const r3 = Range{ .start = 7, .end = 5, .step = 1 };
    const normalized3 = r3.normalize(10).?;
    try std.testing.expectEqual(7, normalized3.@"0");
    try std.testing.expectEqual(0, normalized3.@"1");
}

test "range normalization - zero dimension size" {
    const r = Range{ .start = 0, .end = 0, .step = 1 };
    const normalized = r.normalize(0).?;
    try std.testing.expectEqual(0, normalized.@"0");
    try std.testing.expectEqual(0, normalized.@"1");

    const r2 = Range{ .start = 0, .end = 0, .step = -1 };
    const normalized2 = r2.normalize(0).?;
    try std.testing.expectEqual(0, normalized2.@"0");
    try std.testing.expectEqual(0, normalized2.@"1");
}
