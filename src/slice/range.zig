const std = @import("std");

/// Convert a possibly negative index into a non-negative index based on the dimension size.
/// `dim_size` must not be more than `std.math.MaxInt(isize)` to prevent overflow.
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

pub const Range = struct {
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
        const abs_start = absoluteIndex(self.start, dim_size) orelse return null;

        // Handle end index: can be dim_size (exclusive upper bound)
        const abs_end = if (self.end) |e| blk: {
            if (e >= 0) {
                const abs_e: usize = @intCast(e);
                // Allow end to equal dim_size (exclusive bound)
                if (abs_e > dim_size) break :blk dim_size + 1; // will fail validation
                break :blk abs_e;
            } else {
                break :blk absoluteIndex(e, dim_size) orelse return null;
            }
        } else dim_size;

        // Validate indices
        if (abs_start < dim_size and abs_end <= dim_size and self.step != 0) {
            return .{ abs_start, abs_end, self.step };
        }

        return null;
    }
};

test "asboluteIndex" {
    try std.testing.expectEqual(2, absoluteIndex(2, 5).?);
    try std.testing.expectEqual(4, absoluteIndex(-1, 5).?);
    try std.testing.expectEqual(0, absoluteIndex(-5, 5).?);
    try std.testing.expectEqual(0, absoluteIndex(0, 5).?);
    try std.testing.expectEqual(4, absoluteIndex(4, 5).?);
    try std.testing.expectEqual(null, absoluteIndex(5, 5));
    try std.testing.expectEqual(null, absoluteIndex(-6, 5));
}

test "Range toAbsolute" {
    const r1 = Range{ .start = 2, .end = 5, .step = 1 };
    const abs1 = r1.toAbsolute(10).?;
    try std.testing.expectEqual(.{ 2, 5, 1 }, abs1);

    const r2 = Range{ .start = -5, .end = null, .step = 2 };
    const abs2 = r2.toAbsolute(10).?;
    try std.testing.expectEqual(.{ 5, 10, 2 }, abs2);

    const r3 = Range{ .start = 0, .end = -1, .step = -1 };
    const abs3 = r3.toAbsolute(10).?;
    try std.testing.expectEqual(.{ 0, 9, -1 }, abs3);

    const r4 = Range{ .start = 11, .end = null, .step = 1 };
    try std.testing.expectEqual(null, r4.toAbsolute(10));

    const r5 = Range{ .start = 10, .end = null, .step = 1 };
    try std.testing.expectEqual(null, r5.toAbsolute(10));

    const r6 = Range{ .start = 0, .end = 12, .step = 1 };
    try std.testing.expectEqual(null, r6.toAbsolute(10));

    const r7 = Range{ .start = 0, .end = 5, .step = 0 };
    try std.testing.expectEqual(null, r7.toAbsolute(10));
}
