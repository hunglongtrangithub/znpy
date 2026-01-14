const std = @import("std");

/// Checks if every byte in the given slice is either 0 or 1 using SIMD operations for efficiency.
pub fn isAllZeroOrOne(bytes: []const u8) bool {
    const vector_size = std.simd.suggestVectorLength(u8) orelse {
        // Check without SIMD if scalars are recommended
        for (bytes) |b| {
            if (b > 1) {
                return false;
            }
        }
        return true;
    };

    const Vec = @Vector(vector_size, u8);

    // Initialize accumulator vector to all zeros
    var acc: Vec = @splat(@as(u8, 0));

    // Process in large chunks
    var i: usize = 0;
    while (i + vector_size <= bytes.len) : (i += vector_size) {
        const chunk: Vec = bytes[i..][0..vector_size].*;
        acc |= chunk; // Accumulate all set bits
    }

    // Check the remaining bytes (tail)
    var tail_acc: u8 = 0;
    for (bytes[i..]) |b| {
        tail_acc |= b;
    }

    // Reduce the vector to a single u8
    const final_acc = @reduce(.Or, acc) | tail_acc;

    // If any bit other than the last one is set, it's not 0 or 1
    return (final_acc & 0b1111_1110) == 0;
}

const testing = std.testing;

test "isAllZeroOrOne - empty slice returns true (vacuous)" {
    const empty: []const u8 = &[_]u8{};
    try testing.expect(isAllZeroOrOne(empty));
}

test "isAllZeroOrOne - all zeros or all ones" {
    const all_zeros = &[_]u8{ 0, 0, 0, 0, 0, 0, 0, 0 };
    try testing.expect(isAllZeroOrOne(all_zeros));
    const all_ones = &[_]u8{ 1, 1, 1, 1, 1, 1, 1, 1 };
    try testing.expect(isAllZeroOrOne(all_ones));
}

test "isAllZeroOrOne - mixed zeros and ones" {
    const mixed = &[_]u8{ 0, 1, 0, 1, 1, 0, 1, 0 };
    try testing.expect(isAllZeroOrOne(mixed));
}

test "isAllZeroOrOne - single valid byte" {
    const single_zero = &[_]u8{0};
    try testing.expect(isAllZeroOrOne(single_zero));
    const single_one = &[_]u8{1};
    try testing.expect(isAllZeroOrOne(single_one));
}

test "isAllZeroOrOne - single invalid byte" {
    const single_bad = &[_]u8{2};
    try testing.expect(!isAllZeroOrOne(single_bad));
}

test "isAllZeroOrOne - one bad byte" {
    const bad_at_start = &[_]u8{ 2, 0, 1, 0, 1, 0 };
    try testing.expect(!isAllZeroOrOne(bad_at_start));
    const bad_at_end = &[_]u8{ 0, 1, 0, 1, 0, 2 };
    try testing.expect(!isAllZeroOrOne(bad_at_end));
    const bad_in_middle = &[_]u8{ 0, 1, 0, 5, 1, 0 };
    try testing.expect(!isAllZeroOrOne(bad_in_middle));
    const bad_255 = &[_]u8{ 0, 1, 0, 255, 1, 0 };
    try testing.expect(!isAllZeroOrOne(bad_255));
}

test "isAllZeroOrOne - large array with bad bytes" {
    // At the start
    var large = [_]u8{0} ** 100;
    large[0] = 3;
    try testing.expect(!isAllZeroOrOne(&large));
    // At the end
    large[99] = 7;
    try testing.expect(!isAllZeroOrOne(&large));
    // In the middle
    large[50] = 255;
    try testing.expect(!isAllZeroOrOne(&large));

    // Multiple bad bytes
    const multiple_bad = &[_]u8{ 2, 1, 0, 3, 1, 4 };
    try testing.expect(!isAllZeroOrOne(multiple_bad));
}

test "isAllZeroOrOne - large valid array" {
    var large = [_]u8{0} ** 100;
    for (0..50) |i| {
        large[i] = 1;
    }
    try testing.expect(isAllZeroOrOne(&large));
}

test "isAllZeroOrOne - boundary case around vector size" {
    // Test sizes around typical SIMD vector sizes (16, 32, 64)
    const vector_size = std.simd.suggestVectorLength(u8) orelse 16;

    // Exactly vector_size elements, all valid
    const exact = try testing.allocator.alloc(u8, vector_size);
    defer testing.allocator.free(exact);
    @memset(exact, 1);
    try testing.expect(isAllZeroOrOne(exact));

    // Exactly vector_size + 1 elements with bad byte in tail
    var plus_one = try testing.allocator.alloc(u8, vector_size + 1);
    defer testing.allocator.free(plus_one);
    @memset(plus_one, 0);
    plus_one[vector_size] = 2;
    try testing.expect(!isAllZeroOrOne(plus_one));

    // Exactly vector_size - 1 elements with bad byte at end
    if (vector_size > 1) {
        var minus_one = try testing.allocator.alloc(u8, vector_size - 1);
        defer testing.allocator.free(minus_one);
        @memset(minus_one, 1);
        minus_one[vector_size - 2] = 8;
        try testing.expect(!isAllZeroOrOne(minus_one));
    }
}
