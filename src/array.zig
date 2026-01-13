//! Module for array types: static-ranked vs dynamic-ranked arrays holding mutable vs immutable data.
//!
//! This module provides multidimensional arrays with support for:
//! - **Ownership**: Arrays own their data, views reference existing data
//! - **Mutability**: Both mutable and const (read-only) variants
//! - **Rank flexibility**: Static (compile-time) or dynamic (runtime) dimensionality
//!
//! ## Array Types (Own Data)
//! Array types allocate and manage contiguous memory buffers:
//! - `DynamicArray(T)` - Dynamically-ranked mutable array
//! - `ConstDynamicArray(T)` - Dynamically-ranked read-only array
//! - `StaticArray(T, N)` - Statically-ranked mutable array (N dimensions known at compile time)
//! - `ConstStaticArray(T, N)` - Statically-ranked read-only array
//!
//! ## View Types (Reference Data)
//! View types reference existing data without ownership. Views support striding and may
//! reference non-contiguous data (e.g., slices, transposed views):
//! - `ArrayView(T)` - Mutable view into array data
//! - `ConstArrayView(T)` - Read-only view into array data
//!
//! ## Const Variants
//! All const variants (`ConstArrayView`, `ConstDynamicArray`, `ConstStaticArray`) are designed
//! to view read-only data buffers, thus prevent accidental modification while maintaining full
//! access to array operations that don't require mutation.
//!
//! ## Memory Layout
//! - **Arrays**: Always store data in a single contiguous buffer
//! - **Views**: May reference non-contiguous data via strides, enabling zero-copy slicing
//!   and transposition
const std = @import("std");

const dynamic = @import("array/dynamic.zig");
const static = @import("array/static.zig");
const view = @import("array/view.zig");

pub const ArrayView = view.ArrayView;
pub const ConstArrayView = view.ConstArrayView;
pub const DynamicArray = dynamic.DynamicArray;
pub const ConstDynamicArray = dynamic.ConstDynamicArray;
pub const StaticArray = static.StaticArray;
pub const ConstStaticArray = static.ConstStaticArray;

/// Calculates a pointer offset from a base pointer using stride-based offset.
///
/// - `base_ptr` MUST point to "Logical Index 0" of the array
/// - `offset` MUST be a valid stride offset, such that the resulting pointer MUST be within the allocated buffer
/// - `offset` is in element strides, not bytes
///
/// Returns a single-item pointer to the element at the given stride offset.
/// The returned pointer type preserves constness from the base pointer type.
pub fn ptrFromOffset(comptime T: type, base_ptr: anytype, offset: isize) switch (@typeInfo(@TypeOf(base_ptr))) {
    .pointer => |ptr_info| if (ptr_info.is_const) *const T else *T,
    else => @compileError("base_ptr must be a pointer type"),
} {
    // 1. Get the base address as an integer
    const base_addr = @intFromPtr(base_ptr);

    // 2. Calculate the byte-level offset.
    // We multiply the logical offset by the size of the element.
    const byte_offset = offset * @as(isize, @intCast(@sizeOf(T)));

    // 3. Use wrapping addition to handle negative or positive offsets.
    // Bit-casting the signed isize to usize allows the CPU to use
    // two's-complement arithmetic to "jump" backwards or forwards.
    const target_addr = base_addr +% @as(usize, @bitCast(byte_offset));

    // 4. Return the resulting single-item pointer
    return @ptrFromInt(target_addr);
}

/// Returns a non-null, properly aligned, but "dangling" pointer for a given type.
///
/// **When to use this:**
/// 1. To initialize an `ArrayView` that has a size of 0 in one or more dimensions.
/// 2. As a sentinel value for empty allocations to avoid using `undefined` or `null`.
/// 3. When you need to satisfy Zig's requirement that a `[*]T` must be aligned to `@alignOf(T)`.
///
/// **Why this address?**
/// We use the type's alignment as its address (e.g., `0x4` for `u32`). This is:
/// - **Aligned:** By definition, the value `@alignOf(T)` is divisible by `@alignOf(T)`.
/// - **Non-Null:** Zig pointers cannot be `0`. Since alignment is always >= 1, this is safe.
/// - **Crash-Fast:** If your code accidentally tries to dereference this pointer,
///   the OS will immediately trigger a segfault because the address is in protected memory.
///
/// **Security Note:** This pointer must NEVER be dereferenced. It is a metadata
/// placeholder only.
pub fn danglingPtr(comptime T: type) [*]T {
    // We cast the alignment value (a power of 2) into the pointer type.
    // This is the idiomatic Zig equivalent to Rust's NonNull::dangling().
    return @ptrFromInt(@alignOf(T));
}

test {
    _ = dynamic;
    _ = static;
    _ = view;
}

test "ptrFromOffset - positive stride" {
    // Create a buffer with some i32 values
    var buffer = [_]i32{ 10, 20, 30, 40, 50 };
    const base_ptr: [*]i32 = &buffer;

    // Test positive offsets
    const ptr0 = ptrFromOffset(i32, base_ptr, 0);
    try std.testing.expectEqual(10, ptr0.*);

    const ptr1 = ptrFromOffset(i32, base_ptr, 1);
    try std.testing.expectEqual(20, ptr1.*);

    const ptr2 = ptrFromOffset(i32, base_ptr, 2);
    try std.testing.expectEqual(30, ptr2.*);

    const ptr4 = ptrFromOffset(i32, base_ptr, 4);
    try std.testing.expectEqual(50, ptr4.*);
}

test "ptrFromOffset - zero stride" {
    // Zero offset should return the same pointer
    var buffer = [_]f64{ 3.14, 2.71, 1.41 };
    const base_ptr: [*]f64 = &buffer;

    const ptr = ptrFromOffset(f64, base_ptr, 0);
    try std.testing.expectEqual(@intFromPtr(base_ptr), @intFromPtr(ptr));
    try std.testing.expectEqual(3.14, ptr.*);
}

test "ptrFromOffset - negative stride" {
    // Create a buffer and set base_ptr to the end for Fortran-style access
    var buffer = [_]i64{ 100, 200, 300, 400, 500 };

    // Point to the last element (index 4)
    const base_ptr: [*]i64 = @ptrCast(&buffer[4]);

    // Test negative offsets to access earlier elements
    const ptr0 = ptrFromOffset(i64, base_ptr, 0);
    try std.testing.expectEqual(500, ptr0.*);

    const ptr_neg1 = ptrFromOffset(i64, base_ptr, -1);
    try std.testing.expectEqual(400, ptr_neg1.*);

    const ptr_neg2 = ptrFromOffset(i64, base_ptr, -2);
    try std.testing.expectEqual(300, ptr_neg2.*);

    const ptr_neg4 = ptrFromOffset(i64, base_ptr, -4);
    try std.testing.expectEqual(100, ptr_neg4.*);
}

test "ptrFromOffset - const pointer" {
    // Test with const pointers to ensure type preservation
    const buffer = [_]u16{ 1, 2, 3, 4 };
    const base_ptr: [*]const u16 = &buffer;

    const ptr1 = ptrFromOffset(u16, base_ptr, 1);
    try std.testing.expectEqual(2, ptr1.*);

    // Verify the returned type is also const
    const ptr_type = @TypeOf(ptr1);
    try std.testing.expect(ptr_type == *const u16);
}

test "ptrFromOffset - mixed positive and negative from middle" {
    // Base pointer in the middle, test both directions
    var buffer = [_]i8{ 10, 20, 30, 40, 50, 60, 70 };

    // Point to middle element (index 3, value 40)
    const base_ptr: [*]i8 = @ptrCast(&buffer[3]);

    // Access from middle
    const ptr0 = ptrFromOffset(i8, base_ptr, 0);
    try std.testing.expectEqual(40, ptr0.*);

    // Positive direction
    const ptr_pos1 = ptrFromOffset(i8, base_ptr, 1);
    try std.testing.expectEqual(50, ptr_pos1.*);

    const ptr_pos3 = ptrFromOffset(i8, base_ptr, 3);
    try std.testing.expectEqual(70, ptr_pos3.*);

    // Negative direction
    const ptr_neg1 = ptrFromOffset(i8, base_ptr, -1);
    try std.testing.expectEqual(30, ptr_neg1.*);

    const ptr_neg3 = ptrFromOffset(i8, base_ptr, -3);
    try std.testing.expectEqual(10, ptr_neg3.*);
}
