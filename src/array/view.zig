const std = @import("std");

const array_mod = @import("../array.zig");

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
        /// Pointer to "Logical Index 0" of the array
        data_ptr: [*]T,

        const Self = @This();

        /// Compute the flat array offset for a given multi-dimensional index.
        /// Returns:
        ///   - The computed offset as an isize if the index is valid
        ///   - null if the index is invalid (wrong number of dimensions or out of bounds)
        fn strideOffset(self: *const Self, index: []const usize) ?isize {
            // Dimension mismatch
            if (index.len != self.dims.len) return null;

            var offset: isize = 0;

            for (index, self.dims, self.strides) |idx, dim, stride| {
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
        /// Undefined behavior if any index[i] >= dims[i] or if index.len != dims.len.
        fn strideOffsetUnchecked(self: *const Self, index: []const usize) isize {
            var offset: isize = 0;
            for (index, self.strides) |idx, stride| {
                const idx_isize: isize = @intCast(idx);
                offset += idx_isize * stride;
            }
            return offset;
        }

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
            const offset = self.strideOffset(index) orelse return null;
            return array_mod.ptrFromOffset(T, self.data_ptr, offset);
        }

        /// Get a pointer to the element at the given multi-dimensional index without bounds checking.
        ///
        /// SAFETY: The caller MUST ensure that all indices are within bounds.
        /// Undefined behavior if any index[i] >= dims[i] or if index.len != dims.len.
        ///
        /// This function skips all bounds checking for maximum performance.
        /// Use only when you have already validated the indices.
        pub fn atUnchecked(self: *const Self, index: []const usize) *T {
            const offset = self.strideOffsetUnchecked(index);
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
    };
}

test ArrayView {
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
        /// Pointer to "Logical Index 0" of the array
        data_ptr: [*]const T,

        const Self = @This();

        /// Compute the flat array offset for a given multi-dimensional index.
        /// Returns:
        ///   - The computed offset as an isize if the index is valid
        ///   - null if the index is invalid (wrong number of dimensions or out of bounds)
        fn strideOffset(self: *const Self, index: []const usize) ?isize {
            // Dimension mismatch
            if (index.len != self.dims.len) return null;

            var offset: isize = 0;

            for (index, self.dims, self.strides) |idx, dim, stride| {
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
        /// Undefined behavior if any index[i] >= dims[i] or if index.len != dims.len.
        fn strideOffsetUnchecked(self: *const Self, index: []const usize) isize {
            var offset: isize = 0;
            for (index, self.strides) |idx, stride| {
                const idx_isize: isize = @intCast(idx);
                offset += idx_isize * stride;
            }
            return offset;
        }

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
            const offset = self.strideOffset(index) orelse return null;
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
            const offset = self.strideOffsetUnchecked(index);
            return array_mod.ptrFromOffset(T, self.data_ptr, offset);
        }

        /// Get the element at the given multi-dimensional index.
        /// Returns null if the index is out of bounds.
        pub fn get(self: *const Self, index: []const usize) ?T {
            const ptr = self.at(index) orelse return null;
            return ptr.*;
        }
    };
}

test ConstArrayView {
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
