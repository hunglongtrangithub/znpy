const std = @import("std");

const npy_header = @import("header.zig");
const dimension = @import("dimension.zig");
const elements = @import("elements.zig");

/// A view into a multi-dimensional array with static/dynamic rank.
/// The view does not own the underlying data buffer.
///
/// `T` is the element type.
/// `Rank` is the number of dimensions, or `null` for dynamic rank.
/// `mutable` indicates whether the view allows mutable access to the data.
pub fn ArrayView(comptime T: type, comptime Rank: ?usize, comptime mutable: bool) type {
    return struct {
        /// The dimensions of the array.
        dims: if (Rank) |R| [R]usize else []const usize,
        /// The strides of the array. Always the same length as `dims`.
        strides: if (Rank) |R| [R]isize else []const isize,
        /// This pointer always points to "Logical Index 0" of the array.
        data_ptr: if (mutable) [*]T else [*]const T,

        const Self = @This();

        pub const IndexType = if (Rank) |R| [R]usize else []const usize;

        pub const FromFileBufferError = npy_header.ReadHeaderError || dimension.Shape(Rank).FromHeaderError || elements.ViewDataError;

        pub fn fromFileBuffer(
            // If mutable is true, we REQUIRE []u8 (writeable).
            // If mutable is false, we accept []const u8 (read-only).
            file_buffer: if (mutable) []u8 else []const u8,
            allocator: std.mem.Allocator,
        ) FromFileBufferError!Self {
            var slice_reader = npy_header.SliceReader.init(file_buffer);

            // We don't need to defer header.deinit here since we need header.shape to be stored in the Array struct
            const header = try npy_header.Header.fromSliceReader(&slice_reader, allocator);
            errdefer header.deinit(allocator);

            const byte_buffer = file_buffer[slice_reader.pos..];
            const shape, const num_elements = try dimension.Shape(Rank).fromHeader(header);

            const data_buffer = try elements.Element(T).bytesAsSlice(
                mutable,
                byte_buffer,
                num_elements,
                header.descr,
            );

            const strides = if (Rank) |_|
                // Static rank. Strides is an array on the stack.
                shape.getStrides({})
            else
                // Dynamic rank. Strides is a slice allocated on the heap.
                try shape.getStrides(allocator);

            return Self{
                // shape.dims (if dynamic) is allocated by the allocator, so we can just store the pointer here
                .dims = shape.dims,
                .strides = strides,
                .data_ptr = data_buffer.ptr,
            };
        }

        /// Deinitialize the `ArrayView`, freeing any allocated resources.
        /// Only needed for dynamic rank arrays.
        pub fn deinit(self: Self, allocator: std.mem.Allocator) void {
            if (comptime Rank == null) {
                allocator.free(self.strides);
                allocator.free(self.dims);
            }
        }

        /// Compute the flat array offset for a given multi-dimensional index.
        /// Returns:
        ///   - The computed offset as an isize if the index is valid
        ///   - null if the index is invalid (wrong number of dimensions or out of bounds)
        fn strideOffset(self: *const Self, index: IndexType) ?isize {
            if (Rank == null) {
                // (Dynamic rank) Dimension mismatch
                if (index.len != self.dims.len) return null;
            }

            var offset: isize = 0;

            if (comptime Rank) |R| {
                // Static rank
                inline for (0..R) |i| {
                    if (index[i] >= self.dims[i]) {
                        // Index out of bounds
                        return null;
                    }
                    // SAFETY: This cast is safe due to the bounds check above (dim fits in isize and idx < dim)
                    offset += @as(isize, @intCast(index[i])) * self.strides[i];
                }
                return offset;
            } else {
                // Dynamic rank
                for (index, self.dims, self.strides) |idx, dim, stride| {
                    if (idx >= dim) {
                        // Index out of bounds
                        return null;
                    }
                    // SAFETY: This cast is safe due to the bounds check above (dim fits in isize and idx < dim)
                    const idx_isize: isize = @intCast(idx);
                    offset += idx_isize * stride;
                }
            }

            return offset;
        }

        /// Get a pointer to the element at the given multi-dimensional index.
        pub fn at(self: *const Self, index: IndexType) ?(if (mutable) *T else *const T) {
            const offset = self.strideOffset(index) orelse return null;

            // 1. Get the base address as an integer
            const base_addr = @intFromPtr(self.data_ptr);

            // 2. Calculate the byte-level offset.
            // We multiply the logical offset by the size of the element.
            const byte_offset = offset * @as(isize, @intCast(@sizeOf(T)));

            // 3. Use wrapping addition to handle negative or positive offsets.
            // Bit-casting the signed isize to usize allows the CPU to use
            // two's-complement arithmetic to "jump" backwards or forwards.
            const target_addr = base_addr +% @as(usize, @bitCast(byte_offset));

            // 4. Return the resulting pointer
            return @ptrFromInt(target_addr);
        }

        /// Set the element at the given multi-dimensional index.
        /// Only available if the view was initialized with mutable = true.
        pub fn set(self: *const Self, index: IndexType, value: T) ?void {
            if (comptime !mutable) {
                @compileError("Cannot call set on an immutable ArrayView. Use mutable = true.");
            }
            const ptr = self.at(index) orelse return null;
            ptr.* = value;
        }
    };
}

fn StaticArrayView(comptime T: type, comptime Rank: usize) type {
    return ArrayView(T, Rank, true);
}

test "StaticArrayView(f64, 2) - basic 2D array" {
    const allocator = std.testing.allocator;

    // Create a simple 2x3 f64 array
    const StaticView2D = StaticArrayView(f64, 2);

    var data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const view = StaticView2D{
        .dims = [_]usize{ 2, 3 },
        .strides = [_]isize{ 3, 1 }, // C-order strides
        .data_ptr = &data,
    };

    // Test element access
    try std.testing.expectEqual(@as(f64, 1.0), view.at([_]usize{ 0, 0 }).?.*);
    try std.testing.expectEqual(@as(f64, 2.0), view.at([_]usize{ 0, 1 }).?.*);
    try std.testing.expectEqual(@as(f64, 3.0), view.at([_]usize{ 0, 2 }).?.*);
    try std.testing.expectEqual(@as(f64, 4.0), view.at([_]usize{ 1, 0 }).?.*);
    try std.testing.expectEqual(@as(f64, 5.0), view.at([_]usize{ 1, 1 }).?.*);
    try std.testing.expectEqual(@as(f64, 6.0), view.at([_]usize{ 1, 2 }).?.*);

    _ = allocator;
}

test "StaticArrayView(i32, 3) - 3D array C order" {
    const StaticView3D = StaticArrayView(i32, 3);

    // 2x3x4 array
    var data = [_]i32{0} ** 24;
    for (0..24) |i| {
        data[i] = @intCast(i);
    }

    const view = StaticView3D{
        .dims = [_]usize{ 2, 3, 4 },
        .strides = [_]isize{ 12, 4, 1 }, // C-order: (3*4, 4, 1)
        .data_ptr = &data,
    };

    // Test a few elements
    try std.testing.expectEqual(@as(i32, 0), view.at([_]usize{ 0, 0, 0 }).?.*);
    try std.testing.expectEqual(@as(i32, 1), view.at([_]usize{ 0, 0, 1 }).?.*);
    try std.testing.expectEqual(@as(i32, 4), view.at([_]usize{ 0, 1, 0 }).?.*);
    try std.testing.expectEqual(@as(i32, 12), view.at([_]usize{ 1, 0, 0 }).?.*);
    try std.testing.expectEqual(@as(i32, 23), view.at([_]usize{ 1, 2, 3 }).?.*);
}

test "StaticArrayView(f32, 2) - Fortran order strides" {
    const StaticView2D = StaticArrayView(f32, 2);

    // 3x4 array in Fortran order
    var data = [_]f32{0} ** 12;
    for (0..12) |i| {
        data[i] = @floatFromInt(i);
    }

    const view = StaticView2D{
        .dims = [_]usize{ 3, 4 },
        .strides = [_]isize{ 1, 3 }, // F-order: (1, 3)
        .data_ptr = &data,
    };

    // In Fortran order, data is column-major
    try std.testing.expectEqual(@as(f32, 0.0), view.at([_]usize{ 0, 0 }).?.*);
    try std.testing.expectEqual(@as(f32, 1.0), view.at([_]usize{ 1, 0 }).?.*);
    try std.testing.expectEqual(@as(f32, 2.0), view.at([_]usize{ 2, 0 }).?.*);
    try std.testing.expectEqual(@as(f32, 3.0), view.at([_]usize{ 0, 1 }).?.*);
}

test "StaticArrayView(i32, 1) - 1D array" {
    const StaticView1D = StaticArrayView(i32, 1);

    var data = [_]i32{ 10, 20, 30, 40, 50 };
    const view = StaticView1D{
        .dims = [_]usize{5},
        .strides = [_]isize{1},
        .data_ptr = &data,
    };

    try std.testing.expectEqual(@as(i32, 10), view.at([_]usize{0}).?.*);
    try std.testing.expectEqual(@as(i32, 20), view.at([_]usize{1}).?.*);
    try std.testing.expectEqual(@as(i32, 30), view.at([_]usize{2}).?.*);
    try std.testing.expectEqual(@as(i32, 40), view.at([_]usize{3}).?.*);
    try std.testing.expectEqual(@as(i32, 50), view.at([_]usize{4}).?.*);
}

test "StaticArrayView(f64, 2) - out of bounds access" {
    const StaticView2D = StaticArrayView(f64, 2);

    var data = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    const view = StaticView2D{
        .dims = [_]usize{ 2, 2 },
        .strides = [_]isize{ 2, 1 },
        .data_ptr = &data,
    };

    // Valid access
    try std.testing.expect(view.at([_]usize{ 0, 0 }) != null);
    try std.testing.expect(view.at([_]usize{ 1, 1 }) != null);

    // Out of bounds access
    try std.testing.expectEqual(@as(?*f64, null), view.at([_]usize{ 2, 0 }));
    try std.testing.expectEqual(@as(?*f64, null), view.at([_]usize{ 0, 2 }));
    try std.testing.expectEqual(@as(?*f64, null), view.at([_]usize{ 2, 2 }));
}

test "StaticArrayView(bool, 2) - boolean array" {
    const StaticViewBool = StaticArrayView(bool, 2);

    var data = [_]bool{ true, false, false, true };
    const view = StaticViewBool{
        .dims = [_]usize{ 2, 2 },
        .strides = [_]isize{ 2, 1 },
        .data_ptr = &data,
    };

    try std.testing.expectEqual(true, view.at([_]usize{ 0, 0 }).?.*);
    try std.testing.expectEqual(false, view.at([_]usize{ 0, 1 }).?.*);
    try std.testing.expectEqual(false, view.at([_]usize{ 1, 0 }).?.*);
    try std.testing.expectEqual(true, view.at([_]usize{ 1, 1 }).?.*);
}

test "StaticArrayView(i32, 4) - 4D array" {
    const StaticView4D = StaticArrayView(i32, 4);

    // 2x2x2x2 array
    var data = [_]i32{0} ** 16;
    for (0..16) |i| {
        data[i] = @intCast(i);
    }

    const view = StaticView4D{
        .dims = [_]usize{ 2, 2, 2, 2 },
        .strides = [_]isize{ 8, 4, 2, 1 }, // C-order
        .data_ptr = &data,
    };

    try std.testing.expectEqual(@as(i32, 0), view.at([_]usize{ 0, 0, 0, 0 }).?.*);
    try std.testing.expectEqual(@as(i32, 1), view.at([_]usize{ 0, 0, 0, 1 }).?.*);
    try std.testing.expectEqual(@as(i32, 8), view.at([_]usize{ 1, 0, 0, 0 }).?.*);
    try std.testing.expectEqual(@as(i32, 15), view.at([_]usize{ 1, 1, 1, 1 }).?.*);
}

test "StaticArrayView(u8, 2) - modification through pointer" {
    const StaticView2D = StaticArrayView(u8, 2);

    var data = [_]u8{ 0, 1, 2, 3, 4, 5 };
    const view = StaticView2D{
        .dims = [_]usize{ 2, 3 },
        .strides = [_]isize{ 3, 1 },
        .data_ptr = &data,
    };

    // Modify through the view
    const ptr = view.at([_]usize{ 1, 1 }).?;
    try std.testing.expectEqual(@as(u8, 4), ptr.*);
    ptr.* = 99;

    // Verify modification
    try std.testing.expectEqual(@as(u8, 99), view.at([_]usize{ 1, 1 }).?.*);
    try std.testing.expectEqual(@as(u8, 99), data[4]);
}

test "StaticArrayView(i16, 3) - single element in each dimension" {
    const StaticView3D = StaticArrayView(i16, 3);

    var data = [_]i16{42};
    const view = StaticView3D{
        .dims = [_]usize{ 1, 1, 1 },
        .strides = [_]isize{ 1, 1, 1 },
        .data_ptr = &data,
    };

    try std.testing.expectEqual(@as(i16, 42), view.at([_]usize{ 0, 0, 0 }).?.*);

    // Any other index should be out of bounds
    try std.testing.expectEqual(@as(?*i16, null), view.at([_]usize{ 1, 0, 0 }));
    try std.testing.expectEqual(@as(?*i16, null), view.at([_]usize{ 0, 1, 0 }));
    try std.testing.expectEqual(@as(?*i16, null), view.at([_]usize{ 0, 0, 1 }));
}
