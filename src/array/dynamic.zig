const std = @import("std");

const header_mod = @import("../header.zig");
const shape_mod = @import("../shape.zig");
const elements_mod = @import("../elements.zig");
const array_mod = @import("../array.zig");
const view_mod = @import("./view.zig");
const slice_mod = @import("../slice.zig");

/// A multi-dimensional array with dynamic rank that owns its data.
/// This array owns the data buffer and will free it on deinit.
/// You can read and write elements through this array.
///
/// `T` is the element type.
pub fn DynamicArray(comptime T: type) type {
    const element_type = elements_mod.ElementType.fromZigType(T) catch @compileError("Unsupported element type for DynamicArray");
    return struct {
        /// The shape of the array (dimensions, strides, order, num_elements)
        shape: shape_mod.DynamicShape,
        /// The data buffer for memory management (allocation/deallocation)
        data_buffer: []T,

        const Self = @This();

        pub const FromFileBufferError = elements_mod.ReadHeaderError || shape_mod.DynamicShape.FromHeaderError || elements_mod.ViewDataError;

        pub const InitError = shape_mod.DynamicShape.InitError || std.mem.AllocError;

        /// Initialize a new `DynamicArray` with the given dimensions and order.
        /// A new data buffer will be allocated using the provided allocator.
        pub fn init(
            dims: []const usize,
            order: shape_mod.Order,
            allocator: std.mem.Allocator,
        ) InitError!Self {
            const shape = try shape_mod.DynamicShape.init(
                dims,
                order,
                element_type,
                allocator,
            );

            // Allocate the data buffer
            const data_buffer = try allocator.alloc(T, shape.num_elements);

            return Self{
                .shape = shape,
                .data_buffer = data_buffer,
            };
        }

        /// Deinitialize the array, freeing the data buffer and shape.
        pub fn deinit(self: Self, allocator: std.mem.Allocator) void {
            allocator.free(self.data_buffer);
            self.shape.deinit(allocator);
        }

        /// Create a `DynamicArrayView` from a numpy file buffer.
        /// The buffer must contain a valid numpy array file.
        pub fn fromFileBuffer(file_buffer: []u8, allocator: std.mem.Allocator) FromFileBufferError!Self {
            var slice_reader = header_mod.SliceReader.init(file_buffer);

            // We don't need to defer header.deinit here since we need header.shape (allocated by the allocator) to be stored in the Array struct
            const header = try header_mod.Header.fromSliceReader(&slice_reader, allocator);
            errdefer header.deinit(allocator);

            const byte_buffer = file_buffer[slice_reader.pos..];
            const shape = try shape_mod.DynamicShape.fromHeader(header, allocator);

            const data_buffer = try elements_mod.Element(T).bytesAsSlice(
                byte_buffer,
                shape.num_elements,
                header.descr,
            );

            return Self{
                .shape = shape,
                .data_buffer = data_buffer,
            };
        }

        /// Create a view of this array.
        pub fn asView(self: *const Self) view_mod.ArrayView(T) {
            return .{
                .dims = self.shape.dims,
                .strides = self.shape.strides,
                .data_ptr = self.data_buffer.ptr,
            };
        }

        /// Get a pointer to the element at the given multi-dimensional index.
        ///
        /// Returns:
        /// - A pointer to the element if the index is valid
        /// - `null` if the index is out of bounds
        ///
        /// SAFETY:
        /// - The returned pointer is only valid as long as the array exists
        /// - Do not store the pointer beyond the array's lifetime
        /// - Do not use the pointer after the array has been `deinit()`ed
        ///
        /// For safe value access, prefer `get()` and `set()` methods.
        pub fn at(self: *const Self, index: []const usize) ?*T {
            return self.asView().at(index);
        }

        /// Get a pointer to the element at the given multi-dimensional index without bounds checking.
        ///
        /// SAFETY: The caller MUST ensure that all indices are within bounds.
        /// Undefined behavior if any index[i] >= dims[i] or if index.len != dims.len.
        ///
        /// This function skips all bounds checking for maximum performance.
        /// Use only when you have already validated the indices.
        pub fn atUnchecked(self: *const Self, index: []const usize) *T {
            return self.asView().atUnchecked(index);
        }

        /// Get the element at the given multi-dimensional index.
        /// Returns null if the index is out of bounds.
        pub fn get(self: *const Self, index: []const usize) ?T {
            return self.asView().get(index);
        }

        /// Set the element at the given multi-dimensional index.
        /// Panics if the index is out of bounds.
        pub fn set(self: *const Self, index: []const usize, value: T) void {
            self.asView().set(index, value);
        }

        /// Create a sliced array view from this array.
        /// The returned view has the same mutability as the original.
        ///
        /// The caller owns the returned view's dims and strides arrays.
        pub fn slice(
            self: *const Self,
            slices: []const slice_mod.Slice,
            allocator: std.mem.Allocator,
        ) (slice_mod.SliceError || std.mem.Allocator.Error)!view_mod.ArrayView(T) {
            return try self.asView().slice(slices, allocator);
        }
    };
}

/// A view into a multi-dimensional array with dynamic rank.
/// The view does not own the underlying data buffer.
/// You can only read elements through this view.
///
/// `T` is the element type.
pub fn ConstDynamicArray(comptime T: type) type {
    return struct {
        /// The shape of the array (dimensions, strides, order, num_elements)
        shape: shape_mod.DynamicShape,
        /// The data buffer for memory management (allocation/deallocation)
        data_buffer: []const T,

        const Self = @This();

        pub const FromFileBufferError = header_mod.ReadHeaderError || shape_mod.DynamicShape.FromHeaderError || elements_mod.ViewDataError;

        /// Create a `DynamicArrayView` from a numpy file buffer.
        /// The buffer must contain a valid numpy array file.
        pub fn fromFileBuffer(file_buffer: []const u8, allocator: std.mem.Allocator) FromFileBufferError!Self {
            var slice_reader = header_mod.SliceReader.init(file_buffer);

            // We don't need to defer header.deinit here since we need header.shape (allocated by the allocator) to be stored in the Array struct
            const header = try header_mod.Header.fromSliceReader(&slice_reader, allocator);
            errdefer header.deinit(allocator);

            const byte_buffer = file_buffer[slice_reader.pos..];
            const shape = try shape_mod.DynamicShape.fromHeader(header, allocator);

            const data_buffer = try elements_mod.Element(T).bytesAsSlice(
                byte_buffer,
                shape.num_elements,
                header.descr,
            );

            return Self{
                .shape = shape,
                .data_buffer = data_buffer,
            };
        }

        /// Deallocate the array by deallocating the shape data
        pub fn deinit(self: Self, allocator: std.mem.Allocator) void {
            self.shape.deinit(allocator);
        }

        /// Create a const view of this array.
        fn asView(self: *const Self) view_mod.ConstArrayView(T) {
            return .{
                .dims = self.shape.dims,
                .strides = self.shape.strides,
                .data_ptr = self.data_buffer.ptr,
            };
        }

        /// Get a const pointer to the element at the given multi-dimensional index.
        ///
        /// Returns:
        /// - A pointer to the element if the index is valid
        /// - `null` if the index is out of bounds
        ///
        /// SAFETY:
        /// - The returned pointer is only valid as long as the array exists
        /// - Do not store the pointer beyond the array's lifetime
        /// - Do not use the pointer after the underlying buffer is freed
        ///
        /// For safe value access, prefer the `get()` method.
        pub fn at(self: *const Self, index: []const usize) ?*const T {
            return self.asView().at(index);
        }

        /// Get a const pointer to the element at the given multi-dimensional index without bounds checking.
        ///
        /// SAFETY: The caller MUST ensure that all indices are within bounds.
        /// Undefined behavior if any index[i] >= dims[i] or if index.len != dims.len.
        ///
        /// This function skips all bounds checking for maximum performance.
        /// Use only when you have already validated the indices.
        pub fn atUnchecked(self: *const Self, index: []const usize) *const T {
            return self.asView().atUnchecked(index);
        }

        /// Get the element at the given multi-dimensional index.
        /// Returns null if the index is out of bounds.
        pub fn get(self: *const Self, index: []const usize) ?T {
            return self.asView().get(index);
        }

        /// Create a sliced array view from this array.
        /// The returned view has the same mutability as the original.
        ///
        /// The caller owns the returned view's dims and strides arrays.
        pub fn slice(
            self: *const Self,
            slices: []const slice_mod.Slice,
            allocator: std.mem.Allocator,
        ) (slice_mod.SliceError || std.mem.Allocator.Error)!view_mod.ArrayView(T) {
            return try self.asView().slice(slices, allocator);
        }
    };
}

test "DynamicArray - public at() function" {
    const allocator = std.testing.allocator;
    const Array = DynamicArray(f64);

    var data = [_]f64{ 1.5, 2.5, 3.5, 4.5, 5.5, 6.5 };
    const dims = [_]usize{ 2, 3 };
    const shape = try shape_mod.DynamicShape.init(
        &dims,
        .C,
        elements_mod.ElementType{ .Float64 = null },
        allocator,
    );
    defer shape.deinit(allocator);

    const array = Array{
        .shape = shape,
        .data_buffer = &data,
    };

    // Test that at() is now public and returns correct pointers
    const ptr00 = array.at(&[_]usize{ 0, 0 });
    try std.testing.expect(ptr00 != null);
    try std.testing.expectEqual(@as(f64, 1.5), ptr00.?.*);

    const ptr12 = array.at(&[_]usize{ 1, 2 });
    try std.testing.expect(ptr12 != null);
    try std.testing.expectEqual(@as(f64, 6.5), ptr12.?.*);

    // Test bounds checking
    try std.testing.expectEqual(@as(?*f64, null), array.at(&[_]usize{ 2, 0 }));
    try std.testing.expectEqual(@as(?*f64, null), array.at(&[_]usize{ 0, 3 }));

    // Test dimension mismatch
    try std.testing.expectEqual(@as(?*f64, null), array.at(&[_]usize{0}));
}

test "DynamicArray - atUnchecked() for performance" {
    const allocator = std.testing.allocator;
    const Array = DynamicArray(i32);

    var data = [_]i32{ 10, 20, 30, 40, 50 };
    const dims = [_]usize{5};
    const shape = try shape_mod.DynamicShape.init(
        &dims,
        .C,
        elements_mod.ElementType{ .Int32 = null },
        allocator,
    );
    defer shape.deinit(allocator);

    const array = Array{
        .shape = shape,
        .data_buffer = &data,
    };

    // atUnchecked skips bounds checking
    const ptr0 = array.atUnchecked(&[_]usize{0});
    try std.testing.expectEqual(@as(i32, 10), ptr0.*);

    const ptr4 = array.atUnchecked(&[_]usize{4});
    try std.testing.expectEqual(@as(i32, 50), ptr4.*);

    // Can modify through the pointer
    ptr0.* = 100;
    try std.testing.expectEqual(@as(i32, 100), data[0]);
}

test "ConstDynamicArray - public at() returns const pointer" {
    const allocator = std.testing.allocator;
    const ConstArray = ConstDynamicArray(u32);

    const data = [_]u32{ 1, 2, 3, 4 };
    const dims = [_]usize{ 2, 2 };
    const shape = try shape_mod.DynamicShape.init(
        &dims,
        .C,
        elements_mod.ElementType{ .UInt32 = null },
        allocator,
    );
    defer shape.deinit(allocator);

    const array = ConstArray{
        .shape = shape,
        .data_buffer = &data,
    };

    // Verify at() returns const pointer
    const ptr = array.at(&[_]usize{ 0, 0 });
    try std.testing.expect(ptr != null);
    try std.testing.expectEqual(@as(u32, 1), ptr.?.*);

    // Verify return type is *const u32
    const ptr_type = @TypeOf(ptr.?);
    try std.testing.expect(ptr_type == *const u32);
}

test "ConstDynamicArray - atUnchecked() returns const pointer" {
    const allocator = std.testing.allocator;
    const ConstArray = ConstDynamicArray(i8);

    const data = [_]i8{ 10, 20, 30, 40 };
    const dims = [_]usize{ 2, 2 };
    const shape = try shape_mod.DynamicShape.init(
        &dims,
        .C,
        elements_mod.ElementType.Int8,
        allocator,
    );
    defer shape.deinit(allocator);

    const array = ConstArray{
        .shape = shape,
        .data_buffer = &data,
    };

    const ptr = array.atUnchecked(&[_]usize{ 1, 1 });
    try std.testing.expectEqual(@as(i8, 40), ptr.*);

    // Verify return type is *const i8
    const ptr_type = @TypeOf(ptr);
    try std.testing.expect(ptr_type == *const i8);
}
