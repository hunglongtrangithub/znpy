const std = @import("std");

const header_mod = @import("../header.zig");
const shape_mod = @import("../shape.zig");
const elements_mod = @import("../elements.zig");
const array_mod = @import("../array.zig");

/// A multi-dimensional array with dynamic rank.
/// The view does not own the underlying data buffer.
/// You can read and write elements through this view.
///
/// `T` is the element type.
pub fn DynamicArray(comptime T: type) type {
    const element_type = header_mod.ElementType.fromZigType(T) catch @compileError("Unsupported element type for DynamicArray");
    return struct {
        /// The shape of the array (dimensions, strides, order, num_elements)
        shape: shape_mod.DynamicShape,
        /// The data buffer for memory management (allocation/deallocation)
        data_buffer: []T,
        /// Pointer to "Logical Index 0" of the array (may differ from data_buffer.ptr for negative strides)
        data_ptr: [*]T,

        const Self = @This();

        pub const FromFileBufferError = header_mod.ReadHeaderError || shape_mod.DynamicShape.FromHeaderError || elements_mod.ViewDataError;

        pub const InitError = shape_mod.DynamicShape.InitError || std.mem.AllocError;

        /// Initialize a new `DynamicArray` with the given dimensions and order.
        /// A new data buffer will be allocated using the provided allocator.
        pub fn init(
            dims: []const usize,
            order: header_mod.Order,
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
                .data_ptr = data_buffer.ptr,
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

        /// Set the element at the given multi-dimensional index.
        /// Panics if the index is out of bounds.
        pub fn set(self: *const Self, index: []const usize, value: T) void {
            const ptr = self.at(index).?;
            ptr.* = value;
        }

        /// Get the element at the given multi-dimensional index.
        /// Returns null if the index is out of bounds.
        pub fn get(self: *const Self, index: []const usize) ?T {
            const ptr = self.at(index) orelse return null;
            return ptr.*;
        }

        /// Compute the flat array offset for a given multi-dimensional index.
        /// Returns:
        ///   - The computed offset as an isize if the index is valid
        ///   - null if the index is invalid (wrong number of dimensions or out of bounds)
        fn strideOffset(self: *const Self, index: []const usize) ?isize {
            // Dimension mismatch
            if (index.len != self.shape.dims.len) return null;

            var offset: isize = 0;

            for (index, self.shape.dims, self.shape.strides) |idx, dim, stride| {
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

        /// Get a pointer to the element at the given multi-dimensional index.
        /// Returns null when index is out of bounds.
        fn at(self: *const Self, index: []const usize) ?*T {
            const offset = self.strideOffset(index) orelse return null;
            return array_mod.ptrFromOffset(T, self.data_ptr, offset);
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
        /// Pointer to "Logical Index 0" of the array (may differ from data_buffer.ptr for negative strides)
        data_ptr: [*]const T,

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
                .data_ptr = data_buffer.ptr,
            };
        }

        /// Deallocate the array by deallocating the shape data
        pub fn deinit(self: Self, allocator: std.mem.Allocator) void {
            self.shape.deinit(allocator);
        }

        /// Get the element at the given multi-dimensional index.
        /// Returns null if the index is out of bounds.
        pub fn get(self: *const Self, index: []const usize) ?T {
            const ptr = self.at(index) orelse return null;
            return ptr.*;
        }

        /// Compute the flat array offset for a given multi-dimensional index.
        /// Returns:
        ///   - The computed offset as an isize if the index is valid
        ///   - null if the index is invalid (wrong number of dimensions or out of bounds)
        fn strideOffset(self: *const Self, index: []const usize) ?isize {
            // Dimension mismatch
            if (index.len != self.shape.dims.len) return null;

            var offset: isize = 0;

            for (index, self.shape.dims, self.shape.strides) |idx, dim, stride| {
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

        /// Get a const pointer to the element at the given multi-dimensional index.
        /// Returns null when index is out of bounds.
        fn at(self: *const Self, index: []const usize) ?*const T {
            const offset = self.strideOffset(index) orelse return null;
            return array_mod.ptrFromOffset(T, self.data_ptr, offset);
        }
    };
}
