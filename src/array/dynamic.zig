const std = @import("std");

const header_mod = @import("../header.zig");
const shape_mod = @import("../shape.zig");
const elements_mod = @import("../elements.zig");

/// A view into a multi-dimensional array with dynamic rank.
/// The view does not own the underlying data buffer.
/// You can read and write elements through this view.
///
/// `T` is the element type.
pub fn DynamicArray(comptime T: type) type {
    const element_type = header_mod.ElementType.fromZigType(T) catch @compileError("Unsupported element type for DynamicArray");
    return struct {
        /// The dimensions of the array.
        dims: []const usize,
        /// The strides of the array. Always the same length as `dims`.
        strides: []const isize,
        /// This pointer always points to "Logical Index 0" of the array.
        data_ptr: [*]T,
        /// Number of elements in the array.
        // NOTE: Technically redundant (can be computed from dims), but storing this
        // will fill the padded space in the struct, and we may need it later.
        num_elements: usize,

        const Self = @This();

        pub const FromFileBufferError = header_mod.ReadHeaderError || shape_mod.DynamicShape.FromHeaderError || elements_mod.ViewDataError;

        pub const InitError = shape_mod.DynamicShape.InitError || std.mem.AllocError;

        /// Initialize a new `DynamicArray` with the given dimensions and order.
        /// A new data buffer will be allocated using the provided allocator.
        pub fn init(dims: []const usize, order: header_mod.Order, allocator: std.mem.Allocator) InitError!Self {
            const shape, const num_elements = try shape_mod.DynamicShape.init(
                dims,
                order,
                element_type,
            );
            const strides = try shape.getStrides();

            // Allocate the data buffer
            const data_buffer = try allocator.alloc(T, num_elements);

            return Self{
                .dims = dims,
                .strides = strides,
                .data_ptr = data_buffer.ptr,
                .num_elements = num_elements,
            };
        }

        // TODO: add deinit method to free the data buffer

        /// Create a `DynamicArrayView` from a numpy file buffer.
        /// The buffer must contain a valid numpy array file.
        pub fn fromFileBuffer(file_buffer: []u8, allocator: std.mem.Allocator) FromFileBufferError!Self {
            var slice_reader = header_mod.SliceReader.init(file_buffer);

            // We don't need to defer header.deinit here since we need header.shape (allocated by the allocator) to be stored in the Array struct
            const header = try header_mod.Header.fromSliceReader(&slice_reader, allocator);
            errdefer header.deinit(allocator);

            const byte_buffer = file_buffer[slice_reader.pos..];
            const shape, const num_elements = try shape_mod.DynamicShape.fromHeader(header);

            const data_buffer = try elements_mod.Element(T).bytesAsSlice(
                byte_buffer,
                num_elements,
                header.descr,
            );

            const strides = try shape.getStrides(allocator);

            return Self{
                .dims = shape.dims,
                .strides = strides,
                .data_ptr = data_buffer.ptr,
                .num_elements = num_elements,
            };
        }

        /// Deinitialize the `ArrayView`, freeing any allocated resources.
        /// Only needed for dynamic rank arrays.
        pub fn deinit(self: Self, allocator: std.mem.Allocator) void {
            allocator.free(self.strides);
            allocator.free(self.dims);
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

        /// Get a pointer to the element at the given multi-dimensional index.
        fn at(self: *const Self, index: []const usize) ?*T {
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
    };
}

/// A view into a multi-dimensional array with dynamic rank.
/// The view does not own the underlying data buffer.
/// You can only read elements through this view.
///
/// `T` is the element type.
pub fn ConstDynamicArray(comptime T: type) type {
    return struct {
        /// The dimensions of the array.
        dims: []const usize,
        /// The strides of the array. Always the same length as `dims`.
        strides: []const isize,
        /// This pointer always points to "Logical Index 0" of the array.
        data_ptr: [*]const T,
        /// Number of elements in the array.
        // NOTE: Technically redundant (can be computed from dims), but storing this
        // will fill the padded space in the struct, and we may need it later.
        num_elements: usize,

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
            const shape, const num_elements = try shape_mod.DynamicShape.fromHeader(header);

            const data_buffer = try elements_mod.Element(T).bytesAsSlice(
                byte_buffer,
                num_elements,
                header.descr,
            );

            const strides = try shape.getStrides(allocator);

            return Self{
                .dims = shape.dims,
                .strides = strides,
                .data_ptr = data_buffer.ptr,
                .num_elements = num_elements,
            };
        }

        /// Deinitialize the `ArrayView`, freeing any allocated resources.
        /// Only needed for dynamic rank arrays.
        pub fn deinit(self: Self, allocator: std.mem.Allocator) void {
            allocator.free(self.strides);
            allocator.free(self.dims);
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

        /// Get a const pointer to the element at the given multi-dimensional index.
        fn at(self: *const Self, index: []const usize) ?*const T {
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
    };
}
