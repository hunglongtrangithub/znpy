const std = @import("std");

const npy_header = @import("header.zig");
const dimension = @import("dimension.zig");
const elements = @import("elements.zig");

const log = std.log.scoped(.npy_array);

/// A view into a multi-dimensional array. Does not own the underlying data buffer.
/// Caller can read and write to the data buffer through this view.
pub fn ArrayView(comptime T: type) type {
    return struct {
        /// The dimensions of the array.
        dims: []const usize,
        /// The strides of the array. Always the same length as `dims`.
        strides: []const isize,
        /// The underlying data buffer. Owned by the caller.
        data: []T,

        const Self = @This();

        const FromFileBufferError = npy_header.ReadHeaderError || dimension.Shape.FromHeaderError || elements.ViewDataError;

        pub fn fromFileBuffer(file_buffer: []u8, allocator: std.mem.Allocator) FromFileBufferError!Self {
            var slice_reader = npy_header.SliceReader.init(file_buffer);

            // We don't need to defer header.deinit here since we need header.shape to be stored in the Array struct
            const header = try npy_header.Header.fromSliceReader(&slice_reader, allocator);

            const byte_buffer = file_buffer[slice_reader.pos..];
            const shape: dimension.Shape, const num_elements: usize = try dimension.Shape.fromHeader(header);

            const data_buffer = try elements.Element(T).bytesAsSlice(
                byte_buffer,
                num_elements,
                header.descr,
            );

            const strides = try shape.getStrides(allocator);

            return Self{
                .dims = shape.dims,
                .strides = strides,
                .data = data_buffer,
            };
        }

        pub fn deinit(self: Self, allocator: std.mem.Allocator) void {
            allocator.free(self.strides);
            allocator.free(self.dims);
        }

        /// Compute the flat array offset for a given multi-dimensional index.
        /// Returns:
        ///   - The computed offset as an isize if the index is valid
        ///   - null if the index is invalid (wrong number of dimensions or out of bounds)
        fn strideOffset(self: *const Self, index: []const usize) ?isize {
            if (index.len != self.dims.len) {
                // Dimension mismatch
                return null;
            }
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
        pub fn at(self: *const Self, index: []const usize) ?*T {
            const offset = self.strideOffset(index) orelse return null;
            // SAFETY: offset is guaranteed to be valid due to the checks in strideOffset
            return &self.data[@intCast(offset)];
        }
    };
}
