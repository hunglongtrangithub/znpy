const std = @import("std");
const builtin = @import("builtin");

const header_mod = @import("../header.zig");
const shape_mod = @import("../shape.zig");
const elements_mod = @import("../elements.zig");
const array_mod = @import("../array.zig");
const view_mod = @import("./view.zig");
const slice_mod = @import("../slice.zig");
const pointer_mod = @import("../pointer.zig");

const native_endian = builtin.cpu.arch.endian();

pub const FromFileBufferError = header_mod.ReadHeaderError || shape_mod.DynamicShape.Error || elements_mod.ViewDataError;
pub const FromFileReaderError = header_mod.ReadHeaderError || shape_mod.DynamicShape.Error || elements_mod.ReadDataError;
pub const InitError = shape_mod.DynamicShape.Error || std.mem.Allocator.Error;

/// Generic function to create either a `StaticArray` or `ConstStaticArray` from a numpy file buffer,
/// depending on the mutability of the input buffer.
fn arrayFromFileBuffer(
    comptime T: type,
    file_buffer: anytype,
    allocator: std.mem.Allocator,
) FromFileBufferError!if (pointer_mod.isConstPtr(@TypeOf(file_buffer)))
    ConstDynamicArray(T)
else
    DynamicArray(T) {
    var slice_reader = header_mod.SliceReader.init(file_buffer);

    const header = try header_mod.Header.fromSliceReader(&slice_reader, allocator);
    // We don't defer here since the shape will hold the dims from the header
    errdefer header.deinit(allocator);

    const byte_buffer = file_buffer[slice_reader.pos..];
    const shape = try shape_mod.DynamicShape.fromHeader(header, allocator);

    const data_buffer = try elements_mod.Element(T).bytesAsSlice(
        byte_buffer,
        shape.numElements(),
        header.descr,
    );

    if (comptime pointer_mod.isConstPtr(@TypeOf(file_buffer))) {
        return ConstDynamicArray(T){
            .shape = shape,
            .data_buffer = data_buffer,
        };
    } else {
        return DynamicArray(T){
            .shape = shape,
            .data_buffer = data_buffer,
        };
    }
}

/// A multi-dimensional array with dynamic rank that owns its data.
/// This array owns the data buffer and will free it on deinit.
/// You can read and write elements through this array.
///
/// `T` is the element type.
pub fn DynamicArray(comptime T: type) type {
    const element_type = elements_mod.ElementType.fromZigType(T) catch @compileError("Unsupported element type for DynamicArray");
    return struct {
        /// The shape of the array (dimensions, strides, order)
        shape: shape_mod.DynamicShape,
        /// The data buffer for memory management (allocation/deallocation)
        data_buffer: []T,

        const Self = @This();

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
            const data_buffer = try allocator.alloc(T, shape.numElements());

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

        /// Free only the shape metadata, not the data buffer.
        /// Use this for arrays created with `fromFileBuffer`
        /// where the buffer is externally managed.
        pub fn deinitMetadata(self: Self, allocator: std.mem.Allocator) void {
            self.shape.deinit(allocator);
        }

        /// Create a `DynamicArray` from a numpy file buffer.
        /// The returned array borrows the buffer's data; no copy is made.
        /// Do not use `deinit` on the returned array, as it does not own the buffer,
        /// but make sure to call `deinitMetadata` to free the shape when done.
        pub fn fromFileBuffer(file_buffer: []u8, allocator: std.mem.Allocator) FromFileBufferError!Self {
            return arrayFromFileBuffer(T, file_buffer, allocator);
        }

        /// Create a `DynamicArray` by reading from a file reader in numpy file format.
        pub fn fromFileAlloc(file_reader: *std.io.Reader, allocator: std.mem.Allocator) FromFileReaderError!Self {
            const header = try header_mod.Header.fromReader(file_reader, allocator);
            // header's dims are owned by shape, so no need to defer deinit here
            const shape = try shape_mod.DynamicShape.fromHeader(header, allocator);

            // Allocate the data buffer and read data from the file
            const data_buffer = try allocator.alloc(T, shape.numElements());
            errdefer allocator.free(data_buffer);
            try elements_mod.Element(T).readSlice(
                data_buffer,
                file_reader,
                header.descr,
            );

            return Self{
                .shape = shape,
                .data_buffer = data_buffer,
            };
        }

        /// Write the array (both header and array data) to a writer in numpy file format.
        pub fn writeAll(
            self: *const Self,
            writer: *std.io.Writer,
            allocator: std.mem.Allocator,
        ) header_mod.WriteHeaderError!void {
            const header = header_mod.Header{
                // Force specified endianness to arch's native endian
                .descr = element_type.withEndian(native_endian),
                .order = self.shape.order,
                .shape = self.shape.dims,
            };
            // Write header
            try header.writeAll(writer, allocator);
            // Write data buffer
            try writer.writeAll(std.mem.sliceAsBytes(self.data_buffer));
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

        /// Format the array view using the default formatter.
        /// Intended to be used with `std.io.Writer.print`:
        /// ```zig
        /// var stdout_buffer: [1024]u8 = undefined;
        /// var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
        /// const stdout = &stdout_writer.interface;
        /// try stdout.print("Array:\n{f}\n", .{array});
        pub fn format(self: *const Self, writer: *std.io.Writer) std.io.Writer.Error!void {
            const view = self.asView().asConst();
            try view.format(writer);
        }
    };
}

/// A view into a multi-dimensional array with dynamic rank.
/// The view does not own the underlying data buffer.
/// You can only read elements through this view.
///
/// `T` is the element type.
pub fn ConstDynamicArray(comptime T: type) type {
    const element_type = elements_mod.ElementType.fromZigType(T) catch @compileError("Unsupported element type for ConstDynamicArray");
    return struct {
        /// The shape of the array (dimensions, strides, order)
        shape: shape_mod.DynamicShape,
        /// The data buffer for memory management (allocation/deallocation)
        data_buffer: []const T,

        const Self = @This();

        /// Initialize a new `ConstDynamicArray` with the given dimensions and order.
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
            const data_buffer = try allocator.alloc(T, shape.numElements());

            return Self{
                .shape = shape,
                .data_buffer = data_buffer,
            };
        }

        /// Create a `ConstDynamicArray` from a numpy file buffer.
        /// The returned array borrows the buffer's data; no copy is made.
        /// Do not use `deinit` on the returned array, as it does not own the buffer,
        /// but make sure to call `deinitMetadata` to free the shape when done.
        pub fn fromFileBuffer(file_buffer: []const u8, allocator: std.mem.Allocator) FromFileBufferError!Self {
            return arrayFromFileBuffer(T, file_buffer, allocator);
        }

        /// Free only the shape metadata, not the data buffer.
        /// Use this for arrays created with `fromFileBuffer`
        /// where the buffer is externally managed.
        pub fn deinitMetadata(self: Self, allocator: std.mem.Allocator) void {
            self.shape.deinit(allocator);
        }

        /// Deallocate the array by deallocating the shape data and the data buffer.
        pub fn deinit(self: Self, allocator: std.mem.Allocator) void {
            self.shape.deinit(allocator);
            allocator.free(self.data_buffer);
        }

        /// Create a const view of this array.
        pub fn asView(self: *const Self) view_mod.ConstArrayView(T) {
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
        ) (slice_mod.SliceError || std.mem.Allocator.Error)!view_mod.ConstArrayView(T) {
            return try self.asView().slice(slices, allocator);
        }

        /// Format the array view using the default formatter.
        /// Intended to be used with `std.io.Writer.print`:
        /// ```zig
        /// var stdout_buffer: [1024]u8 = undefined;
        /// var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
        /// const stdout = &stdout_writer.interface;
        /// try stdout.print("Array:\n{f}\n", .{array});
        pub fn format(self: *const Self, writer: *std.io.Writer) std.io.Writer.Error!void {
            const view = self.asView();
            try view.format(writer);
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
    try std.testing.expectEqual(1.5, ptr00.?.*);

    const ptr12 = array.at(&[_]usize{ 1, 2 });
    try std.testing.expect(ptr12 != null);
    try std.testing.expectEqual(6.5, ptr12.?.*);

    // Test bounds checking
    try std.testing.expectEqual(null, array.at(&[_]usize{ 2, 0 }));
    try std.testing.expectEqual(null, array.at(&[_]usize{ 0, 3 }));

    // Test dimension mismatch
    try std.testing.expectEqual(null, array.at(&[_]usize{0}));
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
    try std.testing.expectEqual(10, ptr0.*);

    const ptr4 = array.atUnchecked(&[_]usize{4});
    try std.testing.expectEqual(50, ptr4.*);

    // Can modify through the pointer
    ptr0.* = 100;
    try std.testing.expectEqual(100, data[0]);
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
    try std.testing.expectEqual(1, ptr.?.*);

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
    try std.testing.expectEqual(40, ptr.*);

    // Verify return type is *const i8
    const ptr_type = @TypeOf(ptr);
    try std.testing.expect(ptr_type == *const i8);
}

test "DynamicArray.init" {
    const allocator = std.testing.allocator;

    // Test 1D array
    const Array1D = DynamicArray(f64);
    const dims1d = [_]usize{5};
    var array1d = try Array1D.init(&dims1d, .C, allocator);
    defer array1d.deinit(allocator);

    try std.testing.expectEqual(5, array1d.shape.numElements());
    try std.testing.expectEqualSlices(usize, &dims1d, array1d.shape.dims);
    try std.testing.expect(array1d.shape.order == .C);
    try std.testing.expectEqual(5, array1d.data_buffer.len);

    // Test 2D array
    const Array2D = DynamicArray(i32);
    const dims2d = [_]usize{ 2, 3 };
    var array2d = try Array2D.init(&dims2d, .F, allocator);
    defer array2d.deinit(allocator);

    try std.testing.expectEqual(6, array2d.shape.numElements());
    try std.testing.expectEqualSlices(usize, &dims2d, array2d.shape.dims);
    try std.testing.expect(array2d.shape.order == .F);
    try std.testing.expectEqual(6, array2d.data_buffer.len);

    // Test 3D array
    const Array3D = DynamicArray(u8);
    const dims3d = [_]usize{ 2, 2, 2 };
    var array3d = try Array3D.init(&dims3d, .C, allocator);
    defer array3d.deinit(allocator);

    try std.testing.expectEqual(8, array3d.shape.numElements());
    try std.testing.expectEqualSlices(usize, &dims3d, array3d.shape.dims);
    try std.testing.expect(array3d.shape.order == .C);
    try std.testing.expectEqual(8, array3d.data_buffer.len);
}

test "ConstDynamicArray.init" {
    const allocator = std.testing.allocator;

    // Test 1D array
    const ConstArray1D = ConstDynamicArray(f64);
    const dims1d = [_]usize{5};
    var array1d = try ConstArray1D.init(&dims1d, .C, allocator);
    defer array1d.deinit(allocator);

    try std.testing.expectEqual(5, array1d.shape.numElements());
    try std.testing.expectEqualSlices(usize, &dims1d, array1d.shape.dims);
    try std.testing.expect(array1d.shape.order == .C);
    try std.testing.expectEqual(5, array1d.data_buffer.len);

    // Test 2D array
    const ConstArray2D = ConstDynamicArray(i32);
    const dims2d = [_]usize{ 2, 3 };
    var array2d = try ConstArray2D.init(&dims2d, .F, allocator);
    defer array2d.deinit(allocator);

    try std.testing.expectEqual(6, array2d.shape.numElements());
    try std.testing.expectEqualSlices(usize, &dims2d, array2d.shape.dims);
    try std.testing.expect(array2d.shape.order == .F);
    try std.testing.expectEqual(6, array2d.data_buffer.len);
}

test "DynamicArray.deinit" {
    const allocator = std.testing.allocator;

    // Create an array
    const Array = DynamicArray(f64);
    const dims = [_]usize{ 2, 3 };
    var array = try Array.init(&dims, .C, allocator);

    // Deinit should not crash
    array.deinit(allocator);
}

test "ConstDynamicArray.deinit" {
    const allocator = std.testing.allocator;

    // Create a const array using init
    const ConstArray = ConstDynamicArray(f64);
    const dims = [_]usize{ 2, 3 };
    var array = try ConstArray.init(&dims, .C, allocator);

    // Deinit should free both the data buffer and shape
    array.deinit(allocator);

    // After deinit, the array is freed, test passes if no crash
}

test "DynamicArray.deinitMetadata" {
    const allocator = std.testing.allocator;

    // Create an array with owned data
    const Array = DynamicArray(f64);
    const dims = [_]usize{ 2, 3 };
    var array = try Array.init(&dims, .C, allocator);

    // Set some data
    array.data_buffer[0] = 42.0;
    array.data_buffer[1] = 43.0;

    // Deinit metadata only - should free shape but not data buffer
    array.deinitMetadata(allocator);

    // Data buffer should still be accessible and contain the data
    try std.testing.expectEqual(42.0, array.data_buffer[0]);
    try std.testing.expectEqual(43.0, array.data_buffer[1]);

    // Now manually free the data buffer since it wasn't freed
    allocator.free(array.data_buffer);
}

test "ConstDynamicArray.deinitMetadata" {
    const allocator = std.testing.allocator;

    // Create a const array
    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const dims = [_]usize{ 2, 3 };
    const shape = try shape_mod.DynamicShape.init(
        &dims,
        .C,
        elements_mod.ElementType{ .Float64 = null },
        allocator,
    );

    const ConstArray = ConstDynamicArray(f64);
    var array = ConstArray{
        .shape = shape,
        .data_buffer = &data,
    };

    // Deinit metadata - should free shape but not touch data buffer
    array.deinitMetadata(allocator);

    // Data buffer should still be accessible
    try std.testing.expectEqual(1.0, data[0]);
    try std.testing.expectEqual(2.0, data[1]);
}

test "DynamicArray.get" {
    const allocator = std.testing.allocator;

    const Array = DynamicArray(i32);
    const dims = [_]usize{ 2, 3 };
    var array = try Array.init(&dims, .C, allocator);
    defer array.deinit(allocator);

    // Set some values
    array.data_buffer[0] = 10;
    array.data_buffer[1] = 20;
    array.data_buffer[2] = 30;
    array.data_buffer[3] = 40;
    array.data_buffer[4] = 50;
    array.data_buffer[5] = 60;

    // Test valid indices
    try std.testing.expectEqual(10, array.get(&[_]usize{ 0, 0 }));
    try std.testing.expectEqual(20, array.get(&[_]usize{ 0, 1 }));
    try std.testing.expectEqual(30, array.get(&[_]usize{ 0, 2 }));
    try std.testing.expectEqual(40, array.get(&[_]usize{ 1, 0 }));
    try std.testing.expectEqual(50, array.get(&[_]usize{ 1, 1 }));
    try std.testing.expectEqual(60, array.get(&[_]usize{ 1, 2 }));

    // Test out of bounds
    try std.testing.expectEqual(null, array.get(&[_]usize{ 2, 0 }));
    try std.testing.expectEqual(null, array.get(&[_]usize{ 0, 3 }));
    try std.testing.expectEqual(null, array.get(&[_]usize{ 1, 3 }));

    // Test wrong number of dimensions
    try std.testing.expectEqual(null, array.get(&[_]usize{0}));
    try std.testing.expectEqual(null, array.get(&[_]usize{ 0, 0, 0 }));
}

test "ConstDynamicArray.get" {
    const allocator = std.testing.allocator;

    const data = [_]i32{ 10, 20, 30, 40, 50, 60 };
    const dims = [_]usize{ 2, 3 };
    const shape = try shape_mod.DynamicShape.init(
        &dims,
        .C,
        elements_mod.ElementType{ .Int32 = null },
        allocator,
    );
    defer shape.deinit(allocator);

    const ConstArray = ConstDynamicArray(i32);
    const array = ConstArray{
        .shape = shape,
        .data_buffer = &data,
    };

    // Test valid indices
    try std.testing.expectEqual(10, array.get(&[_]usize{ 0, 0 }));
    try std.testing.expectEqual(20, array.get(&[_]usize{ 0, 1 }));
    try std.testing.expectEqual(30, array.get(&[_]usize{ 0, 2 }));
    try std.testing.expectEqual(40, array.get(&[_]usize{ 1, 0 }));
    try std.testing.expectEqual(50, array.get(&[_]usize{ 1, 1 }));
    try std.testing.expectEqual(60, array.get(&[_]usize{ 1, 2 }));

    // Test out of bounds
    try std.testing.expectEqual(null, array.get(&[_]usize{ 2, 0 }));
    try std.testing.expectEqual(null, array.get(&[_]usize{ 0, 3 }));
}

test "DynamicArray.set" {
    const allocator = std.testing.allocator;

    const Array = DynamicArray(i32);
    const dims = [_]usize{ 2, 3 };
    var array = try Array.init(&dims, .C, allocator);
    defer array.deinit(allocator);

    // Set values
    array.set(&[_]usize{ 0, 0 }, 100);
    array.set(&[_]usize{ 0, 1 }, 200);
    array.set(&[_]usize{ 0, 2 }, 300);
    array.set(&[_]usize{ 1, 0 }, 400);
    array.set(&[_]usize{ 1, 1 }, 500);
    array.set(&[_]usize{ 1, 2 }, 600);

    // Verify values were set
    try std.testing.expectEqual(100, array.data_buffer[0]);
    try std.testing.expectEqual(200, array.data_buffer[1]);
    try std.testing.expectEqual(300, array.data_buffer[2]);
    try std.testing.expectEqual(400, array.data_buffer[3]);
    try std.testing.expectEqual(500, array.data_buffer[4]);
    try std.testing.expectEqual(600, array.data_buffer[5]);
}

test "DynamicArray.slice" {
    const allocator = std.testing.allocator;

    const Array = DynamicArray(i32);
    const dims = [_]usize{ 2, 3 };
    var array = try Array.init(&dims, .C, allocator);
    defer array.deinit(allocator);

    // Set values: [[10, 20, 30], [40, 50, 60]]
    array.set(&[_]usize{ 0, 0 }, 10);
    array.set(&[_]usize{ 0, 1 }, 20);
    array.set(&[_]usize{ 0, 2 }, 30);
    array.set(&[_]usize{ 1, 0 }, 40);
    array.set(&[_]usize{ 1, 1 }, 50);
    array.set(&[_]usize{ 1, 2 }, 60);

    // Slice to get second column: [:, 1]
    const slices = [_]slice_mod.Slice{
        slice_mod.All,
        .{ .Index = 1 },
    };
    var sliced = try array.slice(&slices, allocator);
    defer sliced.deinit(allocator);

    // Should result in 1D array [20, 50]
    try std.testing.expectEqual(1, sliced.dims.len);
    try std.testing.expectEqual(2, sliced.dims[0]);
    try std.testing.expectEqual(20, sliced.get(&[_]usize{0}));
    try std.testing.expectEqual(50, sliced.get(&[_]usize{1}));
}

test "ConstDynamicArray.slice" {
    const allocator = std.testing.allocator;

    const data = [_]i32{ 10, 20, 30, 40, 50, 60 };
    const dims = [_]usize{ 2, 3 };
    const shape = try shape_mod.DynamicShape.init(
        &dims,
        .C,
        elements_mod.ElementType{ .Int32 = null },
        allocator,
    );
    defer shape.deinit(allocator);

    const ConstArray = ConstDynamicArray(i32);
    const array = ConstArray{
        .shape = shape,
        .data_buffer = &data,
    };

    // Slice to get first row: [0, :]
    const slices = [_]slice_mod.Slice{
        .{ .Index = 0 },
        slice_mod.All,
    };
    var sliced = try array.slice(&slices, allocator);
    defer sliced.deinit(allocator);

    // Should result in 1D array [10, 20, 30]
    try std.testing.expectEqual(1, sliced.dims.len);
    try std.testing.expectEqual(3, sliced.dims[0]);
    try std.testing.expectEqual(10, sliced.get(&[_]usize{0}));
    try std.testing.expectEqual(20, sliced.get(&[_]usize{1}));
    try std.testing.expectEqual(30, sliced.get(&[_]usize{2}));
}
