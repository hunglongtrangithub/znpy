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

pub const FromFileBufferError = header_mod.ReadHeaderError || shape_mod.static.FromHeaderError || elements_mod.ViewDataError;
pub const FromFileReaderError = header_mod.ReadHeaderError || shape_mod.static.FromHeaderError || elements_mod.ReadDataError;

/// Generic function to create either a `StaticArray` or `ConstStaticArray` from a numpy file buffer,
/// depending on the mutability of the input buffer.
fn arrayFromFileBuffer(
    comptime T: type,
    comptime rank: usize,
    file_buffer: anytype,
    allocator: std.mem.Allocator,
) FromFileBufferError!if (pointer_mod.isConstPtr(@TypeOf(file_buffer)))
    ConstStaticArray(T, rank)
else
    StaticArray(T, rank) {
    var slice_reader = header_mod.SliceReader.init(file_buffer);

    const header = try header_mod.Header.fromSliceReader(&slice_reader, allocator);
    // We can defer here since the shape will hold its own copy of the dims slice it its struct
    defer header.deinit(allocator);

    const byte_buffer = file_buffer[slice_reader.pos..];
    const shape = try shape_mod.StaticShape(rank).fromHeader(header);

    const data_buffer = try elements_mod.Element(T).bytesAsSlice(
        byte_buffer,
        shape.numElements(),
        header.descr,
    );

    if (comptime pointer_mod.isConstPtr(@TypeOf(file_buffer))) {
        return ConstStaticArray(T, rank){
            .shape = shape,
            .data_buffer = data_buffer,
        };
    } else {
        return StaticArray(T, rank){
            .shape = shape,
            .data_buffer = data_buffer,
        };
    }
}

/// A multi-dimensional array with static rank.
/// The view does not own the underlying data buffer.
/// You can read and write elements through this view.
///
/// `T` is the element type.
/// `rank` is the number of dimensions.
pub fn StaticArray(comptime T: type, comptime rank: usize) type {
    // TODO: consider adding a check to reject ranks that are too large, or limit rank to u8?
    const element_type = elements_mod.ElementType.fromZigType(T) catch @compileError("Unsupported type for StaticArray");

    return struct {
        /// The shape of the array (dimensions, strides, order)
        shape: shape_mod.StaticShape(rank),
        /// The data buffer for memory management (allocation/deallocation)
        data_buffer: []T,

        const Self = @This();

        pub const InitError = shape_mod.static.InitError || std.mem.Allocator.Error;

        /// Initialize a new `StaticArray` with the given dimensions and order.
        /// A new data buffer will be allocated using the provided allocator.
        pub fn init(
            dims: [rank]usize,
            order: shape_mod.Order,
            allocator: std.mem.Allocator,
        ) InitError!Self {
            const shape = try shape_mod.StaticShape(rank).init(
                dims,
                order,
                element_type,
            );

            // Allocate the data buffer
            const data_buffer = try allocator.alloc(T, shape.numElements());

            return Self{
                .shape = shape,
                .data_buffer = data_buffer,
            };
        }

        /// Deinitialize the array, freeing the data buffer.
        pub fn deinit(self: Self, allocator: std.mem.Allocator) void {
            allocator.free(self.data_buffer);
        }

        /// Create a `StaticArray` from a numpy file buffer.
        /// The returned array borrows the buffer's data; no copy is made.
        /// Do not use `deinit` on the returned array, as it does not own the buffer.
        /// No need to free the shape, as it is owned by the array struct.
        pub fn fromFileBuffer(file_buffer: []u8, allocator: std.mem.Allocator) FromFileBufferError!Self {
            return arrayFromFileBuffer(T, rank, file_buffer, allocator);
        }

        /// Create a `StaticArray` by reading from a numpy file reader.
        /// The data buffer is allocated using the provided allocator.
        // To free the array data, call `deinit`.
        pub fn fromFileAlloc(file_reader: *std.io.Reader, allocator: std.mem.Allocator) FromFileReaderError!Self {
            const header = try header_mod.Header.fromReader(file_reader, allocator);
            defer header.deinit(allocator);
            const shape = try shape_mod.StaticShape(rank).fromHeader(header);

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
        ) (header_mod.WriteHeaderError || std.mem.Allocator.Error)!void {
            const header = header_mod.Header{
                // Force specified endianness to arch's native endian
                .descr = element_type.withEndian(native_endian),
                .order = self.shape.order,
                .shape = &self.shape.dims,
            };
            // Write header
            try header.writeAll(writer, allocator);
            // Write data buffer
            try writer.writeAll(std.mem.sliceAsBytes(self.data_buffer));
        }

        /// Create a view of this array.
        pub fn asView(self: *const Self) view_mod.ArrayView(T) {
            return .{
                .dims = &self.shape.dims,
                .strides = &self.shape.strides,
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
        pub fn at(self: *const Self, index: [rank]usize) ?*T {
            return self.asView().at(&index);
        }

        /// Get a pointer to the element at the given multi-dimensional index without bounds checking.
        ///
        /// SAFETY: The caller MUST ensure that all indices are within bounds.
        /// Undefined behavior if any `index[i] >= dims[i]`.
        ///
        /// This function skips all bounds checking for maximum performance.
        /// Use only when you have already validated the indices.
        pub fn atUnchecked(self: *const Self, index: [rank]usize) *T {
            return self.asView().atUnchecked(&index);
        }

        /// Get the element at the given multi-dimensional index.
        /// Returns null if the index is out of bounds.
        pub fn get(self: *const Self, index: [rank]usize) ?T {
            return self.asView().get(&index);
        }

        /// Set the element at the given multi-dimensional index.
        /// Panics if the index is out of bounds.
        pub fn set(self: *Self, index: [rank]usize, value: T) void {
            self.asView().set(&index, value);
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

/// A view into a multi-dimensional array with static rank.
/// The view does not own the underlying data buffer.
/// You can only read elements through this view.
///
/// `T` is the element type.
/// `rank` is the number of dimensions.
pub fn ConstStaticArray(comptime T: type, comptime rank: usize) type {
    // TODO: consider adding a check to reject ranks that are too large, or limit rank to u8?
    return struct {
        /// The shape of the array (dimensions, strides, order)
        shape: shape_mod.StaticShape(rank),
        /// The data buffer for memory management (allocation/deallocation)
        data_buffer: []const T,

        const Self = @This();

        /// Create a `ConstStaticArrayView` from a numpy file buffer.
        /// The buffer must contain a valid numpy array file.
        /// The returned array borrows the buffer's data; no copy is made.
        /// No need to free the shape, as it is owned by the array struct.
        pub fn fromFileBuffer(file_buffer: []const u8, allocator: std.mem.Allocator) FromFileBufferError!Self {
            return arrayFromFileBuffer(T, rank, file_buffer, allocator);
        }

        /// Create a const view of this array for indexing operations.
        pub fn asView(self: *const Self) view_mod.ConstArrayView(T) {
            return .{
                .dims = &self.shape.dims,
                .strides = &self.shape.strides,
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
        pub fn at(self: *const Self, index: [rank]usize) ?*const T {
            return self.asView().at(&index);
        }

        /// Get a const pointer to the element at the given multi-dimensional index without bounds checking.
        ///
        /// SAFETY: The caller MUST ensure that all indices are within bounds.
        /// Undefined behavior if any `index[i] >= dims[i]`.
        ///
        /// This function skips all bounds checking for maximum performance.
        /// Use only when you have already validated the indices.
        pub fn atUnchecked(self: *const Self, index: [rank]usize) *const T {
            return self.asView().atUnchecked(&index);
        }

        /// Get the element at the given multi-dimensional index.
        /// Returns null if the index is out of bounds.
        pub fn get(self: *const Self, index: [rank]usize) ?T {
            return self.asView().get(&index);
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

test "StaticArrayView(f64, 2) - basic 2D array" {
    // Create a simple 2x3 f64 array
    const StaticView2D = StaticArray(f64, 2);

    var data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const view = StaticView2D{
        .shape = shape_mod.StaticShape(2){
            .dims = [_]usize{ 2, 3 },
            .strides = [_]isize{ 3, 1 }, // C-order strides
            .order = .C,
        },
        .data_buffer = &data,
    };

    // Test element access
    try std.testing.expectEqual(@as(f64, 1.0), view.at([_]usize{ 0, 0 }).?.*);
    try std.testing.expectEqual(@as(f64, 2.0), view.at([_]usize{ 0, 1 }).?.*);
    try std.testing.expectEqual(@as(f64, 3.0), view.at([_]usize{ 0, 2 }).?.*);
    try std.testing.expectEqual(@as(f64, 4.0), view.at([_]usize{ 1, 0 }).?.*);
    try std.testing.expectEqual(@as(f64, 5.0), view.at([_]usize{ 1, 1 }).?.*);
    try std.testing.expectEqual(@as(f64, 6.0), view.at([_]usize{ 1, 2 }).?.*);
}

test "StaticArrayView(i32, 3) - 3D array C order" {
    const StaticView3D = StaticArray(i32, 3);

    // 2x3x4 array
    var data = [_]i32{0} ** 24;
    for (0..24) |i| {
        data[i] = @intCast(i);
    }

    const view = StaticView3D{
        .shape = shape_mod.StaticShape(3){
            .dims = [_]usize{ 2, 3, 4 },
            .strides = [_]isize{ 12, 4, 1 }, // C-order: (3*4, 4, 1)
            .order = .C,
        },
        .data_buffer = &data,
    };

    // Test a few elements
    try std.testing.expectEqual(@as(i32, 0), view.at([_]usize{ 0, 0, 0 }).?.*);
    try std.testing.expectEqual(@as(i32, 1), view.at([_]usize{ 0, 0, 1 }).?.*);
    try std.testing.expectEqual(@as(i32, 4), view.at([_]usize{ 0, 1, 0 }).?.*);
    try std.testing.expectEqual(@as(i32, 12), view.at([_]usize{ 1, 0, 0 }).?.*);
    try std.testing.expectEqual(@as(i32, 23), view.at([_]usize{ 1, 2, 3 }).?.*);
}

test "StaticArrayView(f32, 2) - Fortran order strides" {
    const StaticView2D = StaticArray(f32, 2);

    // 3x4 array in Fortran order
    var data = [_]f32{0} ** 12;
    for (0..12) |i| {
        data[i] = @floatFromInt(i);
    }

    const view = StaticView2D{
        .shape = shape_mod.StaticShape(2){
            .dims = [_]usize{ 3, 4 },
            .strides = [_]isize{ 1, 3 }, // F-order: (1, 3)
            .order = .F,
        },
        .data_buffer = &data,
    };

    // In Fortran order, data is column-major
    try std.testing.expectEqual(@as(f32, 0.0), view.at([_]usize{ 0, 0 }).?.*);
    try std.testing.expectEqual(@as(f32, 1.0), view.at([_]usize{ 1, 0 }).?.*);
    try std.testing.expectEqual(@as(f32, 2.0), view.at([_]usize{ 2, 0 }).?.*);
    try std.testing.expectEqual(@as(f32, 3.0), view.at([_]usize{ 0, 1 }).?.*);
}

test "StaticArrayView(i32, 1) - 1D array" {
    const StaticView1D = StaticArray(i32, 1);

    var data = [_]i32{ 10, 20, 30, 40, 50 };
    const view = StaticView1D{
        .shape = shape_mod.StaticShape(1){
            .dims = [_]usize{5},
            .strides = [_]isize{1},
            .order = .C,
        },
        .data_buffer = &data,
    };

    try std.testing.expectEqual(@as(i32, 10), view.at([_]usize{0}).?.*);
    try std.testing.expectEqual(@as(i32, 20), view.at([_]usize{1}).?.*);
    try std.testing.expectEqual(@as(i32, 30), view.at([_]usize{2}).?.*);
    try std.testing.expectEqual(@as(i32, 40), view.at([_]usize{3}).?.*);
    try std.testing.expectEqual(@as(i32, 50), view.at([_]usize{4}).?.*);
}

test "StaticArrayView(f64, 2) - out of bounds access" {
    const StaticView2D = StaticArray(f64, 2);

    var data = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    const view = StaticView2D{
        .shape = shape_mod.StaticShape(2){
            .dims = [_]usize{ 2, 2 },
            .strides = [_]isize{ 2, 1 },
            .order = .C,
        },
        .data_buffer = &data,
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
    const StaticViewBool = StaticArray(bool, 2);

    var data = [_]bool{ true, false, false, true };
    const view = StaticViewBool{
        .shape = shape_mod.StaticShape(2){
            .dims = [_]usize{ 2, 2 },
            .strides = [_]isize{ 2, 1 },
            .order = .C,
        },
        .data_buffer = &data,
    };

    try std.testing.expectEqual(true, view.at([_]usize{ 0, 0 }).?.*);
    try std.testing.expectEqual(false, view.at([_]usize{ 0, 1 }).?.*);
    try std.testing.expectEqual(false, view.at([_]usize{ 1, 0 }).?.*);
    try std.testing.expectEqual(true, view.at([_]usize{ 1, 1 }).?.*);
}

test "StaticArrayView(i32, 4) - 4D array" {
    const StaticView4D = StaticArray(i32, 4);

    // 2x2x2x2 array
    var data = [_]i32{0} ** 16;
    for (0..16) |i| {
        data[i] = @intCast(i);
    }

    const view = StaticView4D{
        .shape = shape_mod.StaticShape(4){
            .dims = [_]usize{ 2, 2, 2, 2 },
            .strides = [_]isize{ 8, 4, 2, 1 }, // C-order
            .order = .C,
        },
        .data_buffer = &data,
    };

    try std.testing.expectEqual(@as(i32, 0), view.at([_]usize{ 0, 0, 0, 0 }).?.*);
    try std.testing.expectEqual(@as(i32, 1), view.at([_]usize{ 0, 0, 0, 1 }).?.*);
    try std.testing.expectEqual(@as(i32, 8), view.at([_]usize{ 1, 0, 0, 0 }).?.*);
    try std.testing.expectEqual(@as(i32, 15), view.at([_]usize{ 1, 1, 1, 1 }).?.*);
}

test "StaticArrayView(u8, 2) - modification through pointer" {
    const StaticView2D = StaticArray(u8, 2);

    var data = [_]u8{ 0, 1, 2, 3, 4, 5 };
    const view = StaticView2D{
        .shape = shape_mod.StaticShape(2){
            .dims = [_]usize{ 2, 3 },
            .strides = [_]isize{ 3, 1 },
            .order = .C,
        },
        .data_buffer = &data,
    };

    // Modify through the view
    const ptr = view.at([_]usize{ 1, 1 }).?;
    try std.testing.expectEqual(4, ptr.*);
    ptr.* = 99;

    // Verify modification
    try std.testing.expectEqual(99, view.at([_]usize{ 1, 1 }).?.*);
    try std.testing.expectEqual(99, data[4]);
}

test "StaticArrayView(i16, 3) - single element in each dimension" {
    const StaticView3D = StaticArray(i16, 3);

    var data = [_]i16{42};
    const view = StaticView3D{
        .shape = shape_mod.StaticShape(3){
            .dims = [_]usize{ 1, 1, 1 },
            .strides = [_]isize{ 1, 1, 1 },
            .order = .C,
        },
        .data_buffer = &data,
    };

    try std.testing.expectEqual(@as(i16, 42), view.at([_]usize{ 0, 0, 0 }).?.*);

    // Any other index should be out of bounds
    try std.testing.expectEqual(@as(?*i16, null), view.at([_]usize{ 1, 0, 0 }));
    try std.testing.expectEqual(@as(?*i16, null), view.at([_]usize{ 0, 1, 0 }));
    try std.testing.expectEqual(@as(?*i16, null), view.at([_]usize{ 0, 0, 1 }));
}

test "StaticArray - atUnchecked() for performance" {
    const Array1D = StaticArray(i32, 1);

    var data = [_]i32{ 10, 20, 30, 40, 50 };
    const array = Array1D{
        .shape = shape_mod.StaticShape(1){
            .dims = [_]usize{5},
            .strides = [_]isize{1},
            .order = .C,
        },
        .data_buffer = &data,
    };

    // atUnchecked skips bounds checking
    const ptr0 = array.atUnchecked([_]usize{0});
    try std.testing.expectEqual(@as(i32, 10), ptr0.*);

    const ptr4 = array.atUnchecked([_]usize{4});
    try std.testing.expectEqual(@as(i32, 50), ptr4.*);

    // Can modify through the pointer
    ptr0.* = 100;
    try std.testing.expectEqual(@as(i32, 100), data[0]);
}

test "ConstStaticArray - at() returns const pointer" {
    const ConstArray2D = ConstStaticArray(u32, 2);

    const data = [_]u32{ 1, 2, 3, 4 };
    const array = ConstArray2D{
        .shape = shape_mod.StaticShape(2){
            .dims = [_]usize{ 2, 2 },
            .strides = [_]isize{ 2, 1 },
            .order = .C,
        },
        .data_buffer = &data,
    };

    // Verify at() returns const pointer
    const ptr = array.at([_]usize{ 0, 0 });
    try std.testing.expect(ptr != null);
    try std.testing.expectEqual(@as(u32, 1), ptr.?.*);

    // Verify return type is *const u32
    const ptr_type = @TypeOf(ptr.?);
    try std.testing.expect(ptr_type == *const u32);
}

test "ConstStaticArray - atUnchecked() returns const pointer" {
    const ConstArray2D = ConstStaticArray(i8, 2);

    const data = [_]i8{ 10, 20, 30, 40 };
    const array = ConstArray2D{
        .shape = shape_mod.StaticShape(2){
            .dims = [_]usize{ 2, 2 },
            .strides = [_]isize{ 2, 1 },
            .order = .C,
        },
        .data_buffer = &data,
    };

    const ptr = array.atUnchecked([_]usize{ 1, 1 });
    try std.testing.expectEqual(40, ptr.*);

    // Verify return type is *const i8
    const ptr_type = @TypeOf(ptr);
    try std.testing.expect(ptr_type == *const i8);
}

test "StaticArray.init" {
    const allocator = std.testing.allocator;

    // Test 1D array
    const Array1D = StaticArray(f64, 1);
    var array1d = try Array1D.init([_]usize{5}, .C, allocator);
    defer array1d.deinit(allocator);

    try std.testing.expectEqual(5, array1d.shape.numElements());
    try std.testing.expectEqualSlices(usize, &[_]usize{5}, &array1d.shape.dims);
    try std.testing.expect(array1d.shape.order == .C);
    try std.testing.expectEqual(5, array1d.data_buffer.len);

    // Test 2D array
    const Array2D = StaticArray(i32, 2);
    var array2d = try Array2D.init([_]usize{ 2, 3 }, .F, allocator);
    defer array2d.deinit(allocator);

    try std.testing.expectEqual(6, array2d.shape.numElements());
    try std.testing.expectEqualSlices(usize, &[_]usize{ 2, 3 }, &array2d.shape.dims);
    try std.testing.expect(array2d.shape.order == .F);
    try std.testing.expectEqual(6, array2d.data_buffer.len);

    // Test 3D array
    const Array3D = StaticArray(u8, 3);
    var array3d = try Array3D.init([_]usize{ 2, 2, 2 }, .C, allocator);
    defer array3d.deinit(allocator);

    try std.testing.expectEqual(8, array3d.shape.numElements());
    try std.testing.expectEqualSlices(usize, &[_]usize{ 2, 2, 2 }, &array3d.shape.dims);
    try std.testing.expect(array3d.shape.order == .C);
    try std.testing.expectEqual(8, array3d.data_buffer.len);
}

test "StaticArray.deinit" {
    const allocator = std.testing.allocator;

    // Create an array
    const Array = StaticArray(f64, 2);
    var array = try Array.init([_]usize{ 2, 3 }, .C, allocator);

    // Deinit should not crash
    array.deinit(allocator);

    // After deinit, we can't access the array safely, but the test passes if no crash occurs
}

test "StaticArray.get" {
    const allocator = std.testing.allocator;

    const Array = StaticArray(i32, 2);
    var array = try Array.init([_]usize{ 2, 3 }, .C, allocator);
    defer array.deinit(allocator);

    // Set some values
    array.set([_]usize{ 0, 0 }, 10);
    array.set([_]usize{ 0, 1 }, 20);
    array.set([_]usize{ 0, 2 }, 30);
    array.set([_]usize{ 1, 0 }, 40);
    array.set([_]usize{ 1, 1 }, 50);
    array.set([_]usize{ 1, 2 }, 60);

    // Test valid indices
    try std.testing.expectEqual(10, array.get([_]usize{ 0, 0 }));
    try std.testing.expectEqual(20, array.get([_]usize{ 0, 1 }));
    try std.testing.expectEqual(30, array.get([_]usize{ 0, 2 }));
    try std.testing.expectEqual(40, array.get([_]usize{ 1, 0 }));
    try std.testing.expectEqual(50, array.get([_]usize{ 1, 1 }));
    try std.testing.expectEqual(60, array.get([_]usize{ 1, 2 }));

    // Test out of bounds
    try std.testing.expectEqual(null, array.get([_]usize{ 2, 0 }));
    try std.testing.expectEqual(null, array.get([_]usize{ 0, 3 }));
    try std.testing.expectEqual(null, array.get([_]usize{ 1, 3 }));
}

test "StaticArray.set" {
    const allocator = std.testing.allocator;

    const Array = StaticArray(i32, 2);
    var array = try Array.init([_]usize{ 2, 3 }, .C, allocator);
    defer array.deinit(allocator);

    // Set values
    array.set([_]usize{ 0, 0 }, 100);
    array.set([_]usize{ 0, 1 }, 200);
    array.set([_]usize{ 0, 2 }, 300);
    array.set([_]usize{ 1, 0 }, 400);
    array.set([_]usize{ 1, 1 }, 500);
    array.set([_]usize{ 1, 2 }, 600);

    // Verify values were set
    try std.testing.expectEqual(100, array.data_buffer[0]);
    try std.testing.expectEqual(200, array.data_buffer[1]);
    try std.testing.expectEqual(300, array.data_buffer[2]);
    try std.testing.expectEqual(400, array.data_buffer[3]);
    try std.testing.expectEqual(500, array.data_buffer[4]);
    try std.testing.expectEqual(600, array.data_buffer[5]);
}
