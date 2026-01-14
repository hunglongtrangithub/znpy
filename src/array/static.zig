const std = @import("std");

const header_mod = @import("../header.zig");
const shape_mod = @import("../shape.zig");
const elements_mod = @import("../elements.zig");
const array_mod = @import("../array.zig");
const view_mod = @import("./view.zig");
const slice_mod = @import("../slice.zig");
const pointer_mod = @import("../pointer.zig");

pub const FromFileBufferError = header_mod.ReadHeaderError || shape_mod.static.FromHeaderError || elements_mod.ViewDataError;

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
        shape.num_elements,
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
        /// The shape of the array (dimensions, strides, order, num_elements)
        shape: shape_mod.StaticShape(rank),
        /// The data buffer for memory management (allocation/deallocation)
        data_buffer: []T,

        const Self = @This();

        pub const InitError = shape_mod.StaticShape(rank).InitError || std.mem.Allocator.Error;

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
            const data_buffer = try allocator.alloc(T, shape.num_elements);

            return Self{
                .shape = shape,
                .data_buffer = data_buffer,
            };
        }

        /// Deinitialize the array, freeing the data buffer.
        pub fn deinit(self: Self, allocator: std.mem.Allocator) void {
            allocator.free(self.data_buffer);
        }

        /// Create a `ArrayView` from a numpy file buffer.
        /// The buffer must contain a valid numpy array file.
        pub fn fromFileBuffer(file_buffer: []u8, allocator: std.mem.Allocator) FromFileBufferError!Self {
            return arrayFromFileBuffer(T, rank, file_buffer, allocator);
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
        /// The shape of the array (dimensions, strides, order, num_elements)
        shape: shape_mod.StaticShape(rank),
        /// The data buffer for memory management (allocation/deallocation)
        data_buffer: []const T,

        const Self = @This();

        /// Create a `ConstStaticArrayView` from a numpy file buffer.
        /// The buffer must contain a valid numpy array file.
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
            .num_elements = 6,
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
            .num_elements = 24,
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
            .num_elements = 12,
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
            .num_elements = 5,
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
            .num_elements = 4,
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
            .num_elements = 4,
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
            .num_elements = 16,
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
            .num_elements = 6,
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
            .num_elements = 1,
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
            .num_elements = 5,
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
            .num_elements = 4,
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
            .num_elements = 4,
        },
        .data_buffer = &data,
    };

    const ptr = array.atUnchecked([_]usize{ 1, 1 });
    try std.testing.expectEqual(@as(i8, 40), ptr.*);

    // Verify return type is *const i8
    const ptr_type = @TypeOf(ptr);
    try std.testing.expect(ptr_type == *const i8);
}
