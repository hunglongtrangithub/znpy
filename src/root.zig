//! By convention, root.zig is the root source file when making a library.
const std = @import("std");

const pointer = @import("pointer.zig");

pub const header = @import("header.zig");
pub const elements = @import("elements.zig");
pub const shape = @import("shape.zig");
pub const array = @import("array.zig");
pub const slice = @import("slice.zig");

pub const ElementType = elements.ElementType;
pub const Element = elements.Element;
pub const Order = shape.Order;
pub const Slice = slice.Slice;
pub const s = slice.format_slice;

test {
    _ = pointer;
    _ = header;
    _ = elements;
    _ = shape;
    _ = array;
    _ = slice;
}

fn testIO(
    comptime T: type,
    dims: []const usize,
    order: Order,
    allocator: std.mem.Allocator,
) !void {
    // Make the array
    const arr = try array.DynamicArray(T).init(
        dims,
        order,
        allocator,
    );
    defer arr.deinit(allocator);

    // Fill array with values
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();
    for (0..arr.data_buffer.len) |i| {
        arr.data_buffer[i] = ElementType.randomValue(T, random);
    }

    var file_buffer: [1024]u8 = undefined;
    const temp_file_path = "temp.npy";
    var temp_dir = std.testing.tmpDir(.{});
    defer temp_dir.cleanup();

    // Write to disk
    {
        const temp_file = try temp_dir.dir.createFile(temp_file_path, .{});
        defer temp_file.close();

        var file_writer = std.fs.File.Writer.init(temp_file, &file_buffer);
        try arr.writeAll(&file_writer.interface, allocator);
        try file_writer.interface.flush();
    }

    // Read from disk
    const temp_file = try temp_dir.dir.openFile(temp_file_path, .{});
    defer temp_file.close();

    var temp_file_reader = std.fs.File.Reader.init(temp_file, &file_buffer);
    const arr2 = try array.DynamicArray(T).fromFileAlloc(&temp_file_reader.interface, allocator);
    defer arr2.deinit(allocator);

    // Assertions
    try std.testing.expectEqualSlices(T, arr.data_buffer, arr2.data_buffer);
    try std.testing.expectEqualSlices(usize, arr.shape.dims, arr2.shape.dims);
    try std.testing.expectEqualSlices(isize, arr.shape.strides, arr2.shape.strides);
    try std.testing.expectEqual(arr.shape.order, arr2.shape.order);
}

test "File IO" {
    const allocator = std.testing.allocator;
    try testIO(i32, &[_]usize{ 10, 5 }, .C, allocator);
    try testIO(f32, &[_]usize{100}, .C, allocator);
    try testIO(i32, &[_]usize{}, .C, allocator);
    try testIO(u64, &[_]usize{ 2, 3, 4, 2 }, .C, allocator);
    try testIO(i32, &[_]usize{ 5, 5 }, .F, allocator);
    try testIO(i32, &[_]usize{ 2, 2, 2, 2, 2, 2, 2, 2 }, .C, allocator);
    try testIO(f32, &[_]usize{1}, .C, allocator);
    try testIO(u8, &[_]usize{ 1024, 1024, 10 }, .C, allocator);
    try testIO(f64, &[_]usize{ 3, 4, 5 }, .F, allocator);
    try testIO(i32, &[_]usize{ 10, 0, 5 }, .C, allocator);
}
