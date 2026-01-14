const std = @import("std");

const znpy = @import("znpy");

const dirname = std.fs.path.dirname;
var stdout_buffer: [1024]u8 = undefined;
var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
const stdout = &stdout_writer.interface;

pub fn main() !void {
    const source_dir = comptime dirname(@src().file) orelse "src";
    const npy_file_path = comptime source_dir ++ "/" ++ "../test-data/shapes/f32_2d_4x5.npy";
    try stdout.print("Loading NPY file from path: {s}\n", .{npy_file_path});
    try stdout.flush();

    var fallback = std.heap.stackFallback(2048, std.heap.page_allocator);
    const allocator = fallback.get();

    const file = std.fs.cwd().openFile(npy_file_path, .{ .mode = .read_only }) catch |e| {
        std.debug.print("Failed to open file: {}\n", .{e});
        return;
    };
    defer file.close();

    const Array = znpy.array.StaticArray(f32, 2);

    var read_buffer: [1024]u8 = undefined;
    var file_reader = std.fs.File.Reader.init(file, &read_buffer);
    const array = try Array.fromFileAlloc(&file_reader.interface, allocator);
    defer array.deinit(allocator);

    try stdout.print("Formatted array:\n{f}\n", .{array});

    // Write the array to a temporary file
    const temp_file_path = "temp.npy";
    const temp_file = try std.fs.cwd().createFile(temp_file_path, .{});
    defer temp_file.close();

    var write_buffer: [1024]u8 = undefined;
    var file_writer = std.fs.File.Writer.init(temp_file, &write_buffer);
    try array.writeAll(&file_writer.interface, allocator);

    try file_writer.interface.flush();

    try stdout.print("\nWritten array to temp file, reading back...\n", .{});
    try stdout.flush();

    // Read the array back from the file
    const temp_file_read = try std.fs.cwd().openFile(temp_file_path, .{ .mode = .read_only });
    defer temp_file_read.close();

    var temp_file_reader = std.fs.File.Reader.init(temp_file_read, &read_buffer);
    const array2 = try Array.fromFileAlloc(&temp_file_reader.interface, allocator);
    defer array2.deinit(allocator);

    try stdout.print("Read back array:\n{f}\n", .{array2});
    try stdout.flush();

    // Compare the two arrays
    const shapes_equal = std.mem.eql(usize, &array.shape.dims, &array2.shape.dims);
    const strides_equal = std.mem.eql(isize, &array.shape.strides, &array2.shape.strides);
    const orders_equal = array.shape.order == array2.shape.order;
    const num_elements_equal = array.shape.num_elements == array2.shape.num_elements;
    const data_equal = std.mem.eql(f32, array.data_buffer, array2.data_buffer);

    try stdout.print("\n=== Comparison Results ===\n", .{});
    try stdout.print("Shapes equal: {}\n", .{shapes_equal});
    try stdout.print("Strides equal: {}\n", .{strides_equal});
    try stdout.print("Orders equal: {}\n", .{orders_equal});
    try stdout.print("Num elements equal: {}\n", .{num_elements_equal});
    try stdout.print("Data equal: {}\n", .{data_equal});

    if (shapes_equal and strides_equal and orders_equal and num_elements_equal and data_equal) {
        try stdout.print("\n✓ All properties match! File I/O successful.\n", .{});
    } else {
        try stdout.print("\n✗ Properties do not match! File I/O failed.\n", .{});
    }
    try stdout.flush();

    // Clean up the temporary file
    std.fs.cwd().deleteFile(temp_file_path) catch {};

    return;
}
