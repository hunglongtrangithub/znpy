const std = @import("std");
const znpy = @import("znpy");

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Load data from a file using DynamicArray
    const source_dir = comptime std.fs.path.dirname(@src().file) orelse "src";
    const npy_file_path = comptime source_dir ++ "/../test-data/shapes/f32_2d_4x5.npy";
    std.debug.print("Loading NPY file from path: {s}\n", .{npy_file_path});

    const file = std.fs.cwd().openFile(npy_file_path, .{ .mode = .read_only }) catch |e| {
        std.debug.print("Failed to open file: {}\n", .{e});
        return;
    };
    defer file.close();

    const Array = znpy.array.DynamicArray(f32);
    var read_buffer: [1024]u8 = undefined;
    var file_reader = std.fs.File.Reader.init(file, &read_buffer);
    const arr = try Array.fromFileAlloc(&file_reader.interface, allocator);
    defer arr.deinit(allocator);

    std.debug.print("Loaded array with shape: ", .{});
    for (arr.shape.dims) |dim| {
        std.debug.print("{} ", .{dim});
    }
    std.debug.print("\n", .{});

    // Get a value
    const val = arr.get(&[_]usize{ 1, 2 });
    std.debug.print("Value at [1,2]: {}\n", .{val.?});

    const view_val = arr.get(&[_]usize{ 0, 1 });
    std.debug.print("Value at [0,1]: {}\n", .{view_val.?});

    // Modify a value
    arr.set(&[_]usize{ 1, 1 }, 99.0);
    std.debug.print("Modified value at [1,1]: {}\n", .{arr.get(&[_]usize{ 1, 1 }).?});

    // Print the array
    std.debug.print("Mofified array:\n{f}\n", .{arr});
}
