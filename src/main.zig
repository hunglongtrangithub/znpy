const std = @import("std");
const znpy = @import("znpy");

pub fn main() !void {
    const npy_file_path = "test.npy";
    const file = std.fs.cwd().openFile(npy_file_path, .{ .mode = .read_only }) catch |e| {
        std.debug.print("Failed to open file: {}\n", .{e});
        return;
    };

    var file_buffer: [1024]u8 = undefined;
    var file_reader = file.reader(file_buffer[0..]);
    znpy.readNpyFile(&file_reader) catch |e| {
        std.debug.print("Error reading .npy file: {any}\n", .{e});
    };
}
