const std = @import("std");

pub const HeaderEncoding = enum {
    Ascii,
    Utf8,
};

const HeaderSizeType = enum {
    U16,
    U32,
};

pub const VersionProps = struct {
    header_size_type: HeaderSizeType,
    encoding: HeaderEncoding,
};

const NpyFileReadError = error{
    InvalidFile,
    UnsupportedVersion,
    ReadError,
};

const NpyHeaderParseError = error{
    EndingNewlineMissing,
};

const MAGIC = "\x93NUMPY";

/// Parses the .npy header content.
/// Parameters:
/// - header_buffer: The buffer containing the header content.
/// - encoding: The encoding type of the header.
/// Returns:
/// - Parsed Python value on success, or an error on failure.
fn parseNpyHeader(header_buffer: []const u8, _: HeaderEncoding) !void {
    // Check for ending newline
    if (header_buffer.len == 0 or header_buffer[header_buffer.len - 1] != '\n') {
        return NpyHeaderParseError.EndingNewlineMissing;
    }
    // Trim newline and spaces right before it
    const trimmed_header = std.mem.trimRight(u8, header_buffer[0 .. header_buffer.len - 1], " ");
    std.debug.print("Trimmed Header: |{s}|\n", .{trimmed_header});
}

/// Reads and validates a .npy file header.
/// Paramters:
/// - file_reader: A file reader for the .npy file.
/// Returns:
/// - NpyFileReadError on failure.
pub fn readNpyFile(file_reader: *std.fs.File.Reader) NpyFileReadError!void {
    var byte_buffer: [8]u8 = undefined;
    file_reader.interface.readSliceAll(&byte_buffer) catch return NpyFileReadError.ReadError;

    if (std.mem.eql(u8, byte_buffer[0..6], MAGIC)) {
        std.debug.print("Valid .npy file detected.\n", .{});
    } else {
        std.debug.print("Invalid .npy file: Magic number mismatch.\n", .{});
        return NpyFileReadError.InvalidFile;
    }

    const major_version = byte_buffer[6];
    const minor_version = byte_buffer[7];

    const version_props: VersionProps = version: {
        if (minor_version != 0) {
            std.debug.print("Unsupported .npy version: {}.{}\n", .{ major_version, minor_version });
            return NpyFileReadError.UnsupportedVersion;
        }
        switch (major_version) {
            1 => break :version .{ .header_size_type = .U16, .encoding = .Ascii },
            2 => break :version .{ .header_size_type = .U32, .encoding = .Ascii },
            3 => break :version .{ .header_size_type = .U32, .encoding = .Utf8 },
            else => {
                std.debug.print("Unsupported .npy major version: {}\n", .{major_version});
                return NpyFileReadError.UnsupportedVersion;
            },
        }
    };

    std.debug.print("Version: {}.{}\n", .{ major_version, minor_version });
    std.debug.print("Version Props: {any}\n", .{version_props});

    // Read the header size in little-endian format
    const header_size: u32 = header_size: switch (version_props.header_size_type) {
        .U16 => {
            var size_buffer: [2]u8 = undefined;
            file_reader.interface.readSliceAll(&size_buffer) catch return NpyFileReadError.ReadError;
            break :header_size std.mem.readInt(u16, &size_buffer, .little);
        },
        .U32 => {
            var size_buffer: [4]u8 = undefined;
            file_reader.interface.readSliceAll(&size_buffer) catch return NpyFileReadError.ReadError;
            break :header_size std.mem.readInt(u32, &size_buffer, .little);
        },
    };

    std.debug.print("Header Size: 0x{x:0>2}\n", .{header_size});

    for (std.mem.asBytes(&header_size), 0..) |byte, i| {
        std.debug.print("Header Byte [{}]: 0x{x:0>2}\n", .{ i, byte });
    }

    // Now read the header content
    // Choose allocator based on header size
    var small_buffer: [1024]u8 = undefined;
    const header_buffer = header_buffer: {
        if (header_size > 1024) {
            // Use heap allocator for large headers
            break :header_buffer std.heap.page_allocator.alloc(u8, header_size) catch return NpyFileReadError.ReadError;
        } else {
            // Use stack buffer for small headers
            break :header_buffer small_buffer[0..header_size];
        }
    };
    defer if (header_size > 1024) std.heap.page_allocator.free(header_buffer);

    file_reader.interface.readSliceAll(header_buffer) catch return NpyFileReadError.ReadError;

    std.debug.print("Header Content: {s}\n", .{header_buffer});

    _ = parseNpyHeader(header_buffer, version_props.encoding) catch |e| {
        std.debug.print("Error parsing header: {any}\n", .{e});
        return NpyFileReadError.InvalidFile;
    };
}
