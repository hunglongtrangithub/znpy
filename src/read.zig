const std = @import("std");
const log = std.log.scoped(.npy_reader);
const header = @import("header/root.zig");
const readNpyHeaderData = header.readNpyHeaderData;
const NpyHeaderData = header.NpyHeaderData;
const Parser = @import("header/parse.zig").Parser;

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
    /// The file is too short to be a valid .npy file.
    TooShort,
    /// Magic sequence does not match the expected value.
    MagicMismatch,
    /// The file does not have a valid header.
    InvalidHeader,
    /// The format version is unsupported.
    UnsupportedVersion,
    /// Error reading from file, due to I/O issues.
    IoError,
};

const NpyHeaderParseError = error{
    EndingNewlineMissing,
    InvalidHeaderFormat,
};

const MAGIC = "\x93NUMPY";

/// Processes the .npy header content into a structured format.
/// Parameters:
/// - header_buffer: The buffer containing the header content.
/// - header_encoding: The encoding type of the header.
/// Returns:
/// - Parsed Python value on success, or an error on failure.
fn processNpyHeader(header_buffer: []const u8, header_encoding: HeaderEncoding, allocator: std.mem.Allocator) NpyHeaderParseError!NpyHeaderData {
    // Check for ending newline
    if (header_buffer.len == 0 or header_buffer[header_buffer.len - 1] != '\n') {
        return NpyHeaderParseError.EndingNewlineMissing;
    }
    // Trim newline and spaces right before it
    const trimmed_header = std.mem.trimRight(u8, header_buffer[0 .. header_buffer.len - 1], " ");

    const header_data = readNpyHeaderData(trimmed_header, header_encoding, allocator) catch {
        return NpyHeaderParseError.InvalidHeaderFormat;
    };

    return header_data;
}

/// Reads and validates a .npy file header.
/// Paramters:
/// - reader: The reader to read the .npy file from, or any reader implementing `std.io.Reader`.
/// Returns:
/// - NpyFileReadError on failure.
pub fn readNpyFile(reader: *std.Io.Reader) (NpyFileReadError || std.mem.Allocator.Error)!void {
    var eight_byte_buffer: [8]u8 = undefined;
    reader.readSliceAll(eight_byte_buffer[0..]) catch |e| {
        switch (e) {
            error.ReadFailed => return NpyFileReadError.IoError,
            error.EndOfStream => return NpyFileReadError.TooShort,
        }
    };

    if (!std.mem.eql(u8, eight_byte_buffer[0..6], MAGIC)) {
        return NpyFileReadError.MagicMismatch;
    }

    const major_version = eight_byte_buffer[6];
    const minor_version = eight_byte_buffer[7];

    const version_props: VersionProps = version: {
        if (minor_version != 0) {
            return NpyFileReadError.UnsupportedVersion;
        }
        switch (major_version) {
            1 => break :version .{ .header_size_type = .U16, .encoding = .Ascii },
            2 => break :version .{ .header_size_type = .U32, .encoding = .Ascii },
            3 => break :version .{ .header_size_type = .U32, .encoding = .Utf8 },
            else => return NpyFileReadError.UnsupportedVersion,
        }
    };

    // Read the header size in little-endian format
    const header_size: u32 = header_size: switch (version_props.header_size_type) {
        .U16 => {
            var size_buffer: [2]u8 = undefined;
            reader.readSliceAll(size_buffer[0..]) catch |e| {
                switch (e) {
                    error.ReadFailed => return NpyFileReadError.IoError,
                    error.EndOfStream => return NpyFileReadError.TooShort,
                }
            };
            break :header_size std.mem.readInt(u16, &size_buffer, .little);
        },
        .U32 => {
            var size_buffer: [4]u8 = undefined;
            reader.readSliceAll(size_buffer[0..]) catch |e| {
                switch (e) {
                    error.ReadFailed => return NpyFileReadError.IoError,
                    error.EndOfStream => return NpyFileReadError.TooShort,
                }
            };
            break :header_size std.mem.readInt(u32, &size_buffer, .little);
        },
    };

    // Now read the header content
    // If header size is larger than 1024 bytes, use heap allocation
    var fallback = std.heap.stackFallback(1024, std.heap.page_allocator);
    const allocator = fallback.get();
    const header_buffer = try allocator.alloc(u8, header_size);
    defer allocator.free(header_buffer);

    reader.readSliceAll(header_buffer) catch |e| {
        switch (e) {
            error.ReadFailed => return NpyFileReadError.IoError,
            error.EndOfStream => return NpyFileReadError.TooShort,
        }
    };

    const header_data = processNpyHeader(header_buffer, version_props.encoding, allocator) catch {
        return NpyFileReadError.InvalidHeader;
    };

    log.info("Successfully read .npy header: {any}", .{header_data});
}
