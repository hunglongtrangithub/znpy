//! Module for reading and parsing .npy file headers.
//!
//! This module provides functionality to parse the header dictionary from NumPy `.npy` files.
//! The header contains metadata about the array stored in the file.
//!
//! ## Supported Grammar
//!
//! The header must be a Python dictionary-like structure with the following grammar
//! (whitespace is ignored):
//!
//! ```
//! header          ::= "{" key_value_pairs "}"
//! key_value_pairs ::= key_value_pair ("," key_value_pair)* ","?
//! key_value_pair  ::= "'" key "'" ":" value
//! key             ::= "descr" | "fortran_order" | "shape"
//! value           ::= string_literal | boolean_literal | tuple
//! string_literal  ::= "'" string "'"
//! boolean_literal ::= "True" | "False"
//! tuple           ::= "()" | "(" number ",)" | "(" number ("," number)+ ","? ")"
//! number          ::= [0-9]+
//! string          ::= any sequence of characters except "'"
//! ```
//!
//! ## Required Keys
//!
//! - `descr`: A string describing the data type. Syntax: `<endianness><type kind><size in bytes>` (e.g., `'<f8'` for little-endian float64)
//! - `fortran_order`: A boolean indicating whether the array is stored in Fortran (column-major) order
//! - `shape`: A tuple of integers representing the array dimensions
//!
//! ## Example
//!
//! ```
//! {'descr': '<f8', 'fortran_order': False, 'shape': (3, 4)}
//! ```
const std = @import("std");

const parse = @import("header/parse.zig");
const descr = @import("header/descr.zig");
const lex = @import("header/lex.zig");

const log = std.log.scoped(.npy_header);

pub const ElementType = descr.ElementType;

/// Specifies how array data is laid out in memory
pub const Order = enum {
    /// Array data is in row-major order (C-contiguous)
    C,
    /// Array data is in column-major order (Fortran-contiguous)
    F,
};

pub const HeaderEncoding = enum {
    /// Header data is in ASCII
    Ascii,
    /// Header data is in UTF-8
    Utf8,
};

const HeaderSizeType = enum {
    /// Header size is an unsigned short (`u16`)
    U16,
    /// Header size is an unsigned int (`u32`)
    U32,
};

const VersionProps = struct {
    header_size_type: HeaderSizeType,
    encoding: HeaderEncoding,
};

const ParseHeaderError = error{
    /// Error reading from file, due to I/O issues (unexpected end-of-file/sequence or read failure).
    IoError,
    /// Magic sequence does not match the expected value.
    MagicMismatch,
    /// The format version is unsupported.
    UnsupportedVersion,
    /// The header size is too large to be a `usize`.
    HeaderSizeOverflow,
    /// The header does not end with a newline character.
    MissingNewline,
    /// The file does not have a valid header.
    InvalidHeader,
    /// Header is not a valid Python literal format.
    InvalidHeaderFormat,
    /// Header's Python literal is not a dictionary.
    ExpectedPythonDict,
    /// Expected key 'fortran_order' is missing.
    ExpectedKeyFortranOrder,
    /// Expected key 'descr' is missing.
    ExpectedKeyDescr,
    /// Expected key 'shape' is missing.
    ExpectedKeyShape,
    /// The value for 'fortran_order' is invalid.
    InvalidValueFortranOrder,
    /// The value for 'descr' is invalid.
    InvalidValueDescr,
    /// The value for 'shape' is invalid.
    InvalidValueShape,
    /// The 'descr' value specifies an unsupported type.
    UnsupportedDescrType,
};

/// A simple slice reader that tracks position without copying data.
pub const SliceReader = struct {
    /// The underlying byte slice to read from (owned by caller).
    slice: []const u8,
    /// Current read position within the slice.
    pos: usize = 0,

    const Self = @This();

    pub const Error = error{
        NotEnoughBytes,
    };

    pub fn init(slice: []const u8) Self {
        return .{ .slice = slice, .pos = 0 };
    }

    /// Returns the next n bytes as a slice without copying.
    /// Returns error.NotEnoughBytes if there aren't enough bytes remaining.
    pub fn readBytes(self: *Self, n: usize) Error![]const u8 {
        const remaining = self.slice.len - self.pos;
        if (n > remaining) {
            return Error.NotEnoughBytes;
        }
        const result = self.slice[self.pos..][0..n];
        self.pos += n;
        return result;
    }
};

pub const ReadHeaderError = ParseHeaderError || std.mem.Allocator.Error;

/// Represents the parsed header information from a .npy file.
pub const Header = struct {
    /// The shape of the array, extracted from the 'shape' key.
    shape: []usize,
    /// The element type descriptor, extracted from the 'descr' key.
    descr: ElementType,
    /// The array order, extracted from the 'fortran_order' key.
    order: Order,

    const Self = @This();

    const MAGIC = "\x93NUMPY";

    /// Reads and parses the header from a slice reader (`SliceReader`).
    /// Returns the parsed header and advances the slice reader position.
    /// This function uses the slice reader to avoid unnecessary copies, unlike `fromReader`.
    pub fn fromSliceReader(slice_reader: *SliceReader, allocator: std.mem.Allocator) ReadHeaderError!Self {
        const eight_bytes = slice_reader.readBytes(8) catch {
            return ParseHeaderError.IoError;
        };

        // Check magic
        if (!std.mem.eql(u8, eight_bytes[0..6], MAGIC)) {
            return ParseHeaderError.MagicMismatch;
        }

        const major_version = eight_bytes[6];
        const minor_version = eight_bytes[7];

        // Get version
        const version_props: VersionProps = version: {
            if (minor_version != 0) {
                return ParseHeaderError.UnsupportedVersion;
            }
            switch (major_version) {
                1 => break :version .{ .header_size_type = .U16, .encoding = .Ascii },
                2 => break :version .{ .header_size_type = .U32, .encoding = .Ascii },
                3 => break :version .{ .header_size_type = .U32, .encoding = .Utf8 },
                else => return ParseHeaderError.UnsupportedVersion,
            }
        };

        // Read the header size in little-endian format and cast to usize
        const header_size: usize = header_size: switch (version_props.header_size_type) {
            .U16 => {
                const size_bytes = slice_reader.readBytes(2) catch {
                    return ParseHeaderError.IoError;
                };
                const size = std.mem.readInt(u16, size_bytes[0..2], .little);
                break :header_size std.math.cast(usize, size) orelse return ParseHeaderError.HeaderSizeOverflow;
            },
            .U32 => {
                const size_bytes = slice_reader.readBytes(4) catch {
                    return ParseHeaderError.IoError;
                };
                const size = std.mem.readInt(u32, size_bytes[0..4], .little);
                break :header_size std.math.cast(usize, size) orelse return ParseHeaderError.HeaderSizeOverflow;
            },
        };

        // Read the header content without copying
        const header_buffer = slice_reader.readBytes(header_size) catch {
            return ParseHeaderError.IoError;
        };

        // Check for ending newline
        if (header_buffer.len == 0 or header_buffer[header_buffer.len - 1] != '\n') {
            return ParseHeaderError.MissingNewline;
        }
        // Trim newline and spaces right before it
        const trimmed_header = std.mem.trimRight(u8, header_buffer[0 .. header_buffer.len - 1], " ");
        return Self.fromPythonString(trimmed_header, version_props.encoding, allocator);
    }

    /// Reads and parses the header from a reader (`std.io.Reader`).
    /// The reader should be positioned at the start of the .npy file.
    /// On success, the reader stops right after the header content.
    pub fn fromReader(reader: *std.io.Reader, allocator: std.mem.Allocator) ReadHeaderError!Self {
        var eight_byte_buffer: [8]u8 = undefined;
        reader.readSliceAll(eight_byte_buffer[0..]) catch {
            return ParseHeaderError.IoError;
        };

        // Check magic
        if (!std.mem.eql(u8, eight_byte_buffer[0..6], MAGIC)) {
            return ParseHeaderError.MagicMismatch;
        }

        const major_version = eight_byte_buffer[6];
        const minor_version = eight_byte_buffer[7];

        // Get version
        const version_props: VersionProps = version: {
            if (minor_version != 0) {
                return ParseHeaderError.UnsupportedVersion;
            }
            switch (major_version) {
                1 => break :version .{ .header_size_type = .U16, .encoding = .Ascii },
                2 => break :version .{ .header_size_type = .U32, .encoding = .Ascii },
                3 => break :version .{ .header_size_type = .U32, .encoding = .Utf8 },
                else => return ParseHeaderError.UnsupportedVersion,
            }
        };

        // Read the header size in little-endian format and cast to usize
        const header_size: usize = header_size: switch (version_props.header_size_type) {
            .U16 => {
                var size_buffer: [2]u8 = undefined;
                reader.readSliceAll(size_buffer[0..]) catch {
                    return ParseHeaderError.IoError;
                };
                const size = std.mem.readInt(u16, &size_buffer, .little);
                break :header_size std.math.cast(usize, size) orelse return ParseHeaderError.HeaderSizeOverflow;
            },
            .U32 => {
                var size_buffer: [4]u8 = undefined;
                reader.readSliceAll(size_buffer[0..]) catch {
                    return ParseHeaderError.IoError;
                };
                const size = std.mem.readInt(u32, &size_buffer, .little);
                break :header_size std.math.cast(usize, size) orelse return ParseHeaderError.HeaderSizeOverflow;
            },
        };

        // Now read the header content
        const header_buffer = try allocator.alloc(u8, header_size);
        defer allocator.free(header_buffer);

        reader.readSliceAll(header_buffer) catch {
            return ParseHeaderError.IoError;
        };

        // Check for ending newline
        if (header_buffer.len == 0 or header_buffer[header_buffer.len - 1] != '\n') {
            return ParseHeaderError.MissingNewline;
        }
        // Trim newline and spaces right before it
        const trimmed_header = std.mem.trimRight(u8, header_buffer[0 .. header_buffer.len - 1], " ");
        return Self.fromPythonString(trimmed_header, version_props.encoding, allocator);
    }

    /// Parses the given string buffer info a Header struct.
    /// The string buffer is expected to be in the format of a Python dictionary.
    pub fn fromPythonString(header_buffer: []const u8, header_encoding: HeaderEncoding, allocator: std.mem.Allocator) ReadHeaderError!Header {
        var parser = parse.Parser.init(header_buffer, header_encoding);
        const ast = parser.parse(allocator) catch {
            return ParseHeaderError.InvalidHeaderFormat;
        };
        defer ast.deinit(allocator);

        var header_data = Header{
            .descr = undefined,
            .order = undefined,
            .shape = undefined,
        };

        // Extract values from AST
        switch (ast) {
            .Map => |map| {
                // Find 'descr'
                const descr_ast = map.get("descr") orelse return ParseHeaderError.ExpectedKeyDescr;
                switch (descr_ast) {
                    .Literal => |lit| switch (lit) {
                        .String => |s| header_data.descr = ElementType.fromString(s) catch {
                            return ParseHeaderError.InvalidValueDescr;
                        },
                        else => return ParseHeaderError.InvalidValueDescr,
                    },
                    else => return ParseHeaderError.InvalidValueDescr,
                }

                // Find 'fortran_order'
                const fortran_order_ast = map.get("fortran_order") orelse return ParseHeaderError.ExpectedKeyFortranOrder;
                switch (fortran_order_ast) {
                    .Literal => |lit| switch (lit) {
                        .Boolean => |b| header_data.order = if (b) Order.F else Order.C,
                        else => return ParseHeaderError.InvalidValueFortranOrder,
                    },
                    else => return ParseHeaderError.InvalidValueFortranOrder,
                }

                // Find 'shape'
                const shape_ast = map.get("shape") orelse return ParseHeaderError.ExpectedKeyShape;
                switch (shape_ast) {
                    .Tuple => |tuple| {
                        // Allocate and copy shape data
                        const shape_slice = try allocator.alloc(usize, tuple.items.len);
                        @memcpy(shape_slice, tuple.items);
                        header_data.shape = shape_slice;
                    },
                    else => return ParseHeaderError.InvalidValueShape,
                }
            },
            else => return ParseHeaderError.ExpectedPythonDict,
        }

        return header_data;
    }

    pub fn deinit(self: Header, allocator: std.mem.Allocator) void {
        allocator.free(self.shape);
    }
};

test {
    _ = parse;
    _ = descr;
    _ = lex;
}
