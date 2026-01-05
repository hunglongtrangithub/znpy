const std = @import("std");
const lex = @import("lex.zig");
const parse = @import("parse.zig");
const read = @import("../read.zig");

/// NpyHeaderData represents the parsed header information from a .npy file.
/// We only accept the value of `descr` as a string only for now.
/// The grammar for the header that we support is as follows (any whitespaces are ignored):
/// ```
/// header ::= "{" key_value_pairs "}"
/// key_value_pairs ::= key_value_pair ("," key_value_pair)* ","?
/// key_value_pair ::= "'" key "'" ":" value
/// key ::= "descr" | "fortran_order" | "shape"
/// value ::= "'" string "'" | "True" | "False" | tuple
/// tuple ::= "()" | "(" number ",)" | "(" number ("," number)+ ")"
/// number ::= [0-9]+
/// string ::= any sequence of characters except "'"
/// ```
pub const NpyHeaderData = struct {
    descr: []const u8,
    fortran_order: bool,
    shape: []usize,
};

pub const ReadHeaderDataError = error{
    InvalidHeaderFormat,
    ExpectedHeaderToBeMap,
    ExpectedKeyFortranOrder,
    InvalidFortranOrderValue,
    ExpectedKeyDescr,
    InvalidDescrValue,
    ExpectedKeyShape,
    InvalidShapeValue,
};

pub fn readNpyHeaderData(header_buffer: []const u8, header_encoding: read.HeaderEncoding, allocator: std.mem.Allocator) (ReadHeaderDataError || std.mem.Allocator.Error)!NpyHeaderData {
    var parser = parse.Parser.init(header_buffer, header_encoding);
    const ast = parser.parse(allocator) catch return ReadHeaderDataError.InvalidHeaderFormat;
    defer ast.deinit(allocator);

    var header_data = NpyHeaderData{
        .descr = undefined,
        .fortran_order = undefined,
        .shape = undefined,
    };

    // Extract values from AST
    switch (ast) {
        .Map => |map| {
            // Find 'descr'
            const descr_ast = map.get("descr") orelse return ReadHeaderDataError.ExpectedKeyDescr;
            switch (descr_ast) {
                .Literal => |lit| switch (lit) {
                    .String => |s| header_data.descr = s,
                    else => return ReadHeaderDataError.InvalidDescrValue,
                },
                else => return ReadHeaderDataError.InvalidDescrValue,
            }

            // Find 'fortran_order'
            const fortran_order_ast = map.get("fortran_order") orelse return ReadHeaderDataError.ExpectedKeyFortranOrder;
            switch (fortran_order_ast) {
                .Literal => |lit| switch (lit) {
                    .Boolean => |b| header_data.fortran_order = b,
                    else => return ReadHeaderDataError.InvalidFortranOrderValue,
                },
                else => return ReadHeaderDataError.InvalidFortranOrderValue,
            }

            // Find 'shape'
            const shape_ast = map.get("shape") orelse return ReadHeaderDataError.ExpectedKeyShape;
            switch (shape_ast) {
                .Tuple => |tuple| {
                    // Allocate and copy shape data
                    const shape_slice = try allocator.alloc(usize, tuple.items.len);
                    @memcpy(shape_slice, tuple.items);
                    header_data.shape = shape_slice;
                },
                else => return ReadHeaderDataError.InvalidShapeValue,
            }
        },
        else => return ReadHeaderDataError.ExpectedHeaderToBeMap,
    }

    return header_data;
}
