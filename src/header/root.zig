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
const lex = @import("lex.zig");
const parse = @import("parse.zig");
const read = @import("../read.zig");
const descr = @import("descr.zig");

pub const Endianness = descr.Endianness;
pub const ElementType = descr.ElementType;

/// NpyHeaderData represents the parsed header information from a .npy file.
pub const NpyHeaderData = struct {
    shape: []usize,
    descr: descr.DescrData,
    fortran_order: bool,
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
    UnsupportedDescrType,
};

pub fn readNpyHeaderData(header_buffer: []const u8, header_encoding: read.HeaderEncoding, allocator: std.mem.Allocator) (ReadHeaderDataError || std.mem.Allocator.Error)!NpyHeaderData {
    var parser = parse.Parser.init(header_buffer, header_encoding);
    const ast = parser.parse(allocator) catch {
        return ReadHeaderDataError.InvalidHeaderFormat;
    };
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
                    .String => |s| header_data.descr = descr.parseDescr(s) catch |e| {
                        switch (e) {
                            descr.ParseDescrError.UnsupportedType => return ReadHeaderDataError.UnsupportedDescrType,
                            else => return ReadHeaderDataError.InvalidDescrValue,
                        }
                    },
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
