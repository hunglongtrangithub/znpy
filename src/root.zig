//! By convention, root.zig is the root source file when making a library.
const std = @import("std");

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
    _ = header;
    _ = elements;
    _ = shape;
    _ = array;
    _ = slice;
}
