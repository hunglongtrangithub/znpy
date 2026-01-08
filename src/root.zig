//! By convention, root.zig is the root source file when making a library.
const std = @import("std");

pub const header = @import("header.zig");
pub const elements = @import("elements.zig");
pub const dimension = @import("dimension.zig");

pub const ElementType = header.ElementType;
pub const Element = elements.Element;

test {
    _ = header;
    _ = elements;
    _ = dimension;
}
