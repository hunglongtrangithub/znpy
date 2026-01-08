//! By convention, root.zig is the root source file when making a library.
const std = @import("std");
pub const header = @import("header.zig");
pub const elements = @import("elements.zig");

pub const ElementType = header.ElementType;
pub const Element = elements.Element;

/// The number of bytes an array takes, given its shape and element type, must not exceed `std.math.maxInt(isize)`
/// Returns the number of elements in the array on success, or `null` if overflow occurs.
pub fn shapeSizeChecked(T: ElementType, shape: []usize) ?usize {
    const isize_max = std.math.maxInt(isize);

    const num_elements = blk: {
        var prod: usize = 1;
        for (shape) |dim| {
            prod, const overflow = @mulWithOverflow(prod, dim);
            if (overflow != 0) {
                return null;
            }
        }
        break :blk prod;
    };

    const num_bytes, const overflow = @mulWithOverflow(num_elements, T.byteSize());
    if ((overflow != 0) or (num_bytes > isize_max)) {
        return null;
    }
    return num_elements;
}

test {
    // Make all tests in other files imported by this module available during testing
    std.testing.refAllDeclsRecursive(@This());
}
