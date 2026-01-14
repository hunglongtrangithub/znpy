//! Module for array types: static-ranked vs dynamic-ranked arrays holding mutable vs immutable data.
//! Static array types have their rank (number of dimensions) known at compile time, and thus some
//! allocations can be avoided, and offset calculation loop can be unrolled by the compiler.
//!
//! This module provides multidimensional arrays with support for:
//! - **Ownership**: Arrays own their data, views reference existing data
//! - **Mutability**: Both mutable and const (read-only) variants
//! - **Rank flexibility**: Static (compile-time) or dynamic (runtime) dimensionality
//!
//! ## Array Types (Own Data)
//! Array types allocate and manage contiguous memory buffers:
//! - `DynamicArray(T)` - Dynamically-ranked mutable array
//! - `ConstDynamicArray(T)` - Dynamically-ranked read-only array
//! - `StaticArray(T, N)` - Statically-ranked mutable array (N dimensions known at compile time)
//! - `ConstStaticArray(T, N)` - Statically-ranked read-only array
//!
//! ## View Types (Reference Data)
//! View types reference existing data without ownership. Views support striding and may
//! reference non-contiguous data (e.g., slices):
//! - `ArrayView(T)` - Mutable view into array data
//! - `ConstArrayView(T)` - Read-only view into array data
//!
//! ## Const Variants
//! All const variants (`ConstArrayView`, `ConstDynamicArray`, `ConstStaticArray`) are designed
//! to view read-only data buffers, thus prevent accidental modification while maintaining full
//! access to array operations that don't require mutation.
//!
//! ## Memory Layout
//! - **Arrays**: Always store data in a single contiguous buffer
//! - **Views**: May reference non-contiguous data via strides
const std = @import("std");

const dynamic = @import("array/dynamic.zig");
const static = @import("array/static.zig");
const view = @import("array/view.zig");
const format = @import("array/format.zig");

pub const ArrayView = view.ArrayView;
pub const ConstArrayView = view.ConstArrayView;
pub const DynamicArray = dynamic.DynamicArray;
pub const ConstDynamicArray = dynamic.ConstDynamicArray;
pub const StaticArray = static.StaticArray;
pub const ConstStaticArray = static.ConstStaticArray;

test {
    _ = dynamic;
    _ = static;
    _ = view;
    _ = format;
}
