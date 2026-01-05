const std = @import("std");
const lex = @import("lex.zig");
const read = @import("../read.zig");
const log = std.log.scoped(.npy_parser);
const Token = lex.Token;

/// The Abstract Syntax Tree for the parsed header content.
/// Right now we only support maps, tuples, and literals.
const Ast = union(enum) {
    Map: std.StringHashMapUnmanaged(Ast),
    Literal: lex.Literal,
    Tuple: std.ArrayList(usize),

    const Self = @This();

    pub fn deinit(self: Self, allocator: std.mem.Allocator) void {
        switch (self) {
            .Map => |map| {
                var it = map.iterator();
                while (it.next()) |entry| {
                    entry.value_ptr.deinit(allocator);
                }
                var mut_map = map;
                mut_map.deinit(allocator);
            },
            .Tuple => |tuple| {
                var mut_tuple = tuple;
                mut_tuple.deinit(allocator);
            },
            .Literal => {},
        }
    }
};

const ParserError = error{
    EmptyInput,
    LiteralFollowedByMoreTokens,
    MisplacedToken,
    InvalidKey,
    MissingColonAfterKey,
    MissingCommaAfterValue,
    InvalidValue,
    InvalidTupleElement,
    MissingTrailingComma,
    InvalidTrailingComma,
    InvalidSyntax,
};

pub const BuildError = lex.LexerError || ParserError || std.mem.Allocator.Error;

pub const Parser = struct {
    const Self = @This();

    lexer: lex.NpyHeaderLexer,

    /// Initializes the parser with the given header buffer and encoding.
    /// Input buffer must outlive the parser.
    /// Parameters:
    /// header_buffer - The buffer containing the header content.
    /// header_encoding - The encoding type of the header.
    pub fn init(header_buffer: []const u8, header_encoding: read.HeaderEncoding) Self {
        return .{
            .lexer = lex.NpyHeaderLexer.init(header_buffer, header_encoding),
        };
    }

    /// Parse a map. The opening brace has already been consumed.
    /// The map keys are strings, and values can be literals, tuples, or nested maps.
    fn parseMap(self: *Self, allocator: std.mem.Allocator) BuildError!std.StringHashMapUnmanaged(Ast) {
        log.info("Parsing map...", .{});
        var map = std.StringHashMapUnmanaged(Ast).empty;
        errdefer map.deinit(allocator);

        const State = union(enum) {
            /// Beginning state (at the opening brace)
            Start,
            /// At a key
            Key: []const u8,
            /// At a value
            Value,
            /// At the closing brace
            Final,
        };

        // Initialize state machine
        var state: State = .Start;
        while (state != .Final) {
            const token = try self.lexer.advance();
            switch (state) {
                .Start => switch (token) {
                    // Empty map
                    .RBrace => state = .Final,
                    .Literal => |k|
                    // Expect string literal as key
                    switch (k) {
                        .String => |s| {
                            log.info("Found key: {s}", .{s});
                            state = .{ .Key = s };
                        },
                        else => return ParserError.InvalidKey,
                    },
                    else => return ParserError.InvalidKey,
                },
                .Key => {
                    // Expect colon after key
                    if (token != .Colon) {
                        return ParserError.MissingColonAfterKey;
                    }
                    // Expect value after colon
                    switch (try self.lexer.advance()) {
                        // Literal value
                        .Literal => |literal| {
                            std.log.info("Found literal value for key {s}: {}", .{ state.Key, literal });
                            try map.put(allocator, state.Key, .{ .Literal = literal });
                        },
                        .LParen => {
                            // Parse tuple value
                            const tuple = try self.parseTuple(allocator);
                            std.log.info("Found tuple value for key {s}: {}", .{ state.Key, tuple });
                            try map.put(allocator, state.Key, .{ .Tuple = tuple });
                        },
                        .LBrace => {
                            // Parse nested map value
                            const nested_map = try self.parseMap(allocator);
                            std.log.info("Found nested map value for key {s}: {}", .{ state.Key, nested_map });
                            try map.put(allocator, state.Key, .{ .Map = nested_map });
                        },
                        else => return ParserError.InvalidValue,
                    }
                    state = .Value;
                },
                .Value => switch (token) {
                    // Comma can only be followed by another key or closing brace
                    .Comma => switch (try self.lexer.advance()) {
                        // Expect another key
                        .Literal => |k| {
                            // Expect string literal as key
                            switch (k) {
                                .String => |s| {
                                    log.info("Found key: {s}", .{s});
                                    state = .{ .Key = s };
                                },
                                else => return ParserError.InvalidKey,
                            }
                        },
                        .RBrace => state = .Final,
                        else => return ParserError.InvalidKey,
                    },
                    .RBrace => state = .Final,
                    else => return ParserError.MissingCommaAfterValue,
                },
                .Final => unreachable,
            }
        }
        return map;
    }

    /// Parse a tuple. The opening parenthesis has already been consumed.
    /// The tuple can only contain number literals.
    fn parseTuple(self: *Self, allocator: std.mem.Allocator) BuildError!std.ArrayList(usize) {
        log.info("Parsing tuple...", .{});
        var list = std.ArrayList(usize).empty;
        errdefer list.deinit(std.heap.page_allocator);

        const State = enum {
            /// Beginning state (at the opening parenthesis)
            Start,
            /// At a comma
            Comma,
            /// At a number literal
            Literal,
            /// At the closing parenthesis
            Final,
        };

        // Initialize state machine
        var state: State = .Start;
        while (state != .Final) {
            const token = try self.lexer.advance();
            switch (state) {
                .Start => {
                    switch (token) {
                        // Empty tuple
                        .RParen => state = .Final,
                        .Literal => |literal| {
                            switch (literal) {
                                .Number => |n| {
                                    try list.append(allocator, n);
                                    state = .Literal;
                                },
                                else => return ParserError.InvalidTupleElement,
                            }
                        },
                        else => return ParserError.InvalidTupleElement,
                    }
                },
                .Literal => {
                    switch (token) {
                        .Comma => state = .Comma,
                        .RParen => switch (list.items.len) {
                            // Single-element tuple must have a trailing comma before closing parenthesis
                            1 => return ParserError.MissingTrailingComma,
                            // Multi-element tuple can close directly
                            else => state = .Final,
                        },
                        else => return ParserError.InvalidSyntax,
                    }
                },
                .Comma => {
                    switch (token) {
                        .Literal => |literal| {
                            switch (literal) {
                                .Number => |n| {
                                    try list.append(allocator, n);
                                    state = .Literal;
                                },
                                else => return ParserError.InvalidTupleElement,
                            }
                        },
                        .RParen => switch (list.items.len) {
                            // Single-element tuple with trailing comma
                            1 => state = .Final,
                            // Trailing comma not allowed for multi-element tuple
                            else => return ParserError.InvalidTrailingComma,
                        },
                        else => return ParserError.InvalidSyntax,
                    }
                },
                .Final => unreachable,
            }
        }

        return list;
    }

    /// Parse the header content into an AST.
    pub fn parse(self: *Self, allocator: std.mem.Allocator) BuildError!Ast {
        const token = try self.lexer.advance();
        switch (token) {
            .EOF => return ParserError.EmptyInput,
            .LBrace => {
                log.info("Parsing top-level map...", .{});
                return .{ .Map = try self.parseMap(allocator) };
            },
            .LParen => {
                log.info("Parsing top-level tuple...", .{});
                return .{ .Tuple = try self.parseTuple(allocator) };
            },
            .RBrace, .RParen, .Colon, .Comma => return ParserError.MisplacedToken,
            .Literal => |literal| {
                log.info("Parsing top-level literal...", .{});
                // No more tokens should follow a literal
                if (try self.lexer.peek() != .EOF) {
                    return ParserError.LiteralFollowedByMoreTokens;
                }
                return .{ .Literal = literal };
            },
        }
    }
};
