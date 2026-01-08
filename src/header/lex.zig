const std = @import("std");

const header = @import("../header.zig");

const log = std.log.scoped(.npy_lexer);

pub const HeaderEncoding = header.HeaderEncoding;

/// A literal value in the .npy header.
pub const Literal = union(enum) {
    String: []const u8,
    Number: usize,
    Boolean: bool,
};

/// Token represents the different kinds of tokens that can be found in the .npy header.
pub const Token = union(enum) {
    /// '{'
    LBrace,
    /// '}'
    RBrace,
    /// ':'
    Colon,
    /// ','
    Comma,
    /// '('
    LParen,
    /// ')'
    RParen,
    // A literal
    Literal: Literal,
    /// End of input
    EOF,
};

pub const LexerError = error{
    /// An unsupported identifier
    UnsupportedIdentifier,
    /// An unexpected character
    UnexpectedCharacter,
    /// An expected end of input
    UnexpectedEndOfInput,
    /// An invalid UTF-8 character
    Utf8InvalidCharacter,
    /// Invalid number literal
    InvalidNumberLiteral,
    /// Invalid byte encountered
    InvalidByte,
};

/// A lexer for tokenizing .npy header input.
pub const Lexer = struct {
    const Self = @This();

    /// The header input (owned by caller)
    input: []const u8,
    /// The start byte of the current character (UTF-8 code point)
    position: usize,
    /// Peeked token cache
    peeked: ?Token,
    /// The encoding of the header
    encoding: HeaderEncoding,

    /// Initialize a new lexer with the given input and encoding.
    /// The input slice must outlive the lexer.
    pub fn init(input: []const u8, encoding: HeaderEncoding) Self {
        return .{
            .input = input,
            .encoding = encoding,
            .position = 0,
            .peeked = null,
        };
    }

    /// Get the current character (UTF-8 code point) without advancing.
    /// Guard against end of input and invalid byte sequences.
    /// The returned slice has a length of 1 for ASCII characters, and length of 1-4 for UTF-8 characters.
    /// Returns null if at end of input.
    fn currentChar(self: *const Self) LexerError!?[]const u8 {
        if (self.position >= self.input.len) return null;
        switch (self.encoding) {
            .Ascii => {
                if (std.ascii.isAscii(self.input[self.position])) {
                    return self.input[self.position .. self.position + 1];
                } else {
                    return LexerError.Utf8InvalidCharacter;
                }
            },
            .Utf8 => {
                const char_len = std.unicode.utf8ByteSequenceLength(self.input[self.position]) catch return LexerError.Utf8InvalidCharacter;
                if (self.position + char_len > self.input.len) {
                    return LexerError.Utf8InvalidCharacter;
                }
                const slice = self.input[self.position .. self.position + char_len];
                if (std.unicode.utf8ValidateSlice(slice)) {
                    return slice;
                } else {
                    return LexerError.Utf8InvalidCharacter;
                }
            },
        }
    }

    /// Advance the position by one character (UTF-8 code point) to the start byte of the next character.
    /// Assumes that the current byte position is valid.
    fn advanceChar(self: *Self) LexerError!void {
        if (self.position >= self.input.len) return;

        switch (self.encoding) {
            .Ascii => {
                self.position += 1;
            },
            .Utf8 => {
                const char_len = std.unicode.utf8ByteSequenceLength(self.input[self.position]) catch {
                    return LexerError.Utf8InvalidCharacter;
                };
                self.position += char_len;
            },
        }
    }

    /// Internal method to lex the next token from the input.
    fn lexNext(self: *Self) LexerError!Token {
        // Skip consecutive whitespaces if any
        while (self.position < self.input.len and std.ascii.isWhitespace(self.input[self.position])) {
            self.position += 1;
        }

        // We expect an ASCII character here. UTF-8 multi-byte characters can only exist in string literals.
        const char = if (self.position < self.input.len) self.input[self.position] else return .EOF;

        const token: Token = switch (char) {
            '{' => blk: {
                self.position += 1;
                break :blk .LBrace;
            },
            '}' => blk: {
                self.position += 1;
                break :blk .RBrace;
            },
            ':' => blk: {
                self.position += 1;
                break :blk .Colon;
            },
            ',' => blk: {
                self.position += 1;
                break :blk .Comma;
            },
            '(' => blk: {
                self.position += 1;
                break :blk .LParen;
            },
            ')' => blk: {
                self.position += 1;
                break :blk .RParen;
            },
            '\'' => blk: {
                self.position += 1; // Skip opening quote
                const start = self.position;
                const end = end: while (try self.currentChar()) |c| {
                    // Stop at closing quote
                    if (c.len == 1 and c[0] == '\'') {
                        break :end self.position;
                    }
                    try self.advanceChar();
                } else return LexerError.UnexpectedEndOfInput;
                self.position += 1; // Skip closing quote
                break :blk .{ .Literal = .{ .String = self.input[start..end] } };
            },
            '0'...'9' => blk: {
                const start = self.position;
                self.position += 1; // Consume first digit
                while (try self.currentChar()) |c| {
                    if (c.len == 1 and std.ascii.isDigit(c[0])) {
                        // We know it's a single-byte ASCII digit, so just increment position by 1
                        self.position += 1;
                    } else {
                        // Stop at first non-digit character
                        break;
                    }
                }
                const number_string = self.input[start..self.position];
                const number = std.fmt.parseInt(usize, number_string, 10) catch return LexerError.InvalidNumberLiteral;
                break :blk .{ .Literal = .{ .Number = number } };
            },
            // Identifiers: True, False, or any seqauence of letters and underscores
            'A'...'Z', 'a'...'z', '_' => blk: {
                const start = self.position;
                self.position += 1; // Consume first character
                while (try self.currentChar()) |c| {
                    switch (c.len) {
                        1 => {},
                        else => break,
                    }
                    switch (c[0]) {
                        'A'...'Z', 'a'...'z', '_' => {},
                        else => break,
                    }
                    // If we reach here, it's a valid identifier character
                    self.position += 1;
                }

                const identifier = self.input[start..self.position];
                if (std.mem.eql(u8, identifier, "True")) {
                    break :blk .{ .Literal = .{ .Boolean = true } };
                } else if (std.mem.eql(u8, identifier, "False")) {
                    break :blk .{ .Literal = .{ .Boolean = false } };
                } else {
                    return LexerError.UnsupportedIdentifier;
                }
            },
            else => return LexerError.InvalidByte,
        };
        return token;
    }

    /// Peek at the next token without consuming it.
    /// If there are no more tokens, return EOF token.
    pub fn peek(self: *Self) LexerError!Token {
        if (self.peeked) |peeked| {
            return peeked;
        }

        const token = try self.lexNext();
        self.peeked = token;
        return token;
    }

    /// Advance to the next token and return it.
    /// If there are no more tokens, return EOF token.
    pub fn advance(self: *Self) LexerError!Token {
        const token = if (self.peeked) |peeked| blk: {
            self.peeked = null;
            break :blk peeked;
        } else try self.lexNext();

        return token;
    }
};

fn expectToken(expected: LexerError!Token, actual: LexerError!Token) !void {
    if (expected) |expected_token| {
        // Expecting a successful token
        const actual_token = actual catch |err| {
            std.debug.print("Expected token {any}, but got error: {any}\n", .{ expected_token, err });
            return error.TestExpectedEqual;
        };
        try std.testing.expectEqual(std.meta.activeTag(expected_token), std.meta.activeTag(actual_token));
        switch (expected_token) {
            .Literal => |lit| switch (lit) {
                .Boolean => |b| try std.testing.expectEqual(b, actual_token.Literal.Boolean),
                .Number => |n| try std.testing.expectEqual(n, actual_token.Literal.Number),
                .String => |s| try std.testing.expectEqualSlices(u8, s, actual_token.Literal.String),
            },
            else => {},
        }
    } else |expected_error| {
        // Expecting an error
        if (actual) |actual_token| {
            std.debug.print("Expected error {any}, but got token: {any}\n", .{ expected_error, actual_token });
            return error.TestExpectedEqual;
        } else |actual_error| {
            try std.testing.expectEqual(expected_error, actual_error);
        }
    }
}

test "empty input returns EOF right away" {
    var lexer = Lexer.init("", .Ascii);
    try expectToken(.EOF, lexer.advance());
    try expectToken(.EOF, lexer.advance());
}

test "all single-character tokens" {
    var lexer = Lexer.init("{ }:,()", .Ascii);
    try expectToken(.LBrace, lexer.advance());
    try expectToken(.RBrace, lexer.advance());
    try expectToken(.Colon, lexer.advance());
    try expectToken(.Comma, lexer.advance());
    try expectToken(.LParen, lexer.advance());
    try expectToken(.RParen, lexer.advance());
    try expectToken(.EOF, lexer.advance());
    try expectToken(.EOF, lexer.advance());
}

test "multiple tokens with whitespaces" {
    var lexer = Lexer.init(" { 'key' : 'value' , } ", .Ascii);
    try expectToken(.LBrace, lexer.advance());
    try expectToken(Token{ .Literal = .{ .String = "key" } }, lexer.advance());
    try expectToken(.Colon, lexer.advance());
    try expectToken(Token{ .Literal = .{ .String = "value" } }, lexer.advance());
    try expectToken(.Comma, lexer.advance());
    try expectToken(.RBrace, lexer.advance());
    try expectToken(.EOF, lexer.advance());
    try expectToken(.EOF, lexer.advance());
}

test "valid number literals" {
    var lexer = Lexer.init("0 123 4567 0123456789", .Ascii);
    try expectToken(Token{ .Literal = .{ .Number = 0 } }, lexer.advance());
    try expectToken(Token{ .Literal = .{ .Number = 123 } }, lexer.advance());
    try expectToken(Token{ .Literal = .{ .Number = 4567 } }, lexer.advance());
    try expectToken(Token{ .Literal = .{ .Number = 123456789 } }, lexer.advance());
    try expectToken(.EOF, lexer.advance());
    try expectToken(.EOF, lexer.advance());
}

test "invalid number literal" {
    // Construct a number literal string that exceeds the maximum usize value
    const max_usize_digits = blk: {
        const max_usize = std.math.maxInt(usize);
        var temp: usize = max_usize;
        var count: usize = 0;
        while (temp > 0) : (temp /= 10) count += 1;
        break :blk count;
    };
    const max_str = try std.testing.allocator.alloc(u8, max_usize_digits + 2);
    defer std.testing.allocator.free(max_str);
    for (0..max_str.len) |i| {
        max_str[i] = '9';
    }

    var lexer = Lexer.init(max_str, .Ascii);
    try expectToken(LexerError.InvalidNumberLiteral, lexer.advance());
}

test "boolean literals" {
    var lexer = Lexer.init(" True False Invalid ", .Ascii);
    try expectToken(Token{ .Literal = .{ .Boolean = true } }, lexer.advance());
    try expectToken(Token{ .Literal = .{ .Boolean = false } }, lexer.advance());
    try expectToken(LexerError.UnsupportedIdentifier, lexer.advance());
}

test "string literals with ASCII encoding" {
    var lexer = Lexer.init("'hello' 'world' 'Zig is great!' '\x80\xFF' ''", .Ascii);
    try expectToken(Token{ .Literal = .{ .String = "hello" } }, lexer.advance());
    try expectToken(Token{ .Literal = .{ .String = "world" } }, lexer.advance());
    try expectToken(Token{ .Literal = .{ .String = "Zig is great!" } }, lexer.advance());
    try expectToken(LexerError.Utf8InvalidCharacter, lexer.advance());
}

test "string literals with UTF-8 encoding" {
    var lexer = Lexer.init(" 'xin chào' 'hello world' 'こんにちは世界' 'Zig ⚡' '\xF8\xF9', ''", .Utf8);
    try expectToken(Token{ .Literal = .{ .String = "xin chào" } }, lexer.advance());
    try expectToken(Token{ .Literal = .{ .String = "hello world" } }, lexer.advance());
    try expectToken(Token{ .Literal = .{ .String = "こんにちは世界" } }, lexer.advance());
    try expectToken(Token{ .Literal = .{ .String = "Zig ⚡" } }, lexer.advance());
    try expectToken(LexerError.Utf8InvalidCharacter, lexer.advance());
}

test "error on truncated UTF-8 sequence in string literal" {
    // Incomplete UTF-8 sequence (0xC3 expects another byte)
    const input = "'\xC3'";
    var lexer = Lexer.init(input, .Utf8);
    try expectToken(LexerError.Utf8InvalidCharacter, lexer.advance());
}

test "error on invalid UTF-8 continuation bytes in string literal" {
    // Invalid continuation byte (0xC3 0x28 - 0x28 is not a valid continuation)
    const input = "'\xC3\x28'";
    var lexer = Lexer.init(input, .Utf8);
    try expectToken(LexerError.Utf8InvalidCharacter, lexer.advance());
}

test "error on unsupported identifier" {
    var lexer = Lexer.init("{ 'key': None }", .Ascii);
    try expectToken(Token.LBrace, lexer.advance());
    try expectToken(Token{ .Literal = .{ .String = "key" } }, lexer.advance());
    try expectToken(Token.Colon, lexer.advance());
    try expectToken(LexerError.UnsupportedIdentifier, lexer.advance());

    lexer = Lexer.init("( custom_word, )", .Ascii);
    try expectToken(Token.LParen, lexer.advance());
    try expectToken(LexerError.UnsupportedIdentifier, lexer.advance());
}

test "error on dangling opening quote" {
    var lexer = Lexer.init("'unclosed", .Ascii);
    try expectToken(LexerError.UnexpectedEndOfInput, lexer.advance());
}

test "error on invalid byte" {
    var lexer = Lexer.init("True#", .Ascii);
    try expectToken(Token{ .Literal = .{ .Boolean = true } }, lexer.advance());
    try expectToken(LexerError.InvalidByte, lexer.advance());
}

test "peek does not consume the token" {
    var lexer = Lexer.init("{", .Ascii);
    const peeked = try lexer.peek();
    try std.testing.expectEqual(Token.LBrace, @as(std.meta.Tag(Token), peeked));

    const advanced = try lexer.advance();
    try std.testing.expectEqual(Token.LBrace, @as(std.meta.Tag(Token), advanced));
}

test "multiple peeks return the same token" {
    var lexer = Lexer.init(":", .Ascii);
    const peek1 = try lexer.peek();
    const peek2 = try lexer.peek();
    const peek3 = try lexer.peek();

    try std.testing.expectEqual(peek1, peek2);
    try std.testing.expectEqual(peek2, peek3);
}

test "advance after peek returns the cached token" {
    var lexer = Lexer.init("'test' 123", .Ascii);

    const peeked = try lexer.peek();
    try expectToken(Token{ .Literal = .{ .String = "test" } }, peeked);

    const advanced = try lexer.advance();
    try std.testing.expectEqual(peeked, advanced);

    // Next token should be the number
    const next = try lexer.advance();
    try expectToken(Token{ .Literal = .{ .Number = 123 } }, next);
}
