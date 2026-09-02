//! Shared tokenizer for re-classifying an already-`--`-restored passthrough args slice
//! (see [`crate::core::args_utils::restore_double_dash`]) into flags, their values, and
//! positionals, matching the GNU/POSIX-ish conventions used by git, cargo, rg, and friends.
//! Callers keep their own list of which flags take a value (inherently per-tool) and pass it in
//! as a predicate instead of reimplementing the token-walking around it.
//!
//! Not merged with `restore_double_dash`: `Token<'a>` borrows straight from `args`, so
//! tokenizing an owned `Vec<String>` built *inside* this module would tie every `Token` to a
//! value dropped when the function returns.

/// What kind of unit a [`Token`] represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind {
    /// The literal `--` separator. Emitted exactly once, for the first `--` encountered. Under
    /// [`Dialect::Posix`] it ends option parsing (everything after is `Positional`); under
    /// [`Dialect::Msbuild`] it's an argument-*forwarding* boundary instead, so classification
    /// continues normally past it, with only its position recorded.
    DashDash,
    /// `--name` (see `Token::text` for the name, without the leading `--`).
    Long,
    /// A positional/value token — either free-standing or consumed by a preceding `Long`/`Short`
    /// as its separate-token value (see `Token::linked`).
    Positional,
    /// One character of a `-x` / `-xyz` short-option cluster (see `Token::text`, without the
    /// leading `-`). A run of only digits (`-20`) is a widely-used shorthand for a numeric
    /// value in its own right (git log/head/tail's `-N` count) rather than a cluster of
    /// per-digit boolean flags, so it is kept as one `Short` token with the whole digit run as
    /// `text`, never decomposed.
    Short,
}

/// One classified unit of an args slice, as produced by [`tokenize`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Token<'a> {
    pub kind: TokenKind,
    /// Flag name without leading dash(es) for `Long`/`Short`; raw text for `Positional`; empty
    /// for `DashDash`.
    pub text: &'a str,
    /// Value attached directly to this token: `--flag=value`, or the trailing remainder of a
    /// short cluster (`-A3` → `Short` "A" with `attached: Some("3")`).
    pub attached: Option<&'a str>,
    /// For `Long`/`Short`: index into the returned `Vec` of the `Positional` token consumed as
    /// this flag's separate-token value (only set when `takes_value` returned `true` and there
    /// was no attached value). For a consumed `Positional`: index of the flag token that owns
    /// it. `None` for a free-standing positional, an unconsumed flag, or `DashDash`.
    pub linked: Option<usize>,
    /// Index into the original `args` slice this token was produced from. Every `Short` token
    /// from the same `-xyz` cluster shares one `source_index` (they came from one arg); a
    /// consumed separate-token value always has its own, since it's a distinct arg. Lets a
    /// caller that needs to rebuild exact per-arg boundaries (e.g. whether `-r`/`-n` were typed
    /// as one cluster or two separate flags) do so without re-scanning `args` itself.
    pub source_index: usize,
    /// True if a `Long` token was written with a literal `--` prefix, as opposed to `-flag` or
    /// `/flag` under [`Dialect::Msbuild`] (all three tokenize uniformly as `Long` there, but
    /// they are *not* uniformly valid dotnet CLI syntax — see [`has_flag`] vs
    /// [`has_double_dash_flag`]). Always `true` for `Long` under [`Dialect::Posix`] (its `Long`
    /// is always `--`); always `false` for `Short`/`Positional`/`DashDash`.
    pub double_dash: bool,
    /// True for the `/flag` spelling under [`Dialect::Msbuild`], which is MSBuild's own switch
    /// syntax rather than dotnet's CLI syntax -- `/l:` is MSBuild's logger-assembly switch, not
    /// dotnet's `-l`/`--logger`. Always `false` otherwise.
    pub slash: bool,
}

impl<'a> Token<'a> {
    /// This token's value, whether attached (`--flag=value`, `-fvalue`) or consumed as a
    /// separate token (`--flag value`, `-f value`). `None` for a boolean flag, an unrecognized
    /// flag, or a non-flag token. `tokens` must be the same slice `self` came from.
    pub fn value(&self, tokens: &[Token<'a>]) -> Option<&'a str> {
        self.attached
            .or_else(|| self.linked.map(|idx| tokens[idx].text))
    }

    /// True for a genuine free-standing positional: `Positional` kind, not itself consumed as
    /// some preceding flag's separate-token value (`Token::linked`).
    pub fn is_free_positional(&self) -> bool {
        self.kind == TokenKind::Positional && self.linked.is_none()
    }
}

/// True if `text` is a non-empty run of ASCII digits, e.g. a `Short` token's text for `-20`
/// (git/head/tail's `-N` count shorthand — see [`TokenKind::Short`]). Exposed so callers that
/// need to tell "this Short token is a digit-run flag" from "this Short token is a single
/// boolean-flag letter" don't re-derive the same predicate the tokenizer itself already used to
/// decide clustering.
pub fn is_digit_run(text: &str) -> bool {
    !text.is_empty() && text.bytes().all(|b| b.is_ascii_digit())
}

/// True if `text` (a `Long` token's name) matches `name` under `dialect`'s naming rules: exact
/// for [`Dialect::Posix`], ASCII case-insensitive for [`Dialect::Msbuild`] (MSBuild-ecosystem
/// tools fold case broadly, e.g. `/nologo` and `/NoLogo` are equally valid).
fn flag_name_matches(text: &str, name: &str, dialect: Dialect) -> bool {
    match dialect {
        Dialect::Msbuild => text.eq_ignore_ascii_case(name),
        Dialect::Posix => text == name,
    }
}

/// Index into `tokens` of the `--` boundary, if one was emitted (see [`TokenKind::DashDash`]).
/// `tokens[i].source_index` recovers its position in the original args slice, for a caller that
/// needs to insert/compare against raw arg indices rather than the token vec's own index.
pub fn dashdash_index(tokens: &[Token<'_>]) -> Option<usize> {
    tokens.iter().position(|t| t.kind == TokenKind::DashDash)
}

/// The tokens before the `--` boundary, or all of them when there is none. Under
/// [`Dialect::Msbuild`] classification continues past `--` (it forwards arguments rather than
/// ending option parsing), so a lookup for the tool's *own* flags has to slice here first --
/// otherwise it reads what the user forwarded to the test runner as if dotnet had seen it.
pub fn before_dashdash<'t, 'a>(tokens: &'t [Token<'a>]) -> &'t [Token<'a>] {
    match dashdash_index(tokens) {
        Some(index) => &tokens[..index],
        None => tokens,
    }
}

/// Where RTK's own flags have to be spliced into `args`: before the user's `--`, since
/// anything past the boundary is a pathspec or an argument forwarded to another program, not
/// an option the tool will read. `args_len` when there is no boundary.
pub fn injection_point(tokens: &[Token<'_>], args_len: usize) -> usize {
    dashdash_index(tokens)
        .map(|index| tokens[index].source_index)
        .unwrap_or(args_len)
}

/// True if `tokens` has a `--` boundary at all.
pub fn has_dashdash(tokens: &[Token<'_>]) -> bool {
    dashdash_index(tokens).is_some()
}

/// True if `name` (matched per `dialect`) appears as a `Long` token anywhere in `tokens`. Under
/// `Dialect::Msbuild`, this matches `-flag`/`--flag`/`/flag` uniformly — correct only for
/// legacy MSBuild.exe passthrough switches (`nologo`, `bl`, `v`); see [`has_double_dash_flag`]
/// for anything else.
pub fn has_flag(tokens: &[Token<'_>], dialect: Dialect, name: &str) -> bool {
    tokens
        .iter()
        .any(|t| t.kind == TokenKind::Long && flag_name_matches(t.text, name, dialect))
}

/// Like [`double_dash_flag_value`], but only reports presence, not the value; only matches a
/// token written with a literal `--` prefix (`Token::double_dash`), not `-flag`/`/flag` under
/// [`Dialect::Msbuild`]. Under that dialect, a single-dash or slash spelling of a modern
/// System.CommandLine option (e.g. dotnet's `--logger`) doesn't just get rejected — it gets
/// misparsed as an unrelated legacy MSBuild switch — so use this (not [`has_flag`]) for any
/// option that isn't a genuine legacy MSBuild.exe passthrough switch.
pub fn has_double_dash_flag(tokens: &[Token<'_>], dialect: Dialect, name: &str) -> bool {
    tokens.iter().any(|t| is_double_dash_flag(t, dialect, name))
}

/// This flag's value, if `name` (matched per `dialect`) appears as a `Long` token written with
/// a literal `--` prefix (`Token::double_dash`) anywhere in `tokens`. See
/// [`has_double_dash_flag`] for why this distinction is load-bearing under `Dialect::Msbuild`.
pub fn double_dash_flag_value<'a>(
    tokens: &[Token<'a>],
    dialect: Dialect,
    name: &str,
) -> Option<&'a str> {
    tokens
        .iter()
        .find(|t| is_double_dash_flag(t, dialect, name))
        .and_then(|t| t.value(tokens))
}

/// Every value for `name` (matched per `dialect`), in order, for a `--`-prefixed flag that can
/// legitimately repeat (e.g. dotnet test's `--logger`, usable more than once) — unlike
/// [`double_dash_flag_value`], which only reports the first match. Occurrences with no value are
/// skipped rather than yielding `None`.
pub fn double_dash_flag_values<'a, 't>(
    tokens: &'t [Token<'a>],
    dialect: Dialect,
    name: &'t str,
) -> impl Iterator<Item = &'a str> + 't {
    tokens
        .iter()
        .filter(move |t| is_double_dash_flag(t, dialect, name))
        .filter_map(|t| t.value(tokens))
}

/// Shared match predicate behind [`has_double_dash_flag`]/[`double_dash_flag_value`]/
/// [`double_dash_flag_values`]: a `Long` token written with a literal `--` prefix, matching
/// `name` per `dialect`'s naming rules.
fn is_double_dash_flag(t: &Token<'_>, dialect: Dialect, name: &str) -> bool {
    t.kind == TokenKind::Long && t.double_dash && flag_name_matches(t.text, name, dialect)
}

/// Which CLI's flag grammar to apply. See [`TokenizeOptions::dialect`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dialect {
    /// MSBuild/dotnet-CLI-ish. `-flag`, `--flag`, and `/flag` are all one atomic flag name —
    /// there is no short-flag clustering — and a value can attach via either `=` or `:`
    /// (`--logger:trx` and `--logger=trx` are both valid). Every atomic flag is tagged
    /// `TokenKind::Long` regardless of which prefix introduced it; `TokenKind::Short` is never
    /// produced in this dialect.
    Msbuild,
    /// GNU/POSIX-ish: git, cargo, rg, golangci-lint. `-xyz` is a cluster of short flags,
    /// scanned char by char; only `=` attaches a value to a long flag.
    Posix,
}

/// Tokenizes `args` into [`Token`]s using [`Dialect::Posix`] conventions. `takes_value(kind,
/// name)` decides whether a flag with no attached value should consume the following token as
/// its separate value; never panics, a value-taking flag with nothing left to consume simply
/// gets `attached: None, linked: None`.
pub fn tokenize<'a, T: AsRef<str>>(
    args: &'a [T],
    takes_value: &dyn Fn(TokenKind, &str) -> bool,
) -> Vec<Token<'a>> {
    tokenize_with_options(args, takes_value, TokenizeOptions::default())
}

/// Grammar quirks beyond the base `takes_value` predicate, for a tool that needs
/// [`tokenize_with_options`]. Defaults (via [`Default`]) match plain [`tokenize`]'s behavior, so
/// a caller only sets the field(s) it actually needs.
///
/// `tokenize_with_options` is generic over `T: AsRef<str>`, not `OsStr`/`OsString`: `OsStr`
/// exposes almost no string-manipulation API (no `strip_prefix`, `split_once`), so tokenizing
/// it would mean re-deriving that machinery byte-by-byte the way `clap_lex` does internally.
pub struct TokenizeOptions<'p> {
    /// Which CLI's flag grammar to apply. Defaults to [`Dialect::Posix`].
    pub dialect: Dialect,
    /// Answers "may this `Short` flag consume a separate next-token value here" for a flag
    /// `takes_value` already said takes *some* value with an empty same-arg remainder; `is_solo`
    /// is true only when the flag is the entire arg on its own (e.g. `-n`), false when clustered
    /// with anything else (e.g. the `n` in `-cn`). git is the motivating case (see `git.rs`'s
    /// `log_takes_separate_value`) but nothing here is git-specific. Defaults to always eligible.
    pub takes_separate_value: &'p dyn Fn(&str, bool) -> bool,
    /// Lets specific flags consume a not-yet-seen literal `--` as their value instead of
    /// treating it as the end-of-options boundary. Confirmed this is a per-*tool*, not per-flag,
    /// split: grep/rg swallow `--` as any value-taking flag's value (`-A`/`-m`/`-e`/`--context`/
    /// `--file`, ...), while git/cargo reject it as a value regardless of which flag is asking.
    /// So a grep/rg caller passes the same predicate as `takes_value`; defaults to `false`.
    pub claims_literal_dash_dash: &'p dyn Fn(TokenKind, &str) -> bool,
}

impl Default for TokenizeOptions<'_> {
    fn default() -> Self {
        Self {
            dialect: Dialect::Posix,
            takes_separate_value: &|_name, _is_solo| true,
            claims_literal_dash_dash: &|_kind, _name| false,
        }
    }
}

/// Like [`tokenize`], but with [`TokenizeOptions`] for the handful of tools whose grammar needs
/// more than a plain `takes_value` predicate to classify correctly.
pub fn tokenize_with_options<'a, T: AsRef<str>>(
    args: &'a [T],
    takes_value: &dyn Fn(TokenKind, &str) -> bool,
    options: TokenizeOptions<'_>,
) -> Vec<Token<'a>> {
    tokenize_dialect_ex(
        args,
        options.dialect,
        takes_value,
        options.takes_separate_value,
        options.claims_literal_dash_dash,
    )
}

/// Groups the mutable scan state threaded through [`tokenize_dialect_ex`]'s helper methods
/// (`push_atomic_flag`/`link_next_value`), so adding a future piece of shared state (as this
/// module already had to once, adding `takes_separate_value`) means adding one field instead of
/// a parameter to every helper and every call site.
struct Scanner<'a, 'p, T> {
    tokens: Vec<Token<'a>>,
    args: &'a [T],
    i: usize,
    dialect: Dialect,
    emitted_dash_dash: bool,
    takes_value: &'p dyn Fn(TokenKind, &str) -> bool,
    takes_separate_value: &'p dyn Fn(&str, bool) -> bool,
    claims_literal_dash_dash: &'p dyn Fn(TokenKind, &str) -> bool,
}

impl<'a, 'p, T: AsRef<str>> Scanner<'a, 'p, T> {
    /// Pushes one atomic (non-clustering) flag token — used for `--flag` in both dialects, and
    /// for `-flag`/`/flag` in [`Dialect::Msbuild`]. `rest` is the flag text with its prefix
    /// already stripped; `prefix` records which one it was. Only the `/flag` spelling is barred
    /// from consuming a separate value: an MSBuild switch attaches its value with `:`
    /// (`/bl:x.binlog`), so `/r` (MSBuild's `restore`) must not swallow the token after it the
    /// way dotnet's own `-r <rid>` does.
    fn push_atomic_flag(&mut self, rest: &'a str, prefix: FlagPrefix) {
        let (name, attached) = split_attached(rest, self.dialect);
        let flag_index = self.tokens.len();
        let source_index = self.i;
        self.tokens.push(Token {
            attached,
            ..token(TokenKind::Long, name, source_index, prefix)
        });
        self.i += 1;

        if attached.is_none()
            && prefix != FlagPrefix::Slash
            && (self.takes_value)(TokenKind::Long, name)
            && self.link_next_value(flag_index, self.i)
        {
            self.i += 1;
        }
    }

    /// If `self.args[value_index]` exists and isn't the still-unseen boundary `--`, pushes it as
    /// a `Positional` token linked to `flag_index` (and links `flag_index` back to it). Returns
    /// whether a value was consumed; does *not* itself advance `self.i`. The still-unseen `--`
    /// can be swallowed as a value only if `self.claims_literal_dash_dash` says this flag claims
    /// it (see [`TokenizeOptions::claims_literal_dash_dash`]).
    fn link_next_value(&mut self, flag_index: usize, value_index: usize) -> bool {
        let Some(next) = self.args.get(value_index) else {
            return false;
        };
        if next.as_ref() == "--" && !self.emitted_dash_dash {
            let flag = &self.tokens[flag_index];
            if !(self.claims_literal_dash_dash)(flag.kind, flag.text) {
                return false;
            }
        }
        let token_index = self.tokens.len();
        self.tokens.push(Token {
            linked: Some(flag_index),
            ..positional(next.as_ref(), value_index)
        });
        self.tokens[flag_index].linked = Some(token_index);
        true
    }
}

/// Core implementation shared by every public `tokenize*` entry point. See
/// [`TokenizeOptions`] for `takes_separate_value`/`claims_literal_dash_dash`'s contracts.
fn tokenize_dialect_ex<'a, T: AsRef<str>>(
    args: &'a [T],
    dialect: Dialect,
    takes_value: &dyn Fn(TokenKind, &str) -> bool,
    takes_separate_value: &dyn Fn(&str, bool) -> bool,
    claims_literal_dash_dash: &dyn Fn(TokenKind, &str) -> bool,
) -> Vec<Token<'a>> {
    let mut scanner = Scanner {
        tokens: Vec::with_capacity(args.len()),
        args,
        i: 0,
        dialect,
        emitted_dash_dash: false,
        takes_value,
        takes_separate_value,
        claims_literal_dash_dash,
    };

    while scanner.i < scanner.args.len() {
        let arg = scanner.args[scanner.i].as_ref();

        // Posix stops classifying at `--`; Msbuild's `--` is a forwarding boundary, so it keeps
        // classifying flags past it (see TokenKind::DashDash).
        if scanner.emitted_dash_dash && scanner.dialect == Dialect::Posix {
            scanner.tokens.push(positional(arg, scanner.i));
            scanner.i += 1;
            continue;
        }

        if arg == "--" {
            if scanner.emitted_dash_dash {
                // A second (or later) literal "--" is never itself the boundary — it's just
                // ordinary text at this point, in both dialects.
                scanner.tokens.push(positional(arg, scanner.i));
            } else {
                scanner
                    .tokens
                    .push(token(TokenKind::DashDash, "", scanner.i, FlagPrefix::Dash));
                scanner.emitted_dash_dash = true;
            }
            scanner.i += 1;
            continue;
        }

        if let Some(rest) = arg.strip_prefix("--") {
            scanner.push_atomic_flag(rest, FlagPrefix::DashDash);
            continue;
        }

        if scanner.dialect == Dialect::Msbuild {
            if let Some(rest) = arg.strip_prefix('/') {
                // A real MSBuild switch name never contains another '/' -- without this guard,
                // an absolute Unix path would misclassify as a Long flag (e.g. "tmp/results").
                // KNOWN LIMITATION: a single-segment path (`/app`) is indistinguishable from a
                // genuine switch by structure alone; this pure function has no I/O to resolve it
                // the way real MSBuild does (a filesystem check), but the impact is narrow --
                // only the loose flag lookup ([`has_flag`]) is affected.
                let name_part = rest.split(['=', ':']).next().unwrap_or(rest);
                if !rest.is_empty() && !name_part.contains('/') {
                    scanner.push_atomic_flag(rest, FlagPrefix::Slash);
                    continue;
                }
            }
            if arg.len() > 1 && arg.starts_with('-') {
                scanner.push_atomic_flag(&arg[1..], FlagPrefix::Dash);
                continue;
            }
        } else if arg.len() > 1 && arg.starts_with('-') {
            let cluster = &arg[1..];

            if is_digit_run(cluster) {
                scanner.tokens.push(token(
                    TokenKind::Short,
                    cluster,
                    scanner.i,
                    FlagPrefix::Dash,
                ));
                scanner.i += 1;
                continue;
            }

            let mut consumed_next = false;
            let source_index = scanner.i;

            for (offset, ch) in cluster.char_indices() {
                let char_len = ch.len_utf8();
                let char_text = &cluster[offset..offset + char_len];
                let flag_index = scanner.tokens.len();
                scanner.tokens.push(token(
                    TokenKind::Short,
                    char_text,
                    source_index,
                    FlagPrefix::Dash,
                ));

                if (scanner.takes_value)(TokenKind::Short, char_text) {
                    let remainder = &cluster[offset + char_len..];
                    if !remainder.is_empty() {
                        scanner.tokens[flag_index].attached = Some(remainder);
                    } else {
                        // is_solo: offset == 0 with an empty remainder means this char is the
                        // *entire* cluster (the arg was e.g. just "-n"); a later offset, or any
                        // remainder, means it's genuinely clustered with something else.
                        let is_solo = offset == 0;
                        if (scanner.takes_separate_value)(char_text, is_solo) {
                            consumed_next = scanner.link_next_value(flag_index, source_index + 1);
                        }
                    }
                    break;
                }
            }

            scanner.i += if consumed_next { 2 } else { 1 };
            continue;
        }

        scanner.tokens.push(positional(arg, scanner.i));
        scanner.i += 1;
    }

    scanner.tokens
}

/// Splits `s` into `(name, attached_value)` on the first dialect-appropriate separator:
/// `=` only for [`Dialect::Posix`], `=` or `:` (whichever comes first) for
/// [`Dialect::Msbuild`] (`--logger:trx` and `--logger=trx` are both valid dotnet CLI syntax).
fn split_attached(s: &str, dialect: Dialect) -> (&str, Option<&str>) {
    let sep_pos = match dialect {
        Dialect::Posix => s.find('='),
        Dialect::Msbuild => s.find(['=', ':']),
    };
    match sep_pos {
        Some(pos) => (&s[..pos], Some(&s[pos + 1..])),
        None => (s, None),
    }
}

/// Base constructor for a freshly-scanned token: `attached`/`linked` default to `None`. Every
/// token-construction site builds on this via struct-update syntax instead of a full literal.
fn token(kind: TokenKind, text: &str, source_index: usize, prefix: FlagPrefix) -> Token<'_> {
    Token {
        kind,
        text,
        attached: None,
        linked: None,
        source_index,
        double_dash: prefix == FlagPrefix::DashDash,
        slash: prefix == FlagPrefix::Slash,
    }
}

/// How a flag was spelled. Under [`Dialect::Msbuild`] all three tokenize as `Long`, but they
/// are not interchangeable: MSBuild's `/flag` attaches its value with `:` and never consumes
/// the next argument, while dotnet's own `-flag`/`--flag` do.
#[derive(Clone, Copy, PartialEq, Eq)]
enum FlagPrefix {
    DashDash,
    Dash,
    Slash,
}

fn positional(text: &str, source_index: usize) -> Token<'_> {
    token(TokenKind::Positional, text, source_index, FlagPrefix::Dash)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn owned(args: &[&str]) -> Vec<String> {
        args.iter().map(|s| s.to_string()).collect()
    }

    fn no_values(_kind: TokenKind, _name: &str) -> bool {
        false
    }

    #[test]
    fn empty_args_yield_no_tokens() {
        let args = owned(&[]);
        assert!(tokenize(&args, &no_values).is_empty());
    }

    #[test]
    fn dash_p_after_double_dash_is_positional_not_a_flag() {
        // Regression: `git log -- -p` must not misread the pathspec "-p" as the patch flag
        // (rtk commits 40e4f3a, f8d636d).
        let args = owned(&["--", "-p"]);
        let tokens = tokenize(&args, &no_values);

        assert_eq!(tokens[0].kind, TokenKind::DashDash);
        assert_eq!(tokens[1].kind, TokenKind::Positional);
        assert_eq!(tokens[1].text, "-p");
    }

    #[test]
    fn second_double_dash_is_positional_text_not_another_separator() {
        let args = owned(&["--", "--", "file"]);
        let tokens = tokenize(&args, &no_values);

        assert_eq!(tokens[0].kind, TokenKind::DashDash);
        assert_eq!(tokens[1].kind, TokenKind::Positional);
        assert_eq!(tokens[1].text, "--");
        assert_eq!(tokens[2].kind, TokenKind::Positional);
        assert_eq!(tokens[2].text, "file");
    }

    #[test]
    fn value_taking_long_flag_consumes_and_links_next_token() {
        // Regression: `--grep -p` must treat "-p" as --grep's value, not the patch flag
        // (rtk commits 9bbf55c, 3cc80b2).
        let args = owned(&["--grep", "-p"]);
        let tokens = tokenize(&args, &|kind, name| {
            kind == TokenKind::Long && name == "grep"
        });

        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].kind, TokenKind::Long);
        assert_eq!(tokens[0].text, "grep");
        assert_eq!(tokens[0].linked, Some(1));
        assert_eq!(tokens[1].kind, TokenKind::Positional);
        assert_eq!(tokens[1].text, "-p");
        assert_eq!(tokens[1].linked, Some(0));
    }

    #[test]
    fn value_taking_long_flag_never_swallows_the_unseen_boundary_dashdash() {
        // Regression: verified against real git that a value-taking flag can never claim the
        // still-unseen boundary "--" as its value -- `git log --grep -- pattern` fails with
        // "Option '--grep' requires a value" rather than treating "--" as the search pattern.
        let args = owned(&["--grep", "--", "pattern"]);
        let tokens = tokenize(&args, &|kind, name| {
            kind == TokenKind::Long && name == "grep"
        });

        assert_eq!(tokens[0].kind, TokenKind::Long);
        assert_eq!(tokens[0].text, "grep");
        assert_eq!(
            tokens[0].linked, None,
            "--grep must not claim -- as its value"
        );
        assert_eq!(tokens[1].kind, TokenKind::DashDash);
        assert_eq!(tokens[2].kind, TokenKind::Positional);
        assert_eq!(tokens[2].text, "pattern");
    }

    #[test]
    fn value_taking_short_flag_never_swallows_the_unseen_boundary_dashdash() {
        let args = owned(&["-A", "--", "pattern"]);
        let takes = |kind: TokenKind, name: &str| kind == TokenKind::Short && name == "A";
        let tokens = tokenize(&args, &takes);

        assert_eq!(tokens[0].kind, TokenKind::Short);
        assert_eq!(tokens[0].text, "A");
        assert_eq!(tokens[0].linked, None, "-A must not claim -- as its value");
        assert_eq!(tokens[1].kind, TokenKind::DashDash);
        assert_eq!(tokens[2].text, "pattern");
    }

    #[test]
    fn value_taking_flag_may_consume_a_dashdash_after_the_boundary_was_already_emitted() {
        // Once past the boundary, a further "--" is ordinary text and fair game as a value --
        // verified against real git: `git log -- -- pattern` succeeds (both are pathspecs).
        // Msbuild is the dialect that keeps classifying flags after the boundary, so it's the
        // one where a flag could even encounter a second "--" as its candidate value.
        let args = owned(&["--", "--logger", "--"]);
        let takes = |kind: TokenKind, name: &str| kind == TokenKind::Long && name == "logger";
        let tokens = tokenize_with_options(
            &args,
            &takes,
            TokenizeOptions {
                dialect: Dialect::Msbuild,
                ..Default::default()
            },
        );

        assert_eq!(tokens[0].kind, TokenKind::DashDash);
        assert_eq!(tokens[1].kind, TokenKind::Long);
        assert_eq!(tokens[1].text, "logger");
        assert_eq!(
            tokens[1].linked,
            Some(2),
            "-- after the boundary was already emitted is just text, and --logger may claim it"
        );
        assert_eq!(tokens[2].kind, TokenKind::Positional);
        assert_eq!(tokens[2].text, "--");
    }

    #[test]
    fn attached_long_value_does_not_consult_predicate() {
        let args = owned(&["--grep=-p"]);
        // A predicate that always panics would fail this test if consulted; false is enough
        // to prove it wasn't needed either way, so assert the value came from the "=" form.
        let tokens = tokenize(&args, &no_values);

        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].text, "grep");
        assert_eq!(tokens[0].attached, Some("-p"));
        assert_eq!(tokens[0].linked, None);
    }

    #[test]
    fn optional_value_long_flags_do_not_consume_next_token() {
        // Regression: -U / --unified / --expand-tabs / --max-parents only take an *attached*
        // value; a following bare token is not theirs (rtk commits 705a2f8, 1a1b306).
        for flag in ["unified", "expand-tabs", "max-parents"] {
            let args = owned(&[&format!("--{flag}"), "-p"]);
            let tokens = tokenize(&args, &no_values);

            assert_eq!(tokens[0].linked, None, "--{flag} should not link a value");
            // "-p" is still its own Short("p") token, just not linked to --{flag} as its value.
            assert_eq!(tokens[1].kind, TokenKind::Short);
            assert_eq!(tokens[1].text, "p");
            assert_eq!(
                tokens[1].linked, None,
                "-p after --{flag} must stay independent"
            );
        }
    }

    #[test]
    fn required_value_long_flags_do_consume_next_token() {
        // --diff-algorithm/--diff-filter take a required, separate-token value (rtk commit
        // 84169e2).
        for flag in ["diff-algorithm", "diff-filter"] {
            let args = owned(&[&format!("--{flag}"), "-p"]);
            let takes = |kind: TokenKind, name: &str| {
                kind == TokenKind::Long && (name == "diff-algorithm" || name == "diff-filter")
            };
            let tokens = tokenize(&args, &takes);

            assert_eq!(tokens[0].linked, Some(1), "--{flag} should link its value");
            assert_eq!(tokens[1].text, "-p");
        }
    }

    #[test]
    fn value_taking_flag_at_end_of_args_degrades_gracefully() {
        let args = owned(&["--grep"]);
        let tokens = tokenize(&args, &|kind, name| {
            kind == TokenKind::Long && name == "grep"
        });

        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].attached, None);
        assert_eq!(tokens[0].linked, None);
    }

    #[test]
    fn short_cluster_of_booleans_yields_one_token_per_char() {
        let args = owned(&["-riI"]);
        let tokens = tokenize(&args, &no_values);

        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].text, "r");
        assert_eq!(tokens[1].text, "i");
        assert_eq!(tokens[2].text, "I");
        assert!(tokens.iter().all(|t| t.kind == TokenKind::Short));
        // All three chars came from the one "-riI" arg.
        assert!(tokens.iter().all(|t| t.source_index == 0));
    }

    #[test]
    fn source_index_distinguishes_one_cluster_from_separate_flags() {
        // "-rn" (one arg, one cluster) vs "-r" "-n" (two separate args) classify
        // identically char-by-char, but a caller that needs to know whether they
        // were typed together can tell via source_index.
        let clustered = owned(&["-rn"]);
        let tokens = tokenize(&clustered, &no_values);
        assert_eq!(tokens[0].source_index, tokens[1].source_index);

        let separate = owned(&["-r", "-n"]);
        let tokens = tokenize(&separate, &no_values);
        assert_ne!(tokens[0].source_index, tokens[1].source_index);
    }

    #[test]
    fn short_cluster_value_flag_takes_attached_remainder() {
        let args = owned(&["-A3"]);
        let takes = |kind: TokenKind, name: &str| kind == TokenKind::Short && name == "A";
        let tokens = tokenize(&args, &takes);

        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].text, "A");
        assert_eq!(tokens[0].attached, Some("3"));
        assert_eq!(tokens[0].linked, None);
    }

    #[test]
    fn short_flag_without_attached_remainder_consumes_next_token() {
        let args = owned(&["-A", "3"]);
        let takes = |kind: TokenKind, name: &str| kind == TokenKind::Short && name == "A";
        let tokens = tokenize(&args, &takes);

        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].linked, Some(1));
        assert_eq!(tokens[1].text, "3");
        assert_eq!(tokens[1].linked, Some(0));
    }

    #[test]
    fn short_cluster_stops_consuming_chars_after_value_taking_one() {
        // "-rA3": r is boolean, A takes the attached "3", nothing after A is scanned.
        let args = owned(&["-rA3"]);
        let takes = |kind: TokenKind, name: &str| kind == TokenKind::Short && name == "A";
        let tokens = tokenize(&args, &takes);

        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "r");
        assert_eq!(tokens[1].text, "A");
        assert_eq!(tokens[1].attached, Some("3"));
    }

    #[test]
    fn digit_run_short_flag_stays_one_token_not_a_cluster() {
        // git log/head/tail's "-20" limit shorthand must not decompose into Short('2'),
        // Short('0') — there's no such thing as boolean digit flags.
        let args = owned(&["-20"]);
        let tokens = tokenize(&args, &no_values);

        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].kind, TokenKind::Short);
        assert_eq!(tokens[0].text, "20");
    }

    #[test]
    fn bare_single_dash_is_positional() {
        let args = owned(&["-"]);
        let tokens = tokenize(&args, &no_values);

        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].kind, TokenKind::Positional);
        assert_eq!(tokens[0].text, "-");
    }

    #[test]
    fn plain_positionals_pass_through_unclassified() {
        let args = owned(&["main", "feature/auth"]);
        let tokens = tokenize(&args, &no_values);

        assert_eq!(tokens.len(), 2);
        assert!(tokens.iter().all(|t| t.kind == TokenKind::Positional));
    }

    // --- Dialect::Msbuild ---

    #[test]
    fn msbuild_single_dash_flag_is_atomic_not_a_cluster() {
        // dotnet's "-nologo" is one flag name, not a POSIX cluster of n/o/l/o/g/o.
        let args = owned(&["-nologo"]);
        let tokens = tokenize_with_options(
            &args,
            &no_values,
            TokenizeOptions {
                dialect: Dialect::Msbuild,
                ..Default::default()
            },
        );

        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].kind, TokenKind::Long);
        assert_eq!(tokens[0].text, "nologo");
    }

    #[test]
    fn msbuild_slash_flag_never_consumes_a_separate_value() {
        // `/r` is MSBuild's boolean `restore`, not dotnet's `-r <rid>`: an MSBuild switch takes
        // its value attached with `:`, so `/r` must leave the next arg alone. Reading it as a
        // value hid a following `-bl:<file>` from dotnet's own binlog detection.
        let takes_value = |kind: TokenKind, name: &str| kind == TokenKind::Long && name == "r";
        let slash = owned(&["/r", "-bl:my.binlog"]);
        let tokens = tokenize_with_options(
            &slash,
            &takes_value,
            TokenizeOptions {
                dialect: Dialect::Msbuild,
                ..Default::default()
            },
        );
        assert_eq!(tokens[0].text, "r");
        assert_eq!(tokens[0].linked, None);
        assert_eq!(tokens[1].kind, TokenKind::Long);
        assert_eq!(tokens[1].text, "bl");
        assert_eq!(tokens[1].attached, Some("my.binlog"));

        // The dash spelling is dotnet's own `-r <rid>`, which does consume the next token.
        let dash = owned(&["-r", "linux-x64"]);
        let tokens = tokenize_with_options(
            &dash,
            &takes_value,
            TokenizeOptions {
                dialect: Dialect::Msbuild,
                ..Default::default()
            },
        );
        assert_eq!(tokens[0].value(&tokens), Some("linux-x64"));
    }

    #[test]
    fn msbuild_slash_prefix_is_recognized_as_a_flag() {
        let args = owned(&["/nologo"]);
        let tokens = tokenize_with_options(
            &args,
            &no_values,
            TokenizeOptions {
                dialect: Dialect::Msbuild,
                ..Default::default()
            },
        );

        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].kind, TokenKind::Long);
        assert_eq!(tokens[0].text, "nologo");
    }

    #[test]
    fn msbuild_slash_alone_is_positional() {
        let args = owned(&["/"]);
        let tokens = tokenize_with_options(
            &args,
            &no_values,
            TokenizeOptions {
                dialect: Dialect::Msbuild,
                ..Default::default()
            },
        );

        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].kind, TokenKind::Positional);
        assert_eq!(tokens[0].text, "/");
    }

    #[test]
    fn msbuild_absolute_path_is_positional_not_a_flag() {
        // Real MSBuild never treats a multi-segment "/a/b" as a switch attempt.
        let takes = |kind: TokenKind, name: &str| kind == TokenKind::Long && name == "nologo";
        let args = owned(&["/tmp/results"]);
        let tokens = tokenize_with_options(
            &args,
            &takes,
            TokenizeOptions {
                dialect: Dialect::Msbuild,
                ..Default::default()
            },
        );

        assert_eq!(tokens[0].kind, TokenKind::Positional);
        assert_eq!(tokens[0].text, "/tmp/results");
    }

    #[test]
    fn msbuild_single_segment_slash_flag_is_still_a_flag() {
        // A genuine single-segment MSBuild switch (no internal '/') must still classify as Long,
        // including when it carries an attached value whose own text contains '/'.
        let args = owned(&["/nologo", "/p:OutDir=/tmp/out"]);
        let tokens = tokenize_with_options(
            &args,
            &no_values,
            TokenizeOptions {
                dialect: Dialect::Msbuild,
                ..Default::default()
            },
        );

        assert_eq!(tokens[0].kind, TokenKind::Long);
        assert_eq!(tokens[0].text, "nologo");
        assert_eq!(tokens[1].kind, TokenKind::Long);
        assert_eq!(tokens[1].text, "p");
        assert_eq!(tokens[1].attached, Some("OutDir=/tmp/out"));
    }

    #[test]
    fn msbuild_colon_and_equals_both_attach_a_value() {
        for arg in ["--logger:trx", "--logger=trx"] {
            let args = owned(&[arg]);
            let tokens = tokenize_with_options(
                &args,
                &no_values,
                TokenizeOptions {
                    dialect: Dialect::Msbuild,
                    ..Default::default()
                },
            );

            assert_eq!(tokens[0].text, "logger", "for {arg}");
            assert_eq!(tokens[0].attached, Some("trx"), "for {arg}");
        }
    }

    #[test]
    fn msbuild_separate_token_value_still_works() {
        let args = owned(&["--results-directory", "/tmp/out"]);
        let takes =
            |kind: TokenKind, name: &str| kind == TokenKind::Long && name == "results-directory";
        let tokens = tokenize_with_options(
            &args,
            &takes,
            TokenizeOptions {
                dialect: Dialect::Msbuild,
                ..Default::default()
            },
        );

        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].linked, Some(1));
        assert_eq!(tokens[1].text, "/tmp/out");
    }

    #[test]
    fn msbuild_dashdash_is_a_forwarding_boundary_not_end_of_options() {
        // dotnet's `--` hands the rest to a different receiving parser (the VSTest/MTP test
        // host); unlike Posix, that doesn't stop classification -- flags after it (e.g.
        // --report-trx, forwarded to the test host) must still be recognized as flags.
        let args = owned(&["--", "-nologo"]);
        let tokens = tokenize_with_options(
            &args,
            &no_values,
            TokenizeOptions {
                dialect: Dialect::Msbuild,
                ..Default::default()
            },
        );

        assert_eq!(tokens[0].kind, TokenKind::DashDash);
        assert_eq!(tokens[1].kind, TokenKind::Long);
        assert_eq!(tokens[1].text, "nologo");
    }

    #[test]
    fn msbuild_flag_after_dashdash_still_consumes_its_separate_value() {
        // Regression: `dotnet test <proj> -- --results-directory /tmp/out` -- the value must
        // still link to its flag even though it's past `--`, matching real forwarded-flag
        // semantics (unlike Posix, where nothing after `--` is ever a flag at all).
        let args = owned(&["--", "--results-directory", "/tmp/out"]);
        let takes =
            |kind: TokenKind, name: &str| kind == TokenKind::Long && name == "results-directory";
        let tokens = tokenize_with_options(
            &args,
            &takes,
            TokenizeOptions {
                dialect: Dialect::Msbuild,
                ..Default::default()
            },
        );

        assert_eq!(tokens[0].kind, TokenKind::DashDash);
        assert_eq!(tokens[1].kind, TokenKind::Long);
        assert_eq!(tokens[1].linked, Some(2));
        assert_eq!(tokens[2].text, "/tmp/out");
    }

    #[test]
    fn msbuild_second_dashdash_is_positional_not_another_boundary() {
        // Regression: DashDash must be emitted exactly once even under Msbuild, where
        // classification doesn't stop at `--` (unlike Posix, where a second `--` already falls
        // into the seen_dash_dash positional catch-all for free).
        let args = owned(&["--", "a", "--", "b"]);
        let tokens = tokenize_with_options(
            &args,
            &no_values,
            TokenizeOptions {
                dialect: Dialect::Msbuild,
                ..Default::default()
            },
        );

        assert_eq!(tokens[0].kind, TokenKind::DashDash);
        assert_eq!(tokens[1].kind, TokenKind::Positional);
        assert_eq!(tokens[1].text, "a");
        assert_eq!(tokens[2].kind, TokenKind::Positional);
        assert_eq!(tokens[2].text, "--");
        assert_eq!(tokens[3].kind, TokenKind::Positional);
        assert_eq!(tokens[3].text, "b");
        assert_eq!(
            tokens
                .iter()
                .filter(|t| t.kind == TokenKind::DashDash)
                .count(),
            1
        );
    }

    #[test]
    fn msbuild_dialect_never_produces_short_tokens() {
        let args = owned(&["-a", "-bc", "/d", "--e"]);
        let tokens = tokenize_with_options(
            &args,
            &no_values,
            TokenizeOptions {
                dialect: Dialect::Msbuild,
                ..Default::default()
            },
        );

        assert!(tokens.iter().all(|t| t.kind != TokenKind::Short));
    }

    #[test]
    fn posix_dialect_unaffected_by_slash_or_colon() {
        // The default (tokenize == Dialect::Posix) must not gain '/' or ':' handling.
        let args = owned(&["feature/auth", "--pretty:oops"]);
        let tokens = tokenize(&args, &no_values);

        assert_eq!(tokens[0].kind, TokenKind::Positional);
        assert_eq!(tokens[0].text, "feature/auth");
        assert_eq!(tokens[1].kind, TokenKind::Long);
        assert_eq!(tokens[1].text, "pretty:oops");
        assert_eq!(tokens[1].attached, None);
    }

    // --- has_flag / has_double_dash_flag / double_dash_flag_value(s) ---

    #[test]
    fn msbuild_has_flag_is_case_insensitive() {
        let args = owned(&["-NoLogo"]);
        let tokens = tokenize_with_options(
            &args,
            &no_values,
            TokenizeOptions {
                dialect: Dialect::Msbuild,
                ..Default::default()
            },
        );

        assert!(has_flag(&tokens, Dialect::Msbuild, "nologo"));
        assert!(has_flag(&tokens, Dialect::Msbuild, "NOLOGO"));
    }

    #[test]
    fn posix_has_flag_and_flag_value_are_case_sensitive() {
        // git/cargo/rg/golangci-lint don't fold case; "--Grep" is not "--grep".
        let args = owned(&["--Grep"]);
        let tokens = tokenize(&args, &no_values);

        assert!(has_flag(&tokens, Dialect::Posix, "Grep"));
        assert!(!has_flag(&tokens, Dialect::Posix, "grep"));
    }

    #[test]
    fn has_flag_ignores_short_and_positional_tokens() {
        // A Short "n" or a positional literally spelled "nologo" must not satisfy a Long
        // flag-name lookup for "nologo".
        let args = owned(&["-n", "nologo"]);
        let tokens = tokenize(&args, &no_values);

        assert!(!has_flag(&tokens, Dialect::Posix, "nologo"));
        assert!(!has_flag(&tokens, Dialect::Posix, "n"));
    }

    #[test]
    fn double_dash_flag_value_is_case_insensitive_but_prefix_strict() {
        let args = owned(&["--Logger:trx"]);
        let tokens = tokenize_with_options(
            &args,
            &no_values,
            TokenizeOptions {
                dialect: Dialect::Msbuild,
                ..Default::default()
            },
        );

        assert_eq!(
            double_dash_flag_value(&tokens, Dialect::Msbuild, "logger"),
            Some("trx")
        );
        assert_eq!(
            double_dash_flag_value(&tokens, Dialect::Msbuild, "LOGGER"),
            Some("trx")
        );
    }

    #[test]
    fn double_dash_flag_rejects_single_dash_and_slash_spellings() {
        // Regression: verified against a real dotnet 9 SDK that dotnet's own
        // System.CommandLine-parsed options (unlike legacy MSBuild.exe passthrough switches
        // like -nologo) are double-dash-only -- "-results-directory"/"/results-directory" get
        // misparsed as unrelated MSBuild switches, not treated as this flag at all.
        let args = owned(&["-results-directory", "/tmp/out"]);
        let tokens = tokenize_with_options(
            &args,
            &no_values,
            TokenizeOptions {
                dialect: Dialect::Msbuild,
                ..Default::default()
            },
        );

        assert!(has_flag(&tokens, Dialect::Msbuild, "results-directory"));
        assert!(!has_double_dash_flag(
            &tokens,
            Dialect::Msbuild,
            "results-directory"
        ));
        assert_eq!(
            double_dash_flag_value(&tokens, Dialect::Msbuild, "results-directory"),
            None
        );

        let args = owned(&["/results-directory", "/tmp/out"]);
        let tokens = tokenize_with_options(
            &args,
            &no_values,
            TokenizeOptions {
                dialect: Dialect::Msbuild,
                ..Default::default()
            },
        );
        assert!(!has_double_dash_flag(
            &tokens,
            Dialect::Msbuild,
            "results-directory"
        ));
    }

    #[test]
    fn double_dash_flag_values_reports_every_occurrence_not_just_the_first() {
        // Regression: dotnet test's --logger can legitimately repeat
        // (`--logger "console;verbosity=normal" --logger trx`) -- unlike
        // double_dash_flag_value, which only reports the first match, every occurrence must be
        // checkable.
        let args = owned(&["--logger:console;verbosity=normal", "--logger", "trx"]);
        let takes = |kind: TokenKind, name: &str| kind == TokenKind::Long && name == "logger";
        let tokens = tokenize_with_options(
            &args,
            &takes,
            TokenizeOptions {
                dialect: Dialect::Msbuild,
                ..Default::default()
            },
        );

        let values: Vec<&str> =
            double_dash_flag_values(&tokens, Dialect::Msbuild, "logger").collect();
        assert_eq!(values, vec!["console;verbosity=normal", "trx"]);
    }
}
