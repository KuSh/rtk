//! Shared tokenizer for re-classifying an already-`--`-restored passthrough args slice
//! (see [`crate::core::args_utils::restore_double_dash`]) into flags, their values, and
//! positionals, matching the GNU/POSIX-ish conventions used by git, cargo, rg, and friends.
//!
//! This exists because every `src/cmds/**` filter that needs to know "does this flag consume
//! the next token as its value, and where does `--` end options" was reimplementing that
//! question independently (git.rs's `LogArg`/`consumes_next_token_as_value`, search.rs's
//! `VALUE_FLAGS_SHORT`/`VALUE_FLAGS_LONG`/`ClusterResult`, golangci_cmd.rs's
//! `GLOBAL_FLAGS_WITH_VALUE`), and each one accumulated its own one-off bugs over time.
//! `tokenize` centralizes the classification; callers keep their own list of which flags take a
//! value (that part is inherently per-tool) but pass it in as a predicate instead of
//! reimplementing the token-walking around it.
//!
//! Deliberately *not* merged with `restore_double_dash`, even though every caller has to do
//! both in sequence: `restore_double_dash` needs both the clap-parsed args and the raw process
//! argv to detect what clap's `trailing_var_arg` stripped, and its restored result has to be an
//! owned `Vec<String>` the caller holds in its own `let` binding — `Token<'a>` borrows straight
//! from `args`, so tokenizing a `Vec<String>` built *inside* this module (from a hypothetical
//! internal `restore_double_dash` call) would tie every `Token` to a value that's dropped when
//! the function returns. Same root cause as why this module doesn't build on `clap_lex`.

/// What kind of unit a [`Token`] represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind {
    /// The literal `--` separator. Emitted exactly once, for the first `--` encountered. Under
    /// [`Dialect::Posix`], this ends option parsing: every token after it is `Positional`
    /// unconditionally and `takes_value` is never consulted again (a second or later `--` comes
    /// back as a plain `Positional` with `text == "--"`, matching real git/GNU semantics). Under
    /// [`Dialect::Msbuild`], `--` is an argument-*forwarding* boundary rather than an
    /// end-of-options marker (dotnet's `--` hands the rest to a different receiving parser that
    /// can share flag names with dotnet's own), so classification continues normally past it —
    /// only its position is recorded.
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
}

impl<'a> Token<'a> {
    /// This token's value, whether attached (`--flag=value`, `-fvalue`) or consumed as a
    /// separate token (`--flag value`, `-f value`). `None` for a boolean flag, an unrecognized
    /// flag, or a non-flag token. `tokens` must be the same slice `self` came from.
    pub fn value(&self, tokens: &[Token<'a>]) -> Option<&'a str> {
        self.attached
            .or_else(|| self.linked.map(|idx| tokens[idx].text))
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

/// True if `text` (a `Long` token's name) matches `name` under `dialect`'s naming rules:
/// exact for [`Dialect::Posix`] (git/cargo/rg/golangci-lint are case-sensitive), ASCII
/// case-insensitive for [`Dialect::Msbuild`] (Windows/MSBuild-ecosystem tools fold case
/// broadly — this isn't a dotnet-CLI particularity, e.g. classic MSBuild.exe's `/nologo` and
/// `/NoLogo` are equally valid). `text` can't be case-folded once at tokenize time without
/// giving up `Token`'s zero-copy `&'a str` (there's no borrowed "lowercased" view), so instead
/// every dialect-aware lookup goes through this and [`has_flag`]/[`double_dash_flag_value`].
fn flag_name_matches(text: &str, name: &str, dialect: Dialect) -> bool {
    match dialect {
        Dialect::Msbuild => text.eq_ignore_ascii_case(name),
        Dialect::Posix => text == name,
    }
}

/// True if `name` (matched per `dialect`) appears as a `Long` token anywhere in `tokens`.
///
/// Under `Dialect::Msbuild`, this deliberately matches `-flag`/`--flag`/`/flag` uniformly —
/// correct for genuine legacy MSBuild.exe passthrough switches (`nologo`, `bl`, `v`), but
/// *not* correct in general for dotnet's own System.CommandLine-parsed options or VSTest
/// options forwarded through `dotnet test`. See [`has_double_dash_flag`] before using this for
/// any Msbuild-dialect flag that isn't one of those legacy switches.
pub fn has_flag(tokens: &[Token<'_>], dialect: Dialect, name: &str) -> bool {
    tokens
        .iter()
        .any(|t| t.kind == TokenKind::Long && flag_name_matches(t.text, name, dialect))
}

/// Like [`double_dash_flag_value`], but only reports presence, not the value; only matches a
/// token written with a literal `--` prefix (`Token::double_dash`), not `-flag`/`/flag` under
/// [`Dialect::Msbuild`].
///
/// Under `Dialect::Msbuild`, [`has_flag`] treats `-flag`, `--flag`, and `/flag` as
/// interchangeable — correct for the handful of options MSBuild.exe itself has
/// always accepted in all three forms (`-nologo`, `-bl`, `-v`/`-verbosity`) and forwards
/// straight through to. It is *not* correct for dotnet's own System.CommandLine-parsed options
/// (`dotnet format`'s `--verify-no-changes`/`--report`, `dotnet test`'s `--logger`/
/// `--results-directory`) or VSTest-console options forwarded through `dotnet test`: verified
/// against a real dotnet 9 SDK that these are double-dash-only, and a single-dash or slash
/// spelling doesn't just get rejected -- it gets *misparsed* as an unrelated MSBuild switch
/// (`-results-directory` is read as MSBuild's own unrecognized switch "results-directory";
/// `-logger` collides with MSBuild's own `-logger` switch, which expects a logger assembly
/// spec, not "trx"). Use this (or [`double_dash_flag_value`]/[`double_dash_flag_values`]) for
/// any option that isn't a genuine legacy MSBuild.exe passthrough switch.
pub fn has_double_dash_flag(tokens: &[Token<'_>], dialect: Dialect, name: &str) -> bool {
    tokens.iter().any(|t| {
        t.kind == TokenKind::Long && t.double_dash && flag_name_matches(t.text, name, dialect)
    })
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
        .find(|t| {
            t.kind == TokenKind::Long && t.double_dash && flag_name_matches(t.text, name, dialect)
        })
        .and_then(|t| t.value(tokens))
}

/// Every value for `name` (matched per `dialect`), in order, for a `--`-prefixed flag that can
/// legitimately be repeated (e.g. dotnet test's `--logger console;verbosity=normal --logger
/// trx`, where each occurrence must be checked, not just the first —
/// [`double_dash_flag_value`] only reports the first match). Occurrences with no value (bare
/// flag, or a value-taking flag with nothing left to consume) are skipped rather than yielding
/// `None`. See [`has_double_dash_flag`] for why the `--`-only restriction is load-bearing under
/// `Dialect::Msbuild`.
pub fn double_dash_flag_values<'a, 't>(
    tokens: &'t [Token<'a>],
    dialect: Dialect,
    name: &'t str,
) -> impl Iterator<Item = &'a str> + 't {
    tokens
        .iter()
        .filter(move |t| {
            t.kind == TokenKind::Long && t.double_dash && flag_name_matches(t.text, name, dialect)
        })
        .filter_map(|t| t.value(tokens))
}

/// Which CLI's flag grammar `tokenize_dialect` should apply. See [`tokenize_dialect`].
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
/// name)` is called for each `Long`/`Short` flag that has no attached value, to decide whether
/// the following whole token should be consumed as its separate-token value; it is never called
/// for tokens at or after `--`.
///
/// Never panics and never fails to classify: a value-taking flag with nothing left to consume
/// simply gets `attached: None, linked: None`, matching RTK's fallback/never-block-the-user
/// convention.
pub fn tokenize<'a, T: AsRef<str>>(
    args: &'a [T],
    takes_value: &dyn Fn(TokenKind, &str) -> bool,
) -> Vec<Token<'a>> {
    tokenize_dialect(args, Dialect::Posix, takes_value)
}

/// Like [`tokenize`], but for tools (git specifically) whose value-taking `Short` flags don't
/// uniformly support a separate-token value the way [`tokenize`]'s default assumes. Real git has
/// two different behaviors that don't fit a single boolean:
///
/// - `-n`/`-l` (`--max-count`/rename-detection-cost shorthand) accept a separate-token value, but
///   *only* when written as their own standalone arg -- confirmed against real git 2.51: `git log
///   -n 2` succeeds, but `git log -cn 2` (clustered with `-c`) fails with "ambiguous argument
///   '2'": clustered, `-n`'s value is the (empty) remainder of the same arg, never the next token.
///   Grep doesn't share this restriction (`grep -im 2 pattern file` works clustered), so it isn't
///   folded into [`tokenize`]'s default.
/// - `-M`/`-U`/`-C`/`-B` (rename/copy/context-detection shorthand) accept *only* an attached value
///   (`-M50`) and never a separate token at all, standalone or clustered -- confirmed against real
///   git that even the standalone form `git log -U 3` fails ("ambiguous argument '3'").
///
/// `takes_separate_value(name, is_solo)` answers "may this `Short` flag consume a separate
/// next-token value here" for a flag `takes_value` already said takes *some* value with an empty
/// same-arg remainder; `is_solo` is true only when the flag is the entire arg on its own (e.g.
/// `-n`), false when clustered with anything else (e.g. the `n` in `-cn`).
pub fn tokenize_git<'a, T: AsRef<str>>(
    args: &'a [T],
    takes_value: &dyn Fn(TokenKind, &str) -> bool,
    takes_separate_value: &dyn Fn(&str, bool) -> bool,
) -> Vec<Token<'a>> {
    tokenize_dialect_ex(args, Dialect::Posix, takes_value, takes_separate_value)
}

/// Like [`tokenize`], but lets the caller pick a [`Dialect`] instead of assuming POSIX
/// conventions.
///
/// Generic over `T: AsRef<str>` so a caller can pass either `&[String]` (the common case,
/// e.g. already-owned args from [`crate::core::args_utils::restore_double_dash`]) or `&[&str]`
/// (handy for tests) without cloning either way — `Token` still borrows straight from `args`,
/// zero-copy. Not generic over `OsStr`/`OsString`: unlike `str`, `OsStr` exposes almost no
/// string-manipulation API by design (no `strip_prefix`, `split_once`, char-boundary slicing),
/// so tokenizing it would mean re-deriving that machinery byte-by-byte the way `clap_lex` does
/// internally — a much bigger change for a case rtk doesn't hit today (its own CLI parsing
/// already assumes UTF-8 args for every subcommand this module serves).
pub fn tokenize_dialect<'a, T: AsRef<str>>(
    args: &'a [T],
    dialect: Dialect,
    takes_value: &dyn Fn(TokenKind, &str) -> bool,
) -> Vec<Token<'a>> {
    tokenize_dialect_ex(args, dialect, takes_value, &|_name, _is_solo| true)
}

/// Core implementation shared by [`tokenize_dialect`] and [`tokenize_git`]. See [`tokenize_git`]
/// for `takes_separate_value`'s contract; [`tokenize_dialect`] passes `|_, _| true`, preserving
/// its original "always eligible" behavior for every existing caller.
fn tokenize_dialect_ex<'a, T: AsRef<str>>(
    args: &'a [T],
    dialect: Dialect,
    takes_value: &dyn Fn(TokenKind, &str) -> bool,
    takes_separate_value: &dyn Fn(&str, bool) -> bool,
) -> Vec<Token<'a>> {
    let mut tokens: Vec<Token<'a>> = Vec::with_capacity(args.len());
    let mut i = 0;
    let mut seen_dash_dash = false;
    let mut emitted_dash_dash = false;

    while i < args.len() {
        let arg = args[i].as_ref();

        if seen_dash_dash {
            tokens.push(positional(arg, i));
            i += 1;
            continue;
        }

        if arg == "--" {
            if emitted_dash_dash {
                // A second (or later) literal "--" is never itself the boundary — it's just
                // ordinary text at this point, in both dialects.
                tokens.push(positional(arg, i));
            } else {
                tokens.push(Token {
                    kind: TokenKind::DashDash,
                    text: "",
                    attached: None,
                    linked: None,
                    source_index: i,
                    double_dash: false,
                });
                emitted_dash_dash = true;
                // In Posix conventions `--` ends option parsing: everything after is a literal
                // positional/pathspec, never a flag. dotnet's `--` means something different —
                // an argument-*forwarding* boundary, not an end-of-options marker: what follows
                // is still real flags, just meant for a different receiving parser (the
                // VSTest/MTP test host) that can share flag names with dotnet's own (e.g.
                // --logger, --results-directory are forwarded VSTest-console options). So only
                // Posix stops classifying here; Msbuild keeps going, just with the separator's
                // position on record via this DashDash token.
                if dialect == Dialect::Posix {
                    seen_dash_dash = true;
                }
            }
            i += 1;
            continue;
        }

        if let Some(rest) = arg.strip_prefix("--") {
            push_atomic_flag(
                &mut tokens,
                args,
                &mut i,
                rest,
                true,
                true,
                emitted_dash_dash,
                dialect,
                takes_value,
            );
            continue;
        }

        if dialect == Dialect::Msbuild {
            if let Some(rest) = arg.strip_prefix('/') {
                // A real MSBuild switch name never contains another '/' (confirmed via a real
                // dotnet 9 SDK, Docker: `dotnet build /abs/path/Project.csproj` builds the
                // absolute Unix path as the project positional, not a switch attempt). Without
                // this guard, an absolute path -- the common case on Linux/macOS -- would be
                // misclassified as a Long flag named e.g. "tmp/results".
                //
                // KNOWN LIMITATION, not fixed here: a single-segment absolute path (`/app`,
                // `/tmp`) is indistinguishable from a genuine switch name by structure alone --
                // confirmed via Docker that real dotnet/MSBuild itself only resolves this
                // ambiguity with a filesystem check (`dotnet build /tmp`, which exists, is
                // accepted as a path attempt; `dotnet build /nonexistentdir`, which doesn't
                // exist, is rejected as "MSB1001: Unknown switch" -- byte-for-byte structural
                // parsing alone can't tell them apart, real dotnet needs a stat() call to do
                // it). This tokenizer is a pure function with no I/O by design, so replicating
                // that exactly isn't possible here; the impact is narrow regardless, since it
                // only matters for the loose flag lookup ([`has_flag`]) and only collides with
                // an actual single-segment path that's spelled exactly like one of the few loose
                // switch names ("nologo", "bl", "v", "verbosity").
                let name_part = rest.split(['=', ':']).next().unwrap_or(rest);
                if !rest.is_empty() && !name_part.contains('/') {
                    push_atomic_flag(
                        &mut tokens,
                        args,
                        &mut i,
                        rest,
                        false,
                        false,
                        emitted_dash_dash,
                        dialect,
                        takes_value,
                    );
                    continue;
                }
            }
            if arg.len() > 1 && arg.starts_with('-') {
                push_atomic_flag(
                    &mut tokens,
                    args,
                    &mut i,
                    &arg[1..],
                    false,
                    true,
                    emitted_dash_dash,
                    dialect,
                    takes_value,
                );
                continue;
            }
        } else if arg.len() > 1 && arg.starts_with('-') {
            let cluster = &arg[1..];

            if is_digit_run(cluster) {
                tokens.push(Token {
                    kind: TokenKind::Short,
                    text: cluster,
                    attached: None,
                    linked: None,
                    source_index: i,
                    double_dash: false,
                });
                i += 1;
                continue;
            }

            let mut consumed_next = false;

            for (offset, ch) in cluster.char_indices() {
                let char_len = ch.len_utf8();
                let char_text = &cluster[offset..offset + char_len];
                let flag_index = tokens.len();
                tokens.push(Token {
                    kind: TokenKind::Short,
                    text: char_text,
                    attached: None,
                    linked: None,
                    source_index: i,
                    double_dash: false,
                });

                if takes_value(TokenKind::Short, char_text) {
                    let remainder = &cluster[offset + char_len..];
                    if !remainder.is_empty() {
                        tokens[flag_index].attached = Some(remainder);
                    } else {
                        // is_solo: offset == 0 with an empty remainder means this char is the
                        // *entire* cluster (the arg was e.g. just "-n"); a later offset, or any
                        // remainder, means it's genuinely clustered with something else.
                        let is_solo = offset == 0;
                        if takes_separate_value(char_text, is_solo) {
                            consumed_next = link_next_value(
                                &mut tokens,
                                args,
                                flag_index,
                                i + 1,
                                emitted_dash_dash,
                            );
                        }
                    }
                    break;
                }
            }

            i += if consumed_next { 2 } else { 1 };
            continue;
        }

        tokens.push(positional(arg, i));
        i += 1;
    }

    tokens
}

/// Pushes one atomic (non-clustering) flag token — used for `--flag` in both dialects, and for
/// `-flag`/`/flag` in [`Dialect::Msbuild`] — splitting off an attached value per `dialect` and,
/// absent one, consulting `takes_value` to maybe consume the next whole token as a separate
/// value. `rest` is the flag text with its prefix (`--`, `-`, or `/`) already stripped;
/// `double_dash` records which prefix that was (see `Token::double_dash`). `emitted_dash_dash`
/// is the caller's current boundary state: the still-unseen boundary `--` can never be
/// swallowed as this flag's value (verified against real git: `git log --grep -- pattern` fails
/// with "Option '--grep' requires a value" rather than treating `--` as the pattern); a `--`
/// encountered after the boundary was already emitted is just ordinary text and fair game
/// (`git log -- -- pattern` works).
#[allow(clippy::too_many_arguments)]
fn push_atomic_flag<'a, T: AsRef<str>>(
    tokens: &mut Vec<Token<'a>>,
    args: &'a [T],
    i: &mut usize,
    rest: &'a str,
    double_dash: bool,
    separate_value: bool,
    emitted_dash_dash: bool,
    dialect: Dialect,
    takes_value: &dyn Fn(TokenKind, &str) -> bool,
) {
    let (name, attached) = split_attached(rest, dialect);
    let flag_index = tokens.len();
    let source_index = *i;
    tokens.push(Token {
        kind: TokenKind::Long,
        text: name,
        attached,
        linked: None,
        source_index,
        double_dash,
    });
    *i += 1;

    if attached.is_none()
        && separate_value
        && takes_value(TokenKind::Long, name)
        && link_next_value(tokens, args, flag_index, *i, emitted_dash_dash)
    {
        *i += 1;
    }
}

/// If `args[value_index]` exists and isn't the still-unseen boundary `--`, pushes it as a
/// `Positional` token linked to `flag_index` (and links `flag_index` back to it). Returns
/// whether a value was consumed. Shared by the `Long`/[`Dialect::Msbuild`]-atomic-flag path and
/// the `Short`-cluster path so the boundary guard lives in exactly one place.
///
/// The still-unseen boundary `--` can never be swallowed as a value (verified against real
/// git: `git log --grep -- pattern` fails with "Option '--grep' requires a value" rather than
/// treating `--` as the pattern) -- once the boundary has already been emitted, a later `--` is
/// just ordinary text and fair game (`git log -- -- pattern` works).
fn link_next_value<'a, T: AsRef<str>>(
    tokens: &mut Vec<Token<'a>>,
    args: &'a [T],
    flag_index: usize,
    value_index: usize,
    emitted_dash_dash: bool,
) -> bool {
    let Some(next) = args.get(value_index) else {
        return false;
    };
    if next.as_ref() == "--" && !emitted_dash_dash {
        return false;
    }
    let token_index = tokens.len();
    tokens.push(Token {
        linked: Some(flag_index),
        ..positional(next.as_ref(), value_index)
    });
    tokens[flag_index].linked = Some(token_index);
    true
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

fn positional(text: &str, source_index: usize) -> Token<'_> {
    Token {
        kind: TokenKind::Positional,
        text,
        attached: None,
        linked: None,
        source_index,
        double_dash: false,
    }
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
        let tokens = tokenize_dialect(&args, Dialect::Msbuild, &takes);

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
        let tokens = tokenize_dialect(&args, Dialect::Msbuild, &no_values);

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
        let tokens = tokenize_dialect(&slash, Dialect::Msbuild, &takes_value);
        assert_eq!(tokens[0].text, "r");
        assert_eq!(tokens[0].linked, None);
        assert_eq!(tokens[1].kind, TokenKind::Long);
        assert_eq!(tokens[1].text, "bl");
        assert_eq!(tokens[1].attached, Some("my.binlog"));

        // The dash spelling is dotnet's own `-r <rid>`, which does consume the next token.
        let dash = owned(&["-r", "linux-x64"]);
        let tokens = tokenize_dialect(&dash, Dialect::Msbuild, &takes_value);
        assert_eq!(tokens[0].value(&tokens), Some("linux-x64"));
    }

    #[test]
    fn msbuild_slash_prefix_is_recognized_as_a_flag() {
        let args = owned(&["/nologo"]);
        let tokens = tokenize_dialect(&args, Dialect::Msbuild, &no_values);

        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].kind, TokenKind::Long);
        assert_eq!(tokens[0].text, "nologo");
    }

    #[test]
    fn msbuild_slash_alone_is_positional() {
        let args = owned(&["/"]);
        let tokens = tokenize_dialect(&args, Dialect::Msbuild, &no_values);

        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].kind, TokenKind::Positional);
        assert_eq!(tokens[0].text, "/");
    }

    #[test]
    fn msbuild_absolute_path_is_positional_not_a_flag() {
        // Regression: confirmed via a real dotnet 9 SDK (Docker) that `dotnet build
        // /abs/path/Project.csproj` builds the absolute Unix path as the project positional --
        // real MSBuild never treats a multi-segment "/a/b" as a switch attempt. Before this fix,
        // any '/'-prefixed token was classified as a Long flag regardless of internal '/'s, so
        // an absolute path (the common case on Linux/macOS) was misread as a flag named e.g.
        // "tmp/results".
        let takes = |kind: TokenKind, name: &str| kind == TokenKind::Long && name == "nologo";
        let args = owned(&["/tmp/results"]);
        let tokens = tokenize_dialect(&args, Dialect::Msbuild, &takes);

        assert_eq!(tokens[0].kind, TokenKind::Positional);
        assert_eq!(tokens[0].text, "/tmp/results");
    }

    #[test]
    fn msbuild_single_segment_slash_flag_is_still_a_flag() {
        // A genuine single-segment MSBuild switch (no internal '/') must still classify as Long,
        // including when it carries an attached value whose own text contains '/'.
        let args = owned(&["/nologo", "/p:OutDir=/tmp/out"]);
        let tokens = tokenize_dialect(&args, Dialect::Msbuild, &no_values);

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
            let tokens = tokenize_dialect(&args, Dialect::Msbuild, &no_values);

            assert_eq!(tokens[0].text, "logger", "for {arg}");
            assert_eq!(tokens[0].attached, Some("trx"), "for {arg}");
        }
    }

    #[test]
    fn msbuild_separate_token_value_still_works() {
        let args = owned(&["--results-directory", "/tmp/out"]);
        let takes =
            |kind: TokenKind, name: &str| kind == TokenKind::Long && name == "results-directory";
        let tokens = tokenize_dialect(&args, Dialect::Msbuild, &takes);

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
        let tokens = tokenize_dialect(&args, Dialect::Msbuild, &no_values);

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
        let tokens = tokenize_dialect(&args, Dialect::Msbuild, &takes);

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
        let tokens = tokenize_dialect(&args, Dialect::Msbuild, &no_values);

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
        let tokens = tokenize_dialect(&args, Dialect::Msbuild, &no_values);

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
        let tokens = tokenize_dialect(&args, Dialect::Msbuild, &no_values);

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
        let tokens = tokenize_dialect(&args, Dialect::Msbuild, &no_values);

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
        let tokens = tokenize_dialect(&args, Dialect::Msbuild, &no_values);

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
        let tokens = tokenize_dialect(&args, Dialect::Msbuild, &no_values);
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
        let tokens = tokenize_dialect(&args, Dialect::Msbuild, &takes);

        let values: Vec<&str> =
            double_dash_flag_values(&tokens, Dialect::Msbuild, "logger").collect();
        assert_eq!(values, vec!["console;verbosity=normal", "trx"]);
    }
}
