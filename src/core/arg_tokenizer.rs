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

/// True if `text` (a `Long` token's name) matches `name` under `dialect`'s naming rules:
/// exact for [`Dialect::Posix`] (git/cargo/rg/golangci-lint are case-sensitive), ASCII
/// case-insensitive for [`Dialect::Msbuild`] (Windows/MSBuild-ecosystem tools fold case
/// broadly — this isn't a dotnet-CLI particularity, e.g. classic MSBuild.exe's `/nologo` and
/// `/NoLogo` are equally valid). `text` can't be case-folded once at tokenize time without
/// giving up `Token`'s zero-copy `&'a str` (there's no borrowed "lowercased" view), so instead
/// every dialect-aware lookup goes through this and [`flag_value`]/[`has_flag`].
fn flag_name_matches(text: &str, name: &str, dialect: Dialect) -> bool {
    match dialect {
        Dialect::Msbuild => text.eq_ignore_ascii_case(name),
        Dialect::Posix => text == name,
    }
}

/// This flag's value, if `name` (matched per `dialect`, see [`flag_name_matches`]) appears as
/// a `Long` token anywhere in `tokens`. `tokens` must have come from `tokenize_dialect(_, dialect,
/// _)` — mixing dialects between tokenizing and looking up gives nonsensical results.
pub fn flag_value<'a>(tokens: &[Token<'a>], dialect: Dialect, name: &str) -> Option<&'a str> {
    tokens
        .iter()
        .find(|t| t.kind == TokenKind::Long && flag_name_matches(t.text, name, dialect))
        .and_then(|t| t.value(tokens))
}

/// True if `name` (matched per `dialect`) appears as a `Long` token anywhere in `tokens`.
pub fn has_flag(tokens: &[Token<'_>], dialect: Dialect, name: &str) -> bool {
    tokens
        .iter()
        .any(|t| t.kind == TokenKind::Long && flag_name_matches(t.text, name, dialect))
}

/// Every value for `name` (matched per `dialect`), in order, for a flag that can legitimately
/// be repeated (e.g. dotnet test's `--logger console;verbosity=normal --logger trx`, where each
/// occurrence must be checked, not just the first — [`flag_value`] only reports the first
/// match). Occurrences with no value (bare flag, or a value-taking flag with nothing left to
/// consume) are skipped rather than yielding `None`.
pub fn flag_values<'a, 't>(
    tokens: &'t [Token<'a>],
    dialect: Dialect,
    name: &'t str,
) -> impl Iterator<Item = &'a str> + 't {
    tokens
        .iter()
        .filter(move |t| t.kind == TokenKind::Long && flag_name_matches(t.text, name, dialect))
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
            push_atomic_flag(&mut tokens, args, &mut i, rest, dialect, takes_value);
            continue;
        }

        if dialect == Dialect::Msbuild {
            if let Some(rest) = arg.strip_prefix('/') {
                if !rest.is_empty() {
                    push_atomic_flag(&mut tokens, args, &mut i, rest, dialect, takes_value);
                    continue;
                }
            }
            if arg.len() > 1 && arg.starts_with('-') {
                push_atomic_flag(&mut tokens, args, &mut i, &arg[1..], dialect, takes_value);
                continue;
            }
        } else if arg.len() > 1 && arg.starts_with('-') {
            let cluster = &arg[1..];

            if cluster.bytes().all(|b| b.is_ascii_digit()) {
                tokens.push(Token {
                    kind: TokenKind::Short,
                    text: cluster,
                    attached: None,
                    linked: None,
                    source_index: i,
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
                });

                if takes_value(TokenKind::Short, char_text) {
                    let remainder = &cluster[offset + char_len..];
                    if !remainder.is_empty() {
                        tokens[flag_index].attached = Some(remainder);
                    } else if let Some(next) = args.get(i + 1) {
                        let value_index = tokens.len();
                        tokens.push(Token {
                            linked: Some(flag_index),
                            ..positional(next.as_ref(), i + 1)
                        });
                        tokens[flag_index].linked = Some(value_index);
                        consumed_next = true;
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
/// value. `rest` is the flag text with its prefix (`--`, `-`, or `/`) already stripped.
fn push_atomic_flag<'a, T: AsRef<str>>(
    tokens: &mut Vec<Token<'a>>,
    args: &'a [T],
    i: &mut usize,
    rest: &'a str,
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
    });
    *i += 1;

    if attached.is_none() && takes_value(TokenKind::Long, name) {
        if let Some(next) = args.get(*i) {
            let value_index = tokens.len();
            tokens.push(Token {
                linked: Some(flag_index),
                ..positional(next.as_ref(), *i)
            });
            tokens[flag_index].linked = Some(value_index);
            *i += 1;
        }
    }
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
    fn msbuild_absolute_path_positional_is_not_mistaken_for_a_flag_with_predicate_false() {
        // "/tmp/results" has no recognized flag name, so it's still a positional even though
        // it starts with '/' — the predicate, not the tokenizer, is what would gate this.
        let takes = |kind: TokenKind, name: &str| kind == TokenKind::Long && name == "nologo";
        let args = owned(&["/tmp/results"]);
        let tokens = tokenize_dialect(&args, Dialect::Msbuild, &takes);

        // It's still classified as a Long flag named "tmp/results" (no recognized value-taking
        // predicate matches it) — proving the *caller* must know not to treat arbitrary
        // '/'-prefixed tokens as flags on Unix; the tokenizer only reports structure.
        assert_eq!(tokens[0].kind, TokenKind::Long);
        assert_eq!(tokens[0].text, "tmp/results");
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

    // --- has_flag / flag_value ---

    #[test]
    fn msbuild_has_flag_and_flag_value_are_case_insensitive() {
        let args = owned(&["-NoLogo", "--Logger:trx"]);
        let tokens = tokenize_dialect(&args, Dialect::Msbuild, &no_values);

        assert!(has_flag(&tokens, Dialect::Msbuild, "nologo"));
        assert!(has_flag(&tokens, Dialect::Msbuild, "NOLOGO"));
        assert_eq!(flag_value(&tokens, Dialect::Msbuild, "logger"), Some("trx"));
        assert_eq!(flag_value(&tokens, Dialect::Msbuild, "LOGGER"), Some("trx"));
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
    fn flag_values_reports_every_occurrence_not_just_the_first() {
        // Regression: dotnet test's --logger can legitimately repeat
        // (`--logger "console;verbosity=normal" --logger trx`) -- unlike flag_value, which
        // only reports the first match, every occurrence must be checkable.
        let args = owned(&["--logger:console;verbosity=normal", "--logger", "trx"]);
        let takes = |kind: TokenKind, name: &str| kind == TokenKind::Long && name == "logger";
        let tokens = tokenize_dialect(&args, Dialect::Msbuild, &takes);

        let values: Vec<&str> = flag_values(&tokens, Dialect::Msbuild, "logger").collect();
        assert_eq!(values, vec!["console;verbosity=normal", "trx"]);
    }
}
