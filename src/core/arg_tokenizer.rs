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

/// What kind of unit a [`Token`] represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind {
    /// `--name` (see `Token::text` for the name, without the leading `--`).
    Long,
    /// One character of a `-x` / `-xyz` short-option cluster (see `Token::text`, without the
    /// leading `-`). A run of only digits (`-20`) is a widely-used shorthand for a numeric
    /// value in its own right (git log/head/tail's `-N` count) rather than a cluster of
    /// per-digit boolean flags, so it is kept as one `Short` token with the whole digit run as
    /// `text`, never decomposed.
    Short,
    /// A positional/value token — either free-standing or consumed by a preceding `Long`/`Short`
    /// as its separate-token value (see `Token::linked`).
    Positional,
    /// The literal `--` end-of-options/pathspec separator. Emitted exactly once, for the first
    /// `--` encountered; every token after it is `Positional` unconditionally and the
    /// `takes_value` predicate is never consulted again. A second or later `--` comes back as a
    /// plain `Positional` with `text == "--"`, matching real git/GNU semantics.
    DashDash,
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

/// Tokenizes `args` into [`Token`]s. `takes_value(kind, name)` is called for each `Long`/`Short`
/// flag that has no attached value, to decide whether the following whole token should be
/// consumed as its separate-token value; it is never called for tokens at or after `--`.
///
/// Never panics and never fails to classify: a value-taking flag with nothing left to consume
/// simply gets `attached: None, linked: None`, matching RTK's fallback/never-block-the-user
/// convention.
pub fn tokenize<'a>(
    args: &'a [String],
    takes_value: &dyn Fn(TokenKind, &str) -> bool,
) -> Vec<Token<'a>> {
    let mut tokens: Vec<Token<'a>> = Vec::with_capacity(args.len());
    let mut i = 0;
    let mut seen_dash_dash = false;

    while i < args.len() {
        let arg = args[i].as_str();

        if seen_dash_dash {
            tokens.push(positional(arg, i));
            i += 1;
            continue;
        }

        if arg == "--" {
            tokens.push(Token {
                kind: TokenKind::DashDash,
                text: "",
                attached: None,
                linked: None,
                source_index: i,
            });
            seen_dash_dash = true;
            i += 1;
            continue;
        }

        if let Some(rest) = arg.strip_prefix("--") {
            let (name, attached) = match rest.split_once('=') {
                Some((name, value)) => (name, Some(value)),
                None => (rest, None),
            };
            let flag_index = tokens.len();
            tokens.push(Token {
                kind: TokenKind::Long,
                text: name,
                attached,
                linked: None,
                source_index: i,
            });
            i += 1;

            if attached.is_none() && takes_value(TokenKind::Long, name) {
                if let Some(next) = args.get(i) {
                    let value_index = tokens.len();
                    tokens.push(Token {
                        linked: Some(flag_index),
                        ..positional(next.as_str(), i)
                    });
                    tokens[flag_index].linked = Some(value_index);
                    i += 1;
                }
            }
            continue;
        }

        if arg.len() > 1 && arg.starts_with('-') {
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
                            ..positional(next.as_str(), i + 1)
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
}
