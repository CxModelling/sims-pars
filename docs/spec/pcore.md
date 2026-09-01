# The PCore language — specification

This is the precise behaviour of a PCore script as compiled by
`sims_pars.pcore`. It is deliberately a *specification of the current language*
(v1 surface): the new front end parses exactly what the legacy regex parser
accepted and lowers it to a byte-identical `BayesianNetwork` for every script in
`tests/pcore/corpus/`. What changed is that errors are now reported with a
source span instead of crashing or being silently dropped.

## Lexical structure

```
whitespace   spaces and tabs; insignificant except inside a string literal
newline      ends a statement
identifier   [A-Za-z_][A-Za-z0-9_]*        -- never starts with a digit
string       "..." or '...' , backslash escapes, contents kept verbatim
```

- **Line comments.** A line whose first non-whitespace characters are `#`, `//`
  or `--` is a comment and is ignored. (The legacy parser had no comment syntax;
  `#`-comment lines crashed it and `//` lines were silently skipped.)
- **Descriptions.** A `#` *after* other content on a statement line begins a
  description that runs to the end of the line. Spaces are allowed. The
  description is attached to the node and is not part of its value.
- **Right-hand sides** (after `~` or `=`) are captured as one raw span up to the
  end of the line, a top-level `}`, or a top-level `#`. Brackets `()[]{}` and
  string literals are tracked, so `cat({"low risk": 1})` is captured whole.

## Grammar

```ebnf
program    = { comment | NEWLINE } model { comment | NEWLINE } ;
model      = "PCore" identifier "{" { statement } "}" ;

statement  = ( sample | assign | exogenous ) [ description ] ( NEWLINE | "}" ) ;
sample     = identifier "~" rhs ;
assign     = identifier "=" rhs ;
exogenous  = identifier ;
rhs        = <raw expression source> ;
```

The legacy parser accepted only the **first** `PCore` block in a file; the new
parser reads every block, and `Program.models` holds them all. Everything after
the first is still ignored by `bayes_net_from_script`.

## Node kinds

`rhs` is classified without evaluating anything:

| Written | Kind | Value at sample time | Contributes to log-prob |
|---|---|---|---|
| `x = 5`, `x = [1, 2]` | **Value** | the literal, evaluated once at compile time | no |
| `x = a * exp(b)` | **Function** | the expression over resolved parents | no |
| `x ~ norm(mu, 1)` | **Distribution** | `.sample()` of the built distribution | yes — `logpdf(x)` |
| `x` (bare) | **Exogenous** | supplied by the caller | no |
| `x = f(y, z)` | **Pseudo** | — placeholder, raises if rendered | — |

- `= rhs` is a **Value** if `rhs` evaluates to a constant with no free names,
  otherwise a **Function**. A leading `f(` makes it a **Pseudo** node.
- Any identifier referenced but never defined becomes an **Exogenous** node
  automatically, at the point of first reference.
- The graph must be acyclic; a cycle is reported (the legacy parser raised
  `AttributeError`).

## Expression sublanguage (`=` right-hand side)

Phase 1 evaluates expressions with `sims_pars.util.safe_eval`: arithmetic,
comparisons, `a if c else b`, list / dict / tuple / set literals, and calls to
names in `MATH_FUNC` (`exp log sin cos tan hypot ceil floor sqrt abs erf pow
logit expit ifelse step`) plus the safe builtins `min max abs round sum pow len
int float bool`. Attribute access, subscripting, comprehensions, lambdas and
imports are rejected — with a span, not a crash.

Operator precedence, `/` vs `//`, `**`, chained comparisons and truthiness are
currently Python's. A written operator table and a dedicated evaluator are
Phase 3 work; until then this section is the contract and the corpus pins the
numeric results.

## Distribution sublanguage (`~` right-hand side)

`tag(args…)` where `tag` is one of the registered distributions
(`k unif norm lnorm gamma invgamma exp beta chisq triangle binom pois cat`).
Arguments are positional or `name=` keyword, and each value is itself an
expression. Omitted parameters are filled from the creator's schema.

- An unknown `tag` is reported with the nearest registered name.
- A missing required argument is reported (the legacy parser raised
  `KeyError: 'default'`).
- Parameter constraints (positivity, `0 ≤ p ≤ 1`, …) are enforced by pydantic;
  a violation is scored as `-inf` at evaluation time.

The stored definition string (`Def` in `to_json()`) matches the legacy parser:
the whitespace-stripped source when any keyword argument is present, otherwise
`tag(field1=v1, field2=v2, …)` with every schema field named.

## Diagnostics

`parse(text)` never raises. `Program.diagnostics` and
`sims_pars.pcore.check(text)` return a list of `Diagnostic(span, severity, code,
message, hint)`. `compile_script(text, strict=True)` (and
`bayes_net_from_script(text, strict=True)`) raise `DiagnosticError` if any
diagnostic is an error; `strict=False` builds what it can and leaves the list on
`bn.pcore_diagnostics`.

| Code | Meaning |
|---|---|
| `E0001` | unterminated string literal |
| `E0100`–`E0104` | malformed `PCore { … }` block structure |
| `E0110`–`E0114` | malformed statement |
| `E0111` | `: type` annotation (not supported until Phase 3) |
| `E0210`–`E0213` | bad `~` distribution definition |
| `E0221`–`E0223` | bad `=` expression / function |
| `E0230` | cyclic dependency |

## Tooling

```bash
sims-pars check model.pcore     # parse and print diagnostics; exit 1 on error
```
