# The PCore language — specification

This is the precise behaviour of a PCore script as compiled by
`sims_pars.pcore`. The v1 surface — what the legacy regex parser accepted — is
still lowered to a byte-identical `BayesianNetwork` for every script in
`tests/pcore/corpus/`; every failure that used to be a raw exception is now a
diagnostic with a source span instead. v2 adds three purely additive
extensions on top of that surface (§ Type annotations, § Plates, §
Composition below) — a v1 script is also a valid v2 script.

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
model      = "PCore" identifier "{" { block } "}" ;
block      = statement | plate | include ;

statement  = identifier [ ":" type ] ( sample | assign | exogenous )
             [ description ] ( NEWLINE | "}" ) ;
sample     = "~" rhs ;
assign     = "=" rhs ;
exogenous  = ;                                    -- bare name, no '~' / '='
rhs        = <raw expression source> ;
type       = "float" | "int" | "bool" | "vector" | "simplex" ;

plate      = "for" identifier "in" INT ".." INT "{" { statement } "}" ;
include    = "include" STRING ;
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

Expressions accept arithmetic, comparisons, `a if c else b`, list / dict /
tuple / set literals, and calls to names in `MATH_FUNC` (`exp log sin cos tan
hypot ceil floor sqrt abs erf pow logit expit ifelse step`) plus the safe
builtins `min max abs round sum pow len int float bool`. Attribute access,
subscripting, comprehensions, lambdas and imports are rejected — with a span,
not a crash.

Two evaluators implement this contract and agree on every case that matters:

- **Runtime evaluation** (sampling, likelihood) still goes through
  `sims_pars.util.safe_eval` — an AST allow-list plus a stripped-builtins
  `eval()` — so every legacy and corpus script keeps its exact historical
  numeric result.
- **Static evaluation** (type-annotation checking, see below; anything the
  lowering pipeline needs to know at compile time) goes through
  `sims_pars.pcore.evaluator`: the same accepted grammar, but interpreted
  against an explicit operator table with no call to `eval()` at all. Its
  module docstring is the canonical, versioned operator-precedence table —
  currently identical to Python's for this subset (`**` binds tighter than
  unary `-`/`+`, `*` `/` `//` `%` bind tighter than `+` `-`, comparisons chain,
  `and` binds tighter than `or`, `a if c else b` is lowest).

`sims_pars.pcore.evaluate(expr, env=None)` is the public entry point to the
static evaluator; it raises `EvalError` (not Python's `NameError` /
`SyntaxError`) for anything it can't compute, including a name that isn't in
`env` — callers use that to tell "this is a closed constant" from "this
depends on parents".

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

## Type annotations (v2)

`name : type` may precede `~` / `=` on any statement: `x: float = 3`,
`p: simplex ~ dirichlet([1, 1, 1])`. `type` is one of `float int bool vector
simplex`.

The annotation is checked only where it can be, statically, at parse time: a
constant `=` right-hand side (one with no free names) is evaluated with the
static evaluator above and its Python value is checked against the declared
type (`int` also accepts a `float` with an integer value; `simplex` requires a
non-negative vector summing to 1 within `1e-6`). A mismatch is a *warning*
(`E0301`) — the node still builds. A `~` node, or a `=` expression that
depends on parents, has no value until sample time, so its annotation is
recorded but not checked. An unknown type name is an error (`E0300`) with a
"did you mean" hint.

## Plates (v2)

```
for i in 1..3 {
    x ~ norm(mu, 1)
    y = x[i] * 2
}
```

expands to three independent nodes per statement in the body — `lo..hi`
inclusive, both bounds integer literals. Expansion happens in the parser,
before lowering ever sees the statements, so a plate is pure sugar over
repeated node declarations:

- **A declared name is suffixed**: `x` becomes `x_1, x_2, x_3`.
- **`name[var]` in a right-hand side** picks the matching sibling for the
  current iteration: `x[i]` becomes `x_1`, `x_2`, `x_3` in turn. This is the
  only way to reference a plated sibling — the bracket makes the intent
  explicit.
- **A bare, whole-word `var`** in a right-hand side becomes the literal
  iteration integer: `y = i * 2` becomes `y_1 = 1 * 2`, `y_2 = 2 * 2`, ...
- A name with **no** `[var]` and **not** the loop variable itself is left
  exactly as written — it refers to one shared node outside the plate, the
  usual plate-notation meaning of an unindexed reference.
- Substitution never touches the contents of a string literal.
- Plates nest (`for j in 1..2 { for i in 1..2 { ... } }` produces `x_1_1,
  x_1_2, x_2_1, x_2_2`); `include` is not allowed inside a plate (`E0127`).
- `hi < lo` is an empty range: a warning (`E0128`), zero nodes, not an error.

## Composition (v2)

```
PCore Main {
    include "priors.pcore"
    x ~ binom(10, p)
}
```

`include "path"` splices every non-exogenous node of another file's *first*
`PCore` block into the current model, before its own statements are built. The
path resolves relative to the including file's directory (`compile_script`
and `check` both take an optional `path=` for this; without one, relative
paths resolve against the current working directory). A node the including
file also defines overrides the included one, same precedence as
`BayesianNetwork.merge`. Includes may nest; a cycle is a located error
(`E0251`/`E0254`), as is a missing file (`E0250`) or a file with its own
errors (`E0254`).

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
| `E0111` | malformed `: type` annotation |
| `E0120`–`E0128` | malformed plate (`for … in lo..hi { … }`) |
| `E0130`–`E0132` | malformed `include` statement |
| `E0210`–`E0213` | bad `~` distribution definition |
| `E0221`–`E0223` | bad `=` expression / function |
| `E0230` | cyclic dependency |
| `E0250`–`E0254` | `include` resolution (missing file, circular, sub-file errors) |
| `E0300` | unknown `: type` name |
| `E0301` | a constant's value doesn't match its `: type` annotation (warning) |

## Tooling

```bash
sims-pars check model.pcore     # parse and print diagnostics; exit 1 on error
```
