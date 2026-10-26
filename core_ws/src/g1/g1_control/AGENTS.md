# Coding Style for g1_control

## Python

- Prefer single quotes whenever possible
- Inline comments: lower case, no trailing period
- Function docstrings: start with uppercase, no trailing period
- Write clear, modular code
- Prefer meaningful but concise names
- Keep lines readable; break long lines
- Parse CLI args only in `if __name__ == '__main__':`
- Pass parsed args explicitly to `main()`
- Imports grouped as: library imports → blank line → project imports, sorted by length
- Single blank line between functions/methods
- Double blank lines between top-level definitions

## Change Rules

- Keep diffs minimal
- Follow nearby file conventions first
- Do not make unrelated refactors

## Documentation style

- Use 4 space indentation
- Follow markdownlint style guidelines
- Capitalize the starting letter of bullet items
- Use lower case for explanations after a colon: like this
- Break lines that are too long for readability
- Use period at the end of each bullet item
