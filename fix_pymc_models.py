#!/usr/bin/env python3
"""Fix common anti-patterns in transpiled PyMC models.

Pattern 1: Replace pt.dot(A, B) and pm.math.dot(A, B) with A @ B
Pattern 2: Remove correction Potentials (constants that don't affect sampling)

SKIP files that were intentionally reverted: accel_gp, surgical_model, garch11, prophet
"""

import re
import sys
from pathlib import Path

PYMC_DIR = Path("posterior_database/models/pymc")

# Files to skip (intentionally kept in original form)
SKIP_FILES = {"accel_gp.py", "surgical_model.py", "garch11.py", "prophet.py"}


def fix_dot_calls(content: str) -> str:
    """Replace pt.dot(A, B) and pm.math.dot(A, B) with A @ B."""
    for prefix in ['pt.dot', 'pm.math.dot']:
        while prefix + '(' in content:
            idx = content.find(prefix + '(')
            if idx == -1:
                break
            paren_start = idx + len(prefix) + 1
            args, paren_end = find_matching_args(content, paren_start)
            if args and len(args) == 2:
                a, b = args[0], args[1]
                # Add parens around second arg if it contains * or + to preserve precedence
                if any(op in b for op in [' * ', ' + ', ' - ']) and not (b.startswith('(') and b.endswith(')')):
                    b = f"({b})"
                replacement = f"{a} @ {b}"
                content = content[:idx] + replacement + content[paren_end + 1:]
            else:
                break
    return content


def find_matching_args(s, start):
    """Find the two arguments inside dot(arg1, arg2) starting after '('."""
    depth = 1
    i = start
    args = []
    arg_start = start
    while i < len(s) and depth > 0:
        if s[i] == '(':
            depth += 1
        elif s[i] == ')':
            depth -= 1
            if depth == 0:
                args.append(s[arg_start:i].strip())
                return args, i
        elif s[i] == ',' and depth == 1:
            args.append(s[arg_start:i].strip())
            arg_start = i + 1
        i += 1
    return None, i


def remove_correction_potentials(content: str) -> str:
    """Remove pm.Potential lines that are pure constants (correction terms)."""
    lines = content.split('\n')
    new_lines = []
    i = 0
    removed = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        is_correction = False
        if 'pm.Potential(' in stripped and any(kw in stripped.lower() for kw in [
            'correction', 'half_dist_correction'
        ]):
            is_correction = True

        if is_correction:
            # Check if multi-line
            if stripped.count('(') > stripped.count(')'):
                # Multi-line - skip until closing
                j = i + 1
                while j < len(lines) and ')' not in lines[j]:
                    j += 1
                i = j + 1
            else:
                i += 1
            removed += 1
            # Remove preceding comment about correction
            while new_lines and new_lines[-1].strip().startswith('#') and any(
                kw in new_lines[-1].lower() for kw in ['correction', 'half', 'stan ']
            ):
                new_lines.pop()
            continue

        new_lines.append(line)
        i += 1

    if removed > 0:
        result = '\n'.join(new_lines)
        while '\n\n\n' in result:
            result = result.replace('\n\n\n', '\n\n')
        return result
    return content


def remove_half_counting(content: str) -> str:
    """Remove n_half_params counting lines."""
    lines = content.split('\n')
    new_lines = [l for l in lines if not (
        l.strip().startswith('n_half_params') and ('= 0' in l or '+= 1' in l)
    )]
    result = '\n'.join(new_lines)
    while '\n\n\n' in result:
        result = result.replace('\n\n\n', '\n\n')
    return result


def fix_file(filepath: Path) -> list[str]:
    """Apply all fixes to a file."""
    content = filepath.read_text()
    original = content
    changes = []

    new = fix_dot_calls(content)
    if new != content:
        changes.append("dot -> @")
        content = new

    new = remove_correction_potentials(content)
    if new != content:
        changes.append("removed corrections")
        content = new

    new = remove_half_counting(content)
    if new != content:
        changes.append("removed n_half_params")
        content = new

    if content != original:
        filepath.write_text(content)
    return changes


def main():
    total = 0
    for f in sorted(PYMC_DIR.glob("*.py")):
        if f.name in SKIP_FILES:
            continue
        changes = fix_file(f)
        if changes:
            print(f"  {f.name}: {', '.join(changes)}")
            total += 1
    print(f"\nFixed {total} files")


if __name__ == "__main__":
    main()
