# ruff: noqa: E722, SIM105
# strict_typing_linter.py: skip-file
# suppression_rationale_linter.py: skip-file
"""Test cases for --report-unused-directives in exception_safety_linter.py.

Each case has a comment indicating the expected behavior:
- STALE: directive should be flagged as unused
- ACTIVE: directive suppresses a real violation, should NOT be flagged
"""

# -- STALE: bare-except on a specific catch (not bare) ------------------------
from __future__ import annotations

try:
    pass
except ValueError:  # exception_safety_linter.py: bare-except — not actually bare
    pass

# -- ACTIVE: bare-except on actual bare except --------------------------------

try:
    pass
except:  # exception_safety_linter.py: bare-except — intentional catch-all
    pass

# -- STALE: swallowed-exception on handler that re-raises ---------------------

try:
    pass
except Exception as e:  # exception_safety_linter.py: swallowed-exception — not swallowed
    raise RuntimeError('wrapped') from e

# -- ACTIVE: swallowed-exception on handler without re-raise ------------------

try:
    pass
except Exception:  # exception_safety_linter.py: swallowed-exception — error boundary
    pass

# -- STALE: multi-code directive, one code stale ------------------------------

try:
    pass
except ValueError:  # exception_safety_linter.py: bare-except, raise-without-from — both stale here
    raise TypeError('wrong') from None
