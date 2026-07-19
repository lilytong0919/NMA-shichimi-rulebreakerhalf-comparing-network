# I will move some shared functions from the mvdm codebase to here as I go along with my data analysis.
from typing import Any, Mapping, Dict, Callable, Optional

# A validator:
# - takes a value of any type
# - either raises if invalid
# - or returns a (possibly transformed) value
Validator = Callable[[Any], Any]


def parse_kwargs(
    defaults: Mapping[str, Any],
    kwargs: Mapping[str, Any],
    *,
    schema: Optional[Mapping[str, Validator]] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Validate keys, merge with dict.update(), then validate values.

    Parameters
    ----------
    defaults : Mapping[str, Any]
        Default keyword arguments.
    kwargs : Mapping[str, Any]
        User-provided keyword arguments (usually **kwargs).
    schema : Mapping[str, Validator], optional
        Mapping from key to validation function.
    strict : bool
        Whether to raise on unknown keyword arguments.

    Returns
    -------
    Dict[str, Any]
        Merged and validated keyword arguments.
    """

    # Copy defaults so we don't mutate the caller's dictionary
    opts: Dict[str, Any] = dict(defaults)

    # ---- key validation ----
    allowed = set(defaults)
    unknown = set(kwargs) - allowed
    if unknown:
        if strict:
            raise KeyError(f"Unknown keyword(s): {sorted(unknown)}")
        else:
            # drop unknown keys if not strict
            kwargs = {k: v for k, v in kwargs.items() if k in allowed}

    # ---- merge ----
    opts.update(kwargs)

    # ---- value validation ----
    if schema is not None:
        for k, validator in schema.items():
            if k in opts:
                opts[k] = validator(opts[k])

    return opts
