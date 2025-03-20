from typing import Optional, Callable, Iterable

import experiment_utils as eu

def log_params(logger: Optional[eu.Logger] = None, **params):
    if logger is None:
        print("Params:")
        print(params)
    else:
        logger.log_params(params)


def log_values(
    step: int,
    logger: Optional[eu.Logger] = None,
    compare_fn: Callable = eu.compare_fns.new,
    **values,
):
    if logger is None:
        print(f"Step: {step}")
        print(values)
    else:
        logger.log_values(values, step, compare_fn)


def log_value(
    key,
    value,
    step: int,
    logger: Optional[eu.Logger] = None,
    compare_fn: Callable = eu.compare_fns.new,
):
    if logger is None:
        print(f"Step: {step}")
        print(f"{key}: {value}")

    else:
        logger.log_value(
            key=key,
            value=value,
            step=step,
            compare_fn=compare_fn,
        )