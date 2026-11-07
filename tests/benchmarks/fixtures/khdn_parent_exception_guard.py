# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors


import functools
import importlib
import sys
import threading

KHDN_PREFIX = 'fla.ops.key_gated_hyperbolic_delta_rule'


def _replace_aliases(replacements, saved):
    for module_name, module in list(sys.modules.items()):
        if module is None or not module_name.startswith('fla.'):
            continue
        for name, value in list(vars(module).items()):
            replacement = replacements.get(id(value))
            if replacement is not None:
                saved.append((module, name, value))
                setattr(module, name, replacement)


def _restore_aliases(saved):
    for owner, name, value in reversed(saved):
        setattr(owner, name, value)


class KHDNExceptionGuard:
    """Reject fallback/replay entries and experimental-policy ineligibility during execution."""

    def __init__(self, precision_policy='strict'):
        if precision_policy not in ('strict', 'mixed_readout', 'bf16_fs6'):
            raise ValueError('unknown KHDN precision policy')
        self.precision_policy = precision_policy
        self.saved, self.events = [], []

    def __enter__(self):
        module = importlib.import_module(KHDN_PREFIX + '.chunk')
        recurrent = importlib.import_module(KHDN_PREFIX + '.fused_recurrent')
        naive = importlib.import_module(KHDN_PREFIX + '.naive')
        targets = (module._host_call, module._host_backward, recurrent.fused_recurrent_key_gated_hyperbolic_delta_rule,
                   naive.naive_recurrent_key_gated_hyperbolic_delta_rule)
        targets = (*targets, module._strict_device_replay)
        replacements = {}
        if self.precision_policy in ('mixed_readout', 'bf16_fs6'):
            helper = '_bf16_fs6_qualification_reason' if self.precision_policy == 'bf16_fs6' else '_qualification_reason'
            qualify = getattr(module, helper)

            @functools.wraps(qualify)
            def require_mixed(*args, **kwargs):
                reason = qualify(*args, **kwargs)
                if reason is not None:
                    self.events.append({'entry': helper, 'reason': reason,
                                        'thread': threading.get_ident()})
                    raise RuntimeError(f'KHDN native {self.precision_policy} measurement rejected strict fallback: {reason}')
                return reason
            replacements[id(qualify)] = require_mixed
        for function in targets:
            def reject(*args, _name=function.__name__, **kwargs):
                self.events.append({'entry': _name, 'thread': threading.get_ident()})
                raise RuntimeError(f'KHDN measurement rejected unexpected fallback/substitution: {_name}')
            replacements[id(function)] = reject
        _replace_aliases(replacements, self.saved)
        if not self.saved:
            raise AssertionError('KHDN exception sentinel could not be installed')
        return self

    def __exit__(self, *exception):
        _restore_aliases(self.saved)

    def evidence(self):
        return {'method': 'raise_before_host_forward_or_backward_or_recurrent_or_reference',
                'events': self.events, 'precision_requested': self.precision_policy,
                'normal_path_wrappers': int(self.precision_policy in ('mixed_readout', 'bf16_fs6')),
                'scope': 'exception/replay entries and requested-policy eligibility rejection installed outside timer; '
                         'original core range and eligibility checks remain timed'}
