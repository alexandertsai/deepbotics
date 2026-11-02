"""Header-guided pacing ramps for the prepared Task09 follow-up only."""

from __future__ import annotations

import math
import time

import task08_pilot as P
from task09_paced_client import PacedClient, RATE_HEADERS


class AdaptivePacedClient(PacedClient):
    def __init__(self, *args, ramp_interval_seconds=120.0, ramp_fraction=.10,
                 reported_limit_fraction=.80, backoff_fraction=.25, min_clean_successes=3,
                 max_prompt_tpm=240000, max_uncached_prompt_tpm=240000,
                 max_generated_tpm=60000, **kwargs):
        settings = (ramp_interval_seconds, ramp_fraction, reported_limit_fraction, backoff_fraction)
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in settings):
            raise ValueError('Adaptive pacing settings must be finite numbers')
        if ramp_interval_seconds <= 0 or not 0 < ramp_fraction <= .5:
            raise ValueError('Ramp interval must be positive and fraction in (0, .5]')
        if not 0 < reported_limit_fraction < 1 or not 0 < backoff_fraction < 1:
            raise ValueError('Limit margin and backoff fraction must be in (0, 1)')
        if type(min_clean_successes) is not int or min_clean_successes < 1:
            raise ValueError('Clean success requirement must be positive')
        maxima = {'prompt': max_prompt_tpm, 'uncached': max_uncached_prompt_tpm, 'generated': max_generated_tpm}
        initials = {'prompt': kwargs.get('prompt_tpm', 120000), 'uncached': kwargs.get('uncached_prompt_tpm', 120000),
                    'generated': kwargs.get('generated_tpm', 30000)}
        if any(type(value) is not int or value < initials[key] for key, value in maxima.items()):
            raise ValueError('Adaptive maximum rates must be integers at least as large as initial rates')
        self.initial_pacing = dict(initials)
        self.maximum_pacing = maxima
        self.ramp_interval_seconds = float(ramp_interval_seconds)
        self.ramp_fraction, self.reported_limit_fraction = ramp_fraction, reported_limit_fraction
        self.backoff_fraction, self.min_clean_successes = backoff_fraction, min_clean_successes
        self.clean_successes, self.last_success_at = 0, -float('inf')
        self.next_ramp_at = time.monotonic() + self.ramp_interval_seconds
        self.observed_provider_limits, self.rate_adjustments = {}, []
        self._initializing_adaptive = True
        super().__init__(*args, **kwargs)
        self._initializing_adaptive = False

    def _cap_unlocked(self, key):
        reported = self.observed_provider_limits.get(key)
        return min(self.maximum_pacing[key], reported * self.reported_limit_fraction if reported else float('inf'))

    def _record_adjustment_unlocked(self, reason, before):
        event = {'at': time.time(), 'reason': reason, 'before_tpm': before,
                 'after_tpm': dict(self.pacing_config), 'observed_provider_limits': dict(self.observed_provider_limits)}
        self.rate_adjustments.append(event)
        P.write_json(self.root / 'pacing-adjustments.json', self.rate_adjustments)

    def _maybe_ramp_unlocked(self):
        if self._initializing_adaptive:
            return
        now = time.monotonic()
        if (now < self.next_ramp_at or now < self.cooldown_until
                or self.clean_successes < self.min_clean_successes
                or now - self.last_success_at > self.ramp_interval_seconds):
            return
        before = dict(self.pacing_config)
        for key, value in before.items():
            self.pacing_config[key] = max(1.0, min(value * (1 + self.ramp_fraction), self._cap_unlocked(key)))
        self.next_ramp_at = now + self.ramp_interval_seconds
        self.clean_successes = 0
        if self.pacing_config != before:
            self._record_adjustment_unlocked('clean_traffic_ramp', before)
            self.condition.notify_all()

    def _rates_unlocked(self):
        self._maybe_ramp_unlocked()
        return super()._rates_unlocked()

    def _apply_headers(self, status, headers, retry_number):
        with self.condition:
            before = dict(self.pacing_config)
            for key, header in RATE_HEADERS.items():
                try:
                    value = float(headers.get(header, ''))
                except (ValueError, TypeError):
                    continue
                if math.isfinite(value) and value > 0:
                    self.observed_provider_limits[key] = value
            delay = super()._apply_headers(status, headers, retry_number)
            for key, value in self.observed_provider_limits.items():
                self.header_limits[key] = value * self.reported_limit_fraction
                self.pacing_config[key] = max(1.0, min(self.pacing_config[key], self._cap_unlocked(key)))
            now = time.monotonic()
            if status == 429:
                for key, value in self.pacing_config.items():
                    floor = min(self.initial_pacing[key], self._cap_unlocked(key))
                    self.pacing_config[key] = max(1.0, min(self._cap_unlocked(key), max(floor, value * (1 - self.backoff_fraction))))
                self.clean_successes = 0
                self.next_ramp_at = max(now + self.ramp_interval_seconds, self.cooldown_until)
                self._record_adjustment_unlocked('429_backoff_toward_initial_rate', before)
            elif 200 <= status < 300:
                self.clean_successes += 1
                self.last_success_at = now
                if self.pacing_config != before:
                    self._record_adjustment_unlocked('provider_limit_reduction', before)
                self._maybe_ramp_unlocked()
            elif self.pacing_config != before:
                self._record_adjustment_unlocked('provider_limit_reduction', before)
            return delay

    def _summary_unlocked(self):
        summary = super()._summary_unlocked()
        summary['adaptive_pacing'] = {
            'enabled': True, 'initial_tpm': dict(self.initial_pacing), 'maximum_tpm': dict(self.maximum_pacing),
            'current_targets_tpm': dict(self.pacing_config), 'observed_provider_limits': dict(self.observed_provider_limits),
            'reported_limit_fraction': self.reported_limit_fraction, 'ramp_fraction': self.ramp_fraction,
            'ramp_interval_seconds': self.ramp_interval_seconds, 'min_clean_successes': self.min_clean_successes,
            'clean_successes_since_adjustment': self.clean_successes,
            'backoff_fraction': self.backoff_fraction,
            'backoff_floor': 'Initial rate, bounded by reported safety cap; inherited cooldown and temporary throughput reduction still apply.',
            'adjustments_total': len(self.rate_adjustments), 'recent_adjustments': self.rate_adjustments[-5:],
            'billing_guard_unchanged': True,
        }
        return summary
