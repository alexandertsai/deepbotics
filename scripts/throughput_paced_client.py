"""Header-informed pacing for future independently budgeted account lanes."""

import math

from task09_paced_client import PacedClient, RATE_HEADERS


class ThroughputPacedClient(PacedClient):
    def __init__(self,*args,prompt_tpm=500000,uncached_prompt_tpm=250000,
                 generated_tpm=60000,max_concurrent=3,**kwargs):
        self.fallback_targets = {"prompt":prompt_tpm,"uncached":uncached_prompt_tpm,"generated":generated_tpm}
        self.raw_reported_limits = {}
        self.header_headroom_fraction = .9
        super().__init__(*args,prompt_tpm=prompt_tpm,uncached_prompt_tpm=uncached_prompt_tpm,
            generated_tpm=generated_tpm,max_concurrent=max_concurrent,**kwargs)

    def _apply_headers(self,status,headers,retry_number):
        with self.condition:
            delay = super()._apply_headers(status,headers,retry_number)
            for key,header in RATE_HEADERS.items():
                try:
                    value = float(headers.get(header,""))
                except (ValueError,TypeError):
                    continue
                if math.isfinite(value) and value > 0:
                    self.raw_reported_limits[key] = value
                    target = max(1,math.floor(value*self.header_headroom_fraction))
                    self.pacing_config[key] = target
                    self.header_limits[key] = target
            self.condition.notify_all()
            return delay

    def _summary_unlocked(self):
        result = super()._summary_unlocked()
        result["pacing"].update(fallback_tpm=dict(self.fallback_targets),
            raw_reported_limit_tpm=dict(self.raw_reported_limits),
            reported_limit_headroom_fraction=self.header_headroom_fraction,
            target_tpm_before_429_scaling=dict(self.pacing_config),
            policy="Reported limits set 90% pacing targets; absent dimensions retain fallback targets. 429 cooldown and throughput reduction still apply.")
        return result

    def _before_accounting(self,record,**kwargs):
        super()._before_accounting(record,**kwargs)
        with self.condition:
            record["raw_reported_limit_tpm"] = dict(self.raw_reported_limits)
            record["pacing_target_tpm_before_429_scaling"] = dict(self.pacing_config)
