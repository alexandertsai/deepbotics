"""Public Task09 exploration with an Astra lead choosing a measured design."""

import json
from pathlib import Path
import time

import task08_pilot as P
import task09_swarm_followup as F

DS_ROLES = (
    ("passive_design", "Explore legal, buildable mechanical gravity compensation while preserving useful workspace."),
    ("mass_layout", "Explore legal mass and layout improvements while preserving the public geometry and construction contracts."),
    ("model_trim", "Improve model-based residual trim across all supplied robot variants."),
    ("transfer_adaptation", "Improve transfer and adaptation using only the permitted runtime model and measurements."),
    ("probe_design", "Improve identification and calibration within the public measurement allowance."),
    ("integrated_validation", "Explore an integrated hardware and software design and check public constraints across all variants."),
)
ROLES = DS_ROLES
ASTRA_ROLE = ("astra_lead", "Lead the engineering effort, integrate useful measured designs, and choose the final buildable candidate using the complete public task objective.")


class Coordinator(F.Coordinator):
    def __init__(self,*args,tests_per_worker=8,**kwargs):
        if not 1 <= tests_per_worker <= 8:
            raise ValueError("Team probe allowance must be between one and eight per worker")
        self.lead_choice = None
        kwargs.setdefault("seconds",1800)
        kwargs.setdefault("tests",56)
        kwargs.setdefault("worker_count",7)
        super().__init__(*args,tests_per_worker=tests_per_worker,**kwargs)

    def measured_candidate(self,index):
        if type(index) is not int:
            raise ValueError("Use a measured probe index or -1 for the starter")
        with self.lock:
            if index == -1:
                chosen = dict(self.cache[self.manifest["sha256"]])
            elif 0 <= index < len(self.entries) and self.entries[index]["status"] == "measured":
                chosen = dict(self.cache[self.entries[index]["sha256"]])
            else:
                raise ValueError("Candidate must have a completed public development measurement")
        from services.controller.storage import verify
        verify(chosen["bundle"],chosen["manifest"])
        return chosen

    def read_candidate(self,index):
        chosen = self.measured_candidate(index)
        bundle = Path(chosen["bundle"])
        files = {}
        for name in F.ENTRY_FILES:
            path = bundle/name
            if path.is_symlink() or not path.resolve().is_relative_to(bundle.resolve()):
                raise ValueError("Measured entry files must remain within their frozen bundle")
            files[name] = path.read_text()
        return {"index":index,"worker":chosen["worker"],"partial_score":chosen["score"],
            "manifest_sha256":chosen["manifest"]["sha256"],"entry_files":files,
            "public_metrics":F.compact_report(chosen["report"])}

    def choose_candidate(self,session,index,reason):
        if session.condition != "astra" or not session.lead:
            raise ValueError("Only the Astra lead may choose the final candidate")
        if not isinstance(reason,str) or not reason.strip() or len(reason)>4000:
            raise ValueError("Provide a concise selection reason")
        if session.submitted is not None or time.monotonic() >= self.deadline:
            raise ValueError("Lead development has ended")
        from services.controller.storage import snapshot
        chosen = self.measured_candidate(index)
        with session.lock,self.lock:
            if self.lead_choice is not None:
                raise ValueError("The lead has already frozen its final choice")
            manifest = snapshot(chosen["bundle"],session.root/"frozen",max_bytes=256*1024**2,
                                workspace_path="code/task09")
            if manifest["files"] != chosen["manifest"]["files"]:
                raise ValueError("Chosen snapshot differs from measured source")
            session.submitted = manifest
            self.lead_choice = {**chosen,"bundle":session.root/"frozen","manifest":manifest,
                "source_index":index,"source_manifest_sha256":chosen["manifest"]["sha256"],
                "lead_worker":session.worker_id,"reason":reason}
            record = {key:self.lead_choice[key] for key in ("source_index","source_manifest_sha256","lead_worker","reason")}
            record.update(source_worker=chosen["worker"],frozen_manifest=manifest,at=time.time())
            P.write_json(self.root/"lead-choice.json",record)
            P.write_json(session.root/"frozen-manifest.json",manifest)
        return {"chosen":True,"index":index,"source_worker":chosen["worker"],
                "frozen_sha256":manifest["sha256"],"lead_submitted":True}

    def select(self):
        from services.controller.storage import snapshot,verify
        with self.lock:
            chosen = None if self.lead_choice is None else dict(self.lead_choice)
        if chosen is None:
            selection = super().select()
            selection.update(selection_policy="public_best_fallback",
                fallback_reason="The Astra lead did not freeze an eligible measured candidate choice")
            P.write_json(self.root/"selection.json",selection)
            return selection
        verify(chosen["bundle"],chosen["manifest"])
        manifest = snapshot(chosen["bundle"],self.root/"selected",max_bytes=256*1024**2,workspace_path="code/task09")
        selection = {"worker":chosen["worker"],"partial_score":chosen["score"],
            "selection_components":F.metric_components(chosen["report"]),"starter_partial_score":self.baseline_score,
            "selection_used_final_scores":False,"selection_policy":"astra_lead_measured_choice",
            "lead_worker":chosen["lead_worker"],"source_index":chosen["source_index"],
            "source_manifest_sha256":chosen["source_manifest_sha256"],"reason":chosen["reason"],"manifest":manifest}
        P.write_json(self.root/"selected-manifest.json",manifest)
        P.write_json(self.root/"selection.json",selection)
        return selection


class Session(F.SwarmSession):
    def __init__(self,root,coordinator,worker_id,role,brief,*,condition="deepseek-rax",lead=False):
        if condition not in ("deepseek-rax","deepseek","astra") or lead and condition != "astra":
            raise ValueError("The team lead must be Astra; explorers may be DeepSeek or Astra")
        self.lead = lead
        super().__init__(root,coordinator,worker_id,role,brief)
        self.condition = "deepseek-rax" if condition == "deepseek" else condition
        self.prompt = self.make_prompt()
        (self.root/"prompt.txt").write_text(self.prompt)

    def make_prompt(self):
        condition = self.condition
        try:
            self.condition = "deepseek-rax"
            text = F.Q.Session.make_prompt(self).replace("after both agents stop","after all team agents stop")
        finally:
            self.condition = condition
        files = "\n".join(f"Starter {name}:\n{(self.coordinator.root/'starter'/name).read_text()}" for name in F.ENTRY_FILES)
        text += f"""
Task09 engineering team. Worker {self.worker_id}; role {self.role}: {self.brief}
All agents share the same fresh public starter, task specification, public tools and RAX methods.
Prioritize the complete public task objective, legal buildable hardware, useful workspace, and transfer/adaptation.
Public probe metrics are partial evidence. Keep claims separate from measurements and preserve a measured incumbent.
You have {self.coordinator.seconds} shared development seconds and at most {self.coordinator.tests_per_worker} new probes.
Implement and request the first probe by your second or third model response. Revise from evidence and test again.
The complete starter entry files are supplied below. Avoid repeated reads and status polling.
Use read_candidate(index) to inspect exactly the entry files of a measured frozen candidate; -1 means the starter.
You may integrate useful public candidate designs into your own branch, then measure the integrated version.
Candidate indices refer to completed public probes in swarm_status. No private outcomes are available.
Each worker owns its branch. Never modify peer artifacts. No additional agents may be spawned.
Only a publicly measured frozen candidate can be chosen. The Astra lead makes the final engineering selection.
Final verification begins after all agents stop and selection freezes; final results are never returned to agents.
Starter public evidence:
{json.dumps(F.compact_report(self.coordinator.baseline_report),separators=(',',':'))}
{files}
"""
        if self.lead:
            text += ("\nYou are the Astra lead. Explore, integrate, and measure improvements while explorers work. "
                "Use choose_candidate(index, reason) near the end to choose any completed public candidate, including your own or starter -1. "
                "Explain your engineering judgment from the public task and measured evidence. This irrevocably freezes your choice and ends your branch. "
                "An integrated design must be measured before it is eligible. If you make no explicit choice, the host falls back to the public proxy's best measured incumbent.\n")
        else:
            text += "\nYou are an explorer. Share concise measured findings and submit a useful frozen branch for the lead to assess.\n"
        return text

    def definitions(self):
        definitions = super().definitions()
        definitions.append({"name":"read_candidate","description":"Read six entry files and public metrics from a verified measured frozen candidate; -1 is the starter.",
            "inputSchema":{"type":"object","properties":{"index":{"type":"integer","minimum":-1}},"required":["index"],"additionalProperties":False}})
        if self.lead:
            definitions.append({"name":"choose_candidate","description":"Astra lead only: irrevocably choose a measured public candidate and end the lead branch.",
                "inputSchema":{"type":"object","properties":{"index":{"type":"integer","minimum":-1},"reason":{"type":"string"}},
                               "required":["index","reason"],"additionalProperties":False}})
        return definitions

    def dispatch(self,name,arguments):
        if name in ("read_candidate","choose_candidate"):
            if self.submitted is not None or time.monotonic() >= self.deadline:
                raise ValueError("This agent's development has ended")
            if name == "read_candidate":
                return self.coordinator.read_candidate(arguments["index"])
            return self.coordinator.choose_candidate(self,arguments["index"],arguments["reason"])
        return super().dispatch(name,arguments)
