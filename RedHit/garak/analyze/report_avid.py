#!/usr/bin/env python

"""Prints a report given a garak report in jsonl"""

import importlib
import sys
import json
import argparse
import pandas as pd

from datetime import date
from avidtools.datamodels.report import Report
from avidtools.datamodels.components import *
from garak import _config

evals = []
meta = None

parser = argparse.ArgumentParser(
    description="Conversion Tool from native garak jsonl to AVID reports"
)
# model type; model name; seed; generations; probe names; eval threshold
parser.add_argument(
    "--report",
    "-r",
    type=str,
    help="location of garak report file",
)
_config.args = parser.parse_args(sys.argv[1:])
_config.args.verbose = 0

# load up a .jsonl output file, take in eval and config rows
report_location = _config.args.report
print(f"📜 Converting garak reports {report_location}")
with open(report_location, "r", encoding="utf-8") as reportfile:
    for line in reportfile:
        record = json.loads(line.strip())
        if record["entry_type"] == "eval":
            evals.append(record)
        elif record["entry_type"] == "config":
            meta = record
if len(evals) == 0:
    raise ValueError("No evaluations to report 🤷")

# preprocess
for i in range(len(evals)):
    module_name, plugin_class_name = evals[i]["probe"].split(".")
    mod = importlib.import_module(f"garak.probes.{module_name}")

    evals[i]["probe"] = f"{module_name}.{plugin_class_name}"
    plugin_instance = getattr(mod, plugin_class_name)()
    evals[i]["probe_tags"] = plugin_instance.tags

evals_df = pd.DataFrame.from_dict(evals)
evals_df = evals_df.assign(score=lambda x: (x["passed"] / x["total"] * 100))
probe_scores = evals_df[["probe", "score"]].groupby("probe").mean()

# set up a generic report template
report_template = Report()
if meta is not None:
    report_template.affects = Affects(
        developer=[],
        deployer=[meta["model_type"]],
        artifacts=[Artifact(type=ArtifactTypeEnum.model, name=meta["model_name"])],
    )

report_template.references = [
    Reference(
        type="source",
        label="garak, an LLM vulnerability scanner",
        url="https://github.com/NVIDIA/garak",
    )
]
report_template.reported_date = date.today()

# now build all the reports
all_reports = []
for probe in probe_scores.index:
    report = report_template.copy()
    probe_data = evals_df.query(f"probe=='{probe}'")

    report.description = LangValue(
        lang="eng",
        value=f"The model {meta['model_name']} from {meta['model_type']} was evaluated by the Garak LLM Vunerability scanner using the probe `{probe}`.",
    )
    report.problemtype = Problemtype(
        classof=ClassEnum.llm, type=TypeEnum.measurement, description=report.description
    )
    report.metrics = [
        Metric(
            name="",
            detection_method=Detection(type=MethodEnum.thres, name="Count failed"),
            results=probe_data[["detector", "passed", "total", "score"]]
            .reset_index()
            .to_dict(),
        )
    ]
    all_tags = probe_data.iloc[0]["probe_tags"]
    if all_tags == all_tags:  # check for NaN
        tags_split = [
            tag.split(":") for tag in all_tags if tag.startswith("avid")
        ]  # supports only avid taxonomy for now
        report.impact = Impact(
            avid=AvidTaxonomy(
                risk_domain=pd.Series([tag[1].title() for tag in tags_split])
                .drop_duplicates()
                .tolist(),  # unique values
                sep_view=[SepEnum[tag[2]] for tag in tags_split],
                lifecycle_view=[LifecycleEnum["L05"]],
                taxonomy_version="",
            )
        )
    all_reports.append(report)

# save final output
write_location = report_location.replace(".report", ".avid")
with open(write_location, "w", encoding="utf-8") as f:
    f.writelines(r.json() + "\n" for r in all_reports)
print(f"📜 AVID reports generated at {write_location}")
