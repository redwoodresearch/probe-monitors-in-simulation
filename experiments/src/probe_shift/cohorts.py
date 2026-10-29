"""Load cohorts by S3 prefix or local folder, for cohorts not (yet) in simulated-control-evals' `real_runs.yml`.

Same layout and same return values as `probe_shift.transcripts` (`<root>/logs/<file>.eval`, samples as dicts), so
`invalidity`, `reader_messages` and the rest work unchanged on what these return. `<root>` is an `s3://bucket/prefix/`
or a local folder holding `logs/`. `COHORTS` names the project's own cohorts until they have a `real_runs.yml` alias;
once one does, cite that alias through `probe_shift.transcripts` instead.
"""

import io
import json
import re
import zipfile
from pathlib import Path

from probe_shift.transcripts import _s3, resolve_attachments

# Project cohorts generated after the paper (experiments/2026-10-06_astra_sim_honest/README.md).
COHORTS = {
    # Astra simulated honest runs matched to la_public10_sim_astra_attack_n1000: September simulator code, 437 tasks.
    "la_public10_sim_astra_honest_n437": "s3://rr-sce-sim-transcripts/transcripts/"
    "2026-10-06_la-public10_sim-astra-medium-honest-origtraj-read-nocritic-fork0/",
    # Today's-simulator pilot (sims main e561f936), 20 tasks: honest and attack, same references and settings.
    "la_public10_sim_astra_honest_simcode_e561f936_n20": "s3://rr-sce-sim-transcripts/transcripts/"
    "2026-10-06_la-public10_sim-astra-medium-honest-origtraj-read-nocritic-fork0-simcode-e561f936-n20/",
    "la_public10_sim_astra_attack_simcode_e561f936_n20": "s3://rr-sce-sim-transcripts/transcripts/"
    "2026-10-06_la-public10_sim-astra-medium-attack-origtraj-read-nocritic-fork0-simcode-e561f936-n20/",
}


def root_of(cohort: str) -> str:
    """A `COHORTS` name, an `s3://` prefix, or a local folder, as the root that holds `logs/`."""
    root = COHORTS.get(cohort, cohort)
    return root if root.startswith("s3://") else str(Path(root).expanduser().resolve())


def _read(root: str, file: str) -> bytes:
    if root.startswith("s3://"):
        bucket, prefix = root.removeprefix("s3://").split("/", 1)
        return _s3().get_object(Bucket=bucket, Key=f"{prefix.rstrip('/')}/logs/{file}")["Body"].read()
    return (Path(root) / "logs" / file).read_bytes()


def eval_files(cohort: str) -> list[str]:
    root = root_of(cohort)
    if not root.startswith("s3://"):
        return sorted(p.name for p in (Path(root) / "logs").glob("*.eval"))
    bucket, prefix = root.removeprefix("s3://").split("/", 1)
    pages = _s3().get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=f"{prefix.rstrip('/')}/logs/")
    return sorted(o["Key"].rsplit("/", 1)[-1] for p in pages for o in p.get("Contents", [])
                  if o["Key"].endswith(".eval") and o["Key"].count("/") == f"{prefix.rstrip('/')}/logs/".count("/"))


def file_samples(cohort: str, file: str) -> list[dict]:
    with zipfile.ZipFile(io.BytesIO(_read(root_of(cohort), file))) as z:
        return [resolve_attachments(json.loads(z.read(n))) for n in z.namelist() if n.startswith("samples/") and n.endswith(".json")]


def load_sample(cohort: str, file: str, sample_id: str, epoch: int) -> dict:
    for sample in file_samples(cohort, file):
        if sample["id"] == sample_id and sample["epoch"] == epoch:
            return sample
    raise KeyError(f"sample {sample_id} epoch {epoch} not in {cohort}/{file}")


def manifest_file(cohort: str) -> str:
    """The cohort's newest manifest: the highest `manifest_v<N>.json`, else `manifest.json`. Published manifests are
    never overwritten; a correction is added beside them as the next version, with a `corrections` entry."""
    root = root_of(cohort)
    if root.startswith("s3://"):
        bucket, prefix = root.removeprefix("s3://").split("/", 1)
        pages = _s3().get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=f"{prefix.rstrip('/')}/manifest",
                                                                Delimiter="/")
        names = [o["Key"].rsplit("/", 1)[-1] for p in pages for o in p.get("Contents", [])]
    else:
        names = [p.name for p in Path(root).glob("manifest*.json")]
    versions = {int(m[1]): n for n in names if (m := re.fullmatch(r"manifest_v(\d+)\.json", n))}
    if versions:
        return versions[max(versions)]
    assert "manifest.json" in names, f"no manifest in {root}"
    return "manifest.json"


def manifest(cohort: str) -> dict:
    """The cohort's newest manifest (settings, per-task reference and outcome), from S3 or the local folder. Cite
    `manifest_file(cohort)` with it."""
    root, name = root_of(cohort), manifest_file(cohort)
    if root.startswith("s3://"):
        bucket, prefix = root.removeprefix("s3://").split("/", 1)
        return json.loads(_s3().get_object(Bucket=bucket, Key=f"{prefix.rstrip('/')}/{name}")["Body"].read())
    return json.loads((Path(root) / name).read_text())
