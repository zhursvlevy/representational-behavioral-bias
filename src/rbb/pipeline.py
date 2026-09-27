"""Run the full pipeline, or one stage of it. Each step runs as a separate process
(fresh GPU state between models); steps skip work whose outputs already exist, so an
interrupted run can be restarted with the same command.

    python -m rbb.pipeline all
    python -m rbb.pipeline stimuli          # one stage
    python -m rbb.pipeline --list all       # print the commands only

Stages, in order:
    imagenet16   materialize the 16-class ImageNet validation subset
    stimuli      cue-conflict stimuli (main set, dense lambdas, crop control)
    ceilings     ceiling stimuli (source photos, stylized white canvases)
    extract      hidden states and answers for every model and condition (GPU, hours)
    probe        fold assignments and grouped cross-validated probes
    analyses     all results/ tables
    figures      all paper figures
    validate     optional: torch probes vs. scikit-learn
"""

import argparse
import subprocess
import sys

from rbb import config

PY = [sys.executable, "-m"]


def conditions(model: str) -> list[str]:
    names = [config.condition("main", s) for s in config.main_scales(model)]
    names += ["ceiling_shape"] + [f"ceiling_texture_l{s}" for s in config.SCALES] + ["geirhos"]
    if model == "llava":
        names += [config.condition("crop", s) for s in config.CROP_SCALES]
    return names


def stage_commands(stage: str) -> list[list[str]]:
    if stage == "imagenet16":
        return [PY + ["rbb.stimuli.imagenet16"]]
    if stage == "stimuli":
        scales = sorted(config.SCALES + config.DENSE_SCALES)
        cmds = [PY + ["rbb.stimuli.cue_conflict", "--scale", str(s)] for s in scales]
        cmds += [PY + ["rbb.stimuli.cue_conflict", "--scale", str(s), "--style-source", "crop",
                       "--limit", str(config.CROP_N_STIMULI)] for s in config.CROP_SCALES]
        return cmds
    if stage == "ceilings":
        return [PY + ["rbb.stimuli.ceilings", "photos"]] + [
            PY + ["rbb.stimuli.ceilings", "white-canvas", "--scale", str(s)] for s in config.SCALES]
    if stage == "extract":
        cmds = [PY + ["rbb.extract", "--model", m, "--condition", c] for m in config.MODELS for c in conditions(m)]
        return cmds + [PY + ["rbb.analysis.recognizability"]]
    if stage == "probe":
        cmds = [PY + ["rbb.probing", "folds", "--set", s] for s in ("main", "crop", "geirhos")]
        for m in config.MODELS:
            for c in conditions(m):
                if c.startswith("ceiling_shape"):
                    args = ["--folds", "main", "--labels", "shape"]
                elif c.startswith("ceiling_texture"):
                    args = ["--folds", "main", "--labels", "texture"]
                else:
                    args = ["--folds", c.split("_l")[0]]
                cmds.append(PY + ["rbb.probing", "grouped", "--model", m, "--condition", c] + args)
        return cmds
    if stage == "analyses":
        return [PY + [module] + extra for module, extra in (
            ("rbb.analysis.behavior", []),
            ("rbb.analysis.ceilings", []),
            ("rbb.analysis.correlation", []),
            ("rbb.analysis.within_condition", ["last-layer"]),
            ("rbb.analysis.within_condition", ["depth"]),
            ("rbb.analysis.crop_control", []),
            ("rbb.analysis.protocol", []),
            ("rbb.analysis.stimulus_distances", []),
        )]
    if stage == "figures":
        return [PY + ["rbb.figures.curves"], PY + ["rbb.figures.stimuli"], PY + ["rbb.figures.appendix"]]
    if stage == "validate":
        return [PY + ["rbb.analysis.validate_probe"]]
    raise ValueError(stage)


STAGES = ["imagenet16", "stimuli", "ceilings", "extract", "probe", "analyses", "figures"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=STAGES + ["validate", "all"])
    parser.add_argument("--list", action="store_true", help="print the commands without running them")
    args = parser.parse_args()

    for stage in STAGES if args.stage == "all" else [args.stage]:
        for cmd in stage_commands(stage):
            print("+", " ".join(cmd[1:]), flush=True)
            if not args.list:
                subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
