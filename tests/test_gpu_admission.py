"""The GPU admission gate: refuse to start beside a lane that still owns the card.

Two things are pinned here. The gate's own arithmetic, and the wire between the unit and the
gate -- a unit that ships without the gate is the bug that motivated it: arbiter came up on a
GPU a llama lane still held, every metric recorded looked fine, and the box was unusable in
practice. That failure is why refusing is a non-zero exit rather than a warning: the unit
carries Restart=on-failure, so a refusal becomes a retry and never an oversubscription.

nvidia-smi is faked on PATH rather than assumed absent. gx10 has a driver installed and
fkzllama has two cards, so a test that leans on the tool being missing would pass on CI by
accident and lie on every machine that actually runs this.
"""
import os
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GATE = os.path.join(REPO, "wait-vram.sh")
UNIT = os.path.join(REPO, "systemd", "arbiter.service")
BUILD = os.path.join(REPO, "build-venv.sh")


def _fake_smi(tmp_path, answers):
    """An nvidia-smi on PATH that answers per card index, draining one queue per card.

    `answers` maps a card index to the values that card will report, one value per call. A
    card with no queue answers with nothing, which is what a missing or failing driver
    reports -- and the gate has to read that as no free memory, never as a licence to start.
    """
    bin_dir = os.path.join(str(tmp_path), "bin")
    os.makedirs(bin_dir, exist_ok=True)
    for index, values in answers.items():
        with open(os.path.join(bin_dir, "answers." + str(index)), "w") as handle:
            handle.writelines(str(value) + "\n" for value in values)
    shim = os.path.join(bin_dir, "nvidia-smi")
    with open(shim, "w") as handle:
        handle.write(
            "#!/bin/sh\n"
            'd=$(dirname "$0")\n'
            'f="$d/answers.$2"\n'
            '[ -f "$f" ] || exit 0\n'
            'val=$(head -n 1 "$f")\n'
            'tail -n +2 "$f" > "$f.t" && mv "$f.t" "$f"\n'
            'printf "%s\\n" "$val"\n'
        )
    os.chmod(shim, 0o755)
    return dict(os.environ, PATH=bin_dir + os.pathsep + os.environ["PATH"])


def _run_gate(tmp_path, answers, *args):
    return subprocess.run(["bash", GATE, *map(str, args)], capture_output=True,
                          text=True, env=_fake_smi(tmp_path, answers))


def test_admits_onto_a_card_that_is_actually_free(tmp_path):
    # 15885 of 16311 MiB: what fkzllama reported the moment the 35B lane released the card.
    done = _run_gate(tmp_path, {1: ["15885"]}, 1, 6144, 0)
    assert done.returncode == 0, done.stderr
    assert "15885 MiB free" in done.stdout
    assert "admitting arbiter" in done.stdout


def test_refuses_a_card_a_llama_lane_still_owns(tmp_path):
    # 980 MiB free with a 14.9 GiB load resident: the 2x2 configuration that must never return.
    done = _run_gate(tmp_path, {1: ["980"]}, 1, 6144, 0)
    assert done.returncode == 1, f"refusal has to be loud, got exit {done.returncode}"
    assert "FATAL" in done.stderr
    assert "refuses to oversubscribe" in done.stderr


def test_waits_for_a_released_card_instead_of_failing_at_once(tmp_path):
    # The card is mid-teardown: 980 now, 15885 one poll later. The gate sits out the wait.
    done = _run_gate(tmp_path, {1: ["980", "15885"]}, 1, 6144, 60)
    assert done.returncode == 0, done.stderr
    assert "waiting for gpu1 vram: 980/6144 MiB" in done.stdout
    assert "admitting arbiter" in done.stdout


def test_stops_waiting_and_fails_loudly_when_the_card_never_empties(tmp_path):
    done = _run_gate(tmp_path, {1: ["980"] * 5}, 1, 6144, 0)
    assert done.returncode == 1
    assert "stuck at 980 MiB free" in done.stderr


def test_strips_the_units_and_spacing_the_tool_wraps_around_the_number(tmp_path):
    # Asked without csv,noheader,nounits the tool decorates the answer; digits alone still count.
    done = _run_gate(tmp_path, {1: ["  15885 MiB "]}, 1, 6144, 0)
    assert done.returncode == 0, done.stderr
    assert "admitting arbiter" in done.stdout


def test_treats_an_unparsable_answer_as_no_free_memory(tmp_path):
    done = _run_gate(tmp_path, {1: ["ERROR: no data available"]}, 1, 6144, 0)
    assert done.returncode == 1
    assert "stuck at 0 MiB" in done.stderr


def test_the_card_asked_for_is_the_card_consulted(tmp_path):
    # Two cards, one free. Reading the neighbour's headroom would put arbiter on a busy card,
    # which is the whole failure: the gate has to look at the index the unit named.
    free_card = {0: ["15885"]}
    assert _run_gate(tmp_path, free_card, 0, 6144, 0).returncode == 0
    assert _run_gate(tmp_path, free_card, 1, 6144, 0).returncode == 1


def test_prints_usage_when_no_card_index_is_given(tmp_path):
    done = _run_gate(tmp_path, {})
    assert done.returncode != 0
    assert "gpu index" in done.stderr


def test_the_unit_wires_the_gate_before_the_server():
    """An ExecStartPre that does not gate is the bug, so the wire is pinned too."""
    with open(UNIT) as handle:
        unit = handle.read()
    gates = [line for line in unit.splitlines() if line.startswith("ExecStartPre=")]
    assert len(gates) == 1, gates
    assert "wait-vram.sh" in gates[0]
    assert "ExecStart=/opt/arbiter/run.sh serve-foreground" in unit
    assert "Environment=ARBITER_DEVICE=cuda" in unit      # never a silent CPU fallback
    assert "Environment=CUDA_VISIBLE_DEVICES=1" in unit   # never drift onto the neighbour card


def test_the_build_script_pins_the_versions_that_decided_the_outcome():
    """The build is reproducible only while the pins are the ones that were measured."""
    with open(BUILD) as handle:
        build = handle.read()
    assert "laya==0.3.4" in build
    assert "mcp>=2" in build                              # the bridge's one third-party import
    assert "download.pytorch.org/whl/cu130" in build      # the CUDA index, both architectures
    assert "guard_policy.py" in build                      # without it the approval gate is dead
    assert "huggingface_hub download" not in build         # weights are copied in, never re-pulled
