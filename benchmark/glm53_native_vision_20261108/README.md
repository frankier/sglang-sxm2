# GLM-5.3-Flash native-image acceptance

See [the serving guide](../../docs/v100/GLM-5.3-Flash.md#native-document-validation-2026-11-08)
for the launch profile, results and scope. [summary.json](summary.json) stores
per-cohort official scores, usage, TTFT, sampled GPU memory by rank and source
hashes. It deliberately distinguishes connected clients, observed active decode
and estimated overlapping output spans.

Artifacts remain on `robefran@turso.cs.helsinki.fi` in the isolated workspace
`~/sglang-issue1-20261108`:

- `results/run-2945071`: exact-image replay; five 16-page cohorts; raw responses;
  server logs; 200 ms GPU telemetry; admission, packed-frame and cache checks;
  official scoring outputs and health bundles.
- `results/mtp-2945107`: final-code native no-MTP OpenAI requests, scheduler-
  confirmed cached follow-up and Anthropic completion. Compact evidence is in
  [follow-up-summary.json](follow-up-summary.json). The `mtp-` directory prefix
  belongs to the extra-check harness; this successful run used MTP=0.
- `validate.py`, `validate.sbatch`: request and server orchestration.
- `eval/`: unchanged evaluation clients copied from the original failed run.
- `score.sh`, `score.sbatch`, `config.env.sh`: pinned official evaluator launch.
- `summarize.py`: metric, scheduler-count and timing aggregation.
- `results/unit-before.log`: new attention regressions fail on the old path.
- `results/unit-final.log`, `results/unit-device.log`: final CPU suites pass.
- `results/mtp-2945090.log`: final V100 attention regressions pass; extra MTP
  startup then fails in draft-weight scale preparation, before any request.

Scoring jobs 2945077, 2945086, 2945087, 2945088 and 2945094 completed. Every
cohort retains all 16 pages. Original failed inference and scoring evidence in
`~/llmturso-work/results/sglang-sxm2-vision/` was not changed.

The acceptance run used the original POSIX shared-memory transport. For the
additional final-code follow-up, a validation-only `disk_mm_shm.c` shim redirects
only `sgl_shm_mm_*` CPU pixel segments to the task's disk-backed `ipc/` directory.
It preserves normal small NCCL IPC and does not change image preprocessing,
attention, weights or admission. This shim is test infrastructure, not a serving
requirement or a production code change. Failed whole-`/dev/shm` disk-bind
attempts and their logs are preserved separately.
