# Which code ran, and where

The submission was produced by one Python package, `src/ragtime`, running as two processes on two
clusters: a long-lived retrieval service holding the index resident, and short-lived pipeline
workers querying it.

| | retrieval service | pipeline workers |
| --- | --- | --- |
| cluster | baobab | bamboo |
| entry point | `ragtime.devkit.rsvc`, run as a module | `run --config <cfg> --stage pipeline` |
| lifetime | one allocation, weeks | one topic, hours |
| holds | the index (~410 GiB resident), the encoders, the reranker | the RAG loop, the model conversation |
| code it imported | a pinned checkout | the working tree of the moment |

The service's launcher sets `RAGTIME_PYPATH` to a frozen symlink farm, so editing the repository
could not change the behaviour of a service that had already paid a ~30-minute cold load. The
workers carried no such pin: each imported the repository as it stood when that worker started.
The service therefore ran older code than the client, by design.

The package here is the client tree. It is exact for the service program and the whole ranking
path, and approximate for the rest of the service's import closure: a few modules there carry
client-side additions made after the pin was taken, and the editorial pass that prepared this
repository for publication touched comments and docstrings in others.

All six shipped runs are `kind: e2e_agentic`, the three `mlir-*` included: Task 2 is that same
agentic pipeline with the search knob moved. The witness is on disk, since a cell directory is
`<run_id>__<variant>__seed<N>` and the deployed mlir cells are `mlir-omt__original__seed0`, keyed
on `passage_lang`, which is the branch `variant()` takes for `e2e_agentic`.

## Where the pin and this tree differ

`src/ragtime/config/schema.py` declares how many seeds each run family expects. The e2e count was
cut from five to one during the campaign, and the pinned service predates the cut, so it still
validates against the old rule. That reaches config validation only, never ranking: the service
validates the config it is handed at boot and then serves queries from the index. The shipped
configs say `seeds: 1`, this schema expects `seeds: 1`, and the runs behind the submission were
single-seed.

## The retrieval service as it ran

The service ran as an `srun --overlap` step into an allocation held for the whole campaign, on six
80 GiB cards, with one replica sharding the four language cells across four of them and a reranker
instance on each of the six. `slurm/retrieval_service_step.sh` is that line, parameterized;
`slurm/retrieval_service.sbatch` submits the same service as a fresh job but derives a smaller
shape unless `SVC_REPLICAS` and `SVC_EXTRA` are set, because it has no knob of its own for the
reranker devices. `slurm/README.md` gives both. The candidate set that reached the cross-encoder
was the top `retrieval.reranker.depth` of the fused ranking; the ranking as it actually ran is in
[`pipeline.md`](pipeline.md).

## The index was validated before it served

The shipped index — `final/f308301501d9/index/949acf17d993`, 9,941,840 passages per rendering —
passed the corpus acceptance pass in [`src/ragtime/preprocess/acceptance.py`](../src/ragtime/preprocess/acceptance.py)
on 2026-08-07, run through [`slurm/acceptance.sbatch`](../slurm/acceptance.sbatch): step 0, phase A,
phase B1 over all ten cells and phase B2 over the three non-English languages, every fragment
`passed` with no failures and the dense hard gate green in all ten cells. SLURM jobs 4286554,
4286580, 4286582 and 4290500 all completed `0:0`, and `main()` returns non-zero on any hard-gate
failure, so the exit codes are a second and independent statement that the gates held.

The directory it validated is the directory that served: acceptance recorded `index_hash
2072b711bfa2` against `reconcile_hash f308301501d9` and `packing_hash 4e12ee30`, and the index tree's
preserved `manifest.json.pre-rekey-backup` carries the same three values and the same
`created_at`. The tree was re-keyed on 2026-08-09, which changed its name and not its contents.
