# Data setup: acquiring the reference data

Every local data source the harness reads declares, in its descriptor (`configs/data/sources/<source>.yaml`), how its
files are acquired: the release, where the files are published, which files make up each table, how they are
checked, the step that prepares them (when one is needed), where they land, and the licence and login notes. One
generic command reads those sections and does the rest. There is no source-specific acquisition code: a new
source, or a new release of an existing one, is a change to its `acquisition` section.

```bash
vbt data status                                   # every declared table: present, verified, ready, tools it unlocks
vbt data acquire --all --plan                     # what everything would take: files, bytes, time, licences
vbt data acquire open_targets --env-file .env     # acquire a source, then point the data layer at it
vbt data acquire open_targets.target tahoe_100m.metadata      # single tables, or extra download groups
vbt data acquire --for-agents genomics-analyst --plan         # the tables an agent's tools read
vbt data acquire --for-tools 'target.*' functional_genomics.query_drug_perturbation
vbt ds check                                      # readiness of what is now present
```

`vbt data acquire` always prints the plan first. With `--plan` it stops there and writes nothing; without it, it
carries the plan out. The steps are:

1. **List** each source once through its transport, or use the declared sizes with `--offline`. Each file is put
   in one state: verified by an earlier manifest, present but unverified, partial (`.part`), or missing.
2. **Check the space.** Refuse before any transfer when the files left to download exceed `--max-gb`, or the free
   disk minus `data.acquisition.reserve_bytes`.
3. **Download** in parallel (`--workers`, default `data.acquisition.workers`). Interrupted files resume from their
   `.part` with range requests. A file takes its name only after its size, its publisher checksum and, where
   declared, its Parquet framing all match.
4. **Write the manifest** `.download-manifest.json` in the upstream downloader's format: `release`, `base`,
   `expected_files`, `complete`, and `files{path: {bytes, sha256, url, <publisher algorithm>}}`. The data layer's
   readiness check (R2) and the upstream doctor read it.
5. **Run the prepare steps** whose inputs are all verified. Each runs an unmodified upstream script.
6. **Pin the release** in `<home>/.vbt-acquisition.json`. A home that holds another release is refused.
7. **Record provenance**: one JSON line in `<data.provenance.dir>/acquisitions.jsonl`.
8. **Print the variables** that point the data layer at the files. `--env-file PATH` writes them as
   `KEY="value"` lines; the project `.env` is read at start-up.

Run it again at any time. Verified files are not transferred twice, and a rerun of a complete acquisition only
lists the source and rewrites the manifest.

## Where the files land

`data.acquisition.root` defaults to `${VBT_DATA_DIR:-data}/sources`; `--dest ROOT` overrides it for one command.
Each source has a home `<root>/<acquisition.dir>` (default `{source}/{release}`), so releases sit side by side.

| source | release (pinned) | home under the root | variable set | declared size |
|---|---|---|---|---|
| `open_targets` | 25.09 | `open_targets/25.09` | `OPEN_TARGETS_DATA_PATH` | 38 tables, 3,508 files, 31,131,380,890 bytes |
| `tahoe_100m` | HF revision `2dc57900…5a95` | `tahoe/<revision>` | `TAHOE_DATA_PATH` (= `<home>/prepared`) | 1,026 DE shards, 88,859,715,303 bytes, + 4 metadata files, 1,451,950 bytes; then the preparation |
| `depmap` | 24Q4 | `depmap/24Q4` | `DEPMAP_DATA_PATH` | 4 files, 850,460,784 bytes |
| `gene_ontology` | GO archive 2026-08-05 (data-version releases/2026-07-26) | `gene_ontology/2026-08-05` | `GO_DATA_PATH` | 32,227,785 bytes |
| `cell_ontology` | CL 2026-06-08 | `cell_ontology/2026-06-08` | `VBT_CL_OBO` | 3,347,518 bytes (+ `uberon_basic`, optional, 12,155,980 bytes) |
| `msigdb` | 2024.1.Hs | `msigdb/2024.1.Hs` | `MSIGDB_DATA_PATH` | 48,690 bytes |
| `zenodo_vbt` | record 22259123 | `zenodo/22259123` | `VBT_ZENODO_DIR` | 30 archive members, 92,001,168 bytes |
| `cellxgene_census` | 2025-11-08 | — | — | read live from the public S3 bucket; nothing to download |

The sizes are those the descriptors declare. They were observed on the real sources on 2026-10-08, and the opt-in
test `VBT_DL_NETWORK=1 pytest tests/datalayer/test_dl_acquisition.py` compares them with live listings. The
variables are passed to the data child through `tool_env` in `configs/default.yaml`.

## Licences and logins

Each plan prints the source's `acquisition.licence` (and `login` when a person must accept terms or log in first);
a refused call quotes them too. As declared:

| source | licence |
|---|---|
| Open Targets Platform | CC0 1.0 |
| Tahoe-100M | CC0 1.0 (dataset card) |
| DepMap 24Q4 (figshare) | CC BY 4.0 |
| Gene Ontology, Cell Ontology | CC BY 4.0 |
| MSigDB Hallmark | CC BY 4.0 for v2022.1 and later. The KEGG-, BioCarta- and STKE-derived C2/M2 sets have further terms and are not acquired |
| Zenodo case-study archive | CC BY 4.0 |
| CELLxGENE Census | per dataset (read live) |

No shipped source needs a login. A gated Hugging Face repository takes `token_env: <VARIABLE>` in its transport
options; the token is read from that variable at transfer time and never stored. A source that only a person can
fetch declares `mode: manual` and a `login` text, and the plan shows that text instead of a transfer.

## Transports

A transport is an acquisition plugin: the `acquisition` plugin kind, a harness-side kind beside the data layer's
five. The builtins are generic, each covering a protocol or a hosting platform:

| plugin | lists with | checks each file by |
|---|---|---|
| `http` | an explicit file list, the server's HTML directory index, or a checksum list (`<hex>  <path>`, e.g. Open Targets' `release_data_integrity`, itself checked against its `.sha1`) | the list's sha1/sha256/md5, or a sha256 pinned in the descriptor |
| `huggingface` | the tree API at a pinned revision (paged through `Link`) | the LFS sha256; the git blob id for files without LFS |
| `s3`, `gcs` | ListObjectsV2 / the JSON API, anonymous | md5 (single-part S3 ETag, GCS `md5Hash`) |
| `json_index` | a JSON document read with JSONPaths (a Zenodo record, a figshare article) | the md5 the index reports |
| `zip_member` | the central directory of a remote zip (range requests), found directly or through a JSON index | the member's CRC-32 |

A new platform is a new plugin, either a module in `vbt.datalayer.plugins.acquisition` or an entry point in group
`vbt.datalayer.acquisition`. It must pass the conformance suite
`src/vbt/datalayer/plugins/conformance/acquisition.py` (A-1 to A-7), which serves the plugin's own index format
from a local HTTP server:

1. A-1: the listing is exactly the published files.
2. A-2: sizes and checksums match the files.
3. A-3: fetches return the files' bytes, and ranged fetches return the rest of a file.
4. A-4: a missing or damaged file raises an error.
5. A-5: a failing index raises an error.
6. A-6: no name can escape the destination directory.
7. A-7: `describe()` returns one line.

Proxies and CA bundles come from the environment (`HTTPS_PROXY`, `SSL_CERT_FILE`).

## Prepare steps

Tahoe is the one shipped source whose tables are not the downloaded files. `vbt data acquire tahoe_100m` first
downloads the DE shards and the metadata tables and verifies each sha256. It then runs the unmodified upstream
script as `{python} -B {upstream}/tools/prepare_tahoe.py <home> <home>/prepared --source-revision <revision>`.
`{upstream}` is `vars.upstream` (`VBT_UPSTREAM`). The script writes every table of the descriptor and
`preparation_manifest.json`, which the R2 check reads.

The step runs only when every input group is verified. It is skipped when its output already holds a complete
manifest. When the output directory exists without a complete manifest, the step refuses to run, because the
script creates its output whole; move the directory aside to prepare again.

`vbt data acquire tahoe_100m.metadata` fetches only the four metadata files (1.45 MB). No Tahoe table becomes
ready without the preparation, and the preparation reads every shard.

A run on this development machine, with one real shard (91,442,763 bytes) in a scratch copy of the descriptors:
the unmodified script read 3,986,181 rows and wrote 155,254 permissive, 117,979 significant and 82,616
high-quality rows in 2.0 s. The seven tables were then `ready` by `vbt data status --check`.

## Status

`vbt data status [SOURCE ...]` lists every declared table with these columns:

| column | meaning |
|---|---|
| present | the files at the location the data layer reads now, or `remote` |
| verified | whether every present file is in the manifest with its size |
| ready | from the readiness cache (`--check` runs `vbt ds check` on the present tables) |
| unlocks | the reviewed tools whose bindings read the table (`--tools` names them) |

When the files are at the acquisition home but the variable is unset, the source line says which variable to set.
`--json` gives the same data in machine-readable form.

## On demand: a refused call says how to acquire

When a tool call is refused `not_ready` because a table's files are absent, partial or of another release, the
reason's `hint` says how to fix it:

```
the table's files are absent; acquire them with `vbt data acquire open_targets.target`
(75.57 MB in 10 file(s), release 25.09); licence: CC0 1.0 ...; then rerun `vbt ds check`
```

The readiness reason also carries the same facts as a structured `acquire` entry: command, bytes, files, prepare
steps, licence, login, and the policy's decision.

What happens next is the operator's policy, `data.acquisition` in the configuration:

```yaml
data:
  acquisition:
    root: /srv/vbt/data/sources   # default ${VBT_DATA_DIR:-data}/sources
    auto: off                     # off | ask | under_budget
    budget_bytes: 5 GB            # under_budget: acquisitions up to this size happen between turns
    workers: auto                 # parallel transfers (auto: 4 per CPU, at most 32)
    rate_mbps: 50                 # the rate the plan assumes until one is measured on this host
    reserve_bytes: 1 GiB          # free disk kept after a download
    retries: 4
    timeout_s: 120
```

| `auto` | what happens after a refusal |
|---|---|
| `off` (default) | the refusal says how; a person runs `vbt data acquire` |
| `ask` | the tables are queued in `<data.cache_dir>/acquisition/pending.json`; `vbt data acquire --pending` acquires them once an operator approves |
| `under_budget` | between turns the system acquires the refused tables when the whole acquisition fits `budget_bytes` (prepare inputs included); larger ones are queued as with `ask`. Each such acquisition is recorded with `by: auto`, the policy and the budget, in `acquisitions.jsonl` and in the run's `data_acquisitions.jsonl` |

The between-turns step is `vbt.data.ondemand.between_turns(config, refusals=..., run_dir=...)`. Without refusals
it takes every table that the readiness cache reports as missing, partial or stale.
`vbt data acquire --missing` does the same from the command line, for example from a scheduled job.

## Releases

Each source acquires one pinned release:

* `acquisition.release` is the release to fetch: a revision for Hugging Face, a record for Zenodo, a dated archive
  for GO and CL.
* The descriptor's `release.expect` is what readiness compares against the manifest.

To move to a new release, change both, and the table paths if they name the release (MSigDB). Then run
`vbt data acquire <source> --plan`. The new release lands in its own home next to the old one, and `--env-file`
points the data layer at it.

## Older commands

* `vbt data ot list|fetch|manifest` is kept. It now goes through the same engine and the `open_targets`
  acquisition section, with `--dest` as the release directory.
* `vbt data zenodo list|fetch|download|presets` extracts any part of the paper's case-study archive, beyond the
  tables the `zenodo_vbt` descriptor declares.
