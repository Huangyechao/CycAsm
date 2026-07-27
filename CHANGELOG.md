# Changelog

All notable changes to CycAsm are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-07-27

First public release.

### Added

- `cycasm.sh`: one-command hybrid assembly pipeline wrapping the full workflow
  (hifiasm assembly → bwa-mem2 / minimap2 alignment → Thpolish polishing),
  with argument parsing, dependency checks, logging, and checkpoint resume.
- Thpolish v0.1.1 bundled under `Thpolish/`: haplotype-aware hybrid polisher
  (Rust core + Python feature scripts) with MIT license.
- S1 reagent models (`S1.filter.v1.0.pkl` and `S1.predict.v1.0.pt`) are
  distributed as release assets due to file size; place them under
  `Thpolish/model/` after download.
- `environment.yml` for one-step conda/mamba environment creation.
- Full documentation: installation, input requirements, quick start,
  step-by-step guide, output description, performance benchmarks, and FAQ.
