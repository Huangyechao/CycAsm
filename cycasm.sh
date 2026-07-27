#!/usr/bin/env bash
# =============================================================================
# CycAsm: CycloneSEQ long-read + DNBSEQ short-read hybrid genome assembly
#
# Steps:
#   1. hifiasm long-read assembly (CycloneSEQ, --ont mode)
#   2. Short-read (bwa-mem2) and long-read (minimap2) alignment
#   3. Thpolish haplotype-aware hybrid polishing
#
# Supports checkpoint resume: completed steps are skipped on re-run.
# =============================================================================
set -euo pipefail

VERSION="1.0.0"

usage() {
    cat <<'EOF'
Usage: cycasm.sh [options]

Required:
  -l, --long-read FILE     CycloneSEQ long reads (FASTQ, gzipped ok)
  -1, --short-r1 FILE      DNBSEQ short reads R1 (FASTQ)
  -2, --short-r2 FILE      DNBSEQ short reads R2 (FASTQ)
  -b, --thpolish-dir DIR   Thpolish source directory (contains py/, model/,
                           target/release/thPolish)

Optional:
  -p, --prefix STR         Assembly output prefix        [sample.asm]
  -o, --outdir DIR         Output directory              [.]
  -r, --reagent STR        Sequencing reagent/kit model  [S1]
                           (selects model/${REAGENT}.filter.v1.0.pkl and
                            model/${REAGENT}.predict.v1.0.pt)
  -t, --threads INT        Number of threads             [64]
  -s, --split INT          Number of genome splits for parallel polishing [30]
      --hap STR            Polish haplotype assembly instead of primary:
                           hap1 | hap2                   [primary]
  -f, --force              Ignore checkpoints and re-run all steps
  -h, --help               Show this help
  -v, --version            Show version
EOF
    exit "${1:-0}"
}

log()  { echo "[$(date '+%F %T')] $*" | tee -a "${LOG_FILE}" >&2; }
die()  { echo "[$(date '+%F %T')] ERROR: $*" >&2; exit 1; }

# ------------------------------ defaults -------------------------------------
LONG_READ=""; SHORT_READ1=""; SHORT_READ2=""; THPOLISH_DIR=""
PREFIX="sample.asm"; OUTDIR="."; REAGENT="S1"; THREADS=64; SPLIT_NUM=30
HAP="primary"; FORCE=0

# --------------------------- argument parsing --------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        -l|--long-read)     LONG_READ="$2";    shift 2 ;;
        -1|--short-r1)      SHORT_READ1="$2";  shift 2 ;;
        -2|--short-r2)      SHORT_READ2="$2";  shift 2 ;;
        -b|--thpolish-dir)  THPOLISH_DIR="$2"; shift 2 ;;
        -p|--prefix)        PREFIX="$2";       shift 2 ;;
        -o|--outdir)        OUTDIR="$2";       shift 2 ;;
        -r|--reagent)       REAGENT="$2";      shift 2 ;;
        -t|--threads)       THREADS="$2";      shift 2 ;;
        -s|--split)         SPLIT_NUM="$2";    shift 2 ;;
            --hap)          HAP="$2";          shift 2 ;;
        -f|--force)         FORCE=1;           shift   ;;
        -h|--help)          usage 0 ;;
        -v|--version)       echo "CycAsm ${VERSION}"; exit 0 ;;
        *)                  echo "Unknown option: $1" >&2; usage 1 ;;
    esac
done

[[ -n "${LONG_READ}"    ]] || { echo "Missing --long-read"    >&2; usage 1; }
[[ -n "${SHORT_READ1}"  ]] || { echo "Missing --short-r1"     >&2; usage 1; }
[[ -n "${SHORT_READ2}"  ]] || { echo "Missing --short-r2"     >&2; usage 1; }
[[ -n "${THPOLISH_DIR}" ]] || { echo "Missing --thpolish-dir" >&2; usage 1; }
[[ "${HAP}" =~ ^(primary|hap1|hap2)$ ]] || die "--hap must be primary|hap1|hap2"

mkdir -p "${OUTDIR}"
OUTDIR="$(cd "${OUTDIR}" && pwd)"
cd "${OUTDIR}"
LOG_FILE="${OUTDIR}/cycasm.log"
CKPT_DIR="${OUTDIR}/.cycasm_ckpt"
[[ ${FORCE} -eq 1 ]] && rm -rf "${CKPT_DIR}"
mkdir -p "${CKPT_DIR}"

# --------------------------- dependency checks -------------------------------
for tool in hifiasm bwa-mem2 minimap2 samtools fxTools python3 awk; do
    command -v "${tool}" >/dev/null 2>&1 || die "dependency not found in PATH: ${tool}"
done
THPOLISH_BIN="${THPOLISH_DIR}/target/release/thPolish"
[[ -x "${THPOLISH_BIN}" ]] || die "Thpolish binary not found: ${THPOLISH_BIN} (build with: LIBTORCH_USE_PYTORCH=1 cargo build --release)"
for f in "${THPOLISH_DIR}/py/depth.py" \
         "${THPOLISH_DIR}/py/find_candidates_predict.py" \
         "${THPOLISH_DIR}/py/prepeare_predict_input_rust.py"; do
    [[ -f "${f}" ]] || die "Thpolish script not found: ${f}"
done
FILTER_MODEL="${THPOLISH_DIR}/model/${REAGENT}.filter.v1.0.pkl"
PREDICT_MODEL="${THPOLISH_DIR}/model/${REAGENT}.predict.v1.0.pt"
[[ -f "${FILTER_MODEL}"  ]] || die "model not found: ${FILTER_MODEL}"
[[ -f "${PREDICT_MODEL}" ]] || die "model not found: ${PREDICT_MODEL}"
for f in "${LONG_READ}" "${SHORT_READ1}" "${SHORT_READ2}"; do
    [[ -f "${f}" ]] || die "input file not found: ${f}"
done

done_step() { [[ -f "${CKPT_DIR}/$1" ]]; }
mark_done() { touch "${CKPT_DIR}/$1"; log "step done: $1"; }

log "CycAsm ${VERSION} starting (prefix=${PREFIX}, reagent=${REAGENT}, threads=${THREADS}, hap=${HAP})"

# --------------------------- Step 1: assembly --------------------------------
case "${HAP}" in
    primary) GFA="${PREFIX}.bp.p_ctg.gfa" ;;
    hap1)    GFA="${PREFIX}.bp.hap1.p_ctg.gfa" ;;
    hap2)    GFA="${PREFIX}.bp.hap2.p_ctg.gfa" ;;
esac
GENOME="genome.fa"

if done_step step1_assembly; then
    log "skip step1 (checkpoint)"
else
    log "step1: hifiasm assembly (CycloneSEQ, --ont mode)"
    hifiasm -o "${PREFIX}" --ont -t "${THREADS}" "${LONG_READ}" 2> hifiasm.log
    [[ -f "${GFA}" ]] || die "expected hifiasm output not found: ${GFA}"
    awk '/^S/{print ">"$2; print $3}' "${GFA}" > "${GENOME}"
    mark_done step1_assembly
fi

# -------------------------- Step 2: alignment --------------------------------
if done_step step2_align_ngs; then
    log "skip step2.1 (checkpoint)"
else
    log "step2.1: short-read alignment (bwa-mem2 + thPolish filter)"
    [[ -f "${GENOME}.0123" ]] || bwa-mem2 index -p "${GENOME}" "${GENOME}"
    bwa-mem2 mem -c 1000 -a -B 2 -O 4,4 -E 1,1 -t "${THREADS}" \
        "${GENOME}" "${SHORT_READ1}" "${SHORT_READ2}" 2> bwa.log \
        | "${THPOLISH_BIN}" filter -t 3 --condensed-nm --max-mismatch 6 - \
        | samtools sort --write-index -@ "${THREADS}" -o ngs.bam
    mark_done step2_align_ngs
fi

if done_step step2_align_long; then
    log "skip step2.2 (checkpoint)"
else
    log "step2.2: long-read alignment (minimap2, CycloneSEQ-tuned)"
    minimap2 -t "${THREADS}" -a -k 16 -w 13 -A 2 -B 4 -O 4,41 -E 2,1 -s 180 \
        -U70,1000000 "${GENOME}" "${LONG_READ}" 2> minimap2.log \
        | samtools sort --write-index -@ "${THREADS}" -o long.bam
    mark_done step2_align_long
fi

if done_step step2_depth; then
    log "skip step2.3 (checkpoint)"
else
    log "step2.3: alignment depth"
    python3 "${THPOLISH_DIR}/py/depth.py" -b ngs.bam  -t "${THREADS}" -o ngs.bam.depth
    python3 "${THPOLISH_DIR}/py/depth.py" -b long.bam -t "${THREADS}" -o long.bam.depth
    mark_done step2_depth
fi

# -------------------------- Step 3: polishing --------------------------------
if done_step step3_polish; then
    log "skip step3 (checkpoint)"
else
    log "step3: Thpolish hybrid polishing (split=${SPLIT_NUM})"
    fxTools --split "${SPLIT_NUM}" "${GENOME}"

    shopt -s nullglob
    GENOME_BASE="$(basename "${GENOME}")"
    TARGET_DIR="${GENOME_BASE}.split${SPLIT_NUM}"
    FASTAS=(); VCFS=()

    for SPLIT_GENOME in "${TARGET_DIR}"/*fasta; do
        [[ -s "${SPLIT_GENOME}" ]] || continue
        BASE_NAME="$(basename "${SPLIT_GENOME}")"
        OUT_DIR="predict/out_${BASE_NAME}"
        mkdir -p "${OUT_DIR}"
        log "polishing fragment: ${SPLIT_GENOME}"

        python3 "${THPOLISH_DIR}/py/find_candidates_predict.py" \
            -b1 ngs.bam -b2 long.bam -r "${SPLIT_GENOME}" -o "${OUT_DIR}" \
            -b1_depth ngs.bam.depth.npz -b2_depth long.bam.depth.npz \
            --threads "${THREADS}" --model_path "${FILTER_MODEL}"

        python3 "${THPOLISH_DIR}/py/prepeare_predict_input_rust.py" \
            -b1 ngs.bam -b2 long.bam -r "${SPLIT_GENOME}" -o "${OUT_DIR}" \
            --bed_in "${OUT_DIR}/prediction.bed.win" \
            --depth_illu ngs.bam.depth.npz --depth_ont long.bam.depth.npz \
            --threads "${THREADS}"

        "${THPOLISH_BIN}" predict "${PREDICT_MODEL}" \
            "${OUT_DIR}/manifest.json" -t "${THREADS}" \
            -o "${OUT_DIR}/predict_${BASE_NAME}.out"

        "${THPOLISH_BIN}" correct "${OUT_DIR}/predict_${BASE_NAME}.out" \
            -r "${SPLIT_GENOME}" --output-prefix "${OUT_DIR}/final_${BASE_NAME}"

        FASTAS+=("${OUT_DIR}/final_${BASE_NAME}.fasta")
        VCFS+=("${OUT_DIR}/final_${BASE_NAME}.vcf")
    done

    [[ ${#FASTAS[@]} -gt 0 ]] || die "no polished fragments produced"
    cat "${FASTAS[@]}" > thPolish_final.fasta
    grep "^#" "${VCFS[0]}" > thPolish_final.vcf
    for vcf in "${VCFS[@]}"; do
        grep -v "^#" "${vcf}" >> thPolish_final.vcf
    done
    mark_done step3_polish
fi

log "all done: ${OUTDIR}/thPolish_final.fasta (+ thPolish_final.vcf)"
