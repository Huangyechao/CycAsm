> **Note:** This tool is currently under active development. Features, interfaces, and results may change without notice. It has currently been tested only on human datasets. Support and performance evaluation for additional species, including non-human animals, plants, bacteria, and fungi, are still in progress.


# ThPolish
A T2T haplotype-aware polishing tool


## Installation

Please install [bwa-mem2](https://github.com/bwa-mem2/bwa-mem2), [minimap2](https://github.com/lh3/minimap2), [samtools](https://github.com/samtools/samtools), [fxTools](https://github.com/moold/fxTools) and [rust](https://rust-lang.org/tools/install/) first.

```bash
#Install Python dependencies
pip3 install pysam numpy joblib xxhash pyfastx safetensors zstandard scikit-learn lightgbm pandas
pip3 install torch==2.10.0+cu126 torchvision==0.25.0+cu126 torchaudio==2.10.0+cu126 --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple
#or using conda/mamba
mamba create -y -n thpolish_env rust pysam numpy joblib xxhash safetensors zstandard scikit-learn lightgbm pytorch=2.10.0 torchvision torchaudio perl perl-ipc-cmd c-compiler gcc_linux-64 gxx_linux-64 clang llvmdev libclang python-xxhash pyfastx

#Build the Rust extension
LIBTORCH_USE_PYTORCH=1 cargo build --release
```

## Run
```bash
set -euo pipefail
shopt -s nullglob

SHORT_READ1="short_read1.fastq.gz"
SHORT_READ2="short_read2.fastq.gz"
LONG_READ="long_read.fastq.gz"
GENOME="genome.fa"
BIN_DIR="PATH_TO_ThPolish"
REAGENT="HD118"

THREADS=64
SPLIT_NUM=30

#1. short read mapping
bwa-mem2 index -p ${GENOME} ${GENOME}
bwa-mem2 mem -c 1000 -a -B 2 -O 4,4 -E 1,1 -t ${THREADS} ${GENOME} ${SHORT_READ1} ${SHORT_READ2} 2> bwa.log |${BIN_DIR}/target/release/thPolish filter -t 3 --condensed-nm --max-mismatch 6 - | samtools sort --write-index -@ ${THREADS} -o ngs.bam

#2. long reads mapping
minimap2 -t ${THREADS} -a -k 16 -w 13 -A 2 -B 4 -O 4,41 -E 2,1 -s 180 -U70,1000000 ${GENOME} ${LONG_READ} 2>minimap2.log|samtools sort --write-index -@ ${THREADS} -o long.bam

#2. calculate mapping depth
python3 ${BIN_DIR}/py/depth.py -b ngs.bam -t ${THREADS} -o ngs.bam.depth
python3 ${BIN_DIR}/py/depth.py -b long.bam -t ${THREADS} -o long.bam.depth

#3. split genome 
fxTools --split ${SPLIT_NUM} ${GENOME}

#4. this is can be run in parallel
GENOME_BASE=$(basename "${GENOME}")
TARGET_DIR="${GENOME_BASE}.split${SPLIT_NUM}"
for SPLIT_GENOME in ${TARGET_DIR}/*fasta; do
    [[ -s "${SPLIT_GENOME}" ]] || continue

    echo $SPLIT_GENOME
    BASE_NAME=$(basename ${SPLIT_GENOME})

    OUT_DIR="predict/out_${BASE_NAME}"
    mkdir -p "${OUT_DIR}"

    python3 ${BIN_DIR}/py/find_candidates_predict.py -b1 ngs.bam -b2 long.bam -r ${SPLIT_GENOME} -o ${OUT_DIR} -b1_depth ngs.bam.depth.npz -b2_depth long.bam.depth.npz --threads ${THREADS} --model_path ${BIN_DIR}/model/${REAGENT}.filter.v1.0.pkl

    python3 ${BIN_DIR}/py/prepeare_predict_input_rust.py -b1 ngs.bam -b2 long.bam -r ${SPLIT_GENOME} -o ${OUT_DIR} --bed_in ${OUT_DIR}/prediction.bed.win --depth_illu ngs.bam.depth.npz --depth_ont long.bam.depth.npz --threads ${THREADS}

    ${BIN_DIR}/target/release/thPolish predict ${BIN_DIR}/model/${REAGENT}.predict.v1.0.pt ${OUT_DIR}/manifest.json -t ${THREADS} -o ${OUT_DIR}/predict_${BASE_NAME}.out

    ${BIN_DIR}/target/release/thPolish correct ${OUT_DIR}/predict_${BASE_NAME}.out -r ${SPLIT_GENOME} --output-prefix ${OUT_DIR}/final_${BASE_NAME}

    FASTAS+=("${OUT_DIR}/final_${BASE_NAME}.fasta")
    VCFS+=("${OUT_DIR}/final_${BASE_NAME}.vcf")
done

#5. merge result
cat "${FASTAS[@]}" > thPolish_final.fasta
grep "^#" "${VCFS[0]}" > thPolish_final.vcf
for vcf in "${VCFS[@]}"; do
    grep -v "^#" "$vcf" >> thPolish_final.vcf
done

```

## Contact
For help, please send an email to hujiang\_at\_genomics\_dot\_cn.
