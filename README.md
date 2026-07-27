# CycAsm

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Release](https://img.shields.io/badge/release-v1.0.0-blue.svg)](../../releases)

**CycAsm：CycloneSEQ 长读长 + DNBSEQ 短读长的长短混合基因组组装流程**

CycAsm 将长读长的连续性和短读长的准确度结合起来，通过标准化流程产出高质量的个体基因组组装结果。流程分为三个核心步骤：

1. **长读长组装**：使用 CycloneSEQ 长读长搭建基因组骨架（[hifiasm](https://github.com/chhylp123/hifiasm)）；
2. **长短读长比对**：将 DNBSEQ 短读长和 CycloneSEQ 长读长分别比对回组装基因组；
3. **单倍型感知协同抛光**：使用自研工具 **Thpolish**，结合长短两种读长的比对信号，校正局部碱基错误，在保持连续性的同时提高碱基准确性、减少分型错误。

```
CycloneSEQ 长读长 ──► hifiasm ──► 基因组骨架 (GFA)
                                       │ GFA → FASTA
                                       ▼
DNBSEQ 短读长 ──► bwa-mem2 比对 ──┐
                                  ├──► Thpolish ──► 抛光后基因组 (FASTA + VCF)
CycloneSEQ 长读长 ─► minimap2 比对 ┘
```

## 仓库结构

```
CycAsm/
├── cycasm.sh            # 一键流程主脚本（参数解析 / 断点续跑 / 日志）
├── environment.yml      # conda/mamba 环境定义
├── Thpolish/            # 长短读长协同抛光工具（随仓库发布）
│   ├── src/             # Rust 核心（filter / predict / correct）
│   ├── py/              # 候选位点筛选与特征准备脚本
│   └── model/           # 模型文件（体积较大，以 Release 附件形式发布，
│                        #  下载后放入本目录）
├── CHANGELOG.md
├── CITATION.cff
└── LICENSE
```

---

## 目录

- [1. 流程概述](#1-流程概述)
- [2. 环境依赖与安装](#2-环境依赖与安装)
- [3. 输入数据要求](#3-输入数据要求)
- [4. 快速开始（一键运行）](#4-快速开始一键运行)
- [5. 分步详解](#5-分步详解)
- [6. 输出文件说明](#6-输出文件说明)
- [7. 性能参考](#7-性能参考)
- [8. 常见问题](#8-常见问题)
- [9. 引用](#9-引用)
- [10. 联系方式](#10-联系方式)

---

## 1. 流程概述

| 步骤 | 工具 | 输入 | 输出 | 作用 |
|------|------|------|------|------|
| ① 组装 | hifiasm | CycloneSEQ 长读长 (FASTQ) | 组装骨架 (GFA) | 利用长读长搭建高连续性骨架 |
| ② 比对 | bwa-mem2 / minimap2 | 短读长 + 长读长 + 骨架 (FASTA) | ngs.bam / long.bam | 为抛光提供比对信号 |
| ③ 抛光 | Thpolish | 两个 BAM + 骨架 (FASTA) | 抛光基因组 (FASTA + VCF) | 单倍型感知协同校正碱基 |

---

## 2. 环境依赖与安装

### 2.1 依赖总览

| 软件 | 用途 | 获取方式 |
|------|------|----------|
| hifiasm（**> 0.21.0-r686**） | 长读长组装（`--ont` 模式） | <https://github.com/chhylp123/hifiasm> |
| Thpolish | 长短读长协同抛光 | 随本仓库发布（`Thpolish/` 目录） |
| bwa-mem2 | 短读长比对 | <https://github.com/bwa-mem2/bwa-mem2> |
| minimap2 | 长读长比对 | <https://github.com/lh3/minimap2> |
| samtools | BAM 排序/索引 | <https://github.com/samtools/samtools> |
| fxTools | 基因组切分 | <https://github.com/moold/fxTools> |
| rust | 编译 Thpolish | <https://rust-lang.org/tools/install/> |

### 2.2 安装 hifiasm

```bash
git clone https://github.com/chhylp123/hifiasm
cd hifiasm && make
# 编译完成后得到可执行文件 ./hifiasm（仅依赖 g++ 和 zlib）
# 注意：CycAsm 使用 --ont 模式组装 CycloneSEQ 数据，要求 hifiasm 版本 > 0.21.0-r686
```

### 2.3 安装 Thpolish

Thpolish 源码随本仓库发布（`Thpolish/` 目录），在其目录下编译即可：

```bash
# 方式一：使用仓库提供的环境定义一键创建 conda 环境（推荐）
mamba env create -f environment.yml
mamba activate cycasm

# 方式二：手动安装 Python 依赖
pip3 install pysam numpy joblib xxhash pyfastx safetensors zstandard scikit-learn lightgbm pandas
pip3 install torch==2.10.0+cu126 torchvision==0.25.0+cu126 torchaudio==2.10.0+cu126 \
    --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple

# 编译 Rust 扩展（在 Thpolish/ 目录下执行）
cd Thpolish
LIBTORCH_USE_PYTORCH=1 cargo build --release
```

> 模型文件（`S1.filter.v1.0.pkl`、`S1.predict.v1.0.pt`）体积较大，请从本仓库的 [Releases](../../releases) 页面下载后放入 `Thpolish/model/` 目录。

> **注意**：Thpolish 目前处于活跃开发阶段，功能、接口和结果可能变动；现阶段仅在人基因组数据上完成测试，其他物种（动植物、细菌、真菌等）的支持与性能评估仍在进行中。

---

## 3. 输入数据要求

| 数据 | 格式 | 建议 | 说明 |
|------|------|------|------|
| CycloneSEQ 长读长 | `long_read.fastq.gz` | 测序深度 ≥ 30×（人基因组约 100–135 Gb） | 用于搭建基因组骨架，并参与抛光 |
| DNBSEQ 短读长 | `short_read1.fastq.gz` / `short_read2.fastq.gz` | 测序深度 ≥ 30× | 双端 FASTQ，用于校正局部碱基 |

> CIMA 队列参考配置：平均长读长数据量 131.74 Gb，平均测序深度 43.9×，结合 DNBSEQ 短读长完成统一组装。

---

## 4. 快速开始（一键运行）

假设 `hifiasm`、`bwa-mem2`、`minimap2`、`samtools`、`fxTools` 已在 `PATH` 中，且 Thpolish 已按 2.3 节编译完成，一条命令跑完全流程：

```bash
# -l  CycloneSEQ 长读长    -1/-2  DNBSEQ 短读长 R1/R2
# -b  Thpolish 源码目录    -r     测序体系/试剂型号（决定使用的模型）
# -p  组装输出前缀         -o     输出目录      -t  线程数
./cycasm.sh \
    -l long_read.fastq.gz \
    -1 short_read1.fastq.gz \
    -2 short_read2.fastq.gz \
    -b ./Thpolish \
    -r S1 \
    -p sample.asm \
    -o results \
    -t 64
```

脚本内置依赖检查、运行日志（`cycasm.log`）和断点续跑：中断后重新执行同一命令，已完成的步骤会自动跳过；加 `-f` 强制全部重跑。完整参数见 `./cycasm.sh -h`。

如需对单倍型组装结果抛光，加 `--hap hap1` 或 `--hap hap2` 即可。

最终抛光后的基因组为 **`thPolish_final.fasta`**，校正记录为 `thPolish_final.vcf`。

> 需要逐条命令手动执行（例如集群分步提交）时，参见第 5 节分步详解。

---

## 5. 分步详解

> 本节命令中 `BIN_DIR` 指 Thpolish 源码目录（即仓库内 `Thpolish/` 的绝对路径），执行前请先设置：`BIN_DIR=/path/to/CycAsm/Thpolish`。

### Step 1：hifiasm 长读长组装

```bash
hifiasm -o sample.asm --ont -t 64 long_read.fastq.gz 2> hifiasm.log
```

- **`--ont` 为必加参数**：CycloneSEQ 为纳米孔型长读长，须使用 hifiasm 的 ONT 模式（需 hifiasm > 0.21.0-r686，且输入须为 FASTQ 格式）；
- 主输出为 `sample.asm.bp.p_ctg.gfa`（primary contigs），同时生成两套初步分型的 `sample.asm.bp.hap1.p_ctg.gfa` / `sample.asm.bp.hap2.p_ctg.gfa`；
- 首次运行会缓存纠错后的 reads 和 overlap（`*.bin` 文件），重复运行时自动复用；加 `-i` 可强制重做 overlap；
- 默认开启单倍型重复清除（purge duplication）；近交/纯合基因组加 `-l0` 关闭；
- 小基因组加 `-f0` 关闭初始 bloom filter（可省 16 GB 内存）；远大于人基因组的物种建议 `-f38` 甚至 `-f39` 节省 k-mer 计数内存；
- **进阶选项**（按需叠加）：
  - `--dual-scaf`：二倍体自支架，利用两套单倍型互相补洞，提升连续性；
  - `--telo-m CCCTAA`：保留更多人基因组端粒（T2T 场景推荐）；
  - `--ul ul.fq.gz`：整合超长读长，冲击 T2T / 单 Contig 组装。

将 GFA 转为 FASTA（Thpolish 需要 FASTA 输入）：

```bash
awk '/^S/{print ">"$2; print $3}' sample.asm.bp.p_ctg.gfa > genome.fa
```

> 如需对两套单倍型分别抛光，将上述命令中的 `bp.p_ctg.gfa` 替换为 `bp.hap1.p_ctg.gfa` / `bp.hap2.p_ctg.gfa` 分别执行即可。

### Step 2：长短读长比对

**短读长**用 bwa-mem2 比对，经 `thPolish filter` 过滤低质量比对后排序：

```bash
bwa-mem2 index -p genome.fa genome.fa
bwa-mem2 mem -c 1000 -a -B 2 -O 4,4 -E 1,1 -t 64 genome.fa short_read1.fastq.gz short_read2.fastq.gz \
    | thPolish filter -t 3 --condensed-nm --max-mismatch 6 - \
    | samtools sort --write-index -@ 64 -o ngs.bam
```

**长读长**用 minimap2 比对（参数针对 CycloneSEQ 长读长调优）：

```bash
minimap2 -t 64 -a -k 16 -w 13 -A 2 -B 4 -O 4,41 -E 2,1 -s 180 -U70,1000000 genome.fa long_read.fastq.gz \
    | samtools sort --write-index -@ 64 -o long.bam
```

然后计算两个 BAM 的比对深度（生成 `.depth.npz`，供候选位点筛选使用）：

```bash
python3 ${BIN_DIR}/py/depth.py -b ngs.bam  -t 64 -o ngs.bam.depth
python3 ${BIN_DIR}/py/depth.py -b long.bam -t 64 -o long.bam.depth
```

### Step 3：Thpolish 协同抛光

Thpolish 采用"切分-并行-合并"策略：

1. **切分**：`fxTools --split 30 genome.fa` 将基因组切成 30 份（`SPLIT_NUM` 可按集群资源调整）；
2. **并行抛光**：每个片段依次执行
   - `find_candidates_predict.py`：结合长短读长深度信号，用 LightGBM 模型（`${REAGENT}.filter.v1.0.pkl`）筛选候选校正位点；
   - `prepeare_predict_input_rust.py`：准备深度学习模型输入特征；
   - `thPolish predict`：用 PyTorch 模型（`${REAGENT}.predict.v1.0.pt`）预测校正结果；
   - `thPolish correct`：将预测应用到该片段，输出抛光后的 FASTA 和 VCF；
3. **合并**：拼接所有片段的 FASTA / VCF 得到最终结果。

> `REAGENT` 变量需与测序使用的体系/试剂型号一致（CycloneSEQ S1 体系填 `S1`），对应 `${BIN_DIR}/model/` 目录下的模型文件（如 `S1.filter.v1.0.pkl` 和 `S1.predict.v1.0.pt`）。

---

## 6. 输出文件说明

| 文件 | 说明 |
|------|------|
| `sample.asm.bp.p_ctg.gfa` | hifiasm primary 组装骨架（GFA 格式） |
| `sample.asm.bp.hap1/hap2.p_ctg.gfa` | 两套初步分型的单倍型组装 |
| `genome.fa` | 由 GFA 转换的 FASTA，抛光输入 |
| `ngs.bam` / `long.bam` | 短/长读长比对结果（已排序、带索引） |
| `predict/out_*/` | 各切分片段的抛光中间结果 |
| **`thPolish_final.fasta`** | **最终抛光后的基因组序列** |
| `thPolish_final.vcf` | 抛光过程的碱基校正记录 |

---

## 7. 性能参考

- 人基因组（~3.1 Gb，30×+ 长读长）：hifiasm 在 48 线程下数小时内完成组装，内存峰值约 137 GB（HG002，36×）；
- 组装质量：HG002 标准品 S1 体系组装 QV 达 62；CIMA 队列（n=291）Contig N50 中位数 44.92 Mb、QV 中位数 48.85（S0 体系）；
- T2T 场景：叠加超长读长后 HG002 Contig N50 达 135 Mb，仅余 13 个 Gap。

---

## 8. 常见问题

**Q1：真实样本的组装 QV 为什么低于 HG002 标准品？**
真实样本覆盖度不均、杂合度更高、复杂区域比例更大，相同口径下 QV 通常略低于标准品，属正常现象。

**Q2：hifiasm 重复运行很慢？**
确认保留了首次运行生成的 `*.bin` 缓存文件，hifiasm 会自动复用 overlap 结果；若换参数重跑（如仅做 trio/Hi-C 分型），可用 `/dev/null` 作为输入跳过 reads 读取。

**Q3：Thpolish 支持非人物种吗？**
现阶段仅在人基因组数据上完成测试，其他物种支持仍在开发评估中。

**Q4：可以只对单倍型组装结果抛光吗？**
可以。将 Step 1 中提取 FASTA 的 GFA 换成 `bp.hap1.p_ctg.gfa` 或 `bp.hap2.p_ctg.gfa`，后续步骤不变。

**Q5：目标是 T2T 组装，参数怎么调？**
常规样本直接使用默认长短联合方案；若目标是 T2T 或单 Contig 解析，建议在 Step 1 叠加超长读长（`--ul`）、`--dual-scaf` 和 `--telo-m CCCTAA`。

---

## 9. 引用

如果使用本流程，请引用 hifiasm：

> Cheng, H., Concepcion, G.T., Feng, X., Zhang, H., Li H. (2021) Haplotype-resolved de novo assembly using phased assembly graphs with hifiasm. *Nat Methods*, **18**:170-175. <https://doi.org/10.1038/s41592-020-01056-5>

> Cheng, H., Asri, M., Lucas, J., Koren, S., Li, H. (2024) Scalable telomere-to-telomere assembly for diploid and polyploid genomes with double graph. *Nat Methods*, **21**:967-970. <https://doi.org/10.1038/s41592-024-02269-8>

---

## 10. 联系方式

- CycAsm 流程问题：请在项目仓库提交 Issue；
- Thpolish 使用问题：hujiang_at_genomics_dot_cn；
- hifiasm 问题：<https://github.com/chhylp123/hifiasm/issues>。
