use anyhow::Result;
use ndarray::{s, Array1, Array2, Array3};
use rayon::prelude::*;
use rust_htslib::bam::{self, Read, Record};
use rust_htslib::faidx;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::sync::{mpsc, Arc};

use crate::{
    depth::DepthArchive,
    option::EncodeArgs
};
// ==========================================
// Part 1: 数据结构定义 (深度文件与编码批次)
// ==========================================

#[derive(Serialize, Deserialize, Clone)]
pub struct EncodedBatch {
    pub chrom: String,
    pub window_range: (i32, i32),
    pub targets_ref_pos: Array1<i32>,
    pub target_cols: Array1<i16>,
    pub ref_seq: Array1<i8>,
    pub seq_len: u16,

    pub illu_bases: Array2<i8>,
    pub illu_cigar: Array2<i8>,
    pub illu_bq: Array2<i8>,
    pub illu_rp: Array2<f32>,
    pub illu_strand: Array1<i8>,
    pub illu_mq: Array1<i8>,
    pub illu_cr: Array1<f32>,
    pub illu_dp: Array1<f32>,

    pub ont_bases: Array2<i8>,
    pub ont_cigar: Array2<i8>,
    pub ont_bq: Array2<i8>,
    pub ont_rp: Array2<f32>,
    pub ont_strand: Array1<i8>,
    pub ont_mq: Array1<i8>,
    pub ont_cr: Array1<f32>,
    pub ont_dp: Array1<f32>,
    pub ont_read_ids: Vec<u64>,
}

#[derive(Serialize, Deserialize)]
pub struct PaddedBatch {
    pub chroms: Vec<String>,
    pub window_range: Vec<(i32, i32)>,      
    pub seq_len: Vec<u16>,
    
    pub targets_ref_pos: Vec<Vec<i32>>,     
    pub target_cols: Vec<Vec<i16>>,          
    pub ref_seq: Array2<i8>,                
    
    // 3D Shape: (Batch, Depth, MaxWidth)
    pub illu_bases: Array3<i8>,
    pub illu_cigar: Array3<i8>,
    pub illu_bq: Array3<i8>,
    pub illu_rp: Array3<f32>,
    pub illu_strand: Array2<i8>, 
    pub illu_mq: Array2<i8>,
    pub illu_cr: Array2<f32>,
    pub illu_dp: Array2<f32>,    

    pub ont_bases: Array3<i8>,
    pub ont_cigar: Array3<i8>,
    pub ont_bq: Array3<i8>,
    pub ont_rp: Array3<f32>,
    pub ont_strand: Array2<i8>,
    pub ont_mq: Array2<i8>,
    pub ont_cr: Array2<f32>,
    pub ont_dp: Array2<f32>,
    pub ont_read_ids: Vec<Vec<u64>>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct NormConfig {
    illu_bq_max: f32,
    illu_mq_max: f32,
    ont_bq_max: f32,
    ont_mq_max: f32,
}

#[derive(Clone, Debug)]
pub struct WindowItem {
    pub chrom: String,
    pub w_s: i32,
    pub w_e: i32,
    pub targets: (i32, i32),
}

#[derive(Serialize)]
struct ManifestRecord {
    path: String,
    sample_count: usize,
    max_width: usize,
}

#[derive(Serialize)]
struct GlobalManifest {
    summary: usize,
    files: Vec<ManifestRecord>,
    norm_config: NormConfig,
}

// ==========================================
// Part 2: 文件读取模块
// ==========================================

fn load_depth_archive(path: &str) -> HashMap<String, Vec<f32>> {
    let file = File::open(path).unwrap_or_else(|_| panic!("Failed to open depth archive: {}", path));
    let decoder = zstd::stream::read::Decoder::new(file).expect("Failed to create zstd decoder");
    let archive: DepthArchive = bincode::deserialize_from(decoder).expect("Failed to deserialize depth archive");
    archive.depths
}

fn load_bed(bed_path: &str) -> Vec<WindowItem> {
    let file = File::open(bed_path).expect("Failed to open BED file");
    let reader = BufReader::new(file);
    let mut windows = Vec::new();
    
    for line in reader.lines().map_while(|l| l.ok()) {
        if line.starts_with('#') || line.trim().is_empty() { continue; }
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() >= 5 {
            windows.push(WindowItem {
                chrom: parts[0].to_string(),
                w_s: parts[1].parse().unwrap(),
                w_e: parts[2].parse().unwrap(),
                targets: (parts[3].parse().unwrap(), parts[4].parse().unwrap()),
            });
        }
    }
    windows
}

// ==========================================
// Part 3: 特征提取与对齐核心逻辑 (PileupEncoder)
// ==========================================

struct PileupEncoder {
    max_depth: usize,
    min_mq: u8,
}

impl PileupEncoder {
    pub const BASE_A: i8 = 0;
    pub const BASE_C: i8 = 1;
    pub const BASE_G: i8 = 2;
    pub const BASE_T: i8 = 3;
    pub const BASE_N: i8 = 4;
    pub const BASE_DEL_IDX: i8 = 5;
    pub const BASE_GAP_IDX: i8 = 6;
    pub const BASE_BATCH_PAD: i8 = 7;

    pub const CIGAR_GAP: i8 = 0;
    pub const CIGAR_MATCH: i8 = 1;
    pub const CIGAR_MISMATCH: i8 = 2;
    pub const CIGAR_DEL: i8 = 3;
    pub const CIGAR_INS: i8 = 4;
    pub const CIGAR_BATCH_PAD: i8 = 5;

    pub const DEFAULT_PAD: i8 = -2;
    pub const DEFAULT_BATCH_PAD: i8 = -3;
    pub const STRAND_FWD: i8 = 0;
    pub const STRAND_REV: i8 = 1;
    pub const STRAND_PAD_IDX: i8 = 2;

    fn new(max_depth: usize, min_mq: u8) -> Self {
        Self { max_depth, min_mq }
    }

    #[inline]
    fn base_to_int(b: u8) -> i8 {
        match b {
            b'A' | b'a' => Self::BASE_A,
            b'C' | b'c' => Self::BASE_C,
            b'G' | b'g' => Self::BASE_G,
            b'T' | b't' => Self::BASE_T,
            _ => Self::BASE_N,
        }
    }

    fn fetch_and_expand_ref(
        &self,
        fasta: &faidx::Reader,
        chrom: &str,
        start: i32,
        end: i32,
        col_map: &HashMap<i32, usize>,
        total_width: usize,
    ) -> Array1<i8> {
        let mut ref_expanded = Array1::from_elem(total_width, Self::BASE_GAP_IDX);
        if let Ok(seq) = fasta.fetch_seq(chrom, start as usize, (end - 1) as usize) {
            for (i, &b) in seq.iter().enumerate() {
                let abs_pos = start + i as i32;
                if let Some(col_idx) = col_map.get(&abs_pos) {
                    if *col_idx < total_width {
                        ref_expanded[*col_idx] = Self::base_to_int(b);
                    }
                }
            }
        }
        ref_expanded
    }

    fn filter_reads(
        &self,
        reads: Vec<Record>,
        win_start: i32,
        win_end: i32,
        targets: (i32, i32),
        platform: &str,
    ) -> Vec<Record> {
        let (loose_max_sc, loose_min_len, strict_max_sc, strict_sc_len, trust_match) = match platform {
            "NANOPORE" => (0.50f64, 2000u32, 0.10f64, 400u32, 50i32),
            "HIFI" => (0.40f64, 1000u32, 0.05f64, 200u32, 50i32),
            _ => (0.20f64, 50u32, 0.05f64, 10u32, 5i32), // ILLUMINA
        };

        let (tgt_min, tgt_max) = targets;
        
        // 临时存储候选读段的数据结构：(Record, aln_len, sc_ratio, win_overlap, is_backbone)
        let mut candidates: Vec<(Record, u32, f64, i32, bool)> = Vec::with_capacity(reads.len());
        let mut high_conf_intervals: Vec<(i32, i32)> = Vec::new();
        let mut total_aln_len: u64 = 0;

        for read in reads {
            if read.is_unmapped() || read.is_duplicate() { continue; }
            if platform != "ILLUMINA" && (read.is_secondary() || read.is_supplementary() || read.mapq() < self.min_mq) {
                continue;
            }

            let start_pos = read.pos() as i32;
            let end_pos = read.cigar().end_pos() as i32;
            
            let target_overlap = std::cmp::max(0, std::cmp::min(end_pos, tgt_max) - std::cmp::max(start_pos, tgt_min));
            if target_overlap <= 0 { continue; }

            let mut aligned_len = 0u32;
            let mut sc_len = 0u32;
            let mut total_len = 0u32; 

            let cigar_view = read.cigar();
            for (i, step) in cigar_view.iter().enumerate() {
                let len = step.len();
                
                // 1. 计算原始读段总长度：仅累加存在于原始 Read 中的碱基操作
                // 包括匹配(M,=,X)、插入(I)以及被裁剪的部分(S,H)
                match step.char() {
                    'M' | 'I' | 'S' | 'H' | '=' | 'X' => total_len += len,
                    _ => {} // 忽略 D, N, P，它们不占用原始 Read 的实际长度
                }

                // 2. 统计比对特征和裁剪特征
                match step.char() {
                    // 修正：对齐 pysam 的 query_alignment_length，排除 D 和 N
                    'M' | '=' | 'X' | 'I' => aligned_len += len, 
                    'S' | 'H' => {
                        if i == 0 || (i == 1 && cigar_view.iter().next().unwrap().char() == 'H') {
                            sc_len += len;
                        } else if i == cigar_view.len() - 1 || (i >= 2 && i == cigar_view.len() - 2 && cigar_view.iter().last().unwrap().char() == 'H') {
                            sc_len += len;
                        }
                    },
                    _ => {}
                }
            }

            if aligned_len < loose_min_len { continue; }
            let sc_ratio = sc_len as f64 / std::cmp::max(total_len, 1) as f64;
            if sc_ratio > loose_max_sc { continue; }

            let win_overlap = std::cmp::max(0, std::cmp::min(end_pos, win_end) - std::cmp::max(start_pos, win_start));
            total_aln_len += aligned_len as u64;

            // 评估置信区间 (Backbone)
            let is_backbone = sc_ratio <= strict_max_sc && sc_len < strict_sc_len;
            if is_backbone {
                let t_start = start_pos + trust_match;
                let t_end = end_pos - trust_match;
                if t_end > t_start {
                    high_conf_intervals.push((t_start, t_end));
                }
            }

            candidates.push((read, aligned_len, sc_ratio, win_overlap, is_backbone));
        }

        if candidates.is_empty() {
            return Vec::new();
        }

        // 必须对区间按照 start_pos 进行排序，以满足区间包含检测的前置条件
        high_conf_intervals.sort_unstable_by_key(|k| k.0);

        let len_weight_factor = total_aln_len as f64 / candidates.len() as f64;
        let mut scored_reads: Vec<(f64, Record)> = Vec::with_capacity(candidates.len());

        for (read, aln_len, sc_ratio, win_overlap, is_backbone) in candidates {
            let start_pos = read.pos() as i32;
            let end_pos = read.cigar().end_pos() as i32;

            if !is_backbone {
                let mut is_contained = false;
                for &(c_start, c_end) in &high_conf_intervals {
                    if c_start > start_pos { break; } // 基于排序数组的提前截断逻辑
                    if c_end >= end_pos {
                        is_contained = true;
                        break;
                    }
                }
                if is_contained { continue; } // 被高置信度区间包含的低置信度序列将被丢弃
            }

            // 执行等价浮点运算
            let mut score: f64 = win_overlap as f64;
            score -= sc_ratio * 100.0;
            score += aln_len as f64 / len_weight_factor;
            score += read.mapq() as f64;

            scored_reads.push((score, read));
        }

        // 按降序排列并截取 max_depth
        scored_reads.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let valid_count = std::cmp::min(scored_reads.len(), self.max_depth);
        
        let mut selected: Vec<Record> = scored_reads.into_iter().take(valid_count).map(|x| x.1).collect();
        selected.sort_by_key(|r| r.pos());
        
        selected
    }

    fn get_max_insertion_map(&self, reads: &[Record], start_pos: i32, end_pos: i32, max_insert_size: u32) -> HashMap<i32, u32> {
        let mut ins_map: HashMap<i32, u32> = HashMap::new();
        for read in reads {
            let mut ref_cursor = read.pos() as i32;
            for step in read.cigar().iter() {
                let len = step.len();
                match step.char() {
                    'M' | '=' | 'X' => ref_cursor += len as i32,
                    'I' => {
                        if len <= max_insert_size {
                            let target_pos = ref_cursor - 1;
                            if target_pos >= start_pos && target_pos < end_pos {
                                let entry = ins_map.entry(target_pos).or_insert(0);
                                *entry = std::cmp::max(*entry, len);
                            }
                        }
                    },
                    'D' | 'N' => ref_cursor += len as i32,
                    _ => {}
                }
            }
        }
        ins_map
    }

    fn build_column_mapping(start_pos: i32, end_pos: i32, ins_map: &HashMap<i32, u32>) -> (HashMap<i32, usize>, usize, Vec<i32>, Vec<i32>) {
        let mut col_map = HashMap::new();
        let num_sites = (end_pos - start_pos) as usize;
        let mut pos_arr = Vec::with_capacity(num_sites);
        let mut col_arr = Vec::with_capacity(num_sites);
        
        let mut curr = 0;
        for pos in start_pos..end_pos {
            col_map.insert(pos, curr);
            pos_arr.push(pos);
            col_arr.push(curr as i32);
            curr += 1 + *ins_map.get(&pos).unwrap_or(&0) as usize;
        }
        (col_map, curr, pos_arr, col_arr)
    }

    fn get_tensors(
        &self,
        reads: &[Record],
        start: i32,
        end: i32,
        col_map: &HashMap<i32, usize>,
        total_width: usize,
        ref_expanded: &Array1<i8>,
        ins_map: &HashMap<i32, u32>
    ) -> (Array2<i8>, Array2<i8>, Array2<i8>, Array2<f32>, Array1<i8>, Array1<i8>, Array1<f32>, Vec<u64>) {
        
        let mut raw_bases = Array2::from_elem((self.max_depth, total_width), Self::BASE_BATCH_PAD);
        let mut raw_cigar = Array2::from_elem((self.max_depth, total_width), Self::CIGAR_BATCH_PAD);
        let mut raw_bq = Array2::from_elem((self.max_depth, total_width), Self::DEFAULT_BATCH_PAD);
        let mut raw_rp = Array2::from_elem((self.max_depth, total_width), Self::DEFAULT_BATCH_PAD as f32);

        let mut read_strands = Array1::from_elem(self.max_depth, Self::STRAND_PAD_IDX);
        let mut read_mqs = Array1::from_elem(self.max_depth, Self::DEFAULT_BATCH_PAD);
        let mut read_crs = Array1::from_elem(self.max_depth, Self::DEFAULT_BATCH_PAD as f32);
        let valid_read_count = std::cmp::min(reads.len(), self.max_depth);
        let mut read_ids = Vec::with_capacity(valid_read_count);

        for (row_idx, read) in reads.iter().enumerate() {
            if row_idx >= self.max_depth { break; }
            
            read_strands[row_idx] = if read.is_reverse() { Self::STRAND_REV } else { Self::STRAND_FWD };
            read_mqs[row_idx] = if read.mapq() == 255 { 0 } else { read.mapq() as i8 };
            read_ids.push(xxhash_rust::xxh64::xxh64(read.qname(), 0));

            let s_col = col_map.get(&std::cmp::max(read.pos() as i32, start)).unwrap_or(&0);
            let e_col = col_map.get(&(read.cigar().end_pos() as i32)).unwrap_or(&total_width);
            
            let e_col_clamp = std::cmp::min(*e_col, total_width);
            for c in *s_col..e_col_clamp {
                raw_bases[[row_idx, c]] = Self::BASE_GAP_IDX;
                raw_cigar[[row_idx, c]] = Self::CIGAR_GAP;
                raw_bq[[row_idx, c]] = Self::DEFAULT_PAD;
                raw_rp[[row_idx, c]] = Self::DEFAULT_PAD as f32;
            }

            self.fill_read_features(
                read, &mut raw_bases, &mut raw_bq, &mut raw_rp, &mut raw_cigar, &mut read_crs,
                row_idx, col_map, start, end, ins_map, ref_expanded
            );
        }

        (raw_bases, raw_cigar, raw_bq, raw_rp, read_strands, read_mqs, read_crs, read_ids)
    }

    fn fill_read_features(
        &self,
        read: &Record,
        bases_t: &mut Array2<i8>, bq_t: &mut Array2<i8>, rp_t: &mut Array2<f32>,
        cigar_t: &mut Array2<i8>, cr_vec: &mut Array1<f32>, row_idx: usize,
        col_map: &HashMap<i32, usize>, win_start: i32, win_end: i32, 
        ins_map: &HashMap<i32, u32>, ref_expanded: &Array1<i8>
    ) {
        let seq = read.seq().as_bytes();
        let qual = read.qual();
        let read_len = seq.len();
        if read_len == 0 { return; }

        let mut read_len_inferred = 0;
        let mut clip_left = 0;
        let mut clip_right = 0;
        let mut h_clip_left = 0;

        let cigar_view = read.cigar();
        for (i, step) in cigar_view.iter().enumerate() {
            let len = step.len();
            
            // 修正：剔除 D 和 N 等非 Query 消耗型操作
            match step.char() {
                'M' | 'I' | 'S' | 'H' | '=' | 'X' => read_len_inferred += len,
                _ => {}
            }

            match step.char() {
                'H' | 'S' => {
                    if i == 0 || (i == 1 && cigar_view.iter().next().unwrap().char() == 'H') {
                        clip_left += len;
                        if step.char() == 'H' { h_clip_left = len; }
                    } else {
                        clip_right += len;
                    }
                },
                _ => {}
            }
        }

        cr_vec[row_idx] = (clip_left + clip_right) as f32 / read_len_inferred.max(1) as f32;

        let mut ref_cursor = read.pos() as i32;
        let mut read_cursor = 0;

        for step in cigar_view.iter() {
            let len = step.len();
            match step.char() {
                'M' | '=' | 'X' => {
                    for _ in 0..len {
                        if ref_cursor >= win_start && ref_cursor < win_end {
                            if let Some(&col_idx) = col_map.get(&ref_cursor) {
                                if read_cursor < read_len {
                                    let mut rp_norm = (read_cursor + h_clip_left as usize) as f32 / read_len_inferred as f32 * 2.0 - 1.0;
                                    if read.is_reverse() { rp_norm = -rp_norm; }
                                    
                                    let base_val = Self::base_to_int(seq[read_cursor]);
                                    let ref_val = ref_expanded[col_idx];
                                    
                                    bases_t[[row_idx, col_idx]] = base_val;
                                    bq_t[[row_idx, col_idx]] = qual[read_cursor] as i8;
                                    rp_t[[row_idx, col_idx]] = rp_norm;
                                    cigar_t[[row_idx, col_idx]] = if ref_val == Self::BASE_GAP_IDX { Self::CIGAR_INS }
                                                                  else if base_val == ref_val { Self::CIGAR_MATCH }
                                                                  else { Self::CIGAR_MISMATCH };
                                }
                            }
                        }
                        ref_cursor += 1; read_cursor += 1;
                    }
                },
                'I' => {
                    let target_pos = ref_cursor - 1;
                    if target_pos >= win_start && target_pos < win_end {
                        if let Some(&base_col) = col_map.get(&target_pos) {
                            let max_ins = *ins_map.get(&target_pos).unwrap_or(&0);
                            if len <= max_ins {
                                for i in 0..len {
                                    let col_idx = base_col + 1 + i as usize;
                                    if col_idx < ref_expanded.len() {
                                        if read_cursor < read_len {
                                            let mut rp_norm = (read_cursor + h_clip_left as usize) as f32 / read_len_inferred as f32 * 2.0 - 1.0;
                                            if read.is_reverse() { rp_norm = -rp_norm; }
                                            bases_t[[row_idx, col_idx]] = Self::base_to_int(seq[read_cursor]);
                                            bq_t[[row_idx, col_idx]] = qual[read_cursor] as i8;
                                            rp_t[[row_idx, col_idx]] = rp_norm;
                                            cigar_t[[row_idx, col_idx]] = Self::CIGAR_INS;
                                        }
                                    }
                                    read_cursor += 1;
                                }
                            } else { read_cursor += len as usize; }
                        } else { read_cursor += len as usize; }
                    } else { read_cursor += len as usize; }
                },
                'D' => {
                    for _ in 0..len {
                        if ref_cursor >= win_start && ref_cursor < win_end {
                            if let Some(&col_idx) = col_map.get(&ref_cursor) {
                                let mut rp_norm = (read_cursor + h_clip_left as usize) as f32 / read_len_inferred as f32 * 2.0 - 1.0;
                                if read.is_reverse() { rp_norm = -rp_norm; }
                                bases_t[[row_idx, col_idx]] = Self::BASE_DEL_IDX;
                                bq_t[[row_idx, col_idx]] = Self::DEFAULT_PAD;
                                rp_t[[row_idx, col_idx]] = rp_norm;
                                cigar_t[[row_idx, col_idx]] = Self::CIGAR_DEL;
                            }
                        }
                        ref_cursor += 1;
                    }
                },
                'S' => read_cursor += len as usize,
                'N' => ref_cursor += len as i32,
                _ => {}
            }
            if ref_cursor > win_end + 3 { break; }
        }
    }
}

// ==========================================
// Part 4: 辅助扫描与展开函数
// ==========================================

fn scan_matrix_for_candidates(
    c1: &Array2<i8>, c2: &Array2<i8>, start_col: usize, end_col: usize, min_support: i32, min_freq: f32
) -> Vec<usize> {
    let max_w_c1 = c1.shape()[1];
    let max_w_c2 = c2.shape()[1];
    let max_w = std::cmp::max(max_w_c1, max_w_c2);
    
    let s = std::cmp::max(0, start_col);
    let e = std::cmp::min(max_w, end_col);
    if s >= e { return vec![]; }

    let width = e - s;
    let mut depth_counts = vec![0i32; width];
    let mut var_counts = vec![0i32; width];
    
    let mut process_mat = |mat: &Array2<i8>| {
        let cols = std::cmp::min(mat.shape()[1], e);
        // 外层循环改为行，保证连续内存访问
        for r in 0..mat.shape()[0] {
            // 内层循环改为列切片访问
            for c in s..cols {
                let val = mat[[r, c]];
                if val != PileupEncoder::CIGAR_BATCH_PAD {
                    depth_counts[c - s] += 1;
                    if val == PileupEncoder::CIGAR_MISMATCH || val == PileupEncoder::CIGAR_DEL || val == PileupEncoder::CIGAR_INS {
                        var_counts[c - s] += 1;
                    }
                }
            }
        }
    };

    process_mat(c1);
    process_mat(c2);

    let mut valid_cols = Vec::new();
    for i in 0..width {
        if depth_counts[i] > 0 {
            let threshold = std::cmp::max(min_support, (depth_counts[i] as f32 * min_freq).ceil() as i32);
            if var_counts[i] >= threshold {
                valid_cols.push(s + i);
            }
        }
    }
    valid_cols
}

fn expand_depth_to_columns(
    depth_ref_slice: &[f32],
    col_map: &HashMap<i32, usize>,
    total_width: usize,
    start_pos: i32,
    padding_value: f32,
) -> Array1<f32> {
    let mut expanded = Array1::from_elem(total_width, padding_value);
    
    let mut sorted_items: Vec<(&i32, &usize)> = col_map.iter().collect();
    sorted_items.sort_by_key(|k| k.0);
    let num_items = sorted_items.len();

    for i in 0..num_items {
        let (ref_pos, col_idx) = sorted_items[i];
        let rel_idx = *ref_pos - start_pos;
        
        let val = if rel_idx >= 0 && (rel_idx as usize) < depth_ref_slice.len() {
            depth_ref_slice[rel_idx as usize]
        } else {
            padding_value
        };

        let next_col_idx = if i < num_items - 1 { *sorted_items[i + 1].1 } else { total_width };

        if *col_idx < total_width {
            let end_fill = std::cmp::min(next_col_idx, total_width);
            for c in *col_idx..end_fill {
                expanded[c] = val;
            }
        }
    }
    expanded
}

// ==========================================
// Part 5: 批次拼接与数据补齐 (Padding & Collation)
// ==========================================

fn pad_and_collate(batch: &[EncodedBatch]) -> PaddedBatch {
    let batch_size = batch.len();
    let max_w = batch.iter().map(|b| b.ref_seq.len()).max().unwrap_or(0);
    let max_depth = batch.first().map(|b| b.illu_bases.shape()[0]).unwrap_or(0);

    // 补齐 ref_seq 数组，使用 BASE_BATCH_PAD (7) 填充空白区域
    let mut padded_ref_seq = Array2::from_elem((batch_size, max_w), PileupEncoder::BASE_BATCH_PAD);

    let mut padded_illu_bases = Array3::from_elem((batch_size, max_depth, max_w), PileupEncoder::BASE_BATCH_PAD);
    let mut padded_illu_cigar = Array3::from_elem((batch_size, max_depth, max_w), PileupEncoder::CIGAR_BATCH_PAD);
    let mut padded_illu_bq = Array3::from_elem((batch_size, max_depth, max_w), PileupEncoder::DEFAULT_BATCH_PAD);
    let mut padded_illu_rp = Array3::from_elem((batch_size, max_depth, max_w), PileupEncoder::DEFAULT_BATCH_PAD as f32);
    
    let mut padded_ont_bases = Array3::from_elem((batch_size, max_depth, max_w), PileupEncoder::BASE_BATCH_PAD);
    let mut padded_ont_cigar = Array3::from_elem((batch_size, max_depth, max_w), PileupEncoder::CIGAR_BATCH_PAD);
    let mut padded_ont_bq = Array3::from_elem((batch_size, max_depth, max_w), PileupEncoder::DEFAULT_BATCH_PAD);
    let mut padded_ont_rp = Array3::from_elem((batch_size, max_depth, max_w), PileupEncoder::DEFAULT_BATCH_PAD as f32);

    // 修正：批次维度 (Width_max - Width_curr) 的填充值必须使用 DEFAULT_BATCH_PAD (-3)，而非 DEFAULT_PAD (-2)
    let mut illu_dp = Array2::from_elem((batch_size, max_w), PileupEncoder::DEFAULT_BATCH_PAD as f32);
    let mut ont_dp = Array2::from_elem((batch_size, max_w), PileupEncoder::DEFAULT_BATCH_PAD as f32);

    let mut illu_strand = Array2::from_elem((batch_size, max_depth), PileupEncoder::STRAND_PAD_IDX);
    let mut illu_mq = Array2::from_elem((batch_size, max_depth), PileupEncoder::DEFAULT_BATCH_PAD);
    let mut illu_cr = Array2::from_elem((batch_size, max_depth), PileupEncoder::DEFAULT_BATCH_PAD as f32);

    let mut ont_strand = Array2::from_elem((batch_size, max_depth), PileupEncoder::STRAND_PAD_IDX);
    let mut ont_mq = Array2::from_elem((batch_size, max_depth), PileupEncoder::DEFAULT_BATCH_PAD);
    let mut ont_cr = Array2::from_elem((batch_size, max_depth), PileupEncoder::DEFAULT_BATCH_PAD as f32);
    let mut ont_read_ids = Vec::with_capacity(batch_size);

    let mut chroms = Vec::with_capacity(batch_size);
    let mut window_ranges = Vec::with_capacity(batch_size);
    let mut seq_lens = Vec::with_capacity(batch_size);
    let mut targets_ref_pos = Vec::with_capacity(batch_size);
    let mut target_cols = Vec::with_capacity(batch_size);

    for (b_idx, item) in batch.iter().enumerate() {
        let w = item.illu_bases.shape()[1];
        chroms.push(item.chrom.clone());
        window_ranges.push(item.window_range);
        seq_lens.push(item.seq_len);
        
        // 提取 Ragged arrays，将 NDArray 转为 std::vec::Vec
        targets_ref_pos.push(item.targets_ref_pos.to_vec());
        target_cols.push(item.target_cols.to_vec());

        padded_ref_seq.slice_mut(s![b_idx, 0..w]).assign(&item.ref_seq);

        padded_illu_bases.slice_mut(s![b_idx, .., 0..w]).assign(&item.illu_bases);
        padded_illu_cigar.slice_mut(s![b_idx, .., 0..w]).assign(&item.illu_cigar);
        padded_illu_bq.slice_mut(s![b_idx, .., 0..w]).assign(&item.illu_bq);
        padded_illu_rp.slice_mut(s![b_idx, .., 0..w]).assign(&item.illu_rp);

        padded_ont_bases.slice_mut(s![b_idx, .., 0..w]).assign(&item.ont_bases);
        padded_ont_cigar.slice_mut(s![b_idx, .., 0..w]).assign(&item.ont_cigar);
        padded_ont_bq.slice_mut(s![b_idx, .., 0..w]).assign(&item.ont_bq);
        padded_ont_rp.slice_mut(s![b_idx, .., 0..w]).assign(&item.ont_rp);

        illu_dp.slice_mut(s![b_idx, 0..w]).assign(&item.illu_dp);
        ont_dp.slice_mut(s![b_idx, 0..w]).assign(&item.ont_dp);

        illu_strand.slice_mut(s![b_idx, ..]).assign(&item.illu_strand);
        illu_mq.slice_mut(s![b_idx, ..]).assign(&item.illu_mq);
        illu_cr.slice_mut(s![b_idx, ..]).assign(&item.illu_cr);

        ont_strand.slice_mut(s![b_idx, ..]).assign(&item.ont_strand);
        ont_mq.slice_mut(s![b_idx, ..]).assign(&item.ont_mq);
        ont_cr.slice_mut(s![b_idx, ..]).assign(&item.ont_cr);
        ont_read_ids.push(item.ont_read_ids.clone());
    }

    PaddedBatch {
        chroms, window_range: window_ranges, seq_len: seq_lens,
        targets_ref_pos, target_cols, ref_seq: padded_ref_seq,
        illu_bases: padded_illu_bases, illu_cigar: padded_illu_cigar,
        illu_bq: padded_illu_bq, illu_rp: padded_illu_rp,
        illu_strand, illu_mq, illu_cr, illu_dp,
        ont_bases: padded_ont_bases, ont_cigar: padded_ont_cigar,
        ont_bq: padded_ont_bq, ont_rp: padded_ont_rp,
        ont_strand, ont_mq, ont_cr, ont_dp, ont_read_ids,
    }
}

pub fn auto_update_config(bam_path: &str, platform: &str, num_reads: usize) -> (f32, f32) {
    let mut bam = bam::IndexedReader::from_path(bam_path).expect("Failed to open BAM");
    
    let mut max_len = 0u64;
    let mut longest_chrom = String::new();
    
    {
        let header_view = bam.header();
        for (tid, name_bytes) in header_view.target_names().into_iter().enumerate() {
            let len = header_view.target_len(tid as u32).unwrap_or(0) as u64;
            if len > max_len {
                max_len = len;
                longest_chrom = String::from_utf8_lossy(name_bytes).to_string();
            }
        }
    }

    let start_pos = max_len / 3;
    
    let _ = bam.fetch((&longest_chrom, start_pos, max_len));

    let mut max_bq = 0;
    let mut max_mq = 0;
    let mut count = 0;

    for r in bam.records() {
        if count >= num_reads { break; }
        if let Ok(read) = r {
            if read.is_unmapped() { continue; }
            let mq = read.mapq();
            if mq != 255 && mq > max_mq { max_mq = mq; }
            
            if count >= num_reads / 4 {
                let qual = read.qual();
                if let Some(&local_max) = qual.iter().max() {
                    if local_max > max_bq { max_bq = local_max; }
                }
            }
            count += 1;
        }
    }

    let mut suggested_bq = (max_bq as f32 / 5.0).ceil() * 5.0;
    if suggested_bq < 40.0 { suggested_bq = 40.0; }
    
    let mut suggested_mq = (max_mq as f32 / 10.0).ceil() * 10.0;
    if suggested_mq < 60.0 { suggested_mq = 60.0; }

    println!("[Auto-Config] {} -> BQ_MAX={}, MQ_MAX={}", platform, suggested_bq, suggested_mq);
    (suggested_bq, suggested_mq)
}

fn save_shard1(buffer: &PaddedBatch, out_dir: &str, file_count: usize) -> String {
    let subdir_idx = (file_count - 1) / 300;
    let subdir_name = format!("{:05}", subdir_idx);
    let file_dir = format!("{}/files/{}", out_dir, subdir_name);
    std::fs::create_dir_all(&file_dir).unwrap();
    
    let filename = format!("shard_{:05}.bin.zst", file_count);
    let full_path = format!("{}/{}", file_dir, filename);
    let rel_path = format!("files/{}/{}", subdir_name, filename);

    let file = File::create(&full_path).unwrap();
    
    let mut encoder = zstd::Encoder::new(file, 3).expect("Failed to create zstd encoder");
    
    bincode::serialize_into(&mut encoder, buffer).expect("Bincode serialization failed");
    
    encoder.finish().expect("Failed to finish zstd encoding");
    
    println!("  [Progress] Saved {} | MaxW: {}", rel_path, buffer.illu_bases.shape()[2]);
    rel_path
}

fn save_shard(buffer: &PaddedBatch, out_dir: &str, file_count: usize) -> String {
    let subdir_idx = (file_count - 1) / 300;
    let subdir_name = format!("{:05}", subdir_idx);
    let file_dir = format!("{}/files/{}", out_dir, subdir_name);
    std::fs::create_dir_all(&file_dir).unwrap();
    
    // 修改 1: 更改文件扩展名以反映实际的数据格式
    let filename = format!("shard_{:05}.json.zst", file_count);
    let full_path = format!("{}/{}", file_dir, filename);
    let rel_path = format!("files/{}/{}", subdir_name, filename);

    let file = File::create(&full_path).unwrap();
    let mut encoder = zstd::Encoder::new(file, 3).expect("Failed to create zstd encoder");
    
    // 修改 2: 使用 serde_json 替代 bincode
    serde_json::to_writer(&mut encoder, buffer).expect("JSON serialization failed");
    
    encoder.finish().expect("Failed to finish zstd encoding");
    
    println!("  [Progress] Saved {} | MaxW: {}", rel_path, buffer.illu_bases.shape()[2]);
    rel_path
}

pub fn run_encode(args: EncodeArgs) -> Result<()> {
    std::fs::create_dir_all(format!("{}/files", args.out_dir)).unwrap();

    rayon::ThreadPoolBuilder::new().num_threads(args.thread).build_global().unwrap();

    println!("[Main] Configuring Normalization...");
    let (illu_bq, illu_mq) = auto_update_config(&args.bam_illumina, "ILLUMINA", 100_000);
    let (ont_bq, ont_mq) = auto_update_config(&args.bam_ont, "NANOPORE", 100_000);
    let norm_config = NormConfig { illu_bq_max: illu_bq, illu_mq_max: illu_mq, ont_bq_max: ont_bq, ont_mq_max: ont_mq };

    // 利用 Arc 加载全基因组深度，跨线程提供无锁只读访问
    println!("[Main] Loading Illumina depth archive from {} ...", args.depth_illu);
    let illu_depths_arc = Arc::new(load_depth_archive(&args.depth_illu));
    
    println!("[Main] Loading ONT depth archive from {} ...", args.depth_ont);
    let ont_depths_arc = Arc::new(load_depth_archive(&args.depth_ont));

    println!("[Main] Loading windows from {}...", args.bed_in);
    let windows = load_bed(&args.bed_in);

    let (tx, rx) = mpsc::sync_channel::<EncodedBatch>(2000);
    let out_dir_clone = args.out_dir.clone();
    let norm_cfg_clone = norm_config.clone();

    // 独立线程：负责落盘保存
    let writer_thread = std::thread::spawn(move || {
        let mut buffer = Vec::new();
        let mut file_count = 0;
        let mut total_samples = 0;
        let mut manifest_records = Vec::new();

        for batch in rx {
            buffer.push(batch);
            if buffer.len() >= 2000 {
                file_count += 1;
                let max_w = buffer.iter().map(|b| b.ref_seq.len()).max().unwrap_or(0);
                let padded_data = pad_and_collate(&buffer);
                let rel_path = save_shard(&padded_data, &out_dir_clone, file_count);
                
                manifest_records.push(ManifestRecord { path: rel_path, sample_count: buffer.len(), max_width: max_w });
                total_samples += buffer.len();
                buffer.clear();
            }
        }
        
        if !buffer.is_empty() {
            file_count += 1;
            let max_w = buffer.iter().map(|b| b.ref_seq.len()).max().unwrap_or(0);
            let padded_data = pad_and_collate(&buffer);
            let rel_path = save_shard(&padded_data, &out_dir_clone, file_count);
            manifest_records.push(ManifestRecord { path: rel_path, sample_count: buffer.len(), max_width: max_w });
            total_samples += buffer.len();
        }

        let manifest = GlobalManifest { summary: total_samples, files: manifest_records, norm_config: norm_cfg_clone };
        let manifest_path = format!("{}/manifest.json", out_dir_clone);
        let mut f = File::create(&manifest_path).unwrap();
        let json_data = serde_json::to_string_pretty(&manifest).unwrap();
        f.write_all(json_data.as_bytes()).unwrap();
        
        println!("\n[Main] Finished. Saved {} shards, Total samples: {}", file_count, total_samples);
    });

    // 并行计算循环
    windows.par_chunks(args.batch_size).for_each_with(tx.clone(), |sender, chunk| {
        let mut bam_illu = bam::IndexedReader::from_path(&args.bam_illumina).unwrap();
        let mut bam_ont = bam::IndexedReader::from_path(&args.bam_ont).unwrap();
        let fasta_ref = faidx::Reader::from_path(&args.ref_fasta).unwrap(); //此处会导致seg.错误
        let encoder = PileupEncoder::new(args.depth, args.min_mq);
        
        for item in chunk {
            let (w_s, w_e) = (item.w_s, item.w_e);

            // 安全截取深度切片
            let global_illu_dp = illu_depths_arc
                .get(&item.chrom)
                .unwrap_or_else(|| panic!("Chromosome {} not found in illu_depths", item.chrom));
            let global_ont_dp = ont_depths_arc
                .get(&item.chrom)
                .unwrap_or_else(|| panic!("Chromosome {} not found in ont_depths", item.chrom));
            
            let local_illu_dp = &global_illu_dp[w_s as usize..w_e as usize];
            let local_ont_dp = &global_ont_dp[w_s as usize..w_e as usize];

            let mut raw_reads_illu = Vec::new();
            if bam_illu.fetch((&item.chrom, w_s as u64, w_e as u64)).is_ok() {
                for r in bam_illu.records() { if let Ok(rec) = r { raw_reads_illu.push(rec); } }
            }
            let mut raw_reads_ont = Vec::new();
            if bam_ont.fetch((&item.chrom, w_s as u64, w_e as u64)).is_ok() {
                for r in bam_ont.records() { if let Ok(rec) = r { raw_reads_ont.push(rec); } }
            }


            let reads_illu = encoder.filter_reads(raw_reads_illu, w_s, w_e, item.targets, "ILLUMINA");
            let reads_ont = encoder.filter_reads(raw_reads_ont, w_s, w_e, item.targets, "NANOPORE");
            if reads_illu.is_empty() && reads_ont.is_empty() { continue; }

            let mut all_reads = Vec::new();
            all_reads.extend_from_slice(&reads_illu);
            all_reads.extend_from_slice(&reads_ont);
            let ins_map = encoder.get_max_insertion_map(&all_reads, w_s, w_e, args.max_insert_size);

            let (col_map, total_width, pos_arr, col_arr) = PileupEncoder::build_column_mapping(w_s, w_e, &ins_map);
            let ref_expanded = encoder.fetch_and_expand_ref(&fasta_ref, &item.chrom, w_s, w_e, &col_map, total_width);
            
            std::io::stdout().flush().unwrap();
            let expanded_illu_dp = expand_depth_to_columns(local_illu_dp, &col_map, total_width, w_s, PileupEncoder::DEFAULT_PAD as f32);
            std::io::stdout().flush().unwrap();
            let expanded_ont_dp = expand_depth_to_columns(local_ont_dp, &col_map, total_width, w_s, PileupEncoder::DEFAULT_PAD as f32);

            std::io::stdout().flush().unwrap();
            let (illu_bases, illu_cigar, illu_bq, illu_rp, illu_strand, illu_mq, illu_cr, _) = 
                encoder.get_tensors(&reads_illu, w_s, w_e, &col_map, total_width, &ref_expanded, &ins_map);
            std::io::stdout().flush().unwrap();
            let (ont_bases, ont_cigar, ont_bq, ont_rp, ont_strand, ont_mq, ont_cr, ont_read_ids) = 
                encoder.get_tensors(&reads_ont, w_s, w_e, &col_map, total_width, &ref_expanded, &ins_map);

            std::io::stdout().flush().unwrap();
            if illu_bases.shape()[1] > 0 || ont_bases.shape()[1] > 0 {
                let start_col_idx = *col_map.get(&item.targets.0).unwrap_or(&0);
                let end_col_idx = *col_map.get(&item.targets.1).unwrap_or(&total_width);
                let valid_col_idxs = scan_matrix_for_candidates(&illu_cigar, &ont_cigar, start_col_idx, end_col_idx, 2, 0.01);

                if !valid_col_idxs.is_empty() {
                    let mut valid_ref_pos = Vec::new();
                    for &c_idx in &valid_col_idxs {
                        let idx_in_pos = col_arr.binary_search(&(c_idx as i32)).unwrap_or_else(|x| x.saturating_sub(1));
                        let idx = std::cmp::min(idx_in_pos, pos_arr.len() - 1);
                        valid_ref_pos.push(pos_arr[idx]);
                    }

                    let batch = EncodedBatch {
                        chrom: item.chrom.clone(), window_range: (w_s, w_e),
                        targets_ref_pos: Array1::from_vec(valid_ref_pos),
                        target_cols: Array1::from_vec(valid_col_idxs.iter().map(|&x| x as i16).collect()),
                        ref_seq: ref_expanded, seq_len: (w_e - w_s) as u16,
                        
                        illu_bases, illu_cigar, illu_bq, illu_rp, illu_strand, illu_mq, illu_cr, 
                        illu_dp: expanded_illu_dp,
                        
                        ont_bases, ont_cigar, ont_bq, ont_rp, ont_strand, ont_mq, ont_cr, ont_read_ids, 
                        ont_dp: expanded_ont_dp,
                    };
                    let _ = sender.send(batch);
                }
            }
        }
    });

    drop(tx);
    writer_thread.join().unwrap();
    Ok(())
}
