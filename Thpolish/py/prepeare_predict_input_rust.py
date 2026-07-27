#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import pysam
import argparse
import numpy as np
import math
import torch
import shutil
import pyfastx
import json
import random
import xxhash
from multiprocessing import Pool, cpu_count
from safetensors.numpy import save
from typing import Tuple, Dict, Any, List
import zstandard as zstd

# ==========================================
# Part: Multiprocessing Shared Globals
# ==========================================

class GlobalShardManager:
    def __init__(self, out_dir, chrom_to_id, shard_size=2000, shuffle_buffer_factor=5):
        self.out_dir = out_dir
        self.shard_size = shard_size
        self.chrom_to_id = chrom_to_id
        
        self.buffer = []
        self.file_count = 0
        self.total_count = 0
        self.manifest = []

        self.encoder = PileupEncoderFactorized()
        self.compressor = zstd.ZstdCompressor(level=3, threads=1)

    def add_samples(self, samples):
        self.buffer.extend(samples)
        self.total_count += len(samples)
        while len(self.buffer) >= self.shard_size:
            chunk = self.buffer[:self.shard_size]
            self.buffer = self.buffer[self.shard_size:]
            self._save_file(chunk)

    def _save_file(self, chunk):

        def get_fill_value(key):
            base_keys = {'illu_bases', 'ont_bases', 'ref_seq'}
            cigar_keys = {'illu_cigar', 'ont_cigar'}
            strand_keys = {'illu_strand', 'ont_strand'}
            other_keys = {
                'illu_bq', 'illu_mq', 'illu_cr', 'illu_rp', 'illu_dp',
                'ont_bq', 'ont_mq', 'ont_cr', 'ont_rp', 'ont_dp'
            }

            if key in base_keys: 
                return self.encoder.BASE_BATCH_PAD
            if key in cigar_keys: 
                return self.encoder.CIGAR_BATCH_PAD
            if key in strand_keys: 
                return self.encoder.STRAND_PAD_IDX
            if key in other_keys: 
                return self.encoder.DEFAULT_BATCH_PAD
            raise KeyError(f"未定义的填充规则字段: {key}")

        self.file_count += 1
        files_per_dir = 300
        subdir_idx = (self.file_count - 1) // files_per_dir
        subdir_name = f"{subdir_idx:05d}"
        file_dir = os.path.join(self.out_dir, "files", subdir_name)
        os.makedirs(file_dir, exist_ok=True)
        
        filename = f"shard_{self.file_count:05d}.safetensors.zst"
        save_path = os.path.join(file_dir, filename)
        relative_path = os.path.join("files", subdir_name, filename)
        
        B = len(chunk)
        max_w = max(len(item['ref_seq']) for item in chunk)
        compact_batch = {}

        # --- 2. 字段分类定义 ---
        width_sensitive_2d = {
            'illu_bases', 'illu_cigar', 'illu_bq', 'illu_rp',
            'ont_bases', 'ont_cigar', 'ont_bq', 'ont_rp'
        }
        width_sensitive_1d = {'ref_seq', 'illu_dp', 'ont_dp'}
        ragged_fields = {'target_cols', 'targets_ref_pos'}

        # 预填充关键元数据
        compact_batch['seq_len'] = np.array([len(item['ref_seq']) for item in chunk], dtype=np.int16)
        # --- 4. 遍历并执行高性能 Padding ---
        keys = chunk[0].keys()
        for k in keys:
            vals = [item[k] for item in chunk]
            first_val = vals[0]
            # 场景 A: 染色体 ID 转换
            if k == 'chrom':
                compact_batch['chrom_id'] = np.array(
                    [self.chrom_to_id.get(v, -1) for v in vals], dtype=np.int32
                )
                continue

            # 场景 B: 固定形状张量 (window_range 或标量)
            elif k == 'window_range':
                compact_batch[k] = np.array(vals, dtype=np.int32)
                continue
            
            first_val = vals[0]
            dtype = first_val.dtype
            # 场景 C: 宽度敏感的 2D 特征 (Depth, Width)
            if k in width_sensitive_2d:
                fill_val = get_fill_value(k)
                D = first_val.shape[0]
                out = np.full((B, D, max_w), fill_val, dtype=dtype)
                for i, arr in enumerate(vals):
                    curr_w = min(arr.shape[1], max_w)
                    out[i, :, :curr_w] = arr[:, :curr_w]
                compact_batch[k] = out

            # 场景 D: 宽度敏感的 1D 特征 (Width,)
            elif k in width_sensitive_1d:
                fill_val = get_fill_value(k)
                out = np.full((B, max_w), fill_val, dtype=dtype)
                for i, arr in enumerate(vals):
                    curr_w = min(len(arr), max_w)
                    out[i, :curr_w] = arr[:curr_w]
                compact_batch[k] = out

            # 场景 E: 不规则数组 (Ragged)
            elif k in ragged_fields:
                field_max_l = max(len(arr) for arr in vals)
                out = np.full((B, field_max_l), -1, dtype=dtype)
                lengths = np.zeros(B, dtype=np.int32)
                for i, arr in enumerate(vals):
                    l = len(arr)
                    lengths[i] = l
                    if l > 0:
                        out[i, :l] = arr
                compact_batch[k] = out
                if k == 'target_cols':
                    compact_batch['target_len'] = lengths

            # 场景 F: 其他已规则化的特征
            else:
                compact_batch[k] = np.stack(vals)

        # --- 5. 序列化与记录 ---
        metadata = {
            "max_width": str(max_w), 
            "sample_count": str(B),
            "format": "safetensors"
        }
        
        raw_bytes = save(compact_batch, metadata=metadata)
        with open(save_path, 'wb') as f:
            f.write(self.compressor.compress(raw_bytes))

        self.manifest.append({
            "path": relative_path,
            "sample_count": B,
            "max_width": max_w
        })
        
        print(f"[I/O] Shard {self.file_count:05d} saved | Samples: {B} | MaxW: {max_w}")

    def close(self, norm_config):
        """收尾：保存缓冲区剩余的所有样本"""
        if self.buffer:
            self._save_file(self.buffer)
        
        # 保存全局索引文件
        manifest_path = os.path.join(self.out_dir, "manifest.json")
        with open(manifest_path, 'w') as f:
            json.dump({
                "summary": self.total_count,
                "files": self.manifest,
                "norm_config": norm_config,
                "chrom_to_id": self.chrom_to_id,
            }, f, indent=4)
        print(f"\n[Done] Manifest saved to {manifest_path}")

# ==========================================
# Part 1: Pileup Encoder
# ==========================================
class PileupEncoderFactorized:
    def __init__(self, max_depth=50, min_mq=1):
        self.max_depth = max_depth
        self.min_mq = min_mq
        self.base_int_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        self.BASE_DEL_IDX = 5
        self.BASE_GAP_IDX = 6
        self.BASE_BATCH_PAD = 7

        self.platform_int_map = {'ILLUMINA': 0, 'PACBIO': 1, 'NANOPORE': 2}
        self.PLATFORM_PAD_IDX = 3
        self.STRAND_FWD = 0
        self.STRAND_REV = 1
        self.STRAND_PAD_IDX = 2

        self.CIGAR_GAP = 0
        self.CIGAR_MATCH = 1
        self.CIGAR_MISMATCH = 2
        self.CIGAR_DEL = 3
        self.CIGAR_INS = 4
        self.CIGAR_BATCH_PAD = 5
        
        # 0:BQ (Base Qual), 1:MQ (Map Qual), 2:RP (Read Pos), 3:DP (Depth), 4:CR (Clip Ratio)
        self.num_cont_channels = 5 
        self.C_BQ = 0    
        self.C_MQ = 1    
        self.C_RP = 2    
        self.C_DP = 3
        self.C_CR = 4

        self.DEFAULT_PAD = -2
        self.DEFAULT_BATCH_PAD = -3

        self.NORM_CONFIG = {'DEFAULT':  {'bq_max': 60.0, 'mq_max': 60.0}}

    def auto_update_config(self, bam_path, platform_name, longest_ref, max_len, num_reads=100000):
        if not os.path.exists(bam_path): return
        p_name = platform_name.upper()
        print(f"[Auto-Config] Scanning {p_name} ({bam_path}) for normalization stats...")
        try:
            samfile = pysam.AlignmentFile(bam_path, "rb")
        except ValueError: return

        print(f"[Auto-Config] Selected longest reference: {longest_ref} start: {int(max_len/3)} end: {max_len}")

        max_bq_obs = 0
        max_mq_obs = 0
        count = 0
        
        for read in samfile.fetch(longest_ref, int(max_len/3), max_len):
            if count >= num_reads: break
            if read.is_unmapped: continue
            
            mq = read.mapping_quality
            if mq != 255:
                if mq > max_mq_obs: max_mq_obs = mq
            
            q_quals = read.query_qualities
            if q_quals and count >= num_reads/4:
                local_max = max(q_quals)
                if local_max > max_bq_obs: max_bq_obs = local_max
            count += 1
            
        samfile.close()
        
        suggested_bq = math.ceil(max_bq_obs / 5.0) * 5.0
        if suggested_bq < 40.0: suggested_bq = 40.0
        
        suggested_mq = math.ceil(max_mq_obs / 10.0) * 10.0
        if suggested_mq < 60.0: suggested_mq = 60.0 
        
        self.NORM_CONFIG[p_name] = {'bq_max': float(suggested_bq), 'mq_max': float(suggested_mq)}
        print(f"[Auto-Config] {p_name} -> Config(BQ_MAX={suggested_bq}, MQ_MAX={suggested_mq})")

    def _fetch_and_expand_ref(self, fasta, chrom, start, end, col_map, total_width):
        try:
            ref_seq_str = fasta.fetch(chrom, start, end).upper()
        except KeyError:
            return np.full((total_width,), self.BASE_GAP_IDX, dtype=np.int8)
        ref_expanded = np.full((total_width,), self.BASE_GAP_IDX, dtype=np.int8)
        seq_len = len(ref_seq_str)
        for i in range(seq_len):
            abs_pos = start + i
            if abs_pos in col_map:
                col_idx = col_map[abs_pos]
                if col_idx < total_width:
                    base_char = ref_seq_str[i]
                    ref_expanded[col_idx] = self.base_int_map.get(base_char, 4)
        return ref_expanded

    def get_tensors(self, reads_list, chrom, start, end, platform, global_map_tuple, \
        ref_expanded_full, depth_expanded_seq):
        platform_name = platform.upper()
        platform_int = self.platform_int_map[platform_name]
        norm_params = self.NORM_CONFIG.get(platform_name, self.NORM_CONFIG['DEFAULT'])
        
        ins_map, col_map, total_width = global_map_tuple
        raw_bases = np.full((self.max_depth, total_width), self.BASE_BATCH_PAD, dtype=np.int8)
        raw_cigar = np.full((self.max_depth, total_width), self.CIGAR_BATCH_PAD, dtype=np.int8)
        raw_bq = np.full((self.max_depth, total_width), self.DEFAULT_BATCH_PAD, dtype=np.int8)
        rp_dtype = np.float16 if platform_int == 0 else np.float32
        raw_rp = np.full((self.max_depth, total_width), self.DEFAULT_BATCH_PAD, dtype=rp_dtype)

        read_strands = np.full((self.max_depth,), self.STRAND_PAD_IDX, dtype=np.int8)
        read_mqs = np.full((self.max_depth,), self.DEFAULT_BATCH_PAD, dtype=np.int8)
        read_crs = np.full((self.max_depth,), self.DEFAULT_BATCH_PAD, dtype=np.float16) # Clip Ratio

        read_ids = np.zeros((self.max_depth,), dtype=np.uint64)

        for row_idx, read in enumerate(reads_list):
            read_strands[row_idx] = self.STRAND_REV if read.is_reverse else self.STRAND_FWD
            read_mqs[row_idx] = 0 if read.mapping_quality == 255 else read.mapping_quality
            read_ids[row_idx] = xxhash.xxh64(read.query_name).intdigest()

            s_col = col_map[max(read.reference_start, start)]
            e_col = col_map[read.reference_end] if read.reference_end < end else total_width
            raw_bases[row_idx, s_col:e_col] = self.BASE_GAP_IDX
            raw_cigar[row_idx, s_col:e_col] = self.CIGAR_GAP    
            raw_bq[row_idx, s_col:e_col] = self.DEFAULT_PAD  
            raw_rp[row_idx, s_col:e_col] = self.DEFAULT_PAD
            self._fill_read(raw_bases, raw_bq, raw_rp, raw_cigar, read_crs,
                            row_idx, read, col_map, start, end, ins_map, ref_expanded_full)
                            
        return {"bases": raw_bases, "cigar": raw_cigar, "bq": raw_bq, "rp": raw_rp, 
            "strand": read_strands, "mq": read_mqs, "cr": read_crs, "read_ids": read_ids}

    def _filter_reads_list(self, reads, win_start, win_end, targets, platform):

        def _is_contained(read_start, read_end, intervals):
            for c_start, c_end in intervals:
                if c_start > read_start:
                    break
                if c_end >= read_end:
                    return True
            return False

        def has_large_indel(read, max_indel_len=30):
            for op, length in read.cigartuples:
                # CIGAR op: 1=INS, 2=DEL
                if (op == 1 or op == 2) and length > max_indel_len:
                    return True
            return False

        if platform == "NANOPORE":
            loose_max_sc = 0.50
            loose_min_len = 2000
            
            strict_max_sc = 0.10
            strict_sc_len = 400
            
            trust_match = 50
            
        elif platform == "HIFI":
            loose_max_sc = 0.40
            loose_min_len = 1000

            strict_sc_len = 200
            strict_max_sc = 0.05
    
            trust_match = 50
        else: # ILLUMINA
            loose_max_sc = 0.20
            loose_min_len = 50

            strict_max_sc = 0.05
            strict_sc_len = 10

            trust_match = 5

        tgt_min = targets[0]
        tgt_max = targets[-1]

        candidates = []
        high_conf_intervals = []

        total_aln_len = 0
        for read in reads:
            if read.is_unmapped or read.is_duplicate: continue
            if platform != "ILLUMINA" and (read.is_secondary or read.is_supplementary or read.mapping_quality < self.min_mq):
                continue

            target_overlap = max(0, min(read.reference_end, tgt_max) - max(read.reference_start, tgt_min))
            if target_overlap <= 0: continue

            aligned_len = read.query_alignment_length
            if aligned_len < loose_min_len: continue

            sc_len = 0
            total_len = read.infer_read_length()
            cigar = read.cigartuples
            if cigar[0][0] == 5:
                sc_len += cigar[0][1]
                if len(cigar) > 1 and cigar[1][0] == 4: # H + S
                    sc_len += cigar[1][1]
            elif cigar[0][0] == 4:
                sc_len += cigar[0][1]
            
            if cigar[-1][0] == 5: # H
                sc_len += cigar[-1][1]
                if len(cigar) > 1 and cigar[-2][0] == 4: # S + H (S is inner)
                    sc_len += cigar[-2][1]
            elif cigar[-1][0] == 4: # S
                sc_len += cigar[-1][1]

            sc_ratio = sc_len / float(total_len)
            if sc_ratio > loose_max_sc: continue

            # This can lead to over-filtering of chromosome boundaries and loss of heterozygous sites.
            # If our model does not have a heterozygous head, we can enable this filter.
            # if has_large_indel(read): continue 

            win_overlap = max(0, min(read.reference_end, win_end) - max(read.reference_start, win_start))
            total_aln_len += aligned_len
            metrics = {
                'aln_len': aligned_len,
                'sc_ratio': sc_ratio,
                'mq': read.mapping_quality,
                'win_overlap': win_overlap,
            }

            if sc_ratio <= strict_max_sc and sc_len < strict_sc_len:
                t_start = read.reference_start + trust_match
                t_end = read.reference_end - trust_match
                if t_end > t_start:
                    high_conf_intervals.append((t_start, t_end))
                candidates.append((read, metrics, True))
            else:
                candidates.append((read, metrics, False))

        if not candidates:
            return []

        scored_reads = []
        high_conf_intervals.sort(key=lambda x: x[0])
        len_weight_factor = total_aln_len / len(candidates)
        for read, m, is_backbone in candidates:
            if not is_backbone:
                if _is_contained(read.reference_start, read.reference_end, high_conf_intervals):
                    continue 

            score = 0.0
            score += m['win_overlap']
            score -= m['sc_ratio'] * 100.0
            score += m['aln_len'] / len_weight_factor
            score += m['mq']
            scored_reads.append((score, read))

        if len(scored_reads) <= self.max_depth:
            selected = [r for s, r in scored_reads]
        else:
            scored_reads.sort(key=lambda x: x[0], reverse=True)
            selected = [r for s, r in scored_reads[:self.max_depth]]

        selected.sort(key=lambda r: r.reference_start)
        return selected

    def _set_feat(self, bases_t, bq_t, rp_t, cigar_t, r, c, base, bq, rp, ref_val):
        if c >= bases_t.shape[1]: return
        base_int = self.base_int_map.get(base.upper(), 4)
        bases_t[r, c] = base_int
        bq_t[r, c] = bq
        rp_t[r, c] = rp  
        cigar_t[r, c] = (self.CIGAR_INS if ref_val == self.BASE_GAP_IDX else 
                         (self.CIGAR_MATCH if base_int == ref_val else self.CIGAR_MISMATCH))

    def _set_del(self, bases_t, bq_t, rp_t, cigar_t, r, c, rp):
        if c >= bases_t.shape[1]: return
        bases_t[r, c] = self.BASE_DEL_IDX
        bq_t[r, c] = self.DEFAULT_PAD
        rp_t[r, c] = rp
        cigar_t[r, c] = self.CIGAR_DEL

    def _fill_read(self, bases_t, bq_t, rp_t, cigar_t, cr_vec, row_idx, read, 
                   col_map, win_start, win_end, ins_map, ref_expanded_full):
        query_seq = read.query_sequence
        query_qual = read.query_qualities if read.query_qualities else [0] * len(query_seq)
        mq = read.mapping_quality
        strand = -1.0 if read.is_reverse else 1.0
        read_len = len(query_seq)
        if read_len == 0: return

        read_len_inferred = read.infer_read_length()

        ref_cursor = read.reference_start
        read_cursor = 0
        
        clip_left = 0
        clip_right = 0
        h_clip_left = 0
        cigar = read.cigartuples
        if cigar[0][0] == 5: # H
            h_len = cigar[0][1]
            clip_left += h_len
            h_clip_left = h_len
            if len(cigar) > 1 and cigar[1][0] == 4: # H + S
                clip_left += cigar[1][1]
        elif cigar[0][0] == 4: # S
            clip_left += cigar[0][1]
        
        if cigar[-1][0] == 5: # H
            clip_right += cigar[-1][1]
            if len(cigar) > 1 and cigar[-2][0] == 4: # S + H
                clip_right += cigar[-2][1]
        elif cigar[-1][0] == 4: # S
            clip_right += cigar[-1][1]
        cr_vec[row_idx] = (clip_left + clip_right) / float(read_len_inferred)

        for op, length in read.cigartuples:
            if op in [0, 7, 8]:
                for _ in range(length):
                    if win_start <= ref_cursor < win_end:
                        if ref_cursor in col_map:
                            col_idx = col_map[ref_cursor]
                            if read_cursor < read_len:  
                                rp_norm = ((read_cursor + h_clip_left) / float(read_len_inferred)) * 2 - 1
                                if read.is_reverse: rp_norm = -rp_norm
                                ref_val = ref_expanded_full[col_idx]
                                self._set_feat(bases_t, bq_t, rp_t, cigar_t, row_idx, col_idx, 
                                       query_seq[read_cursor], query_qual[read_cursor], rp_norm, ref_val)
                    ref_cursor += 1
                    read_cursor += 1
            elif op == 1: #ins
                target_pos = ref_cursor - 1
                if win_start <= target_pos < win_end and target_pos in col_map:
                    base_col = col_map[target_pos] 
                    max_ins = ins_map[target_pos]
                    if length > max_ins:
                        read_cursor += length
                        continue
                    for i in range(length):
                        rp_norm = ((read_cursor + h_clip_left) / float(read_len_inferred)) * 2 - 1
                        if read.is_reverse: rp_norm = -rp_norm
                        col_idx = base_col + 1 + i
                        ref_val = self.BASE_GAP_IDX
                        self._set_feat(bases_t, bq_t, rp_t, cigar_t, row_idx, col_idx, 
                                       query_seq[read_cursor], query_qual[read_cursor], 
                                       rp_norm, ref_val)
                        read_cursor += 1
                else: read_cursor += length
            elif op == 2: 
                for _ in range(length):
                    if win_start <= ref_cursor < win_end and ref_cursor in col_map:
                        col_idx = col_map[ref_cursor]
                        rp_norm = ((read_cursor + h_clip_left) / float(read_len_inferred)) * 2 - 1
                        if read.is_reverse: rp_norm = -rp_norm
                        self._set_del(bases_t, bq_t, rp_t, cigar_t, row_idx, col_idx, rp_norm)
                    ref_cursor += 1
            elif op == 4: read_cursor += length
            elif op == 3: ref_cursor += length
            if ref_cursor > win_end + 3: break

    def _get_max_insertion_map(self, reads, start_pos, end_pos, max_insert_size):
        ins_map = {pos: 0 for pos in range(start_pos, end_pos)}
        for read in reads:
            ref_cursor = read.reference_start
            for op, length in read.cigartuples:
                if op in [0, 7, 8]: ref_cursor += length
                elif op == 1:
                    if length > max_insert_size: continue
                    target_pos = ref_cursor - 1
                    if start_pos <= target_pos < end_pos:
                        ins_map[target_pos] = max(ins_map[target_pos], length)
                elif op in [2, 3]: ref_cursor += length
        return ins_map

    def _build_column_mapping(self, start_pos, end_pos, ins_map):
        col_map = {}
        num_sites = end_pos - start_pos
        pos_arr = np.empty(num_sites, dtype=np.int32)
        col_arr = np.empty(num_sites, dtype=np.int32)
        
        curr = 0
        idx = 0
        for pos in range(start_pos, end_pos):
            col_map[pos] = curr
            
            # 记录顺序：基因组坐标 -> 起始列索引
            pos_arr[idx] = pos
            col_arr[idx] = curr
            
            curr += (1 + ins_map.get(pos, 0))
            idx += 1
            
        return col_map, curr, pos_arr, col_arr

def _expand_depth_to_columns(depth_ref_array, col_map, total_width, start_pos, padding_value):
    expanded = np.full(total_width, padding_value, dtype=np.float16) #float32
    
    sorted_items = sorted(col_map.items())
    num_items = len(sorted_items)
    
    for i in range(num_items):
        ref_pos, col_idx = sorted_items[i]
        
        # --- 获取深度值 ---
        rel_idx = ref_pos - start_pos
        if 0 <= rel_idx < len(depth_ref_array):
            val = depth_ref_array[rel_idx]
        else:
            val = padding_value
            
        # 起始：当前 Reference 的列索引
        # 结束：下一个 Reference 的列索引（如果还有下一个），否则填到最后
        # 中间的空隙即为 Insertions，自动继承当前的 val
        if i < num_items - 1:
            next_col_idx = sorted_items[i+1][1]
        else:
            next_col_idx = total_width
            
        if col_idx < total_width:
            end_fill = min(next_col_idx, total_width)
            expanded[col_idx : end_fill] = val
            
    return expanded

def scan_matrix_for_candidates(c1, c2, start_col, end_col, encoder, min_support=1, min_freq=0.01):
    mats_c = []
    if c1.shape[1] > 0:
        mats_c.append(c1)
    if c2.shape[1] > 0: 
        mats_c.append(c2)
    
    if not mats_c:
        return []
        
    combined_cigars = np.concatenate(mats_c, axis=0)
    
    max_width = combined_cigars.shape[1]
    s = max(0, start_col)
    e = min(max_width, end_col)
    # print(s, e, start_col, end_col)
    
    if s >= e:
        return []

    cigars_roi = combined_cigars[:, s:e]
    
    # --- 3. 计算有效深度 ---
    valid_mask = cigars_roi != encoder.CIGAR_BATCH_PAD
    col_depth = np.sum(valid_mask, axis=0)
    
    # 忽略无数据列
    active_cols_mask = col_depth > 0
    if not np.any(active_cols_mask):
        return []

    # --- 4. 核心统计逻辑 ---
    
    # 统计所有非 Match(1) 且非 Pad(0) 的 reads
    # 2=Mismatch, 3=Del, 4=Ins
    target_vars = [encoder.CIGAR_MISMATCH, encoder.CIGAR_DEL, encoder.CIGAR_INS]
    is_variant_read = np.isin(cigars_roi, target_vars)

    total_variant_counts = np.sum(is_variant_read, axis=0)

    # 动态阈值
    thresholds = np.maximum(min_support, col_depth * min_freq)

    # 判定
    final_mask = (total_variant_counts >= thresholds) & active_cols_mask
    
    return np.where(final_mask)[0] + s

def encoder_worker(args_tuple):
    try:
        chrom, windows_batch, opts, norm_config, batch_id = args_tuple
        if not windows_batch: return None

        encoder = PileupEncoderFactorized(max_depth=opts.depth, min_mq=opts.min_mq)
        encoder.NORM_CONFIG = norm_config
        chunk_data = []

        illu_depth_ref = None
        ont_depth_ref = None

        try:
            with np.load(opts.depth_illu, mmap_mode='r') as data:
                if chrom in data:
                    illu_depth_ref = data[chrom].astype(np.float32)
        except Exception as e: print(f"[Worker Warning] Illu Depth Load Fail: {e}")

        try:
            with np.load(opts.depth_ont, mmap_mode='r') as data:
                if chrom in data:
                    ont_depth_ref = data[chrom].astype(np.float32)
        except Exception as e: print(f"[Worker Warning] ONT Depth Load Fail: {e}")

        with pysam.AlignmentFile(opts.bam_illumina, "rb") as sam_illu, \
             pysam.AlignmentFile(opts.bam_ont, "rb") as sam_ont, \
             pysam.FastaFile(opts.ref) as fasta_ref:
             
            for item in windows_batch:
                w_s, w_e = item['range']
                targets = item['targets']
                target_len = w_e - w_s
                if illu_depth_ref is not None:
                    slice_data = illu_depth_ref[w_s:w_e]
                    
                    if len(slice_data) == target_len:
                        local_depth_illu_ref = slice_data
                    else:
                        local_depth_illu_ref = np.full(target_len, encoder.DEFAULT_PAD, dtype=np.float32)
                        local_depth_illu_ref[:len(slice_data)] = slice_data
                else:
                    local_depth_illu_ref = np.full(target_len, encoder.DEFAULT_PAD, dtype=np.float32)

                if ont_depth_ref is not None:
                    slice_data = ont_depth_ref[w_s:w_e]
                    if len(slice_data) == target_len:
                        local_depth_ont_ref = slice_data
                    else:
                        local_depth_ont_ref = np.full(target_len, encoder.DEFAULT_PAD, dtype=np.float32)
                        local_depth_ont_ref[:len(slice_data)] = slice_data
                else:
                    local_depth_ont_ref = np.full(target_len, encoder.DEFAULT_PAD, dtype=np.float32)

                raw_reads_illu = list(sam_illu.fetch(chrom, w_s, w_e))
                reads_illu = encoder._filter_reads_list(raw_reads_illu, w_s, w_e, targets, "ILLUMINA")
                raw_reads_ont = list(sam_ont.fetch(chrom, w_s, w_e))
                reads_ont = encoder._filter_reads_list(raw_reads_ont, w_s, w_e, targets, "NANOPORE")
                all_reads_for_map = reads_illu + reads_ont
                if not reads_illu and not reads_ont: continue

                ins_map = encoder._get_max_insertion_map(all_reads_for_map, w_s, w_e, opts.max_insert_size)
                col_map, total_width, pos_arr, col_arr = encoder._build_column_mapping(w_s, w_e, ins_map)
                global_map = (ins_map, col_map, total_width)
                
                ref_expanded = encoder._fetch_and_expand_ref(fasta_ref, chrom, w_s, w_e, col_map, total_width)

                depth_expanded_illu = _expand_depth_to_columns(local_depth_illu_ref, col_map, total_width, w_s, encoder.DEFAULT_PAD)
                depth_expanded_ont = _expand_depth_to_columns(local_depth_ont_ref, col_map, total_width, w_s, encoder.DEFAULT_PAD)

                d1 = encoder.get_tensors(reads_illu, chrom, w_s, w_e, 'ILLUMINA', global_map, 
                    ref_expanded, depth_expanded_illu)
                d2 = encoder.get_tensors(reads_ont, chrom, w_s, w_e, 'NANOPORE', global_map, 
                    ref_expanded, depth_expanded_ont)

                
                if d1['bases'].shape[1] > 0 or d2['bases'].shape[1] > 0:

                    t_start, t_end = item['targets']

                    if t_start in col_map: start_col_idx = col_map[t_start]
                    else: start_col_idx = 0
                    
                    if t_end in col_map: end_col_idx = col_map[t_end]
                    else: end_col_idx = total_width

                    valid_col_idxs = scan_matrix_for_candidates(d1['cigar'], d2['cigar'], start_col_idx, end_col_idx, encoder, min_support=2)
        
                    if len(valid_col_idxs) > 0:
                        idx_in_pos_array = np.searchsorted(col_arr, valid_col_idxs, side='right') - 1
                        idx_in_pos_array = np.clip(idx_in_pos_array, 0, len(pos_arr) - 1)
                        valid_ref_pos = pos_arr[idx_in_pos_array]

                        dp_illu_vec = depth_expanded_illu
                        dp_ont_vec = depth_expanded_ont

                        chunk_data.append({
                            'chrom': chrom,
                            'window_range': (w_s, w_e),
                            'targets_ref_pos': valid_ref_pos.astype(np.int32),
                            'target_cols': np.array(valid_col_idxs, dtype=np.int16),
                            'ref_seq': ref_expanded.astype(np.int8),
                            
                            # Illumina 数据 
                            'illu_bases': d1['bases'], 
                            'illu_cigar': d1['cigar'],
                            'illu_bq': d1['bq'], 
                            'illu_rp': d1['rp'].astype(np.float16),
                            'illu_strand': d1['strand'], 
                            'illu_mq': d1['mq'], 
                            'illu_cr': d1['cr'],
                            'illu_dp': dp_illu_vec.astype(np.float16),
                            
                            # ONT 数据
                            'ont_bases': d2['bases'], 
                            'ont_cigar': d2['cigar'],
                            'ont_bq': d2['bq'], 
                            'ont_rp': d2['rp'],
                            'ont_strand': d2['strand'], 
                            'ont_mq': d2['mq'], 
                            'ont_cr': d2['cr'],
                            'ont_dp': dp_ont_vec.astype(np.float16),
                            'ont_read_ids': d2['read_ids'],
                        })

                    # print_chunk_data(chunk_data[-1], norm_config)
        return chunk_data

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(tb)
        print(e)
        sys.exit(1)

def print_chunk_data(item, norm_config):
    INT_TO_CHAR = {0: 'A', 1: 'C', 2: 'G', 3: 'T', 4: 'N', 5: '-', 6: ' ', 7: '#'}
    CIGAR_TO_CHAR = {0: ' ', 1: '.', 2: 'X', 3: '-', 4: 'I', 5: '#'}

    C_RESET = "\033[0m"
    C_HEADER = "\033[95m" 
    C_REF    = "\033[93m"  # Yellow
    C_ONT    = "\033[96m"  # Cyan
    C_NGS    = "\033[92m"  # Green
    C_RED    = "\033[91m"  # Red
    C_DIM    = "\033[90m"  # Gray

    def expand_item(raw_item, config):
        display_item = raw_item.copy()
        
        for prefix, platform_key in [('illu', 'ILLUMINA'), ('ont', 'NANOPORE')]:
            if f'{prefix}_bases' not in raw_item:
                continue
                
            p_cfg = config.get(platform_key, config.get('DEFAULT', {'bq_max': 60.0, 'mq_max': 60.0}))
            bq_max = p_cfg['bq_max']
            mq_max = p_cfg['mq_max']

            d_raw = raw_item[f'{prefix}_bases']  # (Depth, Width)
            depth_val, width_val = d_raw.shape

            # 1. 处理 Quality (BQ & MQ)
            raw_bq = raw_item[f'{prefix}_bq'].astype(np.float32)
            raw_mq = raw_item[f'{prefix}_mq'].astype(np.float32)
            
            bq_norm = np.clip(raw_bq / bq_max, 0.0, 1.0)
            mq_norm = np.clip(raw_mq / mq_max, 0.0, 1.0)
            
            # 广播 MQ 使其与 BQ 形状一致: (Depth, Width)
            mq_expanded = np.tile(mq_norm[:, np.newaxis], (1, width_val))
            
            qual_3d = np.stack([bq_norm, mq_expanded], axis=-1)
            display_item[f'{prefix}_qual'] = qual_3d.transpose(1, 0, 2)

            display_item[f'{prefix}_bases'] = raw_item[f'{prefix}_bases'].T
            display_item[f'{prefix}_cigar'] = raw_item[f'{prefix}_cigar'].T

            raw_strand = raw_item[f'{prefix}_strand']
            strand_2d = np.tile(raw_strand[:, np.newaxis], (1, width_val))
            display_item[f'{prefix}_strand'] = strand_2d.T  # 结果为 (Width, Depth)

        return display_item

    def render_platform_data(prefix, name, color, bq_max_scale):
        if f'{prefix}_bases' not in item: return
        bases = item[f'{prefix}_bases'].T
        strands = item[f'{prefix}_strand'].T
        cigars = item[f'{prefix}_cigar'].T
        # 这里会把 (width, depth, 2) 转回 (depth, width, 2)
        quals = item[f'{prefix}_qual'].transpose(1, 0, 2)
        depth = bases.shape[0]
        valid_rows = [r for r in range(depth) if not np.all(bases[r] == 6)]

        if not valid_rows:
            print(f"{color}[ {name} ] No Reads{C_RESET}")
            return

        print(f"{color}[ {name} Platform ] ({len(valid_rows)} reads){C_RESET}")
        for r in valid_rows:
            row_str, row_bq_str, row_cig_str = "", "", ""
            mq_norm = np.max(quals[r, :, 1])
            map_q = int(mq_norm * 60.0)
            valid_indices = np.where(bases[r] != 6)[0]
            strand_symbol = ">" if (len(valid_indices) > 0 and strands[r, valid_indices[0]] == 0) else "<"

            for c in range(width):
                val = bases[r, c]
                char = INT_TO_CHAR.get(val, '?')
                if val <= 4: char = char.lower() if strands[r, c] == 1 else char.upper()
                row_str += char

                if val == 6: row_bq_str += " "
                else:
                    # 这里的 bq_max_scale 会还原 Phred 刻度
                    phred = max(0, min(int(quals[r, c, 0] * bq_max_scale), 93))
                    row_bq_str += chr(phred + 33)

                c_val = cigars[r, c]
                c_char = CIGAR_TO_CHAR.get(c_val, '?')
                if c_val in [2, 4]: c_char = f"{C_RED}{c_char}{C_RESET}"
                elif c_val == 1: c_char = f"{C_DIM}{c_char}{C_RESET}"
                row_cig_str += c_char

            print(f"Row_{r:<3} {map_q:<2} {strand_symbol:<3} | {row_str} | {row_cig_str} | {row_bq_str}")

    item = expand_item(item, norm_config)

    chrom = item['chrom']
    start, end = item['window_range']
    targets_ref_pos = item['targets_ref_pos']
    target_cols = item['target_cols'] 
    
    ref_seq_ints = item['ref_seq']
    width = len(ref_seq_ints)
    ref_str = "".join([INT_TO_CHAR.get(val, '?') for val in ref_seq_ints])

    marker_chars = [' '] * width
    for t_col in target_cols:
        if t_col < width: marker_chars[t_col] = '*'
    marker_line = "".join(marker_chars)

    print(f"\n{C_HEADER}{'='*30} Window: {chrom}:{start}-{end} (W:{width}) {'='*30}{C_RESET}")
    print(f"Targets: {targets_ref_pos}")
    print(f"{'Target':>14} | {marker_line}")
    print(f"{C_REF}{'Reference':>14} | {ref_str}{C_RESET}")

    cfg_ont = norm_config.get('NANOPORE', norm_config['DEFAULT'])
    cfg_illu = norm_config.get('ILLUMINA', norm_config['DEFAULT'])

    render_platform_data('ont', 'NANOPORE', C_ONT, cfg_ont['bq_max'])
    render_platform_data('illu', 'ILLUMINA', C_NGS, cfg_illu['bq_max']) 
    print("\n")

def load_windows_from_bed(bed_path):
    windows_by_chrom = {}
    if not os.path.exists(bed_path):
        print(f"[Error] BED file not found: {bed_path}")
        return windows_by_chrom

    with open(bed_path, 'r') as f:
        for line in f:
            if line.startswith("#") or not line.strip(): continue
            parts = line.strip().split('\t')
            if len(parts) < 5: continue
            
            chrom = parts[0]
            w_s = int(parts[1])
            w_e = int(parts[2])
            real_s = int(parts[3])
            real_e = int(parts[4])
                        
            item = {
                'range': (w_s, w_e),
                'targets': (real_s, real_e)
            }
            
            if chrom not in windows_by_chrom:
                windows_by_chrom[chrom] = []
            windows_by_chrom[chrom].append(item)
            
    return windows_by_chrom

def collate_and_save(buffer_list, save_path):
    if not buffer_list:
        return
    torch.save(buffer_list, save_path)

def iter_fasta_len(path):
    for chrom, seq in pyfastx.Fastx(path):
        yield (chrom, len(seq))

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-b1", "--bam_illumina", required=True)
    parser.add_argument("-b2", "--bam_ont", required=True)
    parser.add_argument("-r", "--ref", required=True)
    parser.add_argument("-o", "--out_dir", required=True)
    parser.add_argument("--bed_in", required=True, help="Input BED file with windows (from previous step)")
    parser.add_argument("--depth_illu", required=True, help="Path to Illumina normalized depth .npz")
    parser.add_argument("--depth_ont", required=True, help="Path to ONT normalized depth .npz")
    parser.add_argument("--depth", type=int, default=30, help="NA")
    parser.add_argument("--threads", type=int, default=10, help="NA")
    parser.add_argument("--scan_block_size", type=int, default=5_000_000, help="Chunk size for parallel scanning")
    parser.add_argument("--batch_size", type=int, default=1000, help="Windows per encoding task")
    parser.add_argument("--max_win_size", type=int, default=120, help="NA")
    parser.add_argument("--max_insert_size", type=int, default=15, help="NA")
    parser.add_argument("--min_mq", type=int, default=1, help="NA")
    
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)


    print("[Main] Configuring Normalization...")
    ref_infos = list(iter_fasta_len(args.ref))
    longest_ref, max_len = max(ref_infos, key=lambda x: x[1])
    chrom_to_id = {chrom: idx for idx, (chrom, _) in enumerate(ref_infos)}

    encoder = PileupEncoderFactorized(max_depth=args.depth, min_mq=args.min_mq)
    encoder.auto_update_config(args.bam_illumina, 'ILLUMINA', longest_ref, max_len)
    encoder.auto_update_config(args.bam_ont, 'NANOPORE', longest_ref, max_len)
    norm_config = encoder.NORM_CONFIG
    
    print(f"[Main] Loading windows from {args.bed_in}...")
    windows_map = load_windows_from_bed(args.bed_in)

    pool = Pool(processes=args.threads)
    manager = GlobalShardManager(args.out_dir, chrom_to_id, shard_size=2000)
    for chrom, length in ref_infos:
        windows = windows_map.get(chrom, [])
        if not windows:
            continue      
        print(f"\n[Chrom: {chrom}] Loaded {len(windows)} windows.")

        encoding_tasks = []
        num_batches = math.ceil(len(windows) / args.batch_size)
        
        for i in range(num_batches):
            batch_windows = windows[i * args.batch_size : (i + 1) * args.batch_size]
            encoding_tasks.append((chrom, batch_windows, args, norm_config, i))
            
        print(f"  > Encoding {len(encoding_tasks)} batches in parallel...")
        
        for res_list in pool.imap_unordered(encoder_worker, encoding_tasks):
            if res_list:
                manager.add_samples(res_list)
                
    pool.close()
    pool.join()

    manager.close(norm_config)
    print(f"\n[Main] Finished. Saved {len(manager.manifest)} shards, Total samples: {manager.total_count}")

if __name__ == "__main__":
    main()
