#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pysam
import argparse
import numpy as np
import joblib
import random
from multiprocessing import Pool
from typing import List, Tuple, Set

global_predictor = None
def load_and_parse_model(model_path):
    loaded_obj = joblib.load(model_path)
    return loaded_obj['model'], loaded_obj['best_threshold'], loaded_obj['target_depth']

def cluster_candidates(candidates, merge_dist=20, flank_ext=50, max_win_size=130):
    if not candidates: return []
    candidates = sorted(list(set(candidates)))
    n = len(candidates)
    parent = list(range(n))
    bounds = [(c, c) for c in candidates]

    def find(i):
        path = []
        while parent[i] != i:
            path.append(i)
            i = parent[i]
        for node in path: parent[node] = i
        return i

    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            current_min_i, current_max_i = bounds[root_i]
            current_min_j, current_max_j = bounds[root_j]
            new_min = min(current_min_i, current_min_j)
            new_max = max(current_max_i, current_max_j)
            if (new_max - new_min + 2 * flank_ext) <= max_win_size:
                parent[root_j] = root_i
                bounds[root_i] = (new_min, new_max)
                return True
        return False

    gaps = []
    for i in range(n - 1):
        dist = candidates[i+1] - candidates[i]
        if dist <= merge_dist:
            gaps.append((dist, i))
    gaps.sort(key=lambda x: (x[0], x[1]))
    for _, idx in gaps:
        union(idx, idx+1)

    clusters_map = {}
    for i in range(n):
        root = find(i)
        if root not in clusters_map:
            clusters_map[root] = []
        clusters_map[root].append(candidates[i])

    final_windows = []
    sorted_groups = sorted(clusters_map.values(), key=lambda x: x[0])
    for group in sorted_groups:
        group.sort()
        w_s = max(0, group[0] - flank_ext)
        w_e = group[-1] + flank_ext
        real_s = group[0]
        real_e = group[-1] + 1
        final_windows.append({'range': (w_s, w_e), 'real_range': (real_s, real_e), 'targets': group})

    return final_windows


class DualScanner:
    
    def _calc_max_repeat_length(self, ref_bytes):
        seq_len = len(ref_bytes)
        max_repeat_arr = np.ones(seq_len, dtype=np.float32)
        if seq_len < 2: return max_repeat_arr

        for k in [1, 2, 3]:
            if seq_len <= k: continue
            match_mask = (ref_bytes[:-k] == ref_bytes[k:])
            padded = np.r_[False, match_mask, False]
            changes = padded[:-1] != padded[1:]
            change_indices = np.flatnonzero(changes)
            run_starts = change_indices[::2]
            run_ends = change_indices[1::2]
            for s, e in zip(run_starts, run_ends):
                run_len_in_mask = e - s
                total_repeat_bp = run_len_in_mask + k
                region_slice = slice(s, e + k)
                current_vals = max_repeat_arr[region_slice]
                max_repeat_arr[region_slice] = np.maximum(current_vals, total_repeat_bp)
        return max_repeat_arr

    def _scan_bam_into_matrix(self, samfile, chrom, start, end, min_mq, ref_bytes, downsample_ratio=1.0):
        chunk_len = len(ref_bytes)
        actual_end = start + chunk_len
        data_matrix = np.zeros((chunk_len, 12), dtype=np.float32)
        iter_reads = samfile.fetch(chrom, start, actual_end, multiple_iterators=True)
        
        for read in iter_reads:
            if read.is_unmapped or read.is_secondary or read.is_supplementary or \
               read.is_duplicate or read.mapping_quality < min_mq: continue

            if downsample_ratio < 1.0 and random.random() > downsample_ratio:
                continue
            
            query_seq = read.query_sequence
            query_qual = read.query_qualities
            if not query_seq or not query_qual: continue
            
            query_bytes = np.frombuffer(query_seq.upper().encode('ascii'), dtype='S1')
            ref_pos = read.reference_start
            query_pos = 0
            is_rev = read.is_reverse
            mq = read.mapping_quality
            
            for op, length in read.cigartuples:
                if op in [0, 7, 8]:
                    op_start = ref_pos
                    op_end = ref_pos + length
                    intersect_start = max(start, op_start)
                    intersect_end = min(actual_end, op_end)
                    if intersect_end > intersect_start:
                        idx_s = intersect_start - start
                        idx_e = intersect_end - start
                        q_off = intersect_start - op_start
                        q_s = query_pos + q_off
                        q_e = q_s + (intersect_end - intersect_start)
                        ref_chunk = ref_bytes[idx_s:idx_e]
                        query_chunk = query_bytes[q_s:q_e]
                        qual_chunk = np.array(query_qual[q_s:q_e], dtype=np.int32)
                        matches = (ref_chunk == query_chunk)
                        mismatches = ~matches
                        if is_rev: data_matrix[idx_s:idx_e, 1] += matches
                        else:      data_matrix[idx_s:idx_e, 0] += matches
                        if np.any(mismatches):
                            mm_indices = np.where(mismatches)[0]
                            abs_indices = idx_s + mm_indices
                            if is_rev: data_matrix[abs_indices, 3] += 1
                            else:      data_matrix[abs_indices, 2] += 1
                            q_vals = qual_chunk[mm_indices]
                            data_matrix[abs_indices, 6] += q_vals 
                            data_matrix[abs_indices, 7] += mq
                            data_matrix[abs_indices, 10] += 1
                    ref_pos += length
                    query_pos += length
                elif op == 1: # INS
                    target_pos = ref_pos - 1
                    if start <= target_pos < actual_end:
                        idx = target_pos - start
                        ins_quals = query_qual[query_pos : query_pos + length]
                        avg_bq = sum(ins_quals)/len(ins_quals) if ins_quals else 0
                        if is_rev: data_matrix[idx, 5] += 1
                        else:      data_matrix[idx, 4] += 1
                        data_matrix[idx, 8] += avg_bq
                        data_matrix[idx, 9] += mq
                        data_matrix[idx, 11] += 1
                    query_pos += length
                elif op == 2: # DEL
                    op_start = ref_pos
                    op_end = ref_pos + length
                    intersect_start = max(start, op_start)
                    intersect_end = min(actual_end, op_end)
                    if intersect_end > intersect_start:
                        idx_s = intersect_start - start
                        idx_e = intersect_end - start
                        flank_quals = []
                        if query_pos > 0: flank_quals.append(query_qual[query_pos - 1])
                        if query_pos < len(query_qual): flank_quals.append(query_qual[query_pos])
                        avg_bq = sum(flank_quals)/len(flank_quals) if flank_quals else 0
                        if is_rev: data_matrix[idx_s:idx_e, 5] += 1
                        else:      data_matrix[idx_s:idx_e, 4] += 1
                        data_matrix[idx_s:idx_e, 8] += avg_bq
                        data_matrix[idx_s:idx_e, 9] += mq
                        data_matrix[idx_s:idx_e, 11] += 1
                    ref_pos += length
                elif op == 4: query_pos += length
                elif op == 3: ref_pos += length
                if ref_pos > actual_end + 5: break
        return data_matrix

    def find_candidates_in_block(self, sam_illu, sam_ont, chrom, 
                                 padded_start, padded_end, 
                                 pad_left, pad_right, 
                                 min_mq, ref_seq_str,
                                 ratio_illu=1.0, ratio_ont=1.0):
        
        ref_bytes = np.frombuffer(ref_seq_str.upper().encode('ascii'), dtype='S1')
        mat_illu = self._scan_bam_into_matrix(sam_illu, chrom, padded_start, padded_end, min_mq, ref_bytes, ratio_illu)
        mat_ont = self._scan_bam_into_matrix(sam_ont, chrom, padded_start, padded_end, min_mq, ref_bytes, ratio_ont)
        
        # ==================== 特征计算 ====================
        ill_all_alt = mat_illu[:, 10] + mat_illu[:, 11]
        ont_all_alt = mat_ont[:, 10] + mat_ont[:, 11]
        
        block_len = len(ref_bytes)
        kernel = np.ones(21)
        
        local_sum_i = np.convolve(ill_all_alt, kernel, mode='same')
        local_sum_o = np.convolve(ont_all_alt, kernel, mode='same')
        
        is_gc = ((ref_bytes == b'G') | (ref_bytes == b'C')).astype(np.float32)
        gc_val = np.convolve(is_gc, kernel, mode='same') / np.convolve(np.ones(block_len), kernel, mode='same')
        repeat_len_arr = self._calc_max_repeat_length(ref_bytes)

        slice_s = pad_left
        slice_e = block_len - pad_right
        core_mat_i = mat_illu[slice_s:slice_e]
        core_mat_o = mat_ont[slice_s:slice_e]
        core_local_i = local_sum_i[slice_s:slice_e]
        core_local_o = local_sum_o[slice_s:slice_e]
        core_gc = gc_val[slice_s:slice_e]
        core_rep = repeat_len_arr[slice_s:slice_e]
        
        # ==================== 候选筛选 ====================
        ill_total_reads = core_mat_i[:, 10] + core_mat_i[:, 11]
        ont_total_reads = core_mat_o[:, 10] + core_mat_o[:, 11]
        
        mask = (ill_total_reads >= 2) | (ont_total_reads >= 2)
        
        cand_idx = np.where(mask)[0]
        if len(cand_idx) == 0: return []
        
        abs_pos = padded_start + pad_left + cand_idx
        c_i = core_mat_i[cand_idx]
        c_o = core_mat_o[cand_idx]
        
        results = []
        for k in range(len(cand_idx)):
            depth_i = c_i[k, 0] + c_i[k, 1] + c_i[k, 2] + c_i[k, 3] + c_i[k, 4] + c_i[k, 5]
            depth_o = c_o[k, 0] + c_o[k, 1] + c_o[k, 2] + c_o[k, 3] + c_o[k, 4] + c_o[k, 5]
            
            # --- Ill SNP ---
            snp_tot_i = c_i[k, 10]
            snp_vaf_i = snp_tot_i / (depth_i + 1e-6)
            s_f, s_r = c_i[k, 2], c_i[k, 3]
            snp_sb_i = min(s_f, s_r) / (s_f + s_r + 1e-6)
            snp_mbq_i = c_i[k, 6] / (snp_tot_i + 1e-6)
            snp_mmq_i = c_i[k, 7] / (snp_tot_i + 1e-6)
            
            # --- Ill Indel ---
            ind_tot_i = c_i[k, 11]
            ind_vaf_i = ind_tot_i / (depth_i + 1e-6)
            i_f, i_r = c_i[k, 4], c_i[k, 5]
            ind_sb_i = min(i_f, i_r) / (i_f + i_r + 1e-6)
            ind_mbq_i = c_i[k, 8] / (ind_tot_i + 1e-6)
            ind_mmq_i = c_i[k, 9] / (ind_tot_i + 1e-6)
            
            # --- ONT SNP ---
            snp_tot_o = c_o[k, 10]
            snp_vaf_o = snp_tot_o / (depth_o + 1e-6)
            s_f, s_r = c_o[k, 2], c_o[k, 3]
            snp_sb_o = min(s_f, s_r) / (s_f + s_r + 1e-6)
            snp_mbq_o = c_o[k, 6] / (snp_tot_o + 1e-6)
            snp_mmq_o = c_o[k, 7] / (snp_tot_o + 1e-6)
            
            # --- ONT Indel ---
            ind_tot_o = c_o[k, 11]
            ind_vaf_o = ind_tot_o / (depth_o + 1e-6)
            i_f, i_r = c_o[k, 4], c_o[k, 5]
            ind_sb_o = min(i_f, i_r) / (i_f + i_r + 1e-6)
            ind_mbq_o = c_o[k, 8] / (ind_tot_o + 1e-6)
            ind_mmq_o = c_o[k, 9] / (ind_tot_o + 1e-6)
            
            # --- Env ---
            tot_alt_i = snp_tot_i + ind_tot_i
            tot_alt_o = snp_tot_o + ind_tot_o
            ln_i = (core_local_i[cand_idx[k]] - tot_alt_i) / 20.0
            ln_o = (core_local_o[cand_idx[k]] - tot_alt_o) / 20.0
            
            results.append((
                abs_pos[k],
                depth_i, snp_vaf_i, snp_sb_i, snp_mbq_i, snp_mmq_i,
                ind_vaf_i, ind_sb_i, ind_mbq_i, ind_mmq_i,
                depth_o, snp_vaf_o, snp_sb_o, snp_mbq_o, snp_mmq_o,
                ind_vaf_o, ind_sb_o, ind_mbq_o, ind_mmq_o,
                ln_i, ln_o, core_gc[cand_idx[k]], 
                core_rep[cand_idx[k]]
            ))
            
        return results

def worker_scan_and_predict(args_tuple):
    chrom, start, end, min_mq, bam_illu, bam_ont, ref_path, threshold, r_illu, r_ont = args_tuple
    
    random.seed(42 + start)

    scanner = DualScanner()
    pad_size = 50
    candidates = []
    
    with pysam.AlignmentFile(bam_illu, "rb") as s_i, \
         pysam.AlignmentFile(bam_ont, "rb") as s_o, \
         pysam.FastaFile(ref_path) as fa:
        chrom_len = fa.get_reference_length(chrom)
        padded_start = max(0, start - pad_size)
        padded_end = min(chrom_len, end + pad_size)
        actual_pad_left = start - padded_start
        actual_pad_right = padded_end - end
        ref_seq = fa.fetch(chrom, padded_start, padded_end).upper()
        
        candidates = scanner.find_candidates_in_block(
            s_i, s_o, chrom, padded_start, padded_end, actual_pad_left, actual_pad_right, min_mq, ref_seq, r_illu, r_ont
        )
    
    if not candidates:
        return []

    # 准备数据
    all_data = np.array(candidates, dtype=np.float64)
    positions = all_data[:, 0].astype(np.int64)
    X_features = all_data[:, 1:].astype(np.float32)

    # 预测
    if len(X_features) == 0:
        return []
    
    try:
        y_probs = global_predictor.predict_proba(X_features)[:, 1]
    except Exception:
        y_probs = np.zeros(len(X_features))

    # 过滤
    pass_mask = y_probs >= threshold
    
    valid_pos = positions[pass_mask]
    # valid_probs = y_probs[pass_mask]
        
    return valid_pos

def caculate_downsample_ratio(bam_depth, target_depth=30.0, sample_count=10000):

    with np.load(bam_depth, mmap_mode='r') as data:
        ratio = min(target_depth / data['global_median'], 1.0)
        print(f"{os.path.basename(bam_depth)} 的测序深度为{data['global_median']} target_depth={target_depth} ration: {ratio}")
        return ratio

# ================= Main Function =================

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-b1", "--bam_illumina", required=True, help="?")
    parser.add_argument("-b2", "--bam_ont", required=True, help="?")
    parser.add_argument("-b1_depth", "--bam_illumina_depth", required=True, help="Illumina depth npz file")
    parser.add_argument("-b2_depth", "--bam_ont_depth", required=True, help="ONT depth npz file")
    parser.add_argument("-r", "--ref", required=True, help="?")
    parser.add_argument("-o", "--out_dir", required=True, help="?")
    parser.add_argument("--threads", type=int, default=10, help="?")
    parser.add_argument("--scan_block_size", type=int, default=1000000, help="?")
    parser.add_argument("--min_mq", type=int, default=1, help="?")
    parser.add_argument("--model_path", required=True, help="?")
    parser.add_argument("--merge_dist", type=int, default=20, help="merge_dist (bp)")
    parser.add_argument("--flank_ext", type=int, default=50, help="flank_ext (bp)")
    parser.add_argument("--max_win_size", type=int, default=130, help="max_win_size (bp)")
    
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    model_obj, final_threshold, final_target_depth = load_and_parse_model(args.model_path)
    global global_predictor
    global_predictor = model_obj
    
    ratio_illu = caculate_downsample_ratio(args.bam_illumina_depth, final_target_depth)
    ratio_ont = caculate_downsample_ratio(args.bam_ont_depth, final_target_depth)
    # 仅保留 bed 输出
    out_bed_path = os.path.join(args.out_dir, "prediction.bed.win")

    print(f"[Main] 模型: {args.model_path}")
    print(f"[Main] 线程数: {args.threads}")
    print(f"[Main] 扫描并实时预测中...")

    chrom_sizes = []
    with pysam.FastaFile(args.ref) as f:
        for c in f.references: chrom_sizes.append((c, f.get_reference_length(c)))

    with open(out_bed_path, 'w') as f_bed:
        pool = Pool(processes=args.threads)
        
        for chrom, length in chrom_sizes:
            print(f"\n[Chrom: {chrom}] Length: {length}")
            
            scan_tasks = []
            curr = 0
            while curr < length:
                nxt = min(curr + args.scan_block_size, length)
                scan_tasks.append((chrom, curr, nxt, args.min_mq, args.bam_illumina, 
                    args.bam_ont, args.ref, final_threshold, ratio_illu, ratio_ont))
                curr = nxt
            
            chrom_hits = []
            for block_res in pool.imap_unordered(worker_scan_and_predict, scan_tasks):
                chrom_hits.extend(block_res)

            print(f"  -> Found {len(chrom_hits)} variants (passed threshold)")
            windows = cluster_candidates(chrom_hits, merge_dist=args.merge_dist, flank_ext=args.flank_ext, max_win_size=args.max_win_size)

            for win in windows:
                win_start, win_end = win['range']
                real_start, real_end = win['real_range']
                count_in_win = len(win['targets'])
                f_bed.write(f"{chrom}\t{win_start}\t{win_end}\t{real_start}\t{real_end}\t{count_in_win}\n")

        pool.close()
        pool.join()

    print("[Done] 预测完成。")

if __name__ == "__main__":
    main()
