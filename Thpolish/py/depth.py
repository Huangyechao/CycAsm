#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import argparse
import pysam
import numpy as np
import time
from multiprocessing import Pool

# 直方图统计上限，用于计算中位数
HIST_MAX_DEPTH = 1000 

CLIP_MIN = -1.0
CLIP_MAX = 3.0

def worker_compute_chrom(args):
    bam_path, chrom, length = args
    
    # 1. 尝试打开 BAM 文件
    try:
        samfile = pysam.AlignmentFile(bam_path, "rb")
    except Exception as e:
        return chrom, None, None, str(e)

    # 2. 检查索引是否存在
    if not samfile.has_index():
        samfile.close()
        return chrom, None, None, "No Index (.bai/.csi) found"

    try:
        # 3. 差分数组计算深度
        diff_array = np.zeros(length + 1, dtype=np.int32)
        read_count = 0
        
        for read in samfile.fetch(chrom):
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            
            read_count += 1
            start = read.reference_start
            end = read.reference_end if read.reference_end else (start + read.query_length)
            
            if start < 0: start = 0
            if end > length: end = length
            
            if start < end:
                diff_array[start] += 1
                diff_array[end] -= 1
        
        samfile.close()

        # 4. 如果没有 Reads，直接返回 None，节省内存
        if read_count == 0:
            return chrom, None, None, None

        # 5. 还原深度 (CumSum)
        depth_array = np.cumsum(diff_array, dtype=np.int32)
        final_depth = depth_array[:-1] # 去掉辅助位

        # 6. 计算局部直方图 (用于后续合并计算全局中位数)
        clipped_depth = np.clip(final_depth, 0, HIST_MAX_DEPTH)
        hist = np.bincount(clipped_depth, minlength=HIST_MAX_DEPTH+1)

        # 处理直方图溢出
        if len(hist) > HIST_MAX_DEPTH + 1:
            overflow = np.sum(hist[HIST_MAX_DEPTH+1:])
            hist = hist[:HIST_MAX_DEPTH+1]
            hist[-1] += overflow
            
        return chrom, final_depth, hist, None

    except Exception as e:
        return chrom, None, None, str(e)

def calc_global_stats(global_hist):
    """根据全局直方图计算统计量"""
    total_bases = np.sum(global_hist)
    if total_bases == 0:
        return 0.0, 0.0

    # 计算均值
    depths = np.arange(len(global_hist))
    total_coverage = np.sum(depths * global_hist)
    global_mean = total_coverage / total_bases

    # 计算中位数
    cumsum = np.cumsum(global_hist)
    median_idx = np.searchsorted(cumsum, total_bases / 2)
    global_median = float(median_idx)

    return global_mean, global_median

def main():
    parser = argparse.ArgumentParser(description="Calculate Zero-Centered Normalized Depth")
    parser.add_argument("-b", "--bam", required=True, help="Input BAM file")
    parser.add_argument("-o", "--out", required=True, help="Output .npz file")
    parser.add_argument("-t", "--threads", type=int, default=4)
    parser.add_argument("-l", "--min_len", type=int, default=0)
    parser.add_argument("--chroms", nargs="+", help="Specific chromosomes (optional)")
    
    args = parser.parse_args()
    t0 = time.time()

    # 文件检查
    if not os.path.exists(args.bam):
        sys.exit(f"File not found: {args.bam}")
    
    # 简单的索引文件检查警告
    if not os.path.exists(args.bam + ".bai") and not os.path.exists(args.bam + ".csi"):
        print(f"[Warning] BAM index not found nearby. Fetch might fail.")

    print(f"[Init] Reading header...")
    with pysam.AlignmentFile(args.bam, "rb") as f:
        all_refs = f.references
        all_lens = f.lengths
        ref_map = dict(zip(all_refs, all_lens))

    target_chroms = args.chroms if args.chroms else all_refs
    tasks = []
    for c in target_chroms:
        if c in ref_map and ref_map[c] >= args.min_len:
            tasks.append((args.bam, c, ref_map[c]))

    print(f"[Run] Step 1: Calculating raw depth (Threads: {args.threads})...")
    
    # 暂存原始数据: {chrom: int32_array}
    raw_data_store = {}
    
    # 全局直方图
    global_hist = np.zeros(HIST_MAX_DEPTH + 1, dtype=np.int64)
    total_len = 0
        
    with Pool(processes=args.threads) as pool:
        for chrom, depth_arr, local_hist, err in pool.imap_unordered(worker_compute_chrom, tasks):
            if err:
                print(f"  > [Error] {chrom}: {err}")
                continue
            
            if depth_arr is not None:
                raw_data_store[chrom] = depth_arr
                
                l = min(len(local_hist), len(global_hist))
                global_hist[:l] += local_hist[:l]
                total_len += len(depth_arr)
                
                print(f"  > [Loaded] {chrom:<10} (Mean Raw: {np.mean(depth_arr):.1f}x)")

    # --- Step 2: 全局统计 ---
    print("\n[Stats] Calculating Global Median...")
    g_mean, g_median = calc_global_stats(global_hist)
    
    print(f" Total Bases   : {total_len}")
    print(f" Global Mean   : {g_mean:.4f}")
    print(f" Global Median : {g_median:.4f} (Baseline)")

    if g_median < 1.0:
        print("[Warning] Global median is < 1.0. Defaulting to 1.0 to avoid error.")
        g_median = 1.0

    # --- Step 3: 归一化 (Zero-Centered) ---
    print(f"\n[Run] Step 2: Normalizing (Zero-Centered) & Converting to float16...")
    print(f"  > Formula: (Raw / Median) - 1.0")
    print(f"  > Clip Range: [{CLIP_MIN}, {CLIP_MAX}]")

    chrom_keys = list(raw_data_store.keys())
    for chrom in chrom_keys:
        raw_arr = raw_data_store[chrom]
        
        norm_arr = (raw_arr / g_median) - 1.0
        
        norm_arr = np.clip(norm_arr, CLIP_MIN, CLIP_MAX)
        
        raw_data_store[chrom] = norm_arr.astype(np.float16)
        
        del raw_arr
        
    print(f"  > Normalization done.")

    # --- Step 4: 保存 ---
    meta_info = {
        "global_median": g_median,
        "global_mean": g_mean,
        "normalization": "zero-centered",
        "clip_min": CLIP_MIN,
        "clip_max": CLIP_MAX
    }
    
    print(f"[Save] Writing compressed data to {args.out} ...")
    np.savez_compressed(args.out, **raw_data_store, **meta_info)
    
    print(f"[Finished] Total time: {time.time() - t0:.2f}s")

if __name__ == "__main__":
    main()
