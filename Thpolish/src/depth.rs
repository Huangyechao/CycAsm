use anyhow::{Context, Result};
use log::info;
use rayon::prelude::*;
use rust_htslib::bam::{self, ext::BamRecordExtensions, Read};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

use crate::option::DepthArgs;

const HIST_MAX_DEPTH: usize = 1000;
const CLIP_MIN: f32 = -1.0;
const CLIP_MAX: f32 = 3.0;

#[derive(Serialize, Deserialize, Debug)]
pub struct DepthMetadata {
    pub global_median: f32,
    pub global_mean: f32,
    pub normalization: String,
    pub clip_min: f32,
    pub clip_max: f32,
}

#[derive(Serialize, Deserialize)]
pub struct DepthArchive {
    pub metadata: DepthMetadata,
    pub depths: HashMap<String, Vec<f32>>,
}

pub fn run_depth(args: DepthArgs) -> Result<()> {
    info!("Depth computation initialized. Target BAM: {}", args.bam);

    let bam_reader = bam::IndexedReader::from_path(&args.bam)
        .context("Failed to open BAM or missing index (.bai/.csi)")?;
    let header = bam_reader.header().to_owned();
    
    let mut target_chroms = Vec::new();
    let ref_names = header.target_names();
    
    for (tid, name_bytes) in ref_names.into_iter().enumerate() {
        let name = String::from_utf8_lossy(name_bytes).to_string();
        let len = header.target_len(tid as u32).unwrap_or(0) as usize;
        
        if len < args.min_len { continue; }
        if let Some(ref allowed) = args.chroms {
            if !allowed.contains(&name) { continue; }
        }
        target_chroms.push((tid as u32, name, len));
    }

    if target_chroms.is_empty() {
        anyhow::bail!("No chromosomes matched the criteria.");
    }

    info!("Calculating raw depth (Threads: {})...", args.thread);
    
    if args.thread > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.thread.min(target_chroms.len()))
            .build_global().ok();
    }

    let results: Vec<_> = target_chroms.into_par_iter().filter_map(|(tid, name, len)| {
        match compute_chrom(&args.bam, tid, len) {
            Ok(Some((depth_arr, local_hist))) => {
                info!("  > [Loaded] {:<10} (Length: {})", name, len);
                Some((name, depth_arr, local_hist))
            },
            Ok(None) => None, 
            Err(e) => {
                panic!("  > [Error] {}: {}", name, e)
            }
        }
    }).collect();

    let mut global_hist = [0u64; HIST_MAX_DEPTH + 1];
    let mut total_len = 0usize;
    let mut raw_data_store = HashMap::new();

    for (name, depth_arr, local_hist) in results {
        for i in 0..=HIST_MAX_DEPTH {
            global_hist[i] += local_hist[i];
        }
        total_len += depth_arr.len();
        raw_data_store.insert(name, depth_arr);
    }

    let (g_mean, mut g_median) = calc_global_stats(&global_hist, total_len);
    info!("\n[Stats] Total Bases   : {}", total_len);
    info!("[Stats] Global Mean   : {:.4}", g_mean);
    info!("[Stats] Global Median : {:.4} (Baseline)", g_median);

    if g_median < 1.0 {
        info!("[Warning] Global median is < 1.0. Defaulting to 1.0.");
        g_median = 1.0;
    }

    info!("\nNormalizing and clamping data...");
    let mut final_depths = HashMap::with_capacity(raw_data_store.len());
    
    for (chrom, raw_arr) in raw_data_store {
        let norm_arr: Vec<f32> = raw_arr.into_par_iter()
            .map(|val| {
                let norm = (val as f32 / g_median) - 1.0;
                norm.clamp(CLIP_MIN, CLIP_MAX)
            })
            .collect();
        
        final_depths.insert(chrom, norm_arr);
    }

    info!("Writing compressed archive to {} ...", args.out);
    let archive = DepthArchive {
        metadata: DepthMetadata {
            global_median: g_median,
            global_mean: g_mean,
            normalization: "zero-centered".to_string(),
            clip_min: CLIP_MIN,
            clip_max: CLIP_MAX,
        },
        depths: final_depths,
    };

    let out_stream = std::io::stdout().lock();
    let mut encoder = zstd::Encoder::new(out_stream, 3)?; 
    bincode::serialize_into(&mut encoder, &archive)
        .context("Failed to serialize depth archive")?;
    let _ = encoder.finish()?;

    info!("Depth normalization complete.");
    Ok(())
}

fn compute_chrom(bam_path: &str, tid: u32, len: usize) -> Result<Option<(Vec<i32>, [u64; HIST_MAX_DEPTH + 1])>> {
    let mut reader = bam::IndexedReader::from_path(Path::new(bam_path))?;
    reader.fetch(tid)?;

    let mut diff_array = vec![0i32; len + 1];
    let mut read_count = 0;
    let mut record = bam::Record::new();

    while let Some(Ok(())) = reader.read(&mut record) {
        if record.is_unmapped() || record.is_secondary() || record.is_supplementary() {
            continue;
        }
        read_count += 1;

        // let start = record.pos() as usize;
        // let end_pos = record.cigar().end_pos();
        // let end = (end_pos as usize).min(len);

        let start = record.reference_start() as usize;
        let end = record.reference_end() as usize;

        if start < end {
            diff_array[start] += 1;
            diff_array[end] -= 1;
        }
    }

    if read_count == 0 {
        return Ok(None);
    }

    let mut hist = [0u64; HIST_MAX_DEPTH + 1];
    let mut current_depth = 0i32;
    for i in 0..len {
        current_depth += diff_array[i];
        diff_array[i] = current_depth;
        
        let bin = (current_depth as usize).min(HIST_MAX_DEPTH);
        hist[bin] += 1;
    }
    
    diff_array.pop();

    Ok(Some((diff_array, hist)))
}

fn calc_global_stats(global_hist: &[u64; HIST_MAX_DEPTH + 1], total_bases: usize) -> (f32, f32) {
    if total_bases == 0 { return (0.0, 0.0); }

    let mut total_coverage = 0u64;
    for (depth, &count) in global_hist.iter().enumerate() {
        total_coverage += depth as u64 * count;
    }
    let mean = total_coverage as f64 / total_bases as f64;

    let target = total_bases as u64 / 2;
    let mut cumsum = 0u64;
    let mut median = 0.0;

    for (depth, &count) in global_hist.iter().enumerate() {
        cumsum += count;
        if cumsum >= target {
            median = depth as f32;
            break;
        }
    }

    (mean as f32, median)
}
