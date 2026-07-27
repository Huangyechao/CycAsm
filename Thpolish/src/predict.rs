use anyhow::{anyhow, bail, Context, Result};
use crossbeam_channel::{bounded, Receiver, Sender};
use log::info;
use rustc_hash::FxHashMap as HashMap;
use safetensors::{Dtype, SafeTensors};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;
use tch::{Device, IValue, Kind, Tensor};

use crate::option::PredictArgs;

const BASE_DEL_IDX: i64 = 5;
const BASE_GAP_IDX: i64 = 6;
const BASE_BATCH_PAD: i64 = 7;
const STRAND_PAD_IDX: i64 = 2;
const CIGAR_BATCH_PAD: i64 = 5;
const DEFAULT_BATCH_PAD: f64 = -3.0;

struct InferTask {
    inputs_cpu: Vec<Tensor>,
    meta: BatchMeta,
}

struct WriteTask {
    data: Vec<u8>,
}

struct BatchMeta {
    ref_seq: Tensor,
    target_cols: Tensor,
    target_lengths: Tensor,
    chrom_ids: Tensor,
    ref_pos: Tensor,
    ont_bases: Tensor,
    ont_read_ids: Tensor,
    batch_max_w: i64,
    batch_max_t: i64,
    current_batch_size: i64,
}

#[derive(Deserialize, Debug)]
struct Manifest {
    norm_config: HashMap<String, HashMap<String, f32>>,
    chrom_to_id: HashMap<String, i32>,
    files: Vec<ManifestFile>,
}

#[derive(Deserialize, Debug, Clone)]
struct ManifestFile {
    path: String,
}

#[derive(Serialize)]
struct WindowResult {
    chrom: String,
    ref_seq: Vec<i32>,
    sites: Vec<SiteData>,
}

#[derive(Serialize)]
struct SiteData {
    pos: i32,
    offset: i32,
    p_base: Vec<f32>,
    p_het: f32,
    read_data: Vec<(i64, i32)>,
}

fn get_tensor<'a>(
    st: &'a SafeTensors<'a>,
    name: &str,
    target_kind: Kind,
    device: Device,
) -> Result<Tensor> {
    let tensor_view = st
        .tensor(name)
        .map_err(|e| anyhow!("Failed to load tensor '{}': {:?}", name, e))?;

    let shape: Vec<i64> = tensor_view.shape().iter().map(|&x| x as i64).collect();

    let native_kind = match tensor_view.dtype() {
        Dtype::F32 => Kind::Float,
        Dtype::F64 => Kind::Double,
        Dtype::F16 => Kind::Half,
        Dtype::I8 => Kind::Int8,
        Dtype::U8 => Kind::Uint8,
        Dtype::I16 => Kind::Int16,
        Dtype::I32 => Kind::Int,
        Dtype::I64 | Dtype::U64 => Kind::Int64,
        Dtype::BOOL => Kind::Bool,
        _ => bail!(
            "FATAL: Unsupported Safetensors Dtype {:?} for tensor '{}'",
            tensor_view.dtype(),
            name
        ),
    };

    let t = Tensor::from_data_size(tensor_view.data(), &shape, native_kind)
        .to_device(device)
        .to_kind(target_kind);

    Ok(t)
}

fn build_platform_data(
    st: &SafeTensors,
    prefix: &str,
    device: Device,
    bq_max: f32,
    mq_max: f32,
) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
    let bases = get_tensor(st, &format!("{}_bases", prefix), Kind::Int8, device)?
        .to_kind(Kind::Int64)
        .transpose(1, 2)
        .contiguous();

    let seq_len = bases.size()[1];
    let strand = get_tensor(st, &format!("{}_strand", prefix), Kind::Int8, device)?
        .to_kind(Kind::Int64)
        .unsqueeze(1)
        .expand(&[-1, seq_len, -1], false)
        .contiguous();

    let cigar = get_tensor(st, &format!("{}_cigar", prefix), Kind::Int8, device)?
        .to_kind(Kind::Int64)
        .transpose(1, 2)
        .contiguous();

    let mut bq = get_tensor(st, &format!("{}_bq", prefix), Kind::Int8, device)?.to_kind(Kind::Float);
    let mut mq = get_tensor(st, &format!("{}_mq", prefix), Kind::Int8, device)?.to_kind(Kind::Float);
    let rp = get_tensor(st, &format!("{}_rp", prefix), Kind::Float, device)?;
    let dp = get_tensor(st, &format!("{}_dp", prefix), Kind::Float, device)?;
    let cr = get_tensor(st, &format!("{}_cr", prefix), Kind::Float, device)?;

    let bq_max_t = Tensor::from(bq_max).to_device(device).to_kind(Kind::Float);
    let mq_max_t = Tensor::from(mq_max).to_device(device).to_kind(Kind::Float);

    let bq_mask = bq.ge(0.0);
    let bq_valid = bq.masked_select(&bq_mask);
    let bq_norm = bq_valid / bq_max_t;
    let _ = bq.masked_scatter_(&bq_mask, &bq_norm);

    let mq_mask = mq.ge(0.0);
    let mq_valid = mq.masked_select(&mq_mask);
    let mq_norm = mq_valid / mq_max_t;
    let _ = mq.masked_scatter_(&mq_mask, &mq_norm);

    let mq_expanded = mq.unsqueeze(-1).expand(&[-1, -1, seq_len], false);
    let dp_expanded = dp.unsqueeze(1).expand(&[-1, bq.size()[1], -1], false);
    let cr_expanded = cr.unsqueeze(-1).expand(&[-1, -1, seq_len], false);

    let feats = Tensor::stack(&[bq, mq_expanded, rp, dp_expanded, cr_expanded], -1);
    let feats_final = feats.permute(&[0, 2, 1, 3]).contiguous();

    let mask = bases.ne(BASE_BATCH_PAD);

    Ok((bases, strand, feats_final, cigar, mask))
}

#[inline]
fn sigmoid_scalar(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

#[inline]
fn softmax_stats(row: &[f32]) -> (usize, f32, f32) {
    debug_assert!(!row.is_empty());

    let mut pred_idx = 0usize;
    let mut max_logit = row[0];

    for (i, &v) in row.iter().enumerate().skip(1) {
        if v > max_logit {
            max_logit = v;
            pred_idx = i;
        }
    }

    let mut sum_exp = 0.0f32;
    for &v in row {
        sum_exp += (v - max_logit).exp();
    }
    let max_prob = 1.0f32 / sum_exp.max(f32::MIN_POSITIVE);

    (pred_idx, max_prob, max_logit)
}

#[inline]
fn softmax_probs(row: &[f32], max_logit: f32) -> Vec<f32> {
    let mut exp_vals = Vec::with_capacity(row.len());
    let mut sum_exp = 0.0f32;

    for &v in row {
        let e = (v - max_logit).exp();
        sum_exp += e;
        exp_vals.push(e);
    }

    let denom = sum_exp.max(f32::MIN_POSITIVE);
    for v in &mut exp_vals {
        *v /= denom;
    }
    exp_vals
}

fn process_batch_to_msgpack(
    logits_het: &Tensor,
    logits_base: &Tensor,
    meta: BatchMeta,
    id_to_chrom: &[String],
    ref_threshold: f32,
    het_threshold: f32,
) -> Result<Vec<u8>> {
    let logits_base_contig = logits_base.contiguous();
    let logits_het_contig = logits_het.contiguous();

    let probs_base_slice: &[f32] = unsafe {
        std::slice::from_raw_parts(
            logits_base_contig.data_ptr() as *const f32,
            logits_base_contig.numel() as usize,
        )
    };
    let logits_het_slice: &[f32] = unsafe {
        std::slice::from_raw_parts(
            logits_het_contig.data_ptr() as *const f32,
            logits_het_contig.numel() as usize,
        )
    };

    let vec_ref_seq: &[i64] = unsafe {
        std::slice::from_raw_parts(meta.ref_seq.data_ptr() as *const i64, meta.ref_seq.numel() as usize)
    };
    let vec_target_cols: &[i64] = unsafe {
        std::slice::from_raw_parts(
            meta.target_cols.data_ptr() as *const i64,
            meta.target_cols.numel() as usize,
        )
    };
    let vec_target_len: &[i64] = unsafe {
        std::slice::from_raw_parts(
            meta.target_lengths.data_ptr() as *const i64,
            meta.target_lengths.numel() as usize,
        )
    };
    let vec_chrom_ids: &[i32] = unsafe {
        std::slice::from_raw_parts(
            meta.chrom_ids.data_ptr() as *const i32,
            meta.chrom_ids.numel() as usize,
        )
    };
    let vec_ref_pos: &[i32] = unsafe {
        std::slice::from_raw_parts(meta.ref_pos.data_ptr() as *const i32, meta.ref_pos.numel() as usize)
    };
    let vec_ont_bases: &[i64] = unsafe {
        std::slice::from_raw_parts(meta.ont_bases.data_ptr() as *const i64, meta.ont_bases.numel() as usize)
    };
    let vec_ont_ids: &[i64] = unsafe {
        std::slice::from_raw_parts(
            meta.ont_read_ids.data_ptr() as *const i64,
            meta.ont_read_ids.numel() as usize,
        )
    };

    let base_dim = *logits_base_contig.size().last().unwrap_or(&0) as usize;
    if base_dim == 0 {
        bail!("Invalid logits_base last dimension: 0");
    }

    let max_w = meta.batch_max_w as usize;
    let ont_depth = meta.ont_read_ids.size()[1] as usize;
    let t_stride = meta.batch_max_t as usize;
    let current_batch_size = meta.current_batch_size as usize;

    let mut local_buffer = Vec::with_capacity(current_batch_size * 256);
    let mut cursor: usize = 0;

    for i in 0..current_batch_size {
        let num_t = vec_target_len[i] as usize;
        if num_t == 0 {
            continue;
        }

        let current_chrom = id_to_chrom
            .get(vec_chrom_ids[i] as usize)
            .cloned()
            .unwrap_or_default();

        let ref_offset = i * max_w;
        let t_offset = i * t_stride;

        let mut max_col = 0usize;
        let mut sites = Vec::with_capacity(num_t);

        for t_idx in 0..num_t {
            let global_idx = cursor + t_idx;
            let col_idx = vec_target_cols[t_offset + t_idx] as usize;
            max_col = max_col.max(col_idx);

            let row_start = global_idx * base_dim;
            let row_end = row_start + base_dim;
            let base_row = &probs_base_slice[row_start..row_end];

            let (pred_idx, max_p, max_logit) = softmax_stats(base_row);
            let p_het = sigmoid_scalar(logits_het_slice[global_idx]);

            let ref_idx_raw = vec_ref_seq[ref_offset + col_idx];
            let ref_idx = if ref_idx_raw == BASE_GAP_IDX {
                BASE_DEL_IDX as usize
            } else {
                ref_idx_raw as usize
            };

            if pred_idx == ref_idx && max_p >= ref_threshold && p_het < het_threshold {
                continue;
            }

            let p_base = softmax_probs(base_row, max_logit);

            let mut reads = Vec::with_capacity(ont_depth);
            let ont_base_offset = i * (max_w * ont_depth) + col_idx * ont_depth;
            let ont_id_offset = i * ont_depth;

            for d in 0..ont_depth {
                let rid = vec_ont_ids[ont_id_offset + d];
                if rid == 0 {
                    continue;
                }
                let base_val = vec_ont_bases[ont_base_offset + d];
                if base_val != BASE_BATCH_PAD {
                    let adj_base = if base_val == BASE_GAP_IDX {
                        BASE_DEL_IDX
                    } else {
                        base_val
                    };
                    reads.push((rid, adj_base as i32));
                }
            }

            sites.push(SiteData {
                pos: vec_ref_pos[t_offset + t_idx],
                offset: col_idx as i32,
                p_base,
                p_het,
                read_data: reads,
            });
        }

        if !sites.is_empty() {
            let sample_ref_seq: Vec<i32> = vec_ref_seq[ref_offset..ref_offset + max_col + 1]
                .iter()
                .map(|&x| {
                    let adj = if x == BASE_GAP_IDX { BASE_DEL_IDX } else { x };
                    adj as i32
                })
                .collect();

            let window = WindowResult {
                chrom: current_chrom,
                ref_seq: sample_ref_seq,
                sites,
            };

            rmp_serde::encode::write(&mut local_buffer, &window)?;
        }

        cursor += num_t;
    }

    Ok(local_buffer)
}

pub fn run_predict(args: PredictArgs) -> Result<()> {
    let device = Device::cuda_if_available();
    info!("计算设备: {:?}", device);

    let infer_workers = if matches!(device, Device::Cpu) {
        let workers = args.infer_workers;
        if workers > 1 {
            tch::set_num_threads(1);
            tch::set_num_interop_threads(1);
            info!("CPU 模式：启动 {} 个推理线程。", workers);
        } else {
            info!("CPU 模式：单工作线程。采用 PyTorch 默认底层并发策略。");
        }
        workers
    } else {
        if args.infer_workers > 1 {
            info!(
                "检测到 CUDA 设备。为保证计算一致性并防止 VRAM OOM，推理线程数从 {} 强制截断为 1。",
                args.infer_workers
            );
        }
        1
    };

    let manifest_file = File::open(&args.manifest).context("Failed open manifest")?;
    let manifest: Manifest = serde_json::from_reader(BufReader::new(manifest_file))?;

    let max_chrom_id = manifest
        .chrom_to_id
        .values()
        .copied()
        .filter(|&id| id >= 0)
        .max()
        .unwrap_or(0) as usize;
    let mut id_to_chrom: Vec<String> = vec![String::new(); max_chrom_id + 1];
    for (chrom, id) in &manifest.chrom_to_id {
        if *id >= 0 {
            let idx = *id as usize;
            if idx < id_to_chrom.len() {
                id_to_chrom[idx] = chrom.clone();
            }
        }
    }

    let default_cfg = manifest.norm_config.get("DEFAULT").unwrap();
    let illu_cfg = manifest.norm_config.get("ILLUMINA").unwrap_or(default_cfg);
    let ont_cfg = manifest.norm_config.get("NANOPORE").unwrap_or(default_cfg);

    let manifest_dir = Path::new(&args.manifest)
        .parent()
        .unwrap_or(Path::new(""));

    let (tx_file, rx_file): (Sender<ManifestFile>, Receiver<ManifestFile>) =
        bounded(2 * args.infer_workers);
    let (tx_infer, rx_infer): (Sender<InferTask>, Receiver<InferTask>) =
        bounded(2 * infer_workers);
    let (tx_write, rx_write): (Sender<WriteTask>, Receiver<WriteTask>) =
        bounded(2 * infer_workers);

    info!(
        "启动流水线: {} 读取线程, {} 推理线程, 1 写入线程",
        args.read_thread, infer_workers
    );

    let global_model = tch::CModule::load_on_device(args.checkpoint, device)
        .context("Failed to load global model")?;

    std::thread::scope(|s| -> Result<()> {
        s.spawn({
            let tx_file = tx_file.clone();
            let manifest_files = manifest.files.clone();
            move || -> Result<()> {
                for file_info in manifest_files {
                    tx_file.send(file_info).expect("tx_file send failed");
                }
                Ok(())
            }
        });
        drop(tx_file);

        for loader_idx in 0..args.read_thread {
            let rx_file = rx_file.clone();
            let tx_infer = tx_infer.clone();
            let manifest_dir = manifest_dir.to_path_buf();

            let illu_bq_max = illu_cfg["bq_max"];
            let illu_mq_max = illu_cfg["mq_max"];
            let ont_bq_max = ont_cfg["bq_max"];
            let ont_mq_max = ont_cfg["mq_max"];
            let batch_size = args.batch_size as i64;

            s.spawn(move || -> Result<()> {
                let cpu_device = Device::Cpu;

                while let Ok(file_info) = rx_file.recv() {
                    let file_path = manifest_dir.join(&file_info.path);
                    let compressed_file = File::open(&file_path)?;
                    let mut decoder = zstd::Decoder::new(compressed_file)?;
                    let mut buffer = Vec::new();
                    decoder.read_to_end(&mut buffer)?;
                    let st = SafeTensors::deserialize(&buffer)?;

                    let (illu_bases, illu_strand, illu_feats, illu_cigar, _) =
                        build_platform_data(&st, "illu", cpu_device, illu_bq_max, illu_mq_max)?;
                    let (ont_bases, ont_strand, ont_feats, ont_cigar, _) =
                        build_platform_data(&st, "ont", cpu_device, ont_bq_max, ont_mq_max)?;

                    let ont_read_ids = get_tensor(&st, "ont_read_ids", Kind::Int64, cpu_device)?;
                    let ref_seq = get_tensor(&st, "ref_seq", Kind::Int8, cpu_device)?.to_kind(Kind::Int64);
                    let target_cols = get_tensor(&st, "target_cols", Kind::Int, cpu_device)?.to_kind(Kind::Int64);
                    let target_lengths = get_tensor(&st, "target_len", Kind::Int, cpu_device)?.to_kind(Kind::Int64);
                    let chrom_ids = get_tensor(&st, "chrom_id", Kind::Int, cpu_device)?;
                    let ref_pos = get_tensor(&st, "targets_ref_pos", Kind::Int, cpu_device)?;
                    let seq_lens = get_tensor(&st, "seq_len", Kind::Int, cpu_device)?.to_kind(Kind::Int64);

                    let total_samples = ref_seq.size()[0];
                    let mut batch_start: i64 = 0;

                    let max_file_w = seq_lens.max().int64_value(&[]).max(1);
                    let pos_idx_full =
                        Tensor::arange(max_file_w, (Kind::Int64, cpu_device)).unsqueeze(0);

                    while batch_start < total_samples {
                        let current_batch_size = std::cmp::min(batch_size, total_samples - batch_start);
                        let c_seq_lens = seq_lens.narrow(0, batch_start, current_batch_size);
                        let batch_max_w = c_seq_lens.max().int64_value(&[]).max(1);

                        let slice_batch = |t: &Tensor| t.narrow(0, batch_start, current_batch_size);
                        let to_cpu_bw = |t: &Tensor| slice_batch(t).narrow(1, 0, batch_max_w);

                        let c_seq_lens_cpu = c_seq_lens.unsqueeze(1);
                        let pos_idx = pos_idx_full.narrow(1, 0, batch_max_w);
                        let pad_mask_2d = pos_idx.ge_tensor(&c_seq_lens_cpu);

                        let pad_mask_2d_unsqueeze_2 = pad_mask_2d.unsqueeze(2);
                        let pad_mask_2d_unsqueeze_3 = pad_mask_2d_unsqueeze_2.unsqueeze(3);

                        let mut b_illu_bases_cpu = to_cpu_bw(&illu_bases).copy();
                        let _ = b_illu_bases_cpu.masked_fill_(&pad_mask_2d_unsqueeze_2, BASE_BATCH_PAD);

                        let mut b_illu_strand_cpu = to_cpu_bw(&illu_strand).copy();
                        let _ = b_illu_strand_cpu.masked_fill_(&pad_mask_2d_unsqueeze_2, STRAND_PAD_IDX);

                        let mut b_illu_feats_cpu = to_cpu_bw(&illu_feats).copy();
                        let _ = b_illu_feats_cpu.masked_fill_(&pad_mask_2d_unsqueeze_3, DEFAULT_BATCH_PAD);

                        let mut b_illu_cigar_cpu = to_cpu_bw(&illu_cigar).copy();
                        let _ = b_illu_cigar_cpu.masked_fill_(&pad_mask_2d_unsqueeze_2, CIGAR_BATCH_PAD);

                        let b_illu_mask_cpu = b_illu_bases_cpu.ne(BASE_BATCH_PAD);

                        let mut b_ont_bases_cpu = to_cpu_bw(&ont_bases).copy();
                        let _ = b_ont_bases_cpu.masked_fill_(&pad_mask_2d_unsqueeze_2, BASE_BATCH_PAD);

                        let mut b_ont_strand_cpu = to_cpu_bw(&ont_strand).copy();
                        let _ = b_ont_strand_cpu.masked_fill_(&pad_mask_2d_unsqueeze_2, STRAND_PAD_IDX);

                        let mut b_ont_feats_cpu = to_cpu_bw(&ont_feats).copy();
                        let _ = b_ont_feats_cpu.masked_fill_(&pad_mask_2d_unsqueeze_3, DEFAULT_BATCH_PAD);

                        let mut b_ont_cigar_cpu = to_cpu_bw(&ont_cigar).copy();
                        let _ = b_ont_cigar_cpu.masked_fill_(&pad_mask_2d_unsqueeze_2, CIGAR_BATCH_PAD);

                        let b_ont_mask_cpu = b_ont_bases_cpu.ne(BASE_BATCH_PAD);

                        let mut b_ref_seq_cpu = slice_batch(&ref_seq).narrow(1, 0, batch_max_w).copy();
                        let _ = b_ref_seq_cpu.masked_fill_(&pad_mask_2d, BASE_BATCH_PAD);

                        let b_target_lengths = slice_batch(&target_lengths).contiguous();
                        let batch_max_t = b_target_lengths.max().int64_value(&[]).max(1);
                        let b_target_cols_cpu =
                            slice_batch(&target_cols).narrow(1, 0, batch_max_t).contiguous();

                        let inputs_cpu = vec![
                            b_illu_bases_cpu,
                            b_illu_strand_cpu,
                            b_illu_feats_cpu,
                            b_illu_cigar_cpu,
                            b_ont_bases_cpu.shallow_clone(),
                            b_ont_strand_cpu,
                            b_ont_feats_cpu,
                            b_ont_cigar_cpu,
                            b_illu_mask_cpu,
                            b_ont_mask_cpu,
                            b_ref_seq_cpu.shallow_clone(),
                            b_target_cols_cpu.shallow_clone(),
                            b_target_lengths.shallow_clone(),
                        ];

                        let meta = BatchMeta {
                            ref_seq: b_ref_seq_cpu,
                            target_cols: b_target_cols_cpu,
                            target_lengths: b_target_lengths,
                            chrom_ids: slice_batch(&chrom_ids).contiguous(),
                            ref_pos: slice_batch(&ref_pos).narrow(1, 0, batch_max_t).contiguous(),
                            ont_bases: b_ont_bases_cpu,
                            ont_read_ids: slice_batch(&ont_read_ids).contiguous(),
                            batch_max_w,
                            batch_max_t,
                            current_batch_size,
                        };

                        tx_infer
                            .send(InferTask { inputs_cpu, meta })
                            .expect("tx_infer send failed");

                        batch_start += current_batch_size;
                    }
                    info!("loader {} finished file {}", loader_idx, file_info.path);
                }

                Ok(())
            });
        }
        drop(tx_infer);

        for _ in 0..infer_workers {
            let rx_infer = rx_infer.clone();
            let tx_write = tx_write.clone();
            let model = &global_model;
            let id_to_chrom = &id_to_chrom;
            let ref_threshold = args.ref_threshold;
            let het_threshold = args.het_threshold;

            s.spawn(move || -> Result<()> {
                tch::no_grad(|| -> Result<()> {
                    while let Ok(task) = rx_infer.recv() {
                        let inputs: Vec<IValue> = if matches!(device, Device::Cpu) {
                            task.inputs_cpu.into_iter().map(IValue::from).collect()
                        } else {
                            task.inputs_cpu
                                .into_iter()
                                .map(|t| IValue::from(t.to_device(device)))
                                .collect()
                        };

                        let output = model.forward_is(&inputs)?;
                        let (logits_het_dev, logits_base_dev) = match output {
                            IValue::Tuple(tup) if tup.len() >= 2 => {
                                let t0 = if let IValue::Tensor(ref t) = tup[0] {
                                    t.shallow_clone()
                                } else {
                                    bail!("输出0非张量");
                                };
                                let t1 = if let IValue::Tensor(ref t) = tup[1] {
                                    t.shallow_clone()
                                } else {
                                    bail!("输出1非张量");
                                };
                                (t0, t1)
                            }
                            _ => bail!("模型输出格式不匹配"),
                        };

                        let (logits_het, logits_base) = if matches!(device, Device::Cpu) {
                            (logits_het_dev, logits_base_dev)
                        } else {
                            (
                                logits_het_dev.to_device(Device::Cpu),
                                logits_base_dev.to_device(Device::Cpu),
                            )
                        };

                        let data = process_batch_to_msgpack(
                            &logits_het,
                            &logits_base,
                            task.meta,
                            id_to_chrom,
                            ref_threshold,
                            het_threshold,
                        )?;

                        if !data.is_empty() {
                            tx_write
                                .send(WriteTask { data })
                                .expect("tx_write send failed");
                        }
                    }
                    Ok(())
                })
            });
        }
        drop(tx_write);

        s.spawn(move || -> Result<()> {
            let stdout = io::stdout();
            let mut out_writer = BufWriter::new(stdout.lock());

            while let Ok(task) = rx_write.recv() {
                if !task.data.is_empty() {
                    out_writer.write_all(&task.data)?;
                }
            }

            out_writer.flush()?;
            Ok(())
        });

        Ok(())
    })?;

    Ok(())
}
