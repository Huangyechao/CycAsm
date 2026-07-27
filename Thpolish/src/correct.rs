use anyhow::{anyhow, bail, Context, Result};
use log::info;
use rmpv::decode::read_value;
use rmpv::Value;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use std::cmp::Ordering;
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use xxhash_rust::xxh64::xxh64;

use crate::option::CorrectArgs;

#[derive(Debug, Clone)]
struct SiteData {
    pos: i64,
    offset: usize,
    p_base: Vec<f32>,
    p_het: f32,
    read_data: Vec<(i64, usize)>,
}

#[derive(Debug, Clone)]
struct WindowData {
    chrom: String,
    ref_seq: Vec<usize>,
    sites: Vec<SiteData>,
}

type RawEvent = (usize, i64, usize, usize, f32, f32, String, f32);
type VariantRecord = (i64, String, String, i32, String, f32, f32, f32, String);

struct VCFWriter {
    f: BufWriter<File>,
    written_positions: HashSet<(String, i64)>,
    variant_buffer: HashMap<String, Vec<VariantRecord>>,
    int_to_char: HashMap<usize, char>,
    gap_idx: usize,
    del_idx: usize,
}

impl VCFWriter {
    fn new(path: &str) -> Result<Self> {
        let file = File::create(path).with_context(|| format!("failed to create VCF: {path}"))?;
        let mut writer = VCFWriter {
            f: BufWriter::with_capacity(1024 * 1024, file),
            written_positions: HashSet::default(),
            variant_buffer: HashMap::default(),
            int_to_char: {
                let mut m: HashMap<usize, char> = HashMap::default();
                m.insert(0, 'A');
                m.insert(1, 'C');
                m.insert(2, 'G');
                m.insert(3, 'T');
                m.insert(4, 'N');
                m.insert(5, '-');
                m.insert(6, ' ');
                m.insert(7, '#');
                m
            },
            gap_idx: 5,
            del_idx: 5,
        };
        writer.write_header()?;
        Ok(writer)
    }

    fn write_header(&mut self) -> Result<()> {
        writeln!(self.f, "##fileformat=VCFv4.2")?;
        writeln!(self.f, "##source=ThPolish")?;
        writeln!(
            self.f,
            "##FILTER=<ID=LowQual,Description=\"Probabilty below threshold\">"
        )?;
        writeln!(
            self.f,
            "##INFO=<ID=PROB,Number=1,Type=Float,Description=\"Corrected prediction probability\">"
        )?;
        writeln!(
            self.f,
            "##INFO=<ID=DELTA,Number=1,Type=Float,Description=\"Probability gain (Corrected - Ref)\">"
        )?;
        writeln!(
            self.f,
            "##INFO=<ID=PHET,Number=1,Type=Float,Description=\"Heterozygosity score\">"
        )?;
        writeln!(
            self.f,
            "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">"
        )?;
        writeln!(
            self.f,
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE"
        )?;
        Ok(())
    }

    fn find_anchor(&self, ref_seq_arr: &[usize], start_col_idx: usize) -> char {
        let mut curr = start_col_idx as isize - 1;
        while curr >= 0 {
            let val = ref_seq_arr[curr as usize];
            if val < 5 {
                return *self.int_to_char.get(&val).unwrap_or(&'N');
            }
            curr -= 1;
        }
        'N'
    }

    fn process_variants(
        &mut self,
        chrom: &str,
        ref_seq_arr: &[usize],
        raw_events: &[RawEvent],
    ) -> usize {
        let mut col_map: HashMap<usize, RawEvent> = HashMap::default();
        for item in raw_events {
            col_map.insert(item.0, item.clone());
        }

        let mut sorted_cols: Vec<usize> = col_map.keys().copied().collect();
        sorted_cols.sort_unstable();
        if sorted_cols.is_empty() {
            return 0;
        }

        let mut blocks: Vec<Vec<usize>> = Vec::new();
        let mut current_block_cols = vec![sorted_cols[0]];
        for i in 1..sorted_cols.len() {
            let prev_col = sorted_cols[i - 1];
            let curr_col = sorted_cols[i];
            if col_map[&curr_col].1 == col_map[&prev_col].1 || curr_col == prev_col + 1 {
                current_block_cols.push(curr_col);
            } else {
                blocks.push(current_block_cols);
                current_block_cols = vec![curr_col];
            }
        }
        blocks.push(current_block_cols);

        let mut batch_cache: HashMap<i64, (String, String, String, i32, String, f32, f32, f32)> =
            HashMap::default();

        for block_cols in blocks {
            let all_block_events: Vec<RawEvent> = block_cols.iter().map(|c| col_map[c].clone()).collect();
            let first_evt = &all_block_events[0];
            let first_col = first_evt.0;
            let first_pos = first_evt.1;

            if self
                .written_positions
                .contains(&(chrom.to_string(), first_pos))
            {
                continue;
            }

            let mut merged_ref: Vec<char> = Vec::new();
            let mut merged_alt: Vec<char> = Vec::new();
            let anchor = self.find_anchor(ref_seq_arr, first_col);
            merged_ref.push(anchor);
            merged_alt.push(anchor);

            let mut vcf_pos = if first_evt.2 == self.gap_idx {
                first_pos + 1
            } else {
                first_pos
            };

            let mut min_prob = 1.0_f32;
            let mut min_delta = 1.0_f32;
            let mut avg_p_het = 0.0_f32;

            for evt in &all_block_events {
                let (col, pos, r_int, p_int, prob, delta, _filter, p_het) = evt;
                let _ = (col, pos);
                min_prob = min_prob.min(*prob);
                min_delta = min_delta.min(*delta);
                avg_p_het += *p_het;

                if *r_int != self.gap_idx {
                    merged_ref.push(*self.int_to_char.get(r_int).unwrap_or(&'N'));
                }
                if *p_int != self.del_idx {
                    merged_alt.push(*self.int_to_char.get(p_int).unwrap_or(&'N'));
                }
            }

            avg_p_het /= all_block_events.len() as f32;
            let mut final_ref: String = merged_ref.iter().collect();
            let mut final_alt: String = merged_alt.iter().collect();

            while final_ref.len() > 1
                && final_alt.len() > 1
                && final_ref.chars().last() == final_alt.chars().last()
            {
                final_ref.pop();
                final_alt.pop();
            }

            while final_ref.len() > 1
                && final_alt.len() > 1
                && final_ref.chars().next() == final_alt.chars().next()
            {
                final_ref.remove(0);
                final_alt.remove(0);
                vcf_pos += 1;
            }

            if final_ref != final_alt {
                let qual = 99_i32.min((-10.0_f64
                    * ((1.0_f64 - min_prob as f64).max(1e-10_f64)).log10())
                    as i32);
                let should_update = match batch_cache.get(&vcf_pos) {
                    Some(existing) => min_prob > existing.5,
                    None => true,
                };
                if should_update {
                    batch_cache.insert(
                        vcf_pos,
                        (
                            chrom.to_string(),
                            final_ref,
                            final_alt,
                            qual,
                            "PASS".to_string(),
                            min_prob,
                            min_delta,
                            avg_p_het,
                        ),
                    );
                }
                for c in &block_cols {
                    self.written_positions
                        .insert((chrom.to_string(), col_map[c].1));
                }
            }
        }

        let cache_len = batch_cache.len();
        for (vcf_pos, vals) in batch_cache {
            let (chrom, ref_allele, alt_allele, qual, filt, prob, delta, phet) = vals;
            let gt = if phet >= 0.5 { "0/1" } else { "1/1" }.to_string();
            self.variant_buffer.entry(chrom).or_default().push((
                vcf_pos, ref_allele, alt_allele, qual, filt, prob, delta, phet, gt,
            ));
        }
        cache_len
    }

    fn close(&mut self) -> Result<()> {
        if !self.variant_buffer.is_empty() {
            let mut chroms: Vec<String> = self.variant_buffer.keys().cloned().collect();
            chroms.sort();
            for chrom in chroms {
                if let Some(records) = self.variant_buffer.get_mut(&chrom) {
                    records.sort_by_key(|x| x.0);
                    for v in records.iter() {
                        writeln!(
                            self.f,
                            "{}\t{}\t.\t{}\t{}\t{}\t{}\tPROB={:.3};DELTA={:.3};PHET={:.3}\tGT\t{}",
                            chrom, v.0, v.1, v.2, v.3, v.4, v.5, v.6, v.7, v.8
                        )?;
                    }
                }
            }
        }
        self.f.flush()?;
        Ok(())
    }
}

struct GlobalReadIdentityCorrector {
    alpha: f32,
    het_threshold: f32,
    pseudo_count: f32,
    base_threshold: f32,
    read_identity_stats: HashMap<i64, (f32, f32)>,
    read_scores: HashMap<i64, f32>,
    read_coverage: HashMap<i64, f32>,
    haplo_cluster: HashMap<String, HashSet<i64>>,
}

impl GlobalReadIdentityCorrector {
    fn new(
        haplo_file: Option<&str>,
        alpha: f32,
        het_threshold: f32,
        pseudo_count: f32,
        base_threshold: f32,
    ) -> Result<Self> {
        Ok(Self {
            alpha,
            het_threshold,
            pseudo_count,
            base_threshold,
            read_identity_stats: HashMap::default(),
            read_scores: HashMap::default(),
            read_coverage: HashMap::default(),
            haplo_cluster: Self::load_external_haplotypes(haplo_file)?,
        })
    }

    fn to_signed_i64(v: u64) -> i64 {
        v as i64
    }

    fn load_external_haplotypes(haplo_file: Option<&str>) -> Result<HashMap<String, HashSet<i64>>> {
        let mut haplo_cluster: HashMap<String, HashSet<i64>> = HashMap::default();
        let Some(path) = haplo_file else {
            return Ok(haplo_cluster);
        };

        let file = File::open(path).with_context(|| format!("failed to open haplotype file: {path}"))?;
        for line in BufReader::new(file).lines() {
            let line = line.context("failed to read haplotype assignment line")?;
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() == 2 {
                let cluster_id = parts[1].to_string();
                let rid = Self::to_signed_i64(xxh64(parts[0].as_bytes(), 0));
                haplo_cluster.entry(cluster_id).or_default().insert(rid);
            } else if parts.len() >= 3 {
                let cluster_id = format!("{}_{}", parts[1], parts[2]);
                let rid = Self::to_signed_i64(xxh64(parts[0].as_bytes(), 0));
                haplo_cluster.entry(cluster_id).or_default().insert(rid);
            }
        }
        Ok(haplo_cluster)
    }

    fn collect_stats(&mut self, window_data: &WindowData) {
        for site in &window_data.sites {
            let p_het = site.p_het;
            if p_het < self.het_threshold {
                continue;
            }

            let p_base = &site.p_base;
            let ref_idx = window_data.ref_seq[site.offset];
            if ref_idx >= 4 {
                continue;
            }

            let (top1, top2) = top2_like_python_argsort_rev(p_base);
            if ref_idx != top1 && ref_idx != top2 {
                continue;
            }

            let alt_idx = if ref_idx == top1 { top2 } else { top1 };
            if alt_idx >= 4 {
                continue;
            }

            let p_ref = p_base[ref_idx];
            let p_alt = p_base[alt_idx];
            if p_ref < self.base_threshold || p_alt < self.base_threshold {
                continue;
            }

            for &(rid, rb) in &site.read_data {
                assert!(rb < 6);
                if rb != ref_idx && rb != alt_idx {
                    continue;
                }

                if p_base[rb] < self.base_threshold {
                    continue;
                }

                let delta = if rb == ref_idx { p_ref } else { -p_ref };

                let entry = self.read_identity_stats.entry(rid).or_insert((0.0, 0.0));
                entry.0 += p_het * delta;
                entry.1 += p_het;
            }
        }
    }

    fn fill_score_coverage(&mut self) {
        self.read_scores = self
            .read_identity_stats
            .iter()
            .filter_map(|(&rid, &(total_delta, count))| {
                if count > 0.1 {
                    Some((rid, total_delta / (count + self.pseudo_count)))
                } else {
                    None
                }
            })
            .collect();

        self.read_coverage = self
            .read_identity_stats
            .iter()
            .map(|(&rid, &(_total_delta, count))| (rid, count))
            .collect();

        for reads in self.haplo_cluster.values() {
            let mut cluster_scores: Vec<f32> = reads
                .iter()
                .filter_map(|rid| self.read_scores.get(rid).copied())
                .collect();
            if cluster_scores.is_empty() {
                continue;
            }

            cluster_scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
            let median_cluster_score = median_sorted(&cluster_scores);

            for &rid in reads {
                let original_score = *self.read_scores.get(&rid).unwrap_or(&median_cluster_score);
                let cov = *self.read_coverage.get(&rid).unwrap_or(&1.0);

                if median_cluster_score >= 0.0 {
                    self.read_scores.insert(rid, original_score.abs());
                } else if original_score > 0.0 {
                    self.read_scores.insert(rid, -original_score);
                }

                self.read_coverage.entry(rid).or_insert(cov);
            }
        }

        self.haplo_cluster.clear();
    }

    fn apply_correction(&self, window_data: &WindowData) -> Vec<RawEvent> {
        let mut raw_events = Vec::new();
        let ref_seq_arr = &window_data.ref_seq;

        for site in &window_data.sites {
            let p_base = &site.p_base;
            let offset = site.offset;
            let ref_idx = ref_seq_arr[offset];
            let raw_pred_idx = argmax(p_base);

            let mut sorted_probs = p_base.clone();
            sorted_probs.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
            let raw_margin = sorted_probs[0] - sorted_probs[1];
            let current_margin_threshold = 0.3_f32;

            let (final_idx, final_prob) = if site.p_het < self.het_threshold
                && raw_margin > current_margin_threshold
            {
                (raw_pred_idx, p_base[raw_pred_idx])
            } else {
                let mut haplo_prior = vec![1.0_f32; 6];
                for &(rid, rb) in &site.read_data {
                    if let Some(&s_r) = self.read_scores.get(&rid) {
                        let coverage_weight = self.read_coverage[&rid].ln_1p();
                        let score = (s_r * self.alpha).clamp(-20.0, 20.0);
                        let weight = score.exp() * coverage_weight;
                        haplo_prior[rb] += weight;
                    } else {
                        haplo_prior[rb] += 0.1;
                    }
                }

                let mut combined_probs: Vec<f32> = p_base
                    .iter()
                    .zip(haplo_prior.iter())
                    .map(|(p, h)| p * h)
                    .collect();
                let sum: f32 = combined_probs.iter().sum();
                let denom = sum + 1e-9_f32;
                for p in &mut combined_probs {
                    *p /= denom;
                }
                let idx = argmax(&combined_probs);
                (idx, combined_probs[idx])
            };

            if ref_idx != final_idx {
                let ref_p = p_base[ref_idx];
                let delta = final_prob - ref_p;
                raw_events.push((
                    site.offset,
                    site.pos,
                    ref_idx,
                    final_idx,
                    final_prob,
                    delta,
                    "PASS".to_string(),
                    site.p_het,
                ));
            }
        }

        raw_events
    }
}

fn median_sorted(values: &[f32]) -> f32 {
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    }
}

fn argmax(values: &[f32]) -> usize {
    let mut best_idx = 0usize;
    let mut best_val = values[0];
    for (idx, &val) in values.iter().enumerate().skip(1) {
        if val > best_val {
            best_idx = idx;
            best_val = val;
        }
    }
    best_idx
}

fn generate_corrected_fasta(
    ref_fasta_path: &str,
    out_fasta_path: &str,
    variants: &HashMap<String, Vec<VariantRecord>>,
) -> Result<()> {
    fn process_chrom<W: Write>(
        curr_chrom: Option<&str>,
        curr_seq_list: &[String],
        variants: &HashMap<String, Vec<VariantRecord>>,
        fout: &mut W,
    ) -> Result<()> {
        let Some(chrom) = curr_chrom else {
            return Ok(());
        };

        writeln!(fout, ">{chrom}")?;
        let curr_seq = curr_seq_list.join("").to_uppercase();

        if let Some(chrom_variants) = variants.get(chrom) {
            let mut last_idx = 0usize;
            for v in chrom_variants {
                let mut pos = v.0;
                let mut ref_allele = v.1.clone();
                let mut alt_allele = v.2.clone();

                if ref_allele.len() > 1 || alt_allele.len() > 1 {
                    if ref_allele.chars().next() == alt_allele.chars().next() {
                        ref_allele.remove(0);
                        alt_allele.remove(0);
                        pos += 1;
                    }
                }

                if ref_allele.is_empty() && alt_allele.is_empty() {
                    continue;
                }

                let idx = (pos - 1) as usize;
                if idx < last_idx {
                    eprintln!("[Warning] Overlapping variant at {chrom}:{pos}. Skipping.");
                    continue;
                }

                let ref_len = ref_allele.len();
                let end = idx + ref_len;
                let seq_ref = if idx <= curr_seq.len() && end <= curr_seq.len() {
                    &curr_seq[idx..end]
                } else {
                    ""
                };

                if seq_ref == ref_allele.to_uppercase() {
                    if idx > last_idx {
                        write!(fout, "{}", &curr_seq[last_idx..idx])?;
                    }
                    if !alt_allele.is_empty() {
                        write!(fout, "{alt_allele}")?;
                    }
                    last_idx = idx + ref_len;
                } else {
                    eprintln!(
                        "[Warning] REF mismatch at {chrom}:{pos}. VCF expected {ref_allele}, FASTA found {seq_ref}. Skipping."
                    );
                }
            }

            if last_idx < curr_seq.len() {
                write!(fout, "{}", &curr_seq[last_idx..])?;
            }
            writeln!(fout)?;
        } else {
            writeln!(fout, "{curr_seq}")?;
        }

        Ok(())
    }

    let fin = File::open(ref_fasta_path)
        .with_context(|| format!("failed to open reference FASTA: {ref_fasta_path}"))?;
    let mut fout = BufWriter::new(
        File::create(out_fasta_path)
            .with_context(|| format!("failed to create corrected FASTA: {out_fasta_path}"))?,
    );
    let mut curr_chrom: Option<String> = None;
    let mut curr_seq_list: Vec<String> = Vec::new();

    for line in BufReader::new(fin).lines() {
        let line = line.context("failed to read FASTA line")?.trim().to_string();
        if line.starts_with('>') {
            process_chrom(curr_chrom.as_deref(), &curr_seq_list, variants, &mut fout)?;
            curr_chrom = Some(line[1..].split_whitespace().next().unwrap_or("").to_string());
            curr_seq_list.clear();
        } else {
            curr_seq_list.push(line);
        }
    }
    process_chrom(curr_chrom.as_deref(), &curr_seq_list, variants, &mut fout)?;
    fout.flush()?;
    Ok(())
}

fn iter_windows_from_stream(filepath: &str) -> Result<Vec<WindowData>> {
    let file = File::open(filepath).with_context(|| format!("failed to open MsgPack input: {filepath}"))?;
    let mut reader = BufReader::new(file);
    let mut windows = Vec::new();
    let mut window_idx = 0usize;

    loop {
        match read_value(&mut reader) {
            Ok(window) => {
                windows.push(normalize_window(filepath, window_idx, &window)?);
                window_idx += 1;
            }
            Err(err) => {
                if is_eof_error(&err) {
                    break;
                }
                return Err(err).with_context(|| format!("failed to decode MsgPack window {window_idx} from {filepath}"));
            }
        }
    }

    Ok(windows)
}

fn is_eof_error(err: &rmpv::decode::Error) -> bool {
    matches!(err, rmpv::decode::Error::InvalidMarkerRead(e) if e.kind() == io::ErrorKind::UnexpectedEof)
        || matches!(err, rmpv::decode::Error::InvalidDataRead(e) if e.kind() == io::ErrorKind::UnexpectedEof)
}

fn normalize_window(
    filepath: &str,
    window_idx: usize,
    window: &Value,
) -> Result<WindowData> {
    match window {
        Value::Map(map) => {
            let chrom = map_get_string(map, "chrom")?;
            let ref_seq = value_to_usize_vec(map_get(map, "ref_seq")?)?;
            let sites_value = map_get(map, "sites")?;
            let sites = normalize_sites(filepath, window_idx, sites_value)?;
            Ok(WindowData {
                chrom,
                ref_seq,
                sites,
            })
        }
        Value::Array(arr) => {
            if arr.len() != 3 {
                bail!("[{filepath}] Window {window_idx} Array length mismatch.");
            }
            let chrom = value_to_string(&arr[0])?;
            let ref_seq = value_to_usize_vec(&arr[1])?;
            let sites = normalize_sites(filepath, window_idx, &arr[2])?;
            Ok(WindowData {
                chrom,
                ref_seq,
                sites,
            })
        }
        _ => bail!(
            "[{filepath}] Window {window_idx} unsupported type: {}",
            value_type_name(window)
        ),
    }
}

fn normalize_sites(
    filepath: &str,
    window_idx: usize,
    sites_value: &Value,
) -> Result<Vec<SiteData>> {
    let sites_arr = match sites_value {
        Value::Array(arr) => arr,
        _ => bail!("[{filepath}] Window {window_idx} sites unsupported type."),
    };

    let mut normalized = Vec::with_capacity(sites_arr.len());
    for (site_idx, site) in sites_arr.iter().enumerate() {
        let site_data = match site {
            Value::Map(map) => SiteData {
                pos: map_get_i64(map, "pos")?,
                offset: map_get_usize(map, "offset")?,
                p_base: value_to_f32_vec(map_get(map, "p_base")?)?,
                p_het: map_get_f32(map, "p_het")?,
                read_data: value_to_read_data(map_get(map, "read_data")?)?,
            },
            Value::Array(arr) => {
                if arr.len() != 5 {
                    bail!(
                        "[{filepath}] Window {window_idx}, Site {site_idx} SiteData length mismatch."
                    );
                }
                SiteData {
                    pos: value_to_i64(&arr[0])?,
                    offset: value_to_usize(&arr[1])?,
                    p_base: value_to_f32_vec(&arr[2])?,
                    p_het: value_to_f32(&arr[3])?,
                    read_data: value_to_read_data(&arr[4])?,
                }
            }
            _ => {
                bail!(
                    "[{filepath}] Window {window_idx}, Site {site_idx} unsupported type."
                )
            }
        };
        normalized.push(site_data);
    }
    Ok(normalized)
}

fn map_get<'a>(map: &'a [(Value, Value)], key: &str) -> Result<&'a Value> {
    for (k, v) in map {
        if value_key_eq(k, key) {
            return Ok(v);
        }
    }
    Err(anyhow!("missing key: {key}"))
}

fn map_get_string(map: &[(Value, Value)], key: &str) -> Result<String> {
    value_to_string(map_get(map, key)?)
}

fn map_get_i64(map: &[(Value, Value)], key: &str) -> Result<i64> {
    value_to_i64(map_get(map, key)?)
}

fn map_get_usize(map: &[(Value, Value)], key: &str) -> Result<usize> {
    value_to_usize(map_get(map, key)?)
}

fn map_get_f32(map: &[(Value, Value)], key: &str) -> Result<f32> {
    value_to_f32(map_get(map, key)?)
}

fn value_key_eq(value: &Value, key: &str) -> bool {
    match value {
        Value::String(s) => s.as_str().is_some_and(|x| x == key),
        Value::Binary(bytes) => std::str::from_utf8(bytes).is_ok_and(|x| x == key),
        _ => false,
    }
}

fn value_to_string(value: &Value) -> Result<String> {
    match value {
        Value::String(s) => Ok(s.as_str().unwrap_or_default().to_string()),
        Value::Binary(bytes) => Ok(String::from_utf8(bytes.clone())?),
        _ => bail!("expected string, got {}", value_type_name(value)),
    }
}

fn value_to_i64(value: &Value) -> Result<i64> {
    match value {
        Value::Integer(i) => i
            .as_i64()
            .or_else(|| i.as_u64().map(|x| x as i64))
            .ok_or_else(|| anyhow!("integer out of range")),
        _ => bail!("expected int, got {}", value_type_name(value)),
    }
}

fn value_to_usize(value: &Value) -> Result<usize> {
    let n = value_to_i64(value)?;
    if n < 0 {
        bail!("negative integer cannot be converted to usize");
    }
    Ok(n as usize)
}

fn value_to_f32(value: &Value) -> Result<f32> {
    match value {
        Value::F32(x) => Ok(*x),
        Value::F64(x) => Ok(*x as f32),
        Value::Integer(i) => i
            .as_i64()
            .map(|x| x as f32)
            .or_else(|| i.as_u64().map(|x| x as f32))
            .ok_or_else(|| anyhow!("integer out of range")),
        _ => bail!("expected float, got {}", value_type_name(value)),
    }
}

fn value_to_usize_vec(value: &Value) -> Result<Vec<usize>> {
    match value {
        Value::Array(arr) => arr.iter().map(value_to_usize).collect(),
        _ => bail!("expected array, got {}", value_type_name(value)),
    }
}

fn value_to_f32_vec(value: &Value) -> Result<Vec<f32>> {
    match value {
        Value::Array(arr) => arr.iter().map(value_to_f32).collect(),
        _ => bail!("expected array, got {}", value_type_name(value)),
    }
}

fn value_to_read_data(value: &Value) -> Result<Vec<(i64, usize)>> {
    let arr = match value {
        Value::Array(arr) => arr,
        _ => bail!("expected read_data array, got {}", value_type_name(value)),
    };

    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        match item {
            Value::Array(pair) if pair.len() == 2 => {
                out.push((value_to_i64(&pair[0])?, value_to_usize(&pair[1])?));
            }
            _ => bail!("read_data item must be a 2-element array"),
        }
    }
    Ok(out)
}

fn value_type_name(value: &Value) -> &'static str {
    match value {
        Value::Nil => "nil",
        Value::Boolean(_) => "bool",
        Value::Integer(_) => "int",
        Value::F32(_) => "f32",
        Value::F64(_) => "f64",
        Value::String(_) => "string",
        Value::Binary(_) => "binary",
        Value::Array(_) => "array",
        Value::Map(_) => "map",
        Value::Ext(_, _) => "ext",
    }
}

fn top2_like_python_argsort_rev(values: &[f32]) -> (usize, usize) {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| {
        match values[a].partial_cmp(&values[b]).unwrap_or(Ordering::Equal) {
            Ordering::Equal => a.cmp(&b),
            ord => ord,
        }
    });
    order.reverse();

    (order[0], order[1])
}

pub fn run_correct(args: CorrectArgs) -> Result<()> {
    let mut corrector = GlobalReadIdentityCorrector::new(
        args.haplo.as_deref(),
        args.alpha,
        args.het_threshold,
        1.0,
        args.base_threshold,
    )?;

    let mut all_windows = Vec::new();

    info!(">>> Pass 1: Collecting global read identities...");
    for window_data in iter_windows_from_stream(&args.input)? {
        corrector.collect_stats(&window_data);
        all_windows.push(window_data);
    }
    corrector.fill_score_coverage();

    info!(">>> Pass 2: Applying corrections...");
    let vcf_path = format!("{}.vcf", args.output_prefix);
    let mut writer = VCFWriter::new(&vcf_path)?;
    let mut count_variants = 0usize;

    for window in &all_windows {
        let events = corrector.apply_correction(window);
        if !events.is_empty() {
            count_variants += writer.process_variants(&window.chrom, &window.ref_seq, &events);
        }
    }
    writer.close()?;

    if let Some(ref_path) = args.ref_fasta.as_deref() {
        let fasta_path = format!("{}.fasta", args.output_prefix);
        generate_corrected_fasta(ref_path, &fasta_path, &writer.variant_buffer)?;
    }

    println!("Finished. Total variants: {count_variants}");
    Ok(())
}
