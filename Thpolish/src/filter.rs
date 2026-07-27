use anyhow::Result;
use rust_htslib::bam::record::{Aux, Cigar};
use rust_htslib::bam::{self, Read, Record, Reader, Writer, CompressionLevel, Header};
use std::path::Path;
use std::cmp::Reverse;

use crate::option::FilterArgs;

const COMPLEMENT: [u8; 256] = {
    let mut c = [0u8; 256];
    let mut i = 0;
    while i < 256 { c[i] = i as u8; i += 1; }
    c[b'A' as usize] = b'T'; c[b'a' as usize] = b't';
    c[b'C' as usize] = b'G'; c[b'c' as usize] = b'g';
    c[b'G' as usize] = b'C'; c[b'g' as usize] = b'c';
    c[b'T' as usize] = b'A'; c[b't' as usize] = b'a';
    c[b'U' as usize] = b'A'; c[b'u' as usize] = b'a';
    c[b'N' as usize] = b'N'; c[b'n' as usize] = b'n';
    c
};

fn reverse_complement(seq: &[u8]) -> Vec<u8> {
    seq.iter().rev().map(|&b| COMPLEMENT[b as usize]).collect()
}

struct SourceInfo {
    seq_plus: Vec<u8>,
    qual_plus: Vec<u8>,
    seq_minus: Vec<u8>,
    qual_minus: Vec<u8>,
    primary_score: i64,
}

impl SourceInfo {
    fn new(seq: &[u8], qual: &[u8], is_reverse: bool, score: i64) -> Self {
        let (seq_plus, qual_plus, seq_minus, qual_minus) = if is_reverse {
            let p_seq = reverse_complement(seq);
            let p_qual: Vec<u8> = qual.iter().rev().cloned().collect();
            (p_seq, p_qual, seq.to_vec(), qual.to_vec())
        } else {
            let m_seq = reverse_complement(seq);
            let m_qual: Vec<u8> = qual.iter().rev().cloned().collect();
            (seq.to_vec(), qual.to_vec(), m_seq, m_qual)
        };
        Self { seq_plus, qual_plus, seq_minus, qual_minus, primary_score: score }
    }
}

#[inline]
fn apply_hard_clipping<'a>(
    full_seq: &'a [u8],
    full_qual: &'a [u8],
    cigar: &rust_htslib::bam::record::CigarStringView,
) -> (&'a [u8], &'a [u8]) {
    let mut start_trim = 0;
    let mut end_trim = 0;

    for c in cigar.iter() {
        if let Cigar::HardClip(len) = c { start_trim += len; } else { break; }
    }
    for c in cigar.iter().rev() {
        if let Cigar::HardClip(len) = c { end_trim += len; } else { break; }
    }

    let total_len = full_seq.len();
    let slice_end = (total_len as u32).saturating_sub(end_trim) as usize;
    let start_trim = start_trim as usize;

    if start_trim >= slice_end || start_trim >= total_len {
        return (&[], &[]);
    }
    (&full_seq[start_trim..slice_end], &full_qual[start_trim..slice_end])
}

fn get_int_tag(r: &Record, tag: &[u8]) -> Option<i64> {
    match r.aux(tag) {
        Ok(Aux::I8(v)) => Some(v as i64),
        Ok(Aux::U8(v)) => Some(v as i64),
        Ok(Aux::I16(v)) => Some(v as i64),
        Ok(Aux::U16(v)) => Some(v as i64),
        Ok(Aux::I32(v)) => Some(v as i64),
        Ok(Aux::U32(v)) => Some(v as i64),
        Ok(Aux::Float(v)) => Some(v as i64), 
        _ => None,
    }
}

fn count_condensed_mismatches(md: &[u8]) -> i64 {
    let mut count = 0;
    let mut current_num: i64 = 0;
    let mut reading_num = true; 
    let mut in_error_block = false;

    for &b in md {
        if b.is_ascii_digit() {
            if !reading_num {
                reading_num = true;
                current_num = 0;
            }
            current_num = current_num * 10 + (b - b'0') as i64;
        } else {
            if reading_num {
                if current_num > 0 {
                    in_error_block = false;
                }
                reading_num = false;
            }

            if !in_error_block {
                count += 1;
                in_error_block = true;
            }
        }
    }
    count
}

struct Candidate {
    score: i64,
    record: Record,
}

fn process_group(
    group: &mut Vec<Record>, 
    writer: &mut Writer, 
    min_ratio: f64, 
    tags_to_delete: &[Vec<u8>],
    max_mismatch: Option<i64>,
    use_condensed_nm: bool,
    max_records: usize
) -> Result<()> {
    if group.is_empty() { return Ok(()); }
    let shared_qname = group[0].qname().to_vec();

    let mut source_r1 = None;
    let mut source_r2 = None;
    let mut source_u = None;

    for r in group.iter() {
        if (r.flags() & 0x900) == 0 {
            let seq = r.seq().as_bytes();
            if !seq.is_empty() && seq != b"*" {
                let score = get_int_tag(r, b"AS").unwrap_or(0);
                let info = SourceInfo::new(&seq, &r.qual(), r.is_reverse(), score);
                if r.is_paired() {
                    if r.is_first_in_template() { source_r1 = Some(info); }
                    else if r.is_last_in_template() { source_r2 = Some(info); }
                } else { source_u = Some(info); }
            }
        }
    }

    let mut candidates: Vec<Candidate> = Vec::with_capacity(group.len());

    for r in group.drain(..) {
        if r.is_unmapped() { continue; }
        if (r.flags() & 0x900) == 0 { 
            writer.write(&r)?;
            continue ;
        }

        let score = get_int_tag(&r, b"AS").unwrap_or(0);

        if r.seq().is_empty() {
            let source_score = if r.is_paired() {
                if r.is_first_in_template() { source_r1.as_ref().map(|s| s.primary_score) }
                else if r.is_last_in_template() { source_r2.as_ref().map(|s| s.primary_score) }
                else { None }
            } else { source_u.as_ref().map(|s| s.primary_score) };

            if let Some(ps) = source_score {
                if (score as f64) < (ps as f64 * min_ratio) { continue; }
            }
        }

        if let Some(limit) = max_mismatch {
            let nm_keep = if use_condensed_nm {
                match r.aux(b"MD") {
                    Ok(Aux::String(md)) => count_condensed_mismatches(md.as_bytes()) <= limit,
                    _ => get_int_tag(&r, b"NM").unwrap_or(0) <= limit,
                }
            } else {
                get_int_tag(&r, b"NM").unwrap_or(0) <= limit
            };
            if !nm_keep { continue; }
        }

        candidates.push(Candidate { score, record: r });
    }

    if candidates.len() > max_records {
        candidates.select_nth_unstable_by_key(max_records, |c| Reverse(c.score));
        candidates.truncate(max_records);
    }

    for mut c in candidates {
        let r = &mut c.record;

        for tag in tags_to_delete {
            let _ = r.remove_aux(tag);
        }

        if r.seq().is_empty() {
            let source_data = if r.is_paired() {
                if r.is_first_in_template() { source_r1.as_ref() }
                else if r.is_last_in_template() { source_r2.as_ref() }
                else { None }
            } else { source_u.as_ref() };

            if let Some(source) = source_data {
                let (base_seq, base_qual) = if r.is_reverse() {
                    (&source.seq_minus, &source.qual_minus)
                } else {
                    (&source.seq_plus, &source.qual_plus)
                };

                let (f_seq, f_qual) = apply_hard_clipping(base_seq, base_qual, &r.cigar());
                let cigar = r.cigar().to_owned();
                r.set(&shared_qname, Some(&cigar), f_seq, f_qual);
            } else {
                continue; 
            }
        }

        writer.write(r)?;
    }

    Ok(())
}

pub fn run_filter(args: FilterArgs) -> Result<()> {

    let tags_to_delete_bytes: Vec<Vec<u8>> = args.delete_tags
        .iter().map(|s| s.as_bytes().to_vec()).collect();

    let mut reader = if args.bam == "-" {
        Reader::from_stdin()?
    } else {
        Reader::from_path(Path::new(&args.bam))?
    };

    if args.thread > 0 {
        reader.set_threads(args.thread)?;
    }

    let header = Header::from_template(reader.header());

    let mut writer = Writer::from_path(Path::new("-"), &header, bam::Format::Bam)?;
    writer.set_compression_level(CompressionLevel::Uncompressed)?;
    if args.thread > 0 {
        writer.set_threads(args.thread)?;
    }

    let mut group: Vec<Record> = Vec::with_capacity(32);
    let mut spare_records: Vec<Record> = Vec::with_capacity(32);

    let mut temp_record = Record::new();
    let mut current_qname: Option<Vec<u8>> = None;

    while reader.read(&mut temp_record).is_some() {
        let qname_changed = match &current_qname {
            Some(curr) => curr != temp_record.qname(),
            None => true,
        };

        if qname_changed {
            if !group.is_empty() {
                process_group(
                    &mut group, 
                    &mut writer, 
                    args.ratio, 
                    &tags_to_delete_bytes,
                    args.max_mismatch,
                    args.condensed_nm,
                    args.max_records
                )?;
                spare_records.append(&mut group);
            }
            current_qname = Some(temp_record.qname().to_vec());
        }
        
        group.push(temp_record);
        temp_record = spare_records.pop().unwrap_or_else(Record::new);
    }

    if !group.is_empty() {
        process_group(
            &mut group, 
            &mut writer, 
            args.ratio, 
            &tags_to_delete_bytes,
            args.max_mismatch,
            args.condensed_nm,
            args.max_records
        )?;
    }

    Ok(())
}
