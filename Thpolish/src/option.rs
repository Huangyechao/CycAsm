#![allow(dead_code)]
use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};
use libc_stdhandle::stdout;
use path_absolutize::*;
use std::{
    ffi::CString,
    io::{Error, ErrorKind, Result as IoResult},
    os::unix::ffi::OsStrExt,
    path::Path,
};

#[derive(Parser, Debug)]
#[command(
    name = "THPolish",
    version = env!("VERSION"),
    about = "A T2T haplotype-aware polishing tool",
    arg_required_else_help = true,
    subcommand_required = true,
)]
pub struct PolishArgs {
    #[command(subcommand)]
    pub command: Commands,
}

impl PolishArgs {
    
    pub fn parse_with_redirect_io(redirect_io: bool) -> Result<Self> {
        let args = Self::parse();

        if redirect_io {
            args.setup_io()
                .context("Failed to initialize standard I/O redirection")?;
        }

        Ok(args)
    }

    fn setup_io(&self) -> IoResult<()> {
        let out_path = match &self.command {
            Commands::Filter(args) => Some(&args.out),
            Commands::Depth(args) => Some(&args.out),
            Commands::Predict(args) => Some(&args.out),
            _ => None,
        };
        if let Some(path) = out_path {
            if path != "stdout" && path != "-" {
                freopen_stdout(path)?;
            }
        }
        Ok(())
    }
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Filter BAM records based on alignment scores and mismatch profiles
    Filter(FilterArgs),
    /// Calculate zero-centered normalized sequencing depth
    Depth(DepthArgs),

    /// Encode aligned reads and depths into feature matrices
    Encode(EncodeArgs),

    /// Perform inference on encoded shards to detect base-level error loci.
    Predict(PredictArgs),

    /// Apply global haplotype-aware correction and generate VCF/optional FASTA.
    Correct(CorrectArgs),
}


#[derive(Args, Debug)]
#[command(arg_required_else_help = true)]
pub struct FilterArgs {
    /// Input alignment file in BAM format (use '-' for stdin).
    #[arg(required = true, value_parser = parse_input_path)]
    pub bam: String,

    /// Number of working threads.
    #[arg(short = 't', long, default_value_t = 3)]
    pub thread: usize,

    /// Path to the output file.
    #[arg(short = 'o', long, default_value = "stdout")]
    pub out: String,

    /// Minimum alignment score ratio relative to the primary alignment.
    #[arg(short, long, default_value_t = 0.8)]
    pub ratio: f64,

    /// Comma-separated list of auxiliary BAM tags to remove.
    #[arg(short = 'd', long, value_delimiter = ',', default_value = "MD,MC,XS,NM")]
    pub delete_tags: Vec<String>,

    /// Maximum permitted mismatches (evaluates NM tag or equivalent).
    #[arg(short = 'm', long)]
    pub max_mismatch: Option<i64>,

    /// Treat consecutive mismatches and INDELs as a single error event (requires MD tag).
    #[arg(short = 'c', long)]
    pub condensed_nm: bool,

    /// Maximum number of secondary/supplementary alignments retained per read.
    #[arg(short = 'n', long, default_value_t = 100)]
    pub max_records: usize,
}

#[derive(Args, Debug)]
#[command(arg_required_else_help = true)]
pub struct DepthArgs {
    /// Input alignment file in BAM format.
    #[arg(required = true, value_parser = parse_abspath)]
    pub bam: String,

    /// Number of working threads.
    #[arg(short = 't', long, default_value_t = 3)]
    pub thread: usize,

    /// Path to the output file.
    #[arg(short = 'o', long, default_value = "stdout")]
    pub out: String,

    /// Minimum contig/chromosome length to be included in the analysis.
    #[arg(short = 'l', long, default_value_t = 0)]
    pub min_len: usize,

    /// Target chromosomes/contigs for processing (e.g., --chroms chr1 chr2).
    #[arg(short = 'c', long, num_args = 1..)]
    pub chroms: Option<Vec<String>>,
}

#[derive(Args, Debug)]
#[command(arg_required_else_help = true)]
pub struct EncodeArgs {
    /// Number of working threads.
    #[arg(short = 't', long, default_value_t = 10)]
    pub thread: usize,

    /// Output directory for encoded shards and manifest.
    #[arg(short = 'o', long = "out_dir", required = true)]
    pub out_dir: String,

    /// Input Illumina BAM file.
    #[arg(short = '1', long = "bam_illumina", required = true, value_parser = parse_abspath)]
    pub bam_illumina: String,

    /// Input ONT BAM file.
    #[arg(short = '2', long = "bam_ont", required = true, value_parser = parse_abspath)]
    pub bam_ont: String,

    /// Path to Illumina normalized depth archive (.bin.zst).
    #[arg(long = "depth_illu", required = true, value_parser = parse_abspath)]
    pub depth_illu: String,

    /// Path to ONT normalized depth archive (.bin.zst).
    #[arg(long = "depth_ont", required = true, value_parser = parse_abspath)]
    pub depth_ont: String,

    /// Reference FASTA file.
    #[arg(short = 'r', long = "ref", required = true, value_parser = parse_abspath)]
    pub ref_fasta: String,

    /// Input BED file with genomic windows.
    #[arg(long = "bed_in", required = true, value_parser = parse_abspath)]
    pub bed_in: String,

    /// Maximum read depth per window (downsampling threshold).
    #[arg(long, default_value_t = 30)]
    pub depth: usize,

    /// Chunk size for parallel block scanning.
    #[arg(long, default_value_t = 5_000_000)]
    pub scan_block_size: usize,

    /// Number of windows processed per encoding task batch.
    #[arg(long, default_value_t = 1000)]
    pub batch_size: usize,

    /// Maximum allowed insertion size to be mapped into the feature matrix.
    #[arg(long, default_value_t = 15)]
    pub max_insert_size: u32,

    /// Minimum mapping quality (MQ) filter for reads.
    #[arg(long, default_value_t = 1)]
    pub min_mq: u8,
}

#[derive(Args, Debug)]
#[command(arg_required_else_help = true)]
pub struct PredictArgs {
    /// Path to the model checkpoint (.pt).
    #[arg(required = true, value_parser = parse_abspath)]
    pub checkpoint: String,

    /// Path to the encoded shards manifest file (manifest.json).
    #[arg(required = true, value_parser = parse_abspath)]
    pub manifest: String,

    /// Number of DataLoader working threads.
    #[arg(short = 't', long, default_value_t = 2)]
    pub read_thread: usize,

    /// Number of concurrent inference workers in CPU model.
    #[arg(short = 'T', long, default_value_t = 12)]
    pub infer_workers: usize,

    /// Path to the output file.
    #[arg(short = 'o', long, default_value = "stdout")]
    pub out: String,

    /// Inference batch size.
    #[arg(short = 'b', long, default_value_t = 32)]
    pub batch_size: usize,

    /// Threshold for heterozygosity prediction.
    #[arg(short = 'e', long, default_value_t = 0.4)]
    pub het_threshold: f32,

    /// Only skip if predicted base is Ref and its prob > this.
    #[arg(short = 'E', long, default_value_t = 0.8)]
    pub ref_threshold: f32,
}

#[derive(Args, Debug)]
#[command(arg_required_else_help = true)]
pub struct CorrectArgs {
    /// Input tensor stream generated by the predictor.
    #[arg(required = true, value_parser = parse_abspath)]
    pub input: String,

    /// Prefix for output files. Generates <prefix>.vcf and, if --ref is provided, <prefix>.fasta.
    #[arg(short = 'o', long, default_value = "thpolish.correct")]
    pub output_prefix: String,

    /// Reference genome FASTA. Enables consensus FASTA generation when provided.
    #[arg(short = 'r', long, value_parser = parse_abspath)]
    pub ref_fasta: Option<String>,

    /// External read-to-haplotype assignment file. Expected TSV: Read_ID Hap_ID [Phase_Block_ID].
    #[arg(short = 'p', long, value_parser = parse_abspath)]
    pub haplo: Option<String>,

    /// Scaling factor for read identity priors in the probabilistic fusion model.
    #[arg(short = 'a', long, default_value_t = 3.0)]
    pub alpha: f32,

    /// Minimum heterozygosity probability required to trigger phase block aggregation and correction.
    #[arg(short = 'e', long, default_value_t = 0.7)]
    pub het_threshold: f32,

    /// Minimum base prediction probability used to validate read-level observations during identity scoring.
    #[arg(short = 'b', long, default_value_t = 0.3)]
    pub base_threshold: f32,
}

fn parse_input_path(s: &str) -> Result<String, String> {
    if s == "-" {
        return Ok(s.to_string());
    }
    parse_abspath(s)
}

fn parse_abspath(s: &str) -> Result<String, String> {
    let path = Path::new(s);
    let abs_path = path
        .absolutize()
        .map_err(|e| format!("Path derivation failed: {}", e))?
        .to_path_buf();

    if abs_path.exists() {
        Ok(abs_path.to_string_lossy().to_string())
    } else {
        Err(format!("{:?} does not exist!", abs_path))
    }
}

fn freopen_stdout(path: &str) -> IoResult<()> {
    let path = Path::new(path).absolutize()?.to_path_buf();
    
    if path.exists() {
        return Err(Error::new(
            ErrorKind::AlreadyExists,
            format!("{path:?} already exists!"),
        ));
    }

    let mode = CString::new("w")?;
    let c_path = CString::new(path.as_os_str().as_bytes())?;

    if unsafe { libc::freopen(c_path.as_ptr(), mode.as_ptr(), stdout()) }.is_null() {
        return Err(Error::other(format!("Failed to freopen: {path:?}")));
    }
    Ok(())
}
