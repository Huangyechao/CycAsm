use anyhow::{Context, Result};
use log::info;
use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

mod mlib;
use mlib::{FileIO, resource_str, update_checker};

mod option;
use option::{Commands, PolishArgs};

mod filter;
use filter::run_filter;

mod depth;
use depth::run_depth;

mod encode;
use encode::run_encode;

mod predict;
use predict::run_predict;

mod correct;
use correct::run_correct;

fn main() -> Result<()> {
    let args = PolishArgs::parse_with_redirect_io(true)?;
    match args.command {
        Commands::Filter(args) => {
            run_filter(args).context("Filter subcommand execution terminated with an error")?;
        }
        Commands::Depth(args) => {
            run_depth(args).context("Depth subcommand execution terminated with an error")?;
        }
        Commands::Encode(args) => {
            run_encode(args).context("Encode subcommand execution terminated with an error")?;
        }
        Commands::Predict(args) => {
            run_predict(args).context("Predict subcommand execution terminated with an error")?;
        }
        Commands::Correct(args) => {
            run_correct(args).context("Correct subcommand execution terminated with an error")?;
        }
    }

    info!("\n{}", resource_str());
    Ok(())
}
