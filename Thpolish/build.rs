use std::env;
use std::fs;
use std::process::Command;

const REQUIRED_TORCH_VERSION: &str = "2.10";
fn config_torch_env() {
    if env::var("LIBTORCH_USE_PYTORCH").unwrap_or_else(|_| "0".to_string()) != "1" {
        println!("cargo:warning=LIBTORCH_USE_PYTORCH is not 1. Skipping Python PyTorch validation.");
        return;
    }

    let output = Command::new("python3")
        .args([
            "-c",
            "import torch; import os; print(torch.__version__); print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"
        ])
        .output();

    match output {
        Ok(output) if output.status.success() => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let mut lines = stdout.lines();

            if let (Some(raw_version), Some(lib_path)) = (lines.next(), lines.next()) {
                let clean_version = raw_version.split('+').next().unwrap_or(raw_version).trim();
                if !clean_version.starts_with(REQUIRED_TORCH_VERSION) {
                    panic!(
                        "\n\n\
                        [FATAL CONFIGURATION ERROR]\n\
                        Version mismatch detected between Rust `tch` crate and Python PyTorch environment.\n\
                        - Required PyTorch version: {}\n\
                        - Detected PyTorch version: {} (Raw: {})\n\
                        \n\
                        [REMEDIATION]\n\
                        Option A (Upgrade Python environment):\n\
                            pip3 install torch=={}\n\
                        \n\
                        Option B (Downgrade Rust `tch` crate):\n\
                            Modify Cargo.toml to match your current PyTorch version ({})\n\
                        \n\
                        see https://github.com/LaurentMazare/tch-rs\n",
                        REQUIRED_TORCH_VERSION, clean_version, raw_version, REQUIRED_TORCH_VERSION, clean_version
                    );
                }

                println!("cargo:warning=Strict validation passed: PyTorch {} found at {}", clean_version, lib_path);
                println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_path);
            } else {
                panic!("Failed to parse Python output. Expected 2 lines (version and lib_path).");
            }
        }
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            panic!("Python script execution failed. Is PyTorch properly installed? Stderr: {}", stderr);
        }
        Err(e) => {
            panic!("Failed to invoke `python3` command. Please ensure it is in the system PATH. Error: {}", e);
        }
    }
}

fn set_git_version(){
    let version = env::var("CARGO_PKG_VERSION").unwrap();

    let child = Command::new("git").args(["describe", "--always"]).output();
    match child {
        Ok(child) if child.status.success() => {
            let buf = String::from_utf8(child.stdout).expect("failed to read stdout");
            println!("cargo:rustc-env=VERSION={version}-{buf}");
        }
        _ => {
            eprintln!("`git describe` failed, using CARGO_PKG_VERSION version");
            println!("cargo:rustc-env=VERSION={version}");
        }
    }
}

fn set_update_checker() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let git_config = format!("{}/.git/config", manifest_dir);

    let (mut owner, mut repo) = ("unknown".to_string(), "unknown".to_string());

    if let Ok(content) = fs::read_to_string(&git_config) {
        if let Some(line) = content.lines().find(|l| l.trim().starts_with("url =")) {
            let url = line.trim()["url =".len()..].trim();

            // 1. git@github.com:owner/repo.git
            // 2. https://github.com/owner/repo.git
            // 3. git://github.com/owner/repo.git
            let repo_part = if let Some(idx) = url.find("github.com") {
                &url[idx + "github.com/".len()..]
            }else {
                ""
            };

            let repo_part = repo_part.trim_end_matches(".git");

            let mut parts = repo_part.splitn(2, '/');
            if let (Some(o), Some(r)) = (parts.next(), parts.next()) {
                if !o.is_empty() && !r.is_empty() {
                    owner = o.to_string();
                    repo = r.to_string();
                }
            }
        }
    }
    println!("cargo:rustc-env=GIT_OWNER={owner}");
    println!("cargo:rustc-env=GIT_REPO={repo}");
}

fn main() {
    set_git_version();
    config_torch_env();
    set_update_checker();
}
