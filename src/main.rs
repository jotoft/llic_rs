use std::env;
use std::fs;
use std::path::Path;
use std::process;

use llic::{decode_file, encode_file, CompressionMode, CompressionQuality};

fn print_usage() {
    eprintln!("Usage:");
    eprintln!("  llic_rs2 decode <input.llic> <output.pgm>");
    eprintln!("  llic_rs2 encode <input.pgm> <output.llic> [quality] [mode]");
    eprintln!();
    eprintln!("Quality options: lossless (default), very_high, high, medium, low");
    eprintln!("Mode options: default (default), fast");
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 4 {
        print_usage();
        process::exit(1);
    }

    let command = &args[1];
    let input_path = &args[2];
    let output_path = &args[3];

    // Create output directory if it doesn't exist
    if let Some(parent) = Path::new(output_path).parent() {
        if !parent.exists() {
            fs::create_dir_all(parent).unwrap_or_else(|e| {
                eprintln!("Error creating output directory: {}", e);
                process::exit(1);
            });
        }
    }

    match command.as_str() {
        "decode" => match decode_file(input_path, output_path) {
            Ok(_) => {}
            Err(e) => {
                eprintln!("Error decoding file: {}", e);
                process::exit(1);
            }
        },
        "encode" => {
            let quality = if args.len() > 4 {
                match args[4].as_str() {
                    "lossless" => CompressionQuality::Lossless,
                    "very_high" => CompressionQuality::VeryHigh,
                    "high" => CompressionQuality::High,
                    "medium" => CompressionQuality::Medium,
                    "low" => CompressionQuality::Low,
                    _ => {
                        eprintln!("Invalid quality: {}", args[4]);
                        print_usage();
                        process::exit(1);
                    }
                }
            } else {
                CompressionQuality::Lossless
            };

            let mode = if args.len() > 5 {
                match args[5].as_str() {
                    "default" => CompressionMode::Default,
                    "fast" => CompressionMode::Fast,
                    _ => {
                        eprintln!("Invalid mode: {}", args[5]);
                        print_usage();
                        process::exit(1);
                    }
                }
            } else {
                CompressionMode::Default
            };

            match encode_file(input_path, output_path, quality, mode) {
                Ok(_) => {}
                Err(e) => {
                    eprintln!("Error encoding file: {}", e);
                    process::exit(1);
                }
            }
        }
        _ => {
            eprintln!("Invalid command: {}", command);
            print_usage();
            process::exit(1);
        }
    }
}
