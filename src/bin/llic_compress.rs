use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use clap::{Parser, Subcommand};
use image::{GrayImage, ImageReader};
use llic::{pgm::Pgm, LlicContext, Mode, Quality};

/// LLIC container format:
/// - Magic: "LLIC" (4 bytes)
/// - Width: u32 LE
/// - Height: u32 LE
/// - Compressed data (LLIC v3 format from compress_gray8)
const MAGIC: &[u8; 4] = b"LLIC";

#[derive(Parser)]
#[command(name = "llic")]
#[command(about = "LLIC lossless image compressor/decompressor", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Compress an image (PNG/PGM) to LLIC format
    #[command(visible_alias = "c")]
    Compress {
        /// Input image file (PNG or PGM)
        input: String,
        /// Output LLIC file
        output: String,
    },
    /// Decompress an LLIC file to an image (PNG/PGM based on extension)
    #[command(visible_alias = "d")]
    Decompress {
        /// Input LLIC file
        input: String,
        /// Output image file (PNG or PGM, determined by extension)
        output: String,
    },
}

fn load_grayscale_image(path: &str) -> Result<(u32, u32, Vec<u8>), Box<dyn std::error::Error>> {
    let path_lower = path.to_lowercase();

    if path_lower.ends_with(".pgm") {
        let pgm = Pgm::open(path)?;
        Ok((pgm.width(), pgm.height(), pgm.data().to_vec()))
    } else {
        // Use image crate for PNG and other formats
        let img = ImageReader::open(path)?.decode()?;
        let gray = img.to_luma8();
        let (width, height) = gray.dimensions();
        Ok((width, height, gray.into_raw()))
    }
}

fn save_grayscale_image(
    path: &str,
    width: u32,
    height: u32,
    data: &[u8],
) -> Result<(), Box<dyn std::error::Error>> {
    let path_lower = path.to_lowercase();

    if path_lower.ends_with(".pgm") {
        let mut pgm = Pgm::new(width, height);
        pgm.data_mut().copy_from_slice(data);
        pgm.save(path, true)?;
    } else if path_lower.ends_with(".png") {
        let img = GrayImage::from_raw(width, height, data.to_vec())
            .ok_or("Failed to create image from data")?;
        img.save(path)?;
    } else {
        return Err(format!("Unsupported output format: {}. Use .png or .pgm", path).into());
    }

    Ok(())
}

fn compress(input: &str, output: &str) -> Result<(), Box<dyn std::error::Error>> {
    if !Path::new(input).exists() {
        return Err(format!("Input file not found: {}", input).into());
    }

    // Load input image
    let (width, height, image_data) = load_grayscale_image(input)?;
    println!(
        "Loaded image: {}x{} ({} bytes)",
        width,
        height,
        image_data.len()
    );

    // Create compression context
    let context = LlicContext::new(width, height, width, None)?;

    // Allocate output buffer
    let max_size = context.compressed_buffer_size();
    let mut compressed = vec![0u8; max_size];

    // Compress
    let compressed_size = context.compress_gray8(
        &image_data,
        Quality::Lossless,
        Mode::Default,
        &mut compressed,
    )?;
    compressed.truncate(compressed_size);

    let ratio = compressed_size as f64 / image_data.len() as f64;
    let savings = (1.0 - ratio) * 100.0;
    println!(
        "Compressed: {} -> {} bytes ({:.1}x, {:.1}% smaller)",
        image_data.len(),
        compressed_size,
        1.0 / ratio,
        savings
    );

    // Write container format
    let mut file = File::create(output)?;
    file.write_all(MAGIC)?;
    file.write_all(&width.to_le_bytes())?;
    file.write_all(&height.to_le_bytes())?;
    file.write_all(&compressed)?;

    println!("Saved to: {}", output);
    Ok(())
}

fn decompress(input: &str, output: &str) -> Result<(), Box<dyn std::error::Error>> {
    if !Path::new(input).exists() {
        return Err(format!("Input file not found: {}", input).into());
    }

    // Read container
    let mut file = File::open(input)?;
    let mut data = Vec::new();
    file.read_to_end(&mut data)?;

    // Parse header
    if data.len() < 12 {
        return Err("File too small to be valid LLIC container".into());
    }

    if &data[0..4] != MAGIC {
        return Err("Invalid LLIC file (bad magic)".into());
    }

    let width = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
    let height = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
    let compressed_data = &data[12..];

    println!(
        "LLIC container: {}x{}, {} bytes compressed",
        width,
        height,
        compressed_data.len()
    );

    // Create decompression context
    let context = LlicContext::new(width, height, width, None)?;

    // Decompress
    let mut image_data = vec![0u8; (width * height) as usize];
    let (quality, mode) = context.decompress_gray8(compressed_data, &mut image_data)?;

    println!("Decompressed: quality={:?}, mode={:?}", quality, mode);

    // Save output based on extension
    save_grayscale_image(output, width, height, &image_data)?;
    println!("Saved to: {}", output);

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Compress { input, output } => compress(&input, &output)?,
        Commands::Decompress { input, output } => decompress(&input, &output)?,
    }

    Ok(())
}
