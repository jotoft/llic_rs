use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use llic::{LlicContext, pgm::Pgm};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 4 || args.len() > 6 {
        eprintln!("USAGE: {} [c | d] infile outfile [quality] [fast]", args[0]);
        eprintln!("\tc or d: Compress or Decompress.");
        eprintln!("\tquality [0...4]: Quality level. 0 is lossless, 4 is low quality.");
        eprintln!("\tfast: Fast mode off or on.");
        return Ok(());
    }
    
    let mode = &args[1];
    let infile = &args[2];
    let outfile = &args[3];
    
    let compress = mode == "c";
    let decompress = mode == "d";
    
    if !compress && !decompress {
        eprintln!("ERROR: Either compression or decompression must be specified.");
        return Ok(());
    }
    
    if compress {
        eprintln!("ERROR: Compression not yet implemented in Rust version.");
        return Ok(());
    } else {
        // Read compressed file header
        let file = File::open(infile)?;
        let mut reader = BufReader::new(file);
        
        let mut line = String::new();
        reader.read_line(&mut line)?;
        let dims: Vec<u32> = line.trim()
            .split_whitespace()
            .map(|s| s.parse())
            .collect::<Result<Vec<_>, _>>()?;
        
        if dims.len() != 2 {
            eprintln!("ERROR: Invalid compressed file format.");
            return Ok(());
        }
        
        let (width, height) = (dims[0], dims[1]);
        println!("Compressed image size: {} x {}", width, height);
        
        line.clear();
        reader.read_line(&mut line)?;
        let compressed_size: usize = line.trim().parse()?;
        
        // Read compressed data
        let mut compressed_data = vec![0u8; compressed_size];
        reader.read_exact(&mut compressed_data)?;
        
        // Initialize decompressor
        let context = LlicContext::new(width, height, width, None)?;
        
        // Decompress
        let mut output_image = Pgm::new(width, height);
        let (quality, mode) = context.decompress_gray8(
            &compressed_data,
            output_image.data_mut()
        )?;
        
        println!("Decompressed with quality: {:?}, mode: {:?}", quality, mode);
        
        // Save output
        output_image.save(outfile, true)?;
        println!("Saved decompressed image to: {}", outfile);
    }
    
    Ok(())
}