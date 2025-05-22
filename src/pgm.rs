use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use crate::{LlicError, Result};

pub struct Pgm {
    width: u32,
    height: u32,
    data: Vec<u8>,
}

impl Pgm {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            data: vec![0; (width * height) as usize],
        }
    }
    
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut line = String::new();
        
        reader.read_line(&mut line)?;
        if !line.starts_with("P5") && !line.starts_with("P2") {
            return Err(LlicError::UnsupportedFormat);
        }
        let binary = line.starts_with("P5");
        
        line.clear();
        loop {
            reader.read_line(&mut line)?;
            if !line.starts_with('#') {
                break;
            }
            line.clear();
        }
        
        let dims: Vec<u32> = line
            .trim()
            .split_whitespace()
            .map(|s| s.parse().map_err(|_| LlicError::InvalidData))
            .collect::<Result<Vec<u32>>>()?;
        
        if dims.len() != 2 {
            return Err(LlicError::InvalidData);
        }
        
        let (width, height) = (dims[0], dims[1]);
        
        line.clear();
        reader.read_line(&mut line)?;
        let max_val: u32 = line.trim().parse().map_err(|_| LlicError::InvalidData)?;
        
        if max_val != 255 {
            return Err(LlicError::UnsupportedFormat);
        }
        
        let mut data = vec![0u8; (width * height) as usize];
        
        if binary {
            use std::io::Read;
            reader.read_exact(&mut data)?;
        } else {
            let mut values = String::new();
            reader.read_to_string(&mut values)?;
            let values: Vec<u8> = values
                .split_whitespace()
                .map(|s| s.parse().map_err(|_| LlicError::InvalidData))
                .collect::<Result<Vec<u8>>>()?;
            
            if values.len() != data.len() {
                return Err(LlicError::InvalidData);
            }
            
            data.copy_from_slice(&values);
        }
        
        Ok(Self { width, height, data })
    }
    
    pub fn save<P: AsRef<Path>>(&self, path: P, binary: bool) -> Result<()> {
        let mut file = File::create(path)?;
        
        if binary {
            writeln!(file, "P5")?;
        } else {
            writeln!(file, "P2")?;
        }
        
        writeln!(file, "{} {}", self.width, self.height)?;
        writeln!(file, "255")?;
        
        if binary {
            file.write_all(&self.data)?;
        } else {
            for (i, &pixel) in self.data.iter().enumerate() {
                if i > 0 && i % 16 == 0 {
                    writeln!(file)?;
                }
                write!(file, "{} ", pixel)?;
            }
            writeln!(file)?;
        }
        
        Ok(())
    }
    
    pub fn width(&self) -> u32 {
        self.width
    }
    
    pub fn height(&self) -> u32 {
        self.height
    }
    
    pub fn data(&self) -> &[u8] {
        &self.data
    }
    
    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }
}