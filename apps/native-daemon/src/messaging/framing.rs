use byteorder::{ByteOrder, LittleEndian};
use thiserror::Error;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

const MAX_MESSAGE_SIZE_BYTES: usize = 10 * 1024 * 1024; // 10MB limit

#[derive(Error, Debug)]
pub enum FramingError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Message size {0} bytes exceeds maximum limit of 10MB")]
    MessageTooLarge(usize),
    #[error("Unexpected end of file (Chrome disconnected)")]
    UnexpectedEof,
}

pub async fn read_message<R>(reader: &mut R) -> Result<Vec<u8>, FramingError>
where
    R: AsyncReadExt + Unpin,
{
    let mut len_buf = [0u8; 4];
    
    match reader.read_exact(&mut len_buf).await {
        Ok(_) => (),
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Err(FramingError::UnexpectedEof),
        Err(e) => return Err(FramingError::Io(e)),
    }

    // CRITICAL FIX: Strictly LittleEndian to match Chromium's actual C++ implementation
    let len = LittleEndian::read_u32(&len_buf) as usize;

    if len > MAX_MESSAGE_SIZE_BYTES {
        return Err(FramingError::MessageTooLarge(len));
    }

    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf).await?;

    Ok(buf)
}

pub async fn write_message<W>(writer: &mut W, data: &[u8]) -> Result<(), FramingError>
where
    W: AsyncWriteExt + Unpin,
{
    // Check usize length BEFORE casting to prevent u32 wrap-around attacks
    if data.len() > MAX_MESSAGE_SIZE_BYTES {
        return Err(FramingError::MessageTooLarge(data.len()));
    }

    let len = data.len() as u32;
    let mut len_buf = [0u8; 4];
    
    // CRITICAL FIX: Strictly LittleEndian
    LittleEndian::write_u32(&mut len_buf, len);

    writer.write_all(&len_buf).await?;
    writer.write_all(data).await?;
    writer.flush().await?;

    Ok(())
}