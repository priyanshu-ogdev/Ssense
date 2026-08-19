use byteorder::{ByteOrder, LittleEndian};
use thiserror::Error;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::time::{timeout, Duration};
use tracing::{debug, warn};

// ─────────────────────────────────────────────────────────────────
// SOTA SENSE: Asymmetric Chrome Native Messaging Limits
// ─────────────────────────────────────────────────────────────────
// Chrome allows up to 4GB incoming, but we restrict to 10MB to prevent 
// RAM exhaustion attacks from compromised extensions.
const MAX_INCOMING_SIZE_BYTES: usize = 10 * 1024 * 1024; 

// Chrome STRICTLY limits native host outbound messages to 1MB.
// Exceeding this causes Chrome to instantly kill the daemon process.
const MAX_OUTGOING_SIZE_BYTES: usize = 1024 * 1024; 

const IPC_PAYLOAD_TIMEOUT_SECS: u64 = 30; 
const IPC_WRITE_TIMEOUT_SECS: u64 = 10; 

#[derive(Error, Debug)]
pub enum FramingError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Incoming message size {0} bytes exceeds maximum limit of 10MB")]
    IncomingTooLarge(usize),
    
    #[error("Outgoing message size {0} bytes exceeds Chrome's strict 1MB limit. Connection saved.")]
    OutgoingTooLarge(usize),
    
    #[error("Unexpected end of file (Chrome closed the extension)")]
    UnexpectedEof,
    
    #[error("IPC read timeout: Chrome sent header but stalled on payload")]
    ReadTimeout,
    
    #[error("IPC write timeout: Chrome OS buffer is full or suspended")]
    WriteTimeout,
}

pub async fn read_message<R>(reader: &mut R) -> Result<Vec<u8>, FramingError>
where
    R: AsyncReadExt + Unpin,
{
    let mut len_buf = [0u8; 4];
    
    // 🛡️ NO TIMEOUT HERE. We wait peacefully for the user to trigger an action in Chrome.
    match reader.read_exact(&mut len_buf).await {
        Ok(_) => (),
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
            debug!("Chrome Native Messaging host disconnected gracefully (EOF).");
            return Err(FramingError::UnexpectedEof);
        }
        Err(e) => return Err(FramingError::Io(e)),
    }

    // Chrome sends the 32-bit message length in native byte order (Little Endian on x86/ARM)
    let len = LittleEndian::read_u32(&len_buf) as usize;

    if len > MAX_INCOMING_SIZE_BYTES {
        warn!("🚨 Blocked incoming IPC message of size {} (Exceeds 10MB constraint)", len);
        return Err(FramingError::IncomingTooLarge(len));
    }

    let mut buf = vec![0u8; len];
    
    // 🛡️ Timeout applied ONLY to the payload to prevent hanging if Chrome crashes mid-write.
    let read_payload = timeout(
        Duration::from_secs(IPC_PAYLOAD_TIMEOUT_SECS),
        reader.read_exact(&mut buf)
    ).await;

    match read_payload {
        Ok(Ok(_)) => Ok(buf),
        Ok(Err(e)) => Err(FramingError::Io(e)),
        Err(_) => {
            warn!("🚨 IPC Read Timeout: Header received but payload stalled.");
            Err(FramingError::ReadTimeout)
        }
    }
}

pub async fn write_message<W>(writer: &mut W, data: &[u8]) -> Result<(), FramingError>
where
    W: AsyncWriteExt + Unpin,
{
    // 🛡️ Chrome Assassination Prevention
    if data.len() > MAX_OUTGOING_SIZE_BYTES {
        warn!("🚨 Prevented Chrome crash! Attempted to send {} bytes (>1MB limit).", data.len());
        return Err(FramingError::OutgoingTooLarge(data.len()));
    }

    let len = data.len() as u32;
    let mut len_buf = [0u8; 4];
    LittleEndian::write_u32(&mut len_buf, len);

    // 🛡️ Bilateral Write Timeouts. Prevents our async loop from locking if the OS pipe fills.
    let write_op = timeout(Duration::from_secs(IPC_WRITE_TIMEOUT_SECS), async {
        writer.write_all(&len_buf).await?;
        writer.write_all(data).await?;
        writer.flush().await?;
        Ok::<(), std::io::Error>(())
    }).await;

    match write_op {
        Ok(Ok(_)) => Ok(()),
        Ok(Err(e)) => Err(FramingError::Io(e)),
        Err(_) => {
            warn!("🚨 IPC Write Timeout: OS pipe full or Chrome suspended.");
            Err(FramingError::WriteTimeout)
        }
    }
}