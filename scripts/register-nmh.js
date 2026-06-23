// scripts/register-nmh.js
const { writeFileSync, mkdirSync, existsSync } = require('fs');
const { join, resolve } = require('path');
const { homedir, platform } = require('os');
const { execSync } = require('child_process');

const EXT_ID = process.argv[2];

if (!EXT_ID) {
  console.error('❌ Usage: node scripts/register-nmh.js <YOUR_EXTENSION_ID>');
  console.error('   Get your Extension ID from chrome://extensions (Enable Developer Mode)');
  process.exit(1);
}

// Resolve the absolute path to the compiled Rust binary
let binPath = resolve(__dirname, '../apps/native-daemon/target/release/ssense-daemon');
const osPlatform = platform();

if (osPlatform === 'win32') {
  binPath += '.exe';
  // 🚀 SOTA FIX: DO NOT manually escape backslashes. 
  // JSON.stringify automatically handles JSON specification escaping.
}

const checkPath = osPlatform === 'win32' ? binPath : binPath;
if (!existsSync(checkPath)) {
  console.error(`❌ Rust binary not found at ${checkPath}. Run 'make build-daemon' first.`);
  process.exit(1);
}

const manifest = {
  name: 'com.ssense.daemon',
  description: 'Ssense DPDP Edge AI',
  path: binPath,
  type: 'stdio',
  allowed_origins: [`chrome-extension://${EXT_ID}/`]
};

try {
  if (osPlatform === 'win32') {
    // Write JSON to a safe, space-free location on Windows
    const appData = process.env.APPDATA || join(homedir(), 'AppData', 'Roaming');
    const ssenseDir = join(appData, 'Ssense');
    if (!existsSync(ssenseDir)) mkdirSync(ssenseDir, { recursive: true });
    
    const jsonPath = join(ssenseDir, 'com.ssense.daemon.json');
    writeFileSync(jsonPath, JSON.stringify(manifest, null, 2));
    
    // Register in Windows Registry
    execSync(`reg add "HKCU\\Software\\Google\\Chrome\\NativeMessagingHosts\\com.ssense.daemon" /ve /t REG_SZ /d "${jsonPath}" /f`);
    console.log('✅ Registered Native Messaging Host in Windows Registry.');
  } else {
    let dir = osPlatform === 'darwin' 
      ? join(homedir(), 'Library/Application Support/Google/Chrome/NativeMessagingHosts')
      : join(homedir(), '.config/google-chrome/NativeMessagingHosts');
    
    if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
    writeFileSync(join(dir, 'com.ssense.daemon.json'), JSON.stringify(manifest, null, 2));
    console.log(`✅ Registered Native Messaging Host at ${dir}`);
  }
  console.log(`🔗 Linked to Extension ID: ${EXT_ID}`);
} catch (e) {
  console.error('❌ Failed to register NMH:', e.message);
  process.exit(1);
}