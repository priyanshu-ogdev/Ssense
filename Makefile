.PHONY: all setup build-ext build-daemon clean test

# Default target
all: build-ext build-daemon

# 1. Unified Setup: Builds everything and prompts for NMH registration
setup: build-ext build-daemon
	@echo ""
	@printf "Enter your Chrome Extension ID (from chrome://extensions): " && read ext_id && node scripts/register-nmh.js $$ext_id

# 2. Build the TypeScript Extension
build-ext:
	@echo "📦 Building Chrome Extension..."
	cd apps/extension && npm install && npm run build
	@echo "✅ Extension built in apps/extension/dist"

# 3. Build the Rust Daemon (Release Mode)
build-daemon:
	@echo "🦀 Building Rust Native Daemon..."
	cargo build --release
	@echo "✅ Daemon built successfully."

# 4. Run all tests (Workspace level to include libs/rust-utils)
test:
	@echo "🧪 Running Rust Workspace Tests..."
	cargo test --workspace
	@echo "🧪 Running TypeScript Typechecks..."
	cd apps/extension && npx tsc --noEmit

# 5. Clean all artifacts
clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -rf apps/extension/dist
	cargo clean
	@echo "✅ Clean complete."