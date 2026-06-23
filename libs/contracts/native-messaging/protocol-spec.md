# Ssense Native Messaging Protocol Specification

## Overview
Communication between Chrome Extension (TypeScript) and Native Daemon (Rust) via Chrome's Native Messaging API.

## Binary Framing
All messages use the same framing format:
[4 bytes: uint32 little-endian message length] [N bytes: JSON payload]

Maximum message size: 10 MB (enforced by daemon).

## Message Types

### Requests (Extension → Daemon)

| Type | Description |
|------|-------------|
| `AUDIT_POLICY` | Submit privacy policy text for DPDP audit |
| `CHAT` | Conversational query about a site's compliance |
| `GET_TRUST_SCORE` | Retrieve cached trust score for a domain |
| `HEALTH_CHECK` | Verify daemon is alive and model loaded |

### Responses (Daemon → Extension)

| Type | Description |
|------|-------------|
| `AUDIT_POLICY_RESULT` | Full DPDP audit report |
| `CHAT_RESULT` | Natural language response |
| `TRUST_SCORE_RESULT` | Cached score (or null) |
| `HEALTH_CHECK_RESULT` | Daemon status |
| `ERROR` | Error with message |

## Type Synchronization
- TypeScript types: `apps/extension/src/types/native-protocol.ts`
- Rust types: `apps/native-daemon/src/messaging/protocol.rs`

**These MUST be manually kept in sync.** Any change to one requires a matching change to the other.
