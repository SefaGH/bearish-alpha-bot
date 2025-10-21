# WebSocket Connection Flow Diagram

## Before (Old Behavior - BROKEN)
```
┌─────────────────────────────────────────────────────────────┐
│ Trading Loop Starts                                         │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ Start WebSocket Streams                                     │
│ await ws_optimizer.initialize_websockets()                  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ Log: "WebSocket: ⚠️ STREAMS STARTED (waiting...)"          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │ ❌ INFINITE WAIT      │
         │ No timeout            │
         │ No retry              │
         │ Never fails           │
         │ FREEZES FOREVER!      │
         └───────────────────────┘
```

## After (New Behavior - FIXED)
```
┌─────────────────────────────────────────────────────────────┐
│ Trading Loop Starts                                         │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ Establish WebSocket Connection with Retry                  │
│ await _establish_websocket_connection(max_retries=3)       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
        ╔════════════════════════════╗
        ║ Retry Loop (3 attempts)    ║
        ╚════════════╦═══════════════╝
                     │
                     │ ┌─── Attempt 1 ────┐
                     ├─┤                   │
                     │ └──────────────────┘
                     │         │
                     │         ▼
                     │ ┌─────────────────────────────────────┐
                     │ │ Start WebSocket Streams             │
                     │ └──────────────┬──────────────────────┘
                     │                │
                     │                ▼
                     │ ┌─────────────────────────────────────┐
                     │ │ Wait for Connection (timeout: 30s)  │
                     │ │ Polls every 1 second                │
                     │ │ Logs status every 5 seconds         │
                     │ └──────────────┬──────────────────────┘
                     │                │
                     │                ├─── Connected? ───┐
                     │                │                   │
                     │           ✅ YES               ❌ NO
                     │                │                   │
                     │                ▼                   ▼
                     │    ┌───────────────────┐  ┌────────────────┐
                     │    │ SUCCESS!          │  │ TIMEOUT (30s)  │
                     │    │ Return True       │  │ Return False   │
                     │    └────────┬──────────┘  └────────┬───────┘
                     │             │                       │
                     │             │                       ▼
                     │             │          ┌──────────────────────┐
                     │             │          │ Stop Current Streams │
                     │             │          └──────────┬───────────┘
                     │             │                     │
                     │             │                     ▼
                     │             │          ┌──────────────────────┐
                     │             │          │ Wait (exponential)   │
                     │             │          │ Attempt 1: 5s        │
                     │             │          │ Attempt 2: 10s       │
                     │             │          │ Attempt 3: 15s       │
                     │             │          └──────────┬───────────┘
                     │             │                     │
                     │             │                     └──┐
                     │             │                        │
                     │ ┌─── Attempt 2 (if needed) ─────────┤
                     ├─┤                                    │
                     │ └──────────────────────────────────┬─┘
                     │                                    │
                     │ ┌─── Attempt 3 (if needed) ───────┤
                     ├─┤                                  │
                     │ └──────────────────────────────────┘
                     │
                     ▼
        ╔════════════════════════════╗
        ║ All Attempts Complete      ║
        ╚════════════╦═══════════════╝
                     │
                     ├─── Connected? ───┐
                     │                   │
                ✅ YES              ❌ NO
                     │                   │
                     ▼                   ▼
    ┌──────────────────────────┐  ┌──────────────────────────┐
    │ ✅ WebSocket Connected   │  │ ⚠️ Connection Failed     │
    │ Start Trading            │  │ Fall Back to REST API    │
    │ (Real-time data)         │  │ Continue Trading         │
    └──────────────────────────┘  │ (Degraded mode)          │
                                   └──────────────────────────┘
                                             │
                                             ▼
                                   ┌──────────────────────────┐
                                   │ Send Telegram Alert      │
                                   │ "WebSocket unavailable"  │
                                   └──────────────────────────┘
```

## Connection Wait Flow (Detail)
```
┌─────────────────────────────────────────────────────────────┐
│ _wait_for_websocket_connection(timeout=30)                  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ start_time = now()
                     │
                     ▼
        ╔════════════════════════════╗
        ║ Poll Loop (1s interval)    ║
        ╚════════════╦═══════════════╝
                     │
                     ├──► elapsed = now() - start_time
                     │
                     ├──► Check: elapsed >= timeout?
                     │    │
                     │    ├─── YES ──► ❌ Return False (Timeout)
                     │    │
                     │    └─── NO ───► Continue
                     │
                     ├──► status = get_connection_status()
                     │
                     ├──► Check: status.connected?
                     │    │
                     │    ├─── YES ──► ✅ Return True (Connected!)
                     │    │
                     │    └─── NO ───► Continue
                     │
                     ├──► Check: status.error?
                     │    │
                     │    ├─── YES ──► ❌ Return False (Error)
                     │    │
                     │    └─── NO ───► Continue
                     │
                     ├──► Log status (every 5s)
                     │
                     ├──► await sleep(1 second)
                     │
                     └──► Loop back ◄─────────────┘
```

## Health Monitoring Flow
```
┌─────────────────────────────────────────────────────────────┐
│ _monitor_websocket_health()                                 │
│ Runs every 60 seconds during trading                        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
        ╔════════════════════════════╗
        ║ Check Connection Status    ║
        ╚════════════╦═══════════════╝
                     │
                     ├──► status = get_connection_status()
                     │
                     ├──► Check: status.connected?
                     │    │
                     │    ├─── YES ──► ✅ Log "Healthy"
                     │    │            └─► Reset error counter
                     │    │                └─► Continue monitoring
                     │    │
                     │    └─── NO ───► ❌ Log "Not connected"
                     │                 │
                     │                 ▼
                     │    ┌──────────────────────────────┐
                     │    │ Increment error counter      │
                     │    │ (consecutive_errors++)       │
                     │    └──────────────┬───────────────┘
                     │                   │
                     │                   ├─── Check: errors >= 3?
                     │                   │    │
                     │                   │    ├─── YES ──► ⚠️ Fall back to REST API
                     │                   │    │            └─► Send Telegram alert
                     │                   │    │                └─► Stop monitoring
                     │                   │    │
                     │                   │    └─── NO ───► Attempt recovery
                     │                   │                 │
                     │                   │                 ▼
                     │                   │    ┌──────────────────────────────┐
                     │                   │    │ Call _restart_websockets()   │
                     │                   │    │ with exponential backoff     │
                     │                   │    └──────────────────────────────┘
                     │                   │
                     │                   └──► Continue monitoring
                     │
                     └──► await sleep(60 seconds) ◄─────┘
```

## Retry Timing Diagram
```
Time (seconds)
0    5    10   15   20   25   30   35   40   45   50   55   60   65   70   75   80
│────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────│

Attempt 1:
├────────────────────────── Try for 30s ──────────────────────────┤
│                                                                  │
│  [Polling every 1s]                                             │
│  [Logging every 5s]                                             │
│                                                                  │
└─► ❌ TIMEOUT (30s elapsed)                                      

Wait 5s (exponential backoff: 5 * 1)
     ├────┤

Attempt 2:
          ├────────────────────────── Try for 30s ──────────────────────────┤
          │                                                                  │
          │  [Polling every 1s]                                             │
          │  [Logging every 5s]                                             │
          │                                                                  │
          └─► ❌ TIMEOUT (30s elapsed)                                      

          Wait 10s (exponential backoff: 5 * 2)
               ├─────────────┤

Attempt 3:
                             ├────────────────────────── Try for 30s ───────►
                             │
                             │  [Polling every 1s]
                             │  [Logging every 5s]
                             │
                             └─► ❌ TIMEOUT or ✅ SUCCESS

Total max time: 30s + 5s + 30s + 10s + 30s = 105 seconds
With overhead: ~135 seconds maximum
```

## Key Improvements

### Old Behavior
- ❌ No timeout → Infinite wait
- ❌ No retry → Single point of failure
- ❌ No monitoring → Can't detect issues
- ❌ No fallback → Complete failure

### New Behavior
- ✅ 30s timeout → Never hangs forever
- ✅ 3 retries with backoff → Handles transient issues
- ✅ Real-time monitoring → Detects issues immediately
- ✅ REST API fallback → Graceful degradation
- ✅ Clear logging → Easy debugging
- ✅ Telegram alerts → User notification
