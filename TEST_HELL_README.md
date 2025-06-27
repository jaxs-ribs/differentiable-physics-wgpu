# Test Hell Session

An interactive terminal-based test runner that makes you witness each test before proceeding.

## Features

- **Welcome Screen**: ASCII art introduction with test count
- **One Test Per Screen**: Each test gets its own dedicated screen
- **Live Output Streaming**: Watch test output in real-time as it executes
- **Keyboard Navigation**:
  - `SPACE` - Run the current test
  - `N` - Skip to next test
  - `P` - Go to previous test
  - `R` - Re-run current test
  - `Q` - Quit at any time
  - `E` - Export results (from summary screen)
- **Visual Feedback**: Color-coded test status (running, passed, failed, skipped)
- **Summary Screen**: Final results with category breakdown
- **Export Function**: Save detailed results to timestamped file

## Running the Test Hell

```bash
cd physics_core/
python3 run_all_tests.py
```

## What to Expect

1. **Welcome Screen**: You'll be greeted with ASCII art and must press any key to begin
2. **Test Screens**: Each test shows:
   - Test number and total count
   - Category and description
   - Command being executed
   - Current status with duration
   - Live output in a bordered window
3. **Interactive Control**: You decide when to run each test
4. **Summary**: Final screen shows all results with pass/fail breakdown

## Test Categories

- **Core Tests**: Main CI suite with 7 comprehensive tests
- **Custom Operations**: C library, integration, demo, and benchmarks
- **Debugging**: Position corruption, NaN handling, empty contacts, JIT tests

## Export Format

Results can be exported to a timestamped text file containing:
- Session timing information
- Summary statistics
- Detailed results for each test including output

## Implementation Details

- Uses Python's `curses` library for terminal UI
- Streams test output in real-time using threading
- Preserves full test output for review
- Color-coded status indicators
- Responsive to terminal size

## Why "Hell Session"?

You can't just run all tests blindly - you must witness each one, acknowledge its existence, and consciously decide to proceed. It's a test runner that demands your attention.