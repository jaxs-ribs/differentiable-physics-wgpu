# Test Hell Session

An interactive terminal-based test runner that automatically runs tests while you watch.

## Features

- **Welcome Screen**: ASCII art introduction with test count
- **Auto-Running Tests**: Tests start automatically, no manual triggering needed
- **One Test Per Screen**: Each test gets its own dedicated screen with:
  - Test description and docstring preview
  - Live output streaming with color coding
  - Progress indicators and spinners
  - Duration tracking
- **Enhanced Output**: Tests now include verbose logging showing:
  - What's being tested and why
  - Step-by-step progress
  - Expected vs actual results
  - Performance metrics
- **Keyboard Navigation**:
  - `ENTER` - Continue to next test after completion
  - `P` - Go to previous test
  - `R` - Re-run current test
  - `Q` - Quit at any time
  - `E` - Export results (from summary screen)
- **Scrolling Controls**:
  - `↑` / `↓` - Scroll up/down one line
  - `PgUp` / `PgDn` - Scroll up/down one page
  - `Home` - Jump to top of output
  - `End` - Jump to bottom of output
  - Auto-scrolls to bottom when new output arrives
- **Visual Feedback**: 
  - Color-coded test status (running=blue, passed=green, failed=red)
  - Animated spinner for running tests
  - Progress bar showing approximate completion
  - Syntax highlighting for output (errors, success, warnings)
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

## What's New

- **Scrollable Output**: Full scrolling support with arrow keys, page up/down, home/end
- **Docstring Display**: Each test shows its purpose and description from the module docstring
- **Auto-Run**: Tests start automatically without manual triggering
- **Enhanced Logging**: All tests now include verbose output showing progress and details
- **Better Visual Feedback**: Scroll indicators, progress bars, and color-coded output

## Implementation Details

- Uses Python's `curses` library for terminal UI
- Streams test output in real-time using threading
- Preserves full test output for review with scrolling
- Color-coded status indicators
- Responsive to terminal size
- Extracts and displays module docstrings

## Why "Hell Session"?

You can't just run all tests blindly - you must witness each one, acknowledge its existence, and consciously decide to proceed. It's a test runner that demands your attention.