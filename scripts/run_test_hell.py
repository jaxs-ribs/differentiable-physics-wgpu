#!/usr/bin/env python3
"""Interactive test runner with terminal UI - the hell session."""

import curses
import subprocess
import sys
import time
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
import threading
import queue

# Add parent directory and tinygrad to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
tinygrad_path = os.path.join(parent_dir, "external", "tinygrad")
if os.path.exists(tinygrad_path):
    sys.path.insert(0, tinygrad_path)

@dataclass
class TestCase:
    name: str
    command: List[str]
    category: str
    description: str
    docstring: str = ""  # Module docstring from test file
    status: str = "pending"  # pending, running, passed, failed, skipped
    output: str = ""
    duration: float = 0.0
    error: str = ""

class InteractiveTestRunner:
    def __init__(self):
        self.tests = self._collect_tests()
        self.current_test = 0
        self.test_outputs = {}
        self.start_time = None
        self.end_time = None
        self.scroll_offset = 0  # For scrolling output
        
    def _extract_docstring(self, filepath: str) -> str:
        """Extract module docstring from a Python file."""
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
                # Skip shebang and empty lines
                start_idx = 0
                for i, line in enumerate(lines):
                    if line.strip() and not line.startswith('#!'):
                        start_idx = i
                        break
                
                # Look for docstring
                if start_idx < len(lines):
                    content = ''.join(lines[start_idx:])
                    if content.startswith('"""'):
                        end = content.find('"""', 3)
                        if end != -1:
                            return content[3:end].strip()
                    elif content.startswith("'''"):
                        end = content.find("'''", 3)
                        if end != -1:
                            return content[3:end].strip()
        except:
            pass
        return ""
        
    def _collect_tests(self) -> List[TestCase]:
        """Collect all tests from the codebase."""
        tests = []
        
        # Core CI Tests
        test = TestCase(
            name="Core CI Suite",
            command=["python3", "tests/run_ci.py"],
            category="Core Tests",
            description="Main CI test suite with 7 comprehensive tests"
        )
        test.docstring = self._extract_docstring("tests/run_ci.py")
        tests.append(test)
        
        # Custom Operations Tests
        custom_ops_tests = [
            ("C Library Tests", "tests/unit/custom_ops/test_c_library.py", 
             "Direct testing of physics C functions"),
            ("Integration Tests", "tests/unit/custom_ops/test_integration.py",
             "End-to-end custom op testing"),
            ("Basic Demo", "custom_ops/examples/basic_demo.py",
             "Demonstration of custom ops usage"),
            ("Performance Benchmark", "custom_ops/examples/benchmark.py",
             "Shows performance improvements")
        ]
        
        for name, path, desc in custom_ops_tests:
            test = TestCase(
                name=name,
                command=["python3", path],
                category="Custom Operations",
                description=desc
            )
            test.docstring = self._extract_docstring(path)
            tests.append(test)
        
        # Debugging Tests
        debug_tests = [
            ("Position Corruption", "tests/debugging/test_position_corruption.py",
             "Verifies position updates work correctly"),
            ("NaN Propagation", "tests/debugging/test_nan_propagation.py",
             "Tests NaN handling in physics"),
            ("Empty Contacts", "tests/debugging/test_empty_contacts_simple.py",
             "Tests solver with no collisions"),
            ("JIT Early Return", "tests/debugging/test_jit_early_return.py",
             "Tests conditional logic in JIT")
        ]
        
        for name, path, desc in debug_tests:
            test = TestCase(
                name=name,
                command=["python3", path],
                category="Debugging",
                description=desc
            )
            test.docstring = self._extract_docstring(path)
            tests.append(test)
        
        return tests
    
    def run_test(self, test: TestCase, output_queue: queue.Queue):
        """Run a single test and capture output."""
        test.status = "running"
        start_time = time.time()
        
        try:
            # Run the test
            process = subprocess.Popen(
                test.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Stream output line by line
            output_lines = []
            for line in iter(process.stdout.readline, ''):
                if line:
                    output_lines.append(line.rstrip())
                    output_queue.put(("output", line.rstrip()))
            
            process.wait()
            test.duration = time.time() - start_time
            test.output = "\n".join(output_lines)
            
            if process.returncode == 0:
                test.status = "passed"
            else:
                test.status = "failed"
                test.error = f"Exit code: {process.returncode}"
                
        except subprocess.TimeoutExpired:
            test.status = "failed"
            test.error = "Test timed out"
            test.duration = time.time() - start_time
        except Exception as e:
            test.status = "failed"
            test.error = str(e)
            test.duration = time.time() - start_time
    
    def draw_welcome_screen(self, stdscr):
        """Draw the welcome screen with ASCII art."""
        h, w = stdscr.getmaxyx()
        
        # ASCII art
        art = [
            "╔═══════════════════════════════════════════════════════════════╗",
            "║                                                               ║",
            "║  ████████╗███████╗███████╗████████╗    ██╗  ██╗███████╗██╗   ║",
            "║  ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝    ██║  ██║██╔════╝██║   ║",
            "║     ██║   █████╗  ███████╗   ██║       ███████║█████╗  ██║   ║",
            "║     ██║   ██╔══╝  ╚════██║   ██║       ██╔══██║██╔══╝  ██║   ║",
            "║     ██║   ███████╗███████║   ██║       ██║  ██║███████╗███████╗",
            "║     ╚═╝   ╚══════╝╚══════╝   ╚═╝       ╚═╝  ╚═╝╚══════╝╚══════╝",
            "║                                                               ║",
            "║               Welcome to the Test Hell Session                ║",
            "║                                                               ║",
            "╚═══════════════════════════════════════════════════════════════╝",
            "",
            "         You have entered the physics engine test gauntlet",
            "          Each test must be witnessed before proceeding",
            "",
            f"                    {len(self.tests)} tests await your judgment",
            "",
            "",
            "                    Press any key to begin..."
        ]
        
        # Center the art
        start_y = max(0, (h - len(art)) // 2)
        
        for i, line in enumerate(art):
            if start_y + i < h - 1:
                x = max(0, (w - len(line)) // 2)
                try:
                    if i < 12:  # Box lines
                        stdscr.addstr(start_y + i, x, line, curses.color_pair(2))
                    else:  # Text lines
                        stdscr.addstr(start_y + i, x, line, curses.color_pair(1))
                except:
                    pass
    
    def draw_test_screen(self, stdscr, test: TestCase, output_lines: List[str]):
        """Draw the test execution screen."""
        h, w = stdscr.getmaxyx()
        stdscr.clear()
        
        # Header
        header = f"╔{'═' * (w-2)}╗"
        stdscr.addstr(0, 0, header, curses.color_pair(2))
        
        title = f"Test {self.current_test + 1}/{len(self.tests)}: {test.name}"
        title_x = max(2, (w - len(title)) // 2)
        stdscr.addstr(1, title_x, title, curses.color_pair(3) | curses.A_BOLD)
        
        # Test info
        info_y = 3
        stdscr.addstr(info_y, 2, f"Category: {test.category}", curses.color_pair(1))
        stdscr.addstr(info_y + 1, 2, f"Description: {test.description}", curses.color_pair(1))
        
        # Show docstring if available
        current_y = info_y + 3
        if test.docstring:
            stdscr.addstr(current_y, 2, "Purpose:", curses.color_pair(3) | curses.A_BOLD)
            current_y += 1
            # Show first 4-5 lines of docstring
            doc_lines = test.docstring.split('\n')
            max_doc_lines = 5
            for i, line in enumerate(doc_lines[:max_doc_lines]):
                if current_y < info_y + 9 and i < max_doc_lines:
                    # Truncate long lines
                    max_doc_width = w - 6
                    display_line = line.strip()
                    if len(display_line) > max_doc_width:
                        display_line = display_line[:max_doc_width-3] + "..."
                    try:
                        if display_line:  # Only show non-empty lines
                            stdscr.addstr(current_y, 4, display_line[:max_doc_width], curses.color_pair(1) | curses.A_DIM)
                            current_y += 1
                    except:
                        pass
            if len(doc_lines) > max_doc_lines:
                stdscr.addstr(current_y, 4, "...", curses.color_pair(1) | curses.A_DIM)
                current_y += 1
        
        # Status with spinner for running tests
        status_y = current_y + 1
        status_str = f"Status: {test.status.upper()}"
        if test.status == "running":
            spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
            spin_idx = int(time.time() * 10) % len(spinner)
            status_str += f" {spinner[spin_idx]}"
        
        status_color = {
            "pending": 1,
            "running": 4,
            "passed": 5,
            "failed": 6,
            "skipped": 1
        }.get(test.status, 1)
        stdscr.addstr(status_y, 2, status_str, curses.color_pair(status_color) | curses.A_BOLD)
        
        if test.duration > 0:
            stdscr.addstr(status_y, 25, f"Duration: {test.duration:.2f}s", curses.color_pair(1))
        
        # Progress bar for running tests
        if test.status == "running" and test.duration > 0:
            progress_width = min(30, w - 45)
            progress = min(test.duration / 10.0, 1.0)  # Assume 10s max
            filled = int(progress * progress_width)
            bar = "[" + "█" * filled + "░" * (progress_width - filled) + "]"
            try:
                stdscr.addstr(status_y, 45, bar, curses.color_pair(4))
            except:
                pass
        
        # Output window
        output_y = status_y + 2
        output_height = h - output_y - 5  # Leave room for footer and scroll indicator
        
        if output_height > 2:
            # Draw output box
            stdscr.addstr(output_y, 1, f"╔{'═' * (w-4)}╗", curses.color_pair(2))
            for i in range(output_height - 2):
                stdscr.addstr(output_y + 1 + i, 1, "║", curses.color_pair(2))
                stdscr.addstr(output_y + 1 + i, w - 2, "║", curses.color_pair(2))
            stdscr.addstr(output_y + output_height - 1, 1, f"╚{'═' * (w-4)}╝", curses.color_pair(2))
            
            # Calculate visible lines with scroll offset
            total_lines = len(output_lines)
            visible_height = output_height - 2
            
            # Ensure scroll offset is valid
            max_scroll = max(0, total_lines - visible_height)
            self.scroll_offset = max(0, min(self.scroll_offset, max_scroll))
            
            # Get visible lines
            start_idx = self.scroll_offset
            end_idx = min(start_idx + visible_height, total_lines)
            visible_lines = output_lines[start_idx:end_idx]
            
            # Show output lines with color coding
            for i, line in enumerate(visible_lines):
                if i < visible_height:
                    try:
                        # Truncate long lines
                        max_width = w - 6
                        if len(line) > max_width:
                            line = line[:max_width-3] + "..."
                        
                        # Color code certain output
                        color = curses.color_pair(1)
                        if "PASSED" in line or "✓" in line or "passed" in line.lower():
                            color = curses.color_pair(5)
                        elif "FAILED" in line or "✗" in line or "Error" in line or "error" in line.lower():
                            color = curses.color_pair(6)
                        elif "WARNING" in line or "warning" in line.lower():
                            color = curses.color_pair(3)
                        elif "===" in line or "---" in line:
                            color = curses.color_pair(2)
                        
                        stdscr.addstr(output_y + 1 + i, 3, line[:max_width], color)
                    except:
                        pass
            
            # Show scroll indicator if needed
            if total_lines > visible_height:
                scroll_percent = int((self.scroll_offset / max_scroll) * 100) if max_scroll > 0 else 0
                scroll_info = f"Lines {start_idx+1}-{end_idx}/{total_lines} ({scroll_percent}%)"
                try:
                    stdscr.addstr(output_y + output_height - 1, w - len(scroll_info) - 3, scroll_info, curses.color_pair(2))
                except:
                    pass
        
        # Footer with scroll instructions
        footer_y = h - 2
        if test.status == "pending":
            footer = "Starting test automatically..."
        elif test.status == "running":
            footer = "Test running... ↑↓/PgUp/PgDn to scroll, Q to quit"
        elif test.status in ["passed", "failed"]:
            footer = "ENTER: next, P: previous, R: re-run, ↑↓/PgUp/PgDn: scroll, Q: quit"
        else:
            footer = "ENTER: next, P: previous, ↑↓/PgUp/PgDn: scroll"
        
        # Truncate footer if too long
        if len(footer) > w - 4:
            footer = footer[:w-7] + "..."
            
        footer_x = max(0, (w - len(footer)) // 2)
        stdscr.addstr(footer_y, footer_x, footer, curses.color_pair(1))
    
    def draw_summary_screen(self, stdscr):
        """Draw the final summary screen."""
        h, w = stdscr.getmaxyx()
        stdscr.clear()
        
        # Calculate stats
        passed = sum(1 for t in self.tests if t.status == "passed")
        failed = sum(1 for t in self.tests if t.status == "failed")
        skipped = sum(1 for t in self.tests if t.status == "skipped")
        total_duration = sum(t.duration for t in self.tests)
        
        # Header
        art = [
            "╔═══════════════════════════════════════════════════════════════╗",
            "║                      TEST HELL COMPLETE                       ║",
            "╚═══════════════════════════════════════════════════════════════╝"
        ]
        
        start_y = 2
        for i, line in enumerate(art):
            x = max(0, (w - len(line)) // 2)
            stdscr.addstr(start_y + i, x, line, curses.color_pair(2))
        
        # Stats
        stats_y = start_y + 5
        stats = [
            f"Total Tests: {len(self.tests)}",
            f"Passed: {passed}",
            f"Failed: {failed}",
            f"Skipped: {skipped}",
            f"Total Duration: {total_duration:.2f}s"
        ]
        
        for i, stat in enumerate(stats):
            x = max(0, (w - len(stat)) // 2)
            color = 5 if "Passed" in stat else 6 if "Failed" in stat else 1
            stdscr.addstr(stats_y + i, x, stat, curses.color_pair(color))
        
        # Results by category
        results_y = stats_y + 7
        stdscr.addstr(results_y, 2, "Results by Category:", curses.color_pair(1) | curses.A_BOLD)
        
        categories = {}
        for test in self.tests:
            if test.category not in categories:
                categories[test.category] = []
            categories[test.category].append(test)
        
        y = results_y + 2
        for category, tests in categories.items():
            if y < h - 4:
                stdscr.addstr(y, 4, f"{category}:", curses.color_pair(1))
                y += 1
                for test in tests:
                    if y < h - 4:
                        status_char = "✓" if test.status == "passed" else "✗" if test.status == "failed" else "-"
                        color = 5 if test.status == "passed" else 6 if test.status == "failed" else 1
                        stdscr.addstr(y, 6, f"{status_char} {test.name}", curses.color_pair(color))
                        y += 1
                y += 1
        
        # Footer
        footer = "Press Q to quit, E to export results"
        footer_x = max(0, (w - len(footer)) // 2)
        stdscr.addstr(h - 2, footer_x, footer, curses.color_pair(1))
    
    def run(self, stdscr):
        """Main UI loop."""
        # Initialize colors
        curses.start_color()
        curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLACK)    # Normal
        curses.init_pair(2, curses.COLOR_CYAN, curses.COLOR_BLACK)     # Box
        curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)   # Title
        curses.init_pair(4, curses.COLOR_BLUE, curses.COLOR_BLACK)     # Running
        curses.init_pair(5, curses.COLOR_GREEN, curses.COLOR_BLACK)    # Passed
        curses.init_pair(6, curses.COLOR_RED, curses.COLOR_BLACK)      # Failed
        
        # Show welcome screen
        stdscr.clear()
        self.draw_welcome_screen(stdscr)
        stdscr.refresh()
        stdscr.getch()  # Wait for any key
        
        # Main test loop
        self.start_time = time.time()
        output_queue = queue.Queue()
        output_lines = []
        
        while self.current_test < len(self.tests):
            test = self.tests[self.current_test]
            output_lines = []
            test_thread = None
            self.scroll_offset = 0  # Reset scroll for each test
            
            # Auto-start test if pending
            if test.status == "pending":
                output_lines.append("=== Starting test automatically ===")
                test_thread = threading.Thread(
                    target=self.run_test,
                    args=(test, output_queue)
                )
                test_thread.start()
            
            # Draw test screen
            self.draw_test_screen(stdscr, test, output_lines)
            stdscr.refresh()
            
            # Handle input
            while True:
                # Set timeout for refresh when test is running
                if test.status == "running":
                    stdscr.timeout(100)  # 100ms timeout for refresh
                else:
                    stdscr.timeout(-1)  # Blocking mode when not running
                
                # Check for output updates if test is running
                if test.status == "running":
                    try:
                        while True:
                            msg_type, data = output_queue.get_nowait()
                            if msg_type == "output":
                                output_lines.append(data)
                                # Auto-scroll to bottom for new output
                                h, w = stdscr.getmaxyx()
                                output_y = status_y + 2 if 'status_y' in locals() else 15
                                output_height = h - output_y - 5
                                visible_height = output_height - 2
                                if len(output_lines) > visible_height:
                                    self.scroll_offset = len(output_lines) - visible_height
                            self.draw_test_screen(stdscr, test, output_lines)
                            stdscr.refresh()
                    except queue.Empty:
                        pass
                    
                    # Update duration while running
                    if test_thread and test_thread.is_alive():
                        self.draw_test_screen(stdscr, test, output_lines)
                        stdscr.refresh()
                
                key = stdscr.getch()
                
                # Handle scrolling
                h, w = stdscr.getmaxyx()
                output_height = h - 20  # Approximate output area height
                visible_height = max(1, output_height - 2)
                max_scroll = max(0, len(output_lines) - visible_height)
                
                if key == curses.KEY_UP:
                    self.scroll_offset = max(0, self.scroll_offset - 1)
                    self.draw_test_screen(stdscr, test, output_lines)
                    stdscr.refresh()
                elif key == curses.KEY_DOWN:
                    self.scroll_offset = min(max_scroll, self.scroll_offset + 1)
                    self.draw_test_screen(stdscr, test, output_lines)
                    stdscr.refresh()
                elif key == curses.KEY_PPAGE:  # Page Up
                    self.scroll_offset = max(0, self.scroll_offset - visible_height)
                    self.draw_test_screen(stdscr, test, output_lines)
                    stdscr.refresh()
                elif key == curses.KEY_NPAGE:  # Page Down
                    self.scroll_offset = min(max_scroll, self.scroll_offset + visible_height)
                    self.draw_test_screen(stdscr, test, output_lines)
                    stdscr.refresh()
                elif key == curses.KEY_HOME:  # Home key
                    self.scroll_offset = 0
                    self.draw_test_screen(stdscr, test, output_lines)
                    stdscr.refresh()
                elif key == curses.KEY_END:  # End key
                    self.scroll_offset = max_scroll
                    self.draw_test_screen(stdscr, test, output_lines)
                    stdscr.refresh()
                
                # Handle other keys
                elif key == curses.ERR:  # Timeout, just refresh
                    if test.status != "running":
                        continue
                        
                elif key == ord('\n') or key == curses.KEY_ENTER or key == 10 or key == 13:  # ENTER key
                    if test.status in ["passed", "failed", "skipped"]:
                        if self.current_test < len(self.tests) - 1:
                            self.current_test += 1
                            break
                        else:
                            # Show summary
                            self.end_time = time.time()
                            self.draw_summary_screen(stdscr)
                            stdscr.refresh()
                            while True:
                                key = stdscr.getch()
                                if key == ord('q') or key == ord('Q'):
                                    return
                                elif key == ord('e') or key == ord('E'):
                                    self.export_results()
                                    
                elif key == ord('p') or key == ord('P'):
                    if self.current_test > 0 and test.status != "running":
                        self.current_test -= 1
                        break
                        
                elif key == ord('r') or key == ord('R'):
                    if test.status in ["passed", "failed"]:
                        test.status = "pending"
                        test.output = ""
                        test.duration = 0.0
                        test.error = ""
                        output_lines = []
                        self.scroll_offset = 0
                        # Auto-start the re-run
                        output_lines.append("=== Re-running test ===")
                        test_thread = threading.Thread(
                            target=self.run_test,
                            args=(test, output_queue)
                        )
                        test_thread.start()
                        
                elif key == ord('q') or key == ord('Q'):
                    return
    
    def export_results(self):
        """Export test results to a file."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"test_results_{timestamp}.txt"
        
        with open(filename, 'w') as f:
            f.write("TEST HELL SESSION RESULTS\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.start_time))}\n")
            f.write(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.end_time))}\n")
            f.write(f"Total duration: {self.end_time - self.start_time:.2f}s\n\n")
            
            # Summary
            passed = sum(1 for t in self.tests if t.status == "passed")
            failed = sum(1 for t in self.tests if t.status == "failed")
            skipped = sum(1 for t in self.tests if t.status == "skipped")
            
            f.write(f"Total tests: {len(self.tests)}\n")
            f.write(f"Passed: {passed}\n")
            f.write(f"Failed: {failed}\n")
            f.write(f"Skipped: {skipped}\n\n")
            
            # Detailed results
            f.write("DETAILED RESULTS\n")
            f.write("-" * 80 + "\n\n")
            
            for i, test in enumerate(self.tests):
                f.write(f"Test {i+1}: {test.name}\n")
                f.write(f"Category: {test.category}\n")
                f.write(f"Status: {test.status}\n")
                f.write(f"Duration: {test.duration:.2f}s\n")
                if test.error:
                    f.write(f"Error: {test.error}\n")
                if test.output:
                    f.write("Output:\n")
                    f.write("-" * 40 + "\n")
                    f.write(test.output + "\n")
                    f.write("-" * 40 + "\n")
                f.write("\n")

def main():
    """Entry point."""
    try:
        runner = InteractiveTestRunner()
        curses.wrapper(runner.run)
    except KeyboardInterrupt:
        print("\nTest hell session interrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()