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

@dataclass
class TestCase:
    name: str
    command: List[str]
    category: str
    description: str
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
        
    def _collect_tests(self) -> List[TestCase]:
        """Collect all tests from the codebase."""
        tests = []
        
        # Core CI Tests
        tests.append(TestCase(
            name="Core CI Suite",
            command=["python3", "tests/run_ci.py"],
            category="Core Tests",
            description="Main CI test suite with 7 comprehensive tests"
        ))
        
        # Custom Operations Tests
        tests.extend([
            TestCase(
                name="C Library Tests",
                command=["python3", "tests/unit/custom_ops/test_c_library.py"],
                category="Custom Operations",
                description="Direct testing of physics C functions"
            ),
            TestCase(
                name="Integration Tests",
                command=["python3", "tests/unit/custom_ops/test_integration.py"],
                category="Custom Operations",
                description="End-to-end custom op testing"
            ),
            TestCase(
                name="Basic Demo",
                command=["python3", "custom_ops/examples/basic_demo.py"],
                category="Custom Operations",
                description="Demonstration of custom ops usage"
            ),
            TestCase(
                name="Performance Benchmark",
                command=["python3", "custom_ops/examples/benchmark.py"],
                category="Custom Operations",
                description="Shows performance improvements"
            )
        ])
        
        # Debugging Tests
        tests.extend([
            TestCase(
                name="Position Corruption",
                command=["python3", "tests/debugging/test_position_corruption.py"],
                category="Debugging",
                description="Verifies position updates work correctly"
            ),
            TestCase(
                name="NaN Propagation",
                command=["python3", "tests/debugging/test_nan_propagation.py"],
                category="Debugging",
                description="Tests NaN handling in physics"
            ),
            TestCase(
                name="Empty Contacts",
                command=["python3", "tests/debugging/test_empty_contacts_simple.py"],
                category="Debugging",
                description="Tests solver with no collisions"
            ),
            TestCase(
                name="JIT Early Return",
                command=["python3", "tests/debugging/test_jit_early_return.py"],
                category="Debugging",
                description="Tests conditional logic in JIT"
            )
        ])
        
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
        stdscr.addstr(info_y + 2, 2, f"Command: {' '.join(test.command)}", curses.color_pair(1))
        
        # Status
        status_y = info_y + 4
        status_str = f"Status: {test.status.upper()}"
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
        
        # Output window
        output_y = status_y + 2
        output_height = h - output_y - 4
        
        if output_height > 2:
            # Draw output box
            stdscr.addstr(output_y, 1, f"╔{'═' * (w-4)}╗", curses.color_pair(2))
            for i in range(output_height - 2):
                stdscr.addstr(output_y + 1 + i, 1, "║", curses.color_pair(2))
                stdscr.addstr(output_y + 1 + i, w - 2, "║", curses.color_pair(2))
            stdscr.addstr(output_y + output_height - 1, 1, f"╚{'═' * (w-4)}╝", curses.color_pair(2))
            
            # Show output lines
            visible_lines = output_lines[-(output_height-2):]
            for i, line in enumerate(visible_lines):
                if i < output_height - 2:
                    try:
                        # Truncate long lines
                        max_width = w - 6
                        if len(line) > max_width:
                            line = line[:max_width-3] + "..."
                        stdscr.addstr(output_y + 1 + i, 3, line[:max_width])
                    except:
                        pass
        
        # Footer
        footer_y = h - 2
        if test.status == "pending":
            footer = "Press SPACE to run this test, N to skip"
        elif test.status == "running":
            footer = "Test is running... (output streaming above)"
        elif test.status in ["passed", "failed"]:
            footer = "Press N for next test, P for previous, R to re-run, Q to quit"
        else:
            footer = "Press N for next test, P for previous"
        
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
            
            # Draw test screen
            self.draw_test_screen(stdscr, test, output_lines)
            stdscr.refresh()
            
            # Handle input
            while True:
                # Check for output updates if test is running
                if test.status == "running":
                    try:
                        while True:
                            msg_type, data = output_queue.get_nowait()
                            if msg_type == "output":
                                output_lines.append(data)
                            self.draw_test_screen(stdscr, test, output_lines)
                            stdscr.refresh()
                    except queue.Empty:
                        pass
                    
                    # Check if test is still running
                    if test_thread.is_alive():
                        stdscr.timeout(100)  # 100ms timeout for refresh
                    else:
                        stdscr.timeout(-1)  # Back to blocking
                        self.draw_test_screen(stdscr, test, output_lines)
                        stdscr.refresh()
                        continue
                
                key = stdscr.getch()
                
                if key == ord(' ') and test.status == "pending":
                    # Run the test
                    test_thread = threading.Thread(
                        target=self.run_test,
                        args=(test, output_queue)
                    )
                    test_thread.start()
                    
                elif key == ord('n') or key == ord('N'):
                    if test.status == "pending":
                        test.status = "skipped"
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
                    if self.current_test > 0:
                        self.current_test -= 1
                        break
                        
                elif key == ord('r') or key == ord('R'):
                    if test.status in ["passed", "failed"]:
                        test.status = "pending"
                        test.output = ""
                        test.duration = 0.0
                        test.error = ""
                        
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