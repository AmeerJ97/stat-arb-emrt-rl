import os
import gc
import psutil
import shutil
import time
import platform
import ctypes
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────
# Configuration Flags (customize as needed)
# ─────────────────────────────────────────────────────────────────────
ENABLE_PROCESS_TERMINATION = False  # Set to False to disable process termination
MAX_RETRIES = 3                   # Number of deletion retry attempts
RETRY_DELAY = 0                  # Seconds between retries
CURRENT_PID = os.getpid()          # PID of current process

# ─────────────────────────────────────────────────────────────────────
# Terminal Color Codes
# ─────────────────────────────────────────────────────────────────────
COLOR_RESET = "\033[0m"
COLOR_GREEN = "\033[38;5;34m"
COLOR_YELLOW = "\033[93m"
COLOR_RED = "\033[91m"
COLOR_BLUE = "\033[34m"
COLOR_HEADER_BOX = "\033[38;5;240m"
COLOR_HEADER_TEXT = "\033[38;5;34m"

# ─────────────────────────────────────────────────────────────────────
# Self-contained print_header function
# ─────────────────────────────────────────────────────────────────────
def print_header(message: str):
    """Replacement for printing_system.print_header to avoid log file locking"""
    try:
        # Get terminal width
        term_width = os.get_terminal_size().columns
    except:
        term_width = 80
    
    # Ensure minimum width
    term_width = max(term_width, 60)
    
    border = f"{COLOR_HEADER_BOX}╔{'═' * (term_width-2)}╗{COLOR_RESET}"
    header_line = f"{COLOR_HEADER_BOX}║{COLOR_RESET} {COLOR_HEADER_TEXT}{message.center(term_width-4)}{COLOR_RESET} {COLOR_HEADER_BOX}║{COLOR_RESET}"
    footer = f"{COLOR_HEADER_BOX}╚{'═' * (term_width-2)}╝{COLOR_RESET}"
    
    print(f"\n{border}\n{header_line}\n{footer}\n")

# Call the header function
print_header(message="CLEANUP TOOL")

# ─────────────────────────────────────────────────────────────────────
# Function to Print Status Messages with Consistent Formatting
# ─────────────────────────────────────────────────────────────────────
def print_status(message, status="INFO"):
    # Create consistent status labels
    status_labels = {
        "SUCCESS": "[SUCCESS]",
        "WARNING": "[WARNING]",
        "ERROR": "[ERROR]  ",
        "INFO": "[INFO]   "
    }
    
    # Map status to colors
    status_colors = {
        "SUCCESS": COLOR_GREEN,
        "WARNING": COLOR_YELLOW,
        "ERROR": COLOR_RED,
        "INFO": COLOR_BLUE
    }
    
    # Get formatted label and color
    formatted_label = status_labels.get(status, f"[{status}]")
    color = status_colors.get(status, COLOR_RESET)
    
    # Print with consistent formatting
    print(f"{color}{formatted_label} {COLOR_RESET}{message}")

# ─────────────────────────────────────────────────────────────────────
# Windows-specific File Lock Handling
# ─────────────────────────────────────────────────────────────────────
def force_delete_windows(file_path):
    """Attempt to force delete a file on Windows using kernel APIs"""
    try:
        from ctypes import wintypes
        
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        DeleteFile = kernel32.DeleteFileW
        DeleteFile.argtypes = [wintypes.LPCWSTR]
        DeleteFile.restype = wintypes.BOOL
        
        if DeleteFile(str(file_path)):
            return True
            
        MOVEFILE_DELAY_UNTIL_REBOOT = 0x4
        MoveFileEx = kernel32.MoveFileExW
        MoveFileEx.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD]
        MoveFileEx.restype = wintypes.BOOL
        
        if MoveFileEx(str(file_path), None, MOVEFILE_DELAY_UNTIL_REBOOT):
            return True
            
        error_code = ctypes.get_last_error()
        raise OSError(f"Windows error {error_code}")
    except Exception as e:
        raise RuntimeError(f"Force deletion failed: {e}")

# ─────────────────────────────────────────────────────────────────────
# Function to Identify Locking Process
# ─────────────────────────────────────────────────────────────────────
def get_locking_process(file_path):
    """Try to identify which process is locking a file (Windows only)"""
    if platform.system() != 'Windows':
        return "Unknown (only Windows supported)", None
    
    try:
        for proc in psutil.process_iter(['pid', 'name', 'open_files']):
            try:
                if proc.pid == CURRENT_PID:
                    continue  # Skip current process
                    
                files = proc.open_files()
                for f in files:
                    if Path(f.path) == Path(file_path):
                        return proc.name(), proc.pid
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        return "Unknown process", None
    except Exception:
        return "Unknown process", None

# ─────────────────────────────────────────────────────────────────────
# Function to Terminate Locking Process
# ─────────────────────────────────────────────────────────────────────
def terminate_locking_process(pid, process_name):
    """Attempt to terminate a process by PID"""
    try:
        if pid == CURRENT_PID:
            return False, "Cannot terminate current process"
            
        process = psutil.Process(pid)
        process.terminate()
        time.sleep(1)  # Give time for process to exit
        if process.is_running():
            process.kill()
            time.sleep(0.5)
            if process.is_running():
                return False, "Failed to kill process"
        return True, f"Terminated {process_name} (PID: {pid})"
    except psutil.NoSuchProcess:
        return True, "Process already exited"
    except Exception as e:
        return False, f"Error: {str(e)}"

# ─────────────────────────────────────────────────────────────────────
# Define Cleanup Targets (Directories and Files to be Removed)
# ─────────────────────────────────────────────────────────────────────
base_path = Path("C:/Users/Ameer/Desktop/Files/Ice_Crown/Quant_Workspace/code")
targets = [
    base_path / "__pycache__",  # Common Python cache directory
    base_path / "backtest/__pycache__",  # Common Python cache directory
    base_path / "logs/logs.log",     # Log file
]

# ─────────────────────────────────────────────────────────────────────
# Garbage Collection and Memory Usage Snapshot
# ─────────────────────────────────────────────────────────────────────
print_status("Performing garbage collection to free unused memory...", "INFO")

# Get memory usage before GC
proc = psutil.Process(os.getpid())
mem_before = proc.memory_info().rss

# Perform garbage collection
gc.collect()

# Get memory usage after GC
mem_after = proc.memory_info().rss

# Calculate memory saved in MB
mem_saved = mem_before - mem_after
mem_before_mb = mem_before / (1024 * 1024)
mem_after_mb = mem_after / (1024 * 1024)
mem_saved_mb = mem_saved / (1024 * 1024)

# Print memory usage in MB with savings
print_status(f"Memory Before GC: {mem_before_mb:.2f} MB", "INFO")
print_status(f"Memory After GC: {mem_after_mb:.2f} MB", "INFO")
print_status(f"Memory Saved: {mem_saved_mb:.2f} MB", "SUCCESS")

# ─────────────────────────────────────────────────────────────────────
# Cleanup Process: Enhanced with Process Termination
# ─────────────────────────────────────────────────────────────────────
print_status("Starting cleanup of specified targets...", "INFO")

for target in targets:
    for attempt in range(MAX_RETRIES):
        try:
            if target.is_dir():
                shutil.rmtree(target)
                print_status(f"Deleted directory: {target}", "SUCCESS")
                break
            elif target.is_file():
                target.unlink()
                print_status(f"Deleted file: {target}", "SUCCESS")
                break
            else:
                print_status(f"Target not found: {target}", "WARNING")
                break
                
        except PermissionError as e:
            process_name, pid = get_locking_process(target)
            locking_info = f" (locked by {process_name} (PID: {pid}))" if pid else ""
            
            if attempt < MAX_RETRIES - 1:
                print_status(
                    f"Permission denied for {target}{locking_info} "
                    f"(retry {attempt+1}/{MAX_RETRIES} in {RETRY_DELAY} sec)...", 
                    "WARNING"
                )
                time.sleep(RETRY_DELAY)
            else:
                success = False
                
                # Try terminating the locking process
                if pid and ENABLE_PROCESS_TERMINATION:
                    print_status(
                        f"Attempting to terminate locking process: {process_name} (PID: {pid})", 
                        "WARNING"
                    )
                    terminated, message = terminate_locking_process(pid, process_name)
                    if terminated:
                        print_status(message, "SUCCESS")
                        try:
                            # Try deletion immediately after termination
                            if target.is_file():
                                target.unlink()
                            elif target.is_dir():
                                shutil.rmtree(target)
                            print_status(f"Deleted after terminating process: {target}", "SUCCESS")
                            success = True
                        except Exception as e:
                            print_status(f"Deletion failed after process termination: {e}", "WARNING")
                    else:
                        print_status(f"Process termination failed: {message}", "ERROR")
                
                # Try Windows force deletion if still not successful
                if not success and platform.system() == 'Windows' and target.is_file():
                    try:
                        print_status("Attempting Windows force deletion...", "WARNING")
                        force_delete_windows(target)
                        print_status(f"Force deleted file: {target}", "SUCCESS")
                        success = True
                    except Exception as force_error:
                        print_status(f"Force deletion failed: {force_error}", "ERROR")
                
                # Final error if all methods failed
                if not success:
                    print_status(
                        f"Permanently failed to delete {target}: "
                        f"Locked by {process_name} (PID: {pid})", 
                        "ERROR"
                    )
                else:
                    break  # Break out of retry loop on success
                    
        except Exception as e:
            print_status(f"Failed to delete {target}: {e}", "ERROR")
            break

# ─────────────────────────────────────────────────────────────────────
# Final Completion Message
# ─────────────────────────────────────────────────────────────────────
print_status("GC cleanup completed successfully.", "INFO")