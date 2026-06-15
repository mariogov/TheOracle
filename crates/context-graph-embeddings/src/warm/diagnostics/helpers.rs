//! Helper Functions for Diagnostics
//!
//! Utility functions used by the diagnostic system for formatting
//! and data transformation.

use std::time::SystemTime;

// ============================================================================
// Helper Functions
// ============================================================================

/// Format bytes as a human-readable string.
///
/// # Examples
///
/// ```rust,ignore
/// assert_eq!(format_bytes(1024), "1.00KB");
/// assert_eq!(format_bytes(1024 * 1024), "1.00MB");
/// assert_eq!(format_bytes(32 * 1024 * 1024 * 1024), "32.00GB");
/// ```
pub fn format_bytes(bytes: usize) -> String {
    const KB: usize = 1024;
    const MB: usize = KB * 1024;
    const GB: usize = MB * 1024;

    if bytes >= GB {
        format!("{:.2}GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2}MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2}KB", bytes as f64 / KB as f64)
    } else {
        format!("{}B", bytes)
    }
}

/// Get the current ISO 8601 timestamp.
///
/// Returns a string in format: `YYYY-MM-DDTHH:MM:SS.mmmZ`
pub fn current_timestamp() -> String {
    let now = SystemTime::now();
    let duration = now
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = duration.as_secs();
    let millis = duration.subsec_millis();

    // Format as ISO 8601: 2025-01-03T12:00:00.000Z
    let days_since_epoch = secs / 86400;
    let secs_today = secs % 86400;
    let hours = secs_today / 3600;
    let minutes = (secs_today % 3600) / 60;
    let seconds = secs_today % 60;

    // Approximate date calculation
    let mut year = 1970;
    let mut remaining_days = days_since_epoch;

    while remaining_days >= 365 {
        let days_in_year = if is_leap_year(year) { 366 } else { 365 };
        if remaining_days >= days_in_year {
            remaining_days -= days_in_year;
            year += 1;
        } else {
            break;
        }
    }

    let days_in_months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut month = 1;
    for &days_in_month in &days_in_months {
        let days = if month == 2 && is_leap_year(year) {
            29
        } else {
            days_in_month
        };
        if remaining_days >= days {
            remaining_days -= days;
            month += 1;
        } else {
            break;
        }
    }
    let day = remaining_days + 1;

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}Z",
        year, month, day, hours, minutes, seconds, millis
    )
}

/// Check if a year is a leap year.
#[inline]
fn is_leap_year(year: u64) -> bool {
    year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400))
}

#[cfg(test)]
mod helper_tests {
    use super::*;

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(0), "0B");
        assert_eq!(format_bytes(512), "512B");
        assert_eq!(format_bytes(1024), "1.00KB");
        assert_eq!(format_bytes(1024 * 1024), "1.00MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.00GB");
        assert_eq!(format_bytes(32 * 1024 * 1024 * 1024), "32.00GB");
    }

    #[test]
    fn test_timestamp_format() {
        let ts = current_timestamp();
        // Should be in ISO 8601 format: YYYY-MM-DDTHH:MM:SS.mmmZ
        assert!(ts.contains('T'));
        assert!(ts.ends_with('Z'));
        assert_eq!(ts.len(), 24);
    }
}
