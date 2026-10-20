#pragma once

/**
 * spdlog-based logging wrapper.
 *
 * Usage:
 *   #include "io/logger.hpp"
 *   log::info("lattice parameter = {:.4f} a.u.", alat);
 *   log::debug("iteration {} delta = {:.2e}", iter, de);
 */

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace q3 {
namespace log {

/**
 * Initialize the logger. Call once at startup.
 */
inline std::shared_ptr<spdlog::sinks::stdout_color_sink_mt>& console_sink() {
    static std::shared_ptr<spdlog::sinks::stdout_color_sink_mt> sink;
    return sink;
}

/**
 * The level init() left the logger at, so a later change can be undone.
 *
 * Not simply info: a run that asked for a log file deliberately drops the
 * logger to debug so the file sink sees records the console does not, and
 * restoring info instead would empty that file. The console sink needs no
 * such memory -- init() always pins it at info.
 */
inline spdlog::level::level_enum& baseline_logger_level() {
    static spdlog::level::level_enum level = spdlog::level::info;
    return level;
}

/**
 * Move the console sink's own threshold, unless this process was silenced.
 *
 * The sink filters after the logger does, so a logger-level change on its own
 * never reaches the terminal. Moving the sink unconditionally would have the
 * opposite fault: silence() takes it to off on a non-root MPI rank, and that
 * decision must outlive any later level change.
 */
inline void set_console_level_unless_silenced(spdlog::level::level_enum level) {
    if (console_sink() && console_sink()->level() != spdlog::level::off) {
        console_sink()->set_level(level);
    }
}

inline void init(const std::string& log_file = "") {
    console_sink() = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink()->set_pattern("%v"); // Clean output — no timestamp/level prefix
    console_sink()->set_level(spdlog::level::info);

    std::vector<spdlog::sink_ptr> sinks{console_sink()};

    /**
     * The file is a second sink on the same logger, not a logger of its own.
     * It used to be the latter -- registered under its own name and never
     * written to, because every log:: call goes to the default logger -- so a
     * requested log file was reliably created and reliably empty.
     *
     * It is given a lower threshold than the console on purpose: a file is
     * asked for when the console has already proved too terse, so it carries
     * debug records the console drops. That is also why the logger's own
     * threshold moves down with it -- a logger filters before its sinks do, so
     * leaving it at info would discard those records before the file sink ever
     * saw them.
     */
    if (!log_file.empty()) {
        auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_file, true);
        file_sink->set_pattern("%v");
        file_sink->set_level(spdlog::level::debug);
        sinks.push_back(file_sink);
    }

    auto logger = std::make_shared<spdlog::logger>("q3", sinks.begin(), sinks.end());
    baseline_logger_level() = log_file.empty() ? spdlog::level::info : spdlog::level::debug;
    logger->set_level(baseline_logger_level());
    logger->flush_on(spdlog::level::warn);
    spdlog::set_default_logger(logger);
}

/// Set log level: "trace", "debug", "info", "warn", "error", "off"
inline void set_level(const std::string& level) {
    spdlog::level::level_enum parsed_level = spdlog::level::from_str(level);
    spdlog::set_level(parsed_level);
    set_console_level_unless_silenced(parsed_level);
}

/**
 * Apply an input deck's control.verbosity to the log level.
 *
 * `explicit_log_level` is whatever --log-level was given, empty when it was
 * not. An explicit flag always wins: it is the more specific request, and a
 * deck carried from machine to machine must not talk over what the command
 * line asked for.
 *
 * "high" opens up the per-iteration and per-phase diagnostics, which are
 * logged at debug. "default" puts the level back where init() left it rather
 * than assuming info, so that a log file keeps receiving those records and so
 * that one in-process run cannot inherit the previous deck's verbosity.
 */
inline void apply_verbosity(const std::string& verbosity, const std::string& explicit_log_level) {
    if (!explicit_log_level.empty()) {
        return;
    }
    if (verbosity == "high") {
        set_level("debug");
        return;
    }
    spdlog::set_level(baseline_logger_level());
    set_console_level_unless_silenced(spdlog::level::info);
}

/**
 * Push every buffered record to its sink.
 *
 * Needed before any exit that skips destructors -- MPI_Abort, std::_Exit --
 * because spdlog buffers, and a process killed with records still in the
 * buffer loses them. The message being lost is invariably the one explaining
 * why the process is exiting, which is the one worth keeping.
 */
inline void flush() {
    spdlog::default_logger()->flush();
}

/**
 * Silence this process's console output (e.g. non-root MPI ranks).
 *
 * Silences the console sink, not the logger. Setting the logger's own level to
 * off would silence its file sink too, and the one thing a non-root rank has
 * to be able to do is write its own log file -- which is the entire reason
 * init_mpi() gives each rank a numbered one. Falls back to the logger-wide
 * setting only if init() has not run.
 */
inline void silence() {
    if (console_sink()) {
        console_sink()->set_level(spdlog::level::off);
        return;
    }
    spdlog::set_level(spdlog::level::off);
}

/**
 * Report a fatal error on stderr, whatever the current log level.
 *
 * error() routes through spdlog, and silence() takes the console sink off --
 * which means that on a silenced rank a fatal error still reaches that rank's
 * log file, if it has one, but never the console the user is watching. A
 * process about to die owes its user a reason on the console whether or not a
 * log file was asked for, so this bypasses spdlog and
 * writes straight to stderr, flushing before it returns: MPI_Abort neither
 * unwinds nor flushes, and a rank that exits nonzero takes its peers down with
 * it, so anything still sitting in a buffer at that point is never seen.
 */
inline void fatal(const std::string& message) {
    std::fputs(message.c_str(), stderr);
    std::fputc('\n', stderr);
    std::fflush(stderr);
}

// Convenience wrappers that forward to spdlog
template <typename... Args>
inline void trace(spdlog::format_string_t<Args...> fmt, Args&&... args) {
    spdlog::trace(fmt, std::forward<Args>(args)...);
}

template <typename... Args>
inline void debug(spdlog::format_string_t<Args...> fmt, Args&&... args) {
    spdlog::debug(fmt, std::forward<Args>(args)...);
}

template <typename... Args> inline void info(spdlog::format_string_t<Args...> fmt, Args&&... args) {
    spdlog::info(fmt, std::forward<Args>(args)...);
}

template <typename... Args> inline void warn(spdlog::format_string_t<Args...> fmt, Args&&... args) {
    spdlog::warn(fmt, std::forward<Args>(args)...);
}

template <typename... Args>
inline void error(spdlog::format_string_t<Args...> fmt, Args&&... args) {
    spdlog::error(fmt, std::forward<Args>(args)...);
}

/**
 * A blank line, then a section title at the common margin, underlined.
 *
 * Every block-level heading a run prints goes through here, so the margin and
 * the separation between blocks are decided once rather than at each print
 * site -- which is how the output came to carry three competing left margins
 * and three header styles. The title text itself is passed through unchanged,
 * because several of these headings are what the validation suite's stdout
 * parsers key on.
 */
inline void section(const std::string& title) {
    info("");
    info("     {}", title);
    info("     {}", std::string(title.size(), '-'));
}

} // namespace log
} // namespace q3
