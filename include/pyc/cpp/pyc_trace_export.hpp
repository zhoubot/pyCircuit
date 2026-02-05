#pragma once

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

namespace pyc::cpp {

// Timeline record compatible with Konata's O3PipeView format and the Chrome/Perfetto
// trace-event JSON format.
struct PipeviewTimeline {
  std::uint64_t id = 0; // Instruction/uop sequence number (monotonic).
  std::uint64_t pc = 0;
  std::uint32_t upc = 0;
  std::string name{};

  std::uint64_t fetch_cycle = 0;
  std::uint64_t decode_cycle = 0;
  std::uint64_t rename_cycle = 0;
  std::uint64_t dispatch_cycle = 0;
  std::uint64_t issue_cycle = 0;
  std::uint64_t complete_cycle = 0;
  std::uint64_t retire_cycle = 0;

  bool is_store = false;
};

namespace detail {

inline std::string hexPc(std::uint64_t pc) {
  std::ostringstream oss;
  oss << "0x" << std::hex << std::setw(16) << std::setfill('0') << pc;
  return oss.str();
}

inline std::string sanitizePipeviewName(std::string name) {
  for (char &c : name) {
    // O3PipeView uses ':' as a field delimiter; avoid breaking parsing.
    if (c == ':')
      c = ';';
    if (c == '\n' || c == '\r' || c == '\t')
      c = ' ';
  }
  return name;
}

inline void writeJsonEscaped(std::ostream &out, const std::string &text) {
  for (unsigned char c : text) {
    switch (c) {
    case '\"':
      out << "\\\"";
      break;
    case '\\':
      out << "\\\\";
      break;
    case '\b':
      out << "\\b";
      break;
    case '\f':
      out << "\\f";
      break;
    case '\n':
      out << "\\n";
      break;
    case '\r':
      out << "\\r";
      break;
    case '\t':
      out << "\\t";
      break;
    default:
      if (c < 0x20) {
        out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<unsigned>(c) << std::dec
            << std::setw(0);
      } else {
        out << static_cast<char>(c);
      }
      break;
    }
  }
}

inline std::uint64_t fallbackTime(std::uint64_t primary, std::uint64_t fallback) { return primary != 0 ? primary : fallback; }

} // namespace detail

class PipeviewStreamWriter {
public:
  ~PipeviewStreamWriter() { close(); }
  bool open(const std::string &path) {
    out_.open(path);
    return out_.is_open();
  }

  bool isOpen() const { return out_.is_open(); }

  void close() {
    if (out_.is_open())
      out_.close();
  }

  bool write(const PipeviewTimeline &t) {
    if (!out_.is_open())
      return false;
    if (t.dispatch_cycle == 0 || t.retire_cycle == 0)
      return true;

    const std::uint64_t fetch = detail::fallbackTime(t.fetch_cycle, t.dispatch_cycle);
    const std::uint64_t decode = detail::fallbackTime(t.decode_cycle, t.dispatch_cycle);
    const std::uint64_t rename = detail::fallbackTime(t.rename_cycle, decode);
    const std::uint64_t dispatch = t.dispatch_cycle;
    const std::uint64_t issue = detail::fallbackTime(t.issue_cycle, dispatch);
    const std::uint64_t complete = detail::fallbackTime(t.complete_cycle, issue);
    const std::uint64_t retire = t.retire_cycle;

    const std::uint64_t seq = t.id;
    const std::string pc_hex = detail::hexPc(t.pc);
    const std::uint32_t upc = t.upc;
    const std::string name = detail::sanitizePipeviewName(t.name);

    out_ << "O3PipeView:fetch:" << fetch << ":" << pc_hex << ":" << upc << ":" << seq << ":" << name << "\n";
    out_ << "O3PipeView:decode:" << decode << "\n";
    out_ << "O3PipeView:rename:" << rename << "\n";
    out_ << "O3PipeView:dispatch:" << dispatch << "\n";
    out_ << "O3PipeView:issue:" << issue << "\n";
    out_ << "O3PipeView:complete:" << complete << "\n";
    out_ << "O3PipeView:retire:" << retire << ":store:" << (t.is_store ? 1 : 0) << "\n";
    out_.flush();
    return true;
  }

private:
  std::ofstream out_{};
};

class PerfettoTraceWriter {
public:
  ~PerfettoTraceWriter() { close(); }
  bool open(const std::string &path) {
    out_.open(path);
    if (!out_.is_open())
      return false;
    out_ << "{\"traceEvents\":[";
    first_ = true;
    return true;
  }

  bool isOpen() const { return out_.is_open(); }

  void close() {
    if (!out_.is_open())
      return;
    out_ << "]}";
    out_.close();
  }

  bool write(const PipeviewTimeline &t) {
    if (!out_.is_open())
      return false;
    if (t.dispatch_cycle == 0 || t.retire_cycle == 0)
      return true;

    const std::uint64_t dispatch = t.dispatch_cycle;
    const std::uint64_t issue = detail::fallbackTime(t.issue_cycle, dispatch);
    const std::uint64_t complete = detail::fallbackTime(t.complete_cycle, issue);
    const std::uint64_t retire = t.retire_cycle;

    emitSlice(t.name, "dispatch", dispatch, issue, 0);
    emitSlice(t.name, "execute", issue, complete, 1);
    emitSlice(t.name, "commit", complete, retire + 1, 2);
    out_.flush();
    return true;
  }

private:
  void emitSlice(const std::string &name, const char *cat, std::uint64_t start, std::uint64_t end, int tid) {
    if (!first_)
      out_ << ",";
    first_ = false;
    const std::uint64_t dur = (end > start) ? (end - start) : 1;
    out_ << "{\"name\":\"";
    detail::writeJsonEscaped(out_, name);
    out_ << "\",\"cat\":\"" << cat << "\",\"ph\":\"X\",\"ts\":" << start << ",\"dur\":" << dur
         << ",\"pid\":1,\"tid\":" << tid << "}";
  }

  std::ofstream out_{};
  bool first_ = true;
};

inline bool write_o3_pipeview(const std::string &path, const std::vector<PipeviewTimeline> &timelines) {
  std::ofstream out(path);
  if (!out)
    return false;

  std::vector<PipeviewTimeline> sorted = timelines;
  std::sort(sorted.begin(), sorted.end(), [](const auto &a, const auto &b) { return a.id < b.id; });

  for (const auto &t : sorted) {
    if (t.dispatch_cycle == 0 || t.retire_cycle == 0)
      continue;

    const std::uint64_t fetch = detail::fallbackTime(t.fetch_cycle, t.dispatch_cycle);
    const std::uint64_t decode = detail::fallbackTime(t.decode_cycle, t.dispatch_cycle);
    const std::uint64_t rename = detail::fallbackTime(t.rename_cycle, decode);
    const std::uint64_t dispatch = t.dispatch_cycle;
    const std::uint64_t issue = detail::fallbackTime(t.issue_cycle, dispatch);
    const std::uint64_t complete = detail::fallbackTime(t.complete_cycle, issue);
    const std::uint64_t retire = t.retire_cycle;

    const std::uint64_t seq = t.id;
    const std::string pc_hex = detail::hexPc(t.pc);
    const std::uint32_t upc = t.upc;
    const std::string name = detail::sanitizePipeviewName(t.name);

    out << "O3PipeView:fetch:" << fetch << ":" << pc_hex << ":" << upc << ":" << seq << ":" << name << "\n";
    out << "O3PipeView:decode:" << decode << "\n";
    out << "O3PipeView:rename:" << rename << "\n";
    out << "O3PipeView:dispatch:" << dispatch << "\n";
    out << "O3PipeView:issue:" << issue << "\n";
    out << "O3PipeView:complete:" << complete << "\n";
    out << "O3PipeView:retire:" << retire << ":store:" << (t.is_store ? 1 : 0) << "\n";
  }

  return true;
}

inline bool write_perfetto_trace(const std::string &path, const std::vector<PipeviewTimeline> &timelines) {
  std::ofstream out(path);
  if (!out)
    return false;

  out << "{\"traceEvents\":[";

  bool first = true;
  auto emit_slice = [&](const std::string &name, const char *cat, std::uint64_t start, std::uint64_t end, int tid) {
    if (!first)
      out << ",";
    first = false;

    const std::uint64_t dur = (end > start) ? (end - start) : 1;
    out << "{\"name\":\"";
    detail::writeJsonEscaped(out, name);
    out << "\",\"cat\":\"" << cat << "\",\"ph\":\"X\",\"ts\":" << start << ",\"dur\":" << dur
        << ",\"pid\":1,\"tid\":" << tid << "}";
  };

  std::vector<PipeviewTimeline> sorted = timelines;
  std::sort(sorted.begin(), sorted.end(), [](const auto &a, const auto &b) {
    if (a.dispatch_cycle == b.dispatch_cycle)
      return a.id < b.id;
    return a.dispatch_cycle < b.dispatch_cycle;
  });

  for (const auto &t : sorted) {
    if (t.dispatch_cycle == 0 || t.retire_cycle == 0)
      continue;

    const std::uint64_t dispatch = t.dispatch_cycle;
    const std::uint64_t issue = detail::fallbackTime(t.issue_cycle, dispatch);
    const std::uint64_t complete = detail::fallbackTime(t.complete_cycle, issue);
    const std::uint64_t retire = t.retire_cycle;

    emit_slice(t.name, "dispatch", dispatch, issue, 0);
    emit_slice(t.name, "execute", issue, complete, 1);
    emit_slice(t.name, "commit", complete, retire + 1, 2);
  }

  out << "]}";
  return true;
}

} // namespace pyc::cpp
