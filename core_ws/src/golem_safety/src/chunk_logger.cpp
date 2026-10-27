#include "golem_safety/chunk_logger.hpp"

#include <unistd.h>
#include <zlib.h>

#include <algorithm>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>

#if !defined(__BYTE_ORDER__) || __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "the npy and zip writers store values in host byte order, which must be little-endian"
#endif

namespace fs = std::filesystem;

namespace golem_safety
{

LogSample::LogSample(size_t motors)
: mode(motors), q(motors), dq(motors), ddq(motors), tau_est(motors), temperature(motors),
  vol(motors), sensor(motors), motorstate(motors), reserve(motors), q_cmd(motors),
  dq_cmd(motors), tau_cmd(motors), kp_cmd(motors), kd_cmd(motors)
{
}

std::string run_log_directory(const std::string & root)
{
  const std::time_t now = std::time(nullptr);
  std::tm local{};
  localtime_r(&now, &local);
  char stamp[32];
  std::strftime(stamp, sizeof(stamp), "%Y%m%d-%H%M%S", &local);
  return (fs::path(root) / (std::string(stamp) + "-" + std::to_string(::getpid()))).string();
}

namespace
{

// numpy's dtype strings, for the element types a LogSample holds.
template<class T>
struct Descr;
template<>
struct Descr<double> {static constexpr const char * value = "<f8";};
template<>
struct Descr<float> {static constexpr const char * value = "<f4";};
template<>
struct Descr<uint8_t> {static constexpr const char * value = "|u1";};
template<>
struct Descr<int16_t> {static constexpr const char * value = "<i2";};
template<>
struct Descr<uint32_t> {static constexpr const char * value = "<u4";};

// What one sample's field is made of: its element type, the dimensions it adds to the
// chunk's shape after the sample axis, and how it is appended to the chunk's bytes.
template<class T>
struct Field
{
  using Element = T;
  static void tail(std::vector<size_t> &, size_t) {}
  static void append(std::vector<uint8_t> & out, const T & value)
  {
    const auto * bytes = reinterpret_cast<const uint8_t *>(&value);
    out.insert(out.end(), bytes, bytes + sizeof(T));
  }
};

template<class T, size_t K>
struct Field<std::array<T, K>>
{
  using Element = typename Field<T>::Element;
  static void tail(std::vector<size_t> & shape, size_t motors)
  {
    shape.push_back(K);
    Field<T>::tail(shape, motors);
  }
  static void append(std::vector<uint8_t> & out, const std::array<T, K> & value)
  {
    for (const T & item : value) {
      Field<T>::append(out, item);
    }
  }
};

// A std::vector field has one entry per motor.
template<class T>
struct Field<std::vector<T>>
{
  using Element = typename Field<T>::Element;
  static void tail(std::vector<size_t> & shape, size_t motors)
  {
    shape.push_back(motors);
    Field<T>::tail(shape, motors);
  }
  static void append(std::vector<uint8_t> & out, const std::vector<T> & value)
  {
    for (const T & item : value) {
      Field<T>::append(out, item);
    }
  }
};

// One array of a chunk: `data` holds the elements in C order.
struct Column
{
  std::string name;
  std::string descr;
  std::vector<size_t> shape;
  std::vector<uint8_t> data;
};

// The array of `member` over `samples`, shaped (samples, <the field's dimensions>).
template<class T>
Column column(
  const char * name, const std::vector<LogSample> & samples, size_t motors, T LogSample::* member)
{
  Column out;
  out.name = name;
  out.descr = Descr<typename Field<T>::Element>::value;
  out.shape = {samples.size()};
  Field<T>::tail(out.shape, motors);
  for (const LogSample & sample : samples) {
    Field<T>::append(out.data, sample.*member);
  }
  return out;
}

void put16(std::vector<uint8_t> & out, uint16_t value)
{
  out.push_back(static_cast<uint8_t>(value));
  out.push_back(static_cast<uint8_t>(value >> 8));
}

void put32(std::vector<uint8_t> & out, uint32_t value)
{
  put16(out, static_cast<uint16_t>(value));
  put16(out, static_cast<uint16_t>(value >> 16));
}

void put_text(std::vector<uint8_t> & out, const std::string & text)
{
  out.insert(out.end(), text.begin(), text.end());
}

// A version 1.0 .npy file: magic, version, header length, the header dict padded with
// spaces and a newline to a multiple of 64 bytes, then the data.
std::vector<uint8_t> npy_file(const Column & column)
{
  std::string shape = "(";
  for (size_t i = 0; i < column.shape.size(); ++i) {
    shape += (i ? ", " : "") + std::to_string(column.shape[i]);
  }
  shape += column.shape.size() == 1 ? ",)" : ")";
  std::string header =
    "{'descr': '" + column.descr + "', 'fortran_order': False, 'shape': " + shape + ", }";
  const size_t fixed = 6 + 2 + 2;  // magic, version, header length
  header.append((64 - (fixed + header.size() + 1) % 64) % 64, ' ');
  header += '\n';

  std::vector<uint8_t> out;
  out.reserve(fixed + header.size() + column.data.size());
  put_text(out, std::string("\x93NUMPY\x01\x00", 8));
  put16(out, static_cast<uint16_t>(header.size()));
  put_text(out, header);
  out.insert(out.end(), column.data.begin(), column.data.end());
  return out;
}

// A zip archive whose entries are stored uncompressed, which numpy.load reads as an .npz.
class StoredZip
{
public:
  void add(const std::string & name, const std::vector<uint8_t> & data)
  {
    if (data.size() > std::numeric_limits<uint32_t>::max() ||
      file_.size() > std::numeric_limits<uint32_t>::max())
    {
      throw std::runtime_error("an npz entry or the archive passed 4 GiB");
    }
    Entry entry;
    entry.name = name;
    entry.size = static_cast<uint32_t>(data.size());
    entry.crc = static_cast<uint32_t>(
      ::crc32(::crc32(0L, Z_NULL, 0), data.data(), static_cast<uInt>(data.size())));
    entry.offset = static_cast<uint32_t>(file_.size());

    put32(file_, 0x04034b50);  // local file header
    put16(file_, 20);          // version needed to extract
    put16(file_, 0);           // flags
    put16(file_, 0);           // method: stored
    put16(file_, 0);           // modification time
    put16(file_, kDosDate);
    put32(file_, entry.crc);
    put32(file_, entry.size);  // compressed size
    put32(file_, entry.size);  // uncompressed size
    put16(file_, static_cast<uint16_t>(name.size()));
    put16(file_, 0);  // extra field length
    put_text(file_, name);
    file_.insert(file_.end(), data.begin(), data.end());
    entries_.push_back(std::move(entry));
  }

  // The finished archive: the entries, then the central directory and its end record.
  std::vector<uint8_t> finish()
  {
    const auto directory_offset = static_cast<uint32_t>(file_.size());
    for (const Entry & entry : entries_) {
      put32(file_, 0x02014b50);  // central directory header
      put16(file_, 20);          // version made by
      put16(file_, 20);          // version needed to extract
      put16(file_, 0);           // flags
      put16(file_, 0);           // method: stored
      put16(file_, 0);           // modification time
      put16(file_, kDosDate);
      put32(file_, entry.crc);
      put32(file_, entry.size);
      put32(file_, entry.size);
      put16(file_, static_cast<uint16_t>(entry.name.size()));
      put16(file_, 0);  // extra field length
      put16(file_, 0);  // comment length
      put16(file_, 0);  // disk number
      put16(file_, 0);  // internal attributes
      put32(file_, 0);  // external attributes
      put32(file_, entry.offset);
      put_text(file_, entry.name);
    }
    const auto directory_size = static_cast<uint32_t>(file_.size()) - directory_offset;
    put32(file_, 0x06054b50);  // end of central directory
    put16(file_, 0);           // this disk
    put16(file_, 0);           // disk with the central directory
    put16(file_, static_cast<uint16_t>(entries_.size()));
    put16(file_, static_cast<uint16_t>(entries_.size()));
    put32(file_, directory_size);
    put32(file_, directory_offset);
    put16(file_, 0);  // comment length
    return std::move(file_);
  }

private:
  struct Entry
  {
    std::string name;
    uint32_t crc = 0;
    uint32_t size = 0;
    uint32_t offset = 0;
  };

  static constexpr uint16_t kDosDate = 0x0021;  // 1980-01-01, the zip epoch

  std::vector<uint8_t> file_;
  std::vector<Entry> entries_;
};

bool has_suffix(const std::string & text, const std::string & suffix)
{
  return text.size() >= suffix.size() &&
         text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// The npz bytes of `samples`: one .npy per key, in the key order of the LogSample fields.
std::vector<uint8_t> npz_bytes(const std::vector<LogSample> & samples, size_t motors)
{
  const std::vector<Column> columns = {
    column("time_stamp", samples, motors, &LogSample::time_stamp),
    column("mode", samples, motors, &LogSample::mode),
    column("q", samples, motors, &LogSample::q),
    column("dq", samples, motors, &LogSample::dq),
    column("ddq", samples, motors, &LogSample::ddq),
    column("tau_est", samples, motors, &LogSample::tau_est),
    column("temperature", samples, motors, &LogSample::temperature),
    column("vol", samples, motors, &LogSample::vol),
    column("sensor", samples, motors, &LogSample::sensor),
    column("motorstate", samples, motors, &LogSample::motorstate),
    column("reserve", samples, motors, &LogSample::reserve),
    column("imu_quaternion", samples, motors, &LogSample::imu_quaternion),
    column("imu_gyroscope", samples, motors, &LogSample::imu_gyroscope),
    column("imu_accelerometer", samples, motors, &LogSample::imu_accelerometer),
    column("imu_rpy", samples, motors, &LogSample::imu_rpy),
    column("imu_temperature", samples, motors, &LogSample::imu_temperature),
    column("q_cmd", samples, motors, &LogSample::q_cmd),
    column("dq_cmd", samples, motors, &LogSample::dq_cmd),
    column("tau_cmd", samples, motors, &LogSample::tau_cmd),
    column("kp_cmd", samples, motors, &LogSample::kp_cmd),
    column("kd_cmd", samples, motors, &LogSample::kd_cmd),
  };
  StoredZip zip;
  for (const Column & c : columns) {
    zip.add(c.name + ".npy", npy_file(c));
  }
  return zip.finish();
}

bool fits(const LogSample & s, size_t motors)
{
  for (const size_t size :
    {s.mode.size(), s.q.size(), s.dq.size(), s.ddq.size(), s.tau_est.size(),
      s.temperature.size(), s.vol.size(), s.sensor.size(), s.motorstate.size(),
      s.reserve.size(), s.q_cmd.size(), s.dq_cmd.size(), s.tau_cmd.size(), s.kp_cmd.size(),
      s.kd_cmd.size()})
  {
    if (size != motors) {
      return false;
    }
  }
  return true;
}

}  // namespace

ChunkLogger::ChunkLogger(
  std::string base_dir, std::string chunk_prefix, size_t samples_per_chunk)
: base_dir_(std::move(base_dir)), chunk_prefix_(std::move(chunk_prefix)),
  samples_per_chunk_(std::max<size_t>(1, samples_per_chunk))
{
  std::error_code error;
  fs::create_directories(base_dir_, error);
  if (error || !fs::is_directory(base_dir_)) {
    throw std::runtime_error(
            "cannot create the log directory " + base_dir_ + ": " +
            (error ? error.message() : "not a directory"));
  }
  // A run's chunks never mix with an earlier run's.
  for (fs::directory_iterator it(base_dir_, error), end; !error && it != end;
    it.increment(error))
  {
    const std::string name = it->path().filename().string();
    std::error_code ignored;
    if (name.compare(0, chunk_prefix_.size(), chunk_prefix_) == 0 &&
      (has_suffix(name, ".npz") || has_suffix(name, ".npz.tmp")) &&
      it->is_regular_file(ignored))
    {
      fs::remove(it->path(), ignored);
    }
  }
  thread_ = std::thread(&ChunkLogger::run, this);
}

ChunkLogger::~ChunkLogger()
{
  stop();
}

void ChunkLogger::log(LogSample sample)
{
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopping_) {
      return;
    }
    const size_t motors = sample.mode.size();
    if (motors == 0 || !fits(sample, motors) || (motors_ != 0 && motors != motors_)) {
      ++samples_dropped_;
      return;
    }
    motors_ = motors;
    if (queue_.size() >= kMaxQueuedSamples) {
      queue_.pop_front();
      ++samples_dropped_;
    }
    queue_.push_back(std::move(sample));
  }
  wake_.notify_one();
}

void ChunkLogger::stop()
{
  std::lock_guard<std::mutex> stop_lock(stop_mutex_);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stopping_ = true;
  }
  wake_.notify_one();
  if (thread_.joinable()) {
    thread_.join();
  }
}

void ChunkLogger::run()
{
  std::vector<LogSample> chunk;
  chunk.reserve(std::min(samples_per_chunk_, kDefaultSamplesPerChunk));
  while (true) {
    std::deque<LogSample> batch;
    bool stopping = false;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      wake_.wait(lock, [this] {return stopping_ || !queue_.empty();});
      batch.swap(queue_);
      stopping = stopping_;
    }
    for (LogSample & sample : batch) {
      chunk.push_back(std::move(sample));
      if (chunk.size() >= samples_per_chunk_) {
        write_chunk(chunk);
        chunk.clear();
      }
    }
    if (stopping) {
      if (!chunk.empty()) {
        write_chunk(chunk);
      }
      return;
    }
  }
}

void ChunkLogger::write_chunk(const std::vector<LogSample> & samples)
{
  // A chunk that fails keeps its number, so the gap shows where samples were lost.
  const size_t index = next_chunk_++;
  const std::string name = [&] {
      char number[32];
      std::snprintf(number, sizeof(number), "%06zu", index);
      return chunk_prefix_ + number + ".npz";
    }();
  const fs::path final_path = fs::path(base_dir_) / name;
  const fs::path temp_path = fs::path(base_dir_) / (name + ".tmp");
  try {
    const size_t motors = samples.front().mode.size();
    const std::vector<uint8_t> bytes = npz_bytes(samples, motors);
    {
      std::ofstream out(temp_path, std::ios::binary | std::ios::trunc);
      out.write(reinterpret_cast<const char *>(bytes.data()),
                static_cast<std::streamsize>(bytes.size()));
      out.close();
      if (!out) {
        throw std::runtime_error("cannot write " + temp_path.string());
      }
    }
    fs::rename(temp_path, final_path);  // a reader never sees half a chunk
    ++chunks_written_;
  } catch (const std::exception & e) {
    ++write_failures_;
    std::error_code ignored;
    fs::remove(temp_path, ignored);
    std::cerr << "ChunkLogger: chunk " << name << " not written: " << e.what() << std::endl;
  }
}

}  // namespace golem_safety
