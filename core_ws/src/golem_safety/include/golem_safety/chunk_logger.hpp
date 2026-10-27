#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace golem_safety
{

// One logged sample, one field per key of the .npz chunks: the measured motor state
// (/lowstate), the IMU, and the newest command the layer relayed. Each motor field has one
// entry per body slot, and every motor field of a sample has the same length; the dtype of
// a key in the file is its field's element type. The chunk shapes are (samples, motors) for
// a plain motor field, (samples, motors, 2 or 4) for temperature, sensor and reserve,
// (samples, 4 or 3) for the IMU vectors and (samples,) for time_stamp and imu_temperature.
struct LogSample
{
  explicit LogSample(size_t motors = 0);

  double time_stamp = 0.0;  // s, wall clock

  std::vector<uint8_t> mode;
  std::vector<float> q, dq, ddq, tau_est;  // rad, rad/s, rad/s^2, N*m
  std::vector<std::array<int16_t, 2>> temperature;
  std::vector<float> vol;
  std::vector<std::array<uint32_t, 2>> sensor;
  std::vector<uint32_t> motorstate;
  std::vector<std::array<uint32_t, 4>> reserve;

  std::array<float, 4> imu_quaternion{};
  std::array<float, 3> imu_gyroscope{};
  std::array<float, 3> imu_accelerometer{};
  std::array<float, 3> imu_rpy{};
  int16_t imu_temperature = 0;

  std::vector<float> q_cmd, dq_cmd, tau_cmd, kp_cmd, kd_cmd;  // the newest relayed command
};

// The directory of one run under `root`: <root>/<YYYYmmdd-HHMMSS>-<pid>, local time. A
// ChunkLogger clears the chunks of its prefix in its directory when it starts, so a
// directory of its own per run keeps every earlier run's chunks.
std::string run_log_directory(const std::string & root);

// Writes samples to numbered chunk files <base_dir>/<chunk_prefix>000000.npz, ... from a
// background thread. A chunk holds samples_per_chunk samples (at least 1); stop() writes
// the samples left over as a last, shorter chunk. A chunk is an uncompressed .npz that
// numpy.load reads, with the keys, in this order, of the LogSample fields. The motor count
// is the first valid sample's, and a sample with another count is dropped.
//
// log() never blocks on the disk: a sample goes into a bounded queue and the oldest queued
// sample is dropped when it is full. Every method is safe to call from any thread.
class ChunkLogger
{
public:
  static constexpr size_t kDefaultSamplesPerChunk = 1000;
  static constexpr size_t kMaxQueuedSamples = 10000;

  // Creates base_dir (and its parents) and removes the <chunk_prefix>*.npz files in it.
  // Throws std::runtime_error when the directory cannot be created.
  ChunkLogger(std::string base_dir, std::string chunk_prefix, size_t samples_per_chunk);
  ~ChunkLogger();
  ChunkLogger(const ChunkLogger &) = delete;
  ChunkLogger & operator=(const ChunkLogger &) = delete;

  // Queues one sample; ignored after stop().
  void log(LogSample sample);

  // Writes everything queued, then joins the writer thread. Idempotent.
  void stop();

  const std::string & directory() const {return base_dir_;}
  size_t chunks_written() const {return chunks_written_;}
  // Samples not logged: invalid, or dropped from a full queue.
  size_t samples_dropped() const {return samples_dropped_;}
  // Chunks the disk refused; each leaves a gap in the numbering.
  size_t write_failures() const {return write_failures_;}

private:
  void run();
  void write_chunk(const std::vector<LogSample> & samples);

  const std::string base_dir_;
  const std::string chunk_prefix_;
  const size_t samples_per_chunk_;

  std::mutex stop_mutex_;  // one stop() at a time joins the thread
  std::mutex mutex_;       // guards the queue and what is marked below
  std::condition_variable wake_;
  std::deque<LogSample> queue_;                  // mutex_
  size_t motors_ = 0;  // mutex_; 0 until the first valid sample
  bool stopping_ = false;                        // mutex_

  size_t next_chunk_ = 0;  // the writer thread's
  std::atomic<size_t> chunks_written_{0};
  std::atomic<size_t> samples_dropped_{0};
  std::atomic<size_t> write_failures_{0};

  std::thread thread_;
};

}  // namespace golem_safety
