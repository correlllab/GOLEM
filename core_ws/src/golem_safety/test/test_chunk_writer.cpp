// Writes chunks of deterministic samples for test_chunk_files.py, which holds the same
// formulas and compares what numpy reads back. Not installed.
//
//   test_chunk_writer <dir> <chunk_prefix> <samples_per_chunk> <samples> <motors>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

#include "golem_safety/chunk_logger.hpp"

namespace
{

// Every value is a small integer times a power of two, exact in float32.
golem_safety::LogSample sample_at(size_t motors, int k)
{
  golem_safety::LogSample s(motors);
  const auto kf = static_cast<float>(k);
  s.time_stamp = 1700000000.0 + 0.5 * k;
  for (size_t m = 0; m < motors; ++m) {
    const int i = static_cast<int>(m);
    const auto fi = static_cast<float>(i);
    s.mode[m] = static_cast<uint8_t>((k + i) % 251);
    s.q[m] = 0.25F * static_cast<float>(k * 1000 + i);
    s.dq[m] = -0.5F * static_cast<float>(k + i);
    s.ddq[m] = 0.125F * static_cast<float>(7 * k + i);
    s.tau_est[m] = 0.0625F * static_cast<float>(3 * k - i);
    s.temperature[m] = {static_cast<int16_t>(k - i), static_cast<int16_t>(i - 2 * k)};
    s.vol[m] = 24.0F + 0.5F * static_cast<float>(i % 8);
    s.sensor[m] = {static_cast<uint32_t>(1000 * k + i), 0xFFFFFFFFU - static_cast<uint32_t>(k)};
    s.motorstate[m] = static_cast<uint32_t>(3 * k + i);
    s.reserve[m] = {static_cast<uint32_t>(k), static_cast<uint32_t>(i), 7U, 0xDEADBEEFU};
    s.q_cmd[m] = 0.5F * static_cast<float>(k + i);
    s.dq_cmd[m] = 0.25F * static_cast<float>(k - i);
    s.tau_cmd[m] = 0.5F * static_cast<float>(i + 2 * k);
    s.kp_cmd[m] = 100.0F + 0.5F * kf;
    s.kd_cmd[m] = 2.0F + 0.25F * fi + 0.125F * kf;
  }
  s.imu_quaternion = {1.0F, 0.5F * kf, -0.25F, 2.0F};
  s.imu_gyroscope = {0.125F * kf, 0.25F, -0.5F};
  s.imu_accelerometer = {9.5F, -0.5F * kf, 0.75F};
  s.imu_rpy = {0.0625F * kf, -0.0625F, 0.5F};
  s.imu_temperature = static_cast<int16_t>(-k);
  return s;
}

}  // namespace

int main(int argc, char ** argv)
{
  if (argc != 6) {
    std::cerr << "usage: test_chunk_writer <dir> <chunk_prefix> <samples_per_chunk> <samples> "
      "<motors>\n";
    return 2;
  }
  try {
    const size_t per_chunk = std::stoul(argv[3]);
    const int samples = std::stoi(argv[4]);
    const size_t motors = std::stoul(argv[5]);
    golem_safety::ChunkLogger logger(argv[1], argv[2], per_chunk);
    for (int k = 0; k < samples; ++k) {
      logger.log(sample_at(motors, k));
    }
    logger.stop();
    if (logger.samples_dropped() != 0 || logger.write_failures() != 0) {
      std::cerr << "dropped " << logger.samples_dropped() << ", write failures "
                << logger.write_failures() << "\n";
      return 1;
    }
  } catch (const std::exception & e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
  return 0;
}
