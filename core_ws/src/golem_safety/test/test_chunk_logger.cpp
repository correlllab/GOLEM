#include <gtest/gtest.h>

#include <unistd.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <regex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "golem_safety/chunk_logger.hpp"

using golem_safety::ChunkLogger;
using golem_safety::LogSample;

namespace fs = std::filesystem;

namespace {

// A private temp directory, removed with the object.
class ChunkLoggerTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    static int counter = 0;
    dir_ = fs::temp_directory_path() /
      ("golem_safety_chunks_" + std::to_string(::getpid()) + "_" + std::to_string(counter++));
    fs::create_directories(dir_);
  }

  void TearDown() override
  {
    std::error_code ignored;
    fs::remove_all(dir_, ignored);
  }

  // The file names in `dir`, sorted.
  static std::vector<std::string> names_in(const fs::path & dir)
  {
    std::vector<std::string> names;
    for (const auto & entry : fs::directory_iterator(dir)) {
      names.push_back(entry.path().filename().string());
    }
    std::sort(names.begin(), names.end());
    return names;
  }

  fs::path dir_;
};

LogSample sample_of(size_t motors, double time_stamp)
{
  LogSample sample(motors);
  sample.time_stamp = time_stamp;
  return sample;
}

}  // namespace

TEST_F(ChunkLoggerTest, WritesChunkFiles)
{
  // 2.5 chunks of samples: two full chunks and the partial one stop() flushes.
  ChunkLogger logger(dir_.string(), "chunk_", 100);
  for (int i = 0; i < 250; ++i) {
    logger.log(sample_of(29, 1.0e9 + i));
  }
  logger.stop();

  const std::vector<std::string> expected = {
    "chunk_000000.npz", "chunk_000001.npz", "chunk_000002.npz"};
  EXPECT_EQ(names_in(dir_), expected);
  EXPECT_EQ(logger.chunks_written(), 3u);
  EXPECT_EQ(logger.samples_dropped(), 0u);
  EXPECT_EQ(logger.write_failures(), 0u);

  // Chunks 0 and 1 hold 100 samples each, chunk 2 holds 50.
  EXPECT_EQ(fs::file_size(dir_ / expected[0]), fs::file_size(dir_ / expected[1]));
  EXPECT_LT(fs::file_size(dir_ / expected[2]), fs::file_size(dir_ / expected[0]));
}

TEST_F(ChunkLoggerTest, ExactMultipleLeavesNoEmptyChunk)
{
  ChunkLogger logger(dir_.string(), "chunk_", 50);
  for (int i = 0; i < 100; ++i) {
    logger.log(sample_of(29, i));
  }
  logger.stop();
  EXPECT_EQ(names_in(dir_), (std::vector<std::string>{"chunk_000000.npz", "chunk_000001.npz"}));
}

TEST_F(ChunkLoggerTest, DestructorFlushes)
{
  {
    ChunkLogger logger(dir_.string(), "chunk_", 100);
    for (int i = 0; i < 10; ++i) {
      logger.log(sample_of(29, i));
    }
  }
  EXPECT_EQ(names_in(dir_), (std::vector<std::string>{"chunk_000000.npz"}));
}

TEST_F(ChunkLoggerTest, StopIsIdempotentAndLaterSamplesAreIgnored)
{
  ChunkLogger logger(dir_.string(), "chunk_", 100);
  logger.log(sample_of(29, 0.0));
  logger.stop();
  logger.stop();
  logger.log(sample_of(29, 1.0));
  logger.stop();
  EXPECT_EQ(names_in(dir_), (std::vector<std::string>{"chunk_000000.npz"}));
  EXPECT_EQ(logger.samples_dropped(), 0u);
}

TEST_F(ChunkLoggerTest, NoSamplesWritesNoFiles)
{
  ChunkLogger logger(dir_.string(), "chunk_", 100);
  logger.stop();
  EXPECT_TRUE(names_in(dir_).empty());
}

TEST_F(ChunkLoggerTest, ZeroSamplesPerChunkMeansOne)
{
  ChunkLogger logger(dir_.string(), "chunk_", 0);
  for (int i = 0; i < 3; ++i) {
    logger.log(sample_of(29, i));
  }
  logger.stop();
  EXPECT_EQ(names_in(dir_).size(), 3u);
}

TEST_F(ChunkLoggerTest, SamplesThatDoNotFitTheFirstAreDropped)
{
  ChunkLogger logger(dir_.string(), "chunk_", 100);
  logger.log(sample_of(29, 0.0));
  logger.log(sample_of(27, 1.0));  // another motor count than the first sample's

  LogSample ragged = sample_of(29, 2.0);
  ragged.q_cmd.pop_back();  // a motor field that disagrees with the others
  logger.log(ragged);

  logger.log(sample_of(0, 3.0));  // no motors at all
  logger.log(sample_of(29, 4.0));
  logger.stop();

  EXPECT_EQ(logger.samples_dropped(), 3u);
  EXPECT_EQ(names_in(dir_), (std::vector<std::string>{"chunk_000000.npz"}));
}

TEST_F(ChunkLoggerTest, CreatesMissingDirectoriesAndReportsIt)
{
  const fs::path nested = dir_ / "a" / "b";
  ChunkLogger logger(nested.string(), "chunk_", 10);
  EXPECT_EQ(logger.directory(), nested.string());
  EXPECT_TRUE(fs::is_directory(nested));
  logger.stop();
}

TEST_F(ChunkLoggerTest, ClearsOldChunksOfItsPrefixOnly)
{
  for (const char * name :
    {"chunk_000007.npz", "chunk_000008.npz.tmp", "chunks.npz", "other_000001.npz", "notes.txt"})
  {
    std::ofstream(dir_ / name) << "x";
  }
  ChunkLogger logger(dir_.string(), "chunk_", 10);
  logger.stop();
  EXPECT_EQ(
    names_in(dir_),
    (std::vector<std::string>{"chunks.npz", "notes.txt", "other_000001.npz"}));
}

TEST_F(ChunkLoggerTest, ThrowsWhenTheDirectoryCannotBeCreated)
{
  std::ofstream(dir_ / "file") << "x";
  EXPECT_THROW(ChunkLogger((dir_ / "file" / "sub").string(), "chunk_", 10), std::runtime_error);
}

TEST_F(ChunkLoggerTest, ManyProducersLoseNothing)
{
  ChunkLogger logger(dir_.string(), "chunk_", 100);
  std::vector<std::thread> producers;
  for (int t = 0; t < 4; ++t) {
    producers.emplace_back([&logger, t] {
        for (int i = 0; i < 500; ++i) {
          logger.log(sample_of(29, t * 1000 + i));
        }
      });
  }
  for (auto & producer : producers) {
    producer.join();
  }
  logger.stop();
  // 2000 samples, nowhere near the queue bound.
  EXPECT_EQ(logger.samples_dropped(), 0u);
  EXPECT_EQ(logger.chunks_written(), 20u);
}

TEST(RunLogDirectory, IsTheRootWithATimestampAndThePid)
{
  const fs::path run = golem_safety::run_log_directory("/some/root");
  EXPECT_EQ(run.parent_path(), fs::path("/some/root"));
  const std::regex shape("^[0-9]{8}-[0-9]{6}-" + std::to_string(::getpid()) + "$");
  EXPECT_TRUE(std::regex_match(run.filename().string(), shape)) << run;
}
