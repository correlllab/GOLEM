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

// Give each test a private temporary directory, removed after it.
class ChunkLoggerTest : public ::testing::Test
{
protected:
  // Create the test's directory.
  void SetUp() override
  {
    static int counter = 0;
    dir_ = fs::temp_directory_path() /
      ("golem_safety_chunks_" + std::to_string(::getpid()) + "_" + std::to_string(counter++));
    fs::create_directories(dir_);
  }

  // Remove the test's directory.
  void TearDown() override
  {
    std::error_code ignored;
    fs::remove_all(dir_, ignored);
  }

  // Return the file names in `dir`, sorted.
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

// Return a zeroed sample of `motors` slots stamped `time_stamp`.
LogSample sample_of(size_t motors, double time_stamp)
{
  LogSample sample(motors);
  sample.time_stamp = time_stamp;
  return sample;
}

}  // namespace

// Full chunks are written as they fill, and stop() writes the rest as a shorter last chunk.
TEST_F(ChunkLoggerTest, WritesChunkFiles)
{
  ChunkLogger logger(dir_.string(), "chunk_", 100);
  for (int i = 0; i < 250; ++i) {
    logger.log(sample_of(29, 1.0e9 + i));
  }
  logger.stop();

  const std::vector<std::string> expected = {
    "chunk_000000.npz", "chunk_000001.npz", "chunk_000002.npz"};
  EXPECT_EQ(names_in(dir_), expected);
  EXPECT_EQ(logger.samples_dropped(), 0u);
  EXPECT_EQ(logger.write_failures(), 0u);

  EXPECT_EQ(fs::file_size(dir_ / expected[0]), fs::file_size(dir_ / expected[1]));
  EXPECT_LT(fs::file_size(dir_ / expected[2]), fs::file_size(dir_ / expected[0]));
}

// A sample count that fills the last chunk exactly leaves no empty chunk behind.
TEST_F(ChunkLoggerTest, ExactMultipleLeavesNoEmptyChunk)
{
  ChunkLogger logger(dir_.string(), "chunk_", 50);
  for (int i = 0; i < 100; ++i) {
    logger.log(sample_of(29, i));
  }
  logger.stop();
  EXPECT_EQ(names_in(dir_), (std::vector<std::string>{"chunk_000000.npz", "chunk_000001.npz"}));
}

// Destroying the logger writes the samples still queued.
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

// stop() can be called again, and a sample logged after it is ignored, not dropped.
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

// A logger that got no sample writes no file.
TEST_F(ChunkLoggerTest, NoSamplesWritesNoFiles)
{
  ChunkLogger logger(dir_.string(), "chunk_", 100);
  logger.stop();
  EXPECT_TRUE(names_in(dir_).empty());
}

// samples_per_chunk 0 is raised to 1.
TEST_F(ChunkLoggerTest, ZeroSamplesPerChunkMeansOne)
{
  ChunkLogger logger(dir_.string(), "chunk_", 0);
  for (int i = 0; i < 3; ++i) {
    logger.log(sample_of(29, i));
  }
  logger.stop();
  EXPECT_EQ(names_in(dir_).size(), 3u);
}

// Samples whose motor count differs from the first sample's, ragged ones and empty ones are
// dropped and counted.
TEST_F(ChunkLoggerTest, SamplesThatDoNotFitTheFirstAreDropped)
{
  ChunkLogger logger(dir_.string(), "chunk_", 100);
  logger.log(sample_of(29, 0.0));
  logger.log(sample_of(27, 1.0));

  LogSample ragged = sample_of(29, 2.0);
  ragged.q_cmd.pop_back();
  logger.log(ragged);

  logger.log(sample_of(0, 3.0));
  logger.log(sample_of(29, 4.0));
  logger.stop();

  EXPECT_EQ(logger.samples_dropped(), 3u);
  EXPECT_EQ(names_in(dir_), (std::vector<std::string>{"chunk_000000.npz"}));
}

// The constructor creates missing parent directories, and directory() names the result.
TEST_F(ChunkLoggerTest, CreatesMissingDirectoriesAndReportsIt)
{
  const fs::path nested = dir_ / "a" / "b";
  ChunkLogger logger(nested.string(), "chunk_", 10);
  EXPECT_EQ(logger.directory(), nested.string());
  EXPECT_TRUE(fs::is_directory(nested));
  logger.stop();
}

// Starting removes earlier chunks and temporaries of its own prefix and nothing else.
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

// A directory that cannot be created throws std::runtime_error.
TEST_F(ChunkLoggerTest, ThrowsWhenTheDirectoryCannotBeCreated)
{
  std::ofstream(dir_ / "file") << "x";
  EXPECT_THROW(ChunkLogger((dir_ / "file" / "sub").string(), "chunk_", 10), std::runtime_error);
}

// Concurrent log() calls from several threads lose no sample below the queue bound.
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
  EXPECT_EQ(logger.samples_dropped(), 0u);
  EXPECT_EQ(names_in(dir_).size(), 20u);
}

// run_log_directory() is <root>/<YYYYmmdd-HHMMSS>-<pid>.
TEST(RunLogDirectory, IsTheRootWithATimestampAndThePid)
{
  const fs::path run = golem_safety::run_log_directory("/some/root");
  EXPECT_EQ(run.parent_path(), fs::path("/some/root"));
  const std::regex shape("^[0-9]{8}-[0-9]{6}-" + std::to_string(::getpid()) + "$");
  EXPECT_TRUE(std::regex_match(run.filename().string(), shape)) << run;
}
