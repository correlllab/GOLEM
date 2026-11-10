#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <stdexcept>
#include <thread>

#include "golem_safety/spinner.hpp"
#include "rclcpp/rclcpp.hpp"

namespace
{

// An executor whose spin() throws, as one whose callback throws does.
class ThrowingExecutor : public rclcpp::executors::SingleThreadedExecutor
{
public:
  explicit ThrowingExecutor(bool std_exception) : std_exception_(std_exception) {}

  void spin() override
  {
    if (std_exception_) {
      throw std::runtime_error("the heartbeat publish failed");
    }
    throw 42;
  }

private:
  const bool std_exception_;
};

// Wait up to 5 s for the spinner to report a failure; return whether it did.
bool wait_failed(const golem_safety::Spinner & spinner)
{
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!spinner.failed() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return spinner.failed();
}

// Initialise rclcpp around each test.
class SpinnerTest : public ::testing::Test
{
protected:
  void SetUp() override {rclcpp::init(0, nullptr);}
  void TearDown() override {rclcpp::shutdown();}
};

}  // namespace

// A std::exception escaping spin() ends the thread with failed() and its text.
TEST_F(SpinnerTest, AnExceptionFromSpinIsCaughtAndReported)
{
  ThrowingExecutor executor(true);
  golem_safety::Spinner spinner(executor);
  ASSERT_TRUE(wait_failed(spinner));
  EXPECT_EQ(spinner.error(), "the heartbeat publish failed");
  spinner.stop();
}

// Any other exception escaping spin() is caught and reported too.
TEST_F(SpinnerTest, ANonStandardExceptionIsCaughtToo)
{
  ThrowingExecutor executor(false);
  golem_safety::Spinner spinner(executor);
  ASSERT_TRUE(wait_failed(spinner));
  EXPECT_EQ(spinner.error(), "an exception that is not a std::exception");
  spinner.stop();
}

// stop() ends a normal spin without a failure, and a second stop() does nothing.
TEST_F(SpinnerTest, StopEndsANormalSpin)
{
  rclcpp::executors::SingleThreadedExecutor executor;
  auto node = std::make_shared<rclcpp::Node>("spinner_test");
  executor.add_node(node);
  golem_safety::Spinner spinner(executor);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_FALSE(spinner.failed());
  spinner.stop();
  EXPECT_FALSE(spinner.failed());
  spinner.stop();
}
