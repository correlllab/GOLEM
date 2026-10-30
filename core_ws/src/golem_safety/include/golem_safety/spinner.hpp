#pragma once

#include <atomic>
#include <chrono>
#include <exception>
#include <string>
#include <thread>

#include "rclcpp/executor.hpp"

namespace golem_safety
{

// Runs an executor's spin() on a thread of its own, and stops and joins it on every way out
// of its scope, an exception included: a joinable std::thread that is destroyed ends the
// process in std::terminate, with no exit sequence. An exception that escapes spin() (a
// callback that throws) ends the thread with failed() true and its text in error(); the
// owner decides what follows. Declare it after everything the executor's callbacks use, so
// that it is destroyed first.
class Spinner
{
public:
  explicit Spinner(rclcpp::Executor & executor)
  : executor_(executor), thread_([this] {run();}) {}

  ~Spinner() {stop();}

  Spinner(const Spinner &) = delete;
  Spinner & operator=(const Spinner &) = delete;

  // Whether spin() ended with an exception; error() is then its text.
  bool failed() const {return failed_.load(std::memory_order_acquire);}
  const std::string & error() const {return error_;}

  // Returns once spin() has returned; a second call does nothing. A cancel() that arrives
  // before spin() has started is lost, so it is repeated until the thread returns.
  void stop()
  {
    if (!thread_.joinable()) {
      return;
    }
    while (!finished_.load(std::memory_order_acquire)) {
      try {
        executor_.cancel();
      } catch (const std::exception &) {
        // spin() ends by itself once the context is invalid
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    thread_.join();
  }

private:
  void run() noexcept
  {
    try {
      executor_.spin();
    } catch (const std::exception & e) {
      fail(e.what());
    } catch (...) {
      fail("an exception that is not a std::exception");
    }
    finished_.store(true, std::memory_order_release);
  }

  void fail(const char * text) noexcept
  {
    try {
      error_ = text;
    } catch (...) {
    }
    failed_.store(true, std::memory_order_release);
  }

  rclcpp::Executor & executor_;
  std::string error_;  // written before failed_ is set, read after
  std::atomic<bool> failed_{false};
  std::atomic<bool> finished_{false};
  std::thread thread_;  // last: it starts in the constructor, and uses the members above
};

}  // namespace golem_safety
