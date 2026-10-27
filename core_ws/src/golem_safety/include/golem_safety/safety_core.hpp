#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "golem_safety/chunk_logger.hpp"
#include "golem_safety/config.hpp"
#include "golem_safety/robot_profile.hpp"
#include "unitree_hg/msg/low_cmd.hpp"
#include "unitree_hg/msg/low_state.hpp"

namespace golem_safety
{

// A command input. full_body_mode reads Full (every body slot), split_mode reads Lower (the
// profile's lower slots) and Upper (its upper slots), and arm_sdk_mode reads Upper.
enum class Input { Full, Upper, Lower };

// The writer of a command: its DDS publisher GID, as raw bytes.
using WriterId = std::string;

// Something for the node to log, at its level. The core never logs itself.
struct Event
{
  enum Kind { Info, Warn, Error } kind;
  std::string text;
};

// Every decision of the safety layer, with no ROS: the node feeds it the samples it
// receives and publishes what step() returns. Time is passed in as `now`, seconds on the
// boot-time clock, and must not go backwards.
//
// Relay. Each input keeps its newest valid command. step() merges the engaged inputs'
// newest commands, each on its own slots, into zero_command; the slots no input owns keep
// zero_command's values. In arm_sdk_mode the weight slot carries the weight, which ramps
// 0 -> 1 over arm_sdk.engage_seconds from engagement. Header and CRC are stamped last on
// whatever step() and exit_step() return.
//
// Engagement. An input engages on its first valid command. With startup.rule zero_stream
// the core counts as engaged once any input has, and step() returns the merge from its
// first call (zero_command until an input engages). With engage_gate it counts as engaged
// once every input of the mode has, and step() returns nothing until then.
//
// A command is dropped, and the input keeps its last valid command, when it has a
// non-finite q, dq, tau, kp or kd on the input's slots; the drop is logged (Warn) at most
// once a second per input, with the count. A dropped command does not engage its input.
// A valid command is clipped on the input's slots: q to [q_clip_low, q_clip_high], dq and
// tau to +-dq_clip and +-tau_clip, kp and kd to [0, kp_max] and [0, kd_max]. A kp or kd
// above its maximum is logged (Warn) once per joint and gain.
//
// Single writer. Each input has a current writer, the writer of its first command. A
// command from another writer is accepted, and its writer becomes current (Info), when the
// current writer's last command (valid or not) is older than single_writer.handover_seconds;
// otherwise it latches the estop with "second writer on <topic>".
//
// The estop latches, one way until restart, on
//   - a /lowstate whose mode_machine is not the profile's, once the core is engaged (before
//     that it is recorded and not checked further);
//   - a /lowstate q past [q_estop_low, q_estop_high], |dq| past dq_estop, |tau_est| past
//     tau_estop or any of them non-finite, on the checked slots: every slot of the mode's
//     inputs, from the first /lowstate with zero_stream and from engagement with
//     engage_gate. The reason names the joint, the value, the limit and the multiple;
//   - a second writer;
//   - with use_estop, a hardware estop sample that is triggered or not plugged in.
// After the latch step() returns
//   - in arm_sdk_mode, the upper input's last valid command with the weight ramping from
//     its last written value to 0 over arm_sdk.release_seconds, then 0 for ever. The ramp
//     advances only while /lowstate is at most stale.state_seconds old (any age when that
//     is <= 0), so a robot cut off from the layer does not see the weight jump when the
//     link comes back;
//   - otherwise the estop command: estop_response.burst_count times and then nothing with
//     burst_then_silence, on every step with keep_writing.
// Nothing is written after the latch when the core had not taken control (engage_gate,
// before engagement). The first step after the latch raises an Error event with the
// reason and the timing report: decision_ms (when the core latched), action_ms (the `now`
// of that step) and their difference, and for the hardware estop trigger_stamp_ms (the
// sample's arrival) and trigger_to_action_ms; all ms on the boot-time clock.
//
// Not thread-safe: one thread calls every method, except estopped(), which any thread may
// call. estop_reason() is fixed once estopped() has returned true.
class SafetyCore
{
public:
  // Throws std::invalid_argument when `config` is for another robot, when arm_sdk_mode
  // has no weight slot in the profile, or when estop_command does (a Damp kd list that is
  // not one entry per body slot).
  SafetyCore(RobotProfile profile, SafetyConfig config, bool use_estop, int domain_id);

  SafetyCore(const SafetyCore &) = delete;
  SafetyCore & operator=(const SafetyCore &) = delete;

  void on_state(const unitree_hg::msg::LowState & state, double now);
  // A command on an input the mode does not read is ignored, with one Error event.
  void on_command(
    Input input, const unitree_hg::msg::LowCmd & cmd, const WriterId & writer, double now);
  // Ignored without use_estop.
  void on_estop(bool triggered, bool plugged_in, double now);

  // What to publish this cycle, header and CRC stamped; nullopt publishes nothing.
  std::optional<unitree_hg::msg::LowCmd> step(double now);

  // What to publish during shutdown, in place of step(); nullopt means done, and step()
  // answers as exit_step() from the first call on. full_body_mode and split_mode are done
  // at once, unless an estop burst is still being written or the estop command has not
  // been written since the latch. arm_sdk_mode holds the last valid command while the
  // weight ramps to 0 over arm_sdk.release_seconds, whatever the /lowstate age, and is done
  // once it has written weight 0; it is done at once if it never engaged.
  std::optional<unitree_hg::msg::LowCmd> exit_step(double now);

  bool estopped() const {return estopped_.load(std::memory_order_acquire);}
  const std::string & estop_reason() const {return estop_reason_;}

  // The events raised since the last call, oldest first.
  std::vector<Event> drain_events();

  // With logging.enabled, one sample per /lowstate that arrived before the latch (the
  // latching one included), taken once: that /lowstate's motor fields and IMU, and the
  // last command step() or exit_step() returned (zero_command before any), on the
  // profile's body slots. time_stamp is the wall clock at the /lowstate's arrival.
  std::optional<LogSample> take_log_sample();

private:
  struct InputState
  {
    bool used = false;  // the mode reads this input
    SlotRange slots;
    std::string topic;
    std::optional<WriterId> writer;  // the current writer
    double writer_last = 0.0;        // arrival of the current writer's last command
    std::optional<unitree_hg::msg::LowCmd> newest;  // the newest valid command, clipped
    double newest_time = 0.0;                       // its arrival
    size_t drops_unreported = 0;
    std::optional<double> last_drop_report;
    bool misuse_reported = false;
  };

  // The arm_sdk hand-back: the weight ramps from `from` to 0 over release_seconds, while
  // it runs.
  struct Release
  {
    bool begun = false;
    double from = 0.0;
    std::optional<double> start;  // set while the ramp runs
    double elapsed = 0.0;         // ramp time done before the last hold
    std::optional<bool> running;  // last reported
  };

  struct PendingReport
  {
    std::string reason;
    double decision = 0.0;
    std::optional<double> trigger;
  };

  InputState & input_state(Input input);
  bool took_control() const;
  bool state_flowing(double now) const;

  void accept_command(InputState & in, const unitree_hg::msg::LowCmd & cmd, double now);
  void report_drop(InputState & in, const std::string & detail, double now);
  unitree_hg::msg::LowCmd clipped(InputState & in, const unitree_hg::msg::LowCmd & cmd);
  std::optional<std::string> limit_violation(const unitree_hg::msg::LowState & state) const;

  void latch(const std::string & reason, double now, std::optional<double> trigger);
  void report_estop(bool written, double now);

  unitree_hg::msg::LowCmd merged() const;
  void set_weight(unitree_hg::msg::LowCmd & cmd, double weight) const;
  double release_weight(double now, bool exiting);
  std::optional<unitree_hg::msg::LowCmd> relay_output(double now);
  std::optional<unitree_hg::msg::LowCmd> estop_output(double now, bool exiting);
  std::optional<unitree_hg::msg::LowCmd> finish(std::optional<unitree_hg::msg::LowCmd> out);

  const RobotProfile profile_;
  const SafetyConfig config_;
  const bool use_estop_;
  const int domain_id_;  // the ROS domain; 0 is the real robots' command bus
  const unitree_hg::msg::LowCmd zero_cmd_;
  const unitree_hg::msg::LowCmd estop_cmd_;  // built once: nothing on the latch path throws

  std::array<InputState, 3> inputs_;
  std::vector<Input> mode_inputs_;
  std::vector<int> checked_slots_;  // every slot of the mode's inputs, ascending

  bool engaged_ = false;
  std::optional<double> engage_start_;  // arm_sdk weight ramp

  std::optional<unitree_hg::msg::LowState> state_;
  std::optional<double> state_time_;  // the last /lowstate's arrival, kept while estopped
  double state_wall_time_ = 0.0;
  bool state_unlogged_ = false;

  std::atomic<bool> estopped_{false};
  std::string estop_reason_;
  std::optional<PendingReport> pending_report_;
  int estop_commands_written_ = 0;

  double weight_ = 0.0;  // the arm_sdk weight last written
  Release release_;
  bool exiting_ = false;
  bool exit_done_ = false;

  std::vector<bool> kp_clip_reported_, kd_clip_reported_;
  unitree_hg::msg::LowCmd last_written_;
  std::vector<Event> events_;
};

}  // namespace golem_safety
