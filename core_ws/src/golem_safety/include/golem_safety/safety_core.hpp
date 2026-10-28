#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
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
// newest commands, each on its own slots, into zero_command; the slots no engaged input
// owns keep zero_command's values. In arm_sdk_mode the weight slot carries the weight,
// which ramps 0 -> 1 over arm_sdk.engage_seconds from engagement. Header and CRC are
// stamped last on whatever step() and exit_step() return.
//
// Engaging. Nothing engages until the newest /lowstate carries the profile's mode_machine
// and, with use_estop, a hardware estop sample has arrived; a /lowstate from another
// robot only delays engaging. Engaging is tried on each valid command:
//   - startup.rule zero_stream: step() returns the merge from its first call, so
//     zero_command until an input engages. Each input engages on its own first valid
//     command that arrives once those conditions hold (input_engaged, Info).
//   - engage_gate: step() returns nothing until every input of the mode engages, at once,
//     on a valid command that finds every input with a command no older than its stale
//     time (below; when that is > 0), the newest /lowstate at most
//     engage.max_state_age_seconds old, and on every slot of the mode's inputs the newest
//     command's q within engage.max_position_error of the measured q (lowcmd_engage or
//     arm_sdk_engage, Info).
// A command that does not engage raises engage_refused (Warn) with the first reason, at
// most once a second.
//
// Relay ready. relay_ready() is true while the engage conditions that do not depend on the
// inputs hold: the newest /lowstate carries the profile's mode_machine and, with use_estop,
// a hardware estop sample has arrived. Before engaging, a /lowstate from another robot makes
// it false again. A sample that latches the estop never makes it true; once the estop has
// latched it keeps its value, and estopped() is what counts.
//
// Quiet inputs are held and reported, never latched. Once an input has engaged, a gap in
// its commands longer than its stale time (stale.upper_seconds for Upper,
// stale.lower_seconds for Lower and Full) raises input_quiet (Warn) while its last
// command keeps being relayed, and its next command raises input_resumed (Info). Once the
// core has engaged, /lowstate does the same with stale.state_seconds (state_quiet,
// state_resumed). A stale time <= 0 turns its report off.
//
// A command is valid when q, dq, tau, kp and kd are finite on every body slot, the input's
// own or not; any other command latches the estop. A valid command is clipped on the
// input's slots: q to [q_clip_low, q_clip_high], dq and tau to +-dq_clip and +-tau_clip,
// kp and kd to [0, kp_max] and [0, kd_max]. A kp or kd above its maximum raises a
// gain_clipped event (Warn) once per joint and gain.
//
// Single writer. Each input has a current writer, the writer of its first command. A
// command from another writer is accepted, and its writer becomes current (Info), when the
// current writer's last command (valid or not) is older than single_writer.handover_seconds;
// otherwise it latches the estop with "second writer on <topic>".
//
// The /lowstate checks run on the samples that carry the profile's mode_machine, from the
// first /lowstate with zero_stream and from engagement with engage_gate.
//
// The estop latches, one way until restart, on
//   - a command with a non-finite value, before or after engaging: "Command validation
//     failed on <topic>: motor <slot> (<joint>) <field> is <value>";
//   - in arm_sdk_mode, a command with a waist kp, clipped, below arm_sdk.waist_kp_min,
//     before or after engaging: "Command validation failed on <topic>: motor 12
//     (waist_yaw_joint) kp 10.0 is below arm_sdk.waist_kp_min 40.0";
//   - a /lowstate whose mode_machine is not the profile's, once the core is engaged;
//   - on ROS domain 0 with control.state_lag_ms > 0, a /lowstate (checked) that arrives
//     more than state_lag_ms late against the robot's uint32 ms tick: its arrival minus
//     its tick, less the smallest such offset seen, a baseline that may rise by 0.5 ms/s
//     to follow clock drift;
//   - a /lowstate (checked) q past [q_estop_low, q_estop_high], |dq| past dq_estop,
//     |tau_est| past tau_estop or any of them non-finite, on every slot of the mode's
//     inputs. The reason names the joint, the value, the limit and the multiple;
//   - a second writer;
//   - with use_estop, a hardware estop sample that is triggered or not plugged in, and,
//     from the first sample on, a step() more than estop.timeout_seconds after the newest
//     sample.
// The latch sets the flag before any other work, and nothing it does can throw.
// After the latch step() returns
//   - in arm_sdk_mode, the upper input's last valid command with the weight ramping from
//     its last written value to 0 over arm_sdk.release_seconds, then 0 for ever. The ramp
//     advances only while /lowstate is at most stale.state_seconds old (any age when that
//     is <= 0), so a robot cut off from the layer does not see the weight jump when the
//     link comes back. Each change between running and holding raises
//     arm_sdk_release_ramping (Info) or arm_sdk_release_held (Warn);
//   - otherwise the estop command: estop_response.burst_count times and then nothing with
//     burst_then_silence, on every step with keep_writing.
// Nothing is written after the latch when the core had not taken control (engage_gate,
// before engagement). The estop report (Error) is estop_command_written, or
// estop_latched_silent when that first step after the latch wrote nothing: the reason,
// decision_ms (when the core latched), action_ms (the `now` of that step) and their
// difference, and for the hardware estop trigger_stamp_ms (the sample's arrival) and
// trigger_to_action_ms; all ms on the boot-time clock. It is built by the drain_events()
// after that step, so nothing is formatted before the step's command is handed back.
//
// Events. engage_refused, input_engaged, lowcmd_engage, arm_sdk_engage, input_quiet,
// input_resumed, state_quiet, state_resumed, gain_clipped, arm_sdk_release_ramping,
// arm_sdk_release_held and the estop report are one JSON object each: keys sorted,
// ", " and ": " separators, and numbers as Python's json module writes them (times and
// weights rounded to 4 decimals). The other events are text. Raising an event never
// throws: an event that cannot be built is dropped.
//
// Not thread-safe: one thread calls every method, except estopped() and relay_ready(),
// which any thread may call.
class SafetyCore
{
public:
  // `domain_id` is the ROS domain: on 0, the real robots' command bus, the transport-lag
  // check runs. Throws std::invalid_argument when `config` is for another robot, when
  // arm_sdk_mode finds no weight slot or no waist_* joint among the upper slots in the
  // profile, or when estop_command does (a Damp kd list that is not one entry per body
  // slot).
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
  // answers as exit_step() from the first call on.
  //   - full_body_mode and split_mode are done at once, so the robot holds the last
  //     command written. Once estopped, the estop sequence is finished first: the rest of
  //     the burst with burst_then_silence, and with keep_writing the estop command once if
  //     none has been written since the latch.
  //   - arm_sdk_mode holds the last valid command while the weight ramps from its value
  //     to 0 over arm_sdk.release_seconds, whatever the /lowstate age, then writes weight
  //     0 for 0.1 s more, so that one lost packet cannot leave the robot on a higher
  //     weight. It is done at once if it never engaged.
  std::optional<unitree_hg::msg::LowCmd> exit_step(double now);

  bool estopped() const {return estopped_.load(std::memory_order_acquire);}
  // Relaying can begin: see "Relay ready" above.
  bool relay_ready() const {return relay_ready_.load(std::memory_order_acquire);}
  // Empty before the latch; a fixed text when the reason could not be built.
  const std::string & estop_reason() const;

  // The events raised since the last call, oldest first, with the estop report in its
  // place once the step after the latch has run.
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
    std::string source;          // the input's name in events, e.g. split_mode_upper
    double stale_seconds = 0.0;  // quiet-report threshold; <= 0 turns it off
    std::optional<WriterId> writer;  // the current writer
    double writer_last = 0.0;        // arrival of the current writer's last command
    std::optional<unitree_hg::msg::LowCmd> newest;  // the newest valid command, clipped
    double newest_time = 0.0;                       // its arrival
    bool engaged = false;  // its slots carry `newest`
    bool quiet = false;    // input_quiet raised, input_resumed not yet
    bool misuse_reported = false;
  };

  // Why a command does not engage.
  struct Refusal
  {
    enum Kind
    {
      NoState, StateAge, ModeMachine, NoEstopSample, NoCommand, QuietInput, StateNotFinite,
      FarCommand
    } kind;
    // StateAge: the age, s; ModeMachine: the sample's; QuietInput: the gap, s; FarCommand: q
    double value = 0.0;
    double measured = 0.0;  // FarCommand: the measured q
    int slot = 0;           // StateNotFinite, FarCommand
    const InputState * input = nullptr;  // NoCommand, QuietInput
  };

  // The transport-lag check's state: the robot's ms tick, unwrapped, against `now`.
  struct StateLag
  {
    std::optional<uint32_t> last_tick;
    long long wraps = 0;
    std::optional<double> base_ms;  // the earliest arrival seen, drift allowed for
    double base_time = 0.0;
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

  // The estop report's times, s; the reason is estop_reason().
  struct PendingReport
  {
    double decision = 0.0;
    std::optional<double> trigger;
    bool acted = false;      // the first step after the latch has run
    bool written = false;    // and returned a command
    double action = 0.0;
    size_t event_index = 0;  // where the report goes among that step's events
  };

  // A measured value past its estop limit (`limit` set), or a non-finite one.
  struct Violation
  {
    int slot = 0;
    const char * field = "";
    double value = 0.0;
    std::optional<double> limit;
  };

  // A non-finite value in a command.
  struct BadValue
  {
    int slot = 0;
    const char * field = "";
    double value = 0.0;
  };

  InputState & input_state(Input input);
  bool took_control() const;
  bool state_flowing(double now) const;

  void check_state(const unitree_hg::msg::LowState & state, double now);
  void update_relay_ready();

  void accept_command(InputState & in, const unitree_hg::msg::LowCmd & cmd, double now);
  std::optional<BadValue> soft_waist(const unitree_hg::msg::LowCmd & cmd) const;
  void try_engage(InputState & in, double now);
  std::optional<Refusal> refusal(double now) const;
  std::string refusal_text(const Refusal & refusal) const;
  void report_refusal(const Refusal & refusal, double now);
  void report_quiet(double now);
  double state_lag_ms(uint32_t tick, double now);
  void check_estop_input(double now);
  std::optional<BadValue> non_finite_value(const unitree_hg::msg::LowCmd & cmd) const;
  unitree_hg::msg::LowCmd clipped(InputState & in, const unitree_hg::msg::LowCmd & cmd);
  std::optional<Violation> limit_violation(const unitree_hg::msg::LowState & state) const;
  std::string joint(int slot) const;

  // Latches with the reason build_reason() returns: the flag first, then the report and
  // the arm_sdk release (set_latched), then the reason, whose failure leaves the fixed
  // text. Does nothing once latched.
  template<class BuildReason>
  void latch(double now, std::optional<double> trigger, BuildReason && build_reason) noexcept
  {
    if (!set_latched(now, trigger)) {
      return;
    }
    try {
      estop_reason_ = build_reason();
    } catch (...) {
      reason_lost_ = true;
    }
  }
  bool set_latched(double now, std::optional<double> trigger) noexcept;
  std::string estop_report(const PendingReport & report) const;

  // Appends the event build_text() returns; drops it when it cannot be built.
  template<class BuildText>
  void raise(Event::Kind kind, BuildText && build_text) noexcept
  {
    try {
      events_.push_back(Event{kind, build_text()});
    } catch (...) {
    }
  }

  unitree_hg::msg::LowCmd merged() const;
  void set_weight(unitree_hg::msg::LowCmd & cmd, double weight) const;
  double release_weight(double now, bool exiting);
  std::optional<unitree_hg::msg::LowCmd> relay_output(double now);
  std::optional<unitree_hg::msg::LowCmd> estop_output(double now, bool exiting);
  std::optional<unitree_hg::msg::LowCmd> finish(std::optional<unitree_hg::msg::LowCmd> out);

  const RobotProfile profile_;
  const SafetyConfig config_;
  const bool use_estop_;
  const unitree_hg::msg::LowCmd zero_cmd_;
  const unitree_hg::msg::LowCmd estop_cmd_;  // built once: nothing on the latch path throws

  std::array<InputState, 3> inputs_;
  std::vector<Input> mode_inputs_;
  std::vector<int> checked_slots_;  // every slot of the mode's inputs, ascending
  std::vector<int> waist_slots_;    // arm_sdk_mode: the waist_kp_min slots

  bool engaged_ = false;  // any input with zero_stream, every input with engage_gate
  std::optional<double> engage_start_;  // arm_sdk weight ramp
  std::optional<double> refusal_reported_;  // the last engage_refused

  std::optional<unitree_hg::msg::LowState> state_;
  std::optional<double> state_time_;  // the last /lowstate's arrival, kept while estopped
  double state_wall_time_ = 0.0;
  bool state_unlogged_ = false;
  bool state_quiet_ = false;  // state_quiet raised, state_resumed not yet
  const bool lag_checked_;    // ROS domain 0 (the real robots' bus), state_lag_ms > 0
  StateLag lag_;

  std::optional<double> estop_sample_time_;  // the newest hardware estop sample's arrival

  std::atomic<bool> relay_ready_{false};
  std::atomic<bool> estopped_{false};
  std::string estop_reason_;
  bool reason_lost_ = false;
  std::optional<PendingReport> pending_report_;
  int burst_written_ = 0;      // estop commands of the burst written, at most burst_count
  bool estop_written_ = false;  // an estop command has been written since the latch

  double weight_ = 0.0;  // the arm_sdk weight last written
  Release release_;
  bool exiting_ = false;
  bool exit_done_ = false;
  std::optional<double> exit_hold_start_;  // arm_sdk: weight 0 first written on exit

  std::vector<bool> kp_clip_reported_, kd_clip_reported_;
  unitree_hg::msg::LowCmd last_written_;
  std::vector<Event> events_;
};

}  // namespace golem_safety
