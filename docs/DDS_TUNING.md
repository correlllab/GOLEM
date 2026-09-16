# Network / DDS tuning (x86)

The stack moves bursty, high-bandwidth sensor data (camera images, Livox point
clouds) over CycloneDDS. On a gigabit link the wire has headroom, but the default
208 KB kernel UDP receive buffer overflows on bursts and the kernel silently
drops datagrams. Lost DDS fragments stall or drop whole point-cloud/image
samples, so topics go laggy even though the network is not full. The tell is
`RcvbufErrors` climbing in `/proc/net/snmp`.

Three settings fix this and **all three are required**.

## 1. Host kernel buffers

The containers run `network_mode: host`, so these are the host's `net.core`
sysctls. CycloneDDS can only *request* a large socket buffer — the kernel clamps
it to `net.core.rmem_max` — so without this, step 2 has no effect.

```bash
sudo sysctl -w net.core.rmem_max=2147483647
sudo sysctl -w net.core.rmem_default=16777216
sudo sysctl -w net.core.netdev_max_backlog=10000
```

To persist, put the same three lines in `/etc/sysctl.d/10-cyclonedds.conf` and
run `sudo sysctl --system`.

## 2. CycloneDDS config

`core_ws/cyclonedds.xml` requests a 16 MB receive buffer. The `ros` profile wires
it in automatically: `docker-compose.yml` bind-mounts it to
`/home/code/cyclonedds.xml` and sets `CYCLONEDDS_URI` to point at it. Edit the
file and restart the container to change the tuning (for example to pin the DDS
network interface).

## 3. Pin the NIC IRQ off the real-time cores

`eno1` has a single RX queue, so *all* inbound DDS traffic is processed in one
`NET_RX` softirq on whichever core holds its IRQ. `irqbalance` parks that IRQ on
core 11 — inside the MJPC planner's `0-11` set — and the contention drops the
robot's 500 Hz `/lowstate` stream to ~400 Hz with multi-second stalls (TF fades
in RViz, the lower-body controller trips its state-stale safe-hold). The wire and
the robot are fine; the loss is single-core contention.

Move the IRQ to a core outside the RT set (0–12) and stop `irqbalance` putting it
back:

```bash
sudo systemctl stop irqbalance
echo 13 | sudo tee /proc/irq/$(awk -F: '/eno1/{gsub(/ /,"",$1);print $1}' /proc/interrupts)/smp_affinity_list
```

Persist with `sudo systemctl disable --now irqbalance` plus a boot-time unit that
re-applies the affinity. The bringup CPU map reserves core 13 for this — the
estimator takes core 12 and `OTHER_CPUS` starts at 14, so no ROS node lands on
the network core.

Confirm the softirq moved (counts should grow on core 13, not 11):

```bash
grep NET_RX /proc/softirqs
```

## Verify

Under load, `RcvbufErrors` should stop climbing:

```bash
docker exec golem_ros grep '^Udp:' /proc/net/snmp
```
