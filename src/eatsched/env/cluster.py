"""Server cluster representations and energy models for EAT-Sched."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Server:
    """Represents a computing server in an edge or cloud tier."""

    server_id: int
    speed: float = 1.0
    tier: str = "edge"
    power_idle_watts: float = 80.0
    power_busy_watts: float = 250.0
    wan_latency_s: float = 0.0

    # Dynamic state variables [PAPER Sec. C2]
    available: bool = True  # a_e(t) (True=1=idle, False=0=busy)
    remaining_time: float = 0.0  # t_e^r(t)
    group_id: int = 0  # d_e(t) (0 = cold start / no resident model)
    current_task_id: Optional[int] = None

    def power(self) -> float:
        """Instantaneous power consumption based on busy status [ASM-10]."""
        return self.power_idle_watts if self.available else self.power_busy_watts

    def advance_time(self, dt: float) -> Optional[int]:
        """Advance time by dt seconds. Returns task_id if completed during this interval."""
        if self.available:
            return None
        self.remaining_time = max(0.0, self.remaining_time - dt)
        if self.remaining_time <= 0.0:
            completed_task = self.current_task_id
            self.available = True
            self.current_task_id = None
            # Group ID d_e is preserved across tasks to allow process-group reuse [PAPER Sec. C3]
            return completed_task
        return None


class ServerCluster:
    """Manages an entire cluster of edge and cloud servers."""

    def __init__(self, servers: List[Server]) -> None:
        self.servers = servers
        self.total_energy_joules: float = 0.0
        self.next_group_id: int = 1

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> ServerCluster:
        """Instantiate cluster from YAML configuration dictionary."""
        servers: List[Server] = []
        server_cfg = cfg.get("servers", {})
        is_homogenous = server_cfg.get("homogenous", True)

        if is_homogenous:
            num_servers = cfg["num_servers"]
            speed = float(server_cfg.get("speed", 1.0))
            p_idle = float(server_cfg.get("power_idle_watts", 80.0))
            p_busy = float(server_cfg.get("power_busy_watts", 250.0))
            tier = str(server_cfg.get("tier", "edge"))
            wan_latency = float(server_cfg.get("wan_latency_s", 0.0))

            for i in range(num_servers):
                servers.append(
                    Server(
                        server_id=i,
                        speed=speed,
                        tier=tier,
                        power_idle_watts=p_idle,
                        power_busy_watts=p_busy,
                        wan_latency_s=wan_latency,
                    )
                )
        else:
            groups = server_cfg.get("groups", [])
            idx = 0
            for g in groups:
                count = int(g["count"])
                speed = float(g.get("speed", 1.0))
                p_idle = float(g.get("power_idle_watts", 80.0))
                p_busy = float(g.get("power_busy_watts", 250.0))
                tier = str(g.get("tier", "edge"))
                wan_latency = float(g.get("wan_latency_s", 0.0))
                for _ in range(count):
                    servers.append(
                        Server(
                            server_id=idx,
                            speed=speed,
                            tier=tier,
                            power_idle_watts=p_idle,
                            power_busy_watts=p_busy,
                            wan_latency_s=wan_latency,
                        )
                    )
                    idx += 1

        return cls(servers)

    @property
    def num_servers(self) -> int:
        return len(self.servers)

    def get_idle_servers(self) -> List[Server]:
        """Return all servers currently idle (available == True)."""
        return [s for s in self.servers if s.available]

    def get_idle_groups(self) -> Dict[int, List[Server]]:
        """Group idle servers by their existing process group ID (excluding group 0)."""
        groups: Dict[int, List[Server]] = {}
        for s in self.get_idle_servers():
            if s.group_id > 0:
                groups.setdefault(s.group_id, []).append(s)
        return groups

    def allocate_group_id(self) -> int:
        """Generate a new unique process group ID."""
        gid = self.next_group_id
        self.next_group_id += 1
        return gid

    def advance_time(self, dt: float) -> List[int]:
        """Advance simulation time on all servers and accumulate power consumption."""
        completed_tasks: List[int] = []
        for s in self.servers:
            # Accumulate energy E = P * dt
            self.total_energy_joules += s.power() * dt
            completed_id = s.advance_time(dt)
            if completed_id is not None and completed_id not in completed_tasks:
                completed_tasks.append(completed_id)
        return completed_tasks

    def reset(self) -> None:
        """Reset all servers to initial cold idle state."""
        for s in self.servers:
            s.available = True
            s.remaining_time = 0.0
            s.group_id = 0
            s.current_task_id = None
        self.total_energy_joules = 0.0
        self.next_group_id = 1
