"""Deterministic Greedy Server Selector implementing Algorithm 1 from IEEE TMC 2026."""

from __future__ import annotations

from typing import List, Optional, Tuple

from eatsched.env.cluster import Server, ServerCluster


class GreedyServerSelector:
    """Implements Algorithm 1 (Deterministic Greedy Server Allocation) [PAPER Sec. C4]."""

    def __init__(
        self,
        cold_start_s: float = 30.0,
        reconfig_s: float = 1.76,
        reuse_s: float = 0.0,
    ) -> None:
        self.cold_start_s = cold_start_s
        self.reconfig_s = reconfig_s
        self.reuse_s = reuse_s

    def select_servers(
        self,
        cluster: ServerCluster,
        c_k: int,
    ) -> Tuple[Optional[List[Server]], float, str]:
        """Select c_k servers according to Algorithm 1.

        Returns:
            (selected_servers, init_delay_seconds, reuse_type)
            If insufficient idle servers exist, returns (None, 0.0, "invalid")
        """
        idle_servers = cluster.get_idle_servers()
        if len(idle_servers) < c_k:
            return None, 0.0, "invalid"

        # Step 1: Process Group Reuse Check [PAPER Alg. 1 Lines 2-6]
        # Inspect existing idle process groups G_m^t where d_e == m
        idle_groups = cluster.get_idle_groups()  # Dict[group_id, List[Server]]
        matching_groups = [(gid, srvs) for gid, srvs in idle_groups.items() if len(srvs) == c_k]
        if matching_groups:
            # Tie-break: select the group with the smallest group ID m [PAPER Alg. 1 Line 5]
            matching_groups.sort(key=lambda item: item[0])
            selected_group_id, selected_servers = matching_groups[0]
            # Zero init cost for reuse [PAPER Sec. C1]
            return selected_servers, self.reuse_s, "reuse"

        # Step 2: Server Aggregation & Group Eviction [PAPER Alg. 1 Lines 7-13]
        selected: List[Server] = []

        # Collect idle servers with no loaded process (d_e == 0) first [PAPER Alg. 1 Line 8]
        zero_servers = [s for s in idle_servers if s.group_id == 0]
        # Sort by server_id for deterministic allocation
        zero_servers.sort(key=lambda s: s.server_id)

        if len(zero_servers) >= c_k:
            selected = zero_servers[:c_k]
        else:
            selected.extend(zero_servers)

            # Iteratively destroy existing idle process groups in ascending order
            # of size (primary key) and group ID (secondary key) [PAPER Alg. 1 Line 9]
            sorted_groups = sorted(idle_groups.items(), key=lambda item: (len(item[1]), item[0]))

            for _gid, srvs in sorted_groups:
                # Sort servers within group deterministically
                srvs.sort(key=lambda s: s.server_id)
                for s in srvs:
                    selected.append(s)
                    if len(selected) == c_k:
                        break
                if len(selected) == c_k:
                    break

        # Evict old process group assignments and assign a new unique group ID [PAPER Alg. 1 Line 12]
        new_gid = cluster.allocate_group_id()
        had_cold_server = any(s.group_id == 0 for s in selected)

        for s in selected:
            s.group_id = new_gid

        # Step 2e: Initialization cost evaluation [PAPER Table VI]
        if had_cold_server:
            init_delay = self.cold_start_s
            reuse_type = "cold_start"
        else:
            init_delay = self.reconfig_s
            reuse_type = "reconfig"

        return selected, init_delay, reuse_type
