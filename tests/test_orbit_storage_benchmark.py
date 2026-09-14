from scripts.benchmark_orbit_storage import balanced_allocation, sample_processes


def test_tt_storage_sample_is_evenly_balanced():
    processes = sample_processes("tt", 1000)
    assert [events for _, events in processes.values()] == [334, 333, 333]
    assert sum(events for _, events in processes.values()) == 1000


def test_standard_mixture_balances_top_level_groups_and_processes():
    processes = sample_processes("mixture", 1000)
    assert sum(events for _, events in processes.values()) == 1000
    assert [processes[name][1] for name in ("QCD_HT50tobb", "QCD_HT50toInf")] == [125, 125]
    assert [processes[name][1] for name in ("tt_hadronic", "tt_leptonic", "tt_semileptonic")] == [84, 83, 83]
    vjets = [name for name in processes if name.startswith(("WJets", "DYJets", "ZJets"))]
    assert sum(processes[name][1] for name in vjets) == 250
    assert sum(processes[name][1] for name in processes if name.startswith(("WW_", "WZ_", "ZZ_"))) == 250


def test_balanced_allocation_assigns_remainder_in_declared_order():
    assert balanced_allocation(5, ["a", "b", "c"]) == {"a": 2, "b": 2, "c": 1}
