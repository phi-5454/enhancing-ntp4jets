"""Lightweight canonical process taxonomy shared by downstream tools."""

FIVE_CLASS_GROUPS = {
    "QCD": ("QCD_HT50tobb", "QCD_HT50toInf"),
    "tt": (
        "tt0123j_5f_ckm_LO_MLM_hadronic",
        "tt0123j_5f_ckm_LO_MLM_leptonic",
        "tt0123j_5f_ckm_LO_MLM_semiLeptonic",
    ),
    "VJets": (
        "WJetsToLNu_13TeV-madgraphMLM-pythia8",
        "WJetsToQQ_13TeV-madgraphMLM-pythia8",
        "DYJetsToLL_13TeV-madgraphMLM-pythia8",
        "ZJetsTobb_13TeV-madgraphMLM-pythia8",
        "ZJetsTocc_13TeV-madgraphMLM-pythia8",
        "ZJetsToQQ_13TeV-madgraphMLM-pythia8",
        "ZJetsTovv_13TeV-madgraphMLM-pythia8",
    ),
    "VV": (
        "WW_hadronic",
        "WW_leptonic",
        "WW_semileptonic",
        "WZ_hadronic",
        "WZ_leptonic",
        "WZ_semileptonic",
        "ZZ_hadronic",
        "ZZ_leptonic",
        "ZZ_semileptonic",
    ),
    "ggHbb": ("ggHbb",),
}
FIVE_CLASS_TO_LABEL = {name: index for index, name in enumerate(FIVE_CLASS_GROUPS)}
PROCESS_TO_GROUP = {
    process: group for group, processes in FIVE_CLASS_GROUPS.items() for process in processes
}

