import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing


options = VarParsing("analysis")
options.setDefault("outputFile", "orbit_storage.root")
options.register("representation", "plain", VarParsing.multiplicity.singleton, VarParsing.varType.string, "plain or tokenized")
options.register("outputKind", "edm", VarParsing.multiplicity.singleton, VarParsing.varType.string, "edm or nano")
options.register("eventCount", 1, VarParsing.multiplicity.singleton, VarParsing.varType.int, "Number of events")
options.parseArguments()
if len(options.inputFiles) != 1:
    raise ValueError("Exactly one packed input file must be supplied with inputFiles=...")

process = cms.Process("ORBITSTORAGE")
process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(options.eventCount))
process.options = cms.untracked.PSet(
    numberOfThreads=cms.untracked.uint32(1),
    numberOfStreams=cms.untracked.uint32(1),
)
process.orbitPayload = cms.EDProducer(
    "OrbitStoragePayloadProducer",
    inputFile=cms.string(options.inputFiles[0]),
    representation=cms.string(options.representation),
    outputKind=cms.string(options.outputKind),
)
process.path = cms.Path(process.orbitPayload)

common = dict(
    fileName=cms.untracked.string("file:" + options.outputFile),
    outputCommands=cms.untracked.vstring("drop *", "keep *_orbitPayload_*_ORBITSTORAGE"),
    compressionAlgorithm=cms.untracked.string("LZMA"),
    compressionLevel=cms.untracked.int32(9),
)
if options.outputKind == "nano":
    process.output = cms.OutputModule("NanoAODOutputModule", **common)
else:
    process.output = cms.OutputModule("PoolOutputModule", **common)
process.endpath = cms.EndPath(process.output)
