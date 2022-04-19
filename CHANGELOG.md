# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- `demod_smoother()` no longer required in `e3l.Parameters` object

## [0.5.0] - 2022-04-19

### Added
- `premeasure_interleaved` option for `e3l.Experiment.preprocess()` allows premeasure runs to be interleaved with data runs
- `binned_average()` method for `tp.Time_Multitrace`, returning another `tp.Time_Multitrace` object with coarser time step
- `tp.parsers.keysight_hsa_csv()` function for parsing Keysight RSA traces
- Incomplete `tp.parsers.labview_log()` function for reading in LabVIEW parameter values in post-processing

### Changed
- `e3l.Data.track_cav_frequency_iq()` now estimates frequency by measuring slopes in the `tp.MT_Phase` object generated from the demodulated cavity phasor. Extra options added to function for variably sophisticated processing of such slopes
- `e3l.Data.demod_atom_trace()` now uses IQ demodulation instead of multiplication with a cosine

### Deprecated
- Old `e3l.Data.demod_atom_trace()` function still exists but was renamed to `e3l.Data.demod_atom_trace_OLD()` and is deprecated

### Fixed
- In `e3l_watchdog_test.py`, `event_handler.events` list no longer suffers from race condition leading to an uncaught exception
- `e3l.demod_atom_trace()` now works with `tp.Time_Multitrace` objects

### Removed
- Old `e3l.Cav_Phase` object no longer exists (it went unused)

## [0.4.0] - 2022-02-22

### Added
- `premeasure` and `postmeasure` options for `e3l.Experiment.preprocess()` allow for calculation of initial and bare cavity frequency
- `collapse` option for `e3l.Data.track_cav_frequency_iq()`, which collapses all non-time dimensions if set to True (default)

### Changed
- `CHANGELOG.md` now uses inline code formatting
- Shot class renamed to Sequence class
- `Sequence` class now initializes by default using arrays instead of a file and parser. To initialize a `Sequence` object using data from a file, the `Sequence.load()` class method was added.
- `e3l.Data` class now requires arrays to initialize, replacing earlier functionality where it initialized with NoneType attributes

### Removed
- Old non-IQ-based `e3l.Data.track_cav_frequency()` method

## [0.3.0] - 2022-02-14

### Added
- Watchdog example script for three_level
- A moving average function for `Time_Multitrace`
- Uncertainty handling for `Time_Multitrace`

### Changed
- `Time_Multitrace` method code no longer calls nearly identical helper functions
- `Time_Multitrace` time/dim attributes are read-only properties

### Deprecated
- `three_level.py` `Experiment.track_cav_frequency` methods no longer output separate `t`, `V` arrays by default. Instead, they will by default output a `Time_Multitrace` object

### Fixed
- `three_level.py` `Experiments` can now load `Shots` with 0 or 1 triggers properly

## [0.2.0] - 2022-02-07

### Added

- A changelog (this document!) roughly adhering to the standards specified in [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
- Parsing function for `ni_pcie7851r_ai`
- Three_level experiment test scripts and instructions for how to run them
- Digital IQ demodulation capabilities for `Time_Multitrace`
- `MT_Phasor` and `MT_Phase` classes (inheriting methods from `Time_Multitrace`)

### Changed
- Parsing function names changed to give unique identification
- Default cavity resonance probe measurement uses an IQ demodulation strategy, as opposed to an FFT + curve fit

### Removed
- `Time_Trace` and `Frequency_Trace` (instead use `Time_Multitrace` and `Frequency_Multitrace`)

## [0.1.0] - 2022-02-01

### Added

- Version control using the `setuptools_scm` package

## [0.0.0] - 2022-01-25

### Added

- Barebones package structure for a pulsed experiment analysis package
- Class definitions for single traces, multitraces, and shots (multiple traces from a single data file)
- Parser module in `/core/` containing a single parsing function for the NI PCI-5105 oscilloscope board
- Experiment-specific analysis code for the 3L experiment, stored in `/expts/three_level.py`

[Unreleased]: https://github.com/dylan-j-young/thompson-pulsed/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/dylan-j-young/thompson-pulsed/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/dylan-j-young/thompson-pulsed/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/dylan-j-young/thompson-pulsed/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/dylan-j-young/thompson-pulsed/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/dylan-j-young/thompson-pulsed/compare/v0.0.0...v0.1.0
[0.0.0]: https://github.com/dylan-j-young/thompson-pulsed/tree/v0.0.0