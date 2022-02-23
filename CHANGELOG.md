# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/dylan-j-young/thompson-pulsed/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/dylan-j-young/thompson-pulsed/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/dylan-j-young/thompson-pulsed/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/dylan-j-young/thompson-pulsed/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/dylan-j-young/thompson-pulsed/compare/v0.0.0...v0.1.0
[0.0.0]: https://github.com/dylan-j-young/thompson-pulsed/tree/v0.0.0